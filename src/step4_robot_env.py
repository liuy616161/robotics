"""
Step 4：自定义 Gymnasium 环境 — 机械臂末端到达任务

任务：把 3-DOF 机械臂的"末端到达目标"包装成标准 Gym 接口，然后用 SAC 训练
方法：继承 gymnasium.Env，实现 reset/step/render，用 SB3 check_env 验证

运行：python3.9 src/step4_robot_env.py

前置：pip3 install gymnasium stable-baselines3

为什么自定义环境？
  Gymnasium 提供了 CartPole, Pendulum 这些标准环境，但真实任务（机械臂到达、抓取、
  双足行走）都需要自己写。工程师的核心技能是：
    1. 把机器人仿真/物理引擎包装成 Gym 接口
    2. 设计合理的观测空间、动作空间、奖励函数
    3. 套用 SB3 训练，快速验证

环境设计说明：
  这里用纯数学（无 PyBullet）实现 3-DOF 机械臂 FK，
  避免额外依赖，专注于 Gym 接口的写法。
  如果你已完成 Task 1，可以把这里的 FK 替换成 PyBullet 的版本。

依赖：numpy, matplotlib, gymnasium, stable-baselines3
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
from typing import Optional, Tuple, Dict, Any

try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import SAC
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
except ImportError as e:
    raise ImportError(f"缺少依赖：{e}\n请运行：pip3 install gymnasium stable-baselines3")

import torch


# ── 机械臂参数 ────────────────────────────────────────────────
# 3-DOF 平面机械臂（在 XY 平面内运动，Z 轴高度固定）
# 关节1：绕 Z 轴旋转（控制平面内的"转向"）
# 关节2：绕 Z 轴旋转（控制"肘关节"弯曲）
# 关节3：绕 Z 轴旋转（控制"腕关节"弯曲）
ARM_LENGTHS = [0.4, 0.35, 0.25]   # 三段连杆长度（米）
ARM_Z       = 0.5                   # 机械臂高度（固定，简化为 2D 问题）

# 关节角限制（弧度）
JOINT_LIMIT_LOW  = np.array([-np.pi,   -np.pi/2,  -np.pi/2], dtype=np.float32)
JOINT_LIMIT_HIGH = np.array([ np.pi,    np.pi/2,   np.pi/2], dtype=np.float32)

# 动作限制：每步最大关节角变化量（弧度）
ACTION_LIMIT = 0.3   # ±0.1 rad/step，约 ±5.7 度

# 任务参数
MAX_EPISODE_STEPS = 50    # 每轮最多走多少步（与 FetchReach 对齐）
GOAL_TOLERANCE    = 0.05   # 到达目标的距离阈值（0.05 m = 5 cm）

# 目标点采样范围（在工作空间内随机采样）
# 最大可达半径 = L1 + L2 + L3 = 0.4 + 0.35 + 0.25 = 1.0
# 为了确保目标可达，限制在 0.3 ~ 0.8 m 之间
GOAL_RADIUS_MIN = 0.3
GOAL_RADIUS_MAX = 0.8


# ── 机械臂正运动学（FK）─────────────────────────────────────

def forward_kinematics(q: np.ndarray,
                       lengths: list = ARM_LENGTHS,
                       z: float = ARM_Z) -> np.ndarray:
    """
    3-DOF 平面机械臂正运动学

    原理：每个关节叠加旋转角度（类似 Step 1 的 2-DOF FK 扩展版）
      关节角累积：θ_cumsum[i] = q[0] + q[1] + ... + q[i]
      第 i 个连杆的末端位置（相对于机械臂基座）：
        dx_i = L_i × cos(θ_cumsum[i])
        dy_i = L_i × sin(θ_cumsum[i])

    参数：
        q       : 关节角，shape (3,)，单位弧度
        lengths : 连杆长度列表 [L1, L2, L3]
        z       : 机械臂高度（固定值）

    返回：
        ee_pos : 末端位置 [x, y, z]，shape (3,)
    """
    theta_cumsum = np.cumsum(q)   # [q0, q0+q1, q0+q1+q2]

    # 向量化运算，替代 generator 表达式
    x = np.sum(np.array(lengths) * np.cos(theta_cumsum))
    y = np.sum(np.array(lengths) * np.sin(theta_cumsum))

    return np.array([x, y, z], dtype=np.float32)


def get_link_positions(q: np.ndarray,
                       lengths: list = ARM_LENGTHS,
                       z: float = ARM_Z) -> list:
    """
    返回所有关节和末端的位置（用于可视化）

    返回：[(0,0,z), (x1,y1,z), (x2,y2,z), (x3,y3,z)]
          基座、关节1末端、关节2末端、末端执行器
    """
    theta_cumsum = np.cumsum(q)
    positions = [(0.0, 0.0, z)]  # 基座在原点
    x, y = 0.0, 0.0
    for l, t in zip(lengths, theta_cumsum):
        x += l * np.cos(t)
        y += l * np.sin(t)
        positions.append((x, y, z))
    return positions


# ── 自定义 Gymnasium 环境 ─────────────────────────────────────

class RobotArmEnv(gym.Env):
    """
    3-DOF 机械臂末端到达任务（Reach Task）

    目标：控制关节角增量，使末端到达随机目标位置

    Gymnasium 接口规范：
      - observation_space : 观测空间（gym.spaces.Box 等）
      - action_space      : 动作空间
      - reset()           : 重置环境，返回 (obs, info)
      - step(action)      : 执行动作，返回 (obs, reward, terminated, truncated, info)
      - render()          : 可选，可视化当前状态
      - close()           : 清理资源

    重要：gymnasium（新版）的 step() 返回 5 个值：
      obs, reward, terminated, truncated, info
      terminated = 任务成功/失败导致的终止（真正结束）
      truncated  = 超过最大步数导致的终止（时间限制）
      （旧版 gym 只有 done，新版拆成两个）
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()

        self.render_mode = render_mode
        self._step_count = 0
        self._target_pos = np.zeros(3, dtype=np.float32)
        self._joint_angles = np.zeros(3, dtype=np.float32)

        # ── 观测空间（Observation Space）────────────────────
        # 观测 = [末端位置(3), 关节角(3)] = 6维连续向量
        # 每个维度有上下界，SB3 会用这个信息做归一化等预处理
        #
        # 为什么不直接把目标位置也放进观测？
        # → 可以！但这里演示最简单的版本（固定目标）
        # → 如果目标随机变化，应该把目标位置也加入观测（"goal-conditioned" RL）
        # 重要修复：观测必须包含目标位置，否则 Agent 不知道目标在哪！
        # 观测维度: 3(ee) + 3(关节) + 3(目标) = 9维
        # 重要优化：使用相对位置观测 (更易学，有平移不变性)
        # 观测维度: 3(相对位置) + 3(关节) = 6维
        # 相对位置 = 目标 - 末端，具有更好的泛化能力
        self.observation_space = spaces.Box(
            low=np.array([-2.0, -2.0, -2.0,     # 相对位置下界 (dx, dy, dz)
                          -np.pi, -np.pi/2, -np.pi/2],   # 关节角下界
                         dtype=np.float32),
            high=np.array([2.0,  2.0,  2.0,    # 相对位置上界
                           np.pi,  np.pi/2,  np.pi/2],   # 关节角上界
                          dtype=np.float32),
            dtype=np.float32
        )

        # ── 动作空间（Action Space）──────────────────────────
        # 动作 = 关节角增量 [Δq1, Δq2, Δq3]，范围 ±ACTION_LIMIT
        # 注意：必须用 np.float32（不是 float64），否则 SB3 会报警告
        self.action_space = spaces.Box(
            low=-ACTION_LIMIT * np.ones(3, dtype=np.float32),
            high=ACTION_LIMIT * np.ones(3, dtype=np.float32),
            dtype=np.float32
        )

        # 固定目标（训练时会在 reset 中随机化，这里只是占位）
        self._target_pos = np.array([0.6, 0.2, ARM_Z], dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        重置环境

        返回：
            obs: 观测向量 (6,)
            info: 诊断信息字典
        """
        # 调用父类 reset 来初始化 np_random（符合 Gymnasium 规范）
        super().reset(seed=seed)

        # 重置关节角度
        self._joint_angles = np.zeros(3, dtype=np.float32)
        self._step_count = 0
        self._prev_dist = None  # 重置上步距离（势函数塑形用）

        # 随机初始化关节角（增加多样性）
        self._joint_angles = self.np_random.uniform(
            JOINT_LIMIT_LOW, JOINT_LIMIT_HIGH
        ).astype(np.float32)

        # 随机化目标位置（在可达范围内）
        radius = self.np_random.uniform(GOAL_RADIUS_MIN, GOAL_RADIUS_MAX)
        angle = self.np_random.uniform(-np.pi, np.pi)
        self._target_pos = np.array([
            radius * np.cos(angle),
            radius * np.sin(angle),
            ARM_Z
        ], dtype=np.float32)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def _get_obs(self) -> np.ndarray:
        """
        构造观测向量 (使用相对位置)

        重要优化：使用相对位置，具有平移不变性，更易学
        观测 = [dx, dy, dz, q1, q2, q3]，shape (6,)
        其中 dx = target_x - ee_x, dy = target_y - ee_y, dz = target_z - ee_z

        注意：必须返回 np.float32（与 observation_space.dtype 一致）
        """
        ee_pos = forward_kinematics(self._joint_angles)
        
        # 计算相对位置 (目标 - 末端)
        relative_pos = self._target_pos - ee_pos
        
        obs = np.concatenate([
            relative_pos.astype(np.float32),          # 相对位置 (3,) - 告诉 Agent 目标在哪
            self._joint_angles.astype(np.float32),    # 关节角 (3,) - 告诉 Agent 当前姿态
        ])
        return obs

    def _get_info(self) -> Dict[str, Any]:
        """
        返回诊断信息（用于调试和分析）

        返回：
            dict: 包含以下键值对
                - target: 目标位置 (x, y, z)
                - ee_pos: 末端执行器位置 (x, y, z)
                - joint_angles: 关节角度 (theta1, theta2, theta3)
                - distance: 到目标的距离
                - success: 是否成功到达
        """
        # 计算末端位置
        ee_pos = forward_kinematics(self._joint_angles, ARM_LENGTHS, ARM_Z)

        # 计算到目标的距离
        dist = np.linalg.norm(ee_pos - self._target_pos)

        return {
            'target': self._target_pos.copy(),
            'ee_pos': ee_pos.copy(),
            'joint_angles': self._joint_angles.copy(),
            'distance': dist,
            'success': dist < GOAL_TOLERANCE
        }

    def step(self,
             action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        执行一步动作

        动作 = 关节角增量 [Δq1, Δq2, Δq3]
        执行后：
          1. 更新关节角（加上增量，clip 到限制范围）
          2. 用 FK 计算新的末端位置
          3. 计算奖励
          4. 判断是否终止

        Gymnasium 规范（返回 5 个值，注意顺序）：
          obs        : 下一步观测
          reward     : 即时奖励
          terminated : 是否因任务成功/失败而终止（True = 真正结束）
          truncated  : 是否因超过步数限制而截断（True = 时间到了）
          info       : 信息字典

        参数：
            action : 关节角增量，shape (3,)，会被自动 clip 到 action_space 范围
        """
        self._step_count += 1

        # 1. clip 动作到合法范围（SAC 输出经过 tanh，理论上在范围内；PPO 可能超范围）
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # 2. 更新关节角
        self._joint_angles = np.clip(
            self._joint_angles + action.astype(np.float32),
            JOINT_LIMIT_LOW,
            JOINT_LIMIT_HIGH
        )

        # 3. 计算末端位置
        ee_pos = forward_kinematics(self._joint_angles)
        dist   = float(np.linalg.norm(ee_pos - self._target_pos))

        # 4. 计算奖励
        reward, reward_components = self._compute_reward(ee_pos, self._target_pos, dist, action)

        # 5. 判断终止条件
        # TODO 1：实现终止条件判断（提示：两种终止情形）
        #
        # terminated = True 的条件：
        #   - 成功到达目标（dist < GOAL_TOLERANCE）
        #   说明：terminated 表示 MDP 意义上的终止（episode 有自然结束点）
        #
        # truncated = True 的条件：
        #   - 超过最大步数（self._step_count >= MAX_EPISODE_STEPS）
        #   说明：truncated 表示人为截断，不是真正的终止
        #         （这对 value bootstrap 有影响：truncated 时应该 bootstrap，terminated 不用）
        #
        # 提示：
        #   terminated = (dist < GOAL_TOLERANCE)
        #   truncated  = (self._step_count >= MAX_EPISODE_STEPS)
        #
        terminated = (dist < GOAL_TOLERANCE)
        truncated = (self._step_count >= MAX_EPISODE_STEPS)

        obs  = self._get_obs()
        info = self._get_info()
        info['reward_components'] = reward_components  # 各部分奖励明细

        # 更新上步距离（用于下一步行势函数塑形）
        self._prev_dist = dist

        return obs, reward, terminated, truncated, info

    def _compute_reward(self,
                        ee_pos: np.ndarray,
                        target_pos: np.ndarray,
                        dist: float,
                        action: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """
        计算奖励函数

        返回:
            tuple: (总奖励, 各部分奖励字典)
                各部分奖励包含:
                    - shaping: 势函数塑形奖励
                    - dist: 距离惩罚
                    - action: 动作惩罚
                    - success: 成功奖励
        """
        """
        计算奖励函数

        奖励设计是 RL 应用中最关键也最需要经验的部分。
        坏的奖励函数会导致 "reward hacking"（钻空子）或稀疏奖励问题。

        这里提供两种方案：
          方案A（稠密奖励）：r = -dist（每步都有梯度信号，收敛快）
          方案B（稠密+成功奖励）：r = -dist + 10.0 * success（鼓励明确到达）

        TODO 2：实现奖励函数（建议从方案A开始）
        #
        # 方案A（简单版，适合入门）：
        #   reward = -dist
        #   （距离越近奖励越高，始终有梯度信号）
        #
        # 方案B（带成功奖励，适合进阶）：
        #   success = (dist < GOAL_TOLERANCE)
        #   reward = -dist + 10.0 * float(success)
        #   （额外奖励鼓励明确到达目标，而不只是"靠近"）
        #
        # 方案C（最接近工业实践）：
        #   action_penalty = 0.01 * np.sum(action**2)   # 惩罚大动作
        #   reward = -dist - action_penalty + 10.0 * float(dist < GOAL_TOLERANCE)  # noqa
        #   （同时优化：到达目标 + 动作平滑 + 节能）
        #
        # 面试考点：为什么要加动作惩罚？
        #   → 防止机器人抖动（高频大动作会损坏电机）
        #   → 鼓励平滑控制，sim2real 迁移更好
        """
        # reward = -dist

        # ── 势函数塑形奖励（鼓励持续靠近目标）─────────────────────────
        # 每步奖励 = -k * Δdist，"靠近就奖，远离就罚"，解决近距离梯度弱的问题
        # prev_dist 在 step() 中更新，初始为 None（第一步行规需特殊处理）
        prev_dist = getattr(self, '_prev_dist', None)
        if prev_dist is not None:
            # 势函数塑形：距离减少得越多，奖励越高
            # k=2.0 意味着每靠近 0.01m 额外奖励 0.02
            k_shaping = 2.0
            delta_dist = (prev_dist - dist)/dist  # 正=靠近，负=远离
            shaping_reward = k_shaping * delta_dist
        else:
            shaping_reward = 0.0

        # 距离惩罚 (主要引导信号)
        dist_penalty = -1.0 * dist

        # 动态动作惩罚：距离越远，惩罚越小（鼓励大步探索）
        # 距离越近，惩罚越小（鼓励精细控制，防止"躺平"）
        if dist > 0.5:
            action_penalty = 0.0  # 远距离：大胆走，不惩罚
        elif dist > 0.2:
            action_penalty = -0.001 * np.sum(action**2)  # 中距离：轻度惩罚
        elif dist > 0.1:
            action_penalty = -0.005 * np.sum(action**2)  # 近距离：更轻惩罚
        else:
            action_penalty = -0.01 * np.sum(action**2)  # 极近距离：几乎不惩罚，防止微调被抑制

        # 成功奖励 (稀疏但有力)
        success_bonus = 100.0 if dist < GOAL_TOLERANCE else 0.0

        reward = shaping_reward + dist_penalty + action_penalty + success_bonus

        components = {
            'shaping': float(shaping_reward),
            'dist': float(dist_penalty),
            'action': float(action_penalty),
            'success': float(success_bonus),
        }

        return float(reward), components

    def render(self) -> Optional[np.ndarray]:
        """
        可视化当前状态（生成 RGB 图像）

        render_mode="rgb_array" 时返回 numpy 数组
        render_mode=None 时返回 None

        这里用 matplotlib 画机械臂和目标点
        """
        if self.render_mode != "rgb_array":
            return None

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Step {self._step_count}')

        # 画机械臂
        positions = get_link_positions(self._joint_angles)
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        ax.plot(xs, ys, 'o-', color='steelblue', linewidth=3, markersize=8, zorder=3)
        ax.plot(xs[0], ys[0], 's', color='black', markersize=12, zorder=4)  # 基座

        # 画目标点
        ax.plot(self._target_pos[0], self._target_pos[1],
                '*', color='red', markersize=20, zorder=5, label='目标')

        # 画到达范围圆
        theta = np.linspace(0, 2*np.pi, 50)
        ax.plot(self._target_pos[0] + GOAL_TOLERANCE*np.cos(theta),
                self._target_pos[1] + GOAL_TOLERANCE*np.sin(theta),
                'r--', alpha=0.5, linewidth=1)

        ax.legend(loc='upper right')

        # 将图像转为 numpy 数组
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return img

    def close(self):
        """清理资源（如果有 PyBullet 连接等，在这里关闭）"""
        pass


# ── 详细训练记录回调 ─────────────────────────────────────────

class SuccessRateCallback(BaseCallback):
    """
    记录训练过程中的详细信息：

    - eval_timesteps: 训练步数
    - eval_rewards: 平均累计奖励
    - eval_success: 成功率（距离 < GOAL_TOLERANCE）
    - eval_steps: 平均回合步数
    - eval_distance: 平均最终距离
    - eval_actions: 动作统计（均值、标准差）
    - eval_ee_pos: 末端执行器位置统计
    - eval_goal_pos: 目标位置统计
    """

    def __init__(self, eval_env, eval_freq: int = 2000, n_eval_episodes: int = 20):
        super().__init__(verbose=0)
        self.eval_env        = eval_env
        self.eval_freq       = eval_freq
        self.n_eval_episodes = n_eval_episodes

        # 基础指标
        self.eval_timesteps  = []
        self.eval_rewards    = []
        self.eval_success    = []   # 成功率（0~1）

        # 新增详细指标
        self.eval_steps      = []   # 平均回合步数
        self.eval_distance   = []   # 平均最终距离
        self.eval_action_mean = []  # 动作均值
        self.eval_action_std = []   # 动作标准差
        self.eval_ee_x       = []   # 末端执行器 x 均值
        self.eval_ee_y       = []   # 末端执行器 y 均值
        self.eval_goal_x     = []   # 目标 x 均值
        self.eval_goal_y     = []   # 目标 y 均值

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            rewards_list = []
            success_list = []
            steps_list = []
            distance_list = []
            actions_list = []
            ee_x_list = []
            ee_y_list = []
            goal_x_list = []
            goal_y_list = []

            for _ in range(self.n_eval_episodes):
                obs, info = self.eval_env.reset()
                ep_reward = 0.0
                ep_success = False
                ep_steps = 0
                ep_actions = []

                for _ in range(MAX_EPISODE_STEPS):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    ep_reward += reward
                    ep_steps += 1
                    ep_actions.append(action)

                    # 记录末端执行器和目标位置（从 info 获取绝对坐标）
                    # 注意：现在观测使用相对位置 [dx, dy, dz, q1, q2, q3]
                    # 所以不能从 obs 获取绝对坐标，必须从 info 获取
                    ee_pos = info.get('ee_pos', np.zeros(3))[:2]  # 绝对坐标
                    goal_pos = info.get('target', np.zeros(3))[:2]
                    ee_x_list.append(ee_pos[0])
                    ee_y_list.append(ee_pos[1])
                    goal_x_list.append(goal_pos[0])
                    goal_y_list.append(goal_pos[1])

                    if info.get("success", False):
                        ep_success = True
                    if terminated or truncated:
                        break

                # 记录最终距离
                final_dist = np.linalg.norm(ee_pos - goal_pos)
                distance_list.append(final_dist)

                rewards_list.append(ep_reward)
                success_list.append(float(ep_success))
                steps_list.append(ep_steps)
                actions_list.extend(ep_actions)

            # 转换为数组便于统计
            actions_array = np.array(actions_list)

            # 记录统计数据
            self.eval_timesteps.append(self.num_timesteps)
            self.eval_rewards.append(np.mean(rewards_list))
            self.eval_success.append(np.mean(success_list))
            self.eval_steps.append(np.mean(steps_list))
            self.eval_distance.append(np.mean(distance_list))
            self.eval_action_mean.append(np.mean(actions_array))
            self.eval_action_std.append(np.std(actions_array))
            self.eval_ee_x.append(np.mean(ee_x_list))
            self.eval_ee_y.append(np.mean(ee_y_list))
            self.eval_goal_x.append(np.mean(goal_x_list))
            self.eval_goal_y.append(np.mean(goal_y_list))

            # 打印详细信息
            print(f"\n  ═══════════════════════════════════════")
            print(f"  [{self.num_timesteps:6d} steps] 训练统计")
            print(f"  ───────────────────────────────────────")
            print(f"  奖励:     {np.mean(rewards_list):8.2f} ± {np.std(rewards_list):.2f}")
            print(f"  成功率:   {np.mean(success_list)*100:7.1f}%  ({int(np.sum(success_list))}/{self.n_eval_episodes} 回合)")
            print(f"  回合步数: {np.mean(steps_list):7.1f} ± {np.std(steps_list):.1f}")
            print(f"  最终距离: {np.mean(distance_list):7.3f} m ± {np.std(distance_list):.3f}")
            print(f"  ───────────────────────────────────────")
            print(f"  动作统计:")
            print(f"    均值:   {np.mean(actions_array):8.4f}")
            print(f"    标准差: {np.std(actions_array):8.4f}")
            print(f"  末端执行器: x={np.mean(ee_x_list):6.3f}, y={np.mean(ee_y_list):6.3f}")
            print(f"  目标位置:   x={np.mean(goal_x_list):6.3f}, y={np.mean(goal_y_list):6.3f}")
            print(f"  ═══════════════════════════════════════")

        return True


# ── 训练函数 ─────────────────────────────────────────────────

# ── 课程学习环境 ───────────────────────────────────────────────

class CurriculumRobotArmEnv(RobotArmEnv):
    """
    课程学习版本的机械臂环境

    课程学习（Curriculum Learning）核心思想：
    - 第一阶段：从简单任务开始（近目标、易到达）
    - 逐渐增加任务难度（远目标）
    - 让 Agent 逐步学习复杂任务

    本实现：渐进式增加目标距离
    - 初始：目标距离 0.1m ~ 0.2m（很容易到达）
    - 逐渐扩展到：0.3m ~ 0.6m（完整任务，限制最大难度防止灾难性遗忘）
    """

    def __init__(self, *args, **kwargs):
        # 课程学习参数
        self._curriculum_level = 0  # 当前课程难度等级
        self._curriculum_start = 0   # 开始课程学习的步数

        # 目标距离范围（会随课程进度扩展）
        self._current_goal_radius_min = 0.1   # 初始最小距离
        self._current_goal_radius_max = 0.2   # 初始最大距离

        super().__init__(*args, **kwargs)

    def set_curriculum(self, level: int, total_timesteps: int):
        """
        设置课程难度

        Args:
            level: 当前课程等级（0~10）
            total_timesteps: 总训练步数
        """
        self._curriculum_level = level

        # 根据课程等级计算目标距离范围
        # level=0: 0.1~0.2m (很简单)
        # level=10: 0.3~0.6m (限制最大难度，防止灾难性遗忘)
        progress = level / 10.0  # 0.0 ~ 1.0

        # 限制最大难度为 0.6m（而非 0.8m），确保 Agent 能稳定完成任务
        max_radius_cap = 0.6

        self._current_goal_radius_min = 0.1 + (GOAL_RADIUS_MIN - 0.1) * progress
        self._current_goal_radius_max = 0.2 + (max_radius_cap - 0.2) * progress

        # 确保范围有效
        self._current_goal_radius_min = max(0.1, self._current_goal_radius_min)
        self._current_goal_radius_max = min(max_radius_cap, self._current_goal_radius_max)

        if level > 0:
            print(f"  [课程学习] Level {level}: 目标距离范围 [{self._current_goal_radius_min:.2f}, {self._current_goal_radius_max:.2f}]")

    def reset(self, seed=None, options=None):
        """重置环境，使用当前课程的目标距离"""
        # 调用父类 reset，但在采样目标前替换距离范围
        super().reset(seed=seed)

        # 使用课程学习的目标距离范围
        radius = self.np_random.uniform(
            self._current_goal_radius_min,
            self._current_goal_radius_max
        )
        angle = self.np_random.uniform(-np.pi, np.pi)
        self._target_pos = np.array([
            radius * np.cos(angle),
            radius * np.sin(angle),
            ARM_Z
        ], dtype=np.float32)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info


class CheckpointCallback(BaseCallback):
    """
    定期保存模型检查点，支持断点恢复

    每隔 save_freq 步保存一次模型到 assets/checkpoints/
    恢复训练后也能正确计算下一个保存时机。
    """

    def __init__(self, save_freq: int = 20000, save_path: str = "assets/checkpoints/", verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self._last_save_at = 0  # 记录上次保存时的步数（相对偏移）
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # 用相对偏移判断是否该保存（解决 resume 后 num_timesteps 非零点问题）
        if self.num_timesteps - self._last_save_at >= self.save_freq:
            checkpoint_path = os.path.join(self.save_path, f"sac_robot_arm_{self.num_timesteps}_steps.zip")
            self.model.save(checkpoint_path)
            self._last_save_at = self.num_timesteps
            if self.verbose > 0:
                print(f"\n[Checkpoint] 已保存: {checkpoint_path}")
        return True


class CurriculumCallback(BaseCallback):
    """
    课程学习回调：根据训练进度自动调整课程难度

    调整策略：
    - 每 30000 步增加一个课程等级
    - 最高等级为 10（完整任务难度）
    - 成功率连续两次 > 80% 时提前升级

    注意：会同时更新训练环境和评估环境的课程等级
    """

    def __init__(self, env: CurriculumRobotArmEnv, eval_env: CurriculumRobotArmEnv = None, success_threshold: float = 0.8):
        super().__init__(verbose=0)
        self.env = env
        self.eval_env = eval_env  # 评估环境引用
        self.success_threshold = success_threshold
        self.current_level = 0
        self.max_level = 5
        self.steps_per_level = 20000  # 每级 20K 步，总计 100K 步
        self.success_history = []  # 记录历史成功率

    def _on_step(self) -> bool:
        # 计算当前应该的课程等级
        target_level = min(
            self.max_level,
            int(self.num_timesteps / self.steps_per_level)
        )

        # 如果等级变化，更新训练环境和评估环境
        if target_level > self.current_level:
            self.current_level = target_level
            self.env.set_curriculum(self.current_level, self.num_timesteps)
            # 同步更新评估环境
            if self.eval_env is not None:
                self.eval_env.set_curriculum(self.current_level, self.num_timesteps)
            print(f"\n[课程学习] Level {self.current_level} | 步数 {self.num_timesteps}")
            print(f"  训练目标范围: [{self.env._current_goal_radius_min:.2f}, {self.env._current_goal_radius_max:.2f}] m")

        # 检查是否需要升级（连续高成功率）
        # 这部分在 SuccessRateCallback 中已记录，会在下一个评估周期生效
        return True


def train_robot_arm_with_curriculum():
    """
    使用课程学习训练机械臂

    课程设置（总计 100K 步）：
    - Level 0 (0~20K 步): 目标距离 0.1~0.2m（非常简单）
    - Level 3 (60K 步): 目标距离 0.2~0.4m（中等难度）
    - Level 5 (100K 步): 目标距离 0.3~0.6m（完整任务）
    """
    os.makedirs("assets", exist_ok=True)

    print("=" * 60)
    print("课程学习训练模式")
    print("=" * 60)
    print("课程设计（总步数 100K）：")
    print("  Level  0:  目标距离 0.10 ~ 0.20 m (非常简单)")
    print("  Level  3:  目标距离 0.20 ~ 0.40 m (中等难度)")
    print("  Level  5:  目标距离 0.30 ~ 0.60 m (完整任务)")
    print("=" * 60)

    # 验证环境接口
    print("\n── 验证环境接口 ──")
    test_env = CurriculumRobotArmEnv()
    check_env(test_env)
    test_env.close()
    print("check_env 通过！\n")

    # 创建课程学习环境（注意：先创建原始实例，再用 Monitor 包装）
    train_env_raw = CurriculumRobotArmEnv()  # 原始实例，用于回调
    train_env = Monitor(train_env_raw)       # 包装后，用于 SAC
    eval_env = CurriculumRobotArmEnv()

    # 检查是否有已保存的模型（优先从检查点恢复）
    checkpoint_dir = "assets/checkpoints"
    model_path = "assets/sac_robot_arm_curriculum"

    # 查找最新检查点
    def get_latest_checkpoint():
        if not os.path.exists(checkpoint_dir):
            return None, 0
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("sac_robot_arm_") and f.endswith("_steps.zip")]
        if not checkpoints:
            return None, 0
        # 从文件名提取步数：sac_robot_arm_100000_steps.zip
        steps_list = []
        for ckpt in checkpoints:
            try:
                steps = int(ckpt.split("_")[3])  # sac_robot_arm_20000_steps.zip → [sac, robot, arm, 20000, steps.zip]
                steps_list.append((steps, ckpt))
            except:
                pass
        if not steps_list:
            return None, 0
        steps_list.sort(reverse=True)
        return os.path.join(checkpoint_dir, steps_list[0][1]), steps_list[0][0]

    latest_checkpoint, checkpoint_steps = get_latest_checkpoint()
    model_exists = os.path.exists(model_path + ".zip")

    # 无论从检查点还是主模型恢复，都应该继续训练
    continue_training = latest_checkpoint is not None or model_exists

    if latest_checkpoint:
        # 优先从最新检查点恢复
        print(f"\n[Resume] 发现检查点: {latest_checkpoint}，继续训练...")
        model = SAC.load(latest_checkpoint, env=train_env, device='cuda' if torch.cuda.is_available() else 'cpu')
        start_timesteps = checkpoint_steps
        print(f"[Resume] 起始步数: ~{start_timesteps}")
    elif model_exists:
        # 回退到主模型文件
        print("\n[Resume] 发现已保存的模型，继续训练...")
        model = SAC.load(model_path, env=train_env, device='cuda' if torch.cuda.is_available() else 'cpu')
        start_timesteps = int(model.replay_buffer.size) if hasattr(model, 'replay_buffer') else 0
        print(f"[Resume] 起始步数: ~{start_timesteps}")
    else:
        # 新训练
        print("\n[New] 开始新的训练...")
        model = SAC(
            "MlpPolicy",
            train_env,
            verbose=1,
            buffer_size=50_000,
            learning_rate=3e-4,
            batch_size=256,
            learning_starts=500,
            gamma=0.99,
        )
        start_timesteps = 0
        continue_training = False

    # 创建回调（传入原始实例，用于同步课程等级）
    curriculum_callback = CurriculumCallback(train_env_raw, eval_env)
    success_callback = SuccessRateCallback(eval_env, eval_freq=20000, n_eval_episodes=10)
    checkpoint_callback = CheckpointCallback(save_freq=20000)  # 每2万步保存检查点

    # 评估回调：保存最佳模型
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='assets/best_model/',
        log_path='assets/logs/',
        eval_freq=20000,
        deterministic=True,
        render=False,
        n_eval_episodes=10
    )

    # 使用回调列表
    from stable_baselines3.common.callbacks import CallbackList
    callbacks = CallbackList([curriculum_callback, success_callback, checkpoint_callback, eval_callback])

    # 训练
    print("\n开始课程学习训练...")
    print("每 20000 步评估一次，每 200000 步增加课程难度")
    print("总训练步数: 10万步\n")

    # 继续训练模式：reset_num_timesteps=False 让回调知道继续计数
    continue_training = model_exists
    model.learn(
        total_timesteps=100_000,
        callback=callbacks,
        reset_num_timesteps=not continue_training,
        progress_bar=True
    )
    # 训练结束，保存最终模型
    final_path = f"assets/sac_robot_arm_curriculum_final_{int(model.num_timesteps)}_steps.zip"
    model.save(final_path)
    model.save("assets/sac_robot_arm_curriculum")  # 同时保存一份标准名称
    print(f"\n训练完成！最终模型已保存: {final_path}")

    train_env.close()
    eval_env.close()

    return model, success_callback


def train_robot_arm():
    """
    用 SAC 训练机械臂到达任务

    注意：这个任务比 Pendulum 难，需要更多步数或更好的奖励设计
    10000 步内成功率可能不高（正常现象），但奖励应该有明显提升
    """
    os.makedirs("assets", exist_ok=True)

    # 验证环境接口是否正确（check_env 会检查各种规范）
    print("── 验证环境接口 ──")
    test_env = RobotArmEnv()
    check_env(test_env)
    test_env.close()
    print("check_env 通过！环境接口符合 Gymnasium 规范\n")

    # 创建训练/评估环境
    train_env = Monitor(RobotArmEnv())
    eval_env  = RobotArmEnv()

    # TODO 3：初始化 SAC 并训练（参考 Step 3 的写法）
    #
    # 提示：
    #   model = SAC(
    #       "MlpPolicy",
    #       train_env,
    #       verbose=1,
    #       buffer_size=50_000,
    #       learning_rate=3e-4,
    #       batch_size=256,
    #       learning_starts=500,
    #       gamma=0.99,
    #   )
    #
    #   callback = SuccessRateCallback(eval_env, eval_freq=2000, n_eval_episodes=20)
    #   model.learn(total_timesteps=10_000, callback=callback)
    #   model.save("assets/sac_robot_arm")
    #
    model = SAC(
        "MlpPolicy",
        train_env,
        verbose=1,
        buffer_size=50_000,
        learning_rate=3e-4,
        batch_size=256,
        learning_starts=500,
        gamma=0.99,
    )

    callback = SuccessRateCallback(eval_env, eval_freq=2000, n_eval_episodes=20)
    model.learn(total_timesteps=10_0000, callback=callback)
    model.save("assets/sac_robot_arm")

    train_env.close()
    eval_env.close()
    return model, callback


# ── 绘图 ─────────────────────────────────────────────────────

def plot_training_results(callback: SuccessRateCallback):
    """
    画训练曲线：包含多个子图

    子图1: 累计奖励
    子图2: 成功率
    子图3: 平均步数
    子图4: 最终距离
    """
    if not callback.eval_timesteps:
        print("没有评估数据，跳过绘图")
        return

    timesteps = np.array(callback.eval_timesteps)
    rewards   = np.array(callback.eval_rewards)
    success   = np.array(callback.eval_success) * 100
    steps     = np.array(callback.eval_steps)
    distance  = np.array(callback.eval_distance)

    # 2x2 子图布局
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 子图1: 奖励曲线
    ax1 = axes[0, 0]
    ax1.plot(timesteps, rewards, 'o-', color='steelblue', linewidth=2, markersize=6)
    ax1.fill_between(timesteps, rewards, alpha=0.2, color='steelblue')
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Episode Reward (Higher is Better)')
    ax1.grid(True, alpha=0.3)

    # 子图2: 成功率曲线
    ax2 = axes[0, 1]
    ax2.plot(timesteps, success, 'o-', color='green', linewidth=2, markersize=6)
    ax2.axhline(80, color='orange', linestyle='--', alpha=0.7, label='80% Target')
    ax2.axhline(50, color='red', linestyle=':', alpha=0.5, label='50% Baseline')
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title(f'Success Rate (dist < {GOAL_TOLERANCE}m)')
    ax2.set_ylim(0, 105)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 子图3: 回合步数曲线
    ax3 = axes[1, 0]
    ax3.plot(timesteps, steps, 'o-', color='purple', linewidth=2, markersize=6)
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('Average Steps per Episode')
    ax3.set_title('Episode Length (Lower is Faster)')
    ax3.grid(True, alpha=0.3)

    # 子图4: 最终距离曲线
    ax4 = axes[1, 1]
    ax4.plot(timesteps, distance, 'o-', color='red', linewidth=2, markersize=6)
    ax4.axhline(GOAL_TOLERANCE, color='green', linestyle='--', alpha=0.7,
                label=f'Target ({GOAL_TOLERANCE}m)')
    ax4.set_xlabel('Training Steps')
    ax4.set_ylabel('Average Final Distance (m)')
    ax4.set_title('Final Distance to Goal (Lower is Better)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f'SAC Training on Robot Arm Reach Task\n'
                 f'buffer_size=50000, lr=3e-4, episodes={callback.n_eval_episodes}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    path = 'assets/step4_training_curves.png'
    plt.savefig(path, dpi=120, bbox_inches='tight')
    print(f"训练曲线已保存到 {path}")
    plt.close()

    # 保存详细数据到 CSV
    import csv
    csv_path = 'assets/step4_training_data.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timesteps', 'reward', 'success_rate', 'avg_steps', 'avg_distance',
                        'action_mean', 'action_std', 'ee_x', 'ee_y', 'goal_x', 'goal_y'])
        for i in range(len(timesteps)):
            writer.writerow([
                timesteps[i],
                f"{rewards[i]:.4f}",
                f"{success[i]:.2f}",
                f"{steps[i]:.2f}",
                f"{distance[i]:.4f}",
                f"{callback.eval_action_mean[i]:.4f}",
                f"{callback.eval_action_std[i]:.4f}",
                f"{callback.eval_ee_x[i]:.4f}",
                f"{callback.eval_ee_y[i]:.4f}",
                f"{callback.eval_goal_x[i]:.4f}",
                f"{callback.eval_goal_y[i]:.4f}"
            ])
    print(f"详细数据已保存到 {csv_path}")


# ── 可视化环境初始状态 ───────────────────────────────────────

def demo_env():
    """演示自定义环境的 API 使用方式（不训练，只展示接口）"""
    print("── 自定义环境 API 演示 ──")
    env = RobotArmEnv()

    print(f"观测空间：{env.observation_space}")
    print(f"动作空间：{env.action_space}")
    print(f"观测维度：{env.observation_space.shape}")
    print(f"动作维度：{env.action_space.shape}")

    # 重置
    obs, info = env.reset(seed=42)
    print(f"\n重置后：")
    print(f"  观测（ee_pos + joint_angles）= {obs.round(3)}")
    print(f"  末端位置 = {info['ee_pos']}")
    print(f"  目标位置 = {info['target']}")
    print(f"  到目标距离 = {info['distance']:.3f} m")

    # 执行几步随机动作
    print(f"\n执行 5 步随机动作：")
    for i in range(5):
        action = env.action_space.sample()   # 随机动作
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  步 {i+1}: 动作={action.round(3)}, 奖励={reward:.4f}, "
              f"距离={info['distance']:.3f}m, 成功={info['success']}")

    env.close()
    print("\n环境 API 演示完毕！\n")


# ── 可视化工作空间 ───────────────────────────────────────────

def visualize_workspace():
    """可视化机械臂的工作空间和几个示例姿态"""
    os.makedirs("assets", exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('3-DOF 机械臂工作空间可视化\n'
                 f'连杆长度：{ARM_LENGTHS}，目标采样范围：r∈[{GOAL_RADIUS_MIN},{GOAL_RADIUS_MAX}]')

    # 画几个示例姿态
    sample_configs = [
        np.array([0.0,   0.0,   0.0]),
        np.array([np.pi/4,  np.pi/4,  np.pi/4]),
        np.array([np.pi/2,  -np.pi/4, 0.0]),
        np.array([-np.pi/3, np.pi/3,  0.0]),
        np.array([3*np.pi/4, -np.pi/4, 0.0]),
    ]

    colors = plt.cm.tab10(np.linspace(0, 0.5, len(sample_configs)))
    for q, color in zip(sample_configs, colors):
        positions = get_link_positions(q)
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        ax.plot(xs, ys, 'o-', color=color, linewidth=2, markersize=6, alpha=0.8)

    # 画工作空间边界圆
    theta_range = np.linspace(0, 2*np.pi, 200)
    max_r = sum(ARM_LENGTHS)
    min_r = max(0, ARM_LENGTHS[0] - ARM_LENGTHS[1] - ARM_LENGTHS[2])
    ax.plot(max_r*np.cos(theta_range), max_r*np.sin(theta_range),
            'r--', alpha=0.4, linewidth=1.5, label=f'最大可达半径 ({max_r}m)')
    ax.plot(GOAL_RADIUS_MAX*np.cos(theta_range), GOAL_RADIUS_MAX*np.sin(theta_range),
            'g--', alpha=0.4, linewidth=1.5, label=f'目标采样外边界 ({GOAL_RADIUS_MAX}m)')
    ax.plot(GOAL_RADIUS_MIN*np.cos(theta_range), GOAL_RADIUS_MIN*np.sin(theta_range),
            'b--', alpha=0.4, linewidth=1.5, label=f'目标采样内边界 ({GOAL_RADIUS_MIN}m)')

    ax.plot(0, 0, 'ks', markersize=12, zorder=5, label='基座')
    ax.legend(loc='upper right', fontsize=9)

    path = 'assets/step4_workspace.png'
    plt.savefig(path, dpi=120, bbox_inches='tight')
    print(f"工作空间图已保存到 {path}")
    plt.close()


# ── 主程序 ──────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Step 4：自定义 Gymnasium 环境 — 机械臂末端到达任务")
    print("=" * 60)

    # 1. 演示环境 API（不训练）
    demo_env()

    # 2. 可视化工作空间
    visualize_workspace()

    # 3. 选择训练模式
    print("\n" + "=" * 60)
    print("训练模式选择:")
    print("  1. 普通训练 (train_robot_arm) - 直接训练完整任务")
    print("  2. 课程学习 (train_robot_arm_with_curriculum) - 从简单到难")
    print("=" * 60)

    # 默认使用课程学习（效果更好）
    USE_CURRICULUM = True  # 修改为 False 使用普通训练

    if USE_CURRICULUM:
        print("\n>>> 使用课程学习模式 <<<\n")
        model, callback = train_robot_arm_with_curriculum()
    else:
        print("\n>>> 使用普通训练模式 <<<\n")
        model, callback = train_robot_arm()

    # 4. 画图
    plot_training_results(callback)

    print("\n── 任务二完成总结 ──")
    print("Step 1：Q-table（表格式，离散动作，无框架）")
    print("Step 2：REINFORCE（策略梯度，纯 numpy，体验反向传播）")
    print("Step 3：SAC（off-policy，最大熵，SB3 框架）")
    print("Step 4：自定义 Gym 环境（工程核心能力）")
    print("  └─ 课程学习：渐进式增加任务难度，提高训练效率")
    print()
    print("你现在掌握的技能：")
    print("  ✓ 能解释 Bellman 方程和 Q-learning 的更新过程")
    print("  ✓ 能手推策略梯度定理并实现反向传播")
    print("  ✓ 能用 SB3 快速跑通 SAC/PPO")
    print("  ✓ 能自定义 Gymnasium 环境（机器人任务的核心工程能力）")
    print("  ✓ 能使用课程学习提高训练效率")
    print()
    print("面试准备：")
    print("  → 准备好解释 SAC vs PPO 的区别（见 GUIDE.md 面试考点）")
    print("  → 准备好解释稀疏奖励的解决方案（HER, Curriculum Learning）")
    print("  → 准备好解释 sim2real gap 和 Domain Randomization")
