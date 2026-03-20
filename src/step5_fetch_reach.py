"""
Step 5: PyBullet UR5 FetchReach-like 环境

使用 PyBullet 模拟 7-DoF UR5 机械臂，复刻 FetchReach 标准任务设计。

核心设计：
  - 观测空间：Goal-Aware 字典结构 {observation, achieved_goal, desired_goal}
  - 动作空间：末端执行器 Cartesian 位移控制 (dx, dy, dz)
  - 奖励模式：Sparse + Dense 双模式可切换
  - 物理引擎：PyBullet（免费开源，对应工业标准 MuJoCo）

运行：python step5_fetch_reach.py

依赖：pip install pybullet gymnasium stable-baselines3
"""

import numpy as np
import os
import time

# 配置中文字体
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from typing import Optional, Tuple, Dict, Any

try:
    import pybullet as p
    import pybullet_data
except ImportError:
    raise ImportError("缺少 pybullet：请运行 pip install pybullet")

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    raise ImportError("缺少 gymnasium：请运行 pip install gymnasium")

try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
except ImportError:
    raise ImportError("缺少 stable-baselines3：请运行 pip install stable-baselines3")

import torch


# ── 任务参数 ──────────────────────────────────────────────────

# Panda 初始位置
PANDA_INITIAL_POS = [0.0, 0.0, 0.0]
PANDA_INITIAL_ORIENTATION = [0, 0, 0, 1]  # 默认朝上

# 末端执行器初始位置（相对于基座）
EE_INITIAL_POS = np.array([0.5, 0.0, 0.5], dtype=np.float32)

# 目标采样范围
GOAL_MIN_RANGE = -0.15
GOAL_MAX_RANGE = 0.15
GOAL_Z_MIN = 0.4
GOAL_Z_MAX = 0.7

# 成功阈值
SUCCESS_THRESHOLD = 0.05  # 5cm

# 动作限制（每步末端位移上限）
ACTION_LIMIT = 0.05  # 5cm

# 最大步数
MAX_EPISODE_STEPS = 100


# ── PyBullet 工具函数 ────────────────────────────────────────

def get_ee_pose(body_uid: int, ee_link: int = 11) -> Tuple[np.ndarray, np.ndarray]:
    """获取末端执行器位置和四元数"""
    ee_pos, ee_quat = p.getLinkState(body_uid, ee_link, computeForwardKinematics=True)[:2]
    return np.array(ee_pos, dtype=np.float32), np.array(ee_quat, dtype=np.float32)


def get_joint_states(body_uid: int) -> Tuple[np.ndarray, np.ndarray]:
    """获取关节角度和速度"""
    joint_states = p.getJointStates(body_uid, range(p.getNumJoints(body_uid)))
    joint_angles = np.array([js[0] for js in joint_states], dtype=np.float32)
    joint_vels = np.array([js[1] for js in joint_states], dtype=np.float32)
    return joint_angles, joint_vels


def compute_inverse_kinematics(body_uid: int, target_pos: np.ndarray, link_index: int = 11, target_quat: np.ndarray = None) -> np.ndarray:
    """计算逆运动学"""
    if target_quat is None:
        target_quat = [0, 0, 0, 1]
    joint_poses = p.calculateInverseKinematics(body_uid, link_index, target_pos, target_quat)
    return np.array(joint_poses, dtype=np.float32)


# ── 自定义 FetchReach 环境 ────────────────────────────────────

class FetchReachEnv(gym.Env):
    """
    基于 PyBullet UR5 的 FetchReach 风格环境

    观测空间（Goal-Aware 结构）：
      observation: [末端位置(3) + 末端速度(3) + 关节角度(6)] = 15维
      achieved_goal: 末端实际位置 (3,)
      desired_goal: 目标位置 (3,)

    动作空间：
      末端 Cartesian 位移控制 (dx, dy, dz)

    奖励模式：
      Dense: -norm(achieved_goal - desired_goal)
      Sparse: 0 if dist < threshold else -1
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: Optional[str] = None, dense_reward: bool = True):
        super().__init__()

        self.render_mode = render_mode
        self.dense_reward = dense_reward
        self._step_count = 0
        self._goal = np.zeros(3, dtype=np.float32)
        self._initial_ee_pos = EE_INITIAL_POS.copy()

        # PyBullet 客户端 ID
        self._physics_client_id = -1
        self._robot_uid = -1

        # ── 观测空间 ────────────────────────────────────────
        # observation: 末端位置(3) + 关节角度(6) = 9维
        # 使用无界空间，关节角度可能超出 [-1, 1]
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(
                low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
            ),
            'achieved_goal': spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
            ),
            'desired_goal': spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
            ),
        })

        # ── 动作空间 ────────────────────────────────────────
        # 末端执行器 Cartesian 位移
        self.action_space = spaces.Box(
            low=-ACTION_LIMIT, high=ACTION_LIMIT, shape=(3,), dtype=np.float32
        )

    def _setup_pybullet(self):
        """初始化 PyBullet 物理引擎"""
        if self._physics_client_id >= 0:
            return

        if self.render_mode == "human":
            self._physics_client_id = p.connect(p.GUI)
        else:
            self._physics_client_id = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(0.01)

        # 加载 Franka Panda
        self._robot_uid = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=PANDA_INITIAL_POS,
            baseOrientation=PANDA_INITIAL_ORIENTATION,
            useFixedBase=True
        )

        # Panda 末端执行器 link index (11 = panda_grasptarget_hand)
        self._ee_link_index = 11

        # 设置关节最大力
        num_joints = p.getNumJoints(self._robot_uid)
        for i in range(num_joints):
            p.setJointMotorControl2(
                self._robot_uid, i,
                p.POSITION_CONTROL,
                force=p.getJointInfo(self._robot_uid, i)[10]
            )

        # 移动到初始位置
        initial_joint_poses = compute_inverse_kinematics(
            self._robot_uid, EE_INITIAL_POS, self._ee_link_index
        )
        for i in range(min(7, len(initial_joint_poses))):
            p.setJointMotorControl2(
                self._robot_uid, i,
                p.POSITION_CONTROL,
                targetPosition=initial_joint_poses[i],
                force=500
            )

        # 步进模拟
        for _ in range(100):
            p.stepSimulation()

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """获取 Goal-Aware 观测"""
        ee_pos, _ = get_ee_pose(self._robot_uid)
        joint_angles, joint_vels = get_joint_states(self._robot_uid)

        # 观测 = 末端位置(3) + 末端速度(3) + 关节角度(6) = 12维
        # 注意：简化版，不包含速度
        obs = np.concatenate([
            ee_pos.astype(np.float32),
            joint_angles[:6].astype(np.float32),  # 6个关节角度
        ])

        return {
            'observation': obs,
            'achieved_goal': ee_pos.copy(),
            'desired_goal': self._goal.copy(),
        }

    def _compute_reward(self) -> float:
        """计算奖励"""
        ee_pos, _ = get_ee_pose(self._robot_uid)
        dist = float(np.linalg.norm(ee_pos - self._goal))

        if self.dense_reward:
            return -dist
        else:
            return 0.0 if dist < SUCCESS_THRESHOLD else -1.0

    def _is_success(self) -> bool:
        """判断是否成功"""
        ee_pos, _ = get_ee_pose(self._robot_uid)
        dist = np.linalg.norm(ee_pos - self._goal)
        return bool(dist < SUCCESS_THRESHOLD)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """重置环境"""
        super().reset(seed=seed)

        self._setup_pybullet()
        self._step_count = 0

        # 采样目标位置（起点 + 随机偏移）
        if options and 'goal' in options:
            self._goal = options['goal'].copy()
        else:
            self._goal = self._initial_ee_pos + self.np_random.uniform(
                low=[GOAL_MIN_RANGE, GOAL_MIN_RANGE, GOAL_Z_MIN - EE_INITIAL_POS[2]],
                high=[GOAL_MAX_RANGE, GOAL_MAX_RANGE, GOAL_Z_MAX - EE_INITIAL_POS[2]],
                size=3
            ).astype(np.float32)

        # 重置关节到初始位置
        initial_joint_poses = compute_inverse_kinematics(
            self._robot_uid, EE_INITIAL_POS
        )
        for i in range(min(6, len(initial_joint_poses))):
            p.setJointMotorControl2(
                self._robot_uid, i,
                p.POSITION_CONTROL,
                targetPosition=initial_joint_poses[i],
                force=500
            )

        for _ in range(50):
            p.stepSimulation()

        obs = self._get_observation()
        info = {'is_success': False, 'distance': np.linalg.norm(
            obs['achieved_goal'] - obs['desired_goal']
        )}

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """执行一步动作"""
        self._step_count += 1

        # 解析动作（末端位移）
        action = np.clip(action, self.action_space.low, self.action_space.high)
        dx, dy, dz = action

        # 获取当前末端位置
        ee_pos, ee_quat = get_ee_pose(self._robot_uid)

        # 计算新的目标位置
        new_ee_pos = ee_pos + np.array([dx, dy, dz], dtype=np.float32)

        # 限幅
        new_ee_pos = np.clip(new_ee_pos, [0.3, -0.3, 0.2], [0.8, 0.3, 0.9])

        # 逆运动学求解
        target_joint_poses = compute_inverse_kinematics(
            self._robot_uid, new_ee_pos, self._ee_link_index, ee_quat
        )

        # 设置关节目标位置 (Panda 有 7 个关节)
        for i in range(min(7, len(target_joint_poses))):
            p.setJointMotorControl2(
                self._robot_uid, i,
                p.POSITION_CONTROL,
                targetPosition=target_joint_poses[i],
                force=500
            )

        # 步进模拟
        for _ in range(10):
            p.stepSimulation()

        # 计算观测、奖励
        obs = self._get_observation()
        reward = self._compute_reward()
        success = self._is_success()

        dist = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
        terminated = bool(success)
        truncated = bool(self._step_count >= MAX_EPISODE_STEPS)

        info = {
            'is_success': success,
            'distance': dist,
        }

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """渲染"""
        if self.render_mode == "rgb_array":
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0.5, 0, 0.3],
                distance=1.5,
                yaw=45,
                pitch=-30,
                roll=0,
                upAxisIndex=2
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60, aspect=1.0, nearVal=0.1, farVal=100.0
            )
            (_, _, px, _, _) = p.getCameraImage(
                width=640, height=480,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )
            rgb_array = np.array(px, dtype=np.uint8)
            rgb_array = rgb_array[:, :, :3].reshape((480, 640, 3))
            return rgb_array
        return None

    def close(self):
        """关闭环境"""
        if self._physics_client_id >= 0:
            p.disconnect(self._physics_client_id)
            self._physics_client_id = -1


# ── 奖励模式包装器 ────────────────────────────────────────────

class SparseRewardWrapper(gym.ObservationWrapper):
    """稀疏奖励包装器"""

    def __init__(self, env):
        super().__init__(env)
        self.env.dense_reward = False


class DenseRewardWrapper(gym.ObservationWrapper):
    """稠密奖励包装器"""

    def __init__(self, env):
        super().__init__(env)
        self.env.dense_reward = True


# ── 回调函数（复用 Step4）────────────────────────────────────

class SuccessRateCallback(BaseCallback):
    """
    记录训练过程中的详细信息
    """

    def __init__(self, eval_env, eval_freq: int = 2000, n_eval_episodes: int = 20):
        super().__init__(verbose=0)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes

        self.eval_timesteps = []
        self.eval_rewards = []
        self.eval_success = []
        self.eval_steps = []
        self.eval_distance = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            rewards_list = []
            success_list = []
            steps_list = []
            distance_list = []

            for _ in range(self.n_eval_episodes):
                obs, info = self.eval_env.reset()
                ep_reward = 0.0
                ep_success = False
                ep_steps = 0

                for _ in range(MAX_EPISODE_STEPS):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    ep_reward += reward
                    ep_steps += 1

                    if info.get("is_success", False):
                        ep_success = True
                    if terminated or truncated:
                        break

                final_dist = info.get('distance', 0.0)
                distance_list.append(final_dist)
                rewards_list.append(ep_reward)
                success_list.append(float(ep_success))
                steps_list.append(ep_steps)

            self.eval_timesteps.append(self.num_timesteps)
            self.eval_rewards.append(np.mean(rewards_list))
            self.eval_success.append(np.mean(success_list))
            self.eval_steps.append(np.mean(steps_list))
            self.eval_distance.append(np.mean(distance_list))

            print(f"\n  ═══════════════════════════════════════")
            print(f"  [{self.num_timesteps:6d} steps] 训练统计")
            print(f"  ───────────────────────────────────────")
            print(f"  奖励:     {np.mean(rewards_list):8.2f} ± {np.std(rewards_list):.2f}")
            print(f"  成功率:   {np.mean(success_list)*100:7.1f}%")
            print(f"  回合步数: {np.mean(steps_list):7.1f} ± {np.std(steps_list):.1f}")
            print(f"  最终距离: {np.mean(distance_list):7.3f} m")
            print(f"  ═══════════════════════════════════════")

        return True


class CheckpointCallback(BaseCallback):
    """定期保存模型检查点"""

    def __init__(self, save_freq: int = 50000, save_path: str = "assets/checkpoints/", verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self._last_save_at = 0
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_save_at >= self.save_freq:
            checkpoint_path = os.path.join(
                self.save_path, f"sac_fetch_reach_{self.num_timesteps}_steps.zip"
            )
            self.model.save(checkpoint_path)
            self._last_save_at = self.num_timesteps
            if self.verbose > 0:
                print(f"\n[Checkpoint] 已保存: {checkpoint_path}")
        return True


# ── 训练函数 ─────────────────────────────────────────────────

def train_fetch_reach(dense: bool = True, total_timesteps: int = 200_000):
    """训练 FetchReach 环境"""

    os.makedirs("assets", exist_ok=True)

    reward_mode = "dense" if dense else "sparse"
    print("=" * 60)
    print(f"训练 FetchReach（{reward_mode} 奖励）")
    print("=" * 60)

    # 创建环境
    train_env_raw = FetchReachEnv(render_mode=None, dense_reward=dense)
    train_env = Monitor(train_env_raw)
    eval_env = FetchReachEnv(render_mode=None, dense_reward=dense)

    # 验证环境
    print("\n── 验证环境接口 ──")
    check_env(eval_env)
    print("check_env 通过！\n")

    # 创建模型
    print("创建 SAC 模型...")
    model = SAC(
        "MultiInputPolicy",
        train_env,
        verbose=1,
        buffer_size=100_000,
        learning_rate=3e-4,
        batch_size=256,
        learning_starts=1000,
        gamma=0.99,
        tau=0.005,
    )

    # 回调
    success_callback = SuccessRateCallback(eval_env, eval_freq=5000, n_eval_episodes=20)
    checkpoint_callback = CheckpointCallback(save_freq=50000)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='assets/best_model/',
        log_path='assets/logs/',
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=10
    )

    callbacks = CallbackList([success_callback, checkpoint_callback, eval_callback])

    # 训练
    print("\n开始训练...\n")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True
    )

    # 保存
    final_path = f"assets/sac_fetch_reach_{reward_mode}_final_{int(model.num_timesteps)}_steps.zip"
    model.save(final_path)
    model.save("assets/sac_fetch_reach")
    print(f"\n训练完成！模型已保存: {final_path}")

    train_env.close()
    eval_env.close()

    return model, success_callback


def plot_training_results(callback: SuccessRateCallback, reward_mode: str = "dense"):
    """绘制训练曲线"""
    if not callback.eval_timesteps:
        print("没有评估数据，跳过绘图")
        return

    timesteps = np.array(callback.eval_timesteps)
    rewards = np.array(callback.eval_rewards)
    success = np.array(callback.eval_success) * 100
    steps = np.array(callback.eval_steps)
    distance = np.array(callback.eval_distance)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 奖励曲线
    ax1 = axes[0, 0]
    ax1.plot(timesteps, rewards, 'o-', color='steelblue', linewidth=2, markersize=6)
    ax1.fill_between(timesteps, rewards, alpha=0.2, color='steelblue')
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Average Reward')
    ax1.set_title(f'Episode Reward ({reward_mode} reward)')
    ax1.grid(True, alpha=0.3)

    # 成功率曲线
    ax2 = axes[0, 1]
    ax2.plot(timesteps, success, 'o-', color='green', linewidth=2, markersize=6)
    ax2.axhline(80, color='orange', linestyle='--', alpha=0.7, label='80% Target')
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title(f'Success Rate (dist < {SUCCESS_THRESHOLD}m)')
    ax2.set_ylim(0, 105)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 回合步数
    ax3 = axes[1, 0]
    ax3.plot(timesteps, steps, 'o-', color='purple', linewidth=2, markersize=6)
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('Average Steps per Episode')
    ax3.set_title('Episode Length')
    ax3.grid(True, alpha=0.3)

    # 最终距离
    ax4 = axes[1, 1]
    ax4.plot(timesteps, distance, 'o-', color='red', linewidth=2, markersize=6)
    ax4.axhline(SUCCESS_THRESHOLD, color='green', linestyle='--', alpha=0.7,
                label=f'Target ({SUCCESS_THRESHOLD}m)')
    ax4.set_xlabel('Training Steps')
    ax4.set_ylabel('Average Final Distance (m)')
    ax4.set_title('Final Distance to Goal')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f'SAC Training on FetchReach ({reward_mode} reward)\n'
                 f'Total steps={callback.eval_timesteps[-1] if callback.eval_timesteps else 0}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    path = f'assets/step5_training_curves_{reward_mode}.png'
    plt.savefig(path, dpi=120, bbox_inches='tight')
    print(f"训练曲线已保存到 {path}")
    plt.close()


# ── 可视化 ───────────────────────────────────────────────────

def demo_env():
    """演示环境 API"""
    print("── FetchReach 环境 API 演示 ──")
    env = FetchReachEnv(render_mode=None, dense_reward=True)

    print(f"观测空间：{env.observation_space}")
    print(f"动作空间：{env.action_space}")

    obs, info = env.reset(seed=42)
    print(f"\n重置后：")
    print(f"  观测维度: observation={obs['observation'].shape}, "
          f"achieved_goal={obs['achieved_goal'].shape}, "
          f"desired_goal={obs['desired_goal'].shape}")
    print(f"  目标位置 = {obs['desired_goal']}")
    print(f"  初始距离 = {info['distance']:.3f} m")

    print(f"\n执行 5 步随机动作：")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  步 {i+1}: 奖励={reward:.4f}, 距离={info['distance']:.3f}m, "
              f"成功={info['is_success']}")

    env.close()
    print("\n环境 API 演示完毕！\n")


def create_video_gif(model_path: str = "assets/sac_fetch_reach.zip", output_path: str = "assets/fetch_reach_demo.gif"):
    """生成训练效果 GIF"""
    try:
        import imageio

        env = FetchReachEnv(render_mode='rgb_array', dense_reward=True)
        model = SAC.load(model_path)

        frames = []
        obs, _ = env.reset()

        for _ in range(50):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            if terminated or truncated:
                obs, _ = env.reset()

        if frames:
            imageio.mimsave(output_path, frames, fps=10)
            print(f"GIF 已保存: {output_path}")

        env.close()
    except Exception as e:
        print(f"生成 GIF 失败: {e}")


# ── 主程序 ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Step 5: PyBullet UR5 FetchReach-like 环境")
    print("=" * 60)

    # 1. 演示环境 API
    demo_env()

    # 2. 训练选项
    print("\n训练模式选择:")
    print("  1. Dense 奖励训练（推荐，更快收敛）")
    print("  2. Sparse 奖励训练（接近真实 FetchReach）")
    print("  3. 两种都跑（对比实验）")

    choice = "3"

    if choice == "1":
        model, callback = train_fetch_reach(dense=True, total_timesteps=200_000)
        plot_training_results(callback, "dense")
    elif choice == "2":
        model, callback = train_fetch_reach(dense=False, total_timesteps=200_000)
        plot_training_results(callback, "sparse")
    else:
        print("\n>>> Dense 奖励训练 <<<")
        model_dense, callback_dense = train_fetch_reach(dense=True, total_timesteps=200_000)
        plot_training_results(callback_dense, "dense")

        print("\n>>> Sparse 奖励训练 <<<")
        model_sparse, callback_sparse = train_fetch_reach(dense=False, total_timesteps=200_000)
        plot_training_results(callback_sparse, "sparse")

        # 对比曲线
        print("\n>>> 绘制对比曲线 <<<")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        for label, callback, color in [
            ("Dense", callback_dense, 'blue'),
            ("Sparse", callback_sparse, 'red')
        ]:
            if callback.eval_timesteps:
                timesteps = np.array(callback.eval_timesteps)
                success = np.array(callback.eval_success) * 100

                axes[0, 1].plot(timesteps, success, 'o-', color=color,
                               linewidth=2, markersize=4, label=label, alpha=0.7)

        axes[0, 1].set_xlabel('Training Steps')
        axes[0, 1].set_ylabel('Success Rate (%)')
        axes[0, 1].set_title('Dense vs Sparse Reward')
        axes[0, 1].set_ylim(0, 105)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        plt.suptitle('Dense vs Sparse Reward Comparison', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig('assets/step5_reward_comparison.png', dpi=120)
        print("对比曲线已保存: assets/step5_reward_comparison.png")
        plt.close()

    print("\n── Step 5 完成 ──")
    print("下一步：step5b_official_fetch.py (官方 FetchReach 对照)")
    print("       step6_rl_reach_benchmark.py (多算法基准)")
