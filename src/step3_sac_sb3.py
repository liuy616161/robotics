"""
Step 3：SAC（Soft Actor-Critic）用 Stable-Baselines3

任务：在 Pendulum-v1 环境上用 SAC 训练连续控制策略，理解关键超参数，保存并评估模型
环境：Pendulum-v1（倒摆，连续动作，支持 CPU/GPU 训练）

运行：python3.9 src/step3_sac_sb3.py

前置：pip3 install gymnasium stable-baselines3

GPU 加速：
  - 代码自动检测 CUDA GPU，有 GPU 时自动使用
  - 神经网络前向/反向传播在 GPU 上执行
  - 对于简单环境（如 Pendulum）GPU 提升有限
  - 复杂环境（机械臂、人形机器人）GPU 加速显著

SAC 为什么重要？
  - Step 1 的 Q-learning 只能处理离散动作
  - Step 2 的 REINFORCE 梯度方差大、样本效率低
  - SAC = Off-policy（高样本效率）+ 最大熵（强探索）+ 连续动作
  - 智元、宇树等公司的机器人运动控制大量使用 SAC / TD3

依赖：numpy, matplotlib, gymnasium, stable-baselines3, torch
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'AR PL UMing CN', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
import os
import time

try:
    import gymnasium as gym
    from stable_baselines3 import SAC, PPO
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
except ImportError as e:
    raise ImportError(f"缺少依赖：{e}\n请运行：pip3 install gymnasium stable-baselines3")


# ── GPU 配置 ──────────────────────────────────────────────────
import torch

# 自动检测并配置 GPU
if torch.cuda.is_available():
    DEVICE = "cuda"
    GPU_NAME = torch.cuda.get_device_name(0)
    GPU_COUNT = torch.cuda.device_count()
    print(f"检测到 GPU：{GPU_NAME}，共 {GPU_COUNT} 张")
else:
    DEVICE = "cpu"
    GPU_NAME = None
    GPU_COUNT = 0
    print("未检测到 GPU，使用 CPU 训练")

# ── 环境配置 ──────────────────────────────────────────────────
ENV_ID = "Pendulum-v1"   # 倒摆环境（连续动作，CPU友好）

# ── SAC 超参数（每个都有详细解释）───────────────────────────
SAC_PARAMS = {
    # ── 网络结构 ────────────────────────────────────────────
    "policy": "MlpPolicy",   # 多层感知机策略网络（输入状态，输出动作）
                              # 默认结构：[256, 256] 两个隐层，tanh 激活

    "device": DEVICE,        # 设备：'cuda' 使用 GPU，'cpu' 使用 CPU
                              # GPU 加速：神经网络前向/反向传播在 GPU 上进行
                              # 连续控制任务 GPU 提升有限（计算量小），但复杂环境有用

    # ── 核心超参数 ──────────────────────────────────────────
    "buffer_size": 50_000,   # Replay Buffer 大小
                              # 存储过去的 (s, a, r, s', done) 四元组
                              # Off-policy 的关键：可以重复利用历史数据
                              # 太小→覆盖旧数据太快；太大→内存压力

    "learning_rate": 3e-4,   # Adam 优化器学习率（Actor 和 Critic 共用）
                              # 太大→不稳定；太小→收敛太慢
                              # 3e-4 是 SAC 的经典默认值（源自原论文）

    "batch_size": 256,        # 每次从 buffer 随机采样的样本数
                              # 更大 batch → 梯度更稳定，但每步计算更慢

    "gamma": 0.99,            # 折扣因子（与 Step 1/2 相同含义）

    "tau": 0.005,             # 目标网络软更新系数
                              # target_net ← τ × online_net + (1-τ) × target_net
                              # τ=0.005 → 目标网络缓慢跟踪，提供稳定的训练目标
                              # （对比硬更新：每隔 N 步直接复制，SAC 用软更新）

    "ent_coef": "auto",       # 熵系数（温度参数 α）
                              # SAC 的最大熵目标：J = E[Σ r_t + α·H(π)]
                              # "auto" = 自动调整 α，目标熵 = -dim(action_space)
                              # α 大 → 鼓励更多探索；α 小 → 偏向利用

    "learning_starts": 1000,  # 前 N 步随机探索，不更新网络
                              # 目的：先填充 replay buffer，避免过拟合于极少量数据
                              # 在这 1000 步内，动作完全随机

    "verbose": 1,             # 打印训练信息（0=不打印，1=打印，2=调试）
}

TOTAL_TIMESTEPS = 50_000     # 总训练步数（GPU 训练可以更快，多训练一些）
EVAL_FREQ       = 5_000      # 每隔多少步评估一次（记录训练曲线）
N_EVAL_EPISODES = 10         # 每次评估用多少个 episode
SAVE_PATH       = "assets/sac_pendulum"


# ── 自定义回调：记录训练曲线和状态 ─────────────────────────────────

class RewardLoggerCallback(BaseCallback):
    """
    自定义 SB3 回调：定期评估模型并记录奖励和观测状态

    记录内容：
    - eval_timesteps: 评估时的训练步数
    - eval_rewards: 评估平均奖励
    - eval_stds: 奖励标准差
    - eval_obs_mean: 观测向量均值 [cos(θ), sin(θ), θ̇]
    - eval_obs_std: 观测向量标准差
    """

    def __init__(self, eval_env, eval_freq: int = 3000, n_eval_episodes: int = 10):
        super().__init__(verbose=0)
        self.eval_env       = eval_env
        self.eval_freq      = eval_freq
        self.n_eval_episodes = n_eval_episodes

        # 记录评估结果
        self.eval_timesteps = []    # 评估时的总步数
        self.eval_rewards   = []    # 评估平均奖励
        self.eval_stds      = []    # 评估奖励标准差

        # 新增：记录观测向量
        self.eval_obs_mean = []     # 观测向量均值
        self.eval_obs_std = []      # 观测向量标准差
        self.eval_actions = []       # 动作均值

    def _on_step(self) -> bool:
        """每步都会被调用；返回 False 会提前终止训练"""
        if self.n_calls % self.eval_freq == 0:
            # 运行评估并记录观测
            obs_list = []
            reward_list = []
            action_list = []

            for _ in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset()
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                    done = terminated or truncated

                    obs_list.append(obs)
                    action_list.append(action)
                    reward_list.append(reward)

            # 计算统计数据
            import numpy as np
            obs_array = np.array(obs_list)
            action_array = np.array(action_list).flatten()
            reward_array = np.array(reward_list)

            mean_reward = np.mean(reward_array)
            std_reward = np.std(reward_array)
            obs_mean = np.mean(obs_array, axis=0)
            obs_std = np.std(obs_array, axis=0)
            action_mean = np.mean(action_array)

            self.eval_timesteps.append(self.num_timesteps)
            self.eval_rewards.append(mean_reward)
            self.eval_stds.append(std_reward)
            self.eval_obs_mean.append(obs_mean)
            self.eval_obs_std.append(obs_std)
            self.eval_actions.append(action_mean)

            # 打印观测状态
            print(f"\n  [评估] 步数: {self.num_timesteps}")
            print(f"         奖励: {mean_reward:.1f} ± {std_reward:.1f}")
            print(f"         观测: cos(θ)={obs_mean[0]:.3f}, sin(θ)={obs_mean[1]:.3f}, θ̇={obs_mean[2]:.3f}")
            print(f"         动作均值: {action_mean:.3f}")

        return True  # 继续训练


# ── 训练 SAC ─────────────────────────────────────────────────

def train_sac():
    """
    训练 SAC 模型并记录曲线

    TODO 1：补全 SAC 初始化和训练（见下面的 TODO 标记）
    """
    os.makedirs("assets", exist_ok=True)

    # 创建训练环境（Monitor 包装：自动记录每 episode 的奖励和长度）
    train_env = Monitor(gym.make(ENV_ID))
    eval_env  = Monitor(gym.make(ENV_ID))

    # 打印环境空间信息
    obs_space = train_env.observation_space
    act_space = train_env.action_space

    print("=" * 60)
    print(f"训练 SAC on {ENV_ID}")
    print(f"总步数：{TOTAL_TIMESTEPS}，评估频率：每 {EVAL_FREQ} 步")
    print(f"使用设备：{DEVICE}" + (f" ({GPU_NAME})" if GPU_NAME else ""))
    print("=" * 60)

    # 打印观测和动作空间详细信息
    print("\n【环境空间定义】")
    print(f"  观测空间 (Observation Space):")
    print(f"    类型: {type(obs_space).__name__}")
    print(f"    形状: {obs_space.shape}")
    print(f"    范围: low={obs_space.low}, high={obs_space.high}")
    print(f"    ─────────────────────────────")
    print(f"    obs[0] = cos(θ)    范围 [{obs_space.low[0]}, {obs_space.high[0]}]")
    print(f"    obs[1] = sin(θ)    范围 [{obs_space.low[1]}, {obs_space.high[1]}]")
    print(f"    obs[2] = θ̇ (角速度) 范围 [{obs_space.low[2]}, {obs_space.high[2]}]")

    print(f"\n  动作空间 (Action Space):")
    print(f"    类型: {type(act_space).__name__}")
    print(f"    形状: {act_space.shape}")
    print(f"    范围: [{act_space.low[0]}, {act_space.high[0]}]")
    print(f"    action[0] = 力矩 (torque)")

    print(f"\n  奖励函数: r = -(θ² + 0.1×θ̇² + 0.001×u²)")
    print(f"    每步奖励范围: 最优 ≈ 0 (杆竖直)，最差 ≈ -16.27 (杆倒下)")
    print("=" * 60)

    print("\nSAC 关键超参数说明：")
    print(f"  buffer_size    = {SAC_PARAMS['buffer_size']:,}  "
          "（Replay Buffer，可重复利用历史数据）")
    print(f"  learning_rate  = {SAC_PARAMS['learning_rate']}  "
          "（Adam 优化器步长）")
    print(f"  tau            = {SAC_PARAMS['tau']}  "
          "（目标网络软更新系数）")
    print(f"  ent_coef       = {SAC_PARAMS['ent_coef']}     "
          "（熵系数，auto = 自动调整）")
    print(f"  learning_starts= {SAC_PARAMS['learning_starts']}   "
          "（前N步随机探索，填充Buffer）")
    print(f"  device         = {SAC_PARAMS['device']}       "
          "（训练设备，GPU 加速神经网络计算）")

    # TODO 1：初始化 SAC 模型
    # 提示：
    #   model = SAC(**SAC_PARAMS, env=train_env)
    #   注意：SAC_PARAMS 里已经包含了 policy 和 verbose 等参数
    model = SAC(**SAC_PARAMS, env=train_env)
    print(f"SAC 模型已初始化，使用设备：{model.device}")

    # 创建回调（记录训练曲线）
    callback = RewardLoggerCallback(
        eval_env=eval_env,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES
    )

    # TODO 2：训练模型
    # 提示：
    #   start_time = time.time()
    #   model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    #   elapsed = time.time() - start_time
    #   print(f"\n训练耗时：{elapsed:.1f} 秒")
    print("\n开始训练...")
    start_time = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, progress_bar=True)
    elapsed = time.time() - start_time
    print(f"\n训练完成！总耗时：{elapsed:.1f} 秒")
    print(f"平均每秒步数：{TOTAL_TIMESTEPS/elapsed:.0f}")

    # 保存模型
    model.save(SAVE_PATH)
    print(f"模型已保存到 {SAVE_PATH}.zip")

    train_env.close()
    eval_env.close()
    return model, callback


# ── 加载并评估 ───────────────────────────────────────────────

def load_and_evaluate():
    """
    加载保存的模型并进行最终评估

    TODO 3：补全模型加载和评估
    """
    print("\n── 加载模型并评估 ──")

    eval_env = gym.make(ENV_ID)

    # TODO 3：加载模型并评估
    # 提示：
    #   model = SAC.load(SAVE_PATH, env=eval_env)
    #   mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20,
    #                                             deterministic=True)
    #   print(f"最终评估（20轮，确定性策略）：{mean_reward:.1f} ± {std_reward:.1f}")
    model = SAC.load(SAVE_PATH, env=eval_env, device=DEVICE)
    mean_reward, std_reward = evaluate_policy(
        model, eval_env,
        n_eval_episodes=20,
        deterministic=True
    )
    print(f"最终评估（20轮，确定性策略）：{mean_reward:.1f} ± {std_reward:.1f}")

    eval_env.close()
    return mean_reward


# ── 可视化训练曲线 ───────────────────────────────────────────

def plot_training_curve(callback: RewardLoggerCallback):
    """画 SAC 训练曲线：奖励随训练步数的变化"""
    if not callback.eval_rewards:
        print("没有评估数据，跳过绘图")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    timesteps = np.array(callback.eval_timesteps)
    rewards   = np.array(callback.eval_rewards)
    stds      = np.array(callback.eval_stds)

    # 奖励曲线 + 误差带（这是累计奖励，不是单步奖励）
    ax.plot(timesteps, rewards, 'o-', color='steelblue', linewidth=2,
            markersize=5, label=f'累计奖励（{N_EVAL_EPISODES}次评估均值）')
    ax.fill_between(timesteps, rewards - stds, rewards + stds,
                    alpha=0.2, color='steelblue', label='±1 标准差')

    # Pendulum 累计奖励范围（200步）
    # 单步 [0, -16.27]，累计 [0, -3254]
    ax.axhline(0, color='green', linestyle='--', alpha=0.7, label='最优 (0)')
    ax.axhline(-500, color='orange', linestyle=':', alpha=0.5, label='较好 (~-500)')
    ax.axhline(-1500, color='red', linestyle=':', alpha=0.5, label='随机 (~-1500)')

    ax.set_xlabel('训练步数 (Timesteps)')
    ax.set_ylabel('累计奖励')
    ax.set_title(f'SAC on {ENV_ID}\n'
                 f'buffer_size={SAC_PARAMS["buffer_size"]}, '
                 f'lr={SAC_PARAMS["learning_rate"]}, '
                 f'ent_coef={SAC_PARAMS["ent_coef"]}')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = 'assets/step3_reward_curve.png'
    plt.savefig(path, dpi=120, bbox_inches='tight')
    print(f"训练曲线已保存到 {path}")
    plt.close()


# ── SAC vs PPO 对比演示 ──────────────────────────────────────

def compare_sac_vs_ppo():
    """
    可选：同时训练 SAC 和 PPO，对比性能
    注意：这会多花约 1 分钟（PPO on Pendulum 约需要更多步数收敛）

    如果不想运行，直接看注释中的结论即可
    """
    print("\n── SAC vs PPO 概念对比（无需运行代码，看注释理解）──")
    print()
    print("┌─────────────────┬────────────────────┬────────────────────┐")
    print("│                 │        SAC         │        PPO         │")
    print("├─────────────────┼────────────────────┼────────────────────┤")
    print("│ 数据使用方式    │ Off-policy（Buffer）│ On-policy（直接丢）│")
    print("│ 样本效率        │ 高                 │ 低                 │")
    print("│ 动作空间        │ 连续动作为主        │ 离散+连续均可      │")
    print("│ 探索机制        │ 最大熵（自动探索）  │ clip ratio 限制    │")
    print("│ 稳定性          │ 较高               │ 非常稳定           │")
    print("│ 超参数数量      │ 较多               │ 较少               │")
    print("│ 典型应用        │ 机器人精细控制      │ 游戏、通用任务     │")
    print("└─────────────────┴────────────────────┴────────────────────┘")
    print()
    print("机器人公司的选择：")
    print("  智元 / 宇树：SAC（连续控制）+ IsaacGym（并行仿真）+ Domain Randomization")
    print("  游戏 AI：PPO（离散动作）→ AlphaStar, OpenAI Five")
    print("  人形机器人（最新）：TD-MPC2, DreamerV3（基于模型的方法，更样本高效）")


# ── 展示 Pendulum 环境信息 ───────────────────────────────────

def explain_pendulum():
    """打印 Pendulum-v1 环境的详细信息"""
    env = gym.make(ENV_ID)
    print(f"\n── {ENV_ID} 环境信息 ──")
    print(f"  观测空间：{env.observation_space}")
    print(f"    obs[0] = cos(θ)，obs[1] = sin(θ)，obs[2] = θ_dot（角速度）")
    print(f"    （注意：不是直接给角度，而是 cos/sin，避免角度的 ±π 不连续性）")
    print(f"  动作空间：{env.action_space}")
    print(f"    action[0] = 力矩，范围 [-2, 2]，1维连续动作")
    print(f"  奖励函数：r = -(θ² + 0.1×θ_dot² + 0.001×u²)")
    print(f"    每步奖励范围: [0, -16.27]")
    print(f"    θ=0, θ̇=0, u=0 时 r = 0 (最优，杆竖直)")
    print(f"    θ=π, θ̇=8, u=2 时 r = -16.27 (最差，杆倒下)")
    print(f"  收敛判断：单步奖励接近 0（如 -1 以内），表示杆能保持竖直")
    env.close()


# ── 主程序 ──────────────────────────────────────────────────

if __name__ == "__main__":
    print("Step 3：SAC（Soft Actor-Critic）on Pendulum-v1")
    print("框架：Stable-Baselines3（不用从零写，专注理解算法）\n")

    # 打印环境说明
    explain_pendulum()

    # 训练
    model, callback = train_sac()

    # 画图
    plot_training_curve(callback)

    # 加载并评估
    final_reward = load_and_evaluate()

    # 对比说明
    compare_sac_vs_ppo()

    print("\n── 思考题 ──")
    print("1. 为什么 SAC 的 learning_starts=1000 之前要随机探索？")
    print("   提示：replay buffer 为空时采样出来都是一样的数据，训练会过拟合")
    print()
    print("2. ent_coef='auto' 是怎么自动调整的？")
    print("   提示：SAC 设定目标熵 H_target = -dim(action_space)")
    print("         当 H(π) > H_target 时（探索太多），减小 α 让策略更集中")
    print("         当 H(π) < H_target 时（探索太少），增大 α 鼓励探索")
    print()
    print("3. 如果把 buffer_size 从 50000 减小到 1000，会发生什么？")
    print("   提示：旧数据被覆盖太快，相当于 on-policy，失去 off-policy 优势")
    print()
    print(f"模型文件：{SAVE_PATH}.zip")
    print("下一步：Step 4 将把机械臂 IK 任务包装成 Gym 环境，用 SAC 训练")
