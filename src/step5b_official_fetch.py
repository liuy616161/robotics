"""
Step 5B: 官方 FetchReach 完美复刻

使用 gymnasium-robotics 官方库完整复刻 FetchReach，作为 Step5 的完美样例对照。

与 Step5 的对比：
  - 物理引擎：PyBullet UR5 vs MuJoCo (Fetch)
  - 观测空间：简化版 vs 完整 Goal-Aware
  - 奖励函数：自定义 vs 官方 Dense
  - 动作空间：末端位移 vs 末端位移+夹爪

运行：python step5b_official_fetch.py

依赖：pip install gymnasium-robotics gymnasium stable-baselines3
"""

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from typing import Optional, Tuple, Dict, Any

try:
    import gymnasium_robotics
    from gymnasium import make
    from stable_baselines3 import SAC
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
except ImportError as e:
    raise ImportError(
        f"缺少依赖：{e}\n"
        "请运行：pip install gymnasium-robotics gymnasium stable-baselines3"
    )

import torch


# ── 参数 ─────────────────────────────────────────────────────

TOTAL_TIMESTEPS = 100_000
EVAL_FREQ = 5000
N_EVAL_EPISODES = 20
SUCCESS_THRESHOLD = 0.05  # 5cm（官方标准）


# ── 回调函数 ─────────────────────────────────────────────────

class SuccessRateCallback(BaseCallback):
    """记录训练统计"""

    def __init__(self, eval_env, eval_freq: int = 5000, n_eval_episodes: int = 20):
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
                obs, _ = self.eval_env.reset()
                ep_reward = 0.0
                ep_success = False
                ep_steps = 0

                # FetchReach 最大步数
                for _ in range(100):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    ep_reward += reward
                    ep_steps += 1

                    if info.get("is_success", False):
                        ep_success = True
                    if terminated or truncated:
                        break

                dist = np.linalg.norm(
                    obs['achieved_goal'] - obs['desired_goal']
                ) if 'achieved_goal' in obs else 0.0
                distance_list.append(dist)
                rewards_list.append(ep_reward)
                success_list.append(float(ep_success))
                steps_list.append(ep_steps)

            self.eval_timesteps.append(self.num_timesteps)
            self.eval_rewards.append(np.mean(rewards_list))
            self.eval_success.append(np.mean(success_list))
            self.eval_steps.append(np.mean(steps_list))
            self.eval_distance.append(np.mean(distance_list))

            print(f"\n  ═══════════════════════════════════════")
            print(f"  [{self.num_timesteps:6d} steps] 官方 FetchReach 统计")
            print(f"  ───────────────────────────────────────")
            print(f"  奖励:     {np.mean(rewards_list):8.2f} ± {np.std(rewards_list):.2f}")
            print(f"  成功率:   {np.mean(success_list)*100:7.1f}%")
            print(f"  回合步数: {np.mean(steps_list):7.1f} ± {np.std(steps_list):.1f}")
            print(f"  最终距离: {np.mean(distance_list):7.3f} m")
            print(f"  ═══════════════════════════════════════")

        return True


class CheckpointCallback(BaseCallback):
    """定期保存检查点"""

    def __init__(self, save_freq: int = 50000, save_path: str = "assets/checkpoints/", verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self._last_save_at = 0
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_save_at >= self.save_freq:
            checkpoint_path = os.path.join(
                self.save_path, f"sac_official_fetch_{self.num_timesteps}_steps.zip"
            )
            self.model.save(checkpoint_path)
            self._last_save_at = self.num_timesteps
            if self.verbose > 0:
                print(f"\n[Checkpoint] 已保存: {checkpoint_path}")
        return True


# ── 训练函数 ─────────────────────────────────────────────────

def train_official_fetch(env_id: str = "FetchReachDense-v4", total_timesteps: int = 500_000):
    """
    训练官方 FetchReach 环境

    可用环境：
      - FetchReach-v4: Sparse 奖励（成功=0，失败=-1）
      - FetchReachDense-v4: Dense 奖励（-距离）
    """
    os.makedirs("assets", exist_ok=True)

    print("=" * 60)
    print(f"训练官方 FetchReach：{env_id}")
    print("=" * 60)

    # 创建环境
    print("\n创建环境...")
    train_env = make(env_id, render_mode=None)
    eval_env = make(env_id, render_mode=None)

    # 验证环境（官方环境已验证，跳过 SB3 check_env）
    print("环境创建成功！\n")

    # 包装 Monitor
    train_env = Monitor(train_env)

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
        policy_kwargs=dict(net_arch=[256, 256, 256]),
    )

    # 回调
    success_callback = SuccessRateCallback(eval_env, eval_freq=EVAL_FREQ, n_eval_episodes=N_EVAL_EPISODES)
    checkpoint_callback = CheckpointCallback(save_freq=50000)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='assets/best_model/',
        log_path='assets/logs/',
        eval_freq=EVAL_FREQ,
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
    model_name = env_id.replace("-", "_").lower()
    final_path = f"assets/sac_{model_name}_final_{int(model.num_timesteps)}_steps.zip"
    model.save(final_path)
    model.save(f"assets/sac_{model_name}")
    print(f"\n训练完成！模型已保存: {final_path}")

    train_env.close()
    eval_env.close()

    return model, success_callback, env_id


def compare_custom_vs_official(
    custom_callback,
    official_callback,
    custom_name: str = "Step5 (PyBullet)",
    official_name: str = "Step5B (Official Fetch)"
):
    """对比自定义环境和官方环境的训练曲线"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 成功率对比
    ax2 = axes[0, 1]
    if custom_callback.eval_timesteps:
        custom_timesteps = np.array(custom_callback.eval_timesteps)
        custom_success = np.array(custom_callback.eval_success) * 100
        ax2.plot(custom_timesteps, custom_success, 'o-', color='blue',
                linewidth=2, markersize=4, label=custom_name, alpha=0.7)

    if official_callback.eval_timesteps:
        official_timesteps = np.array(official_callback.eval_timesteps)
        official_success = np.array(official_callback.eval_success) * 100
        ax2.plot(official_timesteps, official_success, 'o-', color='green',
                linewidth=2, markersize=4, label=official_name, alpha=0.7)

    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title('Custom vs Official FetchReach')
    ax2.set_ylim(0, 105)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 奖励对比
    ax1 = axes[0, 0]
    if custom_callback.eval_timesteps:
        ax1.plot(custom_callback.eval_timesteps, custom_callback.eval_rewards,
                'o-', color='blue', linewidth=2, markersize=4,
                label=custom_name, alpha=0.7)
    if official_callback.eval_timesteps:
        ax1.plot(official_callback.eval_timesteps, official_callback.eval_rewards,
                'o-', color='green', linewidth=2, markersize=4,
                label=official_name, alpha=0.7)
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Episode Reward Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 步数对比
    ax3 = axes[1, 0]
    if custom_callback.eval_timesteps:
        ax3.plot(custom_callback.eval_timesteps, custom_callback.eval_steps,
                'o-', color='blue', linewidth=2, markersize=4,
                label=custom_name, alpha=0.7)
    if official_callback.eval_timesteps:
        ax3.plot(official_callback.eval_timesteps, official_callback.eval_steps,
                'o-', color='green', linewidth=2, markersize=4,
                label=official_name, alpha=0.7)
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('Average Steps')
    ax3.set_title('Episode Length Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 距离对比
    ax4 = axes[1, 1]
    if custom_callback.eval_timesteps:
        ax4.plot(custom_callback.eval_timesteps, custom_callback.eval_distance,
                'o-', color='blue', linewidth=2, markersize=4,
                label=custom_name, alpha=0.7)
    if official_callback.eval_timesteps:
        ax4.plot(official_callback.eval_timesteps, official_callback.eval_distance,
                'o-', color='green', linewidth=2, markersize=4,
                label=official_name, alpha=0.7)
    ax4.axhline(SUCCESS_THRESHOLD, color='red', linestyle='--', alpha=0.7,
                label=f'Target ({SUCCESS_THRESHOLD}m)')
    ax4.set_xlabel('Training Steps')
    ax4.set_ylabel('Average Distance (m)')
    ax4.set_title('Final Distance Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Custom PyBullet vs Official FetchReach Comparison',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    path = 'assets/step5b_custom_vs_official.png'
    plt.savefig(path, dpi=120, bbox_inches='tight')
    print(f"\n对比曲线已保存: {path}")
    plt.close()


def plot_single_training_results(callback: SuccessRateCallback, env_name: str):
    """绘制单次训练结果"""
    if not callback.eval_timesteps:
        print("没有评估数据，跳过绘图")
        return

    timesteps = np.array(callback.eval_timesteps)
    rewards = np.array(callback.eval_rewards)
    success = np.array(callback.eval_success) * 100
    steps = np.array(callback.eval_steps)
    distance = np.array(callback.eval_distance)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 奖励
    ax1 = axes[0, 0]
    ax1.plot(timesteps, rewards, 'o-', color='steelblue', linewidth=2, markersize=6)
    ax1.fill_between(timesteps, rewards, alpha=0.2, color='steelblue')
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Average Reward')
    ax1.set_title(f'Episode Reward ({env_name})')
    ax1.grid(True, alpha=0.3)

    # 成功率
    ax2 = axes[0, 1]
    ax2.plot(timesteps, success, 'o-', color='green', linewidth=2, markersize=6)
    ax2.axhline(80, color='orange', linestyle='--', alpha=0.7, label='80% Target')
    ax2.axhline(50, color='red', linestyle=':', alpha=0.5, label='50% Baseline')
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title(f'Success Rate (dist < {SUCCESS_THRESHOLD}m)')
    ax2.set_ylim(0, 105)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 步数
    ax3 = axes[1, 0]
    ax3.plot(timesteps, steps, 'o-', color='purple', linewidth=2, markersize=6)
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('Average Steps per Episode')
    ax3.set_title('Episode Length')
    ax3.grid(True, alpha=0.3)

    # 距离
    ax4 = axes[1, 1]
    ax4.plot(timesteps, distance, 'o-', color='red', linewidth=2, markersize=6)
    ax4.axhline(SUCCESS_THRESHOLD, color='green', linestyle='--', alpha=0.7,
                label=f'Target ({SUCCESS_THRESHOLD}m)')
    ax4.set_xlabel('Training Steps')
    ax4.set_ylabel('Average Distance (m)')
    ax4.set_title('Final Distance to Goal')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f'Official FetchReach Training: {env_name}\n'
                 f'Total steps={callback.eval_timesteps[-1]}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    path = f'assets/step5b_training_{env_name.replace(" ", "_").lower()}.png'
    plt.savefig(path, dpi=120, bbox_inches='tight')
    print(f"训练曲线已保存: {path}")
    plt.close()


# ── 环境信息展示 ─────────────────────────────────────────────

def show_env_info():
    """展示官方环境详细信息"""
    print("── 官方 FetchReach 环境信息 ──\n")

    for env_id in ["FetchReach-v4", "FetchReachDense-v4"]:
        print(f"环境: {env_id}")
        env = make(env_id)

        print(f"  观测空间: {env.observation_space}")
        print(f"  动作空间: {env.action_space}")
        print(f"  最大 episode 步数: {env._max_episode_steps}")

        obs, _ = env.reset()
        print(f"  观测键: {obs.keys() if hasattr(obs, 'keys') else 'N/A'}")
        if hasattr(obs, 'keys'):
            for k, v in obs.items():
                print(f"    {k}: shape={v.shape}, dtype={v.dtype}")

        env.close()
        print()


# ── 主程序 ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Step 5B: 官方 FetchReach 完美复刻")
    print("=" * 60)

    # 1. 展示环境信息
    show_env_info()

    # 2. 选择训练模式
    print("\n训练选项:")
    print("  1. FetchReach-v4 (Sparse 奖励)")
    print("  2. FetchReachDense-v4 (Dense 奖励)")
    print("  3. 两种都跑并对比")

    choice = "3"

    if choice == "1":
        model, callback, env_id = train_official_fetch("FetchReach-v4", TOTAL_TIMESTEPS)
        plot_single_training_results(callback, env_id)
    elif choice == "2":
        model, callback, env_id = train_official_fetch("FetchReachDense-v4", TOTAL_TIMESTEPS)
        plot_single_training_results(callback, env_id)
    else:
        # 训练两种并对比
        print("\n>>> 训练 FetchReachDense-v4 <<<")
        model_dense, callback_dense, env_dense = train_official_fetch("FetchReachDense-v4", TOTAL_TIMESTEPS)
        plot_single_training_results(callback_dense, env_dense)

        print("\n>>> 训练 FetchReach-v4 <<<")
        model_sparse, callback_sparse, env_sparse = train_official_fetch("FetchReach-v4", TOTAL_TIMESTEPS)
        plot_single_training_results(callback_sparse, env_sparse)

        # 对比
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for label, callback, color, marker in [
            ("Dense", callback_dense, 'blue', 'o'),
            ("Sparse", callback_sparse, 'red', 's')
        ]:
            if callback.eval_timesteps:
                timesteps = np.array(callback.eval_timesteps)
                success = np.array(callback.eval_success) * 100
                axes[0].plot(timesteps, success, f'{marker}-', color=color,
                            linewidth=2, markersize=4, label=label, alpha=0.7)
                axes[1].plot(timesteps, callback.eval_rewards, f'{marker}-', color=color,
                            linewidth=2, markersize=4, label=label, alpha=0.7)

        axes[0].set_xlabel('Training Steps')
        axes[0].set_ylabel('Success Rate (%)')
        axes[0].set_title('Dense vs Sparse Reward')
        axes[0].set_ylim(0, 105)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel('Training Steps')
        axes[1].set_ylabel('Average Reward')
        axes[1].set_title('Reward Comparison')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle('Official FetchReach: Dense vs Sparse', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig('assets/step5b_official_comparison.png', dpi=120)
        print("\n对比曲线已保存: assets/step5b_official_comparison.png")
        plt.close()

    print("\n── Step 5B 完成 ──")
    print("下一步：step6_rl_reach_benchmark.py (多算法基准)")
