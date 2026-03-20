"""
可视化 step5b 训练的 FetchReach 模型

运行：
    conda activate robotics
    python visualize_step5b.py --model assets/checkpoints/sac_official_fetch_250000_steps.zip --episodes 3

参数：
    --model   模型路径
    --episodes  episodes 数量
    --render   渲染模式 (human/rgb_array)
    --delay    每步延迟秒数 (默认 0.03)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import time
import argparse

import gymnasium as gym
import gymnasium_robotics  # noqa

try:
    from stable_baselines3 import SAC
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False


def find_latest_model():
    """找到最新的官方 FetchReach 模型"""
    import glob
    patterns = [
        'assets/checkpoints/sac_official_fetch_*.zip',
        '../assets/checkpoints/sac_official_fetch_*.zip',
    ]
    for pattern in patterns:
        files = glob.glob(pattern)
        if files:
            files.sort(key=os.path.getmtime, reverse=True)
            return files[0]
    return None


def try_render_modes(env_id):
    """尝试不同的渲染模式，返回可用的模式"""
    modes = [("rgb_array",), ("human",), (None,)]
    for mode in modes:
        try:
            env = gym.make(env_id, render_mode=mode[0])
            obs, _ = env.reset()
            if mode[0]:
                frame = env.render()
            env.close()
            print(f"  渲染模式 {mode[0]} 可用")
            return mode[0] if mode[0] else "human"
        except Exception:
            env.close()
            continue
    return None


def visualize_with_model(model_path, env_id="FetchReachDense-v4", n_episodes=3, delay=0.03, render_mode=None):
    """
    使用模型可视化

    Args:
        model_path: 模型路径
        env_id: 环境 ID
        n_episodes: episodes 数量
        delay: 每步延迟秒数
        render_mode: 渲染模式
    """
    print("=" * 60)
    print("FetchReach 模型可视化")
    print("=" * 60)

    # 加载模型
    if model_path and os.path.exists(model_path):
        print(f"加载模型: {model_path}")
        model = SAC.load(model_path)
    else:
        print("模型未找到，使用随机策略")
        model = None

    # 自动检测渲染模式
    if render_mode is None:
        print("检测可用渲染模式...")
        render_mode = try_render_modes(env_id)
        if render_mode is None:
            print("警告: 无法进行图形渲染，使用 None 模式")
            render_mode = None

    # 创建环境
    env = gym.make(env_id, render_mode=render_mode)

    episodes_data = []
    all_trajectories = []  # 保存轨迹

    for ep in range(n_episodes):
        print(f"\n{'='*50}")
        print(f"Episode {ep + 1}/{n_episodes}")
        print(f"{'='*50}")

        obs, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        trajectory = []

        while True:
            # 获取动作
            if model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            # 保存轨迹
            achieved = obs.get('achieved_goal', np.zeros(3))
            desired = obs.get('desired_goal', np.zeros(3))
            trajectory.append({
                'step': episode_steps,
                'achieved_goal': achieved.copy(),
                'desired_goal': desired.copy(),
                'action': action.copy() if hasattr(action, 'copy') else action,
            })

            # 执行
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_steps += 1

            # 渲染
            if render_mode:
                env.render()

            # 显示信息
            dist = np.linalg.norm(achieved - desired) if len(achieved) == 3 else 0
            print(f"\r  Step {episode_steps:>3} | R={episode_reward:>7.2f} | dist={dist:.3f}m | success={info.get('is_success', False):>5}", end="")

            # 延迟
            time.sleep(delay)

            if terminated or truncated:
                print(f"\n  Episode 完成!")
                print(f"    步数: {episode_steps}")
                print(f"    奖励: {episode_reward:.2f}")
                print(f"    成功: {info.get('is_success', False)}")
                break

        # 保存轨迹
        all_trajectories.append(trajectory)
        episodes_data.append({
            'reward': episode_reward,
            'steps': episode_steps,
            'success': info.get('is_success', False),
            'trajectory': trajectory,
        })

        # episode 间暂停
        time.sleep(0.5)

    env.close()
    return episodes_data, all_trajectories


def plot_trajectories_2d(episodes_data, all_trajectories, model_path=None):
    """绘制 2D 轨迹图"""
    n = len(episodes_data)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else axes

    for i, (data, traj) in enumerate(zip(episodes_data, all_trajectories)):
        ax = axes[i]

        # 绘制轨迹
        achieved_x = [t['achieved_goal'][0] for t in traj]
        achieved_y = [t['achieved_goal'][1] for t in traj]
        desired_x = traj[0]['desired_goal'][0] if traj else 0
        desired_y = traj[0]['desired_goal'][1] if traj else 0

        ax.plot(achieved_x, achieved_y, 'b-', linewidth=1, alpha=0.7, label='轨迹')
        ax.plot(achieved_x[0], achieved_y[0], 'go', markersize=10, label='起点')
        ax.plot(achieved_x[-1], achieved_y[-1], 'ro', markersize=10, label='终点')
        ax.plot(desired_x, desired_y, 'r*', markersize=15, label='目标')

        # 画目标范围圆
        theta = np.linspace(0, 2*np.pi, 50)
        ax.plot(desired_x + 0.05*np.cos(theta), desired_y + 0.05*np.sin(theta),
                'r--', alpha=0.5, label='目标范围')

        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.2, 1.2)
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f"Episode {i+1}: {'成功' if data['success'] else '失败'} | "
                    f"R={data['reward']:.1f} | {data['steps']}步")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # 隐藏多余的 subplot
    for j in range(n, len(axes)):
        axes[j].axis('off')

    model_name = os.path.basename(model_path) if model_path else "random"
    plt.suptitle(f"FetchReach 2D 轨迹 - {model_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs("assets", exist_ok=True)
    path = f"assets/fetch_reach_{model_name.replace('.zip', '')}_trajectory_2d.png"
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"2D 轨迹图已保存: {path}")


def plot_trajectories_3d(episodes_data, all_trajectories, model_path=None):
    """绘制 3D 轨迹图"""
    if not all_trajectories or not all_trajectories[0]:
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = plt.cm.Set1(np.linspace(0, 1, len(all_trajectories)))

    for i, (data, traj) in enumerate(zip(episodes_data, all_trajectories)):
        xs = [t['achieved_goal'][0] for t in traj]
        ys = [t['achieved_goal'][1] for t in traj]
        zs = [t['achieved_goal'][2] for t in traj]
        desired = traj[0]['desired_goal']

        ax.plot(xs, ys, zs, color=colors[i], linewidth=1.5, alpha=0.7,
                label=f"Ep{i+1}: {'成功' if data['success'] else '失败'}")
        ax.scatter([xs[0]], [ys[0]], [zs[0]], color=colors[i], marker='o', s=100)
        ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], color=colors[i], marker='s', s=100)
        ax.scatter([desired[0]], [desired[1]], [desired[2]], color=colors[i],
                   marker='*', s=200)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title("FetchReach 3D 轨迹")
    ax.legend()

    model_name = os.path.basename(model_path) if model_path else "random"
    os.makedirs("assets", exist_ok=True)
    path = f"assets/fetch_reach_{model_name.replace('.zip', '')}_trajectory_3d.png"
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"3D 轨迹图已保存: {path}")


def plot_distance_over_time(episodes_data, all_trajectories, model_path=None):
    """绘制距离随时间变化图"""
    n = len(episodes_data)
    cols = min(2, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 4*rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else axes

    for i, (data, traj) in enumerate(zip(episodes_data, all_trajectories)):
        ax = axes[i]

        # 计算每步距离
        distances = []
        for t in traj:
            d = np.linalg.norm(t['achieved_goal'] - t['desired_goal'])
            distances.append(d)

        steps = range(len(distances))

        ax.plot(steps, distances, 'b-', linewidth=2)
        ax.axhline(0.05, color='green', linestyle='--', alpha=0.7, label='成功阈值 (0.05m)')

        ax.set_xlabel('Step')
        ax.set_ylabel('Distance (m)')
        ax.set_title(f"Episode {i+1}: 距离变化 | {'成功' if data['success'] else '失败'}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(distances) * 1.1 if distances else 1)

    model_name = os.path.basename(model_path) if model_path else "random"
    plt.suptitle(f"FetchReach 距离变化 - {model_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs("assets", exist_ok=True)
    path = f"assets/fetch_reach_{model_name.replace('.zip', '')}_distance.png"
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"距离曲线已保存: {path}")


def plot_action_history(episodes_data, all_trajectories, model_path=None):
    """绘制动作历史"""
    n = len(episodes_data)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else axes

    for i, (data, traj) in enumerate(zip(episodes_data, all_trajectories)):
        ax = axes[i]

        if not traj:
            continue

        # 动作维度
        n_actions = len(traj[0]['action'])
        action_dim = min(n_actions, 4)  # 最多显示 4 个维度

        for dim in range(action_dim):
            action_vals = [t['action'][dim] for t in traj]
            ax.plot(action_vals, alpha=0.7, label=f'action_{dim}')

        ax.set_xlabel('Step')
        ax.set_ylabel('Action')
        ax.set_title(f"Episode {i+1}: 动作历史 | {data['steps']}步")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # 隐藏多余的 subplot
    for j in range(n, len(axes)):
        axes[j].axis('off')

    model_name = os.path.basename(model_path) if model_path else "random"
    plt.suptitle(f"FetchReach 动作历史 - {model_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs("assets", exist_ok=True)
    path = f"assets/fetch_reach_{model_name.replace('.zip', '')}_actions.png"
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"动作历史已保存: {path}")


def plot_eval_summary(episodes_data, model_path=None):
    """绘制评估总结"""
    n = len(episodes_data)
    rewards = [ep['reward'] for ep in episodes_data]
    successes = [ep['success'] for ep in episodes_data]
    steps = [ep['steps'] for ep in episodes_data]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 奖励
    colors = ['green' if s else 'red' for s in successes]
    bars1 = axes[0].bar(range(1, n+1), rewards, color=colors, alpha=0.7)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('Episode Rewards')
    axes[0].axhline(0, color='black', linestyle='--', alpha=0.3)
    for bar, r in zip(bars1, rewards):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{r:.1f}', ha='center', va='bottom', fontsize=9)

    # 成功率
    bar_colors = ['green' if s else 'red' for s in successes]
    axes[1].bar(range(1, n+1), [1 if s else 0 for s in successes], color=bar_colors, alpha=0.7)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Success')
    axes[1].set_title(f'Success Rate: {sum(successes)}/{n} = {sum(successes)/n*100:.0f}%')
    axes[1].set_ylim(0, 1.4)

    # 步数
    bars3 = axes[2].bar(range(1, n+1), steps, color='steelblue', alpha=0.7)
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Steps')
    axes[2].set_title('Episode Length')
    for bar, s in zip(bars3, steps):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{s}', ha='center', va='bottom', fontsize=9)

    model_name = os.path.basename(model_path) if model_path else "random"
    plt.suptitle(f"FetchReach 评估总结 - {model_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs("assets", exist_ok=True)
    path = f"assets/fetch_reach_{model_name.replace('.zip', '')}_eval_summary.png"
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"评估总结已保存: {path}")


def main():
    parser = argparse.ArgumentParser(description='可视化 FetchReach 模型')
    parser.add_argument('--model', type=str, default=None, help='模型路径')
    parser.add_argument('--episodes', type=int, default=5, help='episodes 数量')
    parser.add_argument('--delay', type=float, default=0.03, help='每步延迟秒数')
    parser.add_argument('--render', type=str, default=None, choices=['human', 'rgb_array'], help='渲染模式')
    args = parser.parse_args()

    # 找模型
    model_path = args.model
    if model_path is None:
        model_path = find_latest_model()

    if model_path:
        print(f"使用模型: {model_path}")
    else:
        print("未找到模型，使用随机策略")

    # 可视化
    episodes_data, all_trajectories = visualize_with_model(
        model_path=model_path,
        n_episodes=args.episodes,
        delay=args.delay,
        render_mode=args.render
    )

    # 绘制各种图表
    print("\n生成可视化图表...")
    plot_eval_summary(episodes_data, model_path)
    plot_trajectories_2d(episodes_data, all_trajectories, model_path)
    plot_trajectories_3d(episodes_data, all_trajectories, model_path)
    plot_distance_over_time(episodes_data, all_trajectories, model_path)
    plot_action_history(episodes_data, all_trajectories, model_path)

    print("\n" + "=" * 60)
    print("可视化完成!")
    print("=" * 60)
    print("\n生成的文件:")
    print("  assets/fetch_reach_*_eval_summary.png  (评估总结)")
    print("  assets/fetch_reach_*_trajectory_2d.png  (2D 轨迹)")
    print("  assets/fetch_reach_*_trajectory_3d.png  (3D 轨迹)")
    print("  assets/fetch_reach_*_distance.png       (距离变化)")
    print("  assets/fetch_reach_*_actions.png        (动作历史)")


if __name__ == "__main__":
    main()
