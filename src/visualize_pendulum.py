"""
Pendulum 可视化：加载 SAC 模型，让倒立摆动起来
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import gymnasium as gym
from stable_baselines3 import SAC
import os
from PIL import Image
import io

# 环境配置
ENV_ID = "Pendulum-v1"
MODEL_PATH = "assets/sac_pendulum"

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 加载模型
MODEL_PATH = "assets/sac_pendulum"

print("=" * 60)
print("Pendulum 可视化")
print("=" * 60)

# 打印环境信息
env_info = gym.make(ENV_ID)
obs_space = env_info.observation_space
act_space = env_info.action_space

print("\n【环境空间定义】")
print(f"  观测空间 (Observation Space):")
print(f"    类型: {type(obs_space).__name__}, 形状: {obs_space.shape}")
print(f"    范围: [{obs_space.low[0]:.1f}, {obs_space.high[0]:.1f}] × "
      f"[{obs_space.low[1]:.1f}, {obs_space.high[1]:.1f}] × "
      f"[{obs_space.low[2]:.1f}, {obs_space.high[2]:.1f}]")
print(f"    obs[0] = cos(θ)    角度余弦")
print(f"    obs[1] = sin(θ)    角度正弦")
print(f"    obs[2] = θ̇        角速度 (rad/s)")

print(f"\n  动作空间 (Action Space):")
print(f"    类型: {type(act_space).__name__}, 形状: {act_space.shape}")
print(f"    范围: [{act_space.low[0]}, {act_space.high[0]}]")
print(f"    action[0] = 力矩 u (torque)")

print(f"\n  奖励函数: r = -(θ² + 0.1×θ̇² + 0.001×u²)")
print(f"    每步奖励范围: [0, -16.27]")
print(f"    最优=0 (竖直), 最差=-16.27 (倒下)")
print("=" * 60)

print("\n加载 SAC 模型...")
model = SAC.load(MODEL_PATH, device='cuda')
env_info.close()

# 创建环境
env = gym.make("Pendulum-v1", render_mode="rgb_array")


def generate_pendulum_gif(n_frames=200, fps=20):
    """生成 Pendulum 运行 GIF，实时显示观测向量"""
    print(f"生成 {n_frames} 帧动画...")

    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # 左上：倒立摆动画
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect('equal')
    ax1.set_title('Pendulum - SAC Policy')
    ax1.axis('off')

    pivot = Circle((0, 0), 0.05, color='black')
    ax1.add_patch(pivot)
    rod = Line2D([0, 1], [0, 1], color='steelblue', lw=3)
    ax1.add_line(rod)
    bob = Circle((1, 0), 0.15, color='red')
    ax1.add_patch(bob)

    # 右上：观测向量实时显示
    ax_obs = fig.add_subplot(gs[0, 1:])
    ax_obs.set_xlim(0, n_frames)
    ax_obs.set_ylim(-1.5, 1.5)
    ax_obs.set_xlabel('Frame')
    ax_obs.set_ylabel('Value')
    ax_obs.set_title('Observation Vector (Real-time)')
    ax_obs.grid(True, alpha=0.3)
    ax_obs.axhline(0, color='black', lw=0.5)

    cos_history = []
    sin_history = []
    dot_history = []

    line_cos, = ax_obs.plot([], [], 'r-', lw=1.5, label='cos(theta)')
    line_sin, = ax_obs.plot([], [], 'g-', lw=1.5, label='sin(theta)')
    line_dot, = ax_obs.plot([], [], 'b-', lw=1.5, label='theta_dot')
    ax_obs.legend(loc='upper right', fontsize=8)

    # 左下：角度历史
    ax_theta = fig.add_subplot(gs[1, 0])
    ax_theta.set_xlim(0, n_frames)
    ax_theta.set_ylim(-np.pi - 0.5, np.pi + 0.5)
    ax_theta.set_xlabel('Frame')
    ax_theta.set_ylabel('Angle (rad)')
    ax_theta.set_title('Angle History')
    ax_theta.grid(True, alpha=0.3)
    ax_theta.axhline(0, color='green', linestyle='--', alpha=0.5)

    theta_history = []
    line_theta, = ax_theta.plot([], [], 'b-', lw=1.5)

    # 中下：动作历史
    ax_action = fig.add_subplot(gs[1, 1])
    ax_action.set_xlim(0, n_frames)
    ax_action.set_ylim(-2.5, 2.5)
    ax_action.set_xlabel('Frame')
    ax_action.set_ylabel('Action (torque)')
    ax_action.set_title('Action History')
    ax_action.grid(True, alpha=0.3)
    ax_action.axhline(0, color='black', lw=0.5)

    action_history = []
    line_action, = ax_action.plot([], [], 'purple', lw=1.5)

    # 右下：奖励历史
    ax_reward = fig.add_subplot(gs[1, 2])
    ax_reward.set_xlim(0, n_frames)
    ax_reward.set_ylim(-16, 1)
    ax_reward.set_xlabel('Frame')
    ax_reward.set_ylabel('Reward')
    ax_reward.set_title('Reward History')
    ax_reward.grid(True, alpha=0.3)
    ax_reward.axhline(-0.5, color='green', linestyle='--', alpha=0.5, label='target (~0)')
    ax_reward.legend(loc='lower right', fontsize=8)

    reward_history = []
    line_reward, = ax_reward.plot([], [], 'orange', lw=1.5)

    frames = []
    obs, _ = env.reset()

    for i in range(n_frames):
        # 获取动作
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # 提取状态
        cos_theta, sin_theta, theta_dot = obs
        theta = np.arctan2(sin_theta, cos_theta)

        # 记录历史
        theta_history.append(theta)
        cos_history.append(cos_theta)
        sin_history.append(sin_theta)
        dot_history.append(theta_dot)
        action_history.append(action[0])
        reward_history.append(reward)

        # 更新左上图
        # 注意：Pendulum 中 theta=0 表示杆竖直向上
        # cos(θ)=1, sin(θ)=0 时，杆向上
        x = np.sin(theta)
        y = np.cos(theta)  # 修复：theta=0 时 y=1（向上）
        rod.set_data([0, x], [0, y])
        bob.set_center((x, y))

        # 根据奖励着色 (Pendulum-v1 奖励范围约 [0, -16])
        # 接近 0 = 竖直，接近 -16 = 倒下
        if reward > -0.5:
            bob.set_color('green')   # 竖直/好的状态
        elif reward > -2:
            bob.set_color('orange')   # 一般
        else:
            bob.set_color('red')     # 倒下/差的状态

        # 更新右上观测图
        line_cos.set_data(range(len(cos_history)), cos_history)
        line_sin.set_data(range(len(sin_history)), sin_history)
        line_dot.set_data(range(len(dot_history)), dot_history)

        # 更新左下图
        line_theta.set_data(range(len(theta_history)), theta_history)

        # 更新中下图
        line_action.set_data(range(len(action_history)), action_history)

        # 更新右下图
        line_reward.set_data(range(len(reward_history)), reward_history)

        # 添加实时信息文本
        # 判断状态
        if theta > 2.5 or theta < -2.5:
            state = "倒悬!"
        elif abs(theta) < 0.3:
            state = "竖直"
        else:
            state = "过渡"

        info_text = f"theta={theta:5.2f}rad ({theta/np.pi:4.1f}π)\nobs: cos={cos_theta:.2f}, sin={sin_theta:.2f}\naction: {action[0]:5.2f}  r={reward:6.3f}\n状态: {state}"
        ax1.text(0, -1.3, info_text, ha='center', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 保存帧
        fig.canvas.draw()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img = Image.open(buf)
        frames.append(img.copy())
        buf.close()

        if done:
            obs, _ = env.reset()
            # 重置历史
            theta_history.clear()
            cos_history.clear()
            sin_history.clear()
            dot_history.clear()
            action_history.clear()
            reward_history.clear()

        if i % 50 == 0:
            print(f"  Frame {i}/{n_frames}, reward: {reward:.1f}, obs: {obs}")

    plt.close()

    # 保存 GIF
    print(f"保存 GIF...")
    frames[0].save(
        'assets/pendulum_sac.gif',
        save_all=True,
        append_images=frames[1:],
        duration=1000 // fps,
        loop=0
    )
    print("保存到 assets/pendulum_sac.gif")


def generate_comparison_gif(n_frames=150):
    """对比随机策略 vs SAC"""
    print("\n对比：随机策略 vs SAC")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 随机策略
    ax1 = axes[0]
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect('equal')
    ax1.set_title('Random Policy')
    ax1.axis('off')

    pivot1 = Circle((0, 0), 0.05, color='black')
    ax1.add_patch(pivot1)
    rod1 = Line2D([0, 1], [0, 1], color='gray', lw=3)
    ax1.add_line(rod1)
    bob1 = Circle((1, 0), 0.15, color='red')
    ax1.add_patch(bob1)

    # SAC 策略
    ax2 = axes[1]
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_aspect('equal')
    ax2.set_title('SAC Policy')
    ax2.axis('off')

    pivot2 = Circle((0, 0), 0.05, color='black')
    ax2.add_patch(pivot2)
    rod2 = Line2D([0, 1], [0, 1], color='steelblue', lw=3)
    ax2.add_line(rod2)
    bob2 = Circle((1, 0), 0.15, color='green')
    ax2.add_patch(bob2)

    frames = []

    # 随机策略环境
    env_random = gym.make("Pendulum-v1", render_mode="rgb_array")
    obs_random, _ = env_random.reset()

    # SAC 策略环境
    env_sac = gym.make("Pendulum-v1", render_mode="rgb_array")
    obs_sac, _ = env_sac.reset()

    random_rewards = []
    sac_rewards = []

    for i in range(n_frames):
        # 随机动作
        action_random = env_random.action_space.sample()
        obs_random, reward_r, term_r, trunc_r, _ = env_random.step(action_random)
        random_rewards.append(reward_r)

        # SAC 动作
        action_sac, _ = model.predict(obs_sac, deterministic=True)
        obs_sac, reward_s, term_s, trunc_s, _ = env_sac.step(action_sac)
        sac_rewards.append(reward_s)

        # 更新随机策略图
        cos_r, sin_r, _ = obs_random
        theta_r = np.arctan2(sin_r, cos_r)
        rod1.set_data([0, np.sin(theta_r)], [0, np.cos(theta_r)])  # 修复
        bob1.set_center((np.sin(theta_r), np.cos(theta_r)))

        # 更新 SAC 图
        cos_s, sin_s, _ = obs_sac
        theta_s = np.arctan2(sin_s, cos_s)
        rod2.set_data([0, np.sin(theta_s)], [0, np.cos(theta_s)])  # 修复
        bob2.set_center((np.sin(theta_s), np.cos(theta_s)))

        # 重置
        if term_r or trunc_r:
            obs_random, _ = env_random.reset()
        if term_s or trunc_s:
            obs_sac, _ = env_sac.reset()

        # 添加标题显示平均奖励
        fig.suptitle(f'Frame {i}/{n_frames} | Random: {np.mean(random_rewards):.0f} | SAC: {np.mean(sac_rewards):.0f}')

        # 保存帧
        fig.canvas.draw()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img = Image.open(buf)
        frames.append(img.copy())
        buf.close()

        if i % 30 == 0:
            print(f"  Frame {i}/{n_frames}")

    plt.close()
    env_random.close()
    env_sac.close()

    # 保存
    print("保存对比 GIF...")
    frames[0].save(
        'assets/pendulum_comparison.gif',
        save_all=True,
        append_images=frames[1:],
        duration=80,
        loop=0
    )
    print("保存到 assets/pendulum_comparison.gif")

    print(f"\n结果对比:")
    print(f"  随机策略平均奖励: {np.mean(random_rewards):.1f}")
    print(f"  SAC 策略平均奖励: {np.mean(sac_rewards):.1f}")


def print_stats():
    """打印策略统计信息"""
    print("\n" + "="*50)
    print("SAC Policy 统计分析")
    print("="*50)

    # 运行多次评估
    eval_rewards = []
    eval_lengths = []

    for ep in range(20):
        obs, _ = env.reset()
        ep_reward = 0
        ep_length = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            ep_length += 1
            done = terminated or truncated

        eval_rewards.append(ep_reward)
        eval_lengths.append(ep_length)

    print(f"平均奖励: {np.mean(eval_rewards):.1f} ± {np.std(eval_rewards):.1f}")
    print(f"平均回合长度: {np.mean(eval_lengths):.0f}")
    print(f"最佳奖励: {np.max(eval_rewards):.1f}")
    print(f"最差奖励: {np.min(eval_rewards):.1f}")

    env.close()


if __name__ == "__main__":
    print("=" * 50)
    print("     Pendulum 可视化")
    print("=" * 50)

    # 生成 SAC 动画
    generate_pendulum_gif(200)

    # 对比随机 vs SAC
    generate_comparison_gif(150)

    # 统计信息
    print_stats()

    print("\n" + "="*50)
    print("完成！查看 assets/ 目录")
    print("="*50)
