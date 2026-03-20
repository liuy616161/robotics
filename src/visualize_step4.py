"""
Step 4 验证动画：加载训练好的模型，生成机械臂动画 GIF
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os
from PIL import Image
import io

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 导入 Step 4 的环境
import sys
sys.path.insert(0, '/workplace/export/01-robotics/robot-learning/task2/src')
from step4_robot_env import RobotArmEnv, ARM_LENGTHS, GOAL_RADIUS_MIN, GOAL_RADIUS_MAX

# 加载模型（优先从最新检查点恢复）
MODEL_PATH = "assets/sac_robot_arm_curriculum"
CHECKPOINT_DIR = "assets/checkpoints"


def find_best_model():
    """查找最佳可用模型：最新检查点 > 主模型文件 > None"""
    # 1. 查找最新检查点
    if os.path.exists(CHECKPOINT_DIR):
        checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("sac_robot_arm_") and f.endswith("_steps.zip")]
        if checkpoints:
            steps_list = []
            for ckpt in checkpoints:
                try:
                    steps = int(ckpt.split("_")[3])  # sac_robot_arm_20000_steps.zip → [sac, robot, arm, 20000, steps.zip]
                    steps_list.append((steps, ckpt))
                except:
                    pass
            if steps_list:
                steps_list.sort(reverse=True)
                return os.path.join(CHECKPOINT_DIR, steps_list[0][1]), steps_list[0][0]

    # 2. 回退到主模型文件
    model_path_ext = MODEL_PATH if MODEL_PATH.endswith('.zip') else f"{MODEL_PATH}.zip"
    if os.path.exists(model_path_ext):
        return model_path_ext, None

    return None, None


print("=" * 60)
print("Step 4 验证动画生成")
print("=" * 60)

# 查找可用模型
best_model_path, model_steps = find_best_model()

# 创建环境
env = RobotArmEnv(render_mode="rgb_array")


def get_link_positions(joint_angles):
    """根据关节角度计算各连杆末端坐标"""
    positions = []
    x, y = 0.0, 0.0
    angles = np.cumsum(joint_angles)  # 累计角度

    for i, length in enumerate(ARM_LENGTHS):
        x += length * np.cos(angles[i])
        y += length * np.sin(angles[i])
        positions.append((x, y))

    return [(0.0, 0.0)] + positions


def generate_validation_gif(n_episodes=3, frames_per_episode=100):
    """生成验证动画 GIF"""
    print(f"生成 {n_episodes} 个回合的验证动画...")

    # 尝试加载模型（使用 find_best_model 找到的最佳模型）
    use_model = False
    try:
        from stable_baselines3 import SAC
        if best_model_path is None:
            raise FileNotFoundError("未找到任何可用模型")

        model = SAC.load(best_model_path, env=env, device='cpu')  # 建议用 cpu 生成动画
        if model_steps:
            print(f"已加载检查点: {best_model_path} ({model_steps} 步)")
        else:
            print(f"已加载模型: {best_model_path}")
        use_model = True
    except Exception as e:
        print(f"未找到模型或加载失败: {e}")
        print("将使用随机策略演示")
        use_model = False

    fig, ax = plt.subplots(figsize=(8, 8))
    frames = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        step = 0

        # 从 info 获取真实的目标位置 (绝对坐标)
        goal_pos = np.array(info['target'][:2])
        print(f"\n回合 {episode + 1}/{n_episodes} | 目标: ({goal_pos[0]:.2f}, {goal_pos[1]:.2f})")

        while not done and step < frames_per_episode:
            # 获取动作
            if use_model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            # ── 关键修复：从 info 获取真实状态，而不是解析 obs ──
            # 因为 obs 现在是相对位置 [dx, dy, dz, q1, q2, q3]，直接画图会错
            ee_pos = np.array(info['ee_pos'][:2])       # 真实的末端绝对坐标 (x, y)
            joint_angles = np.array(info['joint_angles'])  # 真实的关节角
            current_dist = info['distance']             # 真实的距离
            success = info['success']                   # 真实的状态
            reward_components = info.get('reward_components', {})  # 各部分奖励

            # 计算连杆位置用于绘图
            positions = get_link_positions(joint_angles)
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]

            # ── 绘图 ──
            ax.clear()
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')

            # 1. 画目标采样范围参考线 (可选，为了美观)
            theta_ref = np.linspace(0, 2*np.pi, 50)
            ax.plot(GOAL_RADIUS_MAX*np.cos(theta_ref), GOAL_RADIUS_MAX*np.sin(theta_ref), 'g--', alpha=0.2, linewidth=1)
            ax.plot(GOAL_RADIUS_MIN*np.cos(theta_ref), GOAL_RADIUS_MIN*np.sin(theta_ref), 'b--', alpha=0.2, linewidth=1)

            # 2. 画目标点
            ax.plot(goal_pos[0], goal_pos[1], 'g*', markersize=18, label='Target', zorder=5)

            # 3. 画目标容差圈
            circle = plt.Circle(goal_pos, 0.05, color='green', fill=False, linestyle='--', alpha=0.5)
            ax.add_patch(circle)

            # 4. 画机械臂
            ax.plot(xs, ys, 'o-', color='steelblue', linewidth=3, markersize=8, label='Arm', zorder=3)

            # 5. 画基座
            ax.plot(0, 0, 'ks', markersize=12, label='Base', zorder=4)

            # 6. 画末端执行器 (强调显示)
            ax.plot(ee_pos[0], ee_pos[1], 'ro', markersize=10, label='End-Effector', zorder=6)

            # 标题信息
            status_str = "SUCCESS!" if success else "Moving..."
            color_str = "green" if success else "black"

            # 各部分奖励明细
            shaping = reward_components.get('shaping', 0)
            dist_r = reward_components.get('dist', 0)
            action_r = reward_components.get('action', 0)
            success_r = reward_components.get('success', 0)

            ax.set_title(
                f'Episode {episode+1} | Step {step}\n'
                f'Dist: {current_dist:.3f}m | {status_str}\n'
                f'Reward: shaping={shaping:+.3f} dist={dist_r:+.3f} action={action_r:+.3f} success={success_r:+.1f} | Total={reward:.3f}',
                fontsize=10, color=color_str, fontweight='bold'
            )

            if step == 1:
                ax.legend(loc='upper right', fontsize=9)

            # 保存帧
            fig.canvas.draw()
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)
            frames.append(img.copy())
            buf.close()

            if step % 20 == 0 or success:
                print(f"  Step {step}: Dist={current_dist:.3f}m | shaping={shaping:+.3f} dist={dist_r:+.3f} action={action_r:+.3f} success={success_r:+.1f} | Total={reward:.3f} {'OK' if success else ''}")

    plt.close(fig)

    # 保存 GIF
    if len(frames) > 0:
        print("\n保存 GIF (这可能需要几秒钟)...")
        output_path = 'assets/robot_arm_validation.gif'
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=100,  # 每帧 100ms
            loop=0
        )
        print(f"保存到 {output_path}")


def run_random_demo(n_steps=100):
    """运行随机策略演示"""
    print(f"\n运行随机策略演示 ({n_steps} 步)...")

    fig, ax = plt.subplots(figsize=(8, 8))

    obs, info = env.reset()

    for step in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()

        # ── 关键修复：从 info 获取真实状态 ──
        ee_pos = np.array(info['ee_pos'][:2])       # 绝对坐标
        joint_angles = np.array(info['joint_angles'])
        goal = np.array(info['target'][:2])
        current_dist = info['distance']

        positions = get_link_positions(joint_angles)

        ax.clear()
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # 画目标采样范围参考线
        theta_ref = np.linspace(0, 2*np.pi, 50)
        ax.plot(GOAL_RADIUS_MAX*np.cos(theta_ref), GOAL_RADIUS_MAX*np.sin(theta_ref), 'g--', alpha=0.2)
        ax.plot(GOAL_RADIUS_MIN*np.cos(theta_ref), GOAL_RADIUS_MIN*np.sin(theta_ref), 'b--', alpha=0.2)

        # 画目标
        ax.plot(goal[0], goal[1], 'g*', markersize=15)

        # 画机械臂
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        ax.plot(xs, ys, 'o-', color='steelblue', linewidth=3, markersize=8)

        # 末端
        ax.plot(ee_pos[0], ee_pos[1], 'ro', markersize=8)

        # 统计
        ax.set_title(f'Random Policy - Step {step}\nDistance: {current_dist:.3f}m')

        fig.canvas.draw()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=80)
        buf.seek(0)
        img = Image.open(buf)
        buf.close()

        if step % 20 == 0:
            print(f"  步数 {step}")

    plt.close()


if __name__ == "__main__":
    os.makedirs("assets", exist_ok=True)

    # 生成验证 GIF
    generate_validation_gif(n_episodes=3, frames_per_episode=50)

    print("\n完成!")
