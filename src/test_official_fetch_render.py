"""
官方 FetchReach 渲染测试 - 使用训练好的模型

运行：
    conda activate robotics
    python test_official_fetch_render.py --model assets/checkpoints/sac_official_fetch_250000_steps.zip
"""

import gymnasium as gym
import gymnasium_robotics
import argparse
import time
import os

def test_fetch_reach_with_model(model_path=None, n_steps=200):
    """使用训练好的模型测试 FetchReach 渲染"""
    print("=" * 60)
    print("FetchReach 模型渲染测试")
    print("=" * 60)

    if model_path and os.path.exists(model_path):
        print(f"加载模型: {model_path}")
        from stable_baselines3 import SAC
        model = SAC.load(model_path)
        print("模型加载成功!")
    else:
        print("警告: 模型未找到，使用随机策略")
        model = None

    # 测试两种环境
    for env_id in ["FetchReach-v4", "FetchReachDense-v4"]:
        print(f"\n{'='*40}")
        print(f"环境: {env_id}")
        print(f"{'='*40}")

        try:
            # 使用 human 渲染模式
            env = gym.make(env_id, render_mode="human")
            print(f"创建环境: OK")
            print(f"观测空间: {env.observation_space}")
            print(f"动作空间: {env.action_space}")

            # 重置
            obs, info = env.reset()
            episode_reward = 0
            episode_steps = 0

            print(f"\n开始渲染 (按 Ctrl+C 退出)...")
            print(f"{'步数':>6} {'奖励':>10} {'成功':>6} {'位置':>30}")
            print("-" * 60)

            for step in range(n_steps):
                # 获取动作
                if model:
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    action = env.action_space.sample()

                # 执行
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                episode_steps += 1

                # 渲染
                env.render()

                # 显示信息
                achieved = obs.get('achieved_goal', [0,0,0])
                desired = obs.get('desired_goal', [0,0,0])
                print(f"\r{episode_steps:>6} {episode_reward:>10.2f} {info.get('is_success', False):>6} "
                      f"pos=[{achieved[0]:.3f}, {achieved[1]:.3f}, {achieved[2]:.3f}]", end="")

                # 延迟 - 方便看清
                time.sleep(0.03)

                if terminated or truncated:
                    print(f"\n\nEpisode 结束!")
                    print(f"  总步数: {episode_steps}")
                    print(f"  总奖励: {episode_reward:.2f}")
                    print(f"  成功: {info.get('is_success', False)}")
                    print(f"  最终距离: {info.get('distance', 'N/A')}")
                    print("-" * 60)

                    # 重置
                    obs, info = env.reset()
                    episode_reward = 0
                    episode_steps = 0
                    time.sleep(0.5)  # 重置后暂停

            env.close()

        except KeyboardInterrupt:
            print(f"\n\n用户中断")
            env.close()
            break
        except Exception as e:
            print(f"\n错误: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


def test_pybullet_render():
    """测试 PyBullet 渲染"""
    print("\n" + "=" * 60)
    print("PyBullet 渲染测试")
    print("=" * 60)

    try:
        import pybullet as p
        import pybullet_data

        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        p.loadURDF("plane.urdf")
        robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

        print("  连接 GUI: OK")
        print("  加载 Panda: OK")

        import time
        print("\n开始模拟 (按 Ctrl+C 退出)...")
        for i in range(500):
            p.stepSimulation()
            time.sleep(0.02)

        p.disconnect()
        print("\n  PyBullet 渲染: OK")

    except Exception as e:
        print(f"  PyBullet 错误: {e}")

    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FetchReach 渲染测试')
    parser.add_argument('--model', type=str, default=None, help='模型路径')
    parser.add_argument('--steps', type=int, default=500, help='最大步数')
    args = parser.parse_args()

    # 找模型
    model_path = args.model
    if model_path is None:
        import glob
        files = glob.glob('assets/checkpoints/sac_official_fetch_*.zip')
        if files:
            files.sort(key=os.path.getmtime, reverse=True)
            model_path = files[0]

    if model_path:
        print(f"使用模型: {model_path}")
    else:
        print("未找到模型，使用随机策略")

    print()
    test_fetch_reach_with_model(model_path, n_steps=args.steps)
    # test_pybullet_render()  # 可选：同时测试 PyBullet
