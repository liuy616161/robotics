"""
Step 1：Q-learning（表格式强化学习）

任务：在 CartPole-v1 环境中，用 Q-table 学会平衡倒立摆
方法：把连续观测离散化 → 建 Q-table → Bellman 方程迭代更新

运行：python3.9 src/step1_q_learning.py

环境要求：pip3 install gymnasium
依赖：numpy, matplotlib, gymnasium

作者提示：先阅读注释和 TODO，理解算法流程后再填写代码
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无显示器时用 Agg 后端
import matplotlib.pyplot as plt
import os

# 尝试导入 gymnasium（需要先安装）
try:
    import gymnasium as gym
except ImportError:
    raise ImportError("请先运行：pip3 install gymnasium")


# ── 超参数配置（每个参数都有解释）────────────────────────────────
ALPHA       = 0.1     # 学习率：每次更新步长，太大震荡，太小收敛慢
GAMMA       = 0.99    # 折扣因子：未来奖励的权重，0.99 表示非常重视未来
EPSILON     = 1.0     # 初始探索率：1.0 = 完全随机探索
EPSILON_MIN = 0.01    # 最小探索率：即使训练好了，保留 1% 的随机性
EPSILON_DECAY = 0.995 # 每轮结束后 epsilon 乘以这个系数（慢慢减少探索）
N_EPISODES  = 500     # 训练轮数（episode）
MAX_STEPS   = 500     # 每轮最多走多少步（CartPole 最大 500 步）

# ── Q-table 离散化参数──────────────────────────────────────────
# CartPole 观测：[小车位置, 小车速度, 杆角度, 杆角速度]
# 这 4 个都是连续值，Q-table 需要离散的索引
# 每个维度分成 N_BINS 个 bin（桶），总状态数 = 各维 bin 数之积

N_BINS = [3, 3, 6, 6]   # 四个观测维度各自的 bin 数
                          # 总状态数：3×3×6×6 = 324，已经不小了！
                          # 实验证明：角度和角速度对控制影响更大，给更多 bin

# 观测值的截断范围（CartPole 的速度理论上无界，但实际不会太大）
OBS_LOW  = np.array([-2.4,  -3.0,  -0.2095, -3.0])  # 各维度最小值
OBS_HIGH = np.array([ 2.4,   3.0,   0.2095,  3.0])  # 各维度最大值


# ── 工具函数 ─────────────────────────────────────────────────

def discretize(obs: np.ndarray) -> tuple:
    """
    将连续观测映射到 Q-table 的整数索引

    原理：把每个连续值的范围均匀分成 N 个 bin，
         用 np.digitize 找到该值落在哪个 bin 里

    参数：
        obs : 连续观测，shape (4,)，即 CartPole 的 [pos, vel, angle, ang_vel]

    返回：
        tuple of int，长度 4，每个值是 0 ~ N_BINS[i]-1 之间的整数
        例如：(1, 2, 3, 5) 表示该状态对应 Q-table 的第 [1,2,3,5] 个格子

    注意：
        - 超出范围的值先 clip 再 digitize（防止索引越界）
        - np.digitize 返回从 1 开始的索引，需要减 1
    """
    # TODO 1：实现观测离散化（提示：用 np.digitize，注意边界处理）
    #
    # 步骤：
    #   for i in range(4):
    #     1. 把 obs[i] clip 到 [OBS_LOW[i], OBS_HIGH[i]]
    #     2. 创建 bin 边界：np.linspace(OBS_LOW[i], OBS_HIGH[i], N_BINS[i]+1)
    #     3. 用 np.digitize 找到 bin 编号（从0开始，clip到合法范围）
    #
    indices = []
    for i in range(4):
        clipped = np.clip(obs[i], OBS_LOW[i], OBS_HIGH[i])
        bins = np.linspace(OBS_LOW[i], OBS_HIGH[i], N_BINS[i]+1)
        idx = np.digitize(clipped, bins) - 1
        idx = np.clip(idx, 0, N_BINS[i] - 1)
        
        indices.append(idx)
    
    return tuple(indices)


def epsilon_greedy(Q: np.ndarray, state: tuple, epsilon: float, n_actions: int) -> int:
    """
    Epsilon-greedy 动作选择策略

    含义：
      以 epsilon 的概率随机选动作（探索 Exploration）
      以 1-epsilon 的概率选 Q 值最大的动作（利用 Exploitation）

    为什么需要探索？
      如果总是选最优动作，可能永远无法发现更好的路径（局部最优）
      探索-利用权衡（Exploration-Exploitation Tradeoff）是 RL 的核心问题

    参数：
        Q        : Q-table，numpy 数组
        state    : 离散化后的状态索引（tuple）
        epsilon  : 探索率
        n_actions: 动作数量（CartPole 是 2）

    返回：
        动作索引（0 或 1）
    """
    # TODO 2：实现 epsilon-greedy 策略（提示：用 np.random.random() 和 np.argmax）
    #
    # 逻辑：
    #   if 随机数 < epsilon:
    #       return 随机动作（np.random.randint(n_actions)）
    #   else:
    #       return Q[state] 中最大值对应的动作（np.argmax(Q[state])）
    #

    if np.random.random() < epsilon:
        return np.random.randint(n_actions)
    else:
        return np.argmax(Q[state])

    
def update_q_table(Q: np.ndarray,
                   state: tuple,
                   action: int,
                   reward: float,
                   next_state: tuple,
                   done: bool) -> float:
    """
    用 Bellman 方程更新 Q-table

    Bellman 方程（TD 更新）：
      Q(s, a) ← Q(s, a) + α × [r + γ × max_a' Q(s', a') - Q(s, a)]
                                 ↑                              ↑
                              TD 目标                       TD 误差

    Q 值的含义：从状态 s 执行动作 a 后，沿最优策略走下去的期望总折扣奖励

    参数：
        Q        : Q-table（会被原地修改）
        state    : 当前离散状态
        action   : 执行的动作
        reward   : 得到的即时奖励
        next_state: 下一个离散状态
        done     : 是否到达终态（done=True 时没有后续奖励）

    返回：
        td_error  : TD 误差（绝对值），用于监控训练
    """
    # TODO 3：实现 Q-table 更新（提示：Bellman 方程，注意 done 时 max Q(s') = 0）
    #
    # 步骤：
    #   1. td_target = reward + gamma * max(Q[next_state]) * (1 - done)
    #      （done=True 表示终态，终态没有后续价值，所以乘 (1-done)）
    #   2. td_error  = td_target - Q[state + (action,)]
    #      （state 是 tuple，加上 (action,) 变成完整索引）
    #   3. Q[state + (action,)] += alpha * td_error
    #   4. return abs(td_error)
    #
    
    td_target = reward + GAMMA * np.max(Q[next_state]) * (1 - done)
    td_error = td_target - Q[state + (action,)]
    Q[state + (action,)] += ALPHA * td_error
    return abs(td_error)


# ── 训练主循环 ────────────────────────────────────────────────

def train(render=False):
    """
    Q-learning 训练主函数

    流程：
      for each episode:
        reset 环境 → 得到初始观测
        for each step:
          离散化观测 → epsilon-greedy 选动作 → 执行动作
          → 得到 reward, next_obs, done
          → 更新 Q-table
          → epsilon 衰减
        记录本轮总奖励
    """
    # 创建环境
    env = gym.make("CartPole-v1")
    n_actions = env.action_space.n  # CartPole 有 2 个动作

    # 初始化 Q-table：全零
    # 形状：(*N_BINS, n_actions) = (3, 3, 6, 6, 2)
    # 即 3×3×6×6 种状态，每种状态有 2 个动作的 Q 值
    Q = np.zeros(N_BINS + [n_actions])
    print(f"Q-table 形状：{Q.shape}，总格子数：{Q.size}")
    print(f"  （{N_BINS[0]}×{N_BINS[1]}×{N_BINS[2]}×{N_BINS[3]} 种状态 × {n_actions} 个动作）")

    epsilon = EPSILON         # 初始探索率
    episode_rewards = []      # 记录每轮总奖励
    td_errors = []            # 记录平均 TD 误差（监控收敛）

    print(f"\n开始训练（{N_EPISODES} 轮）...")

    for episode in range(N_EPISODES):
        # reset() 返回 (obs, info)，取第一个
        obs, _ = env.reset()
        state = discretize(obs)         # 离散化初始观测
        total_reward = 0.0
        episode_td_errors = []

        for step in range(MAX_STEPS):
            # 1. 选动作（epsilon-greedy）
            action = epsilon_greedy(Q, state, epsilon, n_actions)

            # 2. 执行动作
            # gymnasium 的 step() 返回 5 个值：
            #   obs, reward, terminated, truncated, info
            # terminated: 杆倒了或小车越界（真正的失败）
            # truncated:  超过最大步数（时间限制）
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # 3. 离散化下一状态
            next_state = discretize(next_obs)

            # 4. 更新 Q-table
            td_error = update_q_table(Q, state, action, reward, next_state, done)
            episode_td_errors.append(td_error)

            # 5. 更新状态
            state = next_state
            total_reward += reward

            if done:
                break

        # 每轮结束后，衰减 epsilon
        # TODO 4：实现 epsilon 衰减（提示：epsilon = max(epsilon_min, epsilon * decay)）
        # epsilon = ...
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        
        episode_rewards.append(total_reward)
        td_errors.append(np.mean(episode_td_errors))

        # 每 50 轮打印一次进度
        if (episode + 1) % 50 == 0:
            recent_avg = np.mean(episode_rewards[-50:])
            print(f"  Episode {episode+1:4d}/{N_EPISODES}  "
                  f"最近50轮平均奖励: {recent_avg:6.1f}  "
                  f"epsilon: {epsilon:.3f}")

    env.close()
    print(f"\n训练完成！最终 100 轮平均奖励：{np.mean(episode_rewards[-100:]):.1f}")
    return Q, episode_rewards, td_errors


# ── 评估（不探索，完全贪心）─────────────────────────────────

def evaluate(Q: np.ndarray, n_episodes: int = 20) -> float:
    """
    评估训练好的 Q-table 的性能
    评估时 epsilon=0（完全贪心，不随机），看真实能力
    """
    env = gym.make("CartPole-v1")
    rewards = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        for _ in range(MAX_STEPS):
            state = discretize(obs)
            # 贪心选动作：完全不探索
            action = int(np.argmax(Q[state]))
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        rewards.append(total_reward)

    env.close()
    mean_r = np.mean(rewards)
    std_r  = np.std(rewards)
    print(f"\n评估（{n_episodes} 轮，epsilon=0）：平均奖励 {mean_r:.1f} ± {std_r:.1f}")
    return mean_r


# ── 可视化 ───────────────────────────────────────────────────

def plot_results(episode_rewards: list, td_errors: list):
    """画学习曲线：奖励随训练轮数的变化"""
    os.makedirs("assets", exist_ok=True)  # 确保 assets 目录存在

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # 上图：奖励曲线
    episodes = np.arange(1, len(episode_rewards) + 1)
    ax1.plot(episodes, episode_rewards, alpha=0.4, color='steelblue', linewidth=0.8,
             label='每轮奖励')

    # 用滑动平均平滑曲线（窗口 50 轮）
    window = 50
    if len(episode_rewards) >= window:
        smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(np.arange(window, len(episode_rewards)+1), smoothed,
                 color='red', linewidth=2, label=f'{window}轮滑动平均')

    ax1.axhline(195, color='green', linestyle='--', alpha=0.7, label='通关线 (195)')
    ax1.set_xlabel('训练轮数 (Episode)')
    ax1.set_ylabel('总奖励')
    ax1.set_title('Q-learning on CartPole-v1\n蓝色=每轮奖励，红色=50轮均值，绿线=通关线')
    ax1.legend(loc='upper left')
    ax1.set_ylim(0, MAX_STEPS + 10)
    ax1.grid(True, alpha=0.3)

    # 下图：TD 误差（反映 Q 值的变化幅度，应该随训练减小）
    ax2.plot(episodes, td_errors, alpha=0.6, color='orange', linewidth=0.8)
    ax2.set_xlabel('训练轮数 (Episode)')
    ax2.set_ylabel('平均 TD 误差')
    ax2.set_title('TD 误差变化（越小说明 Q 值越稳定）')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = 'assets/step1_learning_curve.png'
    plt.savefig(path, dpi=120, bbox_inches='tight')
    print(f"学习曲线已保存到 {path}")
    plt.close()


# ── Q-table 分析 ─────────────────────────────────────────────

def analyze_q_table(Q: np.ndarray):
    """
    分析训练好的 Q-table：打印最常用的动作分布

    有意思的洞察：
    - Q-table 对角速度最敏感（最后两个维度变化大）
    - 中心区域（小车在中间）的 Q 值比较均匀
    - 边缘区域（小车快要出界）的 Q 值差异悬殊
    """
    print("\n── Q-table 分析 ──")
    print(f"Q 值范围：min={Q.min():.2f}, max={Q.max():.2f}, mean={Q.mean():.2f}")

    # 最优动作分布：看哪个动作被选的多
    best_actions = np.argmax(Q, axis=-1)  # 每个状态的最优动作（0或1）
    n_left  = (best_actions == 0).sum()   # 向左推的状态数
    n_right = (best_actions == 1).sum()   # 向右推的状态数
    print(f"最优动作分布：向左(0)={n_left}个状态，向右(1)={n_right}个状态")

    # 动作差异（|Q(s,0) - Q(s,1)|）越大说明该状态的动作选择越明确
    action_diff = np.abs(Q[..., 0] - Q[..., 1])
    print(f"动作 Q 值差异：mean={action_diff.mean():.2f}，"
          f"max={action_diff.max():.2f}（差异大 → 策略更确定）")


# ── 主程序 ──────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Step 1：Q-learning on CartPole-v1")
    print("算法：表格式 Q-learning（Bellman 方程迭代）")
    print("=" * 60)

    print("\n关键超参数：")
    print(f"  学习率 α = {ALPHA}")
    print(f"  折扣因子 γ = {GAMMA}")
    print(f"  初始探索率 ε = {EPSILON} → 最终 ≈ {EPSILON * EPSILON_DECAY**N_EPISODES:.3f}")
    print(f"  训练轮数 = {N_EPISODES}")

    # 训练
    Q, rewards, td_errors = train()

    # 评估
    evaluate(Q)

    # 画图
    plot_results(rewards, td_errors)

    # 分析
    analyze_q_table(Q)

    print("\n── 思考题（不需要写代码，但要能回答）──")
    print("1. 为什么 Q-table 不适合连续动作空间？")
    print("   提示：如果动作也是连续的，Q-table 维度会爆炸，而且 max_a Q(s,a) 无法穷举")
    print("2. epsilon 为什么要逐渐减小，而不是一直保持 1.0？")
    print("   提示：探索-利用权衡；训练初期需要探索，后期应该利用已学到的知识")
    print("3. CartPole 的 Q-table 有 3×3×6×6×2 = 648 个格子，")
    print("   如果观测是 84×84 的图像（Atari 游戏风格），Q-table 要多大？")
    print("   → 这就是为什么需要深度 Q-network（DQN）：用神经网络代替 Q-table")
