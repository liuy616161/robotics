"""
Step 2：REINFORCE（策略梯度，纯 numpy 实现）

任务：在 CartPole-v1 上用 Monte Carlo 策略梯度（REINFORCE）训练一个 2 层策略网络
方法：纯 numpy 手写神经网络的前向传播和反向传播，不依赖 PyTorch / TensorFlow

运行：python3.9 src/step2_reinforce.py

为什么用 numpy？
  Q-learning 的局限：只适合离散动作，且 Q-table 在高维状态下会爆炸
  策略梯度直接优化策略参数，支持连续动作，是现代 RL（PPO/SAC/TD3）的基础
  用 numpy 手写是为了真正理解反向传播，而不是黑盒调用框架

依赖：numpy, matplotlib, gymnasium
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

try:
    import gymnasium as gym
except ImportError:
    raise ImportError("请先运行：pip3 install gymnasium")


# ── 超参数配置 ────────────────────────────────────────────────
LEARNING_RATE = 0.005  # 梯度上升步长（注意：是最大化回报，所以是上升）
GAMMA         = 0.99   # 折扣因子
N_EPISODES    = 1200   # 训练轮数（纯 numpy 收敛比框架慢，需要多跑）
MAX_STEPS     = 500    # 每轮最大步数
HIDDEN_SIZE   = 16     # 隐层神经元数量
N_STATE       = 4      # CartPole 观测维度（[pos, vel, angle, ang_vel]）
N_ACTIONS     = 2      # CartPole 动作数（0=左, 1=右）
PRINT_EVERY   = 100    # 每隔多少轮打印一次进度
SEED          = 42     # 随机种子（保证结果可复现）


# ── 策略网络（2层 MLP，纯 numpy）─────────────────────────────

class PolicyNetwork:
    """
    2层全连接策略网络

    结构：
        输入层 (4) → 隐层 (16, ReLU) → 输出层 (2, Softmax)

    输出是动作的概率分布：
        [P(动作=0), P(动作=1)]，两者之和 = 1

    参数（需要训练的权重）：
        W1: (16, 4)   输入→隐层的权重矩阵
        b1: (16,)     隐层偏置
        W2: (2, 16)   隐层→输出的权重矩阵
        b2: (2,)      输出偏置

    总参数量：16×4 + 16 + 2×16 + 2 = 64 + 16 + 32 + 2 = 114 个参数
    （对比 Q-table Step 1：648 个格子 × 有限精度）
    """

    def __init__(self, n_state: int, n_hidden: int, n_actions: int, seed: int = 42):
        rng = np.random.default_rng(seed)

        # Xavier 初始化：权重方差 = 2 / (fan_in + fan_out)
        # 目的：让各层输出的方差大致相同，避免梯度消失/爆炸
        scale1 = np.sqrt(2.0 / (n_state + n_hidden))
        scale2 = np.sqrt(2.0 / (n_hidden + n_actions))

        self.W1 = rng.normal(0, scale1, (n_hidden, n_state)).astype(np.float64)
        self.b1 = np.zeros(n_hidden, dtype=np.float64)
        self.W2 = rng.normal(0, scale2, (n_actions, n_hidden)).astype(np.float64)
        self.b2 = np.zeros(n_actions, dtype=np.float64)

        # 保存前向传播的中间结果（反向传播时需要用）
        self._h     = None   # 隐层激活值（ReLU 之后）
        self._prob  = None   # 输出的动作概率

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """ReLU 激活：max(0, x)，负数置零，引入非线性"""
        return np.maximum(0.0, x)

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """
        数值稳定的 Softmax：将 logits 转为概率分布

        softmax(x_i) = exp(x_i) / Σ exp(x_j)

        数值稳定技巧：先减去最大值，防止 exp 溢出
          softmax(x) = softmax(x - max(x))（数学上等价，但数值更稳定）
        """
        x_shifted = x - np.max(x)   # 数值稳定：减最大值
        exp_x = np.exp(x_shifted)
        return exp_x / exp_x.sum()

    def forward(self, state: np.ndarray) -> np.ndarray:
        """
        前向传播：输入状态 → 输出动作概率

        参数：
            state : 观测值，shape (4,)

        返回：
            prob  : 动作概率，shape (2,)，和为 1

        中间变量（保存用于反向传播）：
            self._h    = ReLU(W1 @ state + b1)   隐层激活
            self._prob = softmax(W2 @ h + b2)     输出概率
        """
        # 第一层：线性变换 + ReLU
        z1 = self.W1 @ state + self.b1       # (16,)，线性
        h  = self.relu(z1)                   # (16,)，ReLU 激活
        self._h = h                          # 保存，反传用

        # 第二层：线性变换 + Softmax
        logits = self.W2 @ h + self.b2       # (2,)，未归一化的对数概率
        prob   = self.softmax(logits)        # (2,)，归一化为概率

        self._prob = prob                    # 保存，反传用
        return prob

    def update(self, state: np.ndarray, action: int, advantage: float, lr: float):
        """
        策略梯度更新（反向传播 + 梯度上升）

        策略梯度定理：
          ∇J(θ) = E[advantage × ∇ log π(a|s; θ)]

          其中 advantage = G_t - baseline（折扣回报减去基线）

        对数概率的梯度（softmax + cross-entropy 的组合）：
          ∂(-log π(a)) / ∂logits = softmax(logits) - one_hot(a)
                                  = prob - one_hot(a)

          解释：这是 softmax 交叉熵损失的梯度，可以直接用结论

        梯度上升（最大化期望回报）：
          θ ← θ + lr × advantage × ∇ log π(a|s; θ)

        参数：
            state     : 当前状态（前向传播已在 forward 中完成）
            action    : 实际执行的动作
            advantage : G_t - baseline（控制更新方向和幅度）
            lr        : 学习率
        """
        h    = self._h       # 隐层激活，shape (16,)
        prob = self._prob    # 动作概率，shape (2,)

        # ── TODO 1：实现策略梯度反向传播 ──────────────────────
        #
        # 步骤（请参考 GUIDE.md 中的提示3）：
        #
        # 1. 计算输出层梯度（softmax 对数概率梯度）：
        #    dlogits = prob.copy()
        #    dlogits[action] -= 1.0     # 减去 one-hot(action)
        #    dlogits *= -advantage      # 乘以 -advantage（负号因为是梯度上升）
        #
        # 2. 输出层参数梯度：
        #    dW2 = np.outer(dlogits, h)  # (2, 16)
        #    db2 = dlogits               # (2,)
        #
        # 3. 反传到隐层：
        #    dh = self.W2.T @ dlogits    # (16,)
        #    dh[h <= 0] = 0              # ReLU 的梯度（h<=0 的位置梯度为 0）
        #
        # 4. 输入层参数梯度：
        #    dW1 = np.outer(dh, state)   # (16, 4)
        #    db1 = dh                    # (16,)
        #
        # 5. 梯度上升更新（注意：是减去 dW，因为 dlogits 里已经含负号）：
        #    self.W2 -= lr * dW2
        #    self.b2 -= lr * db2
        #    self.W1 -= lr * dW1
        #    self.b1 -= lr * db1
        #
        raise NotImplementedError("请实现 PolicyNetwork.update() 中的反向传播（TODO 1）")


# ── 折扣回报计算 ─────────────────────────────────────────────

def compute_returns(rewards: list, gamma: float = GAMMA) -> np.ndarray:
    """
    计算每个时间步的折扣回报 G_t

    定义：
      G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ... + γ^{T-t}·r_T
           = Σ_{k=0}^{T-t} γ^k · r_{t+k}

    递推关系：
      G_T = r_T                    （最后一步）
      G_t = r_t + γ · G_{t+1}    （从后往前计算，O(T) 时间复杂度）

    为什么要折扣？
      - 远期奖励不确定性更大，折扣系数 γ < 1 反映这种不确定性
      - 同时保证无限时间轴的回报是有限值（等比级数收敛）
      - γ = 0.99 表示100步之后的奖励只有现在的 0.99^100 ≈ 0.37

    参数：
        rewards : 一条轨迹的奖励序列，长度 T
        gamma   : 折扣因子（0 < γ ≤ 1）

    返回：
        G : 每步的折扣回报，shape (T,)，G[t] = G_t
    """
    # TODO 2：实现折扣回报计算（提示：从后往前递推，见 GUIDE.md 提示1）
    #
    # 步骤：
    #   G = np.zeros(len(rewards))
    #   G[-1] = rewards[-1]
    #   for t in reversed(range(len(rewards) - 1)):
    #       G[t] = rewards[t] + gamma * G[t+1]
    #   return G
    #
    raise NotImplementedError("请实现 compute_returns() 函数（TODO 2）")


def normalize_returns(G: np.ndarray) -> np.ndarray:
    """
    标准化折扣回报（减均值，除标准差）

    为什么要标准化？
      - 不同 episode 的回报量级差异很大（短 episode 回报低，长 episode 回报高）
      - 标准化后梯度方差更小，训练更稳定（这是 REINFORCE with baseline 的简化版本）
      - 严格的 baseline 方法是减去 V(s)，这里用整条轨迹的均值近似

    注意：eps 防止除以零（当所有奖励相同时 std=0）
    """
    mean = G.mean()
    std  = G.std() + 1e-8   # eps = 1e-8，防止除以零
    return (G - mean) / std


# ── 训练主循环 ────────────────────────────────────────────────

def train():
    """
    REINFORCE 训练主函数

    REINFORCE 算法流程（Monte Carlo 策略梯度）：
      for each episode:
        1. 用当前策略 π_θ 收集一条完整轨迹：(s_0, a_0, r_0, s_1, a_1, r_1, ...)
        2. 计算每步的折扣回报 G_t
        3. 对每步 t，用策略梯度更新参数：
           θ ← θ + α × G_t × ∇ log π_θ(a_t | s_t)
    """
    np.random.seed(SEED)
    env = gym.make("CartPole-v1")

    # 初始化策略网络
    policy = PolicyNetwork(N_STATE, HIDDEN_SIZE, N_ACTIONS, seed=SEED)

    episode_rewards = []      # 记录每轮总奖励
    running_reward  = 0.0     # 指数移动平均奖励（监控趋势）

    print("=" * 60)
    print("REINFORCE 训练（纯 numpy，无框架）")
    print(f"网络结构：{N_STATE} → {HIDDEN_SIZE}(ReLU) → {N_ACTIONS}(Softmax)")
    print(f"总参数量：{policy.W1.size + policy.b1.size + policy.W2.size + policy.b2.size}")
    print("=" * 60)

    for episode in range(N_EPISODES):
        obs, _ = env.reset(seed=SEED + episode)

        # ── 收集一条完整轨迹 ────────────────────────────────
        states, actions, rewards = [], [], []

        for step in range(MAX_STEPS):
            state = obs.astype(np.float64)

            # 前向传播：得到动作概率
            prob = policy.forward(state)

            # 按概率分布采样动作（探索！）
            # np.random.choice([0,1], p=[p0, p1])
            action = np.random.choice(N_ACTIONS, p=prob)

            # 执行动作
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 记录这一步
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            obs = next_obs
            if done:
                break

        # ── 计算折扣回报 ─────────────────────────────────
        G = compute_returns(rewards, GAMMA)

        # 标准化回报（baseline 减方差）
        G_normalized = normalize_returns(G)

        # ── 更新策略（每步都更新）────────────────────────
        for t in range(len(states)):
            policy.update(
                state    = states[t],
                action   = actions[t],
                advantage = G_normalized[t],  # 用标准化后的 G_t 作为 advantage
                lr       = LEARNING_RATE
            )

        # ── 记录 & 打印 ──────────────────────────────────
        total_reward = sum(rewards)
        episode_rewards.append(total_reward)

        # 指数移动平均（比算术平均对近期数据更敏感）
        running_reward = 0.05 * total_reward + 0.95 * running_reward

        if (episode + 1) % PRINT_EVERY == 0:
            recent_avg = np.mean(episode_rewards[-PRINT_EVERY:])
            print(f"  Episode {episode+1:5d}/{N_EPISODES}  "
                  f"最近{PRINT_EVERY}轮均值: {recent_avg:6.1f}  "
                  f"运行均值: {running_reward:6.1f}")

    env.close()
    final_avg = np.mean(episode_rewards[-100:])
    print(f"\n训练完成！最终 100 轮平均奖励：{final_avg:.1f}")
    return policy, episode_rewards


# ── 评估 ─────────────────────────────────────────────────────

def evaluate(policy: PolicyNetwork, n_episodes: int = 20) -> float:
    """
    评估策略性能：使用贪心策略（选概率最大的动作，不采样）
    评估时不探索，直接看学到的策略有多好
    """
    env = gym.make("CartPole-v1")
    rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        total_reward = 0.0
        for _ in range(MAX_STEPS):
            state = obs.astype(np.float64)
            prob = policy.forward(state)
            action = int(np.argmax(prob))   # 贪心：选概率最大的动作
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        rewards.append(total_reward)

    env.close()
    mean_r = np.mean(rewards)
    std_r  = np.std(rewards)
    print(f"\n评估（{n_episodes} 轮，贪心策略）：平均奖励 {mean_r:.1f} ± {std_r:.1f}")
    return mean_r


# ── 可视化 ───────────────────────────────────────────────────

def plot_results(episode_rewards: list):
    """画训练学习曲线"""
    os.makedirs("assets", exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    episodes = np.arange(1, len(episode_rewards) + 1)
    ax.plot(episodes, episode_rewards, alpha=0.35, color='steelblue',
            linewidth=0.8, label='每轮奖励')

    # 滑动平均（窗口 100）
    window = 100
    if len(episode_rewards) >= window:
        smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax.plot(np.arange(window, len(episode_rewards)+1), smoothed,
                color='red', linewidth=2, label=f'{window}轮滑动平均')

    ax.axhline(195, color='green', linestyle='--', alpha=0.7, label='通关线 (195)')
    ax.axhline(100, color='orange', linestyle=':', alpha=0.7, label='目标线 (100)')

    ax.set_xlabel('训练轮数 (Episode)')
    ax.set_ylabel('总奖励 (每轮存活步数)')
    ax.set_title('REINFORCE（纯 numpy 策略梯度）on CartPole-v1\n'
                 f'网络：{N_STATE}→{HIDDEN_SIZE}(ReLU)→{N_ACTIONS}(Softmax)，'
                 f'学习率={LEARNING_RATE}，γ={GAMMA}')
    ax.legend(loc='upper left')
    ax.set_ylim(0, MAX_STEPS + 10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = 'assets/step2_learning_curve.png'
    plt.savefig(path, dpi=120, bbox_inches='tight')
    print(f"学习曲线已保存到 {path}")
    plt.close()


# ── 网络权重可视化 ───────────────────────────────────────────

def visualize_weights(policy: PolicyNetwork):
    """
    可视化训练后的网络权重分布
    有助于理解网络是否正常训练（权重分布是否合理）
    """
    os.makedirs("assets", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    weights = [('W1 (输入→隐层)', policy.W1.flatten()),
               ('W2 (隐层→输出)', policy.W2.flatten())]

    for ax, (name, w) in zip(axes, weights):
        ax.hist(w, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
        ax.set_title(f'{name}\nmean={w.mean():.3f}, std={w.std():.3f}')
        ax.set_xlabel('权重值')
        ax.set_ylabel('频数')
        ax.axvline(0, color='red', linewidth=1)
        ax.grid(True, alpha=0.3)

    plt.suptitle('训练后的网络权重分布（健康的网络权重应该均匀分布在 0 附近）')
    plt.tight_layout()
    path = 'assets/step2_weights.png'
    plt.savefig(path, dpi=120, bbox_inches='tight')
    print(f"权重分布图已保存到 {path}")
    plt.close()


# ── 主程序 ──────────────────────────────────────────────────

if __name__ == "__main__":
    print("Step 2：REINFORCE（策略梯度）on CartPole-v1")
    print("方法：纯 numpy 手写策略网络 + Monte Carlo 梯度估计\n")

    # 训练
    policy, rewards = train()

    # 评估
    evaluate(policy)

    # 画图
    plot_results(rewards)
    visualize_weights(policy)

    print("\n── Q-learning vs REINFORCE 对比 ──")
    print("Q-learning（Step 1）：")
    print("  ✓ 简单直观，理论保证收敛")
    print("  ✗ 只适合离散动作，状态空间大时 Q-table 内存爆炸")
    print("  ✗ 需要手动离散化，信息损失")
    print()
    print("REINFORCE（Step 2）：")
    print("  ✓ 支持连续动作，可直接作用于原始状态")
    print("  ✓ 是现代 PPO/SAC 的理论基础")
    print("  ✗ 梯度方差大（Monte Carlo 估计），收敛慢")
    print("  ✗ 纯 numpy 实现速度慢")
    print()
    print("→ Step 3 会用 stable-baselines3 的 SAC 解决上面这些问题")
