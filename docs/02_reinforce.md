# Step 2: REINFORCE（策略梯度算法）

## 目录

1. [为什么需要策略梯度](#1-为什么需要策略梯度)
2. [算法原理](#2-算法原理)
3. [网络结构](#3-网络结构)
4. [反向传播推导](#4-反向传播推导)
5. [代码实现](#5-代码实现)
6. [方差 reduction](#6-方差-reduction)
7. [面试考点](#7-面试考点)

---

## 1. 为什么需要策略梯度

### 1.1 Q-Learning 的瓶颈

| 问题 | 原因 |
|------|------|
| 只能处理离散动作 | Q-table 无法穷举连续动作 |
| 维度灾难 | 状态空间大将导致表格爆炸 |
| 无法处理图像输入 | 状态空间连续且高维 |

### 1.2 策略梯度 vs Q-Learning

```
Q-Learning:  状态 → 查表 → Q值 → 选最大动作

策略梯度:    状态 → 神经网络 → 动作概率分布 → 采样
```

### 1.3 策略梯度的优势

- **连续动作空间**: 输出动作分布（如高斯分布），可以采样出任意连续值
- **随机策略**: 自然地表达不确定性
- **端到端**: 直接优化最终目标

---

## 2. 算法原理

### 2.1 策略梯度定理

```
∇J(θ) = E_τ[ Σ_t ∇log π_θ(a_t | s_t) × G_t ]
```

**直观理解**:
- 如果某个动作的回报 G_t 很高 → 增大选择它的概率
- 如果某个动作的回报很低 → 减小选择它的概率

### 2.2 折扣回报 G_t

```
G_t = r_t + γ×r_{t+1} + γ²×r_{t+2} + ... + γ^{T-t}×r_T
```

递推计算（从后往前）:
```python
G = np.zeros(T)
G[-1] = rewards[-1]
for t in reversed(range(T-1)):
    G[t] = rewards[t] + gamma * G[t+1]
```

### 2.3 REINFORCE 流程

```
1. 采集一个 episode: (s_0, a_0, r_0, s_1, ...)
2. 计算每个时间步的 G_t
3. 对每个 (s_t, a_t, G_t):
   - 前向传播得到 π(a_t|s_t)
   - 计算 loss = -log π(a_t|s_t) × G_t
   - 反向传播更新网络
4. 重复
```

---

## 3. 网络结构

### 3.1 纯 NumPy 实现（无 PyTorch）

```
输入层(4) → 隐藏层(16, ReLU) → 输出层(2, softmax)
```

### 3.2 前向传播

```python
# 输入
state: (4,)

# 隐藏层
h = ReLU(W1 @ state + b1)  # (16,) = (16,4) @ (4,) + (16,)

# 输出 logits
logits = W2 @ h + b2  # (2,) = (2,16) @ (16,) + (2,)

# softmax 得到概率
prob = softmax(logits)  # (2,), sum = 1.0
```

### 3.3 ReLU 激活函数

```
ReLU(x) = max(0, x)

示例:
  x = -0.5 → ReLU = 0
  x =  0.5 → ReLU = 0.5
```

作用: 引入非线性，使网络能学习复杂模式

### 3.4 Softmax 函数

```python
softmax(x)_i = exp(x_i) / Σ_j exp(x_j)

示例:
  logits = [2.0, 1.0]
  exp([2,1]) = [7.39, 2.72]
  sum = 10.11
  prob = [0.73, 0.27]
```

---

## 4. 反向传播推导

### 4.1 计算图

```
state (4,) → W1(16,4) → ReLU → h (16,) → W2(2,16) → softmax → prob (2,)
                    ↑                                    ↓
                    ←←←←←←←←←←←←←←←←←←←←←←←←← action, G_t
```

### 4.2 反向传播步骤

**Step 1: softmax + cross-entropy 梯度**

```python
dlogits = prob.copy()          # dlogits = [p_0, p_1]
dlogits[action] -= 1.0          # dlogits = [p_0 - 1, p_1]
dlogits *= -advantage           # dlogits *= -G_t
```

**推导**: 对于 cross-entropy loss = -log π(a)，梯度恰好是 `prob - one_hot(action)`。

**Step 2: 对 W2, b2 的梯度**

```python
dW2 = np.outer(dlogits, h)  # (2,16) = (2,) outer (16,)
db2 = dlogits.copy()          # (2,)
```

**Step 3: 反传到隐藏层**

```python
dh = W2.T @ dlogits          # (16,) = (16,2) @ (2,)
dh[h <= 0] = 0                # ReLU 梯度: 通过的保留，截断的置零
```

**ReLU 梯度**:
```
Forward:  x > 0 → y = x
Backward: dy/dx = 1 (若 x > 0), 否则 0
```

**Step 4: 对 W1, b1 的梯度**

```python
dW1 = np.outer(dh, state)  # (16,4) = (16,) outer (4,)
db1 = dh.copy()              # (16,)
```

**Step 5: 梯度更新**

```python
W2 -= lr * dW2
b2 -= lr * db2
W1 -= lr * dW1
b1 -= lr * db1
```

### 4.3 数值示例

```python
# 前向
state = [0.1, 0.2, -0.05, 0.3]
h = ReLU(W1 @ state) = [0.0, 0.32, 0.15, ..., 0.08]
logits = [1.2, 0.8]
prob = [0.60, 0.40]  # 60% 选左，40% 选右

# 假设选了 action=1 (右)，且 G_t = +50
dlogits = [0.60, -0.60]
dlogits *= -50 = [-30, 30]

# 解读: 增大左推的概率，减小右推的概率
# (因为右推效果好，G_t 为正)
```

---

## 5. 代码实现

### 5.1 折扣回报计算

```python
def compute_returns(rewards, gamma):
    """计算折扣回报 G_t"""
    G = np.zeros(len(rewards))
    G[-1] = rewards[-1]
    for t in reversed(range(len(rewards)-1)):
        G[t] = rewards[t] + gamma * G[t+1]
    return G
```

### 5.2 策略梯度更新

```python
def policy_gradient_update(network, state, action, advantage, lr):
    """纯 numpy 实现策略梯度更新"""
    # 前向传播
    prob = network.forward(state)

    # 计算梯度
    dlogits = prob.copy()
    dlogits[action] -= 1.0
    dlogits *= -advantage

    # 反向传播
    dW2 = np.outer(dlogits, network.h)
    db2 = dlogits
    dh = network.W2.T @ dlogits
    dh[network.h <= 0] = 0
    dW1 = np.outer(dh, state)
    db1 = dh

    # 更新
    network.W2 -= lr * dW2
    network.b2 -= lr * db2
    network.W1 -= lr * dW1
    network.b1 -= lr * db1
```

### 5.3 训练循环

```python
for episode in range(N_EPISODES):
    # 采集数据
    states, actions, rewards = [], [], []
    obs, _ = env.reset()

    for step in range(MAX_STEPS):
        action = network.sample_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        states.append(obs)
        actions.append(action)
        rewards.append(reward)
        obs = next_obs
        if terminated or truncated:
            break

    # 计算回报
    returns = compute_returns(rewards, GAMMA)

    # 策略梯度更新
    for s, a, G in zip(states, actions, returns):
        policy_gradient_update(network, s, a, G, lr)
```

---

## 6. 方差 Reduction

### 6.1 问题

直接使用 G_t 作为权重会导致**高方差**:
- 不同 episode 回报差异大
- 梯度震荡剧烈

### 6.2 Baseline 技巧

减去一个 baseline V(s)，不改变期望但减小方差:

```python
# 无 baseline
∇J = E[∇log π(a|s) × G_t]

# 有 baseline
∇J = E[∇log π(a|s) × (G_t - V(s))]
```

### 6.3 简化: 减均值

```python
# 减均值
G_t = G_t - np.mean(G_all)

# 或标准化
G_t = (G_t - np.mean(G_all)) / (np.std(G_all) + 1e-8)
```

**效果**: 梯度方差减小，训练更稳定

---

## 7. 面试考点

### Q1: 策略梯度 vs Q-Learning 的核心区别?

| 方面 | 策略梯度 | Q-Learning |
|------|----------|------------|
| 表示 | 神经网络 π(a\|s) | Q-table |
| 动作 | 连续/离散都可以 | 离散 |
| 策略 | 随机 | 确定 |
| 梯度 | 高方差 | 低方差 |
| 收敛性 | 不保证 | 有限 MDP 保证 |

### Q2: 为什么用 log π?

**log-trick 推导**:

```
∇E[R] = ∇ Σ_τ P(τ)R(τ)
       = Σ_τ ∇P(τ) × R(τ)

用恒等式 ∇P = P × ∇log P:
       = Σ_τ P(τ) × ∇log P(τ) × R(τ)
       = E[∇log P(τ) × R(τ)]

又 log P(τ) = Σ_t log π(a_t|s_t)
所以 ∇log P(τ) = Σ_t ∇log π(a_t|s_t)
```

**数值稳定性**: log 函数将乘法转化为加法，数值更稳定

### Q3: REINFORCE 的方差问题?

**原因**: 使用整条轨迹的回报 G_t，噪声大

**解决方案**:
1. 减 baseline V(s)
2. 使用 Actor-Critic (用 V(s) 估计代替 G_t)
3. 标准化回报

### Q4: 为什么要用 discount factor γ?

- 未来不确定性大，应该打折
- 数学上保证收敛
- 平衡近期/远期奖励

### Q5: On-policy vs Off-policy?

**REINFORCE 是 On-policy**:
- 数据采集策略 = 当前正在优化的策略
- 用完就丢，无法复用历史数据
- 样本效率低

---

## 附录: NumPy 实现完整代码框架

```python
class PolicyNetwork:
    def __init__(self, state_dim=4, hidden_dim=16, action_dim=2):
        self.W1 = np.random.randn(hidden_dim, state_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(action_dim, hidden_dim) * 0.1
        self.b2 = np.zeros(action_dim)

    def forward(self, state):
        self.h = np.maximum(0, self.W1 @ state + self.b1)  # ReLU
        logits = self.W2 @ self.h + self.b2
        self.prob = np.exp(logits) / np.sum(np.exp(logits))  # softmax
        return self.prob

    def sample_action(self, state):
        prob = self.forward(state)
        return np.random.choice(len(prob), p=prob)
```
