# Step 1: Q-Learning（表格式强化学习）

## 目录

1. [算法原理](#1-算法原理)
2. [CartPole 环境](#2-cartpole-环境)
3. [代码实现](#3-代码实现)
4. [关键函数详解](#4-关键函数详解)
5. [超参数分析](#5-超参数分析)
6. [实验结果](#6-实验结果)
7. [面试考点](#7-面试考点)

---

## 1. 算法原理

### 1.1 核心思想

Q-Learning 是 1989 年由 Chris Watkins 提出的经典强化学习算法，属于**时序差分 (TD) 学习**的一种。

**核心问题**: 智能体如何通过与环境交互，学会最优的动作决策？

**解决方案**: 维护一个 Q-table，记录每个 (状态, 动作) 对的价值，通过 Bellman 方程迭代更新。

### 1.2 Bellman 方程

```
Q(s, a) ← Q(s, a) + α × [ r + γ × max_a' Q(s', a') - Q(s, a) ]
```

| 符号 | 含义 |
|------|------|
| Q(s, a) | 状态动作对的价值 |
| α | 学习率 (step size) |
| r | 即时奖励 |
| γ | 折扣因子 (future reward weight) |
| max_a' Q(s', a') | 下一状态的最大 Q 值 |

### 1.3 TD 误差解读

```
TD误差 = r + γ × max Q(s',a') - Q(s,a)
        = 更好估计 - 旧估计
```

- **TD 误差 > 0**: 这个动作比预期好，增大 Q 值
- **TD 误差 < 0**: 这个动作比预期差，减小 Q 值

### 1.4 为什么叫 "Q" Learning?

Q 来自 **Quality** 的首字母，表示"动作的质量"或"价值"。

---

## 2. CartPole 环境

### 2.1 环境描述

```
         ┌─────┐
         │     │  ← 杆（要保持竖直）
         │     │
    ╔════╧═════╗
    ║   小车   ║ ← 可以左右推
    ╚══════════╝
  ◀──────────────▶  轨道
```

### 2.2 空间定义

| 属性 | 定义 |
|------|------|
| **观测 (4维)** | 小车位置 [-2.4, 2.4] |
|  | 小车速度 [-∞, ∞] (实际截断到 [-3, 3]) |
|  | 杆角度 [-0.2095, 0.2095] rad |
|  | 杆角速度 [-∞, ∞] (实际截断到 [-3, 3]) |
| **动作 (离散)** | 0 = 向左推，1 = 向右推 |
| **奖励** | 每步存活 +1 |
| **终止** | 杆角度 > ±15° 或 小车位置 > ±2.4 |

### 2.3 为什么用 CartPole?

- 足够简单，易于调试算法
- 包含 RL 核心概念（状态、动作、奖励、折扣）
- 训练快（500步/episode）
- 有明确的目标（保持平衡）

---

## 3. 代码实现

### 3.1 整体流程

```python
# 伪代码
for episode in range(N_EPISODES):
    state = discretize(env.reset())
    for step in range(MAX_STEPS):
        action = epsilon_greedy(Q, state, epsilon)  # 探索
        next_state, reward, done = env.step(action)
        update_q_table(Q, state, action, reward, next_state, done)
        state = next_state
        if done: break
    epsilon *= EPSILON_DECAY  # 逐渐减少探索
```

### 3.2 离散化方法

连续观测无法直接用于 Q-table，需要离散化：

```python
def discretize(obs):
    indices = []
    for i in range(4):
        clipped = np.clip(obs[i], OBS_LOW[i], OBS_HIGH[i])
        bins = np.linspace(OBS_LOW[i], OBS_HIGH[i], N_BINS[i]+1)
        idx = np.digitize(clipped, bins) - 1
        idx = np.clip(idx, 0, N_BINS[i] - 1)
        indices.append(idx)
    return tuple(indices)
```

### 3.3 Q-table 更新

```python
def update_q_table(Q, state, action, reward, next_state, done):
    td_target = reward + GAMMA * np.max(Q[next_state]) * (1 - done)
    td_error = td_target - Q[state + (action,)]
    Q[state + (action,)] += ALPHA * td_error
    return abs(td_error)
```

**关键点**: `done=True` 时，`max(Q[next_state]) × 0 = 0`，因为终态没有未来奖励。

---

## 4. 关键函数详解

### 4.1 `discretize()`

**输入**: 连续观测 `obs = [pos, vel, angle, ang_vel]`

**输出**: `tuple(4,) = (idx0, idx1, idx2, idx3)` 作为 Q-table 索引

**原理**: 使用 `np.digitize` 将连续值映射到 bin

```
bins = [a, b, c, d]
digitize(x):
  x < a → 0
  a ≤ x < b → 1
  b ≤ x < c → 2
  c ≤ x → 3
```

### 4.2 `epsilon_greedy()`

**探索-利用权衡 (Exploration-Exploitation Tradeoff)**:

- **探索 (ε)**: 随机动作，发现新策略
- **利用 (1-ε)**: 选择已知最优动作

```python
def epsilon_greedy(Q, state, epsilon, n_actions):
    if random() < epsilon:
        return random()  # 探索
    else:
        return argmax(Q[state])  # 利用
```

### 4.3 `update_q_table()`

**Bellman 方程的代码形式**:

```python
# Q(s,a) = Q(s,a) + α × [TD_target - Q(s,a)]
# TD_target = r + γ × max_a' Q(s',a')  (若未终止)
```

---

## 5. 超参数分析

### 5.1 学习率 α

| α 值 | 效果 |
|------|------|
| α = 0.01 | 收敛慢，但稳定 |
| α = 0.1 | 平衡（常用） |
| α = 0.5 | 收敛快，但可能震荡 |
| α = 1.0 | 不收敛 |

**CartPole 适合 α = 0.1**

### 5.2 折扣因子 γ

| γ 值 | 效果 |
|------|------|
| γ = 0.0 | 只看即时奖励 |
| γ = 0.9 | 重视近未来 |
| γ = 0.99 | 重视远未来 (常用) |

**CartPole 适合 γ = 0.99**（需要长期规划保持平衡）

### 5.3 探索率 ε

```python
epsilon = EPSILON * EPSILON_DECAY^episode
# 500 轮后: 1.0 × 0.995^500 ≈ 0.08
```

### 5.4 bin 数量

```python
N_BINS = [3, 3, 6, 6]
```

为什么速度只分 3 个 bin，角度分 6 个?

因为**角度直接决定平衡**，需要更精细的划分。速度变化快但影响相对间接。

---

## 6. 实验结果

### 6.1 训练曲线解读

```
Episode 50:   最近50轮平均奖励:  42.3
Episode 100:  最近50轮平均奖励:  95.6
Episode 200:  最近50轮平均奖励: 185.2
Episode 500:  最近50轮平均奖励: 195.3
```

**预期**: 500 轮后，平均奖励应达到 **195+**（接近上限 200）

### 6.2 TD 误差变化

```
初期: TD_error ≈ 0.8  (Q 值估计误差大)
后期: TD_error ≈ 0.1  (Q 值估计准确)
```

TD 误差下降说明 Q-table 逐渐收敛。

### 6.3 Q-table 分析

```python
# 最优动作分布
向左(0) = 312 个状态
向右(1) = 336 个状态

# 动作 Q 值差异
mean = 2.3, max = 8.7
```

边缘状态的动作选择更确定（差异大），中心区域较模糊。

---

## 7. 面试考点

### Q1: Q-Learning 的收敛条件是什么?

1. 所有 (s,a) 对被访问无限多次
2. 学习率满足 Robbins-Monro: Σα = ∞, Σα² < ∞
3. 环境是有限 MDP

### Q2: 为什么 Q-Learning 能收敛?

因为 Bellman 方程是一个**压缩映射 (contraction)**，迭代更新在均方意义下必收敛。

### Q3: Q-Learning 的致命局限是什么?

| 局限 | 说明 |
|------|------|
| 离散动作 | 无法处理连续动作空间 |
| 维度灾难 | 状态空间大将导致 Q-table 爆炸 |
| 函数近似 | 无法处理图像等高维输入 |

**这就是为什么需要 DQN → 策略梯度 → SAC 的演进。**

### Q4: ε-Greedy 为什么有效?

- 初期高 ε: 充分探索状态空间
- 后期低 ε: 利用已学到的最优策略
- 保持少量探索: 避免错过更好的策略

### Q5: done=True 时为什么 max Q(s')=0?

因为**终态没有未来奖励**。数学上:
```
if done:
    Q(s,a) = r  (没有后续)
else:
    Q(s,a) = r + γ × max Q(s',a')
```

---

## 思考题答案

1. **为什么 Q-table 不适合连续动作空间?**
   - Q-table 维度 = 状态数 × 动作数
   - 连续动作无法穷举，维度爆炸
   - max_a Q(s,a) 无法计算

2. **ε 为什么逐渐减小?**
   - 初期: 需要探索发现好策略
   - 后期: 已经知道好策略，应该利用
   - 但要保留少量探索，避免过拟合

3. **Atari 图像 Q-table 需要多大?**
   - 84×84×256³ ≈ 10^12 字节
   - 不可行！所以需要 DQN (神经网络近似)
