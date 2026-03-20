# Step 3: SAC（Soft Actor-Critic）

## 目录

1. [算法概述](#1-算法概述)
2. [Off-Policy vs On-Policy](#2-off-policy-vs-on-policy)
3. [最大熵框架](#3-最大熵框架)
4. [网络结构](#4-网络结构)
5. [超参数详解](#5-超参数详解)
6. [Pendulum-v1 环境](#6-pendulum-v1-环境)
7. [面试考点](#7-面试考点)

---

## 1. 算法概述

### 1.1 什么是 SAC?

SAC (Soft Actor-Critic) 由 Haarnoja 等人于 2018 年提出，是目前**连续控制领域最广泛使用**的强化学习算法。

### 1.2 三大核心特性

| 特性 | 说明 | 优势 |
|------|------|------|
| Off-Policy | 使用 Replay Buffer | 样本效率高 |
| 最大熵 | 策略包含熵项 | 自动探索 |
| 双 Q 网络 | 两个 Critic 取 min | 稳定训练 |

### 1.3 SAC vs 其他算法

```
Q-Learning → DQN → DDPG → TD3 → SAC
             ↓
         只能离散   连续动作  解决不稳定
```

---

## 2. Off-Policy vs On-Policy

### 2.1 核心区别

| 特性 | On-Policy (PPO) | Off-Policy (SAC) |
|------|-----------------|------------------|
| 数据来源 | 当前策略采集 | 历史数据复用 |
| Replay Buffer | 无 | 有 |
| 样本效率 | 低（用完就丢） | 高（反复利用） |
| 实现复杂度 | 低 | 高 |

### 2.2 Replay Buffer

```python
class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=256):
        batch = random.sample(self.buffer, batch_size)
        return zip(*batch)
```

### 2.3 比喻

```
On-policy = 自助餐，吃完这轮就清盘
Off-policy = 火锅，菜可以反复涮
```

---

## 3. 最大熵框架

### 3.1 标准 RL vs 最大熵 RL

```
标准 RL:      max E[Σ r_t]
最大熵 RL:    max E[Σ r_t + α × H(π)]
```

### 3.2 熵 H(π) 的含义

```python
H(π) = -E[log π(a|s)]

# 高斯策略的例子
π(a|s) = N(μ, σ²)
H(π) = 0.5 × ln(2πe × σ²)

σ 大 → H 大 → 策略更随机 → 更多探索
σ 小 → H 小 → 策略更确定 → 更多利用
```

### 3.3 温度系数 α

```python
# α 控制探索-利用的平衡
α 大 → 熵更重要 → 更随机（多探索）
α 小 → 奖励更重要 → 更确定（多利用）

# SAC 的 'auto' 模式
α 自动调整，无需手动设置
```

### 3.4 为什么最大熵有用?

1. **避免局部最优**: 高熵鼓励尝试不同动作
2. **自动探索**: 不需要手动调 ε
3. **鲁棒性**: 策略更平滑，泛化能力更强

---

## 4. 网络结构

### 4.1 SAC 的 5 个网络

```
Actor (策略网络):
  输入: 状态 s
  输出: 动作分布 π(a|s) = N(μ, σ)

Critic 1 (Q 网络):
  输入: (s, a)
  输出: Q1(s, a)

Critic 2 (Q 网络):
  输入: (s, a)
  输出: Q2(s, a)

Target Critic 1 (目标网络):
  缓慢更新的 Q1 副本

Target Critic 2 (目标网络):
  缓慢更新的 Q2 副本
```

### 4.2 双 Q 网络的作用

**问题**: Q-learning 有"过高估计"倾向

```
Q(s,a) = r + γ × max Q(s',a')
         ↓
由于 max 操作，会放大噪声，导致 Q 值偏高
```

**解决**: 取两个 Q 值的最小值

```python
Q_target = min(Q1(s',a'), Q2(s',a'))
```

### 4.3 目标网络更新

```python
# 软更新（比硬更新更稳定）
target_W ← 0.005 × online_W + 0.995 × target_W

# 每步只更新 0.5%，确保目标相对稳定
```

---

## 5. 超参数详解

### 5.1 核心超参数

| 参数 | 推荐值 | 作用 |
|------|--------|------|
| buffer_size | 50K+ | Replay Buffer 容量 |
| learning_rate | 3e-4 | 优化器学习率 |
| batch_size | 256 | 每次采样数 |
| gamma | 0.99 | 折扣因子 |
| tau | 0.005 | 目标网络更新率 |
| ent_coef | 'auto' | 自动熵系数 |

### 5.2 learning_rate

```python
# 太大
lr = 1e-2 → 训练震荡，可能不收敛

# 太小
lr = 1e-5 → 收敛极慢

# 推荐
lr = 3e-4  # 对大多数任务都有效
```

### 5.3 batch_size

```python
# 太小
batch = 32 → 梯度方差大，不稳定

# 太大
batch = 2048 → 计算慢，可能错过稀有样本

# 推荐
batch = 256  # 平衡稳定性和效率
```

### 5.4 tau (目标网络更新)

```python
# 硬更新 (每 N 步完全复制)
if step % N == 0:
    target_W = online_W

# 软更新 (推荐)
target_W = tau × online_W + (1-tau) × target_W
# tau = 0.005 表示每步更新 0.5%
```

### 5.5 learning_starts

```python
# 前 1000 步纯随机，填充 Replay Buffer
# 避免早期用未充分训练的网络采集的数据
learning_starts = 1000
```

---

## 6. Pendulum-v1 环境

### 6.1 环境描述

```
         │
         │ θ ← 杆，需要保持竖直 (θ=0)
        ╱│
       ╱ │
      ╱  │
     ●   │  ← 摆杆（下端固定）
    ╱    │
   ●     │  ← 可以施加力矩
```

### 6.2 空间定义

| 属性 | 定义 |
|------|------|
| **观测 (3维)** | cos(θ), sin(θ), θ̇ |
| **动作 (1维)** | 力矩 u ∈ [-2, 2] |
| **奖励** | -(θ² + 0.1×θ̇² + 0.001×u²) |

### 6.3 为什么用 cos(θ) 和 sin(θ)?

**问题**: 直接给角度 θ，在 ±180° 处不连续

```
θ = 179° 和 θ = -179° 实际很接近
但数值上 179 - (-179) = 358°
```

**解决**: 用 cos(θ) 和 sin(θ) 表示，两者结合可唯一确定角度，且连续

### 6.4 奖励分析

```python
r = -(θ² + 0.1×θ̇² + 0.001×u²)

# 各项含义
θ²:      主要目标，让杆竖直 (θ=0)
0.1×θ̇²:  抑制角速度，鼓励稳定
0.001×u²: 惩罚大动作，节省能量
```

| 状态 | 奖励 |
|------|------|
| 完美竖直 (θ=0, θ̇=0, u=0) | 0 |
| 略微倾斜 (θ=0.1) | -0.01 |
| 倒下 (θ=π, 最大) | -9.87 |
| 极端情况 (θ=π, θ̇=8, u=2) | -16.27 |

### 6.5 训练目标

```
随机策略:    平均奖励 ≈ -1200 (混沌摆动)
SAC 训练后:  平均奖励 ≈ -200 (能稳定竖直)
```

---

## 7. 面试考点

### Q1: SAC vs PPO 的核心区别?

| 方面 | SAC | PPO |
|------|-----|-----|
| 采样 | Off-policy | On-policy |
| 探索 | 最大熵 (自动) | Clip ratio |
| 样本效率 | 高 | 低 |
| 收敛速度 | 快 | 慢但稳定 |
| 适用场景 | 真实机器人 (数据珍贵) | 仿真大规模并行 |

### Q2: 为什么真实机器人常用 SAC?

1. **数据珍贵**: 真实机器人采集数据慢 (1步/50ms)，不能浪费
2. **Off-policy 效率**: 数据可以反复利用
3. **连续动作**: 关节力矩是连续值
4. **自动探索**: 不需要手动调 ε

### Q3: 最大熵 RL 的物理意义?

```python
J(π) = E[Σ r_t + α × H(π)]

# 相当于同时优化两个目标:
# 1. 最大化奖励
# 2. 保持策略高熵 (多探索)

# 机器人控制中的意义:
# - 避免死板的动作序列
# - 对扰动更鲁棒
# - 遇到新情况能灵活应对
```

### Q4: 双 Q 网络为什么取 min?

```python
Q1, Q2 = 两个 Q 值
Q_target = min(Q1, Q2)  # 抑制过高估计

# 数学解释:
# E[min(Q1, Q2)] ≤ E[max(Q1, Q2)]
# 取 min 更保守，避免过高估计导致的训练不稳定
```

### Q5: 目标网络的作用?

1. **提供稳定目标**: 每步目标只变化 0.5%
2. **打破相关性**: 避免训练陷入"移动目标"问题
3. **数学保证**: 软更新的极限是贝尔曼最优

### Q6: auto 熵系数怎么调?

```python
# SAC 的自动熵调整
α 在训练中自动优化，目标是使熵保持在某个目标值

# 目标熵通常设为:
target_entropy = -action_dim  # 连续动作空间

# 例如: action_dim=1 → target_entropy ≈ -1
```

---

## 附录: SB3 SAC 使用模板

```python
from stable_baselines3 import SAC

model = SAC(
    "MlpPolicy",
    env,
    buffer_size=50000,
    learning_rate=3e-4,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    ent_coef='auto',
    learning_starts=1000,
    verbose=1
)

model.learn(total_timesteps=100000)
model.save("sac_pendulum")
```
