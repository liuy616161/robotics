# 算法进化史与对比

## RL 算法演进脉络

```
1989 Q-Learning (Watkins)
    │
    ├── 优点: 简单、有收敛保证
    ├── 局限: 只能离散状态/动作，维度爆炸
    │
    ▼
2013 DQN (DeepMind, Playing Atari)
    │
    ├── 创新: 用神经网络近似 Q 函数
    ├── 技术: Replay Buffer + Target Network
    ├── 局限: 只能离散动作
    │
    ▼
2015 DDPG (DeepMind, Continuous Control)
    │
    ├── 创新: Actor-Critic 架构，支持连续动作
    ├── Actor: 输出确定性动作
    ├── 局限: 训练不稳定，容易崩溃
    │
    ▼
2017 PPO (OpenAI)
    │
    ├── 创新: Clip ratio 限制更新幅度
    ├── 优势: 极其稳定，成为 RL 标准
    ├── 局限: On-policy，样本效率低
    │
    ▼
2018 TD3 (Twin Delayed DDPG)
    │
    ├── 创新: 双 Q 网络 + 延迟更新
    ├── 修复: DDPG 的过高估计和不稳定
    ├── 局限: 探索机制简单
    │
    ▼
2018 SAC (Haarnoja et al., Berkeley)
    │
    ├── 创新: 最大熵框架 + 自动温度调节
    ├── 优势: 连续控制首选
    ├── 现状: 宇树、智元等公司广泛使用
    │
    ▼
2024+ VLA + RL Fine-tuning
    ├── RL 用于微调大模型
    └── Diffusion Policy + RL
```

---

## 算法对比表

| 算法 | 年份 | 采样 | 动作空间 | 探索方式 | 稳定性 | 样本效率 | 代表应用 |
|------|------|------|----------|----------|--------|----------|---------|
| Q-Learning | 1989 | Off | 离散 | ε-greedy | 高 | 高 | 表格游戏 |
| DQN | 2013 | Off | 离散 | ε-greedy | 中 | 高 | Atari 游戏 |
| DDPG | 2015 | Off | 连续 | 高斯噪声 | 低 | 高 | 简单控制 |
| PPO | 2017 | On | 离散+连续 | Clip | 非常高 | 低 | 游戏、通用 |
| TD3 | 2018 | Off | 连续 | 高斯噪声 | 高 | 高 | 机器人 |
| SAC | 2018 | Off | 连续 | 最大熵 | 高 | 高 | 机器人控制 |

---

## 算法选择指南

### 按场景

| 场景 | 推荐算法 | 原因 |
|------|----------|------|
| 离散动作游戏 | DQN / PPO | 离散空间天然适合 |
| 机器人精细操作 | SAC + HER | 样本效率高，自动探索 |
| 人形/四足步态 | PPO + IsaacGym | 大规模并行，稳定性高 |
| 稀疏奖励抓取 | SAC + HER | Off-policy + 目标重标记 |
| Sim2Real 迁移 | SAC + Domain Randomization | 最大熵鲁棒 |

### 按硬件

| 硬件 | 推荐 | 原因 |
|------|------|------|
| 单机 CPU | SAC | Off-policy，样本效率 |
| 单机 GPU | SAC / PPO | 都可以 |
| IsaacGym (4096 并行) | PPO | On-policy 在并行下效率高 |

---

## 为什么机器人控制首选 SAC

### 1. 样本效率

```
真实机器人:
  - 采集速度: ~50ms/步
  - 1分钟 = 1200 步
  - 1小时 = 72,000 步

SAC (Off-policy):
  - 72,000 步可用多次

PPO (On-policy):
  - 72,000 步只能用一次
```

### 2. 连续动作

```
机器人关节力矩: -2.5 ~ 2.5 N·m (连续)
PPO/SAC 都可以处理，但 SAC 更自然
```

### 3. 自动探索

```
PPO: 需要手动调 ε 或添加噪声
SAC: 最大熵框架自动平衡探索利用
```

---

## 面试八股文

### Q1: On-policy vs Off-policy

```
On-policy (PPO, REINFORCE):
  数据采集策略 = 当前正在优化的策略
  数据用完即丢
  优点: 稳定
  缺点: 样本效率低

Off-policy (SAC, DQN, TD3):
  数据采集策略 ≠ 优化策略
  Replay Buffer 存储历史数据
  优点: 样本效率高
  缺点: 需要重要性采样校正（或绕过）
```

### Q2: 稀疏奖励问题

```
问题: 真实机器人任务奖励稀疏（到达才给 +1）

方案:
1. Reward Shaping: 加中间奖励 r = -dist
2. HER: 失败轨迹重标记目标
3. Curriculum: 从简单→困难
4. Demo: 专家演示预训练
```

### Q3: Sim2Real Gap

```
动力学 Gap: 摩擦、弹性、延迟
传感器 Gap: 噪声、漂移
视觉 Gap: 光照、纹理

解决:
1. Domain Randomization (域随机化)
2. System Identification (系统辨识)
3. Sim2Real Fine-tuning
4. 残差 RL
```

### Q4: 为什么 SAC 是机器人控制首选

```
1. 样本效率高 (Off-policy)
2. 最大熵自动探索
3. 连续动作空间自然
4. 实现相对简单
5. 已被工业界验证
```

### Q5: 最大熵 RL 的意义

```
标准 RL: max E[Σ r_t]
最大熵: max E[Σ r_t + α×H(π)]

物理意义:
- 鼓励策略保持随机性
- 自动探索不同行为模式
- 对扰动更鲁棒
```

### Q6: DQN 家族 vs Policy Gradient 家族

```
DQN 家族 (Value-Based):
  - Q-Learning → DQN → TD3
  - 输出 Q 值，动作用 argmax
  - 适合离散动作

Policy Gradient 家族:
  - REINFORCE → PPO → SAC
  - 直接输出策略
  - 适合连续动作
```
