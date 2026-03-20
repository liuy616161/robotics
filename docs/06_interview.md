# 面试核心知识点

## Part 1: 强化学习基础

### Q1: 强化学习 vs 监督学习

| 方面 | 监督学习 | 强化学习 |
|------|----------|----------|
| 数据 | i.i.d. 样本 | 序列交互 |
| 反馈 | 即时正确标签 | 延迟奖励 |
| 探索 | 无需 | 需要 |
| 目标 | 最小化损失 | 最大化累积奖励 |

### Q2: MDP 四元组

```
MDP = (S, A, P, R, γ)

S: 状态空间
A: 动作空间
P: 转移概率 P(s'|s, a)
R: 奖励函数 R(s, a, s')
γ: 折扣因子
```

### Q3: Value Function vs Q Function

```python
V(s) = E[Σ γ^t r_t | s_0 = s]
# 状态 s 的价值 = 从 s 开始的期望折扣回报

Q(s, a) = E[Σ γ^t r_t | s_0 = s, a_0 = a]
# 状态动作对的价值 = 从 s 执行 a 开始的期望折扣回报

关系: V(s) = max_a Q(s, a)
```

### Q4: Bellman 方程

```python
Q(s, a) = r + γ × max_a' Q(s', a')

# V(s) 版本:
V(s) = max_a [r + γ × V(s')]
```

### Q5: 折扣因子 γ 的作用

```
γ = 0.99:
  - 第 100 步的奖励只值现在的 0.99^100 ≈ 0.37
  - 重视远期奖励

γ = 0.95:
  - 第 100 步只值 0.0059
  - 更重视近期奖励
```

---

## Part 2: Q-Learning 与 DQN

### Q6: Q-Learning 更新规则

```python
Q(s, a) ← Q(s, a) + α × [r + γ × max_a' Q(s', a') - Q(s, a)]
#         旧估计  学习率  TD目标 - 旧估计
```

### Q7: DQN 两大创新

```
1. Experience Replay:
   - 存储 (s, a, r, s') 到 Buffer
   - 随机抽样，打破时序相关性

2. Target Network:
   - 固定目标网络 Q_target
   - 避免训练陷入"移动目标"
```

### Q8: DQN 过高估计问题

```python
# max Q(s', a') 会放大噪声
Q(s,a) = r + γ × max Q(s', a')
         = r + γ × (max Q_true(s', a') + noise)
         ≥ r + γ × max Q_true(s', a')

# 过高估计会传播，导致策略偏差
```

---

## Part 3: 策略梯度

### Q9: 策略梯度定理

```python
∇J(θ) = E_τ[Σ_t ∇log π_θ(a_t|s_t) × G_t]
```

### Q10: REINFORCE 算法流程

```
1. 采集 trajectory: (s0, a0, r0, s1, a1, r1, ...)
2. 计算 G_t = Σ γ^k r_{t+k}
3. 策略梯度更新:
   θ ← θ + α × ∇log π(a_t|s_t) × G_t
```

### Q11: Policy Gradient 方差大怎么办

```
1. 减 Baseline: V(s) 或 baseline
2. Actor-Critic: 用 V(s) 估计代替 G_t
3. 标准化: (G - mean) / std
```

---

## Part 4: PPO

### Q12: PPO Clip 机制

```python
ratio = π_new(a|s) / π_old(a|s)

# Clipped surrogate objective
L = min(
    ratio × A,                    # 未 clip
    clip(ratio, 1-ε, 1+ε) × A   # clip 后
)
```

### Q13: PPO 为什么稳定

```
1. Clip 限制更新幅度
2. 目标函数是 min，防止大更新
3. 适合大规模并行
```

---

## Part 5: SAC

### Q14: SAC 最大熵框架

```python
J(π) = E[Σ (r_t + α × H(π(·|s_t)))]

H(π) = -E[log π(a|s)]
```

### Q15: SAC 自动熵调节

```python
# 目标: H(π) ≈ target_entropy
# 自动调整 α 使熵达到目标值
```

### Q16: 双 Q 网络取 min

```python
Q_target = min(Q1(s', a'), Q2(s', a'))
# 抑制过高估计，稳定训练
```

---

## Part 6: 实际工程

### Q17: 稀疏奖励解决

```
1. Reward Shaping: r = -dist + 10×success
2. HER: 重标记失败轨迹的目标
3. Curriculum: 从简单任务开始
4. Demo: 专家演示初始化
```

### Q18: Sim2Real 挑战

```
1. 动力学差异
2. 传感器噪声
3. 视觉变化

解决:
1. Domain Randomization
2. 系统辨识
3. 残差控制
```

### Q19: 调试 RL 的 checklist

```
1. 奖励是否合理范围?
2. terminated / truncated 逻辑?
3. observation_space dtype = float32?
4. reset() 返回 (obs, info)?
5. step() 返回 5 个值?
6. entropy 是否下降?
7. TD error 是否收敛?
```

### Q20: 常见报错

```python
# 1. dtype 错误
AssertionError: reward must be float
→ return float(reward)

# 2. reset 返回值错误
TypeError: cannot unpack non-iterable NoneType
→ return obs, info  # 必须返回二元组

# 3. 观测越界
check_env: observation not in observation_space
→ 检查 _get_obs() 返回值
→ 确保 .astype(np.float32)
```

---

## 代码速记

### Gymnasium 接口

```python
# 5 个返回值!
obs, reward, terminated, truncated, info = env.step(action)

# reset 返回二元组
obs, info = env.reset()
```

### SB3 SAC

```python
model = SAC("MlpPolicy", env,
    buffer_size=50000,
    learning_rate=3e-4,
    batch_size=256,
    gamma=0.99,
    ent_coef='auto'
)
model.learn(total_timesteps=100000)
```
