# Step 4: 自定义 Gymnasium 环境 — 机械臂末端到达任务

## 任务目标

把 3-DOF 机械臂的"末端到达目标"包装成标准 Gymnasium 接口，然后用 SAC 训练。

## 环境设计

### 核心参数（与 FetchReach 对齐）

| 参数 | 值 | 说明 |
|------|-----|------|
| MAX_EPISODE_STEPS | 50 | 每轮最多 50 步 |
| GOAL_TOLERANCE | 0.05m | 成功阈值 5cm |
| 动作空间 | Box(-0.3, 0.3, (3,)) | 关节角增量 [Δq1, Δq2, Δq3]，单位弧度 |
| 观测空间 | Box(-2, 2, (6,)) | 相对位置(3) + 关节角(3) |

### 运动学模型

3-DOF 平面机械臂，正运动学公式：

```
theta_cumsum[i] = q[0] + q[1] + ... + q[i]
x = Σ L_i × cos(theta_cumsum[i])
y = Σ L_i × sin(theta_cumsum[i])
```

连杆长度：[0.4, 0.35, 0.25] m

### 观测空间设计

使用**相对位置观测**（更易学，有平移不变性）：

```
obs = [dx, dy, dz, q1, q2, q3]
     = [目标_x - 末端_x, 目标_y - 末端_y, 目标_z - 末端_z, 关节角1, 关节角2, 关节角3]
```

### 奖励函数

```python
def _compute_reward(self, ee_pos, target_pos, dist, action):
    # 1. 势函数塑形：距离减少则奖励
    k_shaping = 2.0
    delta_dist = (prev_dist - dist) / dist
    shaping_reward = k_shaping * delta_dist

    # 2. 距离惩罚
    dist_penalty = -1.0 * dist

    # 3. 动态动作惩罚：距离越远惩罚越小
    if dist > 0.5:
        action_penalty = 0.0
    elif dist > 0.2:
        action_penalty = -0.001 * np.sum(action**2)
    elif dist > 0.1:
        action_penalty = -0.005 * np.sum(action**2)
    else:
        action_penalty = -0.01 * np.sum(action**2)

    # 4. 成功奖励
    success_bonus = 100.0 if dist < GOAL_TOLERANCE else 0.0

    reward = shaping_reward + dist_penalty + action_penalty + success_bonus
```

设计理由：
- **势函数塑形**：解决近距离梯度弱的问题
- **动态动作惩罚**：远距离鼓励大步探索，近距离鼓励精细控制
- **稀疏成功奖励**：明确的任务完成信号

## 课程学习配置

| Level | 步数范围 | 目标距离范围 |
|-------|----------|--------------|
| 0 | 0~20K | 0.1~0.2m |
| 3 | 60K | 0.2~0.4m |
| 5 | 100K | 0.3~0.6m |

总计 100K 步，每级 20K 步。

## 与 FetchReach 设计对比

| 对比项 | Step4 | FetchReach |
|--------|-------|------------|
| Episode 长度 | 50 步 | 50 步 |
| 动作空间 | 关节角增量 (3D) | 末端位移 (3D) |
| 观测空间 | 相对位置+关节角 | Goal-Aware 字典 |
| 物理引擎 | 无（纯 FK） | MuJoCo |
| 成功阈值 | 5cm | 5cm |
| 训练步数 | 100K | 100K |

## 常见问题

### 为什么用相对位置而不是绝对位置？

相对位置具有平移不变性，Agent 学习的是"目标在哪里"而不是"目标在哪里绝对坐标"，泛化能力更强。

### 为什么用关节角增量而不是末端位移？

纯数学 FK 模型无法直接做逆运动学（多解问题）。用关节角增量控制是简化方案，便于教学实现。

### 课程学习为什么从近距离开始？

近目标简单，Agent 快速学到"靠近目标=奖励"的基本概念。远目标需要更大的动作幅度，后面的课程自然引入。

## 面试考点

1. **Gymnasium 接口规范**：reset 返回 obs+info，step 返回 obs+reward+terminated+truncated+info
2. **terminated vs truncated**：terminated 是任务结束，truncated 是人为截断
3. **奖励函数设计**：势函数塑形解决稀疏奖励，动作惩罚防止抖动
4. **课程学习**：从简单到难，渐进增加任务难度
