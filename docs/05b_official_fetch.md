# Step 5B: 官方 FetchReach 完美复刻

## 概述

使用 `gymnasium-robotics` 官方库完整复刻 FetchReach，作为 Step5 的完美样例对照。

**对比：**

| 对比项 | Step5 (自实现) | Step5B (官方) |
|--------|---------------|---------------|
| 物理引擎 | PyBullet UR5 | MuJoCo (Fetch) |
| 观测空间 | 简化版 | 完整 Goal-Aware |
| 奖励函数 | 自定义 | 官方 Dense |
| 动作空间 | 末端位移 | 末端位移+夹爪 |
| 成功率 | 评估对比 | 基准 |

## 环境

```python
import gymnasium_robotics
from gymnasium import make

# Sparse 奖励
env = make("FetchReach-v2")

# Dense 奖励（推荐）
env = make("FetchReachDense-v2")
```

## 观测空间结构

```python
{
    'observation': array of shape (15,)  # 末端位置 + 速度 + 关节角度
    'achieved_goal': array of shape (3,)  # 实际到达位置
    'desired_goal': array of shape (3,)   # 目标位置
}
```

## 训练

```bash
# Sparse 奖励
python step5b_official_fetch.py  # 选择 1

# Dense 奖励
python step5b_official_fetch.py  # 选择 2

# 两种都跑
python step5b_official_fetch.py  # 选择 3
```

## 对比验证

训练后会自动生成对比曲线：

```
assets/step5b_custom_vs_official.png   # Step5 vs Step5B 对比
assets/step5b_official_comparison.png  # Dense vs Sparse 对比
```

## 关键设计差异

### 1. 物理引擎

- **官方**：MuJoCo，Fetch 机器人模型，更精确的动力学
- **自实现**：PyBullet，UR5 模型，简化但足够教学用

### 2. 观测空间

官方环境的 observation 包含更多状态：
- 末端位置 (3)
- 末端线速度 (3)
- 末端角速度 (3)
- 关节角度 (6)
- 总计：15 维

### 3. 奖励函数

官方 Dense 奖励：
```python
reward = -np.linalg.norm(achieved_goal - desired_goal)
```

与 Step5 一致，但精度更高（浮点误差更小）。

## 下一步

**Step6**：多算法基准框架 — 用 Step5B 的环境测试 SAC, TD3, DDPG, PPO
