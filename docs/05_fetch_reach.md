# Step 5: FetchReach-like PyBullet 环境

## 概述

本 step 使用 PyBullet 模拟 7-DoF UR5 机械臂，复刻 FetchReach 标准任务设计。

**核心设计：**
- 观测空间：Goal-Aware 字典结构 `{observation, achieved_goal, desired_goal}`
- 动作空间：末端执行器 Cartesian 位移控制 `(dx, dy, dz)`
- 奖励模式：Sparse + Dense 双模式可切换
- 物理引擎：PyBullet（免费开源，对应工业标准 MuJoCo）

## 文件结构

```
task2/src/step5_fetch_reach.py   # PyBullet UR5 FetchReach 实现
task2/docs/05_fetch_reach.md     # 本文档
```

## 观测空间设计（Goal-Aware）

FetchReach 的核心设计是 Goal-Aware 观测空间，这是 Hindsight Experience Replay (HER) 的基础：

```python
observation = {
    'observation': np.array([gripper_pos(3), gripper_vel(3), joint_angles(6)]),  # 15维
    'achieved_goal': np.array([x, y, z]),  # 末端实际位置
    'desired_goal': np.array([x, y, z]),   # 目标位置
}
```

**为什么需要 Goal-Aware 观测？**
1. 支持 HER：失败的经验也可以学习到"我实际到了哪里"
2. 支持 Goal-Conditioned RL：同一个策略可以到达不同目标
3. 与真实机器人任务接口一致（工业标准）

## 动作空间

```python
# 末端执行器 Cartesian 位移控制
action_space = Box(-0.05, 0.05, (3,), dtype=np.float32)  # dx, dy, dz (m)
```

**为什么用末端位移而不是关节角？**
- 更容易解释：动作直接对应"往哪个方向走"
- 更容易泛化：不同构型的机械臂可以用相同策略
- 更接近真实任务：人类操作员更容易给出末端位移指令

## 奖励函数

### Dense 模式（推荐，更快收敛）

```python
def compute_reward(dense=True):
    if dense:
        reward = -np.linalg.norm(achieved_goal - desired_goal)  # -dist
    else:
        # Sparse 模式：成功=0，失败=-1
        reward = 0.0 if dist < 0.05 else -1.0
    return reward
```

### Sparse 模式（接近真实工业任务）

```python
reward = 0.0 if dist < 0.05 else -1.0
```

**为什么需要双模式？**
- Dense 奖励：梯度信号强，收敛快，但容易产生"reward hacking"
- Sparse 奖励：更接近真实任务（只有成功/失败），但训练困难，需要 HER 等技巧

## PyBullet UR5 机械臂

```python
import pybullet as p
import pybullet_data

# 加载 UR5 + 夹爪
robot_id = p.loadURDF("ur5_robot/ur5.urdf")
```

**PyBullet vs MuJoCo：**
| 方案 | 优点 | 缺点 |
|------|------|------|
| PyBullet | 免费开源，SB3 官方推荐 | 精度略低 |
| MuJoCo | 精度高，工业标准 | 商业许可 |

## 训练对比

### Dense vs Sparse 奖励

| 指标 | Dense | Sparse |
|------|-------|--------|
| 收敛速度 | 快 | 慢 |
| 最终成功率 | ~90% | ~70% |
| sim2real 迁移 | 一般 | 更好 |

## 回调复用

直接复用 Step4 的三个回调（SuccessRateCallback, CheckpointCallback, CurriculumCallback）。

## 运行

```bash
# 演示环境 API
python step5_fetch_reach.py

# Dense 奖励训练
python step5_fetch_reach.py  # 选择 1

# Sparse 奖励训练
python step5_fetch_reach.py  # 选择 2
```

## 下一步

- **Step5B**：官方 FetchReach 完美复刻（gymnasium-robotics）
- **Step6**：多算法基准框架（SAC vs TD3 vs DDPG vs PPO）
