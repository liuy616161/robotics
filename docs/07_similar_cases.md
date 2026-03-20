# 相似实践案例对比研究

## 目录

1. [Gymnasium-Robotics FetchReach](#1-gymnasium-robotics-fetchreach)
2. [panda-gym PandaReach](#2-panda-gym-pandareach)
3. [Stable-Baselines3 RL Reach](#3-stable-baselines3-rl-reach)
4. [PyBullet Industrial Robotics Gym](#4-pybullet-industrial-robotics-gym)
5. [PyBullet UR5 Pick-and-Place](#5-pybullet-ur5-pick-and-place)
6. [关键设计模式总结](#6-关键设计模式总结)

---

## 1. Gymnasium-Robotics FetchReach

### 概述

FetchReach 是 Farama Foundation 官方维护的机械臂到达任务环境，基于 7-DoF Fetch 移动机械臂，是机器人 RL 研究的事实标准基准。

### 环境配置

| 参数 | 值 |
|------|-----|
| **机器人** | Fetch Mobile Manipulator (7-DoF) |
| **控制频率** | 25 Hz (每步 20 个仿真步，dt=0.002s) |
| **最大 episode 步数** | 50 |
| **目标容差** | 0.05 m |

### 观测空间

```python
# Goal-aware 字典结构
observation: Box(10,)      # [gripper_pos(3), joint_displacement(2), velocities(5)]
desired_goal: Box(3,)      # 目标位置 [x, y, z]
achieved_goal: Box(3,)     # 当前末端位置 [x, y, z]
```

### 动作空间

```python
Box(-1.0, 1.0, (4,), float32)
# Actions 0-2: Cartesian 位移 (dx, dy, dz)
# Action 3: 夹爪开合（到达任务中未使用）
```

### 奖励函数

| 类型 | 公式 |
|------|------|
| **Sparse** | r = 0 if dist < 0.05m else -1 |
| **Dense** | r = -‖achieved_goal - desired_goal‖ |

### 起点与目标采样

- 起点固定：`[1.3419, 0.7491, 0.555]` m
- 目标：在起点基础上随机偏移 `[-0.15, 0.15]` m

### 我们的差距

| 对比项 | FetchReach | 我们的 Step4 |
|--------|-----------|--------------|
| 机械臂 | 7-DoF 真实机器人 | 3-DoF 简化模型 |
| 仿真器 | MuJoCo 物理引擎 | 纯数学 FK |
| 控制频率 | 25 Hz | 无限制 |
| 观测 | 10维 + goal-aware | 6维相对位置 |
| 奖励 | Sparse/Dense 可选 | 势函数塑形 + 多分量 |

---

## 2. panda-gym PandaReach

### 概述

panda-gym 是基于 PyBullet 的 Panda 机械臂（7-DoF Franka Emika）环境库，提供与 FetchReach 类似但开源的任务。

### 任务变体

| 任务 | 描述 | 奖励类型 |
|------|------|----------|
| **PandaReach** | 末端到达目标 | Sparse / Dense |
| **PandaPush** | 推动方块到目标 | Sparse / Dense |
| **PandaPickAndPlace** | 抓取并放置 | Sparse / Dense |

### 动作空间

```python
# 默认：末端执行器位移控制
# 3维连续向量 (dx, dy, dz)
# 也有关节空间控制变体
```

### 奖励函数设计

| 类型 | 设计 |
|------|------|
| **Sparse** | 仅在任务完成时返回正奖励 |
| **Dense** | 距离越近奖励越高 |

### 我们的参考价值

- **密集奖励设计**：`dense = -dist` 或 `-dist - 0.01*action_norm`
- **稀疏+密集混合**：先用密集引导探索，到达后给稀疏成功奖励
- 这与我们的势函数塑形思路一致

---

## 3. Stable-Baselines3 RL Reach

### 概述

RL Reach 是 SB3 官方项目，专门用于可复现的机械臂到达任务研究。

### 论文

> A.umjaud et al., "RL Reach: A Reproducible Robotic Reaching Environment", 2021

### 特点

- **可复现性**：提供标准化基准
- **可定制化**：可修改任务参数
- **自包含**：开箱即用的工具箱

### GitHub

```
https://github.com/PierreExeter/rl_reach
```

### 我们的参考价值

- 课程学习+评估的最佳实践流程
- 多算法对比基准设计

---

## 4. PyBullet Industrial Robotics Gym

### 概述

基于 PyBullet 的工业机械臂训练框架，支持 DDPG、SAC、TD3 等算法。

### 支持的机器人

- KUKA IIWA
- UR5
- Panda
- 双臂机器人

### 训练命令

```bash
# DDPG 训练
python train_ddpg.py

# SAC 训练
python train_sac.py
```

### 特色功能

- **Hindsight Experience Replay (HER)** 支持
- 多种机器人模型可选
- 工业应用场景

### 我们的参考价值

- HER 在稀疏奖励任务中的应用
- 多算法对比框架

---

## 5. PyBullet UR5 Pick-and-Place

### 概述

自定义 UR5 + Robotiq 2F-85 夹爪的抓取放置任务环境。

### 环境组成

| 组件 | 型号 |
|------|------|
| 机械臂 | UR5 (6-DoF) |
| 夹爪 | Robotiq 2F-85 |
| 仿真器 | PyBullet |
| 框架 | Gymnasium + SB3 |

### 观测空间设计

```python
# 典型设计
observation = [
    joint_positions,      # 6维
    joint_velocities,     # 6维
    gripper_state,        # 夹爪状态
    object_position,      # 物体位置
    target_position       # 目标位置
]
```

### 动作空间

```python
# 关节空间控制或末端控制
action = [d_joint1, d_joint2, ..., d_gripper]  # 7维
```

### 奖励函数

```python
# 抓取放置的典型奖励设计
reward = -dist_to_object * 10          # 靠近物体
       + grasp_success * 50            # 成功抓取
       - dist_to_target * 5            # 靠近目标
       + place_success * 100           # 成功放置
```

### 与我们的对比

| 方面 | UR5 Pick-Place | 我们的 Step4 |
|------|----------------|--------------|
| 任务复杂度 | 抓取+放置（多阶段） | 到达（单阶段） |
| 动作维度 | 7维 | 3维 |
| 奖励设计 | 多分量稀疏奖励 | 势函数塑形密集奖励 |
| 仿真器 | PyBullet 物理 | 纯数学 |

---

## 6. 关键设计模式总结

### 6.1 观测空间设计模式

#### 绝对位置 vs 相对位置 vs Goal-Aware

| 模式 | 格式 | 适用场景 |
|------|------|----------|
| **绝对位置** | [ee_pos, target_pos, joints] | 简单任务 |
| **相对位置** | [target-ee_pos, joints] | 我们的方案 |
| **Goal-Aware** | {obs, achieved_goal, desired_goal} | 标准基准 |

#### 我们的选择理由

```python
# 相对位置优势
relative_pos = target_pos - ee_pos  # 具有平移不变性
# → 无论目标在哪，相同相对位置需要相同动作
# → 泛化能力强，输入维度低
```

### 6.2 奖励函数设计模式

#### 稀疏 vs 密集 vs 混合

| 模式 | 公式 | 优缺点 |
|------|------|--------|
| **Sparse** | r = 0 if success else -1 | 信号清晰但探索困难 |
| **Dense** | r = -dist | 信号丰富但可能走捷径 |
| **混合** | r = -dist + shaping + success | **我们采用** |

#### 势函数塑形（Potential-Based Shaping）

```python
# 核心思想：每步奖励 = k * Δpotential
# 不改变最优策略，但加速学习

shaping_reward = k * (prev_dist - dist)  # 靠近就奖
```

**学术背景**：Nguyen et al. (2019) 证明势函数塑形在稀疏奖励机器人任务中显著加速收敛。

### 6.3 课程学习设计模式

#### 难度渐进方式

| 方式 | 实现 | 适用场景 |
|------|------|----------|
| **目标距离渐进** | radius_min/max 逐渐增大 | 我们的方案 |
| **障碍物渐进** | 逐渐添加障碍 | 导航/操作 |
| **自由度渐进** | 逐渐解锁关节 | 多指灵巧操作 |
| **负重渐进** | 逐渐增加负载 | 运动控制 |

#### 节奏控制

```python
# 太快 → 灾难性遗忘
# 太慢 → 训练效率低

steps_per_level = 200000  # 我们的选择
# 经验法则：确保每个难度下有 50+ 成功回合
```

### 6.4 终止条件设计

| 策略 | terminated | truncated | 用途 |
|------|-----------|-----------|------|
| **到达终止** | dist < tolerance | step >= max | 成功回合 |
| **超时截断** | False | True | 失败回合 |
| **两者皆可** | 可同时为 True | | Gymnasium 规范 |

### 6.5 断点恢复与检查点

#### 检查点保存策略

```python
# ❌ 错误：只在训练结束时保存
model.save("final_model")

# ✅ 正确：定期保存 + resume 支持
class CheckpointCallback:
    def __init__(self, save_freq):
        self._last_save_at = 0  # 相对偏移

    def _on_step(self):
        if self.num_timesteps - self._last_save_at >= self.save_freq:
            self.model.save(f"checkpoint_{self.num_timesteps}")
            self._last_save_at = self.num_timesteps
```

#### 为什么用相对偏移？

```python
# resume 后 num_timesteps 从断点继续
# 绝对取模 % 在 resume 后失效
# 相对偏移始终正确
```

---

## 参考文献

1. FetchReach - Gymnasium-Robotics Documentation
   https://robotics.farama.org/envs/fetch/reach/

2. panda-gym Documentation
   https://panda-gym.readthedocs.io/

3. RL Reach: A Reproducible Robotic Reaching Environment
   https://github.com/PierreExeter/rl_reach

4. PyBullet Industrial Robotics Gym
   https://github.com/rparak/PyBullet_Industrial_Robotics_Gym

5. UR5 Reinforcement Learning Grasp Object
   https://github.com/leesweqq/ur5_reinforcement_learning_grasp_object

6. Stable-Baselines3 Custom Env Guide
   https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html

7. SAC Tutorial - Antonin Raffelin
   https://araffin.github.io/post/sac-massive-sim/

8. Reward Shaping and Curriculum Learning - Anca et al.
   https://arxiv.org/abs/2206.02462

9. Sample Efficient Robot Training on PyBullet with SAC
   https://towardsdatascience.com/sample-efficient-robot-training-on-pybullet-simulation-with-sac-algorithm-71d5d1d4587f/

10. HLRobot: Deep Reinforcement Learning for Robotic Manipulation
    https://arxiv.org/html/2505.03356v1
