# 经典坦克大战 - 分层 MADDPG 协同 AI

复刻 FC/NES 经典坦克大战 (Battle City)，基于分层强化学习实现多坦克协同对战。

- 底层技能 (PPO + 课程学习): 导航 / 攻击 / 防守，3 阶段递进训练
- 高层协调 (MADDPG + 注意力 Critic + 通讯模块): GPU 加速，可提取协作权重
- 实时可视化: 协作连线 / 注意力热力图 / 技能标签

---

## 目录

1. [环境安装](#环境安装)
2. [快速开始](#快速开始)
3. [完整训练流程](#完整训练流程)
4. [测试与评估](#测试与评估)
5. [运行模式一览](#运行模式一览)
6. [协作可视化](#协作可视化)
7. [架构设计](#架构设计)
8. [项目结构](#项目结构)
9. [技术规格](#技术规格)
10. [常见问题](#常见问题)

---

## 环境安装

```bash
conda activate youtu-agent
pip install gymnasium stable-baselines3 pygame torch numpy tqdm omegaconf
```

验证安装：

```bash
cd /home/a4201/owncode/tank
python -c "
from envs.game_engine import BattleCityEngine
from envs.single_tank_env import SingleTankSkillEnv
from envs.multi_tank_env import MultiTankTeamEnv
from maddpg.core import MaddpgTrainer, AttentionCritic
print('所有模块导入成功')
"
```

---

## 快速开始

```bash
cd /home/a4201/owncode/tank

# 人类玩家模式 (方向键移动, 空格射击, ESC 退出)
python main.py --mode play

# 使用规则技能看 AI 对战 (无需预训练)
python main.py --mode visualize --use_rule_skills --num_episodes 5

# 协同可视化对战 (看协作连线 + 注意力矩阵)
python main.py --mode visualize_coop --use_rule_skills --num_episodes 5
```

---

## 完整训练流程

### 阶段 1: PPO 课程学习训练底层技能

每个技能经过 3 个课程阶段 (C1→C2→C3) 逐步增难，自动迁移学习。

```bash
cd /home/a4201/owncode/tank

# 推荐: 3 技能并行在 3 个 GPU 上训练 (约 60 分钟)
PYTHONUNBUFFERED=1 nohup conda run -n youtu-agent \
    python scripts/run_ppo_curriculum.py \
    > logs/ppo_curriculum.log 2>&1 &
echo "PID: $!"

# 查看训练进度
tail -f logs/ppo_curriculum.log
```

或单独训练某个技能：

```bash
# 单个技能课程学习 (在指定 GPU 上)
PYTHONUNBUFFERED=1 conda run -n youtu-agent \
    python -m skills.train_skills --skill attack --device cuda:0

PYTHONUNBUFFERED=1 conda run -n youtu-agent \
    python -m skills.train_skills --skill defend --device cuda:1

PYTHONUNBUFFERED=1 conda run -n youtu-agent \
    python -m skills.train_skills --skill navigate --device cuda:2
```

课程阶段设计：

| 技能 | C1 | C2 | C3 | 总步数 |
|------|---------|---------|---------|--------|
| attack | 1v1, 100万步, 门槛70% | 1v2, 100万步, 门槛40% | 1v3, 200万步, 门槛20% | 400万 |
| defend | 1v1, 100万步, 门槛80% | 1v2, 100万步, 门槛50% | 1v3, 200万步, 门槛25% | 400万 |
| navigate | 0敌, 50万步, 门槛90% | 1敌, 50万步, 门槛50% | 3敌, 100万步, 门槛30% | 200万 |

训练完成后模型保存在 `skills/models/`：

```
skills/models/attack_skill.zip         # 最终攻击技能
skills/models/attack_C1_1v1.zip        # 课程阶段 1 检查点
skills/models/attack_C2_1v2.zip        # 课程阶段 2 检查点
skills/models/attack_C3_1v3.zip        # 课程阶段 3 检查点
skills/models/attack_vecnorm.pkl       # VecNormalize 统计量 (必须!)
skills/models/defend_skill.zip
skills/models/navigate_skill.zip
...
```

### 阶段 1.5: 交叉验证 PPO 技能

在多智能体环境中验证 PPO 技能的实际效果：

```bash
PYTHONUNBUFFERED=1 conda run -n youtu-agent \
    python scripts/cross_validate_ppo.py
```

### 阶段 2: MADDPG 渐进协同训练

基于训练好的 PPO 技能，渐进提高难度训练高层协调器。

```bash
cd /home/a4201/owncode/tank

# 三阶段渐进训练 (easy→medium→hard, 共 18000 episodes)
PYTHONUNBUFFERED=1 nohup conda run -n youtu-agent \
    python scripts/run_maddpg_progressive.py --device cuda:0 \
    > logs/maddpg_progressive.log 2>&1 &
echo "PID: $!"

# 查看训练进度
tail -f logs/maddpg_progressive.log
```

或单独训练某个难度：

```bash
# Easy 难度 (2v2, 3000 episodes)
PYTHONUNBUFFERED=1 conda run -n youtu-agent \
    python -m maddpg.train_coord \
    --episodes 3000 --difficulty easy \
    --skill_interval 4 --device cuda:0

# Medium 难度 (2v3, 5000 episodes)
PYTHONUNBUFFERED=1 conda run -n youtu-agent \
    python -m maddpg.train_coord \
    --episodes 5000 --difficulty medium \
    --skill_interval 4 --device cuda:0

# Hard 难度 (2v4, 10000 episodes)
PYTHONUNBUFFERED=1 conda run -n youtu-agent \
    python -m maddpg.train_coord \
    --episodes 10000 --difficulty hard \
    --skill_interval 4 --device cuda:0
```

渐进训练目标：

| 阶段 | 难度 | 对阵 | 回合数 | 目标胜率 |
|------|------|------|--------|----------|
| Stage 1 | easy | 2v2 | 3000 | > 30% |
| Stage 2 | medium | 2v3 | 5000 | > 15% |
| Stage 3 | hard | 2v4 | 10000 | > 5% |

模型保存在 `models/`：

```
models/maddpg_best/        # 最佳模型 (自动更新)
models/maddpg_ep500/       # 检查点 (每 500 回合)
models/maddpg_ep1000/
```

### 阶段 3: 评估与可视化

```bash
# 加载训练好的模型, 协同可视化对战
python main.py --mode visualize_coop --num_episodes 10

# 普通可视化对战
python main.py --mode visualize --num_episodes 10

# 如果没有训练好的 PPO 模型, 加 --use_rule_skills
python main.py --mode visualize_coop --use_rule_skills --num_episodes 10
```

---

## 测试与评估

### 测试 MADDPG 模型胜率

```bash
cd /home/a4201/owncode/tank

# 测试 ep1000 检查点在所有难度下的胜率
PYTHONUNBUFFERED=1 conda run -n youtu-agent \
    python scripts/test_maddpg_model.py \
    --model_dir models/maddpg_ep1000 --difficulty all --n_eval 100

# 测试最佳模型
PYTHONUNBUFFERED=1 conda run -n youtu-agent \
    python scripts/test_maddpg_model.py \
    --model_dir models/maddpg_best --difficulty easy --n_eval 200

# 使用规则技能测试 (对比基线)
PYTHONUNBUFFERED=1 conda run -n youtu-agent \
    python scripts/test_maddpg_model.py \
    --model_dir models/maddpg_best --use_rule_skills --difficulty all
```

### 验证 PPO 底层技能

```bash
# 交叉验证 (attack+attack 2v2, attack+defend 2v2, attack+defend 2v3)
PYTHONUNBUFFERED=1 conda run -n youtu-agent \
    python scripts/cross_validate_ppo.py
```

---

## 运行模式一览

| 模式 | 说明 | 计算资源 |
|------|------|----------|
| `play` | 人类玩家控制 (方向键 + 空格) | - |
| `train_skills` | PPO 课程学习训练底层技能 (串行) | CPU 多核 |
| `train_skills_parallel` | 3 技能多进程并行训练 | CPU 多核 |
| `train_skills_vis` | 可视化训练底层技能 | CPU 多核 |
| `train_coord` | MADDPG 训练高层协同 | GPU (自动) |
| `train_coord_vis` | 可视化训练协同 (含协作连线) | GPU (自动) |
| `train_all` | 一键全流程 (技能并行 + 协同) | CPU + GPU |
| `demo` | 规则技能快速演示 | GPU (自动) |
| `visualize` | Pygame 可视化对战 | GPU (自动) |
| `visualize_coop` | 协同可视化 (协作网络 + 技能标签) | GPU (自动) |

---

## 协作可视化

`visualize_coop` 模式和 `train_coord_vis` 模式会实时显示以下元素：

| 元素 | 位置 | 含义 |
|------|------|------|
| 黄色粗线 | 坦克间 | 强协作 (注意力 > 0.6) |
| 绿色线 | 坦克间 | 中等协作 (0.3 ~ 0.6) |
| 蓝色细线 | 坦克间 | 弱协作 (< 0.3) |
| 连线数值 | 线中点 | 具体协作权重 (0.0 ~ 1.0) |
| 注意力热力图 | 右下角 | n x n 矩阵，深色 = 高关注 |
| 技能标签 | 坦克上方 | 蓝 = 导航 / 红 = 攻击 / 绿 = 防守 |
| 坦克编号 | 坦克中心 | R0/R1 (红方) / B0/B1... (蓝方) |
| 信息面板 | 底部 | 回合 / 步数 / 奖励 / 存活统计 |

### 如何解读

- 两坦克间有粗黄线 → 紧密协调（例如一个攻击一个防守互补）
- 热力图 R0→R1 值高 → R0 的决策依赖 R1 的状态
- 协作权重随时间变化 → 根据战场形势动态调整协同策略

---

## 架构设计

```
┌─────────────────────────────────────────────────────────┐
│          高层协调器 (MADDPG + 注意力 Critic)             │
│                                                         │
│  输入: 各坦克 32 维观测                                  │
│  输出: 技能 ID (0=导航/1=攻击/2=防守) + 2D 目标参数      │
│                                                         │
│  ParamActor       → Gumbel-Softmax 技能选择              │
│  AttentionCritic  → 多头注意力, 可提取协作权重            │
│  CommModule       → 智能体间消息传递 (obs→obs+16)        │
│                                                         │
│  每 4 步决策一次 (skill_interval=4)                      │
└───────────────────┬─────────────────────────────────────┘
                    ▼
┌─────────────────────────────────────────────────────────┐
│          底层技能库 (PPO 课程学习 / 规则 AI)              │
│                                                         │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐        │
│  │ 导航 (PPO) │  │ 攻击 (PPO) │  │ 防守 (PPO) │        │
│  │ 29维→动作  │  │ 29维→动作  │  │ 29维→动作  │        │
│  └────────────┘  └────────────┘  └────────────┘        │
│  + VecNormalize 观测归一化 (训练/推理必须一致)            │
│                                                         │
│  输出: Discrete(6) → NOOP/UP/DOWN/LEFT/RIGHT/FIRE      │
└───────────────────┬─────────────────────────────────────┘
                    ▼
┌─────────────────────────────────────────────────────────┐
│          经典坦克大战引擎 (game_engine.py)                │
│                                                         │
│  20x20 网格 | 砖墙/钢墙/水面/冰面/树林/基地              │
│  红方 (RL) 2 辆 vs 蓝方 (规则 AI) 2~4 辆               │
│  保护基地 + 消灭敌人 = 胜利                              │
└─────────────────────────────────────────────────────────┘
```

---

## 项目结构

```
tank/
├── main.py                     # 一键入口 (10 种运行模式)
├── config/
│   └── config.yaml             # OmegaConf 配置文件
├── scripts/                    # 独立运行脚本
│   ├── run_ppo_curriculum.py   # PPO 课程学习并行训练
│   ├── run_maddpg_progressive.py # MADDPG 渐进训练
│   ├── cross_validate_ppo.py   # PPO 交叉验证
│   └── test_maddpg_model.py    # MADDPG 模型测试
│
├── envs/                       # 游戏环境
│   ├── game_engine.py          # 坦克大战引擎 (地图/坦克/子弹/碰撞/AI)
│   ├── single_tank_env.py      # 单坦克 Gymnasium 环境 (obs=29 维)
│   └── multi_tank_env.py       # 多智能体环境 (obs=32 维, 2v2~2v4)
│
├── skills/                     # 底层技能
│   ├── train_skills.py         # PPO 课程学习训练 (C1→C2→C3 + VecNormalize)
│   ├── skill_wrapper.py        # PPO/规则技能统一封装 (含 VecNormalize 加载)
│   └── models/                 # 保存的 PPO 模型 (.zip + _vecnorm.pkl)
│
├── maddpg/                     # 高层协调
│   ├── core.py                 # ParamActor / AttentionCritic / CommModule /
│   │                           # MaddpgBuffer / MaddpgTrainer
│   └── train_coord.py          # MADDPG 训练主循环 (多环境收集)
│
├── utils/                      # 工具
│   ├── config.py               # OmegaConf 配置加载
│   ├── visualize.py            # Pygame 渲染器 (协作连线/热力图/标签)
│   └── train_monitor.py        # 训练可视化监控器
│
├── models/                     # 保存的 MADDPG 模型
│   ├── maddpg_best/            # 最佳模型
│   ├── maddpg_ep500/           # 检查点
│   └── maddpg_ep1000/
│
├── logs/                       # 训练日志 + TensorBoard
└── MADDPG/                     # 参考: 原始 MADDPG 工程
```

---

## 技术规格

### 观测空间

单坦克 29 维：自身坐标 (2) + 方向 one-hot (4) + 血量 (1) + 冷却 (1) + 子弹 (1) + 墙壁射线 (4) + 敌人射线 (4) + 最近敌人 (6) + 基地 (3) + 存活比 (1) + 目标参数 (2)

多坦克 32 维：单坦克前 23 维 + 友军信息 (dx, dy, hp, alive) 4 维 + 基地 (3) + 统计 (2)

### 动作空间

Discrete(6): NOOP / UP / DOWN / LEFT / RIGHT / FIRE

### 网络结构

| 网络 | 输入 | 隐藏层 | 输出 |
|------|------|--------|------|
| PPO Actor/Critic | 29 维 | [256, 256] ReLU | 6 logits / 1 V值 |
| ParamActor | 48 维 (32+16 comm) | [256, 256] ReLU | 3 技能概率 + 3x2 参数 |
| AttentionCritic | n x (32+act) | 128 dim, 2 heads | 1 Q值 + 注意力权重 |
| CommModule | 32 → 16 维消息 | [64] ReLU | 增强 obs (32+16=48) |

### 超参数

| 参数 | PPO 底层 (课程学习) | MADDPG 高层 |
|------|---------|------------|
| 学习率 | C1: 3e-4 → C3: 1e-4 | Actor 1e-4 / Critic 3e-4 |
| 折扣 γ | 0.99 | 0.8145 (0.95^4) |
| 批量大小 | 256 | 512 |
| 并行环境 | 32 (SubprocVecEnv) | 32 (MultiEnvCollector) |
| 目标网络 | - | τ=0.01 (Polyak) |
| Gumbel 温度 | - | 1.0 → 0.3 (decay=0.99999) |
| 梯度裁剪 | - | 1.0 |
| 熵系数 | C1: 0.01 → C3: 0.001 | 0.01 |
| 网络结构 | [256, 256] | [256, 256] |
| 梯度更新/步 | - | 8 |

### 训练成果 (4x A100 80GB)

**PPO 课程学习:**

| 技能 | C1 胜率 | C2 胜率 | C3 胜率 | 耗时 |
|------|---------|---------|---------|------|
| attack | 89% (1v1) | 41% (1v2) | 20% (1v3) | ~35 分钟 |
| defend | 81% (1v1) | 25% (1v2) | 25% (1v3) | ~35 分钟 |
| navigate | 96% (1敌) | 94% (3敌) | - | ~20 分钟 |

**PPO 交叉验证 (多智能体环境):**

| 组合 | 难度 | 胜率 |
|------|------|------|
| attack+attack | easy (2v2) | **84%** |
| attack+defend | easy (2v2) | **81%** |
| attack+defend | medium (2v3) | **66%** |

---

## 常见问题

### Pygame 初始化失败 (X Error / libGL error)

conda 环境中 `libstdc++` 版本与系统 Mesa 驱动冲突。通过 `main.py` 入口运行会自动修复：

```bash
python main.py --mode play
```

如果直接调用子模块，手动设置：

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python main.py --mode play
```

### tensorboard 未安装

训练会自动跳过 tensorboard 日志，不影响训练。如需查看曲线：

```bash
pip install tensorboard
tensorboard --logdir logs/
```

### 防止终端关闭中断训练

所有长时间训练命令都建议使用 `nohup ... &` 方式运行：

```bash
PYTHONUNBUFFERED=1 nohup conda run -n youtu-agent python xxx.py > logs/xxx.log 2>&1 &
tail -f logs/xxx.log  # 查看进度
```

### 增加难度

```bash
python main.py --mode play --n_blue 8 --map_name classic_2
```
