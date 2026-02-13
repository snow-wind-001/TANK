# Multi-Agent Deep Deterministic Policy Gradient (MADDPG) - PyTorch版本

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Python-3.8+-3776ab?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Status-Active-success.svg" alt="Status">
</p>

## 🙏 致谢

本项目基于 OpenAI 的 [MADDPG](https://github.com/openai/maddpg) 原始实现进行 PyTorch 迁移和功能扩展。

原始论文：[Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/pdf/1706.02275.pdf)

原始作者：Ryan Lowe, Yi Wu, Aviv Tamar, Jean Harb, Pieter Abbeel, Igor Mordatch

感谢 OpenAI 团队开源这一经典的多智能体强化学习算法！

---

## 👨‍🔬 当前维护者

**沈阳理工大学 装备工程学院 深度学习实验室**

📧 联系邮箱：77232623@qq.com

---

## 📖 项目简介

MADDPG (Multi-Agent Deep Deterministic Policy Gradient) 是一种用于混合合作-竞争环境的多智能体强化学习算法。本项目将原始的 TensorFlow 1.x 实现完整迁移到 **PyTorch 2.0+**，并进行了多项功能增强。

### ✨ 主要特性

- 🔥 **PyTorch 2.0+ 实现**：现代化的深度学习框架，支持 GPU 加速
- 🎮 **可视化训练工具**：基于 PyQt5 的图形化训练界面，实时观察智能体行为
- 📊 **完整的训练监控**：学习曲线、奖励追踪、模型保存/加载
- 🧪 **9种环境场景**：从单智能体导航到复杂的多智能体协作/对抗
- 📝 **详细的中文文档**：场景说明、参数配置、训练指南

---

## 🏗️ 网络架构

### Actor 网络
```
输入(观测) → Linear(obs_dim, 64) → ReLU → Linear(64, 64) → ReLU → Linear(64, act_dim) → 分布采样
```

### Critic 网络
```
输入(所有观测+动作) → Linear(total_dim, 64) → ReLU → Linear(64, 64) → ReLU → Linear(64, 1) → Q值
```

### 概率分布类
| 分布类型 | 动作空间类型 | 说明 |
|---------|-------------|------|
| `SoftCategoricalPd` | Discrete | 软分类分布（离散动作） |
| `SoftMultiCategoricalPd` | MultiDiscrete | 软多分类分布（多维离散） |
| `DiagGaussianPd` | Box | 对角高斯分布（连续动作） |
| `BernoulliPd` | MultiBinary | 伯努利分布（二元动作） |

---

## 🎯 支持的场景

| 场景名称 | 类型 | 智能体数 | 难度 | 核心技能 |
|---------|------|---------|------|---------|
| `simple` | 单智能体 | 1 | ⭐ | 导航 |
| `simple_spread` | 协作 | 3 | ⭐⭐ | 覆盖、避碰 |
| `simple_adversary` | 竞争 | 3 | ⭐⭐⭐ | 欺骗、追踪 |
| `simple_tag` | 竞争 | 4 | ⭐⭐⭐ | 追逐、逃跑 |
| `simple_push` | 竞争 | 2 | ⭐⭐ | 物理对抗 |
| `simple_reference` | 协作 | 2 | ⭐⭐⭐ | 通信、协调 |
| `simple_speaker_listener` | 协作 | 2 | ⭐⭐⭐ | 语言学习 |
| `simple_crypto` | 混合 | 3 | ⭐⭐⭐⭐ | 加密通信 |
| `simple_world_comm` | 混合 | 6 | ⭐⭐⭐⭐ | 团队协作 |

详细场景说明请参阅 [PYTORCH_MIGRATION_README.md](./PYTORCH_MIGRATION_README.md)

---

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- NumPy < 2.0 (gym 兼容性)
- gym
- PyQt5 (可视化工具)
- matplotlib

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/snow-wind-001/MADDPG.git
cd MADDPG

# 2. 创建并激活 conda 环境
conda create -n maddpg python=3.8
conda activate maddpg

# 3. 安装依赖
pip install torch torchvision
pip install "numpy<2.0"
pip install gym
pip install pyqt5 matplotlib

# 4. 安装 multiagent 环境
cd multiagent-particle-envs
pip install -e .
cd ..

# 5. 安装 maddpg 包
pip install -e .

# 6. 设置环境变量
export PYTHONPATH=$(pwd)/multiagent-particle-envs:$PYTHONPATH
export SUPPRESS_MA_PROMPT=1
```

---

## 📋 训练命令

### 基本训练

```bash
# 单智能体简单场景
python experiments/torch_train.py --scenario simple --num-episodes 3000

# 多智能体协作场景
python experiments/torch_train.py --scenario simple_spread --num-episodes 25000

# 对抗场景
python experiments/torch_train.py --scenario simple_adversary --num-adversaries 1 --num-episodes 30000

# 自定义参数训练
python experiments/torch_train.py \
    --scenario simple_tag \
    --num-episodes 50000 \
    --max-episode-len 50 \
    --lr 1e-2 \
    --batch-size 1024 \
    --num-units 64 \
    --save-rate 1000 \
    --exp-name my_experiment
```

### 可视化训练

```bash
# 启动 PyQt5 可视化训练工具
python experiments/qt_visualize_train.py
```

### 模型评估

```bash
# 加载模型并展示
python experiments/torch_train.py --scenario simple_spread --display --load-dir ./saved_models/
```

---

## ⚙️ 命令行参数

### 环境参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--scenario` | 环境场景名称 | `"simple"` |
| `--max-episode-len` | 每回合最大步数 | `25` |
| `--num-episodes` | 训练总回合数 | `60000` |
| `--num-adversaries` | 对抗智能体数量 | `0` |
| `--good-policy` | 友方策略算法 | `"maddpg"` |
| `--adv-policy` | 敌方策略算法 | `"maddpg"` |

### 训练参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--lr` | 学习率 | `1e-2` |
| `--gamma` | 折扣因子 | `0.95` |
| `--batch-size` | 批次大小 | `1024` |
| `--num-units` | 隐藏层单元数 | `64` |

### 保存/加载参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--exp-name` | 实验名称 | `None` |
| `--save-dir` | 模型保存目录 | `"/tmp/policy/"` |
| `--save-rate` | 保存间隔（回合数） | `1000` |
| `--load-dir` | 模型加载目录 | `""` |
| `--plots-dir` | 学习曲线保存目录 | `"./learning_curves/"` |

### 评估参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--restore` | 从检查点恢复训练 | `False` |
| `--display` | 显示训练好的策略 | `False` |
| `--benchmark` | 运行基准测试 | `False` |

---

## 📁 项目结构

```
MADDPG/
├── maddpg/
│   ├── __init__.py
│   ├── common/
│   │   ├── torch_util.py          # PyTorch 工具函数
│   │   ├── torch_distributions.py # 概率分布实现
│   │   ├── distributions.py       # 原 TF 分布（保留）
│   │   └── tf_util.py             # 原 TF 工具（保留）
│   └── trainer/
│       ├── torch_maddpg.py        # PyTorch MADDPG 核心
│       ├── maddpg.py              # 原 TF 版本（保留）
│       └── replay_buffer.py       # 经验回放池
├── experiments/
│   ├── torch_train.py             # PyTorch 训练脚本
│   ├── qt_visualize_train.py      # 可视化训练工具
│   ├── long_train.py              # 长时间训练脚本
│   └── train.py                   # 原 TF 训练脚本
├── multiagent-particle-envs/       # 多智能体粒子环境
│   └── multiagent/scenarios/
│       ├── simple.py
│       ├── simple_spread.py
│       ├── simple_adversary.py
│       ├── simple_tag.py
│       ├── simple_push.py
│       ├── simple_reference.py
│       ├── simple_speaker_listener.py
│       ├── simple_crypto.py
│       └── simple_world_comm.py
├── OrignCode/                      # 原始 TF 代码备份
├── learning_curves/                # 学习曲线数据
├── saved_models/                   # 保存的模型
├── PYTORCH_MIGRATION_README.md     # 详细迁移文档
├── CHANGELOG.md                    # 变更日志
└── README.md                       # 本文件
```

---

## 🔬 算法改进

### 相比原版的改进

1. **现代化框架**：使用 PyTorch 替代过时的 TensorFlow 1.x
2. **GPU 加速**：自动检测和使用 GPU 加速训练
3. **内存优化**：PyTorch 动态计算图提供更好的内存效率
4. **可视化工具**：提供 PyQt5 图形化训练界面
5. **兼容性修复**：解决与新版 Python、NumPy、Gym 的兼容问题
6. **中文文档**：完整的中文使用说明和场景文档

### 训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 更新频率 | 每100步 | 网络参数更新间隔 |
| 软更新系数 | 0.99 | 目标网络更新速率 (polyak) |
| Actor 正则化 | `mean(square(flatparam))` | 防止策略过于确定 |
| 梯度裁剪 | 0.5 | 防止梯度爆炸 |

---

## 📊 性能对比

| 指标 | TensorFlow 1.x | PyTorch 2.0+ |
|------|---------------|--------------|
| 训练速度 | 基准 | +15%~30% (GPU) |
| 内存使用 | 较高 | 更优 |
| 调试难度 | 困难 | 简单 |
| 生态兼容 | 老旧 | 现代 |

---

## 📝 引用

如果本项目对您的研究有帮助，请引用原始论文：

```bibtex
@article{lowe2017multi,
  title={Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments},
  author={Lowe, Ryan and Wu, Yi and Tamar, Aviv and Harb, Jean and Abbeel, Pieter and Mordatch, Igor},
  journal={Neural Information Processing Systems (NIPS)},
  year={2017}
}
```

---

## 📜 许可证

本项目采用 MIT 许可证，详见 [LICENSE.txt](./LICENSE.txt)

---

## 🔗 相关链接

- 原始仓库：[openai/maddpg](https://github.com/openai/maddpg)
- 原始环境：[openai/multiagent-particle-envs](https://github.com/openai/multiagent-particle-envs)
- 论文地址：[arxiv.org/pdf/1706.02275.pdf](https://arxiv.org/pdf/1706.02275.pdf)

---

<p align="center">
  <b>沈阳理工大学 装备工程学院 深度学习实验室</b><br>
  📧 77232623@qq.com
</p>
