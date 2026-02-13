# MADDPG PyTorch版本

这是从原始TensorFlow 1.x版本转换而来的PyTorch版本的MADDPG算法实现。

## 主要修改

### 1. 核心文件转换
- **`maddpg/common/torch_util.py`**: 替换`tf_util.py`，提供PyTorch工具函数
- **`maddpg/common/torch_distributions.py`**: 替换`distributions.py`，使用PyTorch实现概率分布
- **`maddpg/trainer/torch_maddpg.py`**: 替换`maddpg.py`，使用PyTorch实现MADDPG算法
- **`experiments/torch_train.py`**: 替换`train.py`，PyTorch版本的训练脚本

### 2. 环境兼容性修复
- 修复了multiagent-particle-envs与新版gym的兼容性问题
- 处理了numpy版本兼容性
- 修复了action space维度问题

### 3. 主要功能保持
- 保持原始MADDPG算法逻辑不变
- 支持单智能体和多智能体训练
- 支持模型保存和加载
- 支持学习曲线记录和可视化

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- NumPy < 2.0 (为了兼容gym)
- gym
- multiagent-particle-envs

## 安装步骤

```bash
# 克隆仓库
git clone https://github.com/openai/maddpg.git
cd maddpg

# 创建并激活conda环境
conda create -n maddpg python=3.8
conda activate maddpg

# 安装依赖
pip install torch torchvision
pip install "numpy<2.0"
pip install gym

# 安装multiagent环境
git clone https://github.com/openai/multiagent-particle-envs.git
cd multiagent-particle-envs
pip install -e .
cd ..

# 安装maddpg包
pip install -e .

# 设置环境变量（避免每次都要设置）
export PYTHONPATH=/path/to/maddpg/multiagent-particle-envs:$PYTHONPATH
export SUPPRESS_MA_PROMPT=1
```

---

## 场景详细说明 (Scenarios)

Multi-Agent Particle Environments (MPE) 提供了一系列用于研究多智能体强化学习的场景。每个场景都有不同的合作/竞争设定和复杂度。

### 场景总览

| 场景名称 | 类型 | 智能体数 | 难度 | 核心技能 |
|---------|------|---------|------|---------|
| simple | 单智能体 | 1 | ⭐ | 导航 |
| simple_spread | 协作 | 3 | ⭐⭐ | 覆盖、避碰 |
| simple_adversary | 竞争 | 3 | ⭐⭐⭐ | 欺骗、追踪 |
| simple_tag | 竞争 | 4 | ⭐⭐⭐ | 追逐、逃跑 |
| simple_push | 竞争 | 2 | ⭐⭐ | 物理对抗 |
| simple_reference | 协作 | 2 | ⭐⭐⭐ | 通信、协调 |
| simple_speaker_listener | 协作 | 2 | ⭐⭐⭐ | 语言学习 |
| simple_crypto | 混合 | 3 | ⭐⭐⭐⭐ | 加密通信 |
| simple_world_comm | 混合 | 6 | ⭐⭐⭐⭐ | 团队协作、隐蔽 |

---

### 1. Simple (单智能体导航)

**文件**: `simple.py`

#### 场景描述
最基础的单智能体环境，一个智能体需要导航到一个固定的地标位置。这是了解环境和MADDPG基础的入门场景。

#### 环境配置
- **智能体数量**: 1个
- **地标数量**: 1个 (红色目标)
- **碰撞**: 无
- **通信**: 无

#### 状态空间
```
观测向量 = [自身速度(2), 到地标的相对位置(2)]
维度: 4
```

#### 动作空间
- 类型: Discrete (5个动作)
- 动作: 无操作, 上, 下, 左, 右

#### 奖励函数
```python
reward = -distance²  # 距离目标越远，惩罚越大
```

#### 学习目标
- 学习基本的导航策略
- 理解位置-动作关系

#### 训练建议
```bash
python experiments/torch_train.py --scenario simple --num-episodes 3000
```

#### 分析
这是一个纯粹的单智能体任务，主要用于：
- 验证环境安装正确
- 测试基本算法实现
- 作为更复杂场景的基准

---

### 2. Simple Spread (协作覆盖)

**文件**: `simple_spread.py`

#### 场景描述
N个智能体需要协作覆盖N个地标。这是最经典的**协作多智能体**场景，智能体需要学会分工合作，每个智能体占据一个地标，同时避免互相碰撞。

#### 环境配置
- **智能体数量**: 3个 (蓝色)
- **地标数量**: 3个 (灰色)
- **碰撞**: 智能体之间可碰撞
- **通信**: 可选（默认静默）
- **合作**: ✅ 完全合作（共享奖励）

#### 状态空间
```
观测向量 = [
    自身速度(2),
    自身位置(2),
    到各地标的相对位置(2×3),
    到其他智能体的相对位置(2×2),
    其他智能体通信信号(2×2)
]
维度: 18
```

#### 动作空间
- 类型: Discrete (5个动作)
- 动作: 无操作, 上, 下, 左, 右

#### 奖励函数
```python
# 全局奖励 = 地标覆盖奖励 + 碰撞惩罚
for landmark in landmarks:
    reward -= min(距离最近智能体到该地标的距离)
for agent in agents:
    if 发生碰撞:
        reward -= 1
```

#### 学习目标
- 学习去中心化的任务分配
- 避免多个智能体去同一地标
- 最小化整体覆盖距离
- 避免碰撞

#### 训练建议
```bash
python experiments/torch_train.py --scenario simple_spread --num-episodes 25000 --max-episode-len 25
```

#### 分析
**关键挑战**:
1. **任务分配问题**: 没有中央协调器，智能体需要自主决定谁去哪个地标
2. **对称性打破**: 初始状态对称，智能体需要学会打破对称性
3. **多目标优化**: 既要覆盖地标又要避免碰撞

**训练现象**:
- 早期: 智能体会聚集在一起或都去同一个地标
- 中期: 逐渐学会分散，但分配不稳定
- 后期: 形成稳定的分工策略

**研究意义**: 这是研究**涌现通信**和**隐式协调**的理想场景。

---

### 3. Simple Adversary (物理欺骗)

**文件**: `simple_adversary.py`

#### 场景描述
经典的**对抗性**场景。1个对抗者（红色）需要追踪目标地标，而2个合作智能体（蓝色）知道目标地标是哪个，需要保护目标——通过**物理欺骗**让对抗者产生困惑。

#### 环境配置
- **智能体数量**: 3个
  - 1个对抗者 (红色): 不知道目标是哪个
  - 2个合作智能体 (蓝色): 知道目标位置
- **地标数量**: 2个 (1个目标地标为绿色)
- **碰撞**: 无
- **通信**: 无

#### 状态空间

**合作智能体（蓝色）**:
```
观测向量 = [
    到目标地标的相对位置(2),      # 知道目标是哪个
    到所有地标的相对位置(2×2),
    到其他智能体的相对位置(2×2)
]
```

**对抗者（红色）**:
```
观测向量 = [
    到所有地标的相对位置(2×2),    # 不知道哪个是目标
    到其他智能体的相对位置(2×2)
]
```

#### 奖励函数

**合作智能体**:
```python
# 希望蓝色智能体靠近目标，红色智能体远离目标
reward = -min(蓝色到目标距离) + sum(红色到目标距离)
```

**对抗者**:
```python
reward = -distance²(自己, 目标)  # 靠近目标获得奖励
```

#### 学习目标
- **合作智能体**: 学会欺骗——让对抗者误以为另一个地标是目标
- **对抗者**: 学会观察合作智能体的行为来推断目标位置

#### 训练建议
```bash
python experiments/torch_train.py --scenario simple_adversary --num-adversaries 1 --num-episodes 30000
```

#### 分析
**博弈论视角**:
- 这是一个**不完全信息博弈**
- 对抗者需要通过观察行为来推断隐藏信息
- 合作者需要学会**欺骗性行为**

**训练现象**:
1. **朴素策略**: 蓝色智能体直接去目标 → 红色学会跟随蓝色
2. **欺骗策略**: 一个蓝色去目标，另一个去假目标做诱饵
3. **高级策略**: 两个蓝色都去非目标，靠近时再突然转向

**研究意义**: 用于研究**信号博弈**、**欺骗学习**和**对抗训练**。

---

### 4. Simple Tag (捕食者-猎物)

**文件**: `simple_tag.py`

#### 场景描述
经典的**追逐-逃跑**游戏。3个较慢的捕食者（红色）协作追捕1个较快的猎物（绿色）。环境中有障碍物作为掩护。

#### 环境配置
- **智能体数量**: 4个
  - 3个捕食者 (红色): 体型较大(0.075)，速度较慢(max=1.0)
  - 1个猎物 (绿色): 体型较小(0.05)，速度较快(max=1.3)
- **地标数量**: 2个 (障碍物，可碰撞)
- **碰撞**: ✅ 所有实体可碰撞
- **边界惩罚**: ✅

#### 状态空间
```
观测向量 = [
    自身速度(2),
    自身位置(2),
    到地标的相对位置(2×2),
    到其他智能体的相对位置(2×3),
    猎物速度(2)  # 仅对捕食者可见
]
```

#### 奖励函数

**捕食者（红色）**:
```python
reward = 0
if 任何捕食者碰到猎物:
    reward += 10  # 团队共享奖励
# 可选: 距离奖励塑形
```

**猎物（绿色）**:
```python
reward = 0
if 被捕食者碰到:
    reward -= 10
# 边界惩罚
for 每个维度:
    if 位置 > 0.9:
        reward -= bound_penalty(位置)
```

#### 学习目标
- **捕食者**: 学会团队协作包围猎物
- **猎物**: 学会躲避、利用障碍物逃跑

#### 训练建议
```bash
python experiments/torch_train.py --scenario simple_tag --num-adversaries 3 --num-episodes 50000
```

#### 分析
**关键特征**:
1. **速度不对称**: 猎物更快，但捕食者数量多
2. **团队协作需求**: 单个捕食者很难抓到猎物
3. **环境利用**: 障碍物可作为躲避工具

**训练现象**:
- **早期**: 捕食者各自追逐，猎物轻松逃脱
- **中期**: 捕食者学会包抄，猎物学会利用墙壁
- **后期**: 形成复杂的追逐-逃跑策略

**研究意义**: 经典的**捕食者-猎物动力学**研究场景。

---

### 5. Simple Push (物理对抗)

**文件**: `simple_push.py`

#### 场景描述
1个对抗者需要**物理推挤**合作智能体，阻止其接近目标地标。合作智能体需要绕过对抗者到达目标。

#### 环境配置
- **智能体数量**: 2个
  - 1个对抗者 (红色)
  - 1个合作智能体 (蓝色/绿色)
- **地标数量**: 2个 (1个是目标)
- **碰撞**: ✅
- **目标**: 随机选择一个地标为目标

#### 状态空间

**合作智能体**:
```
观测向量 = [
    自身速度(2),
    到目标的相对位置(2),
    自身颜色(3),
    到地标的相对位置(2×2),
    地标颜色(3×2),
    到对抗者的相对位置(2)
]
```

**对抗者**:
```
观测向量 = [
    自身速度(2),
    到地标的相对位置(2×2),
    到合作智能体的相对位置(2)
]
```

#### 奖励函数

**合作智能体**:
```python
reward = -distance(自己, 目标)
```

**对抗者**:
```python
reward = min(合作智能体到目标的距离) - distance(自己, 目标)
# 希望合作智能体远离目标，自己可以控制场地
```

#### 学习目标
- **合作智能体**: 绕过对抗者，快速到达目标
- **对抗者**: 预判对手意图，进行阻挡

#### 训练建议
```bash
python experiments/torch_train.py --scenario simple_push --num-adversaries 1 --num-episodes 20000
```

#### 分析
**物理交互特点**:
- 这是一个需要**物理接触**的场景
- 对抗者通过碰撞来阻止而非捕获

**策略发展**:
- 简单策略: 直线前进
- 中级策略: 躲避后绕行
- 高级策略: 假动作、变速

---

### 6. Simple Reference (合作参考)

**文件**: `simple_reference.py`

#### 场景描述
两个智能体需要**互相告知**对方应该去哪个地标。每个智能体只知道对方的目标，需要通过**通信**来传达这一信息。

#### 环境配置
- **智能体数量**: 2个
- **地标数量**: 3个 (红/绿/蓝)
- **碰撞**: 无
- **通信**: ✅ 高维通信 (dim_c=10)
- **合作**: ✅ 完全合作

#### 状态空间
```
观测向量 = [
    自身速度(2),
    到地标的相对位置(2×3),
    我知道对方应该去的地标颜色(3),
    对方的通信信号(10)
]
```

#### 奖励函数
```python
# 每个智能体的奖励 = -对方与其目标的距离²
reward = -distance²(goal_a.position, goal_b.position)
# 其中 goal_a 是我知道的"对方应该去的智能体"
# goal_b 是我知道的"对方应该去的地标"
```

#### 学习目标
- 学习**发送**有意义的通信信号
- 学习**理解**对方的通信信号
- 基于通信信号采取正确行动

#### 训练建议
```bash
python experiments/torch_train.py --scenario simple_reference --num-episodes 30000
```

#### 分析
**通信挑战**:
1. **接地问题**: 如何将符号与实际地标对应
2. **对称性**: 两个智能体需要建立一致的"语言"
3. **多任务**: 既要发送信号又要理解信号

**研究意义**: 用于研究**涌现通信**和**语言接地**。

---

### 7. Simple Speaker Listener (说话者-听众)

**文件**: `simple_speaker_listener.py`

#### 场景描述
**非对称通信**场景。一个不能移动的"说话者"知道目标地标，需要通过通信告诉一个不能说话的"听众"去哪里。

#### 环境配置
- **智能体数量**: 2个
  - 说话者 (Agent 0): 不能移动，只能发送信号
  - 听众 (Agent 1): 不能说话，只能移动
- **地标数量**: 3个 (红/绿/蓝)
- **通信**: ✅ 单向 (说话者→听众)
- **合作**: ✅

#### 状态空间

**说话者**:
```
观测向量 = [目标地标颜色(3)]
# 只需要告诉听众去哪个颜色的地标
```

**听众**:
```
观测向量 = [
    自身速度(2),
    到地标的相对位置(2×3),
    说话者的通信信号(3)
]
```

#### 奖励函数
```python
# 共享奖励 = -听众到目标的距离²
reward = -distance²(听众, 目标地标)
```

#### 学习目标
- **说话者**: 学习将地标颜色编码为通信信号
- **听众**: 学习理解通信信号并导航到正确地标

#### 训练建议
```bash
python experiments/torch_train.py --scenario simple_speaker_listener --num-episodes 25000
```

#### 分析
**语言学习经典场景**:
- 这是最纯粹的**指称游戏**(referential game)
- 说话者需要发展出一套"词汇"
- 听众需要理解这套"词汇"

**训练现象**:
1. 早期: 随机信号，听众随机移动
2. 中期: 信号与地标开始建立关联
3. 后期: 形成稳定的"语言"映射

---

### 8. Simple Crypto (加密通信)

**文件**: `simple_crypto.py`

#### 场景描述
**加密通信**博弈。1个说话者(Alice)需要将目标信息加密后传给1个听众(Bob)，同时防止1个窃听者(Eve)破解。

#### 环境配置
- **智能体数量**: 3个
  - Eve (对抗者): 窃听者，尝试破解消息
  - Bob (听众): 接收者，需要正确解码
  - Alice (说话者): 发送者，需要加密通信
- **地标数量**: 2个 (不同颜色编码)
- **通信**: ✅ 带密钥
- **密钥**: Alice和Bob共享密钥，Eve没有

#### 状态空间

**Alice (说话者)**:
```
观测向量 = [目标颜色(4), 密钥(4)]
# 需要用密钥加密目标信息
```

**Bob (听众)**:
```
观测向量 = [密钥(4), Alice的通信(4)]
# 可以用密钥解密
```

**Eve (窃听者)**:
```
观测向量 = [Alice的通信(4)]
# 只能看到加密后的信息
```

#### 奖励函数

**Alice和Bob (合作)**:
```python
# Bob正确解码获得奖励，Eve正确解码带来惩罚
good_rew = -||Bob输出 - 目标||²
adv_rew = ||Eve输出 - 目标||²  # Eve越错越好
reward = good_rew + adv_rew
```

**Eve**:
```python
reward = -||自己输出 - 目标||²
```

#### 学习目标
- **Alice**: 学习加密算法（结合密钥和目标）
- **Bob**: 学习解密算法（用密钥解读信息）
- **Eve**: 学习破解/猜测算法

#### 训练建议
```bash
python experiments/torch_train.py --scenario simple_crypto --num-adversaries 1 --num-episodes 50000
```

#### 分析
**安全多方计算视角**:
- Alice和Bob需要发展出**密钥依赖**的编码方案
- 理想情况下，没有密钥的Eve无法区分不同目标的信息

**训练动态**:
1. 初始: Alice发送原始信息 → Bob和Eve都能解码
2. 中期: Alice开始使用密钥 → Bob利用密钥解码，Eve效果下降
3. 后期: 加密变得更复杂，形成对抗平衡

**研究意义**: 研究**对抗性通信**和**加密策略涌现**。

---

### 9. Simple World Comm (复杂通信世界)

**文件**: `simple_world_comm.py`

#### 场景描述
最复杂的场景。4个捕食者追捕2个猎物，环境中有**森林区域**可以隐藏，捕食者团队有1个**领队**可以通信协调。

#### 环境配置
- **智能体数量**: 6个
  - 4个捕食者 (红色): 包括1个领队(可通信)
  - 2个猎物 (绿色): 可隐藏在森林中
- **地标数量**: 5个
  - 1个障碍物
  - 2个食物 (猎物可吃)
  - 2个森林 (可隐藏)
- **碰撞**: ✅
- **通信**: ✅ 领队可向队友通信
- **视野遮蔽**: ✅ 森林中的智能体互相看不见

#### 状态空间
```
观测向量 = [
    自身速度(2),
    自身位置(2),
    到实体的相对位置,
    到其他智能体的相对位置,  # 受森林遮蔽影响
    森林内/外状态(2),
    通信信号(4)  # 来自领队
]
```

#### 奖励函数

**捕食者**:
```python
reward = 0
if 碰到猎物:
    reward += 5  # 团队奖励
# 距离塑形
reward -= 0.1 * min(到猎物的距离)
```

**猎物**:
```python
reward = 0
if 被捕获:
    reward -= 5
if 吃到食物:
    reward += 2
# 边界惩罚
reward -= boundary_penalty
```

#### 学习目标
- **捕食者团队**: 协作搜索、通信协调、包围猎物
- **猎物**: 利用森林隐藏、收集食物、逃跑

#### 训练建议
```bash
python experiments/torch_train.py --scenario simple_world_comm --num-adversaries 4 --num-episodes 100000
```

#### 分析
**复杂性来源**:
1. **部分可观测**: 森林造成视野遮蔽
2. **通信协调**: 领队需要有效传达信息
3. **多目标**: 捕食vs吃食物vs生存
4. **大规模**: 6个智能体的协调

**策略层次**:
- 个体策略: 移动、躲避、攻击
- 团队策略: 分工、包抄、通信
- 高级策略: 森林搜索、守株待兔

---

## 场景选择指南

### 按学习难度

```
入门级: simple
基础级: simple_spread, simple_push
中级: simple_adversary, simple_tag, simple_speaker_listener
高级: simple_reference, simple_crypto
专家级: simple_world_comm
```

### 按研究主题

| 研究方向 | 推荐场景 |
|---------|---------|
| 基础导航 | simple |
| 协作学习 | simple_spread, simple_reference |
| 对抗学习 | simple_adversary, simple_tag |
| 涌现通信 | simple_speaker_listener, simple_reference |
| 语言接地 | simple_speaker_listener |
| 加密/安全 | simple_crypto |
| 部分可观测 | simple_world_comm |
| 欺骗策略 | simple_adversary |

### 按计算资源

| 场景 | 训练回合 | 预计时间(CPU) |
|-----|---------|--------------|
| simple | 3,000 | 10分钟 |
| simple_spread | 25,000 | 1小时 |
| simple_adversary | 30,000 | 1.5小时 |
| simple_tag | 50,000 | 3小时 |
| simple_speaker_listener | 25,000 | 1小时 |
| simple_reference | 30,000 | 2小时 |
| simple_crypto | 50,000 | 3小时 |
| simple_world_comm | 100,000 | 8小时 |

---

## 使用方法

### 基本训练

```bash
# 单智能体简单场景
python experiments/torch_train.py --scenario simple --num-episodes 1000 --max-episode-len 25

# 多智能体场景
python experiments/torch_train.py --scenario simple_spread --num-episodes 5000 --max-episode-len 25 --num-adversaries 2

# 自定义参数
python experiments/torch_train.py \
    --scenario simple_tag \
    --num-episodes 10000 \
    --max-episode-len 50 \
    --lr 1e-2 \
    --batch-size 1024 \
    --num-units 64 \
    --save-rate 1000 \
    --exp-name my_experiment
```

### 参数说明

- `--scenario`: 环境场景名称 (simple, simple_spread, simple_tag等)
- `--num-episodes`: 训练总回合数
- `--max-episode-len`: 每回合最大步数
- `--num-adversaries`: 对抗智能体数量
- `--lr`: 学习率
- `--batch-size`: 批量大小
- `--num-units`: 网络隐藏层单元数
- `--save-rate`: 模型保存间隔
- `--exp-name`: 实验名称
- `--save-dir`: 模型保存目录
- `--plots-dir`: 学习曲线保存目录

### 模型保存和加载

训练过程中会自动保存：
- 模型文件：`{save_dir}/agent_{i}.pth`
- 学习曲线：`{plots_dir}/{exp_name}_rewards.pkl`

### 评估模式

```bash
# 评估已训练模型
python experiments/torch_train.py --scenario simple --display --load-dir /path/to/saved/models
```

---

## 主要技术改进

1. **现代化框架**: 使用PyTorch替代过时的TensorFlow 1.x
2. **GPU加速**: 自动检测和使用GPU加速训练
3. **更好的内存管理**: PyTorch的动态计算图提供更好的内存效率
4. **现代化API**: 使用PyTorch的现代神经网络API
5. **兼容性**: 修复了与新版Python和依赖库的兼容性问题

## 与原版TensorFlow代码的对应

### 网络结构
- **Actor网络**: 2层隐藏层 (input→64→64→output)，与原版`mlp_model`一致
- **Critic网络**: 2层隐藏层，输入为所有智能体的观测和动作拼接

### 分布类
完整复现原版所有分布类：
- `SoftCategoricalPd`: 软分类分布（Discrete动作空间）
- `SoftMultiCategoricalPd`: 软多分类分布（MultiDiscrete动作空间）
- `DiagGaussianPd`: 对角高斯分布（Box动作空间）
- `BernoulliPd`: 伯努利分布（MultiBinary动作空间）

### 训练逻辑
- 每100步更新一次网络
- 使用软更新（polyak=0.99）更新目标网络
- Actor正则化: `p_reg = mean(square(flatparam))`
- 梯度裁剪: 0.5

## 文件结构

```
maddpg/
├── maddpg/common/torch_util.py          # PyTorch工具函数
├── maddpg/common/torch_distributions.py # 概率分布实现
├── maddpg/trainer/torch_maddpg.py       # MADDPG核心算法
├── maddpg/trainer/replay_buffer.py      # 经验回放池(原版)
├── experiments/torch_train.py           # 训练脚本
├── experiments/qt_visualize_train.py    # 可视化训练界面
└── multiagent-particle-envs/            # 多智能体环境
    └── multiagent/scenarios/
        ├── simple.py                    # 单智能体导航
        ├── simple_spread.py             # 协作覆盖
        ├── simple_adversary.py          # 物理欺骗
        ├── simple_tag.py                # 捕食者-猎物
        ├── simple_push.py               # 物理对抗
        ├── simple_reference.py          # 合作参考
        ├── simple_speaker_listener.py   # 说话者-听众
        ├── simple_crypto.py             # 加密通信
        └── simple_world_comm.py         # 复杂通信世界
```

## 可视化训练工具

提供基于PyQt5的可视化训练界面，支持实时观察训练过程。

### 安装依赖
```bash
pip install pyqt5 matplotlib
```

### 启动可视化界面
```bash
python experiments/qt_visualize_train.py
```

### 功能说明
- **环境渲染**: 实时显示多智能体环境状态
- **奖励曲线**: 显示训练奖励变化趋势，含移动平均线
- **参数调节**: 可调节场景、学习率、批大小等参数
- **可视化开关**: 可选择开启/关闭环境渲染以提高训练速度
- **观察按钮**: "👁 观察下一轮"可在下一个episode观察智能体行为
- **训练控制**: 支持暂停、继续、停止训练

### 界面截图说明
- 左侧面板: 场景选择、训练参数、可视化设置、控制按钮
- 右上区域: 环境渲染视图
- 右下区域: 奖励曲线图

## 注意事项

1. 由于multiagent-particle-envs已被标记为不维护，建议考虑迁移到PettingZoo
2. 训练时会看到gym相关的警告，可以忽略
3. GPU会自动检测并使用，如果没有GPU则使用CPU
4. 模型保存格式为PyTorch的.pth格式
5. 可视化工具需要图形界面环境（X11）

## 性能对比

PyTorch版本相比原始TensorFlow版本具有：
- 更快的训练速度（特别是在GPU上）
- 更好的内存使用效率
- 更灵活的调试能力
- 与现代深度学习生态系统的更好集成
