# 变更日志

## [2025-12-16] 界面优化 - 标签页布局

### 界面重构

解决窗体内容超出显示区域的问题，采用标签页布局重新组织界面：

**新界面结构**:
```
┌─────────────────────────────┐
│  [开始] [暂停] [停止]       │  ← 控制按钮
│  [👁 观察下一轮]            │  ← 观察按钮
│  状态: 待命                 │
│  ████████████ 50%           │  ← 进度条
├─────────────────────────────┤
│ 🎮场景 │ 🧠网络 │ 💾模型    │  ← 标签页
├─────────────────────────────┤
│  场景选择                   │
│  训练参数                   │  ← 当前标签内容
│  渲染速度                   │
├─────────────────────────────┤
│  📊 Episode: 0  步数: 0     │  ← 统计信息
│     奖励: 0.00  智能体: 0   │
└─────────────────────────────┘
```

**三个标签页**:
1. **🎮 场景**: 场景选择、训练参数(Episodes/回合长度/学习率/折扣γ)、渲染速度
2. **🧠 网络**: 网络结构(隐藏单元/层数)、训练配置(批次/经验池)、参数说明
3. **💾 模型**: 保存模型、加载继续训练、加载推理演示、使用流程说明

**优点**:
- 界面紧凑，不再超出屏幕
- 功能分类清晰，易于查找
- 控制按钮和统计信息始终可见
- 每个标签页有相关说明

---

## [2025-12-16] 扩展训练参数配置

### 新增可配置参数

**训练参数**:
| 参数 | 含义 | 范围 | 默认值 |
|------|------|------|--------|
| Episode数 | 训练的总回合数 | 100-1,000,000 | 5000 |
| Episode长度 | 每回合最大时间步数 | 10-500 | 25 |
| 学习率 | 梯度下降步长 | 0.00001-0.5 | 0.01 |
| 折扣因子γ | 未来奖励折扣率 | 0-0.9999 | 0.95 |

**网络参数**:
| 参数 | 含义 | 范围 | 默认值 |
|------|------|------|--------|
| 批次大小 | 每次采样的经验数量 | 32-8192 | 1024 |
| 隐藏层单元 | 每层神经元数量 | 16-1024 | 64 |
| 网络层数 | 总层数(隐藏层+输出层) | 2-10 | 3 |
| 经验池大小 | 经验回放池容量 | 10,000-10,000,000 | 1,000,000 |

### 代码修改

- `torch_maddpg.py`: 
  - `MADDPGActor` 和 `MADDPGCritic` 支持可配置 `num_layers`
  - `MADDPGAgentTrainer` 支持 `buffer_size` 参数
- `qt_visualize_train.py`:
  - 新增"网络参数"配置组
  - 所有参数带有Tooltip说明
  - 模型保存时包含所有参数
  - 加载模型时恢复参数设置
- 添加 `simple_world_comm` 场景到下拉菜单

---

## [2025-12-16] 修复可视化问题并添加模型管理功能

### Bug修复

1. **修复Simple Speaker Listener场景智能体透明问题**
   - 问题：智能体颜色为深灰色 `[0.25, 0.25, 0.25]`，在白色背景上不明显
   - 问题：听众颜色计算可能超过1.0导致颜色溢出
   - 修复：
     - 添加 `_get_agent_color()` 方法，对颜色值进行裁剪 (clamp to 0-1)
     - 计算亮度，对过暗的颜色进行增强
     - 对 `simple_speaker_listener` 场景特殊处理：
       - 说话者(Speaker)使用紫色 `(155, 89, 182)`
       - 听众(Listener)使用橙色 `(230, 126, 34)`
     - 不可移动的智能体（说话者）用虚线双圆表示
     - 图例显示"Speaker (说话者)"/"Listener (听众)"

### 新功能

2. **模型保存功能 (💾 保存模型)**
   - 暂停训练后可保存当前模型权重
   - 保存内容包括：
     - 场景名称
     - 当前Episode数
     - 奖励历史
     - 所有智能体的Actor/Critic网络权重
     - 目标网络权重
     - 优化器状态
   - 支持自定义保存路径

3. **加载模型继续训练 (📂 加载模型继续训练)**
   - 选择之前保存的 `.pth` 文件
   - 自动恢复：场景、Episode、奖励历史、网络权重
   - 从断点继续训练

4. **加载模型推理 (🔍 加载模型推理)**
   - 加载训练好的模型进行演示
   - 自动开始可视化观察
   - 无探索噪声，展示纯策略行为
   - 循环运行直到手动停止

### 界面改进

- 新增"模型管理"按钮组
- 背景色调整为浅蓝灰色，提高智能体可见性
- 智能体标签优化：S=Speaker, L=Listener, A=Adversary
- 地标和智能体最小尺寸保证，防止太小看不见

---

## [2025-12-16] 完善场景文档

### 文档更新

更新 `PYTORCH_MIGRATION_README.md`，添加详细的场景说明：

**新增内容：**
- 场景总览表格：包含类型、智能体数、难度、核心技能
- 9个场景的详细文档：
  - **simple**: 单智能体导航基础场景
  - **simple_spread**: 协作覆盖，经典MARL基准
  - **simple_adversary**: 物理欺骗，研究信号博弈
  - **simple_tag**: 捕食者-猎物追逐游戏
  - **simple_push**: 物理对抗场景
  - **simple_reference**: 双向通信协调
  - **simple_speaker_listener**: 单向语言学习
  - **simple_crypto**: 加密通信博弈
  - **simple_world_comm**: 复杂团队协作

**每个场景包含：**
- 场景描述与背景
- 环境配置（智能体数、地标数、碰撞、通信）
- 状态空间详解
- 动作空间说明
- 奖励函数公式
- 学习目标
- 训练建议命令
- 研究意义分析

**场景选择指南：**
- 按学习难度分级（入门→专家）
- 按研究主题分类（协作、对抗、通信等）
- 按计算资源需求估算

---

## [2025-12-16] 添加PyQt5可视化训练工具

### 新功能

添加了基于PyQt5的训练可视化工具 `experiments/qt_visualize_train.py`

**功能特性：**
- **[👁 观察下一轮] 按钮**: 点击后在下一个Episode实时观察智能体移动
- 自定义环境渲染（无需pyglet，直接使用Qt绑制）
- 显示训练奖励曲线（含移动平均线）
- 可选择不同场景进行训练
- 可调节训练参数（学习率、批大小、Episode数等）
- 可调节渲染速度滑块
- 暂停/继续/停止训练控制
- 实时观察状态指示

**使用方法：**
```bash
conda activate maddpg
export SUPPRESS_MA_PROMPT=1
export PYTHONPATH=/path/to/maddpg/multiagent-particle-envs:$PYTHONPATH
python experiments/qt_visualize_train.py
```

**依赖：**
```bash
pip install pyqt5 matplotlib
```

---

## [2025-12-16] PyTorch移植代码完善

### 问题修复

1. **MADDPGCritic缺少local_q_func属性**
   - 问题：`MADDPGCritic`类缺少`local_q_func`属性，导致在`update()`方法中访问`self.critic.local_q_func`时出错
   - 修复：在`MADDPGCritic.__init__`中添加`local_q_func`参数和属性

2. **网络结构与原版不一致**
   - 问题：`MLP`类默认`num_layers=2`，导致网络结构与原版不一致
   - 修复：将默认值改为`num_layers=3`，确保与原版TensorFlow的mlp_model一致（2个隐藏层 + 1个输出层）

3. **MultiDiscrete动作空间维度计算错误**
   - 问题：`__init__`和`_get_act_dim`方法中，MultiDiscrete检测逻辑位置错误，导致动作维度计算为2而非正确的总维度（如15）
   - 修复：将MultiDiscrete检测移到最前面，确保正确计算`sum(high - low + 1)`

4. **动作采样未使用分布类**
   - 问题：PyTorch版本在`action()`和`update()`方法中简化了动作采样，没有使用完整的分布类
   - 修复：恢复使用`pdtype.pdfromflat(p).sample()`进行动作采样，与原版TensorFlow代码一致

5. **Actor正则化计算与原版不一致**
   - 问题：PyTorch版本对采样后的动作进行正则化，而原版是对分布参数（flatparam）正则化
   - 修复：改为`p_reg = torch.mean(torch.square(act_pd.flatparam()))`

### 测试验证

所有场景测试通过：
- `simple` - 单智能体场景 ✓
- `simple_spread` - 多智能体协作场景 ✓
- `simple_adversary` - 对抗场景 ✓
- `simple_reference` - MultiDiscrete动作空间（通信）场景 ✓
- `simple_tag` - 多对抗智能体场景 ✓
- DDPG模式（local_q_func=True）✓

### 训练命令示例

```bash
# 激活环境
conda activate maddpg

# 设置环境变量
export SUPPRESS_MA_PROMPT=1
export PYTHONPATH=/path/to/maddpg/multiagent-particle-envs:$PYTHONPATH

# 运行训练
python experiments/torch_train.py --scenario simple_spread --num-episodes 60000 --max-episode-len 25

# 使用DDPG模式
python experiments/torch_train.py --scenario simple --good-policy ddpg

# 对抗场景
python experiments/torch_train.py --scenario simple_adversary --num-adversaries 1
```

