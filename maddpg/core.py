"""
MADDPG 核心实现 (含注意力协调机制)

包含:
  - ParamActor: 参数化 Actor (离散技能选择 + 连续参数输出)
  - CentralizedCritic: 集中式 Critic (全局状态-动作评估)
  - AttentionCritic: 注意力协调 Critic (MAAC 风格, 可提取协作权重)
  - CommModule: 智能体间通讯模块 (消息传递)
  - MaddpgBuffer: 多智能体经验回放缓冲区
  - MaddpgTrainer: MADDPG 训练器 (含软更新, Gumbel-Softmax, 协作可视化)

参考:
  Lowe et al., "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments", NeurIPS 2017
  Iqbal & Sha, "Actor-Attention-Critic for Multi-Agent RL", ICML 2019 (MAAC)
"""

from __future__ import annotations

import copy
import random
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
#  参数化 Actor 网络
# =====================================================================

class ParamActor(nn.Module):
    """参数化 Actor: 输出离散技能概率 + 每个技能的连续参数.

    Architecture:
        obs -> shared_fc(256, 256) -> skill_logits (num_skills)
                                   -> params (num_skills * param_dim)

    使用 Gumbel-Softmax 实现端到端可微的离散采样.
    """

    def __init__(
        self,
        obs_dim: int,
        num_skills: int,
        param_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.num_skills = num_skills
        self.param_dim = param_dim

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # 离散技能选择头
        self.skill_head = nn.Linear(hidden_dim, num_skills)
        # 连续参数头: 每个技能独立输出 param_dim 维参数
        self.param_head = nn.Linear(hidden_dim, num_skills * param_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        obs: torch.Tensor,
        hard: bool = False,
        deterministic: bool = False,
        tau: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: (batch, obs_dim)
            hard: Gumbel-Softmax hard 模式 (straight-through)
            deterministic: 确定性选择 (argmax)
            tau: Gumbel-Softmax 温度

        Returns:
            skill_probs: (batch, num_skills) - Gumbel-Softmax 概率
            params: (batch, num_skills, param_dim) - 每个技能的参数
            skill_logits: (batch, num_skills) - 原始 logits
        """
        x = self.shared(obs)
        skill_logits = self.skill_head(x)
        params_raw = self.param_head(x).view(-1, self.num_skills, self.param_dim)
        params = torch.tanh(params_raw)  # 归一化到 [-1, 1]

        if deterministic:
            skill_probs = F.one_hot(
                skill_logits.argmax(dim=-1), self.num_skills
            ).float()
        else:
            skill_probs = F.gumbel_softmax(
                skill_logits, tau=tau, hard=hard, dim=-1
            )

        return skill_probs, params, skill_logits

    def select_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[int, np.ndarray]:
        """推理接口: 返回 (skill_id, param) 元组."""
        with torch.no_grad():
            skill_probs, params, _ = self.forward(
                obs, hard=True, deterministic=deterministic
            )
            skill_id = skill_probs.argmax(dim=-1).item()
            selected_param = params[0, skill_id].cpu().numpy()
            return skill_id, selected_param

    def get_action_repr(
        self, obs: torch.Tensor, tau: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """训练时获取可微分的动作表征 (用于 Critic 输入).

        Returns:
            skill_probs: (batch, num_skills) - 可微分概率
            selected_param: (batch, param_dim) - 加权平均参数
        """
        skill_probs, params, _ = self.forward(obs, hard=False, tau=tau)
        # 加权平均参数 (soft attention): 比 gather 更稳定
        # skill_probs: (batch, num_skills) -> (batch, num_skills, 1)
        weights = skill_probs.unsqueeze(-1)
        # params: (batch, num_skills, param_dim)
        selected_param = (weights * params).sum(dim=1)  # (batch, param_dim)
        return skill_probs, selected_param


# =====================================================================
#  集中式 Critic 网络
# =====================================================================

class CentralizedCritic(nn.Module):
    """集中式 Critic: 输入所有智能体的观测和动作, 输出 Q 值.

    Architecture:
        [obs_all, act_all] -> fc(256, 256) -> Q_value (1)

    动作表征 = skill_probs (num_skills) + selected_param (param_dim)
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        num_skills: int,
        param_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        # 每个智能体的动作维度 = 技能概率 + 参数
        act_repr_dim = num_skills + param_dim
        input_dim = n_agents * (obs_dim + act_repr_dim)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        obs_list: list[torch.Tensor],
        skill_probs_list: list[torch.Tensor],
        params_list: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            obs_list: n_agents x (batch, obs_dim)
            skill_probs_list: n_agents x (batch, num_skills)
            params_list: n_agents x (batch, param_dim) - 已选技能的参数

        Returns:
            Q_value: (batch, 1)
        """
        parts: list[torch.Tensor] = []
        for obs, skill_probs, params in zip(obs_list, skill_probs_list, params_list):
            parts.extend([obs, skill_probs, params])
        x = torch.cat(parts, dim=-1)
        return self.net(x)


# =====================================================================
#  注意力协调 Critic (MAAC 风格)
# =====================================================================

class AttentionCritic(nn.Module):
    """注意力协调 Critic: 借鉴 MADDPG 集中式 Critic + MAAC 注意力机制.

    核心改进 (对比 CentralizedCritic):
      1. 每个智能体的 obs+action 独立编码, 而非简单拼接
      2. 使用多头注意力捕捉智能体间协调关系
      3. 可提取注意力权重作为 "协作强度", 用于可视化

    Architecture:
        agent_i: obs_i+act_i -> encoder -> query
        all agents: obs_j+act_j -> encoder -> keys/values
        attention(query, keys, values) -> attended_repr
        [query, attended_repr] -> Q_head -> Q_value

    借鉴 MADDPG 工程:
      - 集中式训练: Critic 接收所有智能体信息 (对应原 MADDPG 的 obs_n + act_n 拼接)
      - 分布式执行: Actor 仅依赖局部观测
      - 共享回放: 使用相同索引跨智能体采样 (与原工程 replay_buffer 一致)
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        num_skills: int,
        param_dim: int,
        hidden_dim: int = 128,
        n_heads: int = 2,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads

        act_repr_dim = num_skills + param_dim
        agent_input_dim = obs_dim + act_repr_dim

        # 共享编码器 (所有智能体共用, 提高参数效率)
        self.encoder = nn.Sequential(
            nn.Linear(agent_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 多头注意力 (捕捉智能体间协调关系)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # Q 值输出头
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        obs_list: list[torch.Tensor],
        skill_probs_list: list[torch.Tensor],
        params_list: list[torch.Tensor],
        agent_idx: int = 0,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs_list: n_agents x (batch, obs_dim)
            skill_probs_list: n_agents x (batch, num_skills)
            params_list: n_agents x (batch, param_dim)
            agent_idx: 当前评估的智能体索引
            return_attention: 是否返回注意力权重 (协作可视化)

        Returns:
            Q_value: (batch, 1)
            attn_weights: (batch, n_agents) - 仅在 return_attention=True 时返回
        """
        batch_size = obs_list[0].shape[0]

        # 1. 编码所有智能体 (obs + action_repr)
        encodings: list[torch.Tensor] = []
        for obs, sp, p in zip(obs_list, skill_probs_list, params_list):
            act_repr = torch.cat([sp, p], dim=-1)
            x = torch.cat([obs, act_repr], dim=-1)
            enc = self.encoder(x)  # (batch, hidden_dim)
            encodings.append(enc)

        # 2. 堆叠为序列: (batch, n_agents, hidden_dim)
        agent_stack = torch.stack(encodings, dim=1)

        # 3. 当前智能体的编码作为 Query
        query = encodings[agent_idx].unsqueeze(1)  # (batch, 1, hidden_dim)

        # 4. 多头注意力: query 关注所有智能体
        attn_out, attn_weights = self.attention(
            query, agent_stack, agent_stack
        )
        # attn_out: (batch, 1, hidden_dim)
        # attn_weights: (batch, 1, n_agents) -> 协作权重!
        attn_out = self.attn_norm(attn_out + query)  # 残差连接

        # 5. Q 值 = f(self_encoding, attended_context)
        combined = torch.cat([
            encodings[agent_idx],
            attn_out.squeeze(1),
        ], dim=-1)
        q_value = self.q_head(combined)

        if return_attention:
            # attn_weights: (batch, 1, n_agents) -> (batch, n_agents)
            return q_value, attn_weights.squeeze(1)
        return q_value


# =====================================================================
#  智能体间通讯模块
# =====================================================================

class CommModule(nn.Module):
    """智能体间通讯模块 (借鉴 MADDPG 环境级 state.c 通讯机制).

    每个智能体生成一个消息向量, 其他智能体接收并融合.
    这实现了显式的信息共享, 补充注意力 Critic 的隐式协调.

    工作流:
      1. 每个智能体的 Actor 编码自身观测 -> 消息向量
      2. CommModule 收集所有消息, 通过注意力融合
      3. 融合后的通讯信息附加到各智能体的观测上
      4. 增强后的观测用于 Actor 的技能选择
    """

    def __init__(
        self,
        obs_dim: int,
        msg_dim: int = 16,
        hidden_dim: int = 64,
        n_agents: int = 2,
    ) -> None:
        super().__init__()
        self.msg_dim = msg_dim
        self.n_agents = n_agents

        # 消息编码器: obs -> message
        self.msg_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, msg_dim),
            nn.Tanh(),
        )

        # 消息融合 (注意力加权)
        self.msg_attention = nn.Sequential(
            nn.Linear(msg_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        obs_list: list[torch.Tensor],
        return_messages: bool = False,
    ) -> list[torch.Tensor] | tuple[list[torch.Tensor], list[torch.Tensor]]:
        """生成通讯增强后的观测.

        Args:
            obs_list: n_agents x (batch, obs_dim)
            return_messages: 是否返回原始消息 (用于可视化)

        Returns:
            enhanced_obs_list: n_agents x (batch, obs_dim + msg_dim)
            messages: n_agents x (batch, msg_dim) - 仅 return_messages=True
        """
        # 1. 生成消息
        messages = [self.msg_encoder(obs) for obs in obs_list]

        # 2. 为每个智能体融合其他智能体的消息
        enhanced_obs: list[torch.Tensor] = []
        for i in range(len(obs_list)):
            # 收集他人消息, 用注意力加权
            other_msgs = [messages[j] for j in range(len(obs_list)) if j != i]
            if other_msgs:
                my_msg = messages[i]
                # 注意力权重
                weights = []
                for other_msg in other_msgs:
                    pair = torch.cat([my_msg, other_msg], dim=-1)
                    w = self.msg_attention(pair)  # (batch, 1)
                    weights.append(w)
                weights = torch.softmax(torch.cat(weights, dim=-1), dim=-1)  # (batch, n-1)

                # 加权融合
                stacked = torch.stack(other_msgs, dim=1)  # (batch, n-1, msg_dim)
                fused = (weights.unsqueeze(-1) * stacked).sum(dim=1)  # (batch, msg_dim)
            else:
                fused = torch.zeros_like(messages[i])

            # 拼接原观测 + 融合消息
            enhanced = torch.cat([obs_list[i], fused], dim=-1)
            enhanced_obs.append(enhanced)

        if return_messages:
            return enhanced_obs, messages
        return enhanced_obs


# =====================================================================
#  经验回放缓冲区
# =====================================================================

class MaddpgBuffer:
    """多智能体经验回放缓冲区."""

    def __init__(
        self,
        capacity: int,
        n_agents: int,
        obs_dim: int,
        num_skills: int,
        param_dim: int,
    ) -> None:
        self.capacity = capacity
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.num_skills = num_skills
        self.param_dim = param_dim
        self.buffer: deque[tuple] = deque(maxlen=capacity)

    def add(
        self,
        obs: list[np.ndarray],
        skill_probs: list[np.ndarray],
        params: list[np.ndarray],
        reward: float,
        next_obs: list[np.ndarray],
        done: bool,
    ) -> None:
        """存储一步经验."""
        self.buffer.append((
            [o.copy() for o in obs],
            [sp.copy() for sp in skill_probs],
            [p.copy() for p in params],
            reward,
            [no.copy() for no in next_obs],
            done,
        ))

    def sample(
        self, batch_size: int, device: str = "cpu"
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        torch.Tensor,
        list[torch.Tensor],
        torch.Tensor,
    ]:
        """采样一批经验."""
        batch = random.sample(self.buffer, batch_size)

        obs_batch: list[torch.Tensor] = []
        skill_probs_batch: list[torch.Tensor] = []
        params_batch: list[torch.Tensor] = []
        next_obs_batch: list[torch.Tensor] = []

        for i in range(self.n_agents):
            obs_batch.append(
                torch.FloatTensor(np.array([b[0][i] for b in batch])).to(device)
            )
            skill_probs_batch.append(
                torch.FloatTensor(np.array([b[1][i] for b in batch])).to(device)
            )
            params_batch.append(
                torch.FloatTensor(np.array([b[2][i] for b in batch])).to(device)
            )
            next_obs_batch.append(
                torch.FloatTensor(np.array([b[4][i] for b in batch])).to(device)
            )

        reward_batch = torch.FloatTensor(
            np.array([b[3] for b in batch])
        ).unsqueeze(1).to(device)
        done_batch = torch.FloatTensor(
            np.array([float(b[5]) for b in batch])
        ).unsqueeze(1).to(device)

        return (
            obs_batch,
            skill_probs_batch,
            params_batch,
            reward_batch,
            next_obs_batch,
            done_batch,
        )

    def __len__(self) -> int:
        return len(self.buffer)


# =====================================================================
#  MADDPG 训练器
# =====================================================================

class MaddpgTrainer:
    """MADDPG 完整训练器 (含注意力协调 + 通讯模块).

    特性:
      - 参数化动作空间 (Gumbel-Softmax 离散 + 连续参数)
      - 集中训练, 分布执行 (CTDE)
      - 注意力协调 Critic (可提取协作权重用于可视化)
      - 智能体间通讯模块 (可选)
      - 目标网络软更新 (Polyak, 与原 MADDPG 工程一致)
      - 梯度裁剪

    借鉴 MADDPG 工程:
      - 集中式 Critic (torch_maddpg.py: obs_n + act_n 拼接)
      - 共享索引回放 (replay_buffer.py: make_index + sample_index)
      - Polyak 软更新 (polyak=0.99 → tau=0.01)
      - 梯度裁剪 (grad_norm_clipping=0.5)
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        num_skills: int,
        param_dim: int,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.95,
        tau: float = 0.01,
        gumbel_tau: float = 1.0,
        gumbel_tau_decay: float = 0.99999,
        gumbel_tau_min: float = 0.3,
        grad_clip: float = 1.0,
        use_attention: bool = True,
        use_comm: bool = False,
        device: str = "cpu",
        hidden_dim: int = 256,
    ) -> None:
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.num_skills = num_skills
        self.param_dim = param_dim
        self.gamma = gamma
        self.tau = tau
        self.gumbel_tau = gumbel_tau
        self.gumbel_tau_decay = gumbel_tau_decay
        self.gumbel_tau_min = gumbel_tau_min
        self.grad_clip = grad_clip
        self.use_attention = use_attention
        self.use_comm = use_comm
        self.device = device
        self.hidden_dim = hidden_dim

        # GPU 上使用更大的网络: A100 算力充足, 增大隐层可提升GPU利用率
        # hidden_dim: Actor/ConcatCritic 隐层宽度
        # critic_hidden: AttentionCritic 隐层宽度 (通常为 hidden_dim 的一半)
        critic_hidden = max(128, hidden_dim // 2)
        n_heads = max(2, hidden_dim // 128)  # 注意力头数随宽度增加

        # ---- 通讯模块 (可选) ----
        self.comm: Optional[CommModule] = None
        actor_obs_dim = obs_dim
        if use_comm:
            self.comm = CommModule(obs_dim, msg_dim=16, n_agents=n_agents).to(device)
            actor_obs_dim = obs_dim + 16  # 增强观测维度

        # ---- 初始化网络 ----
        self.actors: list[ParamActor] = [
            ParamActor(
                actor_obs_dim, num_skills, param_dim,
                hidden_dim=hidden_dim,
            ).to(device)
            for _ in range(n_agents)
        ]

        # Critic: 注意力版本 或 原始拼接版本
        if use_attention:
            self.critics: list[nn.Module] = [
                AttentionCritic(
                    n_agents, obs_dim, num_skills, param_dim,
                    hidden_dim=critic_hidden, n_heads=n_heads,
                ).to(device)
                for _ in range(n_agents)
            ]
        else:
            self.critics: list[nn.Module] = [
                CentralizedCritic(
                    n_agents, obs_dim, num_skills, param_dim,
                    hidden_dim=hidden_dim,
                ).to(device)
                for _ in range(n_agents)
            ]

        # 目标网络 (深拷贝)
        self.target_actors: list[ParamActor] = [
            copy.deepcopy(actor) for actor in self.actors
        ]
        self.target_critics: list[nn.Module] = [
            copy.deepcopy(critic) for critic in self.critics
        ]

        # 冻结目标网络梯度
        for ta in self.target_actors:
            for p in ta.parameters():
                p.requires_grad = False
        for tc in self.target_critics:
            for p in tc.parameters():
                p.requires_grad = False

        # ---- 优化器 ----
        actor_params = []
        for a in self.actors:
            actor_params.extend(list(a.parameters()))
        if self.comm is not None:
            actor_params.extend(list(self.comm.parameters()))

        self.actor_optimizers = [
            torch.optim.Adam(a.parameters(), lr=lr_actor)
            for a in self.actors
        ]
        self.critic_optimizers = [
            torch.optim.Adam(c.parameters(), lr=lr_critic)
            for c in self.critics
        ]
        if self.comm is not None:
            self.comm_optimizer = torch.optim.Adam(self.comm.parameters(), lr=lr_actor)

        self._update_count = 0

        # ---- 协作权重缓存 (最近一次推理的注意力权重) ----
        self._last_cooperation_weights: Optional[np.ndarray] = None
        self._last_comm_messages: Optional[list[np.ndarray]] = None

    # ------------------------------------------------------------------
    # 动作选择 (推理)
    # ------------------------------------------------------------------

    def select_actions(
        self, obs_list: list[np.ndarray], deterministic: bool = False
    ) -> list[tuple[int, np.ndarray]]:
        """为所有智能体选择动作, 同时更新协作权重缓存.

        Args:
            obs_list: 每个智能体的观测 (numpy array).

        Returns:
            list of (skill_id, param) 元组.
        """
        with torch.no_grad():
            obs_tensors = [
                torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                for obs in obs_list
            ]

            # 通讯增强: CommModule 将 obs_dim → obs_dim + msg_dim
            if self.comm is not None:
                obs_tensors = self.comm(obs_tensors)

            actions: list[tuple[int, np.ndarray]] = []
            for i in range(self.n_agents):
                skill_id, param = self.actors[i].select_action(
                    obs_tensors[i], deterministic
                )
                actions.append((skill_id, param))

            # 提取协作权重 (注意力 Critic)
            if self.use_attention:
                # Critic 使用原始 obs_dim 的观测
                raw_obs = [
                    torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    for obs in obs_list
                ]
                self._update_cooperation_weights(
                    raw_obs, obs_tensors, actions,
                )

        return actions

    def _update_cooperation_weights(
        self,
        raw_obs_tensors: list[torch.Tensor],
        enhanced_obs_tensors: list[torch.Tensor],
        actions: list[tuple[int, np.ndarray]],
    ) -> None:
        """从注意力 Critic 提取协作权重矩阵.

        结果: (n_agents, n_agents) 矩阵, [i,j] 表示 agent_i 对 agent_j 的关注度.

        Args:
            raw_obs_tensors: 原始观测 (用于 Critic)
            enhanced_obs_tensors: 通讯增强后的观测 (用于 Actor)
        """
        # 构造动作表征 (Actor 使用增强后的观测)
        sp_list: list[torch.Tensor] = []
        p_list: list[torch.Tensor] = []
        for i in range(self.n_agents):
            sp, p = self.actors[i].get_action_repr(enhanced_obs_tensors[i])
            sp_list.append(sp)
            p_list.append(p)

        # 每个智能体的 Critic 提取注意力权重 (Critic 使用原始观测)
        coop_matrix = np.zeros((self.n_agents, self.n_agents), dtype=np.float32)
        for i in range(self.n_agents):
            _, attn_w = self.critics[i](
                raw_obs_tensors, sp_list, p_list,
                agent_idx=i, return_attention=True,
            )
            coop_matrix[i] = attn_w.squeeze(0).cpu().numpy()

        self._last_cooperation_weights = coop_matrix

    def get_cooperation_weights(self) -> Optional[np.ndarray]:
        """获取最近一次推理的协作权重矩阵.

        Returns:
            (n_agents, n_agents) np.ndarray, [i,j] = agent_i 对 agent_j 的关注度.
            None 如果尚未计算.
        """
        return self._last_cooperation_weights

    # ------------------------------------------------------------------
    # 训练更新
    # ------------------------------------------------------------------

    def update(self, buffer: MaddpgBuffer, batch_size: int = 128) -> dict[str, float]:
        """执行一步 MADDPG 更新.

        Returns:
            dict: 训练指标 (各智能体的 critic_loss, actor_loss).
        """
        if len(buffer) < batch_size:
            return {}

        (
            obs_batch,
            skill_probs_batch,
            params_batch,
            reward_batch,
            next_obs_batch,
            done_batch,
        ) = buffer.sample(batch_size, self.device)

        metrics: dict[str, float] = {}

        for i in range(self.n_agents):
            # 每个 agent 独立重新计算通讯增强观测 (避免计算图冲突)
            if self.comm is not None:
                enhanced_obs = self.comm(list(obs_batch))
                with torch.no_grad():
                    enhanced_next_obs = self.comm(list(next_obs_batch))
            else:
                enhanced_obs = list(obs_batch)
                enhanced_next_obs = list(next_obs_batch)

            # ===== 1. 更新 Critic =====
            with torch.no_grad():
                # 使用目标 Actor 生成 next 动作 (用增强后的 next_obs)
                next_skill_probs_list: list[torch.Tensor] = []
                next_params_list: list[torch.Tensor] = []
                for j in range(self.n_agents):
                    n_sp, n_p = self.target_actors[j].get_action_repr(
                        enhanced_next_obs[j], tau=self.gumbel_tau
                    )
                    next_skill_probs_list.append(n_sp)
                    next_params_list.append(n_p)

                # 计算目标 Q 值 (Critic 使用原始 obs)
                if self.use_attention:
                    target_q = self.target_critics[i](
                        next_obs_batch, next_skill_probs_list, next_params_list,
                        agent_idx=i,
                    )
                else:
                    target_q = self.target_critics[i](
                        next_obs_batch, next_skill_probs_list, next_params_list,
                    )
                target_value = reward_batch + (1.0 - done_batch) * self.gamma * target_q

            # 当前 Q 值 (Critic 使用原始 obs)
            if self.use_attention:
                current_q = self.critics[i](
                    obs_batch, skill_probs_batch, params_batch, agent_idx=i,
                )
            else:
                current_q = self.critics[i](
                    obs_batch, skill_probs_batch, params_batch,
                )
            critic_loss = F.mse_loss(current_q, target_value)

            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critics[i].parameters(), self.grad_clip)
            self.critic_optimizers[i].step()

            # ===== 2. 更新 Actor + CommModule =====
            new_skill_probs_list: list[torch.Tensor] = []
            new_params_list: list[torch.Tensor] = []
            for j in range(self.n_agents):
                if j == i:
                    sp, p = self.actors[j].get_action_repr(
                        enhanced_obs[j], tau=self.gumbel_tau
                    )
                    new_skill_probs_list.append(sp)
                    new_params_list.append(p)
                else:
                    new_skill_probs_list.append(skill_probs_batch[j].detach())
                    new_params_list.append(params_batch[j].detach())

            # 策略梯度: 最大化 Q 值 (Critic 使用原始 obs)
            if self.use_attention:
                q_value = self.critics[i](
                    obs_batch, new_skill_probs_list, new_params_list, agent_idx=i,
                )
            else:
                q_value = self.critics[i](
                    obs_batch, new_skill_probs_list, new_params_list,
                )
            actor_loss = -q_value.mean()

            # 技能选择的熵正则化 (鼓励探索)
            _, _, logits = self.actors[i](enhanced_obs[i], tau=self.gumbel_tau)
            log_probs = F.log_softmax(logits, dim=-1)
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            actor_loss -= 0.01 * entropy  # 熵奖励

            self.actor_optimizers[i].zero_grad()
            if self.comm is not None and hasattr(self, 'comm_optimizer'):
                self.comm_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actors[i].parameters(), self.grad_clip)
            self.actor_optimizers[i].step()
            if self.comm is not None and hasattr(self, 'comm_optimizer'):
                self.comm_optimizer.step()

            metrics[f"agent_{i}/critic_loss"] = critic_loss.item()
            metrics[f"agent_{i}/actor_loss"] = actor_loss.item()
            metrics[f"agent_{i}/entropy"] = entropy.item()
            metrics[f"agent_{i}/q_mean"] = q_value.mean().item()

        # ===== 3. 软更新目标网络 =====
        self._soft_update()

        # ===== 4. 衰减 Gumbel 温度 =====
        self.gumbel_tau = max(
            self.gumbel_tau_min, self.gumbel_tau * self.gumbel_tau_decay
        )
        metrics["gumbel_tau"] = self.gumbel_tau

        self._update_count += 1
        return metrics

    def _soft_update(self) -> None:
        """目标网络 Polyak 软更新."""
        for i in range(self.n_agents):
            for tp, p in zip(
                self.target_actors[i].parameters(), self.actors[i].parameters()
            ):
                tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)
            for tp, p in zip(
                self.target_critics[i].parameters(), self.critics[i].parameters()
            ):
                tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    # ------------------------------------------------------------------
    # 模型保存/加载
    # ------------------------------------------------------------------

    def save(self, save_dir: str) -> None:
        """保存所有网络权重."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        for i in range(self.n_agents):
            torch.save(
                self.actors[i].state_dict(),
                os.path.join(save_dir, f"actor_{i}.pth"),
            )
            torch.save(
                self.critics[i].state_dict(),
                os.path.join(save_dir, f"critic_{i}.pth"),
            )
        # 保存 CommModule 权重 (如果启用)
        if self.comm is not None:
            torch.save(
                self.comm.state_dict(),
                os.path.join(save_dir, "comm.pth"),
            )
        # 保存训练元信息
        torch.save(
            {
                "update_count": self._update_count,
                "gumbel_tau": self.gumbel_tau,
                "use_comm": self.comm is not None,
                "use_attention": self.use_attention,
            },
            os.path.join(save_dir, "meta.pth"),
        )
        print(f"[MADDPG] 模型已保存到 {save_dir}")

    def load(self, save_dir: str) -> None:
        """加载所有网络权重."""
        import os
        for i in range(self.n_agents):
            self.actors[i].load_state_dict(
                torch.load(
                    os.path.join(save_dir, f"actor_{i}.pth"),
                    map_location=self.device,
                    weights_only=True,
                )
            )
            self.critics[i].load_state_dict(
                torch.load(
                    os.path.join(save_dir, f"critic_{i}.pth"),
                    map_location=self.device,
                    weights_only=True,
                )
            )
            # 同步目标网络
            self.target_actors[i].load_state_dict(self.actors[i].state_dict())
            self.target_critics[i].load_state_dict(self.critics[i].state_dict())

        # 加载 CommModule 权重 (如果存在)
        comm_path = os.path.join(save_dir, "comm.pth")
        if self.comm is not None and os.path.exists(comm_path):
            self.comm.load_state_dict(
                torch.load(comm_path, map_location=self.device, weights_only=True)
            )
            print(f"  [CommModule] 通讯模块权重已加载")

        meta_path = os.path.join(save_dir, "meta.pth")
        if os.path.exists(meta_path):
            meta = torch.load(meta_path, map_location="cpu", weights_only=False)
            self._update_count = meta.get("update_count", 0)
            self.gumbel_tau = meta.get("gumbel_tau", self.gumbel_tau)

        print(f"[MADDPG] 模型已从 {save_dir} 加载")
