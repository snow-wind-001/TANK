import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import maddpg.common.torch_util as U
from maddpg.common.torch_distributions import make_pdtype
from maddpg import AgentTrainer
from maddpg.trainer.replay_buffer import ReplayBuffer


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r
        r = r*(1.-done)
        discounted.append(r)
    return discounted[::-1]

def make_update_exp(vars, target_vars, polyak=0.99):
    expression = []
    for var, target_var in zip(vars, target_vars):
        target_var.data.copy_(polyak * target_var.data + (1.0 - polyak) * var.data)
    return expression

class MLP(nn.Module):
    """
    多层感知机网络，与原版TensorFlow的mlp_model一致
    原版结构: input → hidden (ReLU) → hidden (ReLU) → output
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(MLP, self).__init__()
        layers = []
        input_size = input_dim

        for i in range(num_layers):
            if i == num_layers - 1:
                # 最后一层，无激活函数
                layers.append(nn.Linear(input_size, output_dim))
            else:
                layers.append(nn.Linear(input_size, hidden_dim))
                layers.append(nn.ReLU())
                input_size = hidden_dim

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class MADDPGActor(nn.Module):
    def __init__(self, obs_dim, act_dim, num_units=64, num_layers=3):
        super(MADDPGActor, self).__init__()
        self.net = MLP(obs_dim, num_units, act_dim, num_layers=num_layers)

    def forward(self, obs):
        return self.net(obs)

class MADDPGCritic(nn.Module):
    def __init__(self, input_dim, num_units=64, local_q_func=False, num_layers=3):
        super(MADDPGCritic, self).__init__()
        self.input_dim = input_dim
        self.num_units = num_units
        self.local_q_func = local_q_func  # 添加local_q_func属性

        # Create the network with configurable layers
        # 默认: 2层隐藏层 (input→64→64→1)
        layers = []
        in_dim = self.input_dim
        for i in range(num_layers):
            if i == num_layers - 1:
                # 最后一层输出Q值
                layers.append(nn.Linear(in_dim, 1))
            else:
                layers.append(nn.Linear(in_dim, self.num_units))
                layers.append(nn.ReLU())
                in_dim = self.num_units
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Directly process the concatenated input [obs, actions]
        """
        return self.net(x)

class MADDPGAgentTrainer(AgentTrainer):
    def __init__(self, name, model, obs_shape_n, act_space_n, agent_index, args, local_q_func=False):
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args
        self.device = U.get_device()
        self.local_q_func = local_q_func  # 存储local_q_func属性

        # Store original action space for later use
        self.action_space = act_space_n[agent_index]
        
        # 检测MultiDiscrete动作空间 - 必须在其他检查之前
        # MultiDiscrete有high和low属性，且是数组形式
        self.is_multidiscrete = (hasattr(act_space_n[agent_index], 'high') and 
                                  hasattr(act_space_n[agent_index], 'low') and
                                  hasattr(act_space_n[agent_index].high, '__len__'))

        # Get action dimensions
        act_space = act_space_n[agent_index]

        # Handle different action space types - MultiDiscrete必须先检查
        if self.is_multidiscrete:
            # MultiDiscrete space - output concatenated one-hot (softmax) actions
            # 动作维度 = sum(high - low + 1)，即所有离散动作选项的总和
            self.act_dim = int(sum(act_space.high - act_space.low + 1))
        elif hasattr(act_space, 'n') and not hasattr(act_space, 'shape'):
            # Discrete space
            self.act_dim = act_space.n
        elif hasattr(act_space, 'shape'):
            shape = act_space.shape
            if isinstance(shape, (tuple, list)):
                if len(shape) == 0:  # Empty shape, likely Discrete
                    self.act_dim = act_space.n if hasattr(act_space, 'n') else 1
                else:  # Multi-dimensional space (Box)
                    self.act_dim = shape[0]
            else:
                # Single integer dimension
                self.act_dim = shape
        else:
            raise ValueError(f"Cannot determine action dimension for action space: {act_space}, type: {type(act_space)}")

        # Initialize observation shapes
        self.obs_shape_n = obs_shape_n
        self.act_space_n = act_space_n

        # 获取可配置参数，提供默认值以兼容旧代码
        num_layers = getattr(args, 'num_layers', 3)  # 默认3层（2隐藏层+1输出层）
        buffer_size = getattr(args, 'buffer_size', 1e6)  # 经验回放池大小
        
        # Create actor networks - 支持可配置层数
        self.actor = MADDPGActor(obs_shape_n[agent_index][0], self.act_dim, args.num_units, num_layers).to(self.device)
        self.actor_target = MADDPGActor(obs_shape_n[agent_index][0], self.act_dim, args.num_units, num_layers).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Create critic networks
        # 根据local_q_func决定critic的输入维度
        if local_q_func:
            # DDPG模式：只使用当前智能体的观测和动作
            critic_obs_dim = obs_shape_n[agent_index][0]
            critic_act_dim = self._get_act_dim(act_space_n[agent_index])
        else:
            # MADDPG模式：使用所有智能体的观测和动作
            critic_obs_dim = sum(obs_shape_n[i][0] for i in range(self.n))
            critic_act_dim = sum(self._get_act_dim(act_space_n[i]) for i in range(self.n))

        # Total input dimension for critic (obs + actions)
        total_input_dim = critic_obs_dim + critic_act_dim

        # 传递local_q_func和num_layers参数给critic
        self.critic = MADDPGCritic(total_input_dim, args.num_units, local_q_func, num_layers).to(self.device)
        self.critic_target = MADDPGCritic(total_input_dim, args.num_units, local_q_func, num_layers).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Create optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.lr)

        # Create experience buffer - 支持可配置大小
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None

        # Create action distribution - 与原版一致使用分布类
        self.pdtype = make_pdtype(act_space_n[agent_index])

    def _get_act_dim(self, act_space):
        """获取动作空间的维度 - MultiDiscrete必须首先检查"""
        # MultiDiscrete space - 必须首先检查
        if (hasattr(act_space, 'high') and hasattr(act_space, 'low') and 
            hasattr(act_space.high, '__len__')):
            # MultiDiscrete: sum of (high - low + 1) for one-hot encoding
            return int(sum(act_space.high - act_space.low + 1))
        elif hasattr(act_space, 'n') and not hasattr(act_space, 'shape'):
            # Discrete space
            return act_space.n
        elif hasattr(act_space, 'shape'):
            shape = act_space.shape
            if isinstance(shape, (tuple, list)):
                if len(shape) == 0:  # Empty shape, likely Discrete
                    return act_space.n if hasattr(act_space, 'n') else 1
                else:  # Multi-dimensional space (Box)
                    return shape[0]
            else:
                # Single integer dimension
                return shape
        else:
            raise ValueError(f"Cannot determine action dimension for action space: {act_space}, type: {type(act_space)}")

    def action(self, obs):
        """
        获取动作 - 与原版TensorFlow一致，使用分布类采样
        原版: act_sample = act_pd.sample() where act_pd = act_pdtype.pdfromflat(p)
        
        对于MultiDiscrete动作空间，环境期望扁平化的向量（所有softmax输出拼接），
        环境内部会根据action_space.high - action_space.low + 1拆分动作
        """
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            # 获取策略网络输出 (logits/parameters)
            p = self.actor(obs)
            
            # 使用分布类采样动作 - 与原版一致
            act_pd = self.pdtype.pdfromflat(p)
            action = act_pd.sample()
            
        action = action.cpu().numpy()[0]
        
        # 直接返回扁平化的动作向量
        # 对于MultiDiscrete，环境会在_set_action中自动拆分
        return action

    def _multidiscrete_action_from_logits(self, logits):
        """Convert logits to discrete actions for MultiDiscrete space"""
        low = self.action_space.low
        high = self.action_space.high

        discrete_actions = []
        current_idx = 0

        for i, (l, h) in enumerate(zip(low, high)):
            num_options = h - l + 1
            # Get logits for this discrete action
            action_logits = logits[current_idx:current_idx + num_options]
            # Sample from logits (use argmax for deterministic behavior)
            discrete_action = np.argmax(action_logits)
            discrete_actions.append(discrete_action)
            current_idx += num_options

        return np.array(discrete_actions)

    def _discrete_to_onehot(self, discrete_actions):
        """Convert discrete actions to one-hot encoding for critic"""
        low = self.action_space.low
        high = self.action_space.high
        onehot_actions = []

        for i, (discrete_action, l, h) in enumerate(zip(discrete_actions, low, high)):
            num_options = h - l + 1
            onehot = np.zeros(num_options)
            onehot[discrete_action] = 1.0
            onehot_actions.append(onehot)

        return np.concatenate(onehot_actions)

    def experience(self, obs, act, rew, new_obs, done, terminal):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, t):
        """
        更新策略和值函数 - 与原版TensorFlow一致
        """
        if len(self.replay_buffer) < self.max_replay_buffer_len: # replay buffer is not large enough
            return
        if not t % 100 == 0:  # only update every 100 steps
            return

        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)

        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index

        for i in range(self.n):
            obs, act, rew, obs_next, done = agents[i].replay_buffer.sample_index(index)
            obs_n.append(torch.tensor(obs, dtype=torch.float32, device=self.device))
            obs_next_n.append(torch.tensor(obs_next, dtype=torch.float32, device=self.device))
            act_tensor = torch.tensor(act, dtype=torch.float32, device=self.device)
            act_n.append(act_tensor)

        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        act = torch.tensor(act, dtype=torch.float32, device=self.device)
        rew = torch.tensor(rew, dtype=torch.float32, device=self.device)
        obs_next = torch.tensor(obs_next, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device)

        # train q network
        num_sample = 1
        target_q = 0.0

        for i in range(num_sample):
            target_act_next_n = []
            for j in range(self.n):
                with torch.no_grad():
                    obs_next_j = obs_next_n[j]
                    # 获取目标策略网络输出
                    target_p = agents[j].actor_target(obs_next_j)
                    # 使用分布类采样目标动作 - 与原版一致
                    target_act_pd = agents[j].pdtype.pdfromflat(target_p)
                    target_action_j = target_act_pd.sample()
                    target_act_next_n.append(target_action_j)

            # Concatenate observations and actions for critic
            if self.local_q_func:
                target_q_input = torch.cat([obs_next, target_act_next_n[self.agent_index]], dim=1)
            else:
                target_q_input = torch.cat(obs_next_n + target_act_next_n, dim=1)

            target_q_next = self.critic_target(target_q_input).squeeze(-1)
            target_q += rew + self.args.gamma * (1.0 - done) * target_q_next

        target_q /= num_sample

        # Current Q values
        if self.local_q_func:
            current_q_input = torch.cat([obs, act_n[self.agent_index]], dim=1)
        else:
            current_q_input = torch.cat(obs_n + act_n, dim=1)
        current_q = self.critic(current_q_input).squeeze(-1)

        # Critic loss
        critic_loss = F.mse_loss(current_q, target_q.detach())

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        # train p network
        # 获取当前策略网络输出
        obs_i = obs_n[self.agent_index]
        p = self.actor(obs_i)
        
        # 使用分布类包装参数 - 与原版一致
        act_pd = self.pdtype.pdfromflat(p)
        
        # 采样动作用于Q值计算
        actor_actions = act_pd.sample()
        
        # 正则化项 - 与原版一致：对分布参数(flatparam)进行正则化
        # 原版: p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))
        p_reg = torch.mean(torch.square(act_pd.flatparam()))

        # Create actions list for critic
        current_act_n = []
        for i in range(self.n):
            if i == self.agent_index:
                current_act_n.append(actor_actions)
            else:
                current_act_n.append(act_n[i])

        # Get critic value
        if self.local_q_func:
            actor_q_input = torch.cat([obs_i, actor_actions], dim=1)
        else:
            actor_q_input = torch.cat(obs_n + current_act_n, dim=1)
        actor_q = self.critic(actor_q_input).squeeze(-1)

        # Actor loss (negative Q value for maximization) + regularization
        # 原版: loss = pg_loss + p_reg * 1e-3
        pg_loss = -torch.mean(actor_q)
        actor_loss = pg_loss + p_reg * 1e-3

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

        # Update target networks - 与原版一致的软更新
        make_update_exp(list(self.actor.parameters()), list(self.actor_target.parameters()), 0.99)
        make_update_exp(list(self.critic.parameters()), list(self.critic_target.parameters()), 0.99)

        return [critic_loss.item(), actor_loss.item(), torch.mean(target_q).item(), torch.mean(rew).item(), torch.mean(target_q_next).item(), torch.std(target_q).item()]