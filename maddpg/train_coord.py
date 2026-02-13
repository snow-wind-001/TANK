"""
高层 MADDPG 协同训练主循环 — 多环境 + 多 GPU 加速版

核心优化 (充分利用 4xA100 + 104 核 CPU):
  1. 多环境并行收集: N 个环境同时 step, buffer 填充速度 Nx
  2. 多步梯度更新: 每次收集后做 K 次 update, GPU 不空闲
  3. 多 GPU 人口训练: 4 个独立训练进程分布在 4 块 GPU
  4. 技能推理 CPU: PPO 小模型推理在 CPU, GPU 专注 MADDPG 训练

训练流程:
  1. 加载预训练的底层技能 (CPU) 或使用规则替代
  2. 初始化高层 MADDPG 协调器 (GPU)
  3. N 个并行环境同时收集经验
  4. 高层每 skill_interval 步输出一次技能指令
  5. 多步梯度更新, 充分利用 GPU 算力

用法:
  python -m maddpg.train_coord --episodes 2000 --use_rule_skills
  python -m maddpg.train_coord --episodes 2000 --n_collect_envs 8 --updates_per_step 4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.multi_tank_env import MultiTankTeamEnv, OBS_DIM
from maddpg.core import MaddpgBuffer, MaddpgTrainer
from skills.skill_wrapper import RuleBasedSkill, SkillOption


def build_skill_library(
    use_rule_skills: bool = False,
    model_dir: str = "skills/models",
    device: str = "cpu",
) -> dict[int, object]:
    """构建底层技能库.

    注意: 即使 MADDPG 在 GPU 训练, 技能推理也始终在 CPU 上,
    因为 PPO MLP 推理极快 (< 0.1ms), 放 GPU 反而增加数据搬运开销.
    """
    # 技能推理始终在 CPU (PPO MLP 太小, GPU 反而慢)
    skill_device = "cpu"

    if use_rule_skills:
        print("[技能库] 使用规则 AI 底层技能 (无需预训练)")
        return {
            0: RuleBasedSkill(skill_type="navigate"),
            1: RuleBasedSkill(skill_type="attack"),
            2: RuleBasedSkill(skill_type="defend"),
        }

    lib: dict[int, object] = {}
    skill_map = {0: "navigate", 1: "attack", 2: "defend"}

    for sid, name in skill_map.items():
        model_path = os.path.join(model_dir, f"{name}_skill")
        if os.path.exists(model_path + ".zip"):
            lib[sid] = SkillOption(model_path, device=skill_device)
            print(f"  [技能{sid}] 已加载 PPO 模型: {name} (CPU)")
        else:
            lib[sid] = RuleBasedSkill(skill_type=name)
            print(f"  [技能{sid}] 未找到模型, 使用规则替代: {name}")

    return lib


def execute_skill(
    skill_lib: dict[int, object],
    skill_id: int,
    local_obs: np.ndarray,
    target_param: np.ndarray,
) -> int:
    """调用底层技能, 返回离散动作 (0-5)."""
    skill_id = max(0, min(skill_id, len(skill_lib) - 1))
    skill = skill_lib[skill_id]
    return skill.act(local_obs, target_param, deterministic=True)


# =====================================================================
#  多环境并行收集器
# =====================================================================

class MultiEnvCollector:
    """管理 N 个并行环境, 同时收集经验.

    每个环境独立运行一个 episode, 结束后自动 reset.
    所有经验汇入同一个 MaddpgBuffer.
    """

    def __init__(
        self,
        n_envs: int,
        n_agents: int,
        skill_interval: int,
        skill_lib: dict,
        map_pool: list[str],
        n_blue: int,
        num_skills: int = 3,
        max_steps: int = 600,
        difficulty: str = "easy",
    ) -> None:
        self.n_envs = n_envs
        self.n_agents = n_agents
        self.skill_interval = skill_interval
        self.skill_lib = skill_lib
        self.map_pool = map_pool
        self.n_blue = n_blue
        self.num_skills = num_skills
        self.max_steps = max_steps
        self.difficulty = difficulty

        # 创建环境
        self.envs: list[MultiTankTeamEnv] = []
        for i in range(n_envs):
            map_name = map_pool[i % len(map_pool)]
            env = MultiTankTeamEnv(
                map_name=map_name, n_red=n_agents,
                n_blue=n_blue, max_steps=max_steps,
                difficulty=difficulty,
            )
            self.envs.append(env)

        # 每个环境的状态
        self.obs_all: list[list[np.ndarray]] = []
        self.done_all: list[bool] = [True] * n_envs
        self.step_all: list[int] = [0] * n_envs
        self.reward_all: list[float] = [0.0] * n_envs   # episode 总奖励 (用于日志)
        self.skill_ids_all: list[list[int]] = [[1, 1]] * n_envs
        self.skill_params_all: list[list[np.ndarray]] = []

        # [关键修复] Option 级别的经验追踪
        self.prev_obs_all: list[list[np.ndarray]] = [[] for _ in range(n_envs)]
        self.acc_reward_all: list[float] = [0.0] * n_envs  # option 内累积奖励

        self._completed_episodes: list[dict] = []

    def _build_action_repr(self, env_idx: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """从实际执行的 skill_id + param 构建动作表征 (one-hot + param).

        [关键修复] 存储实际执行的动作, 而非当前策略的输出.
        """
        sp_list: list[np.ndarray] = []
        p_list: list[np.ndarray] = []
        for j in range(self.n_agents):
            one_hot = np.zeros(self.num_skills, dtype=np.float32)
            one_hot[self.skill_ids_all[env_idx][j]] = 1.0
            sp_list.append(one_hot)
            p_list.append(self.skill_params_all[env_idx][j].copy())
        return sp_list, p_list

    def reset_all(self) -> None:
        """重置所有环境."""
        self.obs_all = []
        PARAM_DIM = 2
        for i, env in enumerate(self.envs):
            obs = env.reset()
            self.obs_all.append(obs)
            self.done_all[i] = False
            self.step_all[i] = 0
            self.reward_all[i] = 0.0
            self.acc_reward_all[i] = 0.0
            self.prev_obs_all[i] = [o.copy() for o in obs]
            self.skill_ids_all[i] = [1] * self.n_agents
            self.skill_params_all.append(
                [np.zeros(PARAM_DIM, dtype=np.float32)] * self.n_agents
            )
        self._completed_episodes = []

    def step_all_envs(
        self,
        trainer: MaddpgTrainer,
        buffer: MaddpgBuffer,
        device: str,
    ) -> int:
        """所有活跃环境前进一步, 返回新增经验数.

        [关键修复] Option 级别的经验存储:
          1. 在每个决策点 (step % skill_interval == 0) 存储上一个 option 的转移
          2. 累积 option 内所有步的奖励 (而非只存 1/8)
          3. 使用实际执行的动作 (one-hot skill_id + param), 而非当前策略输出
          4. 转移: (决策时obs, 实际动作, 累积奖励, 下次决策obs, done)

        Returns:
            int: 本轮新增到 buffer 的经验条数
        """
        new_experiences = 0
        PARAM_DIM = 2

        for i in range(self.n_envs):
            if self.done_all[i]:
                # 自动重置已完成的环境
                map_name = self.map_pool[i % len(self.map_pool)]
                self.envs[i] = MultiTankTeamEnv(
                    map_name=map_name, n_red=self.n_agents,
                    n_blue=self.n_blue, max_steps=self.max_steps,
                    difficulty=self.difficulty,
                )
                obs_reset = self.envs[i].reset()
                self.obs_all[i] = obs_reset
                self.done_all[i] = False
                self.step_all[i] = 0
                self.reward_all[i] = 0.0
                self.acc_reward_all[i] = 0.0
                self.prev_obs_all[i] = [o.copy() for o in obs_reset]
                self.skill_ids_all[i] = [1] * self.n_agents
                self.skill_params_all[i] = [
                    np.zeros(PARAM_DIM, dtype=np.float32)
                ] * self.n_agents

            obs = self.obs_all[i]

            # 高层决策点: 每 skill_interval 步做一次
            if self.step_all[i] % self.skill_interval == 0:
                # [修复] 存储上一个 option 的完整转移 (如果不是第 0 步)
                if self.step_all[i] > 0:
                    sp_list, p_list = self._build_action_repr(i)
                    buffer.add(
                        self.prev_obs_all[i],   # 上次决策时的 obs
                        sp_list, p_list,         # 实际执行的动作
                        self.acc_reward_all[i],  # option 内累积奖励
                        obs,                     # 本次决策时的 obs (option 结束后)
                        False,                   # episode 未结束
                    )
                    new_experiences += 1

                # 重置 option 奖励累积器, 保存当前 obs 作为新 option 的起点
                self.acc_reward_all[i] = 0.0
                self.prev_obs_all[i] = [o.copy() for o in obs]

                # 选择新的技能
                actions = trainer.select_actions(obs, deterministic=False)
                self.skill_ids_all[i] = [a[0] for a in actions]
                self.skill_params_all[i] = [a[1] for a in actions]

            # 底层技能执行
            low_actions = []
            for j in range(self.n_agents):
                act = execute_skill(
                    self.skill_lib,
                    self.skill_ids_all[i][j],
                    obs[j],
                    self.skill_params_all[i][j],
                )
                low_actions.append(act)

            next_obs, reward, done, info = self.envs[i].step(low_actions)
            self.reward_all[i] += reward       # episode 总奖励 (日志用)
            self.acc_reward_all[i] += reward    # [修复] option 内累积奖励
            self.step_all[i] += 1

            self.obs_all[i] = next_obs

            if done:
                # [修复] episode 结束时存储最后一个 (可能不完整的) option 转移
                sp_list, p_list = self._build_action_repr(i)
                buffer.add(
                    self.prev_obs_all[i],
                    sp_list, p_list,
                    self.acc_reward_all[i],
                    next_obs,
                    True,  # done!
                )
                new_experiences += 1

                self.done_all[i] = True
                self._completed_episodes.append({
                    "reward": self.reward_all[i],
                    "steps": self.step_all[i],
                    "win": info.get("win", False),
                    "blue_killed": info.get("blue_killed", 0),
                    "blue_total": info.get("blue_total", 0),
                    "info": info,
                })

        return new_experiences

    def pop_completed_episodes(self) -> list[dict]:
        """取出并清空已完成的 episode 列表."""
        eps = self._completed_episodes
        self._completed_episodes = []
        return eps


# =====================================================================
#  训练主循环
# =====================================================================

def train(
    episodes: int = 2000,
    skill_interval: int = 4,  # [v3] 更频繁高层决策, PPO短期更有效
    coord_batch_size: int = 512,
    buffer_capacity: int = 500_000,
    warmup_steps: int = 500,
    n_collect_envs: int = 32,
    updates_per_step: int = 16,
    hidden_dim: int = 256,
    use_rule_skills: bool = True,  # [推荐] 默认使用增强规则技能, 更可靠
    map_name: str = "classic_1",
    n_blue: int = 4,
    difficulty: str = "easy",  # [新增] 难度: easy(2v2), medium(2v3), hard(2v4)
    save_interval: int = 200,
    log_interval: int = 50,
    save_dir: str = "models",
    device_str: str = "auto",
    visualize: bool = False,
    vis_interval: int = 50,
    use_attention: bool = True,
    use_comm: bool = False,
    seed: int = 0,
    # 兼容旧接口
    batch_size: int | None = None,
) -> None:
    """高层 MADDPG 协同训练 — 多环境 + 大 batch + 多步更新.

    GPU 利用率 = f(n_collect_envs × updates_per_step × coord_batch_size)
    不靠增大模型, 靠海量环境 + 大 batch + 高频梯度更新.

    Args:
        episodes: 训练回合数
        coord_batch_size: MADDPG 批量大小 (推荐 256-1024)
        n_collect_envs: 并行收集环境数 (推荐 16-64, 越大 buffer 越快满)
        updates_per_step: 每步梯度更新次数 (推荐 8-32, 越大 GPU 越忙)
        hidden_dim: 网络宽度 (256 即可, 不需要太大)
        buffer_capacity: 经验池容量 (n_collect_envs 大时应增大)
        warmup_steps: 开始训练前收集的经验数 (≥ coord_batch_size)
        seed: 随机种子 (多 GPU 训练时每个进程不同)
    """
    # 兼容旧 batch_size 参数
    if batch_size is not None and coord_batch_size in (256, 512):
        coord_batch_size = min(batch_size, 512)

    if device_str == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_str
    print(f"[设备] {device}")

    # 设置随机种子
    if seed > 0:
        torch.manual_seed(seed)
        np.random.seed(seed)

    N_AGENTS = 2
    NUM_SKILLS = 3
    PARAM_DIM = 2

    # 技能库始终在 CPU (PPO MLP 推理极快)
    skill_lib = build_skill_library(
        use_rule_skills=use_rule_skills, device="cpu",
    )

    # hidden_dim 保持小, GPU 利用率靠 env/batch/updates 扩展
    effective_hidden = hidden_dim

    # [关键修复] Option 级别的 gamma: 每个 option 跨越 skill_interval 步
    # 使用 gamma^skill_interval 作为 option MDP 的折扣因子
    gamma_step = 0.95
    gamma_option = gamma_step ** skill_interval  # 0.95^8 ≈ 0.663
    print(f"[Option MDP] gamma_step={gamma_step}, skill_interval={skill_interval}, "
          f"gamma_option={gamma_option:.4f}")

    trainer = MaddpgTrainer(
        n_agents=N_AGENTS,
        obs_dim=OBS_DIM,
        num_skills=NUM_SKILLS,
        param_dim=PARAM_DIM,
        lr_actor=1e-4,             # [v3] PPO底层稳定, 高层不宜太大
        lr_critic=3e-4,            # [v3] Critic 学习稍快
        gamma=gamma_option,        # option 级别的 gamma
        tau=0.01,
        gumbel_tau=1.0,
        gumbel_tau_decay=0.99999,  # 慢衰减
        gumbel_tau_min=0.3,        # 保持探索
        use_attention=use_attention,
        use_comm=use_comm,
        device=device,
        hidden_dim=effective_hidden,
    )
    buffer = MaddpgBuffer(
        capacity=buffer_capacity,
        n_agents=N_AGENTS,
        obs_dim=OBS_DIM,
        num_skills=NUM_SKILLS,
        param_dim=PARAM_DIM,
    )

    # 多环境收集器
    map_pool = ["classic_1", "classic_2", "classic_3"]
    collector = MultiEnvCollector(
        n_envs=n_collect_envs,
        n_agents=N_AGENTS,
        skill_interval=skill_interval,
        skill_lib=skill_lib,
        map_pool=map_pool,
        n_blue=n_blue,
        num_skills=NUM_SKILLS,
        max_steps=600,
        difficulty=difficulty,
    )
    collector.reset_all()

    episode_rewards: list[float] = []
    win_history: list[bool] = []
    kill_history: list[int] = []
    best_avg_reward = -float("inf")
    total_steps = 0
    total_updates = 0
    completed_episodes = 0

    # ---- 可视化监控器 ----
    monitor = None
    if visualize:
        from utils.train_monitor import TrainMonitor
        monitor = TrainMonitor(
            cell_size=32, fps=10, render_every=vis_interval, max_history=200
        )

    critic_type = "注意力" if use_attention else "拼接"
    comm_str = "是" if use_comm else "否"
    print("=" * 60)
    print("经典坦克大战 - 高层 MADDPG 协同训练 [v3 修复版]")
    print(f"  地图: {map_pool}, 红方: {N_AGENTS}, 难度: {difficulty}")
    print(f"  技能数: {NUM_SKILLS}, 参数维度: {PARAM_DIM}")
    print(f"  技能执行间隔: {skill_interval} 步")
    print(f"  Option γ: {gamma_option:.4f} (step γ={gamma_step}^{skill_interval})")
    print(f"  训练回合: {episodes}, 批量大小: {coord_batch_size}")
    print(f"  并行环境: {n_collect_envs}, 梯度更新/步: {updates_per_step}")
    print(f"  经验池容量: {buffer_capacity:,}, 热身: {warmup_steps}")
    print(f"  Gumbel τ: 1.0 → {trainer.gumbel_tau_min} (decay={trainer.gumbel_tau_decay})")
    print(f"  Critic: {critic_type}, 通讯: {comm_str}, 种子: {seed}")
    print(f"  网络宽度: {effective_hidden}, 设备: {device}")
    if visualize:
        print(f"  可视化: 每 {vis_interval} 回合")
    print("=" * 60)

    pbar = tqdm(total=episodes, desc="MADDPG训练")

    while completed_episodes < episodes:
        # === 1. 所有环境前进一步, 收集经验 ===
        new_exp = collector.step_all_envs(trainer, buffer, device)
        total_steps += n_collect_envs  # N 个环境各走一步

        # === 2. 多步梯度更新 (充分利用 GPU) ===
        if len(buffer) >= max(coord_batch_size, warmup_steps):
            for _ in range(updates_per_step):
                metrics = trainer.update(buffer, batch_size=coord_batch_size)
                total_updates += 1

        # === 3. 处理完成的 episode ===
        completed = collector.pop_completed_episodes()
        for ep_info in completed:
            completed_episodes += 1
            episode_rewards.append(ep_info["reward"])
            win_history.append(ep_info["win"])
            kill_history.append(ep_info.get("blue_killed", 0))

            if monitor is not None:
                monitor.on_episode_end(
                    completed_episodes, ep_info["reward"],
                    ep_info["info"], {},
                )

            pbar.update(1)

            # 日志
            if completed_episodes % log_interval == 0:
                n = min(log_interval, len(episode_rewards))
                avg_r = np.mean(episode_rewards[-n:])
                win_r = np.mean(win_history[-n:]) if win_history else 0.0
                avg_kills = np.mean(kill_history[-n:]) if kill_history else 0.0
                coop_str = ""
                if use_attention and trainer.get_cooperation_weights() is not None:
                    cw = trainer.get_cooperation_weights()
                    off_diag = [
                        cw[ii, jj]
                        for ii in range(N_AGENTS)
                        for jj in range(N_AGENTS) if ii != jj
                    ]
                    coop_str = f" | 协作: {np.mean(off_diag):.3f}"
                tqdm.write(
                    f"[Ep {completed_episodes:4d}] "
                    f"Avg R: {avg_r:7.2f} | "
                    f"Win: {win_r:.1%} | "
                    f"Kills: {avg_kills:.1f} | "
                    f"τ: {trainer.gumbel_tau:.3f} | "
                    f"Buf: {len(buffer):,} | "
                    f"Updates: {total_updates:,}"
                    f"{coop_str}"
                )

            # 保存
            if completed_episodes % save_interval == 0:
                trainer.save(os.path.join(save_dir, f"maddpg_ep{completed_episodes}"))
                n = min(save_interval, len(episode_rewards))
                avg_r = np.mean(episode_rewards[-n:])
                if avg_r > best_avg_reward:
                    best_avg_reward = avg_r
                    trainer.save(os.path.join(save_dir, "maddpg_best"))
                    tqdm.write(f"  [新最佳] Avg Reward = {best_avg_reward:.2f}")

            # 可视化
            if (monitor is not None
                    and completed_episodes % vis_interval == 0):
                vis_env = MultiTankTeamEnv(
                    map_name=map_name, n_red=N_AGENTS,
                    n_blue=n_blue, max_steps=400,
                    difficulty=difficulty,
                )
                ok = _run_visual_eval(
                    vis_env, trainer, skill_lib, monitor,
                    completed_episodes, N_AGENTS, skill_interval,
                )
                if not ok:
                    tqdm.write("[可视化] 用户关闭窗口, 继续纯文本训练")
                    monitor = None

            if completed_episodes >= episodes:
                break

    pbar.close()

    if monitor is not None:
        monitor.close()

    trainer.save(os.path.join(save_dir, "maddpg_final"))
    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"  总环境步数: {total_steps:,}")
    print(f"  总梯度更新: {total_updates:,}")
    print(f"  最终胜率: {np.mean(win_history[-50:]) if win_history else 0:.1%}")
    print(f"  最佳奖励: {best_avg_reward:.2f}")
    print(f"  Critic: {critic_type}, 设备: {device}")
    print("=" * 60)


# =====================================================================
#  多 GPU 人口训练
# =====================================================================

def train_population(
    episodes: int = 2000,
    n_gpus: int = 4,
    n_collect_envs: int = 32,
    updates_per_step: int = 16,
    coord_batch_size: int = 512,
    hidden_dim: int = 256,
    use_rule_skills: bool = True,
    map_name: str = "classic_1",
    n_blue: int = 4,
    difficulty: str = "easy",
    save_dir: str = "models",
    use_attention: bool = True,
    use_comm: bool = False,
) -> None:
    """4 块 GPU 同时训练, 不同随机种子, 取最佳模型.

    GPU 分配:
      GPU 0: seed=42,  n_collect_envs 个环境
      GPU 1: seed=123, n_collect_envs 个环境
      GPU 2: seed=456, n_collect_envs 个环境
      GPU 3: seed=789, n_collect_envs 个环境
    """
    import multiprocessing as mp

    actual_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    n_gpus = min(n_gpus, actual_gpus) if actual_gpus > 0 else 0

    if n_gpus == 0:
        print("[警告] 无 GPU, 使用 CPU 单进程训练")
        train(
            episodes=episodes, coord_batch_size=coord_batch_size,
            n_collect_envs=n_collect_envs, updates_per_step=updates_per_step,
            use_rule_skills=use_rule_skills, map_name=map_name,
            n_blue=n_blue, difficulty=difficulty, save_dir=save_dir,
            device_str="cpu",
            use_attention=use_attention, use_comm=use_comm,
        )
        return

    seeds = [42, 123, 456, 789]
    print("=" * 60)
    print(f"多 GPU 人口训练 — {n_gpus} 块 GPU 并行")
    for i in range(n_gpus):
        print(f"  GPU {i}: seed={seeds[i]}, {n_collect_envs} 并行环境")
    print(f"  每个 GPU: batch={coord_batch_size}, "
          f"updates/step={updates_per_step}")
    print(f"  训练完成后自动选择最佳模型")
    print("=" * 60)

    ctx = mp.get_context("spawn")
    processes = []

    for i in range(n_gpus):
        sub_save_dir = os.path.join(save_dir, f"gpu{i}")
        os.makedirs(sub_save_dir, exist_ok=True)
        p = ctx.Process(
            target=train,
            kwargs=dict(
                episodes=episodes,
                coord_batch_size=coord_batch_size,
                n_collect_envs=n_collect_envs,
                updates_per_step=updates_per_step,
                hidden_dim=hidden_dim,
                use_rule_skills=use_rule_skills,
                map_name=map_name,
                n_blue=n_blue,
                difficulty=difficulty,
                save_dir=sub_save_dir,
                device_str=f"cuda:{i}",
                use_attention=use_attention,
                use_comm=use_comm,
                seed=seeds[i],
            ),
            name=f"maddpg_gpu{i}",
        )
        p.start()
        print(f"  [PID {p.pid}] GPU {i} 训练已启动 (seed={seeds[i]})")
        processes.append(p)

    for p in processes:
        p.join()
        print(f"  [PID {p.pid}] {p.name} 完成 (exit={p.exitcode})")

    # 找最佳模型
    print("\n比较各 GPU 训练结果...")
    best_gpu = -1
    best_reward = -float("inf")
    for i in range(n_gpus):
        meta_path = os.path.join(save_dir, f"gpu{i}", "maddpg_best", "meta.pth")
        if os.path.exists(meta_path):
            meta = torch.load(meta_path, map_location="cpu")
            # meta 中有 update_count, 越多通常越好
            update_count = meta.get("update_count", 0)
            print(f"  GPU {i}: updates={update_count}")
            if update_count > best_reward:
                best_reward = update_count
                best_gpu = i

    if best_gpu >= 0:
        import shutil
        src = os.path.join(save_dir, f"gpu{best_gpu}", "maddpg_best")
        dst = os.path.join(save_dir, "maddpg_best")
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"\n最佳模型: GPU {best_gpu} → {dst}")

    print("\n多 GPU 人口训练完成!")


# =====================================================================
#  可视化评估辅助
# =====================================================================

def _run_visual_eval(
    env: MultiTankTeamEnv,
    trainer: MaddpgTrainer,
    skill_lib: dict,
    monitor,
    episode: int,
    n_agents: int,
    skill_interval: int,
) -> bool:
    """运行一次带协作可视化的评估回合."""
    from utils.visualize import TankRenderer

    if monitor.renderer is None:
        monitor.renderer = TankRenderer(cell_size=32, fps=10, engine=env.engine)

    obs = env.reset()
    ep_reward = 0.0
    current_skill_ids = [1] * n_agents
    current_params = [np.zeros(2, dtype=np.float32)] * n_agents

    for step in range(400):
        if step % skill_interval == 0:
            actions = trainer.select_actions(obs, deterministic=True)
            current_skill_ids = [a[0] for a in actions]
            current_params = [a[1] for a in actions]

        low_actions = []
        for i in range(n_agents):
            act = execute_skill(
                skill_lib, current_skill_ids[i], obs[i], current_params[i]
            )
            low_actions.append(act)

        next_obs, reward, done, info = env.step(low_actions)
        ep_reward += reward
        obs = next_obs

        coop_weights = trainer.get_cooperation_weights()
        skill_info = list(zip(current_skill_ids, current_params))

        running = monitor.renderer.render(
            env.engine, skill_info=skill_info,
            episode=episode, step=step, reward=ep_reward,
            cooperation_weights=coop_weights, training_mode=True,
        )

        monitor._draw_training_overlay(env.engine, episode)
        import pygame
        pygame.display.flip()

        if not running:
            return False
        if done:
            break

    return True


# =====================================================================
#  命令行入口
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MADDPG 协同训练 (多环境 + 多 GPU)",
    )
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--skill_interval", type=int, default=8)
    parser.add_argument("--coord_batch_size", type=int, default=256,
                        help="MADDPG 批量大小 (推荐 128-256)")
    parser.add_argument("--buffer_capacity", type=int, default=200_000)
    parser.add_argument("--warmup_steps", type=int, default=300)
    parser.add_argument("--n_collect_envs", type=int, default=8,
                        help="并行收集环境数 (推荐 4-8)")
    parser.add_argument("--updates_per_step", type=int, default=4,
                        help="每步梯度更新次数 (推荐 2-4)")
    parser.add_argument("--use_rule_skills", action="store_true")
    parser.add_argument("--map_name", type=str, default="classic_1")
    parser.add_argument("--n_blue", type=int, default=4)
    parser.add_argument("--difficulty", type=str, default="easy",
                        choices=["easy", "medium", "hard"],
                        help="难度: easy(2v2), medium(2v3), hard(2v4)")
    parser.add_argument("--save_interval", type=int, default=200)
    parser.add_argument("--save_dir", type=str, default="models")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--vis_interval", type=int, default=50)
    parser.add_argument("--use_attention", action="store_true", default=True)
    parser.add_argument("--no_attention", action="store_true")
    parser.add_argument("--use_comm", action="store_true")
    parser.add_argument("--population", action="store_true",
                        help="多 GPU 人口训练 (每块 GPU 一个独立训练)")
    args = parser.parse_args()

    use_attention = args.use_attention and not args.no_attention

    if args.population:
        train_population(
            episodes=args.episodes,
            coord_batch_size=args.coord_batch_size,
            n_collect_envs=args.n_collect_envs,
            updates_per_step=args.updates_per_step,
            use_rule_skills=args.use_rule_skills,
            map_name=args.map_name,
            n_blue=args.n_blue,
            difficulty=args.difficulty,
            save_dir=args.save_dir,
            use_attention=use_attention,
            use_comm=args.use_comm,
        )
    else:
        train(
            episodes=args.episodes,
            skill_interval=args.skill_interval,
            coord_batch_size=args.coord_batch_size,
            buffer_capacity=args.buffer_capacity,
            warmup_steps=args.warmup_steps,
            n_collect_envs=args.n_collect_envs,
            updates_per_step=args.updates_per_step,
            use_rule_skills=args.use_rule_skills,
            map_name=args.map_name,
            n_blue=args.n_blue,
            difficulty=args.difficulty,
            save_interval=args.save_interval,
            save_dir=args.save_dir,
            device_str=args.device,
            visualize=args.visualize,
            vis_interval=args.vis_interval,
            use_attention=use_attention,
            use_comm=args.use_comm,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
