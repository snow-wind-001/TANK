#!/usr/bin/env python3
"""多 GPU 并行 MADDPG 训练 (高效版).

核心优化 (对比 v1):
  1. 批量推理: 所有环境的观测打包为单次 GPU forward, 而非逐个推理
  2. 多线程 env.step: 用 ThreadPoolExecutor 并行执行环境步进
  3. 更大吞吐量: 默认 200 envs/GPU = 600 总并行环境
  4. 更频繁梯度更新: batch=4096, updates_per_step=64

架构:
  Worker0 (cuda:1, 200 envs) ─┐
  Worker1 (cuda:2, 200 envs) ─┼─→ 共享 Queue → Trainer (cuda:1) → 权重同步
  Worker2 (cuda:3, 200 envs) ─┘

数据流: 批量推理 → 多线程 step → 收集经验 → Queue → Trainer 梯度更新
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from multiprocessing import Process, Queue, Event, Value
from ctypes import c_int64

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def collector_worker(
    worker_id: int,
    gpu_id: int,
    n_envs: int,
    difficulty: str,
    skill_interval: int,
    use_rule_skills: bool,
    exp_queue: Queue,
    weight_dir: str,
    stop_event: Event,
    episodes_counter: Value,
    total_episodes: int,
    wins_counter: Value,
    kills_counter: Value,
) -> None:
    """高效经验收集 Worker.

    关键优化:
      1. select_actions_batch: 所有 n_envs 的观测堆叠为 batch 一次推理
      2. 纯顺序 env.step (避免 GIL 争用, CPU 密集代码最快方式)
      3. 减少 CPU→GPU 数据搬运次数
    """
    device = f"cuda:{gpu_id}"
    print(f"[Worker {worker_id}] 启动: GPU={gpu_id}, envs={n_envs}, "
          f"difficulty={difficulty}")

    from envs.multi_tank_env import MultiTankTeamEnv, OBS_DIM
    from maddpg.core import MaddpgTrainer
    from maddpg.train_coord import build_skill_library, execute_skill

    N_AGENTS = 2
    NUM_SKILLS = 3
    PARAM_DIM = 2

    # 技能库 (CPU)
    skill_lib = build_skill_library(use_rule_skills=use_rule_skills, device="cpu")

    # 推理用 Trainer
    trainer = MaddpgTrainer(
        n_agents=N_AGENTS, obs_dim=OBS_DIM, num_skills=NUM_SKILLS,
        param_dim=PARAM_DIM, use_attention=True, use_comm=True,
        device=device, hidden_dim=256,
    )
    if os.path.exists(weight_dir):
        trainer.load(weight_dir)
        print(f"[Worker {worker_id}] 已加载初始权重: {weight_dir}")

    # 设为 eval 模式以加速推理
    for actor in trainer.actors:
        actor.eval()
    if trainer.comm is not None:
        trainer.comm.eval()

    # 创建环境
    map_pool = ["classic_1", "classic_2", "classic_3"]
    envs = []
    for i in range(n_envs):
        env = MultiTankTeamEnv(
            map_name=map_pool[i % len(map_pool)],
            n_red=N_AGENTS, n_blue=4, max_steps=600,
            difficulty=difficulty,
        )
        envs.append(env)

    # 环境状态 (全部用 numpy array 管理)
    obs_all: list[list[np.ndarray]] = [env.reset() for env in envs]
    done_all = np.zeros(n_envs, dtype=bool)
    step_all = np.zeros(n_envs, dtype=np.int32)
    acc_reward_all = np.zeros(n_envs, dtype=np.float32)
    prev_obs_all: list[list[np.ndarray]] = [
        [o.copy() for o in obs] for obs in obs_all
    ]
    skill_ids_all = np.ones((n_envs, N_AGENTS), dtype=np.int32)
    skill_params_all = np.zeros((n_envs, N_AGENTS, PARAM_DIM), dtype=np.float32)

    local_episodes = 0
    weight_reload_interval = 50  # 更频繁重载权重

    while not stop_event.is_set():
        if episodes_counter.value >= total_episodes:
            break

        # ============ Phase 1: 重置已结束的环境 ============
        done_indices = np.where(done_all)[0]
        for i in done_indices:
            envs[i] = MultiTankTeamEnv(
                map_name=map_pool[i % len(map_pool)],
                n_red=N_AGENTS, n_blue=4, max_steps=600,
                difficulty=difficulty,
            )
            obs_all[i] = envs[i].reset()
            done_all[i] = False
            step_all[i] = 0
            acc_reward_all[i] = 0.0
            prev_obs_all[i] = [o.copy() for o in obs_all[i]]
            skill_ids_all[i] = 1
            skill_params_all[i] = 0.0

        # ============ Phase 2: 批量高层决策 ============
        # 找出需要做高层决策的环境
        active_mask = ~done_all
        need_decision = active_mask & (step_all % skill_interval == 0)
        decision_indices = np.where(need_decision)[0]

        if len(decision_indices) > 0:
            # 对需要存储经验的环境 (step > 0), 先入队
            for i in decision_indices:
                if step_all[i] > 0:
                    sp_list = []
                    p_list = []
                    for j in range(N_AGENTS):
                        oh = np.zeros(NUM_SKILLS, np.float32)
                        oh[skill_ids_all[i, j]] = 1.0
                        sp_list.append(oh)
                        p_list.append(skill_params_all[i, j].copy())
                    exp_queue.put((
                        [o.copy() for o in prev_obs_all[i]],
                        sp_list, p_list,
                        float(acc_reward_all[i]),
                        [o.copy() for o in obs_all[i]],
                        False,
                    ))
                    acc_reward_all[i] = 0.0
                    prev_obs_all[i] = [o.copy() for o in obs_all[i]]

            # 批量推理: 一次 GPU forward 处理所有需要决策的环境
            batch_obs = [obs_all[i] for i in decision_indices]
            batch_actions = trainer.select_actions_batch(
                batch_obs, deterministic=False,
            )

            # 分发结果
            for idx, i in enumerate(decision_indices):
                for j in range(N_AGENTS):
                    skill_ids_all[i, j] = batch_actions[idx][j][0]
                    skill_params_all[i, j] = batch_actions[idx][j][1]

        # ============ Phase 3: 底层技能执行 (CPU, 可并行) ============
        active_indices = np.where(active_mask)[0]
        low_actions_all: dict[int, list[int]] = {}
        for i in active_indices:
            low_actions = []
            for j in range(N_AGENTS):
                act = execute_skill(
                    skill_lib, skill_ids_all[i, j],
                    obs_all[i][j], skill_params_all[i, j],
                )
                low_actions.append(act)
            low_actions_all[i] = low_actions

        # ============ Phase 4: 顺序 env.step() (GIL 下最快) ============
        for i in active_indices:
            next_obs, reward, done, info = envs[i].step(low_actions_all[i])
            acc_reward_all[i] += reward
            step_all[i] += 1
            obs_all[i] = next_obs

            if done:
                # 存储最终转移
                sp_list = []
                p_list = []
                for j in range(N_AGENTS):
                    oh = np.zeros(NUM_SKILLS, np.float32)
                    oh[skill_ids_all[i, j]] = 1.0
                    sp_list.append(oh)
                    p_list.append(skill_params_all[i, j].copy())
                exp_queue.put((
                    [o.copy() for o in prev_obs_all[i]],
                    sp_list, p_list,
                    float(acc_reward_all[i]),
                    [o.copy() for o in next_obs],
                    True,
                ))
                done_all[i] = True

                is_win = info.get("win", False)
                with episodes_counter.get_lock():
                    episodes_counter.value += 1
                if is_win:
                    with wins_counter.get_lock():
                        wins_counter.value += 1
                with kills_counter.get_lock():
                    kills_counter.value += info.get("blue_killed", 0)

                local_episodes += 1

                # 更频繁重载权重
                if local_episodes % weight_reload_interval == 0:
                    sync_path = os.path.join("models", "maddpg_latest")
                    if os.path.exists(sync_path):
                        try:
                            trainer.load(sync_path)
                            for actor in trainer.actors:
                                actor.eval()
                            if trainer.comm is not None:
                                trainer.comm.eval()
                        except Exception:
                            pass

    print(f"[Worker {worker_id}] 结束, 收集 {local_episodes} episodes")


def trainer_loop(
    gpu_id: int,
    exp_queue: Queue,
    resume_dir: str,
    save_dir: str,
    save_interval: int,
    total_episodes: int,
    episodes_counter: Value,
    wins_counter: Value,
    kills_counter: Value,
    stop_event: Event,
    batch_size: int = 4096,
    updates_per_step: int = 64,
    buffer_capacity: int = 2_000_000,
    log_interval: int = 50,
    max_pull_per_iter: int = 2000,
) -> None:
    """训练器主循环: 从 Queue 取经验, 更新模型, 定期保存.

    优化:
      - 更大 batch_size (4096) 充分利用 GPU 计算
      - 更频繁梯度更新 (64 per step)
      - 更大 buffer (2M) 降低相关性
    """
    device = f"cuda:{gpu_id}"
    print(f"[Trainer] 启动: GPU={gpu_id}, batch={batch_size}, "
          f"updates/step={updates_per_step}, buffer_cap={buffer_capacity}")

    from envs.multi_tank_env import OBS_DIM
    from maddpg.core import MaddpgTrainer, MaddpgBuffer

    N_AGENTS = 2
    NUM_SKILLS = 3
    PARAM_DIM = 2
    skill_interval = 4
    gamma_option = 0.95 ** skill_interval

    trainer = MaddpgTrainer(
        n_agents=N_AGENTS, obs_dim=OBS_DIM, num_skills=NUM_SKILLS,
        param_dim=PARAM_DIM, lr_actor=1e-4, lr_critic=3e-4,
        gamma=gamma_option, tau=0.01,
        gumbel_tau=1.0, gumbel_tau_decay=0.99999, gumbel_tau_min=0.3,
        use_attention=True, use_comm=True,
        device=device, hidden_dim=256,
    )
    if resume_dir and os.path.exists(resume_dir):
        trainer.load(resume_dir)
        print(f"[Trainer] 已恢复权重: {resume_dir} "
              f"(updates={trainer._update_count}, tau={trainer.gumbel_tau:.4f})")

    buffer = MaddpgBuffer(
        capacity=buffer_capacity, n_agents=N_AGENTS,
        obs_dim=OBS_DIM, num_skills=NUM_SKILLS, param_dim=PARAM_DIM,
    )

    os.makedirs(save_dir, exist_ok=True)
    latest_dir = os.path.join(save_dir, "maddpg_latest")
    best_dir = os.path.join(save_dir, "maddpg_best")
    best_win_rate = 0.0
    total_updates = 0
    last_log_ep = 0
    last_save_ep = 0
    start_time = time.time()

    # 初始保存权重供 workers 加载
    trainer.save(latest_dir)

    print(f"[Trainer] 等待经验...")
    while episodes_counter.value < total_episodes and not stop_event.is_set():
        # 从 Queue 批量取经验
        n_pulled = 0
        while not exp_queue.empty() and n_pulled < max_pull_per_iter:
            try:
                exp = exp_queue.get_nowait()
                obs_list, sp_list, p_list, reward, next_obs, done = exp
                buffer.add(obs_list, sp_list, p_list, reward, next_obs, done)
                n_pulled += 1
            except Exception:
                break

        # 梯度更新 — 关键计算密集区
        if len(buffer) >= max(batch_size, 1000):
            for _ in range(updates_per_step):
                metrics = trainer.update(buffer, batch_size=batch_size)
                total_updates += 1
        else:
            time.sleep(0.01)

        current_ep = episodes_counter.value

        # 日志
        if current_ep >= last_log_ep + log_interval and current_ep > 0:
            elapsed = time.time() - start_time
            win_rate = wins_counter.value / max(current_ep, 1)
            avg_kills = kills_counter.value / max(current_ep, 1)
            eps_per_sec = current_ep / elapsed
            q_size = exp_queue.qsize() if hasattr(exp_queue, 'qsize') else -1
            print(f"[Ep {current_ep}/{total_episodes}] "
                  f"胜率={win_rate:.1%} | 击杀={avg_kills:.2f} | "
                  f"updates={total_updates} | tau={trainer.gumbel_tau:.3f} | "
                  f"buffer={len(buffer)} | queue={q_size} | "
                  f"{eps_per_sec:.1f} ep/s | elapsed={elapsed:.0f}s")
            last_log_ep = current_ep

        # 保存
        if current_ep >= last_save_ep + save_interval and current_ep > 0:
            trainer.save(latest_dir)
            ep_dir = os.path.join(save_dir, f"maddpg_ep{current_ep}")
            trainer.save(ep_dir)

            win_rate = wins_counter.value / max(current_ep, 1)
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                trainer.save(best_dir)
                print(f"[Trainer] 新最佳! 胜率={win_rate:.1%}")

            last_save_ep = current_ep

    # 最终保存
    final_dir = os.path.join(save_dir, "maddpg_final")
    trainer.save(final_dir)
    trainer.save(latest_dir)
    stop_event.set()
    elapsed = time.time() - start_time
    win_rate = wins_counter.value / max(episodes_counter.value, 1)
    print(f"\n[Trainer] 训练完成! {episodes_counter.value} episodes, "
          f"{total_updates} updates, 胜率={win_rate:.1%}, 耗时={elapsed:.0f}s")


if __name__ == "__main__":
    import argparse
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="多 GPU MADDPG 训练 (高效版)")
    parser.add_argument("--gpus", type=str, default="1,2,3",
                        help="使用的 GPU 编号, 逗号分隔")
    parser.add_argument("--workers_per_gpu", type=int, default=4,
                        help="每张 GPU 启动几个 worker 进程 (默认 4)")
    parser.add_argument("--envs_per_worker", type=int, default=200,
                        help="每个 worker 的并行环境数 (默认 200)")
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--difficulty", type=str, default="easy",
                        choices=["easy", "medium", "hard"])
    parser.add_argument("--resume", type=str, default="models/maddpg_best",
                        help="续训模型目录")
    parser.add_argument("--save_dir", type=str, default="models")
    parser.add_argument("--batch_size", type=int, default=4096,
                        help="训练 batch size (默认 4096)")
    parser.add_argument("--updates_per_step", type=int, default=64,
                        help="每次 pull 后的梯度更新次数 (默认 64)")
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--use_rule_skills", action="store_true")
    args = parser.parse_args()

    gpu_ids = [int(g) for g in args.gpus.split(",")]
    n_gpus = len(gpu_ids)
    n_workers = n_gpus * args.workers_per_gpu
    total_envs = n_workers * args.envs_per_worker

    print("=" * 70)
    print("  多 GPU MADDPG 并行训练 (高效版 v3 - 多 Worker)")
    print(f"  GPUs: {gpu_ids} ({n_gpus} 张)")
    print(f"  每 GPU Worker 数: {args.workers_per_gpu}")
    print(f"  总 Worker 数: {n_workers}")
    print(f"  每 Worker 环境数: {args.envs_per_worker}")
    print(f"  总并行环境: {total_envs}")
    print(f"  CPU 核心: {os.cpu_count()}, Worker 进程: {n_workers}")
    print(f"  目标 episodes: {args.episodes}")
    print(f"  难度: {args.difficulty}")
    print(f"  续训: {args.resume}")
    print(f"  训练 batch_size: {args.batch_size}")
    print(f"  updates_per_step: {args.updates_per_step}")
    print("=" * 70)

    # 共享状态
    exp_queue = mp.Queue(maxsize=100000)
    stop_event = mp.Event()
    episodes_counter = mp.Value(c_int64, 0)
    wins_counter = mp.Value(c_int64, 0)
    kills_counter = mp.Value(c_int64, 0)

    # 启动 Collector Workers: 每 GPU 多个 worker
    workers = []
    worker_idx = 0
    for gid in gpu_ids:
        for w in range(args.workers_per_gpu):
            p = mp.Process(
                target=collector_worker,
                args=(
                    worker_idx, gid, args.envs_per_worker,
                    args.difficulty, 4,
                    args.use_rule_skills, exp_queue, args.resume,
                    stop_event, episodes_counter, args.episodes,
                    wins_counter, kills_counter,
                ),
            )
            p.daemon = True
            p.start()
            workers.append(p)
            print(f"  Worker {worker_idx} (cuda:{gid}) 已启动, PID={p.pid}")
            worker_idx += 1

    # Trainer 在主进程
    print(f"\n  Trainer 在 cuda:{gpu_ids[0]} 上运行")
    print("=" * 70)

    try:
        trainer_loop(
            gpu_id=gpu_ids[0],
            exp_queue=exp_queue,
            resume_dir=args.resume,
            save_dir=args.save_dir,
            save_interval=args.save_interval,
            total_episodes=args.episodes,
            episodes_counter=episodes_counter,
            wins_counter=wins_counter,
            kills_counter=kills_counter,
            stop_event=stop_event,
            batch_size=args.batch_size,
            updates_per_step=args.updates_per_step,
        )
    except KeyboardInterrupt:
        print("\n[中断] 正在停止...")
    finally:
        stop_event.set()
        for p in workers:
            p.join(timeout=10)
            if p.is_alive():
                p.kill()
        print("所有进程已停止。")
