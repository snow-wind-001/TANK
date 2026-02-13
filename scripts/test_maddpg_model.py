#!/usr/bin/env python3
"""测试 MADDPG 模型胜率.

加载已保存的 MADDPG 检查点, 在不同难度下评估胜率。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from envs.multi_tank_env import MultiTankTeamEnv, OBS_DIM
from maddpg.core import MaddpgTrainer
from maddpg.train_coord import build_skill_library


def load_trainer(model_dir: str, device: str = "cpu") -> MaddpgTrainer:
    """从检查点加载 MaddpgTrainer."""
    import torch

    # 先读 meta 获取配置信息
    meta_path = os.path.join(model_dir, "meta.pth")
    use_comm = False
    use_attention = True
    if os.path.exists(meta_path):
        meta = torch.load(meta_path, map_location="cpu", weights_only=False)
        use_comm = meta.get("use_comm", False)
        use_attention = meta.get("use_attention", True)
        print(f"  meta: {meta}")

    # 通过 actor 权重的第一层形状自动检测是否使用了 CommModule
    actor_path = os.path.join(model_dir, "actor_0.pth")
    if os.path.exists(actor_path):
        actor_state = torch.load(actor_path, map_location="cpu", weights_only=True)
        first_layer_key = "shared.0.weight"
        if first_layer_key in actor_state:
            in_dim = actor_state[first_layer_key].shape[1]
            if in_dim > OBS_DIM:
                use_comm = True
                print(f"  [自动检测] Actor 输入维度={in_dim} > obs_dim={OBS_DIM}, 启用 CommModule")
            else:
                use_comm = False
                print(f"  [自动检测] Actor 输入维度={in_dim}, 未使用 CommModule")

    # 检查 comm.pth 是否存在
    comm_path = os.path.join(model_dir, "comm.pth")
    has_comm_weights = os.path.exists(comm_path)
    if use_comm and not has_comm_weights:
        print(f"  [警告] Actor 使用了 CommModule 但 comm.pth 不存在!")
        print(f"  CommModule 将使用随机权重, 结果可能不准确。")
        print(f"  建议: 重新训练 MADDPG (save/load 已修复, 会自动保存 comm.pth)")
    elif has_comm_weights:
        print(f"  发现 comm.pth, 将加载通讯模块权重")

    trainer = MaddpgTrainer(
        n_agents=2,
        obs_dim=OBS_DIM,
        num_skills=3,
        param_dim=2,
        use_attention=use_attention,
        use_comm=use_comm,
        device=device,
        hidden_dim=256,
    )
    trainer.load(model_dir)
    return trainer


def evaluate(
    trainer: MaddpgTrainer,
    skill_lib: dict,
    difficulty: str = "easy",
    n_eval: int = 100,
    max_steps: int = 600,
    skill_interval: int = 4,
) -> dict:
    """在指定难度下评估胜率."""
    wins = 0
    losses = 0
    draws = 0
    total_kills = 0
    total_reward = 0.0

    for ep in range(n_eval):
        env = MultiTankTeamEnv(
            map_name="classic_1",
            n_red=2,
            n_blue=4,
            max_steps=max_steps,
            difficulty=difficulty,
        )
        obs_list = env.reset()

        skill_ids = [1, 2]  # 初始: attack, defend
        params = [np.zeros(2, dtype=np.float32)] * 2
        ep_reward = 0.0

        for step in range(max_steps):
            # 高层决策
            if step % skill_interval == 0:
                actions_hl = trainer.select_actions(obs_list, deterministic=True)
                for i in range(len(obs_list)):
                    skill_ids[i] = actions_hl[i][0]
                    params[i] = actions_hl[i][1]

            # 底层执行
            low_actions = []
            for i in range(len(obs_list)):
                sid = skill_ids[i]
                skill = skill_lib[sid]
                action = skill.act(obs_list[i], params[i], deterministic=True)
                low_actions.append(action)

            obs_list, reward, done, info = env.step(low_actions)
            ep_reward += reward

            if done:
                break

        total_reward += ep_reward
        is_win = info.get("win", False)
        base_alive = info.get("base_alive", True)
        red_alive = info.get("red_alive", 0)

        if is_win:
            wins += 1
        elif not base_alive or red_alive == 0:
            losses += 1
        else:
            draws += 1

        total_kills += info.get("blue_killed", 0)

        if (ep + 1) % 20 == 0:
            print(f"  进度: {ep + 1}/{n_eval}, 当前胜率: {wins / (ep + 1):.1%}")

    return {
        "difficulty": difficulty,
        "win_rate": wins / n_eval,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "avg_kills": total_kills / n_eval,
        "avg_reward": total_reward / n_eval,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试 MADDPG 模型")
    parser.add_argument("--model_dir", type=str, default="models/maddpg_best",
                        help="模型目录")
    parser.add_argument("--difficulty", type=str, default="all",
                        choices=["easy", "medium", "hard", "all"])
    parser.add_argument("--n_eval", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--use_rule_skills", action="store_true",
                        help="使用规则技能替代 PPO")
    parser.add_argument("--skill_interval", type=int, default=4)
    args = parser.parse_args()

    print(f"=" * 60)
    print(f"  MADDPG 模型测试")
    print(f"  模型: {args.model_dir}")
    print(f"  设备: {args.device}")
    print(f"  技能: {'规则' if args.use_rule_skills else 'PPO'}")
    print(f"=" * 60)

    print(f"\n加载模型: {args.model_dir}")
    trainer = load_trainer(args.model_dir, args.device)

    print(f"\n加载技能库...")
    skill_lib = build_skill_library(
        use_rule_skills=args.use_rule_skills,
        device="cpu",
    )

    difficulties = ["easy", "medium", "hard"] if args.difficulty == "all" else [args.difficulty]

    results = []
    for diff in difficulties:
        print(f"\n{'=' * 60}")
        print(f"  评估: {diff} ({args.n_eval} 局)")
        print(f"{'=' * 60}")

        r = evaluate(
            trainer, skill_lib, difficulty=diff,
            n_eval=args.n_eval, skill_interval=args.skill_interval,
        )
        results.append(r)

        print(f"\n  胜率:     {r['win_rate']:.1%} ({r['wins']}/{args.n_eval})")
        print(f"  败率:     {r['losses'] / args.n_eval:.1%}")
        print(f"  平局:     {r['draws'] / args.n_eval:.1%}")
        print(f"  平均击杀: {r['avg_kills']:.2f}")
        print(f"  平均奖励: {r['avg_reward']:.2f}")

    if len(results) > 1:
        print(f"\n{'=' * 70}")
        print(f"  汇总 ({args.model_dir})")
        print(f"{'=' * 70}")
        for r in results:
            print(f"  {r['difficulty']:8s} | 胜率 {r['win_rate']:.1%} | "
                  f"击杀 {r['avg_kills']:.2f} | 奖励 {r['avg_reward']:.1f}")
        print(f"{'=' * 70}")
