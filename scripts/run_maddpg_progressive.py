#!/usr/bin/env python3
"""MADDPG 渐进难度训练.

阶段 1: easy  (2v2), 3000 episodes → 目标胜率 > 30%
阶段 2: medium (2v3), 5000 episodes → 目标胜率 > 15%
阶段 3: hard  (2v4), 10000 episodes → 目标胜率 > 5%

使用训练好的 PPO 底层技能, 不使用规则技能.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from maddpg.train_coord import train


# 渐进训练阶段
STAGES = [
    {
        "difficulty": "easy",
        "episodes": 3000,
        "target_win_rate": 0.30,
        "label": "Stage 1: Easy (2v2)",
    },
    {
        "difficulty": "medium",
        "episodes": 5000,
        "target_win_rate": 0.15,
        "label": "Stage 2: Medium (2v3)",
    },
    {
        "difficulty": "hard",
        "episodes": 10000,
        "target_win_rate": 0.05,
        "label": "Stage 3: Hard (2v4)",
    },
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--start_stage", type=int, default=0,
                        help="从第几阶段开始 (0-based)")
    args = parser.parse_args()

    for idx, stage in enumerate(STAGES):
        if idx < args.start_stage:
            continue

        print(f"\n{'#' * 70}")
        print(f"  {stage['label']}")
        print(f"  difficulty={stage['difficulty']}, episodes={stage['episodes']}")
        print(f"  目标胜率: {stage['target_win_rate']:.0%}")
        print(f"{'#' * 70}\n")

        # 第一阶段从零开始, 后续阶段加载上一阶段的模型
        resume = "models/maddpg_latest" if idx > 0 else None

        train(
            episodes=stage["episodes"],
            skill_interval=4,
            use_rule_skills=False,  # 使用 PPO 技能
            use_attention=True,
            use_comm=True,
            n_collect_envs=32,
            coord_batch_size=512,
            updates_per_step=8,
            difficulty=stage["difficulty"],
            save_dir="models",
            save_interval=500,
            device_str=args.device,
        )

        print(f"\n  {stage['label']} 训练完成!")
        print(f"{'#' * 70}\n")

    print("\n所有 MADDPG 渐进训练完成!")
