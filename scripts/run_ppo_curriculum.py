#!/usr/bin/env python3
"""PPO 课程学习并行训练启动器.

在 3 个 GPU 上同时训练 attack/defend/navigate 技能,
每个技能经过 C1→C2→C3 三个课程阶段.
"""
from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path

# 确保项目根目录在路径中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from skills.train_skills import _train_skill_curriculum


def train_on_gpu(skill: str, gpu_id: int) -> None:
    """在指定 GPU 上执行课程学习训练."""
    _train_skill_curriculum(
        skill_type=skill,
        save_dir="skills/models",
        device=f"cuda:{gpu_id}",
        n_envs=32,
        n_steps=2048,
        visualize=False,
    )


if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    skills_gpus = [("attack", 0), ("defend", 1), ("navigate", 2)]
    procs = []
    for skill, gpu in skills_gpus:
        p = ctx.Process(
            target=train_on_gpu,
            args=(skill, gpu),
            name=f"train_{skill}_gpu{gpu}",
        )
        p.start()
        print(f"[PID {p.pid}] {skill} → cuda:{gpu}")
        procs.append(p)

    for p in procs:
        p.join()
        status = "成功" if p.exitcode == 0 else f"失败(code={p.exitcode})"
        print(f"[PID {p.pid}] {p.name} — {status}")

    all_ok = all(p.exitcode == 0 for p in procs)
    if all_ok:
        print("\n✓ 所有课程学习训练完成!")
    else:
        print("\n⚠ 部分训练失败, 请检查日志")
        sys.exit(1)
