#!/usr/bin/env python3
"""PPO 技能在 MultiTankTeamEnv 中的交叉验证.

测试组合:
  1. 双 PPO-attack 在 2v2 easy
  2. PPO-attack + PPO-defend 在 2v2 easy
  3. PPO-attack + PPO-defend 在 2v3 medium
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.multi_tank_env import MultiTankTeamEnv
from envs.single_tank_env import SingleTankSkillEnv


def load_ppo_with_vecnorm(skill_type: str, model_dir: str = "skills/models"):
    """加载 PPO 模型和对应的 VecNormalize."""
    import os

    model_path = os.path.join(model_dir, f"{skill_type}_skill")
    vecnorm_path = os.path.join(model_dir, f"{skill_type}_vecnorm.pkl")

    model = PPO.load(model_path, device="cpu")
    print(f"  加载 {skill_type} PPO: {model_path}")

    vec_normalize = None
    if os.path.exists(vecnorm_path):
        dummy = DummyVecEnv([lambda: SingleTankSkillEnv(
            skill_type=skill_type, map_name="classic_1",
            max_steps=300, n_enemies=1,
        )])
        vec_normalize = VecNormalize.load(vecnorm_path, dummy)
        vec_normalize.training = False
        vec_normalize.norm_reward = False
        print(f"  加载 VecNormalize: {vecnorm_path}")
    else:
        print(f"  ⚠ 无 VecNormalize: {vecnorm_path}")

    return model, vec_normalize


def multi_to_single_obs(
    multi_obs: np.ndarray,
    skill_type: str,
) -> np.ndarray:
    """MultiTankTeamEnv 32-dim → SingleTankSkillEnv 29-dim 转换.

    Multi obs 布局 (32 dim):
      [0:2]   self_x, self_y
      [2:6]   dir_oh (4)
      [6:9]   hp, cd, has_b
      [9:13]  ray_wall (4)
      [13:17] ray_enemy (4)
      [17:20] e1_dx, e1_dy, e1_hp
      [20:23] e2_dx, e2_dy, e2_hp
      [23:27] ally_dx, ally_dy, ally_hp, ally_alive  ← multi 独有
      [27:30] base_dx, base_dy, base_alive
      [30:32] n_enemies, n_allies

    Single obs 布局 (29 dim):
      [0:23]  同 multi [0:23]
      [23:26] base_dx, base_dy, base_alive  ← from multi [27:30]
      [26]    n_alive                       ← from multi [30]
      [27:29] target_dx, target_dy          ← 需要推算
    """
    shared = multi_obs[:23]                  # 0-22: 自身 + 射线 + 敌人
    base = multi_obs[27:30]                  # base_dx, base_dy, base_alive
    n_alive = multi_obs[30:31]               # n_enemies

    # target_dx, target_dy:
    #   attack → 最近敌人位置 = e1_dx, e1_dy (已在 obs[17:19])
    #   defend → 基地位置 = base_dx, base_dy
    #   navigate → 无特定目标, 用 (0, 0)
    if skill_type == "attack":
        target = multi_obs[17:19]            # e1_dx, e1_dy
    elif skill_type == "defend":
        target = multi_obs[27:29]            # base_dx, base_dy
    else:  # navigate
        target = np.array([0.0, 0.0], dtype=np.float32)

    return np.concatenate([shared, base, n_alive, target]).astype(np.float32)


def predict_with_vecnorm(
    model, obs_29: np.ndarray, vec_normalize
) -> int:
    """使用 VecNormalize 归一化后预测."""
    if vec_normalize is not None:
        obs_norm = vec_normalize.normalize_obs(obs_29.reshape(1, -1))
    else:
        obs_norm = obs_29.reshape(1, -1)
    action, _ = model.predict(obs_norm, deterministic=True)
    return int(action)


def run_cross_validation(
    skill_assignments: list[str],
    difficulty: str = "easy",
    n_eval: int = 100,
    max_steps: int = 600,
) -> dict:
    """运行交叉验证.

    Args:
        skill_assignments: 每个红方坦克的技能类型 (如 ["attack", "attack"])
        difficulty: 环境难度
        n_eval: 评估局数
        max_steps: 最大步数

    Returns:
        统计结果字典
    """
    n_agents = len(skill_assignments)

    # 加载所有需要的模型 (去重)
    unique_skills = set(skill_assignments)
    models = {}
    vecnorms = {}
    for skill in unique_skills:
        models[skill], vecnorms[skill] = load_ppo_with_vecnorm(skill)

    label = "+".join(skill_assignments)
    print(f"\n{'=' * 60}")
    print(f"  交叉验证: {label} | {difficulty} | {n_eval} 局")
    print(f"{'=' * 60}")

    wins = 0
    losses = 0
    draws = 0
    total_kills = 0
    total_red_alive = 0

    for ep in range(n_eval):
        env = MultiTankTeamEnv(
            map_name="classic_1",
            n_red=n_agents,
            n_blue=4,
            max_steps=max_steps,
            difficulty=difficulty,
        )
        obs_all = env.reset()

        for step in range(max_steps):
            actions = []
            for i in range(n_agents):
                skill = skill_assignments[i]
                obs_32 = obs_all[i]
                obs_29 = multi_to_single_obs(obs_32, skill)
                action = predict_with_vecnorm(
                    models[skill], obs_29, vecnorms[skill]
                )
                actions.append(action)

            obs_all, rewards, done, info = env.step(actions)
            if done:
                break

        # 统计
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
        total_red_alive += red_alive

    win_rate = wins / n_eval
    print(f"  胜率:     {win_rate:.1%} ({wins}/{n_eval})")
    print(f"  败率:     {losses / n_eval:.1%} ({losses})")
    print(f"  平局:     {draws / n_eval:.1%} ({draws})")
    print(f"  平均击杀: {total_kills / n_eval:.2f}")
    print(f"  红方存活: {total_red_alive / n_eval:.2f}/{n_agents}")
    print(f"{'=' * 60}\n")

    return {
        "label": label,
        "difficulty": difficulty,
        "win_rate": win_rate,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "avg_kills": total_kills / n_eval,
    }


if __name__ == "__main__":
    results = []

    # 测试 1: 双 PPO-attack 在 2v2 easy
    r = run_cross_validation(
        ["attack", "attack"], difficulty="easy", n_eval=100,
    )
    results.append(r)

    # 测试 2: PPO-attack + PPO-defend 在 2v2 easy
    r = run_cross_validation(
        ["attack", "defend"], difficulty="easy", n_eval=100,
    )
    results.append(r)

    # 测试 3: PPO-attack + PPO-defend 在 2v3 medium
    r = run_cross_validation(
        ["attack", "defend"], difficulty="medium", n_eval=100,
    )
    results.append(r)

    # 汇总
    print("\n" + "=" * 70)
    print("  交叉验证汇总")
    print("=" * 70)
    for r in results:
        print(f"  {r['label']:20s} | {r['difficulty']:6s} | "
              f"胜率 {r['win_rate']:.1%} | 击杀 {r['avg_kills']:.2f}")
    print("=" * 70)
