"""
经典坦克大战 - 分层 MADDPG 协同训练 (配置参数化 + 4xA100 加速)

配置系统:
  config/config.yaml 为默认参数, 支持 CLI 覆盖:
    python main.py --mode train_coord                          # 默认配置
    python main.py --mode train_coord --config config/fast_test.yaml
    python main.py --mode train_coord coord.batch_size=1024 coord.n_collect_envs=64

  GPU 利用率 = f(n_collect_envs × updates_per_step × batch_size)
  不靠增大模型, 靠多环境 + 大 batch + 多梯度步充分喂饱 GPU.

运行模式:
  train_skills           训练底层技能 (导航/攻击/防守)
  train_skills_vis       可视化训练底层技能
  train_skills_parallel  3 个技能分配到 3 块 GPU 并行
  train_coord            MADDPG 协同训练 (单 GPU, 多环境)
  train_coord_vis        可视化协同训练
  train_coord_multi_gpu  4 块 GPU 人口训练 (不同种子, 取最佳)
  train_all              一键全流程: 技能 → 多 GPU 协同
  demo                   规则技能快速演示
  visualize              Pygame 可视化对战
  visualize_coop         协同可视化 (协作连线 + 注意力热力图)
  play                   人类玩家控制

用法:
  # [推荐] 4xGPU 人口训练 (充分利用硬件)
  python main.py --mode train_coord_multi_gpu

  # 自定义配置
  python main.py --mode train_coord coord.n_collect_envs=64 coord.batch_size=1024

  # 快速测试 (2 分钟)
  python main.py --mode train_coord --config config/fast_test.yaml

  # 最大吞吐 (4xA100 全力)
  python main.py --mode train_coord_multi_gpu --config config/max_throughput.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import (
    get_coord_kwargs,
    get_population_kwargs,
    get_skills_kwargs,
    load_config,
    print_config,
)


# =====================================================================
#  工具函数
# =====================================================================

def _fix_opengl_for_conda() -> None:
    """修复 conda 环境下 libstdc++ 与系统 mesa 驱动的兼容性问题."""
    sys_stdcpp = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
    if not os.path.exists(sys_stdcpp):
        return
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    ld_preload = os.environ.get("LD_PRELOAD", "")
    if conda_prefix and sys_stdcpp not in ld_preload:
        conda_stdcpp = os.path.join(conda_prefix, "lib", "libstdc++.so.6")
        if os.path.exists(conda_stdcpp):
            os.environ["LD_PRELOAD"] = (
                sys_stdcpp + (":" + ld_preload if ld_preload else "")
            )
            os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
            os.execv(sys.executable, [sys.executable] + sys.argv)


def _resolve_device(device_str: str) -> str:
    """解析设备字符串."""
    import torch

    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print(f"[警告] 请求 {device_str} 但 CUDA 不可用, 回退到 CPU")
        return "cpu"
    return device_str


def _detect_model_config(model_dir: str) -> dict:
    """从模型检查点自动检测 use_comm / use_attention 配置."""
    import torch
    config = {"use_comm": False, "use_attention": True}

    meta_path = os.path.join(model_dir, "meta.pth")
    if os.path.exists(meta_path):
        meta = torch.load(meta_path, map_location="cpu", weights_only=False)
        config["use_comm"] = meta.get("use_comm", False)
        config["use_attention"] = meta.get("use_attention", True)

    # 通过 actor 权重第一层形状兜底检测
    actor_path = os.path.join(model_dir, "actor_0.pth")
    if os.path.exists(actor_path):
        state = torch.load(actor_path, map_location="cpu", weights_only=True)
        key = "shared.0.weight"
        if key in state:
            from envs.multi_tank_env import OBS_DIM
            if state[key].shape[1] > OBS_DIM:
                config["use_comm"] = True

    return config


def _auto_coord_device(device_str: str) -> str:
    """为 MADDPG 自动选择 GPU."""
    if device_str != "auto":
        return device_str
    import torch
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        dev = f"cuda:{min(3, n_gpus - 1)}"
        print(f"[MADDPG] 自动分配 GPU: {dev}")
        return dev
    return "cpu"


# =====================================================================
#  模式: 训练底层技能
# =====================================================================

def mode_train_skills(cfg: DictConfig) -> None:
    """训练底层技能 (支持 GPU)."""
    from skills.train_skills import train_navigate, train_attack, train_defend

    sk = get_skills_kwargs(cfg)
    skill = cfg.skills.skill

    if skill in ("navigate", "all"):
        train_navigate(sk["timesteps"], **{k: v for k, v in sk.items() if k != "timesteps"})
    if skill in ("attack", "all"):
        train_attack(sk["timesteps"], **{k: v for k, v in sk.items() if k != "timesteps"})
    if skill in ("defend", "all"):
        train_defend(sk["timesteps"], **{k: v for k, v in sk.items() if k != "timesteps"})
    print("\n底层技能训练完成!")


def mode_train_skills_parallel(cfg: DictConfig) -> None:
    """多 GPU 并行训练所有 3 个底层技能 (最快)."""
    from skills.train_skills import train_all_parallel

    sk = get_skills_kwargs(cfg)
    train_all_parallel(
        timesteps=sk["timesteps"],
        save_dir="skills/models",
        n_envs=sk["n_envs"],
        batch_size=sk["batch_size"],
        n_steps=sk["n_steps"],
    )


def mode_train_skills_vis(cfg: DictConfig) -> None:
    """可视化底层技能训练."""
    _fix_opengl_for_conda()
    from skills.train_skills import train_navigate, train_attack, train_defend

    sk = get_skills_kwargs(cfg)
    vis_freq = cfg.skills.vis_freq
    skill = cfg.skills.skill

    common = dict(
        visualize=True, vis_freq=vis_freq,
        device=sk["device"], n_envs=sk["n_envs"],
        batch_size=sk["batch_size"], n_steps=sk["n_steps"],
    )

    if skill in ("navigate", "all"):
        train_navigate(sk["timesteps"], **common)
    if skill in ("attack", "all"):
        train_attack(sk["timesteps"], **common)
    if skill in ("defend", "all"):
        train_defend(sk["timesteps"], **common)
    print("\n底层技能可视化训练完成!")


# =====================================================================
#  模式: 训练高层协同 (MADDPG)
# =====================================================================

def mode_train_coord(cfg: DictConfig) -> None:
    """MADDPG 协同训练 — 多环境 + 多步更新 + GPU 加速.

    GPU 利用率通过 n_collect_envs × updates_per_step × batch_size 控制.
    """
    from maddpg.train_coord import train

    kwargs = get_coord_kwargs(cfg)
    kwargs["device_str"] = _auto_coord_device(kwargs["device_str"])
    train(**kwargs)


def mode_train_coord_vis(cfg: DictConfig) -> None:
    """可视化高层协同训练: 含协作连线/注意力矩阵/技能标签."""
    _fix_opengl_for_conda()
    from maddpg.train_coord import train

    kwargs = get_coord_kwargs(cfg)
    kwargs["device_str"] = _auto_coord_device(kwargs["device_str"])
    kwargs["visualize"] = True
    kwargs["vis_interval"] = cfg.visualization.vis_interval
    train(**kwargs)


def mode_train_coord_multi_gpu(cfg: DictConfig) -> None:
    """4 块 GPU 同时训练 MADDPG, 不同随机种子, 取最佳模型."""
    from maddpg.train_coord import train_population

    kwargs = get_population_kwargs(cfg)
    train_population(**kwargs)


def mode_demo(cfg: DictConfig) -> None:
    """规则技能快速演示 (无需预训练)."""
    from maddpg.train_coord import train

    kwargs = get_coord_kwargs(cfg)
    kwargs["device_str"] = cfg.defaults.device
    kwargs["use_rule_skills"] = True
    print("=" * 60)
    print("快速演示: 使用规则 AI 底层技能, 无需预训练")
    print("=" * 60)
    train(**kwargs)


# =====================================================================
#  模式: Pygame 可视化对战
# =====================================================================

def mode_visualize(cfg: DictConfig) -> None:
    _fix_opengl_for_conda()
    import numpy as np
    import torch

    from envs.multi_tank_env import MultiTankTeamEnv, OBS_DIM
    from maddpg.core import MaddpgTrainer
    from maddpg.train_coord import build_skill_library, execute_skill
    from utils.visualize import TankRenderer

    N_AGENTS = 2
    NUM_SKILLS = 3
    PARAM_DIM = 2
    device = _resolve_device(cfg.defaults.device)

    env = MultiTankTeamEnv(
        map_name=cfg.defaults.map_name, n_red=N_AGENTS,
        n_blue=cfg.defaults.n_blue, max_steps=600,
    )
    renderer = TankRenderer(cell_size=32, fps=10)
    # 技能推理始终在 CPU
    skill_lib = build_skill_library(
        use_rule_skills=cfg.coord.use_rule_skills, device="cpu",
    )

    model_dir = os.path.join(cfg.defaults.save_dir, "maddpg_best")
    model_cfg = _detect_model_config(model_dir) if os.path.exists(model_dir) else {}

    trainer = MaddpgTrainer(
        n_agents=N_AGENTS, obs_dim=OBS_DIM, num_skills=NUM_SKILLS,
        param_dim=PARAM_DIM,
        use_attention=model_cfg.get("use_attention", cfg.coord.use_attention),
        use_comm=model_cfg.get("use_comm", False),
        device=device,
        hidden_dim=cfg.coord.hidden_dim,
    )

    if os.path.exists(model_dir):
        trainer.load(model_dir)
        print(f"[可视化] 已加载模型: {model_dir}")
        print(f"  use_comm={model_cfg.get('use_comm')}, use_attention={model_cfg.get('use_attention')}")
    else:
        print("[可视化] 未找到模型, 使用随机策略")

    skill_interval = cfg.coord.skill_interval
    num_episodes = cfg.visualization.num_episodes

    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        step = 0
        current_skill_info = [(1, np.zeros(PARAM_DIM))] * N_AGENTS

        while not done:
            if step % skill_interval == 0:
                actions = trainer.select_actions(obs, deterministic=True)
                current_skill_info = actions

            low_actions = []
            for i in range(N_AGENTS):
                sid, param = current_skill_info[i]
                act = execute_skill(skill_lib, sid, obs[i], param)
                low_actions.append(act)

            next_obs, reward, done, info = env.step(low_actions)
            ep_reward += reward
            step += 1
            obs = next_obs

            running = renderer.render(
                env.engine, episode=ep + 1, step=step, reward=ep_reward,
            )
            if not running:
                renderer.close()
                return

        result = "RED WIN" if info.get("win") else "TIMEOUT/BLUE WIN"
        kills = info.get("blue_killed", 0)
        print(f"[Ep {ep+1}] {result} | R:{ep_reward:.1f} | Kills:{kills} | Steps:{step}")
        time.sleep(1.0)

    renderer.close()


# =====================================================================
#  模式: 可视化协同对战 (含协作网络)
# =====================================================================

def mode_visualize_coop(cfg: DictConfig) -> None:
    """可视化协同对战: 含协作连线、注意力矩阵、技能标签."""
    _fix_opengl_for_conda()
    import numpy as np
    import torch

    from envs.multi_tank_env import MultiTankTeamEnv, OBS_DIM
    from maddpg.core import MaddpgTrainer
    from maddpg.train_coord import build_skill_library, execute_skill
    from utils.visualize import TankRenderer

    N_AGENTS = 2
    NUM_SKILLS = 3
    PARAM_DIM = 2
    device = _resolve_device(cfg.defaults.device)

    env = MultiTankTeamEnv(
        map_name=cfg.defaults.map_name, n_red=N_AGENTS,
        n_blue=cfg.defaults.n_blue, max_steps=600,
    )
    renderer = TankRenderer(cell_size=32, fps=10)
    skill_lib = build_skill_library(
        use_rule_skills=cfg.coord.use_rule_skills, device="cpu",
    )

    model_dir = os.path.join(cfg.defaults.save_dir, "maddpg_best")
    model_cfg = _detect_model_config(model_dir) if os.path.exists(model_dir) else {}

    trainer = MaddpgTrainer(
        n_agents=N_AGENTS, obs_dim=OBS_DIM, num_skills=NUM_SKILLS,
        param_dim=PARAM_DIM,
        use_attention=model_cfg.get("use_attention", True),
        use_comm=model_cfg.get("use_comm", False),
        device=device,
        hidden_dim=cfg.coord.hidden_dim,
    )

    if os.path.exists(model_dir):
        trainer.load(model_dir)
        print(f"[协同可视化] 已加载模型: {model_dir}")
        print(f"  use_comm={model_cfg.get('use_comm')}, use_attention={model_cfg.get('use_attention')}")
    else:
        print("[协同可视化] 未找到模型, 使用随机策略 (仍可看到协作权重)")

    print("\n协作可视化说明:")
    print("  - 黄色连线 = 强协作 | 绿色连线 = 中等协作 | 蓝色连线 = 弱协作")
    print("  - 右下角: 注意力热力图 (R0/R1 之间的关注度)")
    print("  - 坦克上方: 当前执行的技能名称")
    print("  - 按 ESC 退出\n")

    skill_interval = cfg.coord.skill_interval
    num_episodes = cfg.visualization.num_episodes

    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        step = 0
        current_skill_info = [(1, np.zeros(PARAM_DIM))] * N_AGENTS

        while not done:
            if step % skill_interval == 0:
                actions = trainer.select_actions(obs, deterministic=True)
                current_skill_info = actions

            low_actions = []
            for i in range(N_AGENTS):
                sid, param = current_skill_info[i]
                act = execute_skill(skill_lib, sid, obs[i], param)
                low_actions.append(act)

            next_obs, reward, done, info = env.step(low_actions)
            ep_reward += reward
            step += 1
            obs = next_obs

            coop_weights = trainer.get_cooperation_weights()
            running = renderer.render(
                env.engine,
                skill_info=current_skill_info,
                episode=ep + 1,
                step=step,
                reward=ep_reward,
                cooperation_weights=coop_weights,
                training_mode=False,
            )
            if not running:
                renderer.close()
                return

        result = "RED WIN" if info.get("win") else "TIMEOUT/BLUE WIN"
        kills = info.get("blue_killed", 0)
        print(f"[Ep {ep+1}] {result} | R:{ep_reward:.1f} | Kills:{kills} | Steps:{step}")
        time.sleep(1.0)

    renderer.close()


# =====================================================================
#  模式: 人类玩家控制
# =====================================================================

def mode_play(cfg: DictConfig) -> None:
    """人类玩家控制模式 (方向键移动, 空格射击)."""
    _fix_opengl_for_conda()
    import pygame

    from envs.game_engine import Action, BattleCityEngine, Dir
    from utils.visualize import TankRenderer

    engine = BattleCityEngine(map_name=cfg.defaults.map_name)
    engine.reset()

    W, H = engine.width, engine.height
    player = engine.add_tank(W // 2 - 3, H - 3, "red", "normal", Dir.UP)

    spawn_xs = [1, W // 2, W - 2]
    for i in range(cfg.defaults.n_blue):
        engine.add_tank(spawn_xs[i % 3], 0, "blue", "normal", Dir.DOWN)

    renderer = TankRenderer(cell_size=32, fps=8)
    renderer.render(engine, episode=0, step=0, reward=0)

    step = 0
    running = True
    while running:
        action = Action.NOOP
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            action = Action.UP
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            action = Action.DOWN
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            action = Action.LEFT
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            action = Action.RIGHT
        elif keys[pygame.K_SPACE]:
            action = Action.FIRE

        actions = {player.tank_id: action}
        for tank in engine.tanks:
            if tank.team == "blue" and tank.alive:
                actions[tank.tank_id] = engine.blue_ai_action(tank)

        engine.step(actions)
        step += 1

        if not renderer.render(engine, episode=0, step=step, reward=0):
            break

        if not player.alive:
            print("GAME OVER - 你的坦克被摧毁!")
            time.sleep(2)
            break
        if not engine.base_alive:
            print("GAME OVER - 基地被摧毁!")
            time.sleep(2)
            break
        if not any(t.alive for t in engine.tanks if t.team == "blue"):
            print(f"VICTORY! 用时 {step} 步")
            time.sleep(2)
            break

    renderer.close()


# =====================================================================
#  模式: 一键训练全流程
# =====================================================================

def mode_train_all(cfg: DictConfig) -> None:
    """一键训练: 先并行训练技能, 再 4 块 GPU 人口训练协同."""
    import torch

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    print("=" * 60)
    print("一键训练全流程 (底层技能 + 多GPU协同)")
    print(f"  可用 GPU: {n_gpus}")
    print("=" * 60)

    # 阶段 1: 并行训练底层技能
    print("\n" + "=" * 40)
    print("[阶段 1/2] 并行训练底层技能")
    print("=" * 40)
    from skills.train_skills import train_all_parallel

    sk = get_skills_kwargs(cfg)
    train_all_parallel(
        timesteps=sk["timesteps"],
        n_envs=sk["n_envs"],
        batch_size=sk["batch_size"],
        n_steps=sk["n_steps"],
    )

    # 阶段 2: 多 GPU 人口训练协同
    print("\n" + "=" * 40)
    print("[阶段 2/2] 多 GPU 人口训练高层 MADDPG 协同")
    print("=" * 40)

    if n_gpus >= 2:
        from maddpg.train_coord import train_population
        kwargs = get_population_kwargs(cfg)
        kwargs["use_rule_skills"] = False
        train_population(**kwargs)
    else:
        from maddpg.train_coord import train
        kwargs = get_coord_kwargs(cfg)
        kwargs["device_str"] = "cuda:0" if n_gpus > 0 else "cpu"
        kwargs["use_rule_skills"] = False
        train(**kwargs)

    print("\n" + "=" * 60)
    print("全流程训练完成!")
    print("=" * 60)


# =====================================================================
#  命令行解析 + 配置加载
# =====================================================================

ALL_MODES = [
    "train_skills", "train_skills_vis", "train_skills_parallel",
    "train_coord", "train_coord_vis", "train_coord_multi_gpu",
    "train_all",
    "demo",
    "visualize", "visualize_coop",
    "play",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="经典坦克大战 - 配置参数化训练 (OmegaConf + 4xA100)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
配置系统:
  所有参数在 config/config.yaml 中定义, 支持两种覆盖方式:

  1. 覆盖配置文件:
     python main.py --mode train_coord --config config/fast_test.yaml

  2. CLI 点号覆盖 (任意嵌套参数):
     python main.py --mode train_coord coord.batch_size=1024 coord.n_collect_envs=64

运行示例:
  # [推荐] 4xGPU 人口训练
  python main.py --mode train_coord_multi_gpu

  # 快速测试 (2 分钟)
  python main.py --mode train_coord --config config/fast_test.yaml

  # 最大吞吐
  python main.py --mode train_coord_multi_gpu --config config/max_throughput.yaml

  # 手动精调
  python main.py --mode train_coord coord.n_collect_envs=64 coord.batch_size=1024

  # 人类操控
  python main.py --mode play
""",
    )
    parser.add_argument(
        "--mode", type=str, default="play", choices=ALL_MODES,
        help="运行模式",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="覆盖配置文件路径 (如 config/fast_test.yaml)",
    )

    # 兼容旧 CLI 参数 (优先级低于配置文件)
    parser.add_argument("--use_rule_skills", action="store_true",
                        help="使用规则 AI 底层技能 (覆盖 coord.use_rule_skills)")
    parser.add_argument("--no_attention", action="store_true",
                        help="禁用注意力 Critic (覆盖 coord.use_attention)")
    parser.add_argument("--use_comm", action="store_true",
                        help="启用通讯模块 (覆盖 coord.use_comm)")

    args, unknown = parser.parse_known_args()

    # --- 加载配置 ---
    # 从 unknown 中提取点号覆盖 (如 coord.batch_size=1024)
    dot_overrides = [u for u in unknown if "=" in u and not u.startswith("-")]
    cfg = load_config(cli_overrides=dot_overrides, config_path=args.config)

    # --- 兼容旧 CLI flag ---
    if args.use_rule_skills:
        cfg.coord.use_rule_skills = True
    if args.no_attention:
        cfg.coord.use_attention = False
    if args.use_comm:
        cfg.coord.use_comm = True

    # --- 打印配置 ---
    print_config(cfg, title=f"模式: {args.mode}")

    # --- 创建目录 ---
    os.makedirs(cfg.defaults.save_dir, exist_ok=True)
    os.makedirs("skills/models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # --- 分发到模式函数 ---
    mode_map = {
        "train_skills": mode_train_skills,
        "train_skills_vis": mode_train_skills_vis,
        "train_skills_parallel": mode_train_skills_parallel,
        "train_coord": mode_train_coord,
        "train_coord_vis": mode_train_coord_vis,
        "train_coord_multi_gpu": mode_train_coord_multi_gpu,
        "train_all": mode_train_all,
        "demo": mode_demo,
        "visualize": mode_visualize,
        "visualize_coop": mode_visualize_coop,
        "play": mode_play,
    }
    mode_map[args.mode](cfg)


if __name__ == "__main__":
    main()
