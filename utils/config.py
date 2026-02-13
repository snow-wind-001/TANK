"""
配置管理: OmegaConf 加载 YAML + CLI 覆盖.

设计原则:
  1. config/config.yaml 作为默认参数来源
  2. --config 指定覆盖配置文件 (如 config/fast_test.yaml)
  3. CLI 点号参数覆盖 (如 coord.batch_size=1024)
  4. 优先级: CLI > --config > config/config.yaml

用法:
  cfg = load_config()                    # 从 sys.argv 自动解析
  cfg = load_config(["coord.batch_size=1024"])  # 手动覆盖

  # 访问
  print(cfg.coord.batch_size)            # 1024
  print(cfg.defaults.device)             # "auto"

  # 转为 dict (传给函数)
  coord_kwargs = OmegaConf.to_container(cfg.coord, resolve=True)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"


def _find_project_config() -> Path:
    """找到项目根目录下的默认配置文件."""
    if DEFAULT_CONFIG.exists():
        return DEFAULT_CONFIG
    # 兼容: 从当前工作目录查找
    cwd_config = Path.cwd() / "config" / "config.yaml"
    if cwd_config.exists():
        return cwd_config
    raise FileNotFoundError(
        f"找不到默认配置文件: {DEFAULT_CONFIG}\n"
        "请确保 config/config.yaml 存在."
    )


def load_config(
    cli_overrides: Optional[list[str]] = None,
    config_path: Optional[str] = None,
) -> DictConfig:
    """加载配置: 默认 YAML → 覆盖 YAML → CLI 参数.

    Args:
        cli_overrides: 点号覆盖列表, 如 ["coord.batch_size=1024"].
                      若为 None, 自动从 sys.argv 提取.
        config_path: 覆盖配置文件路径 (如 config/fast_test.yaml).
                    若为 None, 从 sys.argv 的 --config 提取.

    Returns:
        DictConfig: 合并后的配置对象.
    """
    # 1. 加载默认配置
    default_path = _find_project_config()
    cfg = OmegaConf.load(default_path)
    assert isinstance(cfg, DictConfig)

    # 2. 从 sys.argv 提取 --config 和点号覆盖
    if cli_overrides is None:
        cli_overrides = []
        argv = sys.argv[1:]
        skip_next = False
        for i, arg in enumerate(argv):
            if skip_next:
                skip_next = False
                continue
            if arg == "--config":
                if i + 1 < len(argv):
                    config_path = config_path or argv[i + 1]
                    skip_next = True
            elif "=" in arg and not arg.startswith("-"):
                # 点号覆盖: coord.batch_size=1024
                cli_overrides.append(arg)

    # 3. 合并覆盖配置文件
    if config_path is not None:
        override_path = Path(config_path)
        if not override_path.is_absolute():
            override_path = PROJECT_ROOT / override_path
        if override_path.exists():
            override_cfg = OmegaConf.load(override_path)
            cfg = OmegaConf.merge(cfg, override_cfg)
            print(f"[配置] 已加载覆盖: {override_path}")
        else:
            print(f"[配置] 警告: 覆盖文件不存在: {override_path}")

    # 4. 合并 CLI 点号覆盖
    if cli_overrides:
        cli_cfg = OmegaConf.from_dotlist(cli_overrides)
        cfg = OmegaConf.merge(cfg, cli_cfg)
        for ov in cli_overrides:
            print(f"[配置] CLI 覆盖: {ov}")

    return cfg


def cfg_to_dict(cfg: DictConfig) -> dict[str, Any]:
    """将 DictConfig 转为普通 dict (resolve 所有引用)."""
    return OmegaConf.to_container(cfg, resolve=True)


def print_config(cfg: DictConfig, title: str = "当前配置") -> None:
    """格式化打印配置."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)


def get_coord_kwargs(cfg: DictConfig) -> dict[str, Any]:
    """从配置中提取 MADDPG 训练参数 (可直接传给 train())."""
    c = cfg.coord
    d = cfg.defaults
    return dict(
        episodes=c.episodes,
        skill_interval=c.skill_interval,
        coord_batch_size=c.batch_size,
        n_collect_envs=c.n_collect_envs,
        updates_per_step=c.updates_per_step,
        hidden_dim=c.hidden_dim,
        buffer_capacity=c.buffer_capacity,
        warmup_steps=c.warmup_steps,
        use_rule_skills=c.use_rule_skills,
        use_attention=c.use_attention,
        use_comm=c.get("use_comm", False),
        map_name=d.map_name,
        n_blue=d.n_blue,
        difficulty=d.get("difficulty", "easy"),
        save_dir=d.save_dir,
        save_interval=c.save_interval,
        log_interval=c.log_interval,
        device_str=d.device,
        seed=d.get("seed", 0),
    )


def get_population_kwargs(cfg: DictConfig) -> dict[str, Any]:
    """从配置中提取多 GPU 人口训练参数."""
    c = cfg.coord
    p = cfg.population
    d = cfg.defaults
    return dict(
        episodes=c.episodes,
        n_gpus=p.n_gpus,
        n_collect_envs=c.n_collect_envs,
        updates_per_step=c.updates_per_step,
        coord_batch_size=c.batch_size,
        hidden_dim=c.hidden_dim,
        use_rule_skills=c.use_rule_skills,
        map_name=d.map_name,
        n_blue=d.n_blue,
        difficulty=d.get("difficulty", "easy"),
        save_dir=d.save_dir,
        use_attention=c.use_attention,
        use_comm=c.get("use_comm", False),
    )


def get_skills_kwargs(cfg: DictConfig) -> dict[str, Any]:
    """从配置中提取底层技能训练参数."""
    s = cfg.skills
    d = cfg.defaults
    return dict(
        timesteps=s.timesteps,
        n_envs=s.n_envs,
        batch_size=s.batch_size,
        n_steps=s.n_steps,
        device=d.device,
    )
