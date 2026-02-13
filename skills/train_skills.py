"""
底层技能训练脚本 - 经典坦克大战版 (GPU 加速 + 可视化训练)

使用 Stable-Baselines3 PPO 训练三个独立的底层技能:
  1. 导航技能 (navigate): 在地图中避障移动到目标点
  2. 攻击技能 (attack):   追击并消灭敌方坦克
  3. 防守技能 (defend):   保护基地, 拦截朝基地移动的敌人

GPU 加速特性:
  - 自动检测 GPU, 支持指定 CUDA 设备 (cuda:0, cuda:1, ...)
  - SubprocVecEnv 多进程并行环境 (默认 16 个)
  - 大 batch_size + 大 n_steps 充分利用 GPU 算力
  - 支持 4xA100 多 GPU 并行: 每个技能分配到不同 GPU

用法:
  python -m skills.train_skills --skill navigate --timesteps 200000
  python -m skills.train_skills --skill all --timesteps 200000 --device cuda:0
  python -m skills.train_skills --skill all --timesteps 200000 --n_envs 32 --visualize
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =====================================================================
#  SB3 可视化回调
# =====================================================================

class VisualEvalCallback:
    """SB3 回调: 定期运行可视化评估回合.

    每隔 eval_freq 步, 在一个独立的 SingleTankSkillEnv 中
    运行一个完整的评估回合, 使用 TankRenderer 实时渲染.
    """

    def __init__(
        self,
        skill_type: str,
        map_name: str = "classic_1",
        eval_freq: int = 10_000,
        max_eval_steps: int = 200,
        fps: int = 12,
    ) -> None:
        from stable_baselines3.common.callbacks import BaseCallback

        self._BaseCallback = BaseCallback
        self.skill_type = skill_type
        self.map_name = map_name
        self.eval_freq = eval_freq
        self.max_eval_steps = max_eval_steps
        self.fps = fps

        self._monitor: Optional[object] = None
        self._eval_env: Optional[object] = None
        self._step_count = 0
        self._running = True

    def _make_callback_class(self):
        """动态创建 BaseCallback 子类 (避免顶层 import SB3)."""
        parent = self

        class _Callback(self._BaseCallback):
            def __init__(self_cb, verbose=0):
                super().__init__(verbose)
                self_cb._last_eval_ts = 0

            def _on_step(self_cb) -> bool:
                # 使用 num_timesteps (跨所有并行环境的真实步数)
                ts = self_cb.num_timesteps
                if ts - self_cb._last_eval_ts >= parent.eval_freq:
                    self_cb._last_eval_ts = ts
                    return parent._run_visual_eval(self_cb.model)
                return parent._running

        return _Callback()

    def get_callback(self):
        return self._make_callback_class()

    def _run_visual_eval(self, model) -> bool:
        """运行一次可视化评估."""
        from envs.single_tank_env import SingleTankSkillEnv
        from utils.train_monitor import TrainMonitor

        if self._eval_env is None:
            self._eval_env = SingleTankSkillEnv(
                skill_type=self.skill_type,
                map_name=self.map_name,
                max_steps=self.max_eval_steps,
                n_enemies=3,
            )
        if self._monitor is None:
            self._monitor = TrainMonitor(
                cell_size=32, fps=self.fps, render_every=1
            )

        def policy_fn(obs):
            action, _ = model.predict(obs, deterministic=True)
            return int(action)

        self._step_count += 1
        ok = self._monitor.render_single_eval(
            self._eval_env, policy_fn,
            episode=self._step_count,
            max_steps=self.max_eval_steps,
        )
        self._running = ok
        return ok


# =====================================================================
#  设备检测工具
# =====================================================================

def _resolve_device(device_str: str) -> str:
    """解析设备字符串, 返回有效的 torch device 名.

    Args:
        device_str: 'auto', 'cpu', 'cuda', 'cuda:0', 'cuda:1', ...

    Returns:
        有效的 device 字符串, 例如 'cuda:0' 或 'cpu'
    """
    import torch

    if device_str == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        return "cpu"
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print(f"  [警告] 请求 {device_str} 但 CUDA 不可用, 回退到 CPU")
        return "cpu"
    return device_str


def _get_gpu_count() -> int:
    """获取可用 GPU 数量."""
    import torch

    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


# =====================================================================
#  课程学习配置
# =====================================================================

# 每个技能的课程阶段: (n_enemies, timesteps, win_rate_threshold)
CURRICULUM = {
    "attack": [
        {"n_enemies": 1, "timesteps": 1_000_000, "threshold": 0.70, "label": "C1 1v1"},
        {"n_enemies": 2, "timesteps": 1_000_000, "threshold": 0.40, "label": "C2 1v2"},
        {"n_enemies": 3, "timesteps": 2_000_000, "threshold": 0.20, "label": "C3 1v3"},
    ],
    "defend": [
        {"n_enemies": 1, "timesteps": 1_000_000, "threshold": 0.80, "label": "C1 1v1"},
        {"n_enemies": 2, "timesteps": 1_000_000, "threshold": 0.50, "label": "C2 1v2"},
        {"n_enemies": 3, "timesteps": 2_000_000, "threshold": 0.25, "label": "C3 1v3"},
    ],
    "navigate": [
        {"n_enemies": 0, "timesteps": 500_000,   "threshold": 0.90, "label": "C1 0敌"},
        {"n_enemies": 1, "timesteps": 500_000,   "threshold": 0.50, "label": "C2 1敌"},
        {"n_enemies": 3, "timesteps": 1_000_000, "threshold": 0.30, "label": "C3 3敌"},
    ],
}

# 各课程阶段的超参数 (ent_coef, learning_rate)
STAGE_HYPERPARAMS = {
    0: {"ent_coef": 0.01,  "learning_rate": 3e-4},   # C1: 较多探索
    1: {"ent_coef": 0.005, "learning_rate": 2e-4},   # C2: 中等探索
    2: {"ent_coef": 0.001, "learning_rate": 1e-4},   # C3: 收敛
}


# =====================================================================
#  课程学习训练核心
# =====================================================================

def _make_curriculum_env(
    skill_type: str,
    n_enemies: int,
    n_envs: int,
):
    """创建用于课程学习某阶段的并行环境."""
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    from envs.single_tank_env import SingleTankSkillEnv

    maps = ["classic_1", "classic_2", "classic_3"]

    def make_env(map_name: str):
        def _init():
            return SingleTankSkillEnv(
                skill_type=skill_type,
                map_name=map_name,
                max_steps=300,
                n_enemies=n_enemies,
            )
        return _init

    env_fns = [make_env(maps[i % len(maps)]) for i in range(n_envs)]
    env = SubprocVecEnv(env_fns)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return env


def _verify_skill_stage(
    model: object,
    skill_type: str,
    n_enemies: int,
    vec_normalize=None,
    n_eval: int = 100,
) -> float:
    """验证某课程阶段的胜率, 返回胜率值.

    关键: 使用 VecNormalize 归一化观测, 确保与训练时一致.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from envs.single_tank_env import SingleTankSkillEnv

    n_enemies_eval = max(n_enemies, 1)

    # 创建 VecNormalize 包裹的评估环境
    eval_env = DummyVecEnv([lambda: SingleTankSkillEnv(
        skill_type=skill_type,
        map_name="classic_1",
        max_steps=300,
        n_enemies=n_enemies_eval,
    )])

    if vec_normalize is not None:
        # 用训练环境的 VecNormalize 统计量包裹评估环境
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
        eval_env.obs_rms = vec_normalize.obs_rms  # 复制统计量
        eval_env.ret_rms = vec_normalize.ret_rms
        eval_env.training = False  # 不更新统计量
        eval_env.norm_reward = False

    wins = 0
    deaths = 0
    base_destroyed = 0
    total_kills = 0

    for ep in range(n_eval):
        obs = eval_env.reset()
        done = False

        for step in range(300):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = eval_env.step(action)
            if dones[0]:
                info = infos[0]
                break
        else:
            info = infos[0]

        enemies_left = info.get("enemies_alive", n_enemies_eval)
        total_kills += n_enemies_eval - enemies_left

        if skill_type == "navigate":
            if info.get("player_alive", True) and info.get("base_alive", True) and dones[0]:
                wins += 1
        else:
            if not info.get("player_alive", True):
                deaths += 1
            elif not info.get("base_alive", True):
                base_destroyed += 1
            elif enemies_left == 0:
                wins += 1

    eval_env.close()
    win_rate = wins / n_eval
    return win_rate


def _train_skill_curriculum(
    skill_type: str,
    save_dir: str = "skills/models",
    device: str = "auto",
    n_envs: int = 32,
    n_steps: int = 2048,
    visualize: bool = False,
    vis_freq: int = 50_000,
) -> None:
    """课程学习训练单个技能: C1→C2→C3 逐步增难, 迁移学习.

    每个阶段:
      1. 创建新环境 (新的 n_enemies)
      2. 从上一阶段模型权重继续训练 (迁移学习)
      3. 训练完成后验证胜率
      4. 通过门槛 → 保存 → 下一阶段
      5. 未通过 → 额外训练 50% 步数, 再验证

    Args:
        skill_type: 'navigate', 'attack', 'defend'
        save_dir: 模型保存目录
        device: 设备
        n_envs: 并行环境数
        n_steps: PPO n_steps
        visualize: 是否可视化
        vis_freq: 可视化频率
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize

    device = _resolve_device(device)
    skill_names = {"navigate": "导航", "attack": "攻击", "defend": "防守"}
    stages = CURRICULUM[skill_type]

    print("\n" + "=" * 70)
    print(f"  {skill_names[skill_type]}技能 ({skill_type}) — 课程学习训练")
    print(f"  设备: {device}, 并行环境: {n_envs}")
    print(f"  课程: {' → '.join(s['label'] for s in stages)}")
    print("=" * 70)

    os.makedirs(save_dir, exist_ok=True)
    model = None
    prev_vecnorm_path = None

    for stage_idx, stage in enumerate(stages):
        n_enemies = stage["n_enemies"]
        timesteps = stage["timesteps"]
        threshold = stage["threshold"]
        label = stage["label"]
        hyper = STAGE_HYPERPARAMS[stage_idx]

        print(f"\n{'─' * 60}")
        print(f"  [{label}] n_enemies={n_enemies}, "
              f"timesteps={timesteps:,}, 门槛={threshold:.0%}")
        print(f"  ent_coef={hyper['ent_coef']}, "
              f"lr={hyper['learning_rate']}")
        print(f"{'─' * 60}")

        # 1. 创建新环境
        env = _make_curriculum_env(skill_type, n_enemies, n_envs)

        # 如果有上一阶段的 VecNormalize, 加载统计量并继续更新
        if prev_vecnorm_path and os.path.exists(prev_vecnorm_path):
            print(f"  [迁移] 加载上阶段 VecNormalize: {prev_vecnorm_path}")
            env = VecNormalize.load(prev_vecnorm_path, env.venv)
            env.training = True
            env.norm_reward = True

        # 2. 创建/加载模型
        net_arch = [256, 256]
        batch_size = 256

        # tensorboard
        tb_log = None
        try:
            import tensorboard  # noqa: F401
            tb_log = f"logs/tb_{skill_type}_{label.replace(' ', '_')}"
        except ImportError:
            pass

        if model is None:
            # C1: 从零创建
            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                learning_rate=hyper["learning_rate"],
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=hyper["ent_coef"],
                device=device,
                policy_kwargs=dict(net_arch=net_arch),
                tensorboard_log=tb_log,
            )
            print(f"  [新建] PPO net_arch={net_arch}, device={device}")
        else:
            # C2+: 迁移学习 — 从上一阶段权重继续
            # 保存临时模型, 然后用新环境重新加载
            tmp_path = os.path.join(save_dir, f"_tmp_{skill_type}")
            model.save(tmp_path)
            model = PPO.load(
                tmp_path,
                env=env,
                device=device,
                learning_rate=hyper["learning_rate"],
                ent_coef=hyper["ent_coef"],
                tensorboard_log=tb_log,
            )
            print(f"  [迁移] 从上阶段权重继续训练")

        # 3. 训练
        callbacks = []
        if visualize:
            vis_cb = VisualEvalCallback(
                skill_type=skill_type,
                eval_freq=vis_freq,
                fps=12,
            )
            callbacks.append(vis_cb.get_callback())

        model.learn(
            total_timesteps=timesteps,
            progress_bar=True,
            callback=callbacks if callbacks else None,
            reset_num_timesteps=True,
        )

        # 4. 验证 (传入 VecNormalize 确保观测归一化一致)
        win_rate = _verify_skill_stage(
            model, skill_type, n_enemies,
            vec_normalize=env, n_eval=100,
        )
        print(f"  [{label}] 验证胜率: {win_rate:.1%} (门槛: {threshold:.0%})")

        if win_rate < threshold:
            # 额外训练 50%
            extra = int(timesteps * 0.5)
            print(f"  ⚠ 未达门槛, 额外训练 {extra:,} 步...")
            model.learn(
                total_timesteps=extra,
                progress_bar=True,
                reset_num_timesteps=True,
            )
            win_rate = _verify_skill_stage(
                model, skill_type, n_enemies,
                vec_normalize=env, n_eval=100,
            )
            print(f"  [{label}] 重新验证: {win_rate:.1%}")
            if win_rate < threshold:
                print(f"  ⚠ 仍未达门槛, 继续下一阶段")
            else:
                print(f"  ✓ 补训后通过!")
        else:
            print(f"  ✓ 通过!")

        # 5. 保存该阶段模型
        stage_save = os.path.join(save_dir, f"{skill_type}_{label.replace(' ', '_')}")
        model.save(stage_save)

        # 保存 VecNormalize
        vecnorm_path = os.path.join(save_dir, f"{skill_type}_vecnorm.pkl")
        env.save(vecnorm_path)
        prev_vecnorm_path = vecnorm_path

        env.close()
        print(f"  [保存] {stage_save}")

    # 最终保存 (覆盖为标准名称)
    final_path = os.path.join(save_dir, f"{skill_type}_skill")
    model.save(final_path)
    print(f"\n[{skill_names[skill_type]}技能] 课程学习完成! 模型: {final_path}")

    # 最终完整验证 (3 敌人, 使用训练的 VecNormalize 统计量)
    # 从保存的 vecnorm 文件加载
    _verify_skill(
        model, skill_type, skill_names[skill_type],
        save_dir=save_dir, n_eval=100,
    )


# =====================================================================
#  旧版训练函数 (保留兼容)
# =====================================================================

def _train_skill(
    skill_type: str,
    timesteps: int,
    save_dir: str,
    device: str = "auto",
    n_envs: int = 16,
    batch_size: int = 512,
    n_steps: int = 2048,
    visualize: bool = False,
    vis_freq: int = 10_000,
) -> None:
    """训练单个技能 — 直接调用课程学习版本.

    旧版参数仍然保留兼容, 但内部转发到课程学习流程.
    """
    _train_skill_curriculum(
        skill_type=skill_type,
        save_dir=save_dir,
        device=device,
        n_envs=n_envs,
        n_steps=n_steps,
        visualize=visualize,
        vis_freq=vis_freq,
    )


# =====================================================================
#  训练后自动验证
# =====================================================================

def _verify_skill(
    model: object,
    skill_type: str,
    skill_name: str,
    vec_normalize=None,
    save_dir: str = "skills/models",
    n_eval: int = 100,
) -> None:
    """训练后自动运行评估, 报告胜率和详细统计.

    使用 VecNormalize 确保观测归一化与训练一致.

    Args:
        model: 训练好的 SB3 PPO 模型
        skill_type: 技能类型
        skill_name: 技能中文名
        vec_normalize: 训练时的 VecNormalize 环境 (用于复制统计量)
        save_dir: 模型保存目录 (用于加载 VecNormalize)
        n_eval: 评估局数
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from envs.single_tank_env import SingleTankSkillEnv

    print(f"\n{'=' * 50}")
    print(f"  {skill_name}技能 ({skill_type}) — 自动验证 ({n_eval} 局)")
    print(f"{'=' * 50}")

    # 创建 VecNormalize 包裹的评估环境
    eval_env = DummyVecEnv([lambda: SingleTankSkillEnv(
        skill_type=skill_type,
        map_name="classic_1",
        max_steps=300,
        n_enemies=3,
    )])

    # 加载 VecNormalize 统计量
    if vec_normalize is not None:
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
        eval_env.obs_rms = vec_normalize.obs_rms
        eval_env.ret_rms = vec_normalize.ret_rms
        eval_env.training = False
        eval_env.norm_reward = False
        print(f"  (使用训练环境的 VecNormalize 统计量)")
    else:
        # 尝试从磁盘加载
        vecnorm_path = os.path.join(save_dir, f"{skill_type}_vecnorm.pkl")
        if os.path.exists(vecnorm_path):
            eval_env = VecNormalize.load(vecnorm_path, eval_env)
            eval_env.training = False
            eval_env.norm_reward = False
            print(f"  (从 {vecnorm_path} 加载 VecNormalize)")
        else:
            print(f"  ⚠ 警告: 无 VecNormalize, 使用原始观测")

    wins = 0
    deaths = 0
    base_destroyed = 0
    timeouts = 0
    total_kills = 0
    total_reward = 0.0

    for ep in range(n_eval):
        obs = eval_env.reset()
        ep_reward = 0.0

        for step in range(300):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = eval_env.step(action)
            ep_reward += reward[0]
            if dones[0]:
                info = infos[0]
                break
        else:
            info = infos[0]

        total_reward += ep_reward
        enemies_left = info.get("enemies_alive", 3)
        total_kills += 3 - enemies_left

        if not info.get("player_alive", True):
            deaths += 1
        elif not info.get("base_alive", True):
            base_destroyed += 1
        elif enemies_left == 0:
            wins += 1
        else:
            timeouts += 1

    eval_env.close()
    win_rate = wins / n_eval
    print(f"  胜率:       {win_rate:.1%} ({wins}/{n_eval})")
    print(f"  死亡率:     {deaths / n_eval:.1%} ({deaths})")
    print(f"  基地被毁:   {base_destroyed / n_eval:.1%} ({base_destroyed})")
    print(f"  超时:       {timeouts / n_eval:.1%} ({timeouts})")
    print(f"  平均击杀:   {total_kills / n_eval:.2f}/3")
    print(f"  平均奖励:   {total_reward / n_eval:.2f}")

    if win_rate < 0.30:
        print(f"\n  ⚠️ 警告: {skill_name}技能胜率 {win_rate:.1%} < 30%, "
              f"建议增加训练量或调整超参数")
    else:
        print(f"\n  ✓ {skill_name}技能验证通过 (胜率 {win_rate:.1%})")
    print(f"{'=' * 50}\n")


# =====================================================================
#  外部接口
# =====================================================================

def train_navigate(
    timesteps: int = 2_000_000, save_dir: str = "skills/models",
    visualize: bool = False, vis_freq: int = 10_000,
    device: str = "auto", n_envs: int = 16,
    batch_size: int = 512, n_steps: int = 2048,
) -> None:
    _train_skill(
        "navigate", timesteps, save_dir, device, n_envs,
        batch_size, n_steps, visualize, vis_freq,
    )


def train_attack(
    timesteps: int = 2_000_000, save_dir: str = "skills/models",
    visualize: bool = False, vis_freq: int = 10_000,
    device: str = "auto", n_envs: int = 16,
    batch_size: int = 512, n_steps: int = 2048,
) -> None:
    _train_skill(
        "attack", timesteps, save_dir, device, n_envs,
        batch_size, n_steps, visualize, vis_freq,
    )


def train_defend(
    timesteps: int = 2_000_000, save_dir: str = "skills/models",
    visualize: bool = False, vis_freq: int = 10_000,
    device: str = "auto", n_envs: int = 16,
    batch_size: int = 512, n_steps: int = 2048,
) -> None:
    _train_skill(
        "defend", timesteps, save_dir, device, n_envs,
        batch_size, n_steps, visualize, vis_freq,
    )


def train_skill_on_gpu(
    skill_type: str,
    gpu_id: int,
    timesteps: int = 2_000_000,
    save_dir: str = "skills/models",
    n_envs: int = 16,
    batch_size: int = 512,
    n_steps: int = 2048,
) -> None:
    """在指定 GPU 上训练单个技能 (用于多 GPU 并行).

    Args:
        skill_type: 'navigate', 'attack', 'defend'
        gpu_id: GPU 编号 (0, 1, 2, 3)
        timesteps: 训练步数
    """
    device = f"cuda:{gpu_id}"
    _train_skill(
        skill_type, timesteps, save_dir, device, n_envs,
        batch_size, n_steps,
    )


def train_all_parallel(
    timesteps: int = 2_000_000,
    save_dir: str = "skills/models",
    n_envs: int = 16,
    batch_size: int = 512,
    n_steps: int = 2048,
) -> None:
    """并行训练所有 3 个技能 (3 个进程同时运行).

    策略: PPO + MLP 策略在 CPU 上更高效 (环境步进是瓶颈, 非网络计算),
    但我们用 SubprocVecEnv + 多进程并行, 3 个技能同时训练, 充分利用多核 CPU.
    每个进程内部 16 个 SubprocVecEnv 环境并行步进.
    MADDPG 阶段再使用 GPU (注意力 Critic 计算量大).
    """
    import multiprocessing as mp

    skills = ["navigate", "attack", "defend"]

    print("=" * 60)
    print("多进程并行训练 — 3 个技能同时运行")
    for skill in skills:
        print(f"  {skill}: CPU + {n_envs} SubprocVecEnv 并行环境")
    print(f"  batch={batch_size}, n_steps={n_steps}")
    n_gpus = _get_gpu_count()
    if n_gpus > 0:
        print(f"  ({n_gpus} 块 GPU 保留给 MADDPG 协同训练)")
    print("=" * 60)

    # 使用 spawn 确保子进程独立
    ctx = mp.get_context("spawn")
    processes = []

    for skill in skills:
        p = ctx.Process(
            target=_train_skill,
            args=(skill, timesteps, save_dir, "cpu", n_envs, batch_size, n_steps),
            name=f"train_{skill}",
        )
        p.start()
        print(f"  [PID {p.pid}] {skill} 训练已启动")
        processes.append(p)

    # 等待所有训练完成
    for p in processes:
        p.join()
        print(f"  [PID {p.pid}] {p.name} 完成 (exit_code={p.exitcode})")

    print("\n所有底层技能并行训练完成!")


# =====================================================================
#  命令行入口
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="训练底层坦克技能 (经典坦克大战) — GPU 加速版",
    )
    parser.add_argument(
        "--skill", type=str, default="all",
        choices=["navigate", "attack", "defend", "all"],
        help="训练哪个技能 (默认: all)",
    )
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--save_dir", type=str, default="skills/models")
    parser.add_argument(
        "--device", type=str, default="auto",
        help="设备: auto/cpu/cuda/cuda:0/cuda:1/...",
    )
    parser.add_argument(
        "--n_envs", type=int, default=16,
        help="并行环境数 (GPU 推荐 16-32, CPU 推荐 4-8)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=512,
        help="PPO 批量大小 (GPU 推荐 512-1024)",
    )
    parser.add_argument(
        "--n_steps", type=int, default=2048,
        help="PPO 每次收集步数 (GPU 推荐 2048-4096)",
    )
    parser.add_argument(
        "--parallel", action="store_true",
        help="多 GPU 并行训练所有技能 (3 GPU 各训练 1 个技能)",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="启用可视化训练模式",
    )
    parser.add_argument(
        "--vis_freq", type=int, default=10_000,
        help="可视化评估频率 (每隔多少步)",
    )
    args = parser.parse_args()

    if args.parallel:
        train_all_parallel(
            timesteps=args.timesteps,
            save_dir=args.save_dir,
            n_envs=args.n_envs,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
        )
        return

    if args.skill in ("navigate", "all"):
        train_navigate(
            args.timesteps, args.save_dir, args.visualize, args.vis_freq,
            args.device, args.n_envs, args.batch_size, args.n_steps,
        )
    if args.skill in ("attack", "all"):
        train_attack(
            args.timesteps, args.save_dir, args.visualize, args.vis_freq,
            args.device, args.n_envs, args.batch_size, args.n_steps,
        )
    if args.skill in ("defend", "all"):
        train_defend(
            args.timesteps, args.save_dir, args.visualize, args.vis_freq,
            args.device, args.n_envs, args.batch_size, args.n_steps,
        )
    print("\n底层技能训练完成!")


if __name__ == "__main__":
    main()
