"""
底层技能封装器 (Option 接口) - 经典坦克大战版

将底层技能 (PPO/规则) 包装为统一的 act(obs, target_param) -> discrete_action 接口,
供高层 MADDPG 协调器调用.

技能列表:
  - 0: navigate (导航) - 移动到目标位置
  - 1: attack   (攻击) - 追击并消灭目标敌人
  - 2: defend   (防守) - 守护基地, 拦截敌人

动作空间: Discrete(6) = [NOOP, UP, DOWN, LEFT, RIGHT, FIRE]
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from envs.game_engine import Action, Dir


class SkillOption:
    """统一的底层技能调用接口 (加载 PPO 模型 + VecNormalize).

    高层协调器调用时, 传入:
      - local_obs: 来自 MultiTankTeamEnv 的局部观测 (32 维)
      - target_param: 高层输出的连续参数 (2 维, [-1, 1])
    返回:
      - discrete_action: 0-5
    """

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        import os

        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import VecNormalize

        self.model = PPO.load(model_path, device=device)
        for param in self.model.policy.parameters():
            param.requires_grad = False

        # [关键修复] 加载训练时使用的 VecNormalize 统计量
        # 没有归一化 → PPO 收到的是完全错误的输入分布 → 输出随机
        vecnorm_path = model_path.replace("_skill", "_vecnorm.pkl")
        self.vec_normalize: VecNormalize | None = None
        if os.path.exists(vecnorm_path):
            from stable_baselines3.common.vec_env import DummyVecEnv

            from envs.single_tank_env import SingleTankSkillEnv

            # 使用与训练相同的环境类型作为 dummy, 确保 obs space 一致
            # 从 model_path 推断技能类型
            skill_type = "attack"  # 默认
            for st in ["navigate", "attack", "defend"]:
                if st in model_path:
                    skill_type = st
                    break

            dummy_env = DummyVecEnv([lambda: SingleTankSkillEnv(
                skill_type=skill_type, map_name="classic_1",
                max_steps=100, n_enemies=1,
            )])
            self.vec_normalize = VecNormalize.load(vecnorm_path, dummy_env)
            self.vec_normalize.training = False  # 推理模式, 不更新统计量
            self.vec_normalize.norm_reward = False
            print(f"    [SkillOption] 已加载 VecNormalize: {vecnorm_path}")
        else:
            print(f"    [SkillOption] 警告: 未找到 {vecnorm_path}, PPO 推理可能不准确")

    def act(
        self,
        local_obs: np.ndarray,
        target_param: np.ndarray,
        deterministic: bool = True,
    ) -> int:
        """将高层观测+目标参数转换为离散动作.

        将 target_param 注入到观测的最后 2 维 (替换原 target_dx, target_dy).
        """
        # 对于 multi_tank_env 的 32 维观测, 最后 2 维是统计信息
        # 我们需要构造与 single_tank_env 兼容的 29 维观测
        # 方案: 取前 27 维 + target_param 2 维 = 29 维
        if len(local_obs) >= 32:
            # multi -> single 转换: 去掉 ally 信息(4维) 和末尾统计(2维), 添加 target
            # 前 9 维(自身) + 8维(射线) + 6维(敌人) + 3维(基地) + 1维(敌人比) = 27
            base_obs = np.concatenate([
                local_obs[:9],       # 自身状态
                local_obs[9:17],     # 射线
                local_obs[17:23],    # 最近两敌人
                local_obs[27:30],    # 基地信息
                [local_obs[30]],     # 敌人存活比
            ])
            skill_obs = np.concatenate([base_obs, target_param]).astype(np.float32)
        else:
            # 已是 29 维, 替换最后 2 维
            skill_obs = local_obs.copy()
            skill_obs[-2:] = target_param

        # [关键修复] 使用 VecNormalize 归一化观测, 与训练时一致
        if self.vec_normalize is not None:
            skill_obs = self.vec_normalize.normalize_obs(skill_obs)

        action, _ = self.model.predict(skill_obs, deterministic=deterministic)
        return int(action)


class RuleBasedSkill:
    """基于规则的底层技能 (无需预训练模型).

    用于 MADDPG 训练初期或无预训练模型时的备选方案.
    """

    def __init__(self, skill_type: str = "navigate") -> None:
        self.skill_type = skill_type

    def act(
        self,
        local_obs: np.ndarray,
        target_param: np.ndarray,
        deterministic: bool = True,
    ) -> int:
        """基于规则生成离散动作.

        target_param: (2,) 归一化参数 [-1, 1], 语义取决于技能类型:
          - navigate: 目标方向偏移 (dx, dy)
          - attack:   目标敌人方向 (dx, dy)
          - defend:   防守位置偏移 (dx, dy)
        """
        # 从观测中提取自身位置和方向
        self_x = local_obs[0]  # 归一化 [0, 1]
        self_y = local_obs[1]
        # 方向 one-hot: indices 2-5
        dir_oh = local_obs[2:6]
        current_dir = int(np.argmax(dir_oh))
        # 火力状态
        fire_cd = local_obs[7]
        has_bullet = local_obs[8]

        if self.skill_type == "navigate":
            return self._navigate_action(self_x, self_y, target_param, current_dir)
        elif self.skill_type == "attack":
            return self._attack_action(local_obs, target_param, current_dir, fire_cd, has_bullet)
        elif self.skill_type == "defend":
            return self._defend_action(local_obs, target_param, current_dir, fire_cd, has_bullet)
        return Action.NOOP

    def _navigate_action(
        self, self_x: float, self_y: float,
        target_param: np.ndarray, current_dir: int,
    ) -> int:
        """导航: 朝目标方向移动."""
        dx = target_param[0]
        dy = target_param[1]

        # 选择差距更大的轴优先移动
        if abs(dy) > abs(dx) + 0.05:
            return Action.DOWN if dy > 0 else Action.UP
        elif abs(dx) > abs(dy) + 0.05:
            return Action.RIGHT if dx > 0 else Action.LEFT
        else:
            # 接近等距, 随机选一个方向
            if abs(dx) < 0.02 and abs(dy) < 0.02:
                return Action.NOOP  # 已到达
            return Action.DOWN if dy > 0 else Action.UP

    def _attack_action(
        self, obs: np.ndarray, target_param: np.ndarray,
        current_dir: int, fire_cd: float, has_bullet: float,
    ) -> int:
        """攻击: 朝敌人移动, 主动对齐到同行/列后射击.

        增强逻辑 (v2):
          1. 射线检测到敌人 → 转向 + 射击
          2. 近距离(< 0.15) → 优先消除较小偏移，对齐后射击
          3. 同行/列(偏差 < 2格) → 转向射击
          4. 远距离 → 优先走能对齐的轴 (消除较小偏移优先)
        """
        can_fire = fire_cd <= 0 and has_bullet < 0.5
        dir_to_action = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]

        # 射线敌人距离 (obs[13:17] = UP, DOWN, LEFT, RIGHT)
        ray_enemy = obs[13:17]

        # 策略1: 射线检测到任意方向有敌人 → 立即转向 + 射击
        best_ray = -1
        best_ray_dist = 1.0
        for d_idx in range(4):
            if ray_enemy[d_idx] < 0.5 and ray_enemy[d_idx] < best_ray_dist:
                best_ray = d_idx
                best_ray_dist = ray_enemy[d_idx]
        if best_ray >= 0:
            if current_dir == best_ray and can_fire:
                return Action.FIRE
            return dir_to_action[best_ray]  # 转向

        # 目标方向 (敌人相对位置, 归一化到地图大小)
        dx, dy = target_param[0], target_param[1]
        abs_dx, abs_dy = abs(dx), abs(dy)
        dist = abs_dx + abs_dy

        # 1格 = 0.05 (20x20地图), 2格 = 0.10
        ALIGN_THRESHOLD = 0.10  # 2格内算"几乎对齐"

        # 策略2: 近距离 (< 5格) 且几乎同列/同行 → 转向射击
        if abs_dx < ALIGN_THRESHOLD and abs_dy > 0.02:
            # 几乎同列 → 纵向射击
            target_act = Action.DOWN if dy > 0 else Action.UP
            target_d = Dir.DOWN if dy > 0 else Dir.UP
            if current_dir == target_d and can_fire:
                return Action.FIRE
            return target_act

        if abs_dy < ALIGN_THRESHOLD and abs_dx > 0.02:
            # 几乎同行 → 横向射击
            target_act = Action.RIGHT if dx > 0 else Action.LEFT
            target_d = Dir.RIGHT if dx > 0 else Dir.LEFT
            if current_dir == target_d and can_fire:
                return Action.FIRE
            return target_act

        # 策略3: 中近距离 → 优先消除较小偏移 (更快进入对齐状态)
        # 例: dx=0.15, dy=0.30 → 先消除 dx (只需移3格就同列)
        if abs_dx <= abs_dy:
            # 水平偏移更小 → 先横向对齐 (进入同列后可纵向射击)
            if abs_dx > 0.02:
                return Action.RIGHT if dx > 0 else Action.LEFT
            # 已经同列了, 朝敌人走
            return Action.DOWN if dy > 0 else Action.UP
        else:
            # 纵向偏移更小 → 先纵向对齐 (进入同行后可横向射击)
            if abs_dy > 0.02:
                return Action.DOWN if dy > 0 else Action.UP
            # 已经同行了, 朝敌人走
            return Action.RIGHT if dx > 0 else Action.LEFT

    def _defend_action(
        self, obs: np.ndarray, target_param: np.ndarray,
        current_dir: int, fire_cd: float, has_bullet: float,
    ) -> int:
        """防守: 守在基地前方, 主动拦截接近的敌人.

        增强逻辑:
          1. 射线方向有敌人 → 转向射击 (不要求已对齐)
          2. 在基地前方 → 面朝上方封锁通道, 主动射击
          3. 不在基地附近 → 快速回防
        """
        can_fire = fire_cd <= 0 and has_bullet < 0.5
        dir_to_action = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]

        # 射线敌人距离
        ray_enemy = obs[13:17]  # UP, DOWN, LEFT, RIGHT

        # 策略1: 任意方向射线检测到敌人 → 转向 + 射击
        for d_idx in range(4):
            if ray_enemy[d_idx] < 0.5:
                if current_dir == d_idx and can_fire:
                    return Action.FIRE
                return dir_to_action[d_idx]  # 转向敌人

        # 基地方向
        base_dx = obs[27] if len(obs) > 27 else target_param[0]
        base_dy = obs[28] if len(obs) > 28 else target_param[1]

        # 目标: 基地前方 (稍微偏上)
        target_dx = target_param[0] if abs(target_param[0]) > 0.02 else base_dx
        target_dy = target_param[1] if abs(target_param[1]) > 0.02 else base_dy - 0.15

        dist = abs(target_dx) + abs(target_dy)

        # 策略2: 已在基地附近 → 面朝上方, 封锁通道射击
        if dist < 0.08:
            if current_dir != Dir.UP:
                return Action.UP  # 转向上方
            if can_fire:
                return Action.FIRE  # 封锁射击
            return Action.NOOP

        # 策略3: 不在基地附近 → 快速回防
        if abs(target_dy) >= abs(target_dx):
            return Action.DOWN if target_dy > 0 else Action.UP
        else:
            return Action.RIGHT if target_dx > 0 else Action.LEFT
