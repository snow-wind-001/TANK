"""
经典坦克大战 - 单坦克技能训练环境

基于 BattleCityEngine, 提供 Gymnasium 标准接口.
支持三种底层技能训练模式:
  - 'navigate': 在地图中导航到指定目标点 (避障移动)
  - 'attack':   追击并消灭指定敌方坦克 (靠近+射击)
  - 'defend':   保护基地, 拦截朝基地移动的敌人

观测空间 (29 维):
  [self_x, self_y, dir_onehot(4), health_ratio, fire_cd_ratio, has_bullet,  # 9
   ray_wall_4dirs(4), ray_enemy_4dirs(4),                                     # 8
   nearest_enemy: dx, dy, health (3),                                         # 3
   second_enemy:  dx, dy, health (3),                                         # 3
   base: dx, dy, alive (3),                                                   # 3
   n_enemies_alive_ratio (1),                                                  # 1
   target_dx, target_dy (2)]                                                   # 2

动作空间: Discrete(6) = [NOOP, UP, DOWN, LEFT, RIGHT, FIRE]
"""

from __future__ import annotations

import math
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.game_engine import (
    Action, BattleCityEngine, Dir, DIR_DELTA, Tile, Tank, MAP_W, MAP_H,
)

OBS_DIM = 29


class SingleTankSkillEnv(gym.Env):
    """单坦克技能训练 Gymnasium 环境."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        skill_type: str = "navigate",
        map_name: str = "classic_1",
        max_steps: int = 300,
        n_enemies: int = 3,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        assert skill_type in ("navigate", "attack", "defend"), f"Unknown skill: {skill_type}"
        self.skill_type = skill_type
        self.map_name = map_name
        self.max_steps = max_steps
        self.n_enemies = n_enemies
        self.render_mode = render_mode

        self.observation_space = spaces.Box(-1.0, 1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(6)

        self.engine = BattleCityEngine(map_name=map_name)
        self._step_count = 0
        self._player_tank: Optional[Tank] = None
        self._target_x: int = 0
        self._target_y: int = 0

    # ------------------------------------------------------------------
    #  Gymnasium 接口
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        rng = self.np_random
        self.engine.reset()
        self._step_count = 0

        # 红方坦克 (玩家) 在基地附近出生
        px = self.engine.base_x - 3 + rng.integers(0, 7)
        px = int(np.clip(px, 1, self.engine.width - 2))
        py = int(np.clip(self.engine.height - 3, 1, self.engine.height - 2))
        # 确保出生点可用
        while not self.engine._can_move_to(px, py, Tank(tank_id=-1, team="none")):
            px = int(rng.integers(1, self.engine.width - 1))
            py = int(rng.integers(self.engine.height - 5, self.engine.height - 2))
        self._player_tank = self.engine.add_tank(px, py, "red", "normal")

        # 蓝方敌人在顶部出生
        spawn_xs = [1, self.engine.width // 2, self.engine.width - 2]
        for i in range(self.n_enemies):
            ex = spawn_xs[i % len(spawn_xs)]
            ey = 0
            while not self.engine._can_move_to(ex, ey, Tank(tank_id=-1, team="none")):
                ex = int(rng.integers(1, self.engine.width - 1))
                ey = int(rng.integers(0, 3))
            self.engine.add_tank(ex, ey, "blue", rng.choice(["normal", "fast"]))

        # 设定技能目标
        self._set_target(rng)

        return self._obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._step_count += 1

        # 构建动作字典
        actions: dict[int, int] = {self._player_tank.tank_id: action}
        # 蓝方 AI 控制
        for tank in self.engine.tanks:
            if tank.team == "blue" and tank.alive:
                actions[tank.tank_id] = self.engine.blue_ai_action(tank)

        events = self.engine.step(actions)

        # 计算奖励
        reward = self._compute_reward(events)

        # 终止条件
        terminated = False
        if not self._player_tank.alive:
            reward -= 10.0
            terminated = True
        if not self.engine.base_alive:
            reward -= 20.0
            terminated = True
        # 技能完成
        if self._check_skill_done():
            reward += 15.0
            terminated = True

        truncated = self._step_count >= self.max_steps
        info = {
            "step": self._step_count,
            "base_alive": self.engine.base_alive,
            "player_alive": self._player_tank.alive,
            "enemies_alive": len(self.engine.get_tanks_by_team("blue")),
            "events": events,
        }
        return self._obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    #  技能目标
    # ------------------------------------------------------------------

    def _set_target(self, rng: np.random.Generator) -> None:
        if self.skill_type == "navigate":
            # 随机目标点 (确保可通行)
            for _ in range(50):
                tx = int(rng.integers(1, self.engine.width - 1))
                ty = int(rng.integers(1, self.engine.height - 3))
                if self.engine.grid[ty, tx] == Tile.EMPTY:
                    self._target_x, self._target_y = tx, ty
                    return
            self._target_x, self._target_y = self.engine.width // 2, self.engine.height // 2

        elif self.skill_type == "attack":
            # 目标: 最近的敌人位置
            nearest = self.engine.get_nearest_enemy(self._player_tank)
            if nearest:
                self._target_x, self._target_y = nearest.x, nearest.y
            else:
                self._target_x, self._target_y = self.engine.width // 2, 0

        elif self.skill_type == "defend":
            # 目标: 基地位置
            self._target_x = self.engine.base_x
            self._target_y = self.engine.base_y

    def _check_skill_done(self) -> bool:
        if self.skill_type == "navigate":
            return (self._player_tank.x == self._target_x and
                    self._player_tank.y == self._target_y)
        elif self.skill_type == "attack":
            return len(self.engine.get_tanks_by_team("blue")) == 0
        elif self.skill_type == "defend":
            return len(self.engine.get_tanks_by_team("blue")) == 0
        return False

    # ------------------------------------------------------------------
    #  奖励
    # ------------------------------------------------------------------

    def _compute_reward(self, events: list[tuple]) -> float:
        """计算奖励 (含增强奖励塑形).

        通用事件奖励 + 技能专属奖励塑形:
          attack: 对齐射击奖励, 靠近敌人奖励
          defend: 封锁位置奖励, 离岗惩罚
          navigate: 靠近目标奖励
        """
        reward = -0.01  # 时间惩罚

        tank = self._player_tank
        if not tank.alive:
            return reward

        for event in events:
            if event[0] == "tank_killed" and event[1] == "blue":
                reward += 8.0   # 击杀蓝方
            if event[0] == "tank_hit" and event[1] == "blue":
                reward += 2.0   # 命中蓝方 (多血量坦克)
            if event[0] == "tank_killed" and event[1] == "red":
                reward -= 5.0   # 己方被击杀
            if event[0] == "base_destroyed":
                reward -= 15.0  # 基地被摧毁

        if self.skill_type == "navigate":
            # 靠近目标奖励
            dist = abs(tank.x - self._target_x) + abs(tank.y - self._target_y)
            reward -= 0.01 * dist

        elif self.skill_type == "attack":
            # 更新攻击目标 (追踪最近敌人)
            nearest = self.engine.get_nearest_enemy(tank)
            if nearest:
                self._target_x, self._target_y = nearest.x, nearest.y
                dist = abs(tank.x - nearest.x) + abs(tank.y - nearest.y)
                reward -= 0.005 * dist  # 靠近敌人

                # [奖励塑形] 对齐奖励: 与敌人同行或同列时额外奖励
                if tank.x == nearest.x or tank.y == nearest.y:
                    reward += 0.1  # 在射击线上

                    # 面朝敌人方向时更高奖励
                    dx = nearest.x - tank.x
                    dy = nearest.y - tank.y
                    facing_enemy = False
                    if dx > 0 and int(tank.direction) == 3:  # RIGHT
                        facing_enemy = True
                    elif dx < 0 and int(tank.direction) == 2:  # LEFT
                        facing_enemy = True
                    elif dy > 0 and int(tank.direction) == 1:  # DOWN
                        facing_enemy = True
                    elif dy < 0 and int(tank.direction) == 0:  # UP
                        facing_enemy = True
                    if facing_enemy:
                        reward += 0.05  # 面朝敌人且在射线上

        elif self.skill_type == "defend":
            base_x, base_y = self.engine.base_x, self.engine.base_y
            dist_base = abs(tank.x - base_x) + abs(tank.y - base_y)

            # 基础: 在基地附近奖励
            if dist_base < 5:
                reward += 0.02
            else:
                reward -= 0.005 * dist_base

            # [奖励塑形] 封锁位置奖励: 在基地上方 3 格内面朝上方
            if (abs(tank.x - base_x) <= 2
                    and 0 < base_y - tank.y <= 3
                    and int(tank.direction) == 0):  # UP
                reward += 0.05  # 理想封锁位置

            # [奖励塑形] 离岗惩罚: 敌人接近基地但自己不在附近
            enemies = self.engine.get_tanks_by_team("blue")
            for enemy in enemies:
                enemy_to_base = abs(enemy.x - base_x) + abs(enemy.y - base_y)
                if enemy_to_base < 5 and dist_base > 6:
                    reward -= 0.1  # 敌人逼近基地但自己不在防守
                    break  # 只惩罚一次

        return reward

    # ------------------------------------------------------------------
    #  观测
    # ------------------------------------------------------------------

    def _obs(self) -> np.ndarray:
        tank = self._player_tank
        W, H = self.engine.width, self.engine.height

        # 自身状态
        self_x = tank.x / W if tank.alive else 0.0
        self_y = tank.y / H if tank.alive else 0.0
        dir_oh = [0.0] * 4
        if tank.alive:
            dir_oh[int(tank.direction)] = 1.0
        hp = tank.health / tank.max_health if tank.alive else 0.0
        cd = tank.fire_cooldown / tank.max_fire_cooldown if tank.alive else 0.0
        has_b = 1.0 if tank.has_bullet else 0.0

        # 射线检测 (4方向)
        ray_wall = [0.0] * 4
        ray_enemy = [0.0] * 4
        if tank.alive:
            for i, d in enumerate([Dir.UP, Dir.DOWN, Dir.LEFT, Dir.RIGHT]):
                dist_w, _ = self.engine.raycast(tank.x, tank.y, d)
                ray_wall[i] = dist_w / max(W, H)

                # 检测该方向最近敌人距离
                dx, dy = DIR_DELTA[d]
                min_enemy_dist = max(W, H)
                for enemy in self.engine.get_tanks_by_team("blue"):
                    if d == Dir.UP and enemy.x == tank.x and enemy.y < tank.y:
                        min_enemy_dist = min(min_enemy_dist, tank.y - enemy.y)
                    elif d == Dir.DOWN and enemy.x == tank.x and enemy.y > tank.y:
                        min_enemy_dist = min(min_enemy_dist, enemy.y - tank.y)
                    elif d == Dir.LEFT and enemy.y == tank.y and enemy.x < tank.x:
                        min_enemy_dist = min(min_enemy_dist, tank.x - enemy.x)
                    elif d == Dir.RIGHT and enemy.y == tank.y and enemy.x > tank.x:
                        min_enemy_dist = min(min_enemy_dist, enemy.x - tank.x)
                ray_enemy[i] = min_enemy_dist / max(W, H)

        # 最近两个敌人
        enemies = self.engine.get_enemies_sorted(tank)
        e1_dx, e1_dy, e1_hp = 0.0, 0.0, 0.0
        e2_dx, e2_dy, e2_hp = 0.0, 0.0, 0.0
        if len(enemies) >= 1:
            e1 = enemies[0]
            e1_dx = (e1.x - tank.x) / W
            e1_dy = (e1.y - tank.y) / H
            e1_hp = e1.health / e1.max_health
        if len(enemies) >= 2:
            e2 = enemies[1]
            e2_dx = (e2.x - tank.x) / W
            e2_dy = (e2.y - tank.y) / H
            e2_hp = e2.health / e2.max_health

        # 基地信息
        base_dx = (self.engine.base_x - tank.x) / W if tank.alive else 0.0
        base_dy = (self.engine.base_y - tank.y) / H if tank.alive else 0.0
        base_alive = 1.0 if self.engine.base_alive else 0.0

        # 敌人存活比
        max_enemies = max(1, self.n_enemies)
        n_alive = len(enemies) / max_enemies

        # 目标参数
        target_dx = (self._target_x - tank.x) / W if tank.alive else 0.0
        target_dy = (self._target_y - tank.y) / H if tank.alive else 0.0

        obs = np.array([
            self_x, self_y,
            *dir_oh,
            hp, cd, has_b,
            *ray_wall,
            *ray_enemy,
            e1_dx, e1_dy, e1_hp,
            e2_dx, e2_dy, e2_hp,
            base_dx, base_dy, base_alive,
            n_alive,
            target_dx, target_dy,
        ], dtype=np.float32)

        return obs
