"""
经典坦克大战 - 多智能体团队协同环境

红方 (RL 控制, 2辆坦克) vs 蓝方 (规则 AI, 波次进攻)
目标: 保护基地 + 消灭所有蓝方坦克

高层 MADDPG 接口:
  obs = env.reset()          -> List[np.ndarray]
  obs, reward, done, info = env.step(actions: List[int])

每个红方坦克的观测 (OBS_DIM=32):
  [self_x, self_y, dir_onehot(4), health_ratio, fire_cd_ratio, has_bullet,  # 9
   ray_wall_4dirs(4), ray_enemy_4dirs(4),                                     # 8
   nearest_enemy: dx, dy, health (3),                                         # 3
   second_enemy:  dx, dy, health (3),                                         # 3
   ally: dx, dy, health, alive (4),                                           # 4
   base: dx, dy, alive (3),                                                   # 3
   n_enemies_alive_ratio, n_allies_alive_ratio (2)]                            # 2

动作空间: Discrete(6) per agent = [NOOP, UP, DOWN, LEFT, RIGHT, FIRE]
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from envs.game_engine import (
    Action, BattleCityEngine, Dir, DIR_DELTA, Tile, Tank, MAP_W, MAP_H,
)

OBS_DIM = 32


class MultiTankTeamEnv:
    """2v(N) 红蓝团队对抗环境, 完整坦克大战体验.

    红方: 2辆坦克, RL 控制
    蓝方: 波次进攻的敌方坦克, 规则 AI 控制
    胜利: 消灭所有蓝方坦克
    失败: 基地被摧毁 或 红方全灭

    difficulty:
      - "easy":   2v2, 蓝方全 normal, 蓝方 AI 弱化
      - "medium": 2v3, 蓝方混合类型
      - "hard":   2v4, 蓝方含 heavy/armor (原始设定)
    """

    # 难度预设: (n_blue, max_on_field, spawn_interval, blue_types)
    DIFFICULTY_PRESETS: dict[str, dict] = {
        "easy":   {"n_blue": 2, "max_on_field": 2, "spawn_interval": 120,
                   "blue_types": ["normal"] * 10},
        "medium": {"n_blue": 3, "max_on_field": 3, "spawn_interval": 100,
                   "blue_types": ["normal"] * 7 + ["fast"] * 3},
        "hard":   {"n_blue": 4, "max_on_field": 4, "spawn_interval": 80,
                   "blue_types": ["normal"] * 5 + ["fast"] * 3 + ["heavy"] * 1 + ["armor"] * 1},
    }

    def __init__(
        self,
        map_name: str = "classic_1",
        n_red: int = 2,
        n_blue: int = 4,
        max_steps: int = 600,
        blue_spawn_interval: int = 80,
        max_blue_on_field: int = 4,
        difficulty: Optional[str] = None,
    ) -> None:
        self.map_name = map_name
        self.n_red = n_red
        self.max_steps = max_steps
        self.difficulty = difficulty

        # 如果指定了 difficulty, 用预设覆盖参数
        if difficulty is not None and difficulty in self.DIFFICULTY_PRESETS:
            preset = self.DIFFICULTY_PRESETS[difficulty]
            self.n_blue_total = preset["n_blue"]
            self.max_blue_on_field = preset["max_on_field"]
            self.blue_spawn_interval = preset["spawn_interval"]
            self._blue_types = preset["blue_types"]
        else:
            self.n_blue_total = n_blue
            self.max_blue_on_field = max_blue_on_field
            self.blue_spawn_interval = blue_spawn_interval
            self._blue_types = ["normal"] * 5 + ["fast"] * 3 + ["heavy"] * 1 + ["armor"] * 1

        self.engine = BattleCityEngine(map_name=map_name)
        self._step_count = 0
        self._blue_spawned = 0
        self._blue_killed = 0
        self._spawn_timer = 0

        self.red_tanks: list[Tank] = []
        self.red_ids: list[int] = []

    @property
    def obs_dim(self) -> int:
        return OBS_DIM

    # ------------------------------------------------------------------
    #  公开接口
    # ------------------------------------------------------------------

    def reset(self) -> list[np.ndarray]:
        """重置环境, 返回每个红方坦克的观测."""
        self.engine.reset()
        self._step_count = 0
        self._blue_spawned = 0
        self._blue_killed = 0
        self._spawn_timer = 0

        # 红方在基地附近出生
        W, H = self.engine.width, self.engine.height
        red_spawns = [
            (W // 2 - 3, H - 3),
            (W // 2 + 3, H - 3),
        ]
        self.red_tanks = []
        self.red_ids = []
        for i in range(self.n_red):
            x, y = red_spawns[i % len(red_spawns)]
            x = int(np.clip(x, 1, W - 2))
            y = int(np.clip(y, 1, H - 2))
            # 确保出生点可用
            while not self.engine._can_move_to(x, y, Tank(tank_id=-1, team="none")):
                x = int(np.random.randint(1, W - 1))
                y = int(np.random.randint(H - 5, H - 2))
            tank = self.engine.add_tank(x, y, "red", "normal", direction=Dir.UP)
            self.red_tanks.append(tank)
            self.red_ids.append(tank.tank_id)

        # 蓝方初始波次
        self._spawn_blue_wave(count=min(2, self.n_blue_total))

        return self._get_obs()

    def step(
        self, red_actions: list[int]
    ) -> tuple[list[np.ndarray], float, bool, dict[str, Any]]:
        """执行一步环境交互.

        Args:
            red_actions: 每个红方坦克的动作 (Discrete 0-5).

        Returns:
            next_obs, team_reward, done, info
        """
        self._step_count += 1

        # 构建动作字典
        actions: dict[int, int] = {}
        for i, tank in enumerate(self.red_tanks):
            if tank.alive:
                actions[tank.tank_id] = int(red_actions[i])
            else:
                actions[tank.tank_id] = Action.NOOP

        # 蓝方 AI
        for tank in self.engine.tanks:
            if tank.team == "blue" and tank.alive:
                actions[tank.tank_id] = self.engine.blue_ai_action(tank)

        # 引擎步进
        events = self.engine.step(actions)

        # 蓝方补员
        self._spawn_timer += 1
        if self._spawn_timer >= self.blue_spawn_interval:
            self._spawn_timer = 0
            self._spawn_blue_wave(count=1)

        # 计算团队奖励
        team_reward = self._compute_team_reward(events)

        # 更新击杀计数
        for ev in events:
            if ev[0] == "tank_killed" and ev[1] == "blue":
                self._blue_killed += 1

        # 终止条件
        done = False
        win = False

        # 基地被摧毁
        if not self.engine.base_alive:
            team_reward -= 30.0
            done = True

        # 红方全灭
        if all(not t.alive for t in self.red_tanks):
            team_reward -= 20.0
            done = True

        # 蓝方全部消灭 (已出完 + 场上没有存活)
        if self._blue_spawned >= self.n_blue_total:
            blue_alive = self.engine.get_tanks_by_team("blue")
            if len(blue_alive) == 0:
                team_reward += 30.0
                done = True
                win = True

        # 超时
        if self._step_count >= self.max_steps:
            done = True

        info = {
            "step": self._step_count,
            "base_alive": self.engine.base_alive,
            "red_alive": sum(1 for t in self.red_tanks if t.alive),
            "blue_alive": len(self.engine.get_tanks_by_team("blue")),
            "blue_killed": self._blue_killed,
            "blue_total": self.n_blue_total,
            "win": win,
            "events": events,
        }

        return self._get_obs(), team_reward, done, info

    # ------------------------------------------------------------------
    #  蓝方波次系统
    # ------------------------------------------------------------------

    def _spawn_blue_wave(self, count: int = 1) -> None:
        """在顶部出生点生成蓝方坦克."""
        W = self.engine.width
        spawn_points = [(1, 0), (W // 2, 0), (W - 2, 0)]
        rng = np.random

        blue_on_field = len(self.engine.get_tanks_by_team("blue"))
        for _ in range(count):
            if self._blue_spawned >= self.n_blue_total:
                return
            if blue_on_field >= self.max_blue_on_field:
                return

            # 选择出生点
            sp = spawn_points[self._blue_spawned % len(spawn_points)]
            sx, sy = sp
            # 确保不重叠
            attempts = 0
            while not self.engine._can_move_to(sx, sy, Tank(tank_id=-1, team="none")):
                sx = int(rng.randint(max(0, sp[0] - 2), min(W, sp[0] + 3)))
                sy = int(rng.randint(0, 3))
                attempts += 1
                if attempts > 20:
                    return  # 暂时放弃

            # 坦克类型 (根据 difficulty 预设或默认配置)
            btype = rng.choice(self._blue_types)
            self.engine.add_tank(sx, sy, "blue", btype, direction=Dir.DOWN)
            self._blue_spawned += 1
            blue_on_field += 1

    # ------------------------------------------------------------------
    #  奖励
    # ------------------------------------------------------------------

    def _compute_team_reward(self, events: list[tuple]) -> float:
        reward = -0.01  # 时间惩罚

        for ev in events:
            if ev[0] == "tank_killed":
                if ev[1] == "blue":
                    reward += 10.0   # 击杀蓝方
                elif ev[1] == "red":
                    reward -= 8.0    # 红方阵亡
            elif ev[0] == "tank_hit":
                if ev[1] == "blue":
                    reward += 2.0    # 命中蓝方
                elif ev[1] == "red":
                    reward -= 1.0    # 被命中惩罚
            elif ev[0] == "brick_destroyed":
                pass  # 摧毁砖墙中性
            elif ev[0] == "base_destroyed":
                reward -= 20.0

        # 基地安全奖励
        if self.engine.base_alive:
            reward += 0.01

        # === 奖励塑形: 提供更密集的学习信号 ===
        W, H = self.engine.width, self.engine.height
        max_dist = W + H

        enemies_alive = self.engine.get_tanks_by_team("blue")
        for tank in self.red_tanks:
            if not tank.alive:
                continue

            # 1. 接近敌人奖励 (鼓励主动接敌)
            if enemies_alive:
                dists = [abs(e.x - tank.x) + abs(e.y - tank.y) for e in enemies_alive]
                min_dist = min(dists)
                # 越近奖励越高, 最大 +0.02/步
                reward += 0.02 * (1.0 - min_dist / max_dist)

            # 2. 基地防御奖励 (敌人靠近基地时, 红方应在基地附近)
            for enemy in enemies_alive:
                enemy_to_base = abs(enemy.x - self.engine.base_x) + abs(enemy.y - self.engine.base_y)
                if enemy_to_base < 6:  # 敌人靠近基地
                    tank_to_base = abs(tank.x - self.engine.base_x) + abs(tank.y - self.engine.base_y)
                    if tank_to_base < 8:  # 红方在基地附近防守
                        reward += 0.01

        return reward

    # ------------------------------------------------------------------
    #  观测
    # ------------------------------------------------------------------

    def _get_obs(self) -> list[np.ndarray]:
        """为每个红方坦克生成观测."""
        obs_list: list[np.ndarray] = []
        for idx, tank in enumerate(self.red_tanks):
            obs_list.append(self._build_obs(tank, idx))
        return obs_list

    def _build_obs(self, tank: Tank, idx: int) -> np.ndarray:
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

        # 射线 (4方向墙壁距离 + 4方向敌人距离)
        ray_wall = [1.0] * 4
        ray_enemy = [1.0] * 4
        if tank.alive:
            for i, d in enumerate([Dir.UP, Dir.DOWN, Dir.LEFT, Dir.RIGHT]):
                dist_w, _ = self.engine.raycast(tank.x, tank.y, d)
                ray_wall[i] = dist_w / max(W, H)

                dx, dy = DIR_DELTA[d]
                min_ed = max(W, H)
                for enemy in self.engine.get_tanks_by_team("blue"):
                    if d == Dir.UP and enemy.x == tank.x and enemy.y < tank.y:
                        min_ed = min(min_ed, tank.y - enemy.y)
                    elif d == Dir.DOWN and enemy.x == tank.x and enemy.y > tank.y:
                        min_ed = min(min_ed, enemy.y - tank.y)
                    elif d == Dir.LEFT and enemy.y == tank.y and enemy.x < tank.x:
                        min_ed = min(min_ed, tank.x - enemy.x)
                    elif d == Dir.RIGHT and enemy.y == tank.y and enemy.x > tank.x:
                        min_ed = min(min_ed, enemy.x - tank.x)
                ray_enemy[i] = min_ed / max(W, H)

        # 最近两个敌人
        enemies = self.engine.get_enemies_sorted(tank)
        e1_dx, e1_dy, e1_hp = 0.0, 0.0, 0.0
        e2_dx, e2_dy, e2_hp = 0.0, 0.0, 0.0
        if tank.alive and len(enemies) >= 1:
            e1 = enemies[0]
            e1_dx = (e1.x - tank.x) / W
            e1_dy = (e1.y - tank.y) / H
            e1_hp = e1.health / e1.max_health
        if tank.alive and len(enemies) >= 2:
            e2 = enemies[1]
            e2_dx = (e2.x - tank.x) / W
            e2_dy = (e2.y - tank.y) / H
            e2_hp = e2.health / e2.max_health

        # 友方信息
        ally = self.red_tanks[1 - idx] if self.n_red == 2 else None
        ally_dx, ally_dy, ally_hp, ally_alive = 0.0, 0.0, 0.0, 0.0
        if ally is not None and tank.alive:
            ally_dx = (ally.x - tank.x) / W
            ally_dy = (ally.y - tank.y) / H
            ally_hp = ally.health / ally.max_health if ally.alive else 0.0
            ally_alive = 1.0 if ally.alive else 0.0

        # 基地信息
        base_dx = (self.engine.base_x - tank.x) / W if tank.alive else 0.0
        base_dy = (self.engine.base_y - tank.y) / H if tank.alive else 0.0
        base_alive = 1.0 if self.engine.base_alive else 0.0

        # 统计
        n_enemies = len(enemies) / max(1, self.n_blue_total)
        n_allies = sum(1 for t in self.red_tanks if t.alive) / max(1, self.n_red)

        obs = np.array([
            self_x, self_y,
            *dir_oh,
            hp, cd, has_b,
            *ray_wall,
            *ray_enemy,
            e1_dx, e1_dy, e1_hp,
            e2_dx, e2_dy, e2_hp,
            ally_dx, ally_dy, ally_hp, ally_alive,
            base_dx, base_dy, base_alive,
            n_enemies, n_allies,
        ], dtype=np.float32)

        return obs
