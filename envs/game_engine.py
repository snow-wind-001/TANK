"""
经典坦克大战 (Battle City) 游戏引擎

完整复刻 FC/NES 坦克大战核心机制:
  - 20x20 网格地图
  - 6 种地形: 砖墙(可摧毁), 钢墙(不可摧毁), 水面(不可通行), 冰面(滑行), 树林(隐蔽), 空地
  - 需要保护的基地 (鹰标志)
  - 坦克4方向移动 + 射击, 每辆坦克同时只有1颗子弹
  - 子弹飞行 + 碰撞检测 (摧毁砖墙/击杀坦克/摧毁基地)
  - 多种坦克类型 (普通/快速/重甲/装甲)
  - 敌方 AI (巡逻/追击/攻击基地)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np

# =====================================================================
#  常量
# =====================================================================

# 地形类型
class Tile(IntEnum):
    EMPTY = 0
    BRICK = 1      # 砖墙: 可被子弹摧毁
    STEEL = 2      # 钢墙: 不可摧毁
    WATER = 3      # 水面: 坦克不可通行, 子弹可飞越
    ICE = 4        # 冰面: 坦克滑行 (额外移动一格)
    TREE = 5       # 树林: 坦克可通行, 视觉隐蔽
    BASE = 6       # 基地: 被摧毁则失败


# 动作
class Action(IntEnum):
    NOOP = 0       # 不动
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    FIRE = 5       # 射击 (朝当前方向)


# 朝向
class Dir(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


# 动作 -> 朝向映射
ACTION_TO_DIR = {
    Action.UP: Dir.UP,
    Action.DOWN: Dir.DOWN,
    Action.LEFT: Dir.LEFT,
    Action.RIGHT: Dir.RIGHT,
}

# 朝向 -> 位移 (dx, dy), y轴向下
DIR_DELTA = {
    Dir.UP: (0, -1),
    Dir.DOWN: (0, 1),
    Dir.LEFT: (-1, 0),
    Dir.RIGHT: (1, 0),
}

# 坦克类型配置: (max_health, speed, fire_cooldown, bullet_speed)
TANK_TYPES = {
    "normal": (1, 1, 15, 2),
    "fast":   (1, 2, 15, 2),
    "heavy":  (2, 1, 20, 2),
    "armor":  (3, 1, 12, 3),
}

BULLET_DAMAGE = 1
MAP_W = 20
MAP_H = 20


# =====================================================================
#  数据结构
# =====================================================================

@dataclass
class Bullet:
    """子弹."""
    x: int
    y: int
    direction: Dir
    speed: int
    owner_id: int        # 所属坦克 ID
    owner_team: str      # 'red' | 'blue'
    alive: bool = True


@dataclass
class Tank:
    """坦克."""
    tank_id: int
    team: str            # 'red' | 'blue'
    tank_type: str = "normal"
    x: int = 0
    y: int = 0
    direction: Dir = Dir.UP
    health: int = 1
    max_health: int = 1
    speed: int = 1
    fire_cooldown: int = 0
    max_fire_cooldown: int = 15
    bullet_speed: int = 2
    alive: bool = True
    has_bullet: bool = False   # 当前子弹是否还在飞

    def reset_type(self) -> None:
        hp, spd, cd, bs = TANK_TYPES[self.tank_type]
        self.health = hp
        self.max_health = hp
        self.speed = spd
        self.max_fire_cooldown = cd
        self.bullet_speed = bs
        self.fire_cooldown = 0
        self.has_bullet = False
        self.alive = True


# =====================================================================
#  预定义地图
# =====================================================================

def _parse_map(text: str) -> np.ndarray:
    """解析文本地图. 字符映射: .=空, B=砖, S=钢, W=水, I=冰, T=树, E=基地."""
    char_map = {
        ".": Tile.EMPTY, "B": Tile.BRICK, "S": Tile.STEEL,
        "W": Tile.WATER, "I": Tile.ICE,   "T": Tile.TREE,
        "E": Tile.BASE,
    }
    lines = [l for l in text.strip().split("\n") if l.strip()]
    h = len(lines)
    w = len(lines[0].strip())
    grid = np.zeros((h, w), dtype=np.int8)
    for r, line in enumerate(lines):
        for c, ch in enumerate(line.strip()):
            grid[r, c] = char_map.get(ch, Tile.EMPTY)
    return grid


# 经典关卡 1 (20x20)
CLASSIC_MAP_1 = _parse_map("""
....................
..BB..BB..BB..BB..B.
..BB..BB..BB..BB..B.
..BB..BB.SSS.BB..BB.
..BB..BB.....BB..BB.
......BB..BB........
.BB.....BB.....BB...
.BB..WW.BB.WW..BB..
.....WW....WW.......
..BB....BB....BB....
..BB..BB..BB..BB....
......BB..BB........
.BB..TT......TT.BB.
.BB..TT......TT.BB.
.....TT..BB..TT....
..BB.....BB.....BB..
..........BB........
......BBBBBB........
......B.EE.B........
......BBBBBB........
""")

# 经典关卡 2 (20x20) - 更多水面和钢墙
CLASSIC_MAP_2 = _parse_map("""
....................
.BB..SS..BB..SS..BB.
.BB..SS..BB..SS..BB.
.BB......BB......BB.
......WW....WW......
..BB.WWW..BWWW.BB..
..BB.WWW..BWWW.BB..
......WW....WW......
.SS..BB..SS..BB..SS.
.......BB.BB........
..TT...BB.BB...TT..
..TT.............TT.
.BB..BB......BB..BB.
.......BB..BB.......
..SS...BB..BB...SS..
.......BB..BB.......
......BBBBBB........
......BBBBBB........
......B.EE.B........
......BBBBBB........
""")

# 经典关卡 3 (20x20) - 开阔战场
CLASSIC_MAP_3 = _parse_map("""
....................
..B...B..SS..B...B..
..B...B......B...B..
..B...B..BB..B...B..
..............BB....
.TTT..BB..BB..TTT...
.TTT..........TTT...
.TTT..II..II..TTT...
......II..II........
..BB......BB..BB....
..BB......BB..BB....
......BB......BB....
..SS..BB..BB..SS....
..................B.
..BB..TT..TT..BB...
..BB..TT..TT..BB...
..........BB........
......BBBBBB........
......B.EE.B........
......BBBBBB........
""")

MAPS = {
    "classic_1": CLASSIC_MAP_1,
    "classic_2": CLASSIC_MAP_2,
    "classic_3": CLASSIC_MAP_3,
}


def random_map(width: int = MAP_W, height: int = MAP_H, seed: int | None = None) -> np.ndarray:
    """随机生成地图, 保留底部基地区域."""
    rng = np.random.RandomState(seed)
    grid = np.zeros((height, width), dtype=np.int8)

    # 随机放置砖墙
    brick_mask = rng.rand(height - 4, width) < 0.18
    grid[:height - 4][brick_mask] = Tile.BRICK

    # 随机放置钢墙 (较少)
    steel_mask = rng.rand(height - 4, width) < 0.04
    grid[:height - 4][steel_mask] = Tile.STEEL

    # 随机水面区域 (2-3 个 2x2 水池)
    for _ in range(rng.randint(2, 4)):
        wx, wy = rng.randint(2, width - 3), rng.randint(3, height - 6)
        grid[wy:wy + 2, wx:wx + 2] = Tile.WATER

    # 随机树林 (2-3 片)
    for _ in range(rng.randint(2, 4)):
        tx, ty = rng.randint(1, width - 3), rng.randint(2, height - 6)
        grid[ty:ty + 2, tx:tx + 3] = Tile.TREE

    # 清理出生点区域 (顶部3行, 底部4行)
    grid[:2, :] = Tile.EMPTY
    grid[height - 4:, :] = Tile.EMPTY

    # 基地区域 (确保在边界内)
    bx = width // 2 - 1
    by = height - 4  # 留出底部空间
    grid[by, bx - 1:bx + 3] = Tile.BRICK        # 上墙
    grid[by + 1, bx - 1] = Tile.BRICK            # 左墙
    grid[by + 1, bx + 2] = Tile.BRICK            # 右墙
    grid[by + 2, bx - 1] = Tile.BRICK
    grid[by + 2, bx + 2] = Tile.BRICK
    grid[by + 3, bx - 1:bx + 3] = Tile.BRICK    # 下墙
    grid[by + 1, bx] = Tile.BASE
    grid[by + 1, bx + 1] = Tile.BASE
    grid[by + 2, bx] = Tile.BASE
    grid[by + 2, bx + 1] = Tile.BASE

    # 确保出生点无阻挡
    spawn_cols = [1, width // 2, width - 2]
    for sc in spawn_cols:
        grid[0:2, max(0, sc - 1):sc + 2] = Tile.EMPTY
    player_cols = [width // 2 - 3, width // 2 + 3]
    for pc in player_cols:
        grid[height - 3:height - 1, max(0, pc - 1):pc + 2] = Tile.EMPTY

    return grid


# =====================================================================
#  游戏引擎
# =====================================================================

class BattleCityEngine:
    """经典坦克大战游戏引擎.

    管理地图, 坦克, 子弹, 碰撞检测, 胜负判定.
    """

    def __init__(
        self,
        map_name: str = "classic_1",
        custom_map: np.ndarray | None = None,
    ) -> None:
        if custom_map is not None:
            self.initial_grid = custom_map.copy()
        elif map_name in MAPS:
            self.initial_grid = MAPS[map_name].copy()
        else:
            self.initial_grid = random_map()

        self.height, self.width = self.initial_grid.shape
        self.grid: np.ndarray = self.initial_grid.copy()
        self.tanks: list[Tank] = []
        self.bullets: list[Bullet] = []
        self.base_alive: bool = True
        self._next_tank_id: int = 0
        self._events: list[tuple] = []

        # 找到基地位置 (取中心)
        base_ys, base_xs = np.where(self.initial_grid == Tile.BASE)
        if len(base_xs) > 0:
            self.base_x = int(base_xs.mean())
            self.base_y = int(base_ys.mean())
        else:
            self.base_x = self.width // 2
            self.base_y = self.height - 2

    # ------------------------------------------------------------------
    #  初始化
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """重置引擎状态."""
        self.grid = self.initial_grid.copy()
        self.tanks.clear()
        self.bullets.clear()
        self.base_alive = True
        self._next_tank_id = 0
        self._events.clear()

    def add_tank(
        self,
        x: int,
        y: int,
        team: str,
        tank_type: str = "normal",
        direction: Dir = Dir.UP,
    ) -> Tank:
        """添加一辆坦克到场上."""
        tank = Tank(
            tank_id=self._next_tank_id,
            team=team,
            tank_type=tank_type,
            x=x,
            y=y,
            direction=direction,
        )
        tank.reset_type()
        if team == "blue":
            tank.direction = Dir.DOWN
        else:
            tank.direction = direction
        self._next_tank_id += 1
        self.tanks.append(tank)
        return tank

    # ------------------------------------------------------------------
    #  主循环
    # ------------------------------------------------------------------

    def step(self, actions: dict[int, int]) -> list[tuple]:
        """执行一个游戏帧.

        Args:
            actions: {tank_id: Action} 每辆坦克的动作.

        Returns:
            事件列表: [('brick_destroyed', x, y), ('tank_killed', team, id), ...]
        """
        self._events.clear()

        # 1. 处理坦克动作 (移动 + 射击)
        for tank in self.tanks:
            if not tank.alive:
                continue
            action = actions.get(tank.tank_id, Action.NOOP)
            action = Action(action)

            if action == Action.FIRE:
                self._fire(tank)
            elif action in ACTION_TO_DIR:
                tank.direction = ACTION_TO_DIR[action]
                self._move_tank(tank)
            # NOOP: 什么都不做

        # 2. 更新子弹 (每帧移动 bullet_speed 步)
        self._update_bullets()

        # 3. 冷却计时
        for tank in self.tanks:
            if tank.fire_cooldown > 0:
                tank.fire_cooldown -= 1

        return list(self._events)

    # ------------------------------------------------------------------
    #  移动
    # ------------------------------------------------------------------

    def _move_tank(self, tank: Tank) -> bool:
        """移动坦克, 返回是否成功."""
        dx, dy = DIR_DELTA[tank.direction]
        nx, ny = tank.x + dx, tank.y + dy

        if not self._can_move_to(nx, ny, tank):
            return False

        tank.x, tank.y = nx, ny

        # 冰面滑行: 额外移动一格
        if self.grid[ny, nx] == Tile.ICE:
            sx, sy = nx + dx, ny + dy
            if self._can_move_to(sx, sy, tank):
                tank.x, tank.y = sx, sy

        return True

    def _can_move_to(self, x: int, y: int, tank: Tank) -> bool:
        """检测坦克能否移动到 (x, y)."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        tile = self.grid[y, x]
        if tile in (Tile.BRICK, Tile.STEEL, Tile.WATER, Tile.BASE):
            return False
        # 检测其他坦克
        for other in self.tanks:
            if other is not tank and other.alive and other.x == x and other.y == y:
                return False
        return True

    # ------------------------------------------------------------------
    #  射击
    # ------------------------------------------------------------------

    def _fire(self, tank: Tank) -> None:
        """坦克射击 (每辆坦克同时只能有1颗子弹)."""
        if tank.fire_cooldown > 0 or tank.has_bullet:
            return
        dx, dy = DIR_DELTA[tank.direction]
        bx, by = tank.x + dx, tank.y + dy
        bullet = Bullet(
            x=bx, y=by,
            direction=tank.direction,
            speed=tank.bullet_speed,
            owner_id=tank.tank_id,
            owner_team=tank.team,
        )
        self.bullets.append(bullet)
        tank.has_bullet = True
        tank.fire_cooldown = tank.max_fire_cooldown

    # ------------------------------------------------------------------
    #  子弹更新
    # ------------------------------------------------------------------

    def _update_bullets(self) -> None:
        """更新所有子弹位置并检测碰撞."""
        # 子弹可能有不同速度, 取最大速度做多步模拟
        max_speed = max((b.speed for b in self.bullets), default=0)
        for _ in range(max_speed):
            for bullet in self.bullets:
                if not bullet.alive:
                    continue
                # 只有速度足够的子弹在这个子步移动
                if _ >= bullet.speed:
                    continue
                dx, dy = DIR_DELTA[bullet.direction]
                bullet.x += dx
                bullet.y += dy
                self._check_bullet_collision(bullet)

        # 清理死亡子弹并更新坦克 has_bullet 标记
        dead_owners = set()
        alive_bullets = []
        for b in self.bullets:
            if b.alive:
                alive_bullets.append(b)
            else:
                dead_owners.add(b.owner_id)
        self.bullets = alive_bullets
        for tank in self.tanks:
            if tank.tank_id in dead_owners:
                tank.has_bullet = False

    def _check_bullet_collision(self, bullet: Bullet) -> None:
        """检测子弹碰撞."""
        x, y = bullet.x, bullet.y

        # 出界
        if not (0 <= x < self.width and 0 <= y < self.height):
            bullet.alive = False
            return

        tile = self.grid[y, x]

        # 砖墙: 摧毁
        if tile == Tile.BRICK:
            self.grid[y, x] = Tile.EMPTY
            bullet.alive = False
            self._events.append(("brick_destroyed", x, y))
            return

        # 钢墙: 子弹消失
        if tile == Tile.STEEL:
            bullet.alive = False
            return

        # 基地: 摧毁基地
        if tile == Tile.BASE:
            self.grid[y, x] = Tile.EMPTY
            self.base_alive = False
            bullet.alive = False
            self._events.append(("base_destroyed", x, y))
            return

        # 水面 / 树林: 子弹穿过
        # (水面子弹飞越, 树林子弹穿过)

        # 坦克碰撞
        for tank in self.tanks:
            if not tank.alive:
                continue
            if tank.tank_id == bullet.owner_id:
                continue  # 不伤自己
            if tank.team == bullet.owner_team:
                continue  # 不伤友军
            if tank.x == x and tank.y == y:
                tank.health -= BULLET_DAMAGE
                bullet.alive = False
                if tank.health <= 0:
                    tank.alive = False
                    self._events.append(("tank_killed", tank.team, tank.tank_id))
                else:
                    self._events.append(("tank_hit", tank.team, tank.tank_id))
                return

        # 子弹互相碰撞
        for other in self.bullets:
            if other is bullet or not other.alive:
                continue
            if other.x == x and other.y == y:
                bullet.alive = False
                other.alive = False
                return

    # ------------------------------------------------------------------
    #  观测辅助
    # ------------------------------------------------------------------

    def raycast(self, x: int, y: int, direction: Dir, max_dist: int = 20) -> tuple[int, int]:
        """从 (x,y) 向 direction 发射射线, 返回 (距离, 碰到的类型).

        类型: 0=边界, 1=砖墙, 2=钢墙, 3=水面, 4=坦克, 5=基地, 6=空(到达max_dist)
        """
        dx, dy = DIR_DELTA[direction]
        for dist in range(1, max_dist + 1):
            nx, ny = x + dx * dist, y + dy * dist
            if not (0 <= nx < self.width and 0 <= ny < self.height):
                return dist, 0  # 边界
            tile = self.grid[ny, nx]
            if tile in (Tile.BRICK, Tile.STEEL):
                return dist, int(tile)
            if tile == Tile.WATER:
                return dist, 3
            if tile == Tile.BASE:
                return dist, 5
            # 检测坦克
            for tank in self.tanks:
                if tank.alive and tank.x == nx and tank.y == ny:
                    return dist, 4
        return max_dist, 6  # 未碰到任何物体

    def get_tanks_by_team(self, team: str) -> list[Tank]:
        return [t for t in self.tanks if t.team == team and t.alive]

    def get_nearest_enemy(self, tank: Tank) -> Optional[Tank]:
        enemies = [t for t in self.tanks if t.team != tank.team and t.alive]
        if not enemies:
            return None
        return min(enemies, key=lambda e: abs(e.x - tank.x) + abs(e.y - tank.y))

    def get_enemies_sorted(self, tank: Tank) -> list[Tank]:
        enemies = [t for t in self.tanks if t.team != tank.team and t.alive]
        return sorted(enemies, key=lambda e: abs(e.x - tank.x) + abs(e.y - tank.y))

    # ------------------------------------------------------------------
    #  蓝方 AI
    # ------------------------------------------------------------------

    def blue_ai_action(self, tank: Tank) -> int:
        """蓝方规则 AI: 巡逻 → 追击 → 攻击基地.

        行为 (已适度弱化, 给红方学习机会):
          1. 优先朝红方坦克或基地移动
          2. 前方有目标时有概率射击 (非 100%)
          3. 偶尔犹豫 (NOOP) 模拟反应时间
          4. 路径规划偶尔走非最优方向
        """
        rng = np.random

        # [弱化] 犹豫机制: 10% 概率什么都不做 (模拟反应时间)
        if rng.rand() < 0.10:
            return Action.NOOP

        # 找最近的红方坦克
        nearest_red = self.get_nearest_enemy(tank)
        target_x, target_y = self.base_x, self.base_y  # 默认目标: 基地

        if nearest_red is not None:
            dist_to_red = abs(nearest_red.x - tank.x) + abs(nearest_red.y - tank.y)
            if dist_to_red < 8:  # 近距离追击红方
                target_x, target_y = nearest_red.x, nearest_red.y

        # 判断是否应该射击: 前方有红方坦克或基地
        dx, dy = DIR_DELTA[tank.direction]
        for dist in range(1, 10):
            fx, fy = tank.x + dx * dist, tank.y + dy * dist
            if not (0 <= fx < self.width and 0 <= fy < self.height):
                break
            tile = self.grid[fy, fx]
            if tile in (Tile.BRICK, Tile.STEEL):
                # 前方有墙, 一定概率射击摧毁砖墙
                if tile == Tile.BRICK and rng.rand() < 0.2:
                    return Action.FIRE
                break
            # 前方有红方坦克
            for red in self.tanks:
                if red.alive and red.team == "red" and red.x == fx and red.y == fy:
                    # [弱化] 70% 概率射击 (原 100%)
                    if rng.rand() < 0.70:
                        return Action.FIRE
                    break  # 看到了但没射
            # 前方是基地? (蓝方的目标是摧毁基地)

        # 朝目标方向移动
        diff_x = target_x - tank.x
        diff_y = target_y - tank.y

        # [弱化] 15% 概率走随机方向 (降低路径规划效率)
        if rng.rand() < 0.15:
            return rng.choice([Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT])

        # 优先选择差距更大的轴
        possible_actions = []
        if abs(diff_y) >= abs(diff_x):
            if diff_y > 0:
                possible_actions.append(Action.DOWN)
            elif diff_y < 0:
                possible_actions.append(Action.UP)
            if diff_x > 0:
                possible_actions.append(Action.RIGHT)
            elif diff_x < 0:
                possible_actions.append(Action.LEFT)
        else:
            if diff_x > 0:
                possible_actions.append(Action.RIGHT)
            elif diff_x < 0:
                possible_actions.append(Action.LEFT)
            if diff_y > 0:
                possible_actions.append(Action.DOWN)
            elif diff_y < 0:
                possible_actions.append(Action.UP)

        # 尝试移动, 如果被阻挡则尝试其他方向
        for act in possible_actions:
            d = ACTION_TO_DIR[act]
            ddx, ddy = DIR_DELTA[d]
            if self._can_move_to(tank.x + ddx, tank.y + ddy, tank):
                # 移动时偶尔射击 (降低概率)
                if rng.rand() < 0.08:
                    return Action.FIRE
                return act

        # 全部方向被挡, 随机转向或射击
        if rng.rand() < 0.3:
            return Action.FIRE
        return rng.choice([Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT])
