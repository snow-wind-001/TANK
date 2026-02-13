"""
经典坦克大战 Pygame 可视化渲染器 (含协作可视化)

完整复刻经典视觉风格:
  - 瓦片地图: 砖墙(棕红), 钢墙(银白), 水面(蓝色动画), 冰面(浅蓝), 树林(绿色), 基地(鹰)
  - 坦克精灵: 三角形 + 方向指示, 红方红色, 蓝方蓝色
  - 子弹: 小方块
  - 底部信息面板: 回合/步数/奖励/存活统计

协作可视化:
  - 协作连线: 坦克间的注意力权重连线 (颜色/粗细表示协作强度)
  - 技能标签: 坦克上方显示当前执行的技能名称
  - 坦克编号: 每个坦克显示唯一编号
  - 注意力矩阵: 右下角热力图
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from envs.game_engine import BattleCityEngine, Dir, Tile, Tank, Bullet

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

# ---- 经典配色 ----
COLOR_BG = (0, 0, 0)           # 黑色背景 (经典)
COLOR_BRICK = (178, 89, 0)     # 砖墙棕色
COLOR_BRICK_DARK = (139, 69, 0)
COLOR_STEEL = (192, 192, 192)  # 钢墙银色
COLOR_STEEL_DARK = (128, 128, 128)
COLOR_WATER = (0, 100, 200)    # 水面蓝色
COLOR_WATER_LIGHT = (50, 140, 240)
COLOR_ICE = (180, 220, 255)    # 冰面浅蓝
COLOR_TREE = (0, 128, 0)       # 树林绿色
COLOR_TREE_LIGHT = (0, 180, 0)
COLOR_BASE = (255, 215, 0)     # 基地金色 (鹰)
COLOR_BASE_DEAD = (100, 100, 100)

COLOR_RED_TANK = (220, 50, 50)
COLOR_RED_DARK = (180, 30, 30)
COLOR_BLUE_TANK = (50, 100, 220)
COLOR_BLUE_DARK = (30, 70, 180)
COLOR_BULLET_RED = (255, 200, 100)
COLOR_BULLET_BLUE = (100, 200, 255)

COLOR_WHITE = (240, 240, 240)
COLOR_GRAY = (150, 150, 150)
COLOR_PANEL_BG = (20, 20, 30)
COLOR_YELLOW = (255, 220, 50)
COLOR_GREEN = (50, 220, 50)

# 协作可视化配色
COLOR_COOP_HIGH = (255, 220, 50)     # 强协作 (黄色)
COLOR_COOP_MID = (100, 200, 100)     # 中等协作 (绿色)
COLOR_COOP_LOW = (60, 80, 140)       # 弱协作 (蓝灰)
COLOR_SKILL_NAV = (80, 200, 255)     # 导航技能 (浅蓝)
COLOR_SKILL_ATK = (255, 80, 80)      # 攻击技能 (红)
COLOR_SKILL_DEF = (80, 255, 120)     # 防守技能 (绿)
COLOR_TANK_ID_BG = (0, 0, 0, 160)   # ID 背景半透明黑

SKILL_NAMES = {0: "导航", 1: "攻击", 2: "防守"}
SKILL_COLORS = {0: COLOR_SKILL_NAV, 1: COLOR_SKILL_ATK, 2: COLOR_SKILL_DEF}


class TankRenderer:
    """经典坦克大战 Pygame 渲染器."""

    def __init__(
        self,
        cell_size: int = 32,
        fps: int = 15,
        engine: Optional[BattleCityEngine] = None,
    ) -> None:
        if not PYGAME_AVAILABLE:
            raise ImportError("Pygame 未安装. 请运行: pip install pygame")

        self.cell_size = cell_size
        self.fps = fps
        self.engine = engine

        # 窗口尺寸由引擎地图决定, 在 init_display 中设置
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.font_small: Optional[pygame.font.Font] = None
        self.font_medium: Optional[pygame.font.Font] = None
        self.font_large: Optional[pygame.font.Font] = None
        self._initialized = False
        self._init_failed = False
        self._frame_count = 0

        self.panel_height = 100

    def _init_display(self, engine: BattleCityEngine) -> bool:
        """延迟初始化 Pygame (首次渲染时调用).

        Returns:
            True 初始化成功, False 初始化失败 (无头环境/驱动问题)
        """
        if self._initialized:
            return True
        if self._init_failed:
            return False

        try:
            import os
            import ctypes

            # ---- 修复 conda/Cursor 环境下的 OpenGL 驱动冲突 ----
            # 问题: conda 的 libstdc++.so.6 版本低于系统 mesa/LLVM 所需
            # 方案: 预加载系统 libstdc++ + 强制软件渲染
            _sys_stdcpp = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
            if os.path.exists(_sys_stdcpp):
                try:
                    ctypes.CDLL(_sys_stdcpp, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
            os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
            os.environ.setdefault("LIBGL_DRIVERS_PATH",
                                  "/usr/lib/x86_64-linux-gnu/dri")

            pygame.init()
            self.engine = engine
            map_w = engine.width * self.cell_size
            map_h = engine.height * self.cell_size
            self.screen = pygame.display.set_mode((map_w, map_h + self.panel_height))
            pygame.display.set_caption("经典坦克大战 - 分层 MADDPG 协同 AI")
            self.clock = pygame.time.Clock()
            self.font_small = pygame.font.SysFont("sans-serif", 13)
            self.font_medium = pygame.font.SysFont("sans-serif", 16)
            self.font_large = pygame.font.SysFont("sans-serif", 22, bold=True)
            self._initialized = True
            return True
        except Exception as e:
            print(f"[渲染器] Pygame 初始化失败: {e}")
            print("[渲染器] 将跳过可视化, 训练继续运行")
            self._init_failed = True
            return False

    # ------------------------------------------------------------------
    #  主渲染
    # ------------------------------------------------------------------

    def render(
        self,
        engine: BattleCityEngine,
        skill_info: Optional[list[tuple[int, np.ndarray]]] = None,
        episode: int = 0,
        step: int = 0,
        reward: float = 0.0,
        cooperation_weights: Optional[np.ndarray] = None,
        comm_messages: Optional[list[np.ndarray]] = None,
        training_mode: bool = False,
    ) -> bool:
        """渲染一帧 (含协作可视化).

        Args:
            engine: 游戏引擎实例
            skill_info: [(skill_id, param), ...] 每个红方坦克的技能
            episode: 当前回合
            step: 当前步数
            reward: 累计奖励
            cooperation_weights: (n_agents, n_agents) 注意力协作权重矩阵
            comm_messages: 各智能体的通讯消息向量 (用于显示通讯状态)
            training_mode: 是否处于训练可视化模式

        Returns:
            True 继续, False 用户关闭窗口
        """
        if not self._init_display(engine):
            return True  # 初始化失败, 静默跳过渲染但不中断训练

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False

        cs = self.cell_size
        self.screen.fill(COLOR_BG)

        # 1. 绘制地图瓦片
        self._draw_map(engine)

        # 2. 协作连线 (在坦克之下, 作为底层)
        red_tanks = [t for t in engine.tanks if t.team == "red" and t.alive]
        if cooperation_weights is not None and len(red_tanks) >= 2:
            self._draw_cooperation_lines(red_tanks, cooperation_weights)

        # 3. 绘制坦克 (先蓝后红, 红方在上层)
        for tank in engine.tanks:
            if tank.team == "blue" and tank.alive:
                self._draw_tank(tank, COLOR_BLUE_TANK, COLOR_BLUE_DARK)
        for tank in engine.tanks:
            if tank.team == "red" and tank.alive:
                self._draw_tank(tank, COLOR_RED_TANK, COLOR_RED_DARK)

        # 4. 坦克编号 + 技能标签 (在坦克之上)
        self._draw_tank_ids(engine)
        if skill_info:
            self._draw_skill_labels(red_tanks, skill_info)

        # 5. 绘制树林 (覆盖在坦克上方, 经典效果)
        for r in range(engine.height):
            for c in range(engine.width):
                if engine.grid[r, c] == Tile.TREE:
                    self._draw_tree_overlay(c, r)

        # 6. 绘制子弹
        for bullet in engine.bullets:
            if bullet.alive:
                self._draw_bullet(bullet)

        # 7. 绘制已摧毁坦克的标记
        for tank in engine.tanks:
            if not tank.alive:
                self._draw_destroyed(tank)

        # 8. 注意力矩阵热力图 (右下角)
        if cooperation_weights is not None:
            self._draw_attention_heatmap(engine, cooperation_weights)

        # 9. 通讯状态指示 (如果有)
        if comm_messages is not None:
            self._draw_comm_indicator(red_tanks, comm_messages)

        # 10. 信息面板
        self._draw_panel(engine, skill_info, episode, step, reward, training_mode)

        pygame.display.flip()
        self.clock.tick(self.fps)
        self._frame_count += 1
        return True

    # ------------------------------------------------------------------
    #  地图绘制
    # ------------------------------------------------------------------

    def _draw_map(self, engine: BattleCityEngine) -> None:
        cs = self.cell_size
        for r in range(engine.height):
            for c in range(engine.width):
                tile = engine.grid[r, c]
                x, y = c * cs, r * cs

                if tile == Tile.BRICK:
                    self._draw_brick(x, y)
                elif tile == Tile.STEEL:
                    self._draw_steel(x, y)
                elif tile == Tile.WATER:
                    self._draw_water(x, y)
                elif tile == Tile.ICE:
                    pygame.draw.rect(self.screen, COLOR_ICE, (x, y, cs, cs))
                    # 冰面高光
                    pygame.draw.line(self.screen, COLOR_WHITE,
                                     (x + 3, y + 3), (x + cs - 6, y + cs // 3), 1)
                elif tile == Tile.TREE:
                    pass  # 树林底层为空地, 覆盖层单独绘制
                elif tile == Tile.BASE:
                    self._draw_base(x, y, alive=engine.base_alive)

    def _draw_brick(self, x: int, y: int) -> None:
        cs = self.cell_size
        pygame.draw.rect(self.screen, COLOR_BRICK, (x, y, cs, cs))
        # 砖块纹理
        half = cs // 2
        pygame.draw.line(self.screen, COLOR_BRICK_DARK, (x, y + half), (x + cs, y + half), 1)
        pygame.draw.line(self.screen, COLOR_BRICK_DARK, (x + half, y), (x + half, y + half), 1)
        pygame.draw.line(self.screen, COLOR_BRICK_DARK, (x, y), (x, y + cs), 1)
        pygame.draw.rect(self.screen, COLOR_BRICK_DARK, (x, y, cs, cs), 1)

    def _draw_steel(self, x: int, y: int) -> None:
        cs = self.cell_size
        pygame.draw.rect(self.screen, COLOR_STEEL, (x, y, cs, cs))
        # 金属纹理
        pygame.draw.rect(self.screen, COLOR_STEEL_DARK, (x, y, cs, cs), 2)
        pygame.draw.line(self.screen, COLOR_WHITE,
                         (x + 2, y + 2), (x + cs // 3, y + 2), 1)

    def _draw_water(self, x: int, y: int) -> None:
        cs = self.cell_size
        pygame.draw.rect(self.screen, COLOR_WATER, (x, y, cs, cs))
        # 波纹动画 (基于帧计数)
        offset = (self._frame_count // 4) % 6
        for wave_y in range(y + offset, y + cs, 6):
            pygame.draw.line(self.screen, COLOR_WATER_LIGHT,
                             (x + 2, wave_y), (x + cs - 2, wave_y), 1)

    def _draw_tree_overlay(self, col: int, row: int) -> None:
        """树林覆盖层 (绘制在坦克之上)."""
        cs = self.cell_size
        x, y = col * cs, row * cs
        # 半透明绿色 (Pygame 不支持直接半透明, 用密集圆点模拟)
        cx, cy = x + cs // 2, y + cs // 2
        r = cs // 2 - 1
        pygame.draw.circle(self.screen, COLOR_TREE, (cx, cy), r)
        pygame.draw.circle(self.screen, COLOR_TREE_LIGHT, (cx - 3, cy - 3), r // 2)

    def _draw_base(self, x: int, y: int, alive: bool) -> None:
        cs = self.cell_size
        color = COLOR_BASE if alive else COLOR_BASE_DEAD
        # 鹰/基地符号
        pygame.draw.rect(self.screen, color, (x + 2, y + 2, cs - 4, cs - 4))
        # 简化的鹰标志 (三角形)
        inner = COLOR_BG if alive else COLOR_GRAY
        cx, cy = x + cs // 2, y + cs // 2
        pts = [(cx, cy - cs // 4), (cx - cs // 4, cy + cs // 4), (cx + cs // 4, cy + cs // 4)]
        pygame.draw.polygon(self.screen, inner, pts)

    # ------------------------------------------------------------------
    #  坦克绘制
    # ------------------------------------------------------------------

    def _draw_tank(self, tank: Tank, color: tuple, dark: tuple) -> None:
        cs = self.cell_size
        cx = tank.x * cs + cs // 2
        cy = tank.y * cs + cs // 2
        r = cs // 2 - 2

        # 坦克主体 (圆形)
        pygame.draw.circle(self.screen, color, (cx, cy), r)
        pygame.draw.circle(self.screen, dark, (cx, cy), r, 2)

        # 炮管 (朝向线)
        dx_map = {Dir.UP: (0, -1), Dir.DOWN: (0, 1), Dir.LEFT: (-1, 0), Dir.RIGHT: (1, 0)}
        ddx, ddy = dx_map[tank.direction]
        barrel_len = cs // 2 + 2
        bx = cx + ddx * barrel_len
        by = cy + ddy * barrel_len
        pygame.draw.line(self.screen, dark, (cx, cy), (bx, by), 3)

        # 血量条
        if tank.max_health > 1:
            bar_w = cs - 4
            bar_h = 3
            bar_x = tank.x * cs + 2
            bar_y = tank.y * cs - 5
            pygame.draw.rect(self.screen, (40, 40, 40), (bar_x, bar_y, bar_w, bar_h))
            hp_ratio = max(0, tank.health / tank.max_health)
            hp_color = COLOR_GREEN if hp_ratio > 0.5 else COLOR_YELLOW if hp_ratio > 0.3 else COLOR_RED_TANK
            pygame.draw.rect(self.screen, hp_color, (bar_x, bar_y, int(bar_w * hp_ratio), bar_h))

    def _draw_destroyed(self, tank: Tank) -> None:
        cs = self.cell_size
        cx = tank.x * cs + cs // 2
        cy = tank.y * cs + cs // 2
        # 灰色十字
        pygame.draw.line(self.screen, COLOR_GRAY, (cx - 6, cy - 6), (cx + 6, cy + 6), 2)
        pygame.draw.line(self.screen, COLOR_GRAY, (cx - 6, cy + 6), (cx + 6, cy - 6), 2)

    # ------------------------------------------------------------------
    #  子弹绘制
    # ------------------------------------------------------------------

    def _draw_bullet(self, bullet: Bullet) -> None:
        cs = self.cell_size
        x = bullet.x * cs + cs // 2
        y = bullet.y * cs + cs // 2
        color = COLOR_BULLET_RED if bullet.owner_team == "red" else COLOR_BULLET_BLUE
        pygame.draw.rect(self.screen, color, (x - 2, y - 2, 5, 5))

    # ------------------------------------------------------------------
    #  信息面板
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    #  协作可视化
    # ------------------------------------------------------------------

    def _draw_cooperation_lines(
        self, red_tanks: list[Tank], coop_weights: np.ndarray
    ) -> None:
        """绘制坦克间协作连线.

        连线粗细和颜色反映注意力权重 (协作强度):
          - 粗黄线 = 高协作
          - 中绿线 = 中等协作
          - 细蓝线 = 弱协作
        """
        cs = self.cell_size
        n = min(len(red_tanks), coop_weights.shape[0])

        for i in range(n):
            for j in range(i + 1, n):
                # 取双向权重的最大值
                w = max(float(coop_weights[i, j]), float(coop_weights[j, i]))
                if w < 0.05:
                    continue

                # 颜色插值
                intensity = min(1.0, w * 2.5)
                if intensity > 0.6:
                    color = COLOR_COOP_HIGH
                elif intensity > 0.3:
                    color = COLOR_COOP_MID
                else:
                    color = COLOR_COOP_LOW
                width = max(1, int(intensity * 5))

                x1 = red_tanks[i].x * cs + cs // 2
                y1 = red_tanks[i].y * cs + cs // 2
                x2 = red_tanks[j].x * cs + cs // 2
                y2 = red_tanks[j].y * cs + cs // 2

                # 绘制发光效果 (两层线)
                if width > 2:
                    glow_r = min(255, color[0] + 30)
                    glow_g = min(255, color[1] + 30)
                    glow_b = min(255, color[2] + 30)
                    pygame.draw.line(self.screen, (glow_r, glow_g, glow_b),
                                     (x1, y1), (x2, y2), width + 2)
                pygame.draw.line(self.screen, color, (x1, y1), (x2, y2), width)

                # 中点权重数值
                mx, my = (x1 + x2) // 2, (y1 + y2) // 2
                self._draw_text_with_bg(f"{w:.2f}", mx - 12, my - 8, color)

    def _draw_skill_labels(
        self, red_tanks: list[Tank], skill_info: list[tuple[int, np.ndarray]]
    ) -> None:
        """在坦克上方绘制技能标签 (带彩色背景)."""
        cs = self.cell_size
        for i, tank in enumerate(red_tanks):
            if i >= len(skill_info):
                break
            sid = skill_info[i][0]
            name = SKILL_NAMES.get(sid, f"S{sid}")
            color = SKILL_COLORS.get(sid, COLOR_WHITE)

            x = tank.x * cs + cs // 2
            y = tank.y * cs - 16

            # 背景框
            text = self.font_small.render(name, True, color)
            tw, th = text.get_width(), text.get_height()
            bg_rect = pygame.Rect(x - tw // 2 - 3, y - 1, tw + 6, th + 2)
            bg_surf = pygame.Surface((bg_rect.w, bg_rect.h), pygame.SRCALPHA)
            bg_surf.fill((0, 0, 0, 160))
            self.screen.blit(bg_surf, bg_rect.topleft)
            self.screen.blit(text, (x - tw // 2, y))

    def _draw_tank_ids(self, engine: BattleCityEngine) -> None:
        """在坦克上绘制编号."""
        cs = self.cell_size
        red_idx, blue_idx = 0, 0
        for tank in engine.tanks:
            if not tank.alive:
                continue
            if tank.team == "red":
                label = f"R{red_idx}"
                color = COLOR_RED_TANK
                red_idx += 1
            else:
                label = f"B{blue_idx}"
                color = COLOR_BLUE_TANK
                blue_idx += 1

            x = tank.x * cs + cs // 2
            y = tank.y * cs + cs // 2
            text = self.font_small.render(label, True, COLOR_WHITE)
            tw = text.get_width()
            self.screen.blit(text, (x - tw // 2, y - 5))

    def _draw_attention_heatmap(
        self, engine: BattleCityEngine, coop_weights: np.ndarray
    ) -> None:
        """在右下角绘制注意力权重热力图 + 协作关系说明."""
        n = coop_weights.shape[0]
        cell = 30
        margin = 8
        map_w = engine.width * self.cell_size
        map_h = engine.height * self.cell_size

        start_x = map_w - margin - n * cell - 40
        start_y = map_h - margin - n * cell - 50

        # 半透明背景
        bg_w = n * cell + 50
        bg_h = n * cell + 45
        bg_surf = pygame.Surface((bg_w, bg_h), pygame.SRCALPHA)
        bg_surf.fill((0, 0, 0, 180))
        self.screen.blit(bg_surf, (start_x - 8, start_y - 20))

        # 标题
        title = self.font_small.render("Attn 协作矩阵", True, COLOR_YELLOW)
        self.screen.blit(title, (start_x, start_y - 16))

        # 热力图
        for i in range(n):
            for j in range(n):
                w = float(coop_weights[i, j])
                # 颜色: 深蓝(0) -> 绿(0.3) -> 黄(0.6) -> 红(1.0)
                if w < 0.3:
                    r = int(50 * w / 0.3)
                    g = int(80 + 140 * w / 0.3)
                    b = int(140 - 90 * w / 0.3)
                elif w < 0.6:
                    t = (w - 0.3) / 0.3
                    r = int(50 + 200 * t)
                    g = int(220 - 20 * t)
                    b = int(50 - 30 * t)
                else:
                    t = min(1.0, (w - 0.6) / 0.4)
                    r = int(250)
                    g = int(200 - 150 * t)
                    b = int(20)
                color = (min(255, r), min(255, g), min(255, b))

                x = start_x + j * cell
                y = start_y + i * cell
                pygame.draw.rect(self.screen, color, (x, y, cell - 2, cell - 2))

                # 数值文本
                txt = self.font_small.render(f"{w:.2f}", True, COLOR_WHITE)
                self.screen.blit(txt, (x + 1, y + 8))

            # 行标签
            label = self.font_small.render(f"R{i}", True, COLOR_RED_TANK)
            self.screen.blit(label, (start_x - 20, start_y + i * cell + 8))

        # 列标签
        for j in range(n):
            label = self.font_small.render(f"R{j}", True, COLOR_RED_TANK)
            self.screen.blit(label, (start_x + j * cell + 6, start_y + n * cell + 2))

    def _draw_comm_indicator(
        self, red_tanks: list[Tank], comm_messages: list[np.ndarray]
    ) -> None:
        """绘制通讯状态指示器 (坦克旁边的小信号图标)."""
        cs = self.cell_size
        for i, tank in enumerate(red_tanks):
            if i >= len(comm_messages):
                break
            msg = comm_messages[i]
            # 消息强度 = L2 范数
            strength = float(np.linalg.norm(msg))
            if strength < 0.01:
                continue

            x = tank.x * cs + cs - 2
            y = tank.y * cs + 2

            # 信号波纹 (1-3 层)
            n_arcs = min(3, max(1, int(strength * 3)))
            for arc in range(n_arcs):
                r = 4 + arc * 3
                alpha = max(60, 220 - arc * 60)
                color = (50, 200, 255)
                pygame.draw.arc(
                    self.screen, color,
                    (x - r, y - r, r * 2, r * 2),
                    0.3, 1.2, 1 + arc,
                )

    def _draw_text_with_bg(
        self, text: str, x: int, y: int, color: tuple, bg_alpha: int = 160
    ) -> None:
        """绘制带半透明背景的文本."""
        surf = self.font_small.render(text, True, color)
        tw, th = surf.get_width(), surf.get_height()
        bg = pygame.Surface((tw + 4, th + 2), pygame.SRCALPHA)
        bg.fill((0, 0, 0, bg_alpha))
        self.screen.blit(bg, (x - 2, y - 1))
        self.screen.blit(surf, (x, y))

    # ------------------------------------------------------------------
    #  信息面板
    # ------------------------------------------------------------------

    def _draw_panel(
        self,
        engine: BattleCityEngine,
        skill_info: Optional[list] = None,
        episode: int = 0,
        step: int = 0,
        reward: float = 0.0,
        training_mode: bool = False,
    ) -> None:
        panel_y = engine.height * self.cell_size
        pygame.draw.rect(
            self.screen, COLOR_PANEL_BG,
            (0, panel_y, engine.width * self.cell_size, self.panel_height)
        )

        # 标题
        mode_str = "[训练可视化] " if training_mode else ""
        title = self.font_large.render(
            f"{mode_str}经典坦克大战 - MADDPG AI", True, COLOR_YELLOW
        )
        self.screen.blit(title, (8, panel_y + 4))

        # 统计信息
        red_alive = len(engine.get_tanks_by_team("red"))
        blue_alive = len(engine.get_tanks_by_team("blue"))
        base_str = "安全" if engine.base_alive else "已摧毁"
        base_col = COLOR_GREEN if engine.base_alive else COLOR_RED_TANK

        stats = [
            (f"回合:{episode}", COLOR_GRAY),
            (f"步数:{step}", COLOR_GRAY),
            (f"奖励:{reward:.1f}", COLOR_GRAY),
            (f"红方:{red_alive}", COLOR_RED_TANK),
            (f"蓝方:{blue_alive}", COLOR_BLUE_TANK),
        ]
        x_offset = 8
        for text_str, col in stats:
            text = self.font_medium.render(text_str, True, col)
            self.screen.blit(text, (x_offset, panel_y + 35))
            x_offset += text.get_width() + 20

        # 基地状态
        base_text = self.font_medium.render(f"基地:{base_str}", True, base_col)
        self.screen.blit(base_text, (x_offset, panel_y + 35))

        # 技能信息
        if skill_info:
            skill_strs = []
            for i, (sid, _) in enumerate(skill_info):
                name = SKILL_NAMES.get(sid, f"技能{sid}")
                skill_strs.append(f"坦克{i}:{name}")
            skill_text = self.font_medium.render(
                "  |  ".join(skill_strs), True, COLOR_YELLOW
            )
            self.screen.blit(skill_text, (8, panel_y + 60))

        # 胜负提示
        if not engine.base_alive:
            result = self.font_large.render("基地被摧毁! GAME OVER", True, COLOR_RED_TANK)
            w = engine.width * self.cell_size
            self.screen.blit(result, (w - result.get_width() - 10, panel_y + 60))
        elif blue_alive == 0 and len(engine.tanks) > len(engine.get_tanks_by_team("red")):
            result = self.font_large.render("胜利! ALL CLEAR", True, COLOR_GREEN)
            w = engine.width * self.cell_size
            self.screen.blit(result, (w - result.get_width() - 10, panel_y + 60))

    def close(self) -> None:
        if self._initialized:
            pygame.quit()
            self._initialized = False
