"""
训练可视化监控器

实现功能:
  1. 单智能体训练可视化: 每 N 回合实时展示当前策略
  2. 多智能体协同训练可视化: 展示协同过程 + 协作连线
  3. 训练曲线叠加: 奖励/胜率/损失实时更新
  4. 协作网络可视化: 注意力权重热力图 + 坦克间连线

使用方式:
  monitor = TrainMonitor(renderer, mode='multi')
  monitor.on_episode_end(episode, reward, info, engine, ...)  # 每回合结束时调用
  monitor.render_eval_episode(engine, trainer, ...)            # 定期完整评估可视化
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from envs.game_engine import BattleCityEngine
from utils.visualize import TankRenderer


class TrainMonitor:
    """训练可视化监控器, 单/多智能体通用."""

    def __init__(
        self,
        cell_size: int = 32,
        fps: int = 10,
        max_history: int = 200,
        render_every: int = 50,
    ) -> None:
        """
        Args:
            cell_size: 瓦片像素大小
            fps: 可视化帧率
            max_history: 训练曲线最大历史长度
            render_every: 每隔多少回合可视化一次完整评估
        """
        self.cell_size = cell_size
        self.fps = fps
        self.render_every = render_every
        self.renderer: Optional[TankRenderer] = None

        # 训练历史
        self.reward_history: deque[float] = deque(maxlen=max_history)
        self.win_history: deque[bool] = deque(maxlen=max_history)
        self.loss_history: deque[float] = deque(maxlen=max_history)

        # 协作权重历史 (多智能体)
        self.coop_weights_history: deque[np.ndarray] = deque(maxlen=50)

        self._episode_count = 0
        self._running = True

    def _ensure_renderer(self, engine: BattleCityEngine) -> None:
        if self.renderer is None:
            self.renderer = TankRenderer(
                cell_size=self.cell_size, fps=self.fps, engine=engine,
            )

    # ------------------------------------------------------------------
    #  训练回调
    # ------------------------------------------------------------------

    def on_episode_end(
        self,
        episode: int,
        reward: float,
        info: dict,
        metrics: Optional[dict[str, float]] = None,
    ) -> None:
        """每回合结束时调用, 记录训练统计."""
        self._episode_count = episode
        self.reward_history.append(reward)
        self.win_history.append(info.get("win", False))
        if metrics:
            avg_loss = np.mean([v for k, v in metrics.items() if "critic_loss" in k])
            self.loss_history.append(avg_loss)

    def should_render(self, episode: int) -> bool:
        """是否应该进行可视化评估."""
        return self._running and (episode + 1) % self.render_every == 0

    # ------------------------------------------------------------------
    #  单智能体可视化评估
    # ------------------------------------------------------------------

    def render_single_eval(
        self,
        env,
        policy_fn,
        episode: int = 0,
        max_steps: int = 200,
    ) -> bool:
        """单智能体可视化评估.

        Args:
            env: SingleTankSkillEnv 实例
            policy_fn: obs -> action 的策略函数
            episode: 当前训练回合
            max_steps: 评估最大步数

        Returns:
            True 继续训练, False 用户关闭窗口
        """
        self._ensure_renderer(env.engine)
        obs, _ = env.reset()
        ep_reward = 0.0

        for step in range(max_steps):
            action = policy_fn(obs)
            obs, reward, term, trunc, info = env.step(action)
            ep_reward += reward

            # 渲染 (含训练统计)
            running = self.renderer.render(
                env.engine,
                episode=episode,
                step=step,
                reward=ep_reward,
            )
            if not running:
                self._running = False
                return False

            # 绘制训练曲线叠加
            self._draw_training_overlay(env.engine, episode)

            if term or trunc:
                break

        return True

    # ------------------------------------------------------------------
    #  多智能体协同可视化评估
    # ------------------------------------------------------------------

    def render_multi_eval(
        self,
        env,
        trainer,
        skill_lib: dict,
        execute_skill_fn,
        episode: int = 0,
        max_steps: int = 300,
        skill_interval: int = 8,
    ) -> bool:
        """多智能体协同可视化评估, 含协作连线.

        Args:
            env: MultiTankTeamEnv 实例
            trainer: MaddpgTrainer 实例
            skill_lib: 底层技能库
            execute_skill_fn: 技能执行函数
            episode: 当前训练回合
            max_steps: 评估步数
            skill_interval: 技能间隔

        Returns:
            True 继续, False 退出
        """
        self._ensure_renderer(env.engine)
        obs = env.reset()
        ep_reward = 0.0
        n_agents = len(obs)

        current_skill_ids = [1] * n_agents
        current_params = [np.zeros(2, dtype=np.float32)] * n_agents

        for step in range(max_steps):
            # 高层决策
            if step % skill_interval == 0:
                actions = trainer.select_actions(obs, deterministic=True)
                current_skill_ids = [a[0] for a in actions]
                current_params = [a[1] for a in actions]

            # 底层执行
            low_actions = []
            for i in range(n_agents):
                act = execute_skill_fn(
                    skill_lib, current_skill_ids[i], obs[i], current_params[i]
                )
                low_actions.append(act)

            next_obs, reward, done, info = env.step(low_actions)
            ep_reward += reward
            obs = next_obs

            # 获取协作权重
            coop_weights = trainer.get_cooperation_weights()
            if coop_weights is not None:
                self.coop_weights_history.append(coop_weights.copy())

            # 基础渲染
            skill_info = list(zip(current_skill_ids, current_params))
            running = self.renderer.render(
                env.engine,
                skill_info=skill_info,
                episode=episode,
                step=step,
                reward=ep_reward,
            )
            if not running:
                self._running = False
                return False

            # 绘制协作可视化叠加
            self._draw_cooperation_overlay(env, coop_weights, current_skill_ids)
            self._draw_training_overlay(env.engine, episode)

            # 刷新显示 (覆盖 renderer 的 flip)
            if PYGAME_AVAILABLE:
                pygame.display.flip()

            if done:
                break

        return True

    # ------------------------------------------------------------------
    #  协作可视化叠加
    # ------------------------------------------------------------------

    def _draw_cooperation_overlay(
        self,
        env,
        coop_weights: Optional[np.ndarray],
        skill_ids: list[int],
    ) -> None:
        """在渲染画面上叠加协作信息."""
        if not PYGAME_AVAILABLE or self.renderer is None or self.renderer.screen is None:
            return

        screen = self.renderer.screen
        cs = self.renderer.cell_size
        red_tanks = [t for t in env.red_tanks if t.alive]

        if len(red_tanks) < 2:
            return

        # 1. 绘制协作连线 (坦克之间)
        if coop_weights is not None:
            for i in range(len(red_tanks)):
                for j in range(len(red_tanks)):
                    if i == j:
                        continue
                    weight = float(coop_weights[i, j])
                    if weight < 0.1:
                        continue

                    # 连线颜色: 从蓝 (弱协作) 到黄 (强协作)
                    intensity = min(1.0, weight * 2.0)
                    r = int(255 * intensity)
                    g = int(220 * intensity)
                    b = int(50 * (1 - intensity))
                    color = (r, g, b)
                    width = max(1, int(intensity * 4))

                    x1 = red_tanks[i].x * cs + cs // 2
                    y1 = red_tanks[i].y * cs + cs // 2
                    x2 = red_tanks[j].x * cs + cs // 2
                    y2 = red_tanks[j].y * cs + cs // 2

                    pygame.draw.line(screen, color, (x1, y1), (x2, y2), width)

                    # 连线中点显示权重值
                    mx, my = (x1 + x2) // 2, (y1 + y2) // 2
                    font = self.renderer.font_small
                    if font:
                        text = font.render(f"{weight:.2f}", True, (255, 255, 200))
                        screen.blit(text, (mx - 12, my - 8))

        # 2. 绘制注意力矩阵 (右上角小窗口)
        if coop_weights is not None:
            self._draw_attention_matrix(coop_weights)

    def _draw_attention_matrix(self, coop_weights: np.ndarray) -> None:
        """在右上角绘制注意力权重热力图."""
        if self.renderer is None or self.renderer.screen is None:
            return

        screen = self.renderer.screen
        n = coop_weights.shape[0]
        cell = 28
        margin = 8
        start_x = screen.get_width() - margin - n * cell - 50
        start_y = margin

        # 标题
        font = self.renderer.font_small
        if font:
            title = font.render("协作矩阵", True, (255, 220, 50))
            screen.blit(title, (start_x, start_y))
        start_y += 18

        # 热力图
        for i in range(n):
            for j in range(n):
                w = float(coop_weights[i, j])
                # 颜色映射: 0=深蓝 -> 1=亮黄
                r = int(min(255, w * 500))
                g = int(min(220, w * 440))
                b = int(max(0, 100 - w * 200))
                x = start_x + j * cell
                y = start_y + i * cell
                pygame.draw.rect(screen, (r, g, b), (x, y, cell - 2, cell - 2))
                # 数值
                if font:
                    txt = font.render(f"{w:.1f}", True, (255, 255, 255))
                    screen.blit(txt, (x + 2, y + 6))

            # 行标签
            if font:
                label = font.render(f"T{i}", True, (200, 50, 50))
                screen.blit(label, (start_x - 22, start_y + i * cell + 6))

        # 列标签
        if font:
            for j in range(n):
                label = font.render(f"T{j}", True, (200, 50, 50))
                screen.blit(label, (start_x + j * cell + 4, start_y + n * cell + 2))

    # ------------------------------------------------------------------
    #  训练统计叠加
    # ------------------------------------------------------------------

    def _draw_training_overlay(self, engine: BattleCityEngine, episode: int) -> None:
        """在画面上叠加训练曲线."""
        if not PYGAME_AVAILABLE or self.renderer is None or self.renderer.screen is None:
            return
        if len(self.reward_history) < 2:
            return

        screen = self.renderer.screen
        font = self.renderer.font_small

        # 奖励曲线 (左上角)
        margin = 8
        chart_w = 160
        chart_h = 60
        x0, y0 = margin, margin

        # 背景
        surf = pygame.Surface((chart_w, chart_h + 18), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 140))
        screen.blit(surf, (x0, y0))

        if font:
            title = font.render(
                f"Ep:{episode} R(avg):{np.mean(list(self.reward_history)[-20:]):.1f}",
                True, (200, 200, 200),
            )
            screen.blit(title, (x0 + 4, y0 + 2))

        # 绘制曲线
        rewards = list(self.reward_history)
        if len(rewards) >= 2:
            min_r, max_r = min(rewards), max(rewards)
            if max_r - min_r < 1e-6:
                max_r = min_r + 1
            points = []
            for idx, r in enumerate(rewards):
                px = x0 + int(idx / max(1, len(rewards) - 1) * (chart_w - 4)) + 2
                py = y0 + 18 + chart_h - 2 - int((r - min_r) / (max_r - min_r) * (chart_h - 4))
                points.append((px, py))
            if len(points) >= 2:
                pygame.draw.lines(screen, (50, 220, 50), False, points, 2)

        # 胜率 (如果有)
        if self.win_history:
            wr = np.mean(list(self.win_history)[-20:])
            if font:
                wr_text = font.render(f"Win:{wr:.0%}", True, (255, 220, 50))
                screen.blit(wr_text, (x0 + chart_w + 8, y0 + 2))

    # ------------------------------------------------------------------
    #  清理
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
