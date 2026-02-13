#!/usr/bin/env python3
"""
MADDPG训练可视化工具 - 基于PyQt5
支持实时观察训练过程，包括环境渲染和奖励曲线

使用方法:
    python experiments/qt_visualize_train.py

功能:
    - 实时渲染多智能体环境
    - 显示训练奖励曲线
    - 可开关可视化功能
    - 支持多种场景选择
    - 可调节训练参数
    - "观察下一轮"按钮用于观察智能体行为
    - 保存/加载模型权重
"""

import sys
import os
import numpy as np
import torch
import time
import pickle
from collections import deque
from threading import Thread, Event
import queue

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'multiagent-particle-envs'))

# 设置环境变量
os.environ['SUPPRESS_MA_PROMPT'] = '1'

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QGroupBox, QGridLayout, QProgressBar, QSplitter,
    QFrame, QStatusBar, QSlider, QMessageBox, QFileDialog,
    QTabWidget, QScrollArea, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QFont, QPen, QBrush

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import maddpg.common.torch_util as U
from maddpg.trainer.torch_maddpg import MADDPGAgentTrainer


class TrainingSignals(QObject):
    """训练信号，用于线程间通信"""
    update_frame = pyqtSignal(object)  # 更新渲染帧（传递world对象）
    update_stats = pyqtSignal(dict)  # 更新统计信息
    training_finished = pyqtSignal()  # 训练结束


class EnvironmentRenderer(QWidget):
    """环境渲染器 - 自定义绘制多智能体环境"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.world = None
        self.env = None
        self.setMinimumSize(400, 400)
        self.setStyleSheet("background-color: white; border: 2px solid #3498db; border-radius: 5px;")
        
        # 颜色映射 - 用于当智能体没有颜色或颜色太浅时
        self.agent_colors = [
            QColor(231, 76, 60),    # 红色
            QColor(46, 204, 113),   # 绿色
            QColor(52, 152, 219),   # 蓝色
            QColor(155, 89, 182),   # 紫色
            QColor(241, 196, 15),   # 黄色
            QColor(26, 188, 156),   # 青色
            QColor(230, 126, 34),   # 橙色
            QColor(149, 165, 166),  # 灰色
        ]
        self.landmark_color = QColor(52, 73, 94)  # 深灰色
        
        # 场景名称（用于特殊处理）
        self.scenario_name = ""
        
    def set_env(self, env, scenario_name=""):
        """设置环境引用"""
        self.env = env
        self.scenario_name = scenario_name
        if env:
            self.world = env.world
        
    def update_world(self, world):
        """更新世界状态"""
        self.world = world
        self.update()
    
    def _get_agent_color(self, agent, index):
        """获取智能体颜色，确保可见性"""
        if hasattr(agent, 'color') and agent.color is not None:
            # 裁剪颜色值到 [0, 1] 范围
            r = min(max(agent.color[0], 0), 1.0)
            g = min(max(agent.color[1], 0), 1.0)
            b = min(max(agent.color[2], 0), 1.0)
            
            # 计算亮度 (perceived brightness)
            brightness = 0.299 * r + 0.587 * g + 0.114 * b
            
            # 如果颜色太暗（如深灰色 0.25,0.25,0.25），使用预定义颜色
            # 或者在 simple_speaker_listener 场景中特殊处理
            if brightness < 0.35:
                # 对于说话者/听众场景，使用更醒目的颜色
                if self.scenario_name == "simple_speaker_listener":
                    if index == 0:  # Speaker
                        return QColor(155, 89, 182)  # 紫色
                    else:  # Listener
                        return QColor(230, 126, 34)  # 橙色
                else:
                    # 增强暗色
                    r = min(r + 0.4, 1.0)
                    g = min(g + 0.4, 1.0)
                    b = min(b + 0.4, 1.0)
            
            return QColor(
                int(r * 255),
                int(g * 255),
                int(b * 255)
            )
        else:
            return self.agent_colors[index % len(self.agent_colors)]
    
    def _get_agent_label(self, agent, index):
        """获取智能体标签"""
        if self.scenario_name == "simple_speaker_listener":
            if index == 0:
                return "S"  # Speaker
            else:
                return "L"  # Listener
        elif hasattr(agent, 'adversary') and agent.adversary:
            return "A"  # Adversary
        else:
            return str(index)
        
    def paintEvent(self, event):
        """绘制事件 - 自定义绘制智能体和地标"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 填充背景 - 使用浅蓝灰色背景使智能体更明显
        painter.fillRect(self.rect(), QColor(240, 245, 250))
        
        if self.world is None:
            # 显示提示文字
            painter.setPen(QColor(100, 100, 100))
            font = QFont("Arial", 12)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignCenter, 
                           "点击 [👁 观察下一轮] 按钮\n观察智能体行为")
            return
        
        # 计算坐标转换参数
        width = self.width()
        height = self.height()
        margin = 40
        scale = min(width - 2*margin, height - 2*margin) / 4.0  # 假设环境范围 [-2, 2]
        center_x = width / 2
        center_y = height / 2
        
        def world_to_screen(pos):
            """世界坐标转屏幕坐标"""
            x = center_x + pos[0] * scale
            y = center_y - pos[1] * scale  # Y轴翻转
            return int(x), int(y)
        
        # 绘制网格
        painter.setPen(QPen(QColor(210, 220, 230), 1))
        for i in range(-2, 3):
            x = center_x + i * scale
            painter.drawLine(int(x), margin, int(x), height - margin)
            y = center_y + i * scale
            painter.drawLine(margin, int(y), width - margin, int(y))
        
        # 绘制坐标轴
        painter.setPen(QPen(QColor(150, 160, 170), 2))
        painter.drawLine(int(center_x), margin, int(center_x), height - margin)
        painter.drawLine(margin, int(center_y), width - margin, int(center_y))
        
        # 绘制地标
        if hasattr(self.world, 'landmarks'):
            for i, landmark in enumerate(self.world.landmarks):
                if hasattr(landmark, 'state') and hasattr(landmark.state, 'p_pos'):
                    pos = landmark.state.p_pos
                    x, y = world_to_screen(pos)
                    
                    # 获取地标颜色（裁剪到有效范围）
                    if hasattr(landmark, 'color') and landmark.color is not None:
                        r = min(max(landmark.color[0], 0), 1.0)
                        g = min(max(landmark.color[1], 0), 1.0)
                        b = min(max(landmark.color[2], 0), 1.0)
                        color = QColor(int(r * 255), int(g * 255), int(b * 255))
                    else:
                        color = self.landmark_color
                    
                    # 获取地标大小
                    landmark_size = getattr(landmark, 'size', 0.1)
                    size = int(landmark_size * scale * 2)
                    size = max(size, 12)  # 最小12像素
                    
                    # 绘制地标（方形）
                    painter.setBrush(QBrush(color))
                    painter.setPen(QPen(color.darker(120), 2))
                    painter.drawRect(x - size//2, y - size//2, size, size)
                    
                    # 绘制地标编号
                    painter.setPen(QColor(255, 255, 255))
                    font = QFont("Arial", 7)
                    painter.setFont(font)
                    painter.drawText(x - 3, y + 3, str(i))
        
        # 绘制智能体
        if hasattr(self.world, 'agents'):
            for i, agent in enumerate(self.world.agents):
                if hasattr(agent, 'state') and hasattr(agent.state, 'p_pos'):
                    pos = agent.state.p_pos
                    x, y = world_to_screen(pos)
                    
                    # 获取智能体颜色
                    color = self._get_agent_color(agent, i)
                    
                    # 获取智能体大小
                    agent_size = getattr(agent, 'size', 0.15)
                    radius = int(agent_size * scale)
                    radius = max(radius, 15)  # 最小15像素
                    
                    # 检查是否可移动（说话者不能移动）
                    is_movable = getattr(agent, 'movable', True)
                    
                    # 绘制智能体（圆形，不可移动的用双圆表示）
                    painter.setBrush(QBrush(color))
                    painter.setPen(QPen(color.darker(120), 2))
                    painter.drawEllipse(x - radius, y - radius, radius * 2, radius * 2)
                    
                    # 如果不可移动（如说话者），绘制双圆
                    if not is_movable:
                        painter.setBrush(Qt.NoBrush)
                        painter.setPen(QPen(color.darker(150), 2, Qt.DashLine))
                        painter.drawEllipse(x - radius - 4, y - radius - 4, 
                                          (radius + 4) * 2, (radius + 4) * 2)
                    
                    # 绘制速度方向（箭头）- 只对可移动的智能体
                    if is_movable and hasattr(agent.state, 'p_vel'):
                        vel = agent.state.p_vel
                        vel_mag = np.sqrt(vel[0]**2 + vel[1]**2)
                        if vel_mag > 0.01:  # 只有速度足够大时才画箭头
                            vel_scale = scale * 0.5
                            end_x = x + int(vel[0] * vel_scale)
                            end_y = y - int(vel[1] * vel_scale)  # Y轴翻转
                            painter.setPen(QPen(color.darker(150), 3))
                            painter.drawLine(x, y, end_x, end_y)
                    
                    # 绘制智能体标签
                    painter.setPen(QColor(255, 255, 255))
                    font = QFont("Arial", 9, QFont.Bold)
                    painter.setFont(font)
                    label = self._get_agent_label(agent, i)
                    painter.drawText(x - 5, y + 4, label)
        
        # 绘制图例
        legend_x = 10
        legend_y = 20
        painter.setPen(QColor(50, 50, 50))
        font = QFont("Arial", 9)
        painter.setFont(font)
        
        if hasattr(self.world, 'agents'):
            for i, agent in enumerate(self.world.agents):
                color = self._get_agent_color(agent, i)
                
                painter.setBrush(QBrush(color))
                painter.setPen(QPen(color.darker(120), 1))
                painter.drawEllipse(legend_x, legend_y + i * 20, 14, 14)
                
                painter.setPen(QColor(50, 50, 50))
                # 显示更详细的名称
                if self.scenario_name == "simple_speaker_listener":
                    if i == 0:
                        name = "Speaker (说话者)"
                    else:
                        name = "Listener (听众)"
                elif hasattr(agent, 'name'):
                    name = agent.name
                    if hasattr(agent, 'adversary') and agent.adversary:
                        name += " (对抗)"
                else:
                    name = f"Agent {i}"
                    
                painter.drawText(legend_x + 20, legend_y + i * 20 + 11, name)


class RewardPlotCanvas(FigureCanvas):
    """奖励曲线绘制画布"""
    
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 4), dpi=100, facecolor='#f0f0f0')
        super().__init__(self.fig)
        self.setParent(parent)
        
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#ffffff')
        self.ax.set_title('Training Reward Curve', fontsize=12, fontweight='bold')
        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Reward')
        self.ax.grid(True, alpha=0.3)
        
        self.rewards = []
        self.line, = self.ax.plot([], [], 'b-', linewidth=2, label='Total Reward')
        self.ax.legend(loc='upper left')
        
        self.fig.tight_layout()
        
    def update_plot(self, rewards):
        """更新奖励曲线"""
        self.rewards = rewards
        if len(rewards) > 0:
            x = list(range(len(rewards)))
            self.line.set_data(x, rewards)
            self.ax.relim()
            self.ax.autoscale_view()
            
            # 添加移动平均线
            if len(rewards) >= 10:
                window = min(50, len(rewards))
                avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                if hasattr(self, 'avg_line'):
                    self.avg_line.set_data(range(window-1, len(rewards)), avg)
                else:
                    self.avg_line, = self.ax.plot(range(window-1, len(rewards)), avg, 
                                                   'r--', linewidth=1.5, alpha=0.7, label=f'MA({window})')
                    self.ax.legend(loc='upper left')
            
        self.draw()


class MADDPGVisualizer(QMainWindow):
    """MADDPG训练可视化主窗口"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MADDPG 训练可视化工具")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #ecf0f1;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #3498db;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #2c3e50;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #1f618d;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
            QComboBox, QSpinBox, QDoubleSpinBox {
                padding: 5px;
                border: 1px solid #bdc3c7;
                border-radius: 3px;
            }
        """)
        
        # 训练相关变量
        self.env = None
        self.trainers = None
        self.is_training = False
        self.is_paused = False
        self.visualization_enabled = True
        self.training_thread = None
        self.stop_event = Event()
        self.data_queue = queue.Queue()
        
        # 观察控制
        self.watch_next_episode = False  # 是否观察下一个Episode
        self.is_watching = False  # 当前是否在观察模式
        
        # 信号
        self.signals = TrainingSignals()
        self.signals.update_frame.connect(self.on_update_frame)
        self.signals.update_stats.connect(self.on_update_stats)
        self.signals.training_finished.connect(self.on_training_finished)
        
        # 训练统计
        self.episode_rewards = []
        self.all_rewards = []
        self.current_episode = 0
        self.current_step = 0
        
        # 当前场景名称
        self.current_scenario = ""
        
        # 模型保存路径
        self.model_save_dir = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
        
        self.init_ui()
        
        # 定时器用于更新UI
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.process_data_queue)
        self.update_timer.start(30)  # 30ms更新一次
        
    def init_ui(self):
        """初始化UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # 左侧 - 控制面板
        left_panel = self.create_control_panel()
        main_layout.addWidget(left_panel, 1)
        
        # 右侧 - 可视化区域
        right_panel = self.create_visualization_panel()
        main_layout.addWidget(right_panel, 3)
        
        # 状态栏
        self.statusBar().showMessage("就绪 - Ready")
        
    def create_control_panel(self):
        """创建控制面板 - 使用标签页组织"""
        panel = QWidget()
        panel.setMaximumWidth(350)
        main_layout = QVBoxLayout(panel)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # ============ 顶部固定区域：控制按钮和状态 ============
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(5)
        
        # 训练控制按钮（横向排列）
        ctrl_layout = QHBoxLayout()
        self.start_btn = QPushButton("▶ 开始")
        self.start_btn.clicked.connect(self.start_training)
        self.start_btn.setToolTip("开始训练")
        ctrl_layout.addWidget(self.start_btn)
        
        self.pause_btn = QPushButton("⏸ 暂停")
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setToolTip("暂停/继续训练")
        ctrl_layout.addWidget(self.pause_btn)
        
        self.stop_btn = QPushButton("⏹ 停止")
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("QPushButton { background-color: #e74c3c; }")
        self.stop_btn.setToolTip("停止训练")
        ctrl_layout.addWidget(self.stop_btn)
        top_layout.addLayout(ctrl_layout)
        
        # 观察按钮
        self.watch_btn = QPushButton("👁 观察下一轮")
        self.watch_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                font-size: 13px;
                padding: 8px;
            }
            QPushButton:hover { background-color: #2ecc71; }
            QPushButton:disabled { background-color: #bdc3c7; }
        """)
        self.watch_btn.clicked.connect(self.trigger_watch_next)
        self.watch_btn.setEnabled(False)
        self.watch_btn.setToolTip("在下一个Episode开始时可视化观察智能体行为")
        top_layout.addWidget(self.watch_btn)
        
        # 状态标签
        self.watch_status_label = QLabel("状态: 待命")
        self.watch_status_label.setStyleSheet("color: #7f8c8d; font-size: 11px;")
        top_layout.addWidget(self.watch_status_label)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximumHeight(15)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #3498db;
                border-radius: 3px;
                text-align: center;
                font-size: 10px;
            }
            QProgressBar::chunk { background-color: #3498db; }
        """)
        top_layout.addWidget(self.progress_bar)
        
        main_layout.addWidget(top_widget)
        
        # ============ 标签页区域 ============
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #bdc3c7;
                border-radius: 3px;
            }
            QTabBar::tab {
                background: #ecf0f1;
                padding: 6px 12px;
                margin-right: 2px;
                border-top-left-radius: 3px;
                border-top-right-radius: 3px;
            }
            QTabBar::tab:selected {
                background: #3498db;
                color: white;
            }
        """)
        
        # Tab 1: 场景与基础参数
        tab1 = self._create_basic_tab()
        self.tab_widget.addTab(tab1, "🎮 场景")
        
        # Tab 2: 网络参数
        tab2 = self._create_network_tab()
        self.tab_widget.addTab(tab2, "🧠 网络")
        
        # Tab 3: 模型管理
        tab3 = self._create_model_tab()
        self.tab_widget.addTab(tab3, "💾 模型")
        
        main_layout.addWidget(self.tab_widget, 1)
        
        # ============ 底部固定区域：统计信息 ============
        stats_group = QGroupBox("📊 训练统计")
        stats_layout = QGridLayout(stats_group)
        stats_layout.setSpacing(3)
        
        stats_layout.addWidget(QLabel("Episode:"), 0, 0)
        self.episode_label = QLabel("0")
        self.episode_label.setStyleSheet("font-weight: bold; color: #2980b9;")
        stats_layout.addWidget(self.episode_label, 0, 1)
        
        stats_layout.addWidget(QLabel("步数:"), 0, 2)
        self.steps_label = QLabel("0")
        self.steps_label.setStyleSheet("font-weight: bold; color: #2980b9;")
        stats_layout.addWidget(self.steps_label, 0, 3)
        
        stats_layout.addWidget(QLabel("奖励:"), 1, 0)
        self.reward_label = QLabel("0.00")
        self.reward_label.setStyleSheet("font-weight: bold; color: #27ae60;")
        stats_layout.addWidget(self.reward_label, 1, 1)
        
        stats_layout.addWidget(QLabel("智能体:"), 1, 2)
        self.agents_label = QLabel("0")
        stats_layout.addWidget(self.agents_label, 1, 3)
        
        main_layout.addWidget(stats_group)
        
        return panel
    
    def _create_basic_tab(self):
        """创建基础参数标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(8)
        
        # 场景选择
        scene_group = QGroupBox("场景选择")
        scene_layout = QGridLayout(scene_group)
        
        scene_layout.addWidget(QLabel("场景:"), 0, 0)
        self.scenario_combo = QComboBox()
        self.scenario_combo.addItems([
            "simple", "simple_spread", "simple_reference",
            "simple_speaker_listener", "simple_push", "simple_tag",
            "simple_adversary", "simple_crypto", "simple_world_comm"
        ])
        self.scenario_combo.setCurrentText("simple_spread")
        self.scenario_combo.setToolTip("选择训练场景")
        scene_layout.addWidget(self.scenario_combo, 0, 1)
        
        scene_layout.addWidget(QLabel("对抗数:"), 1, 0)
        self.adversary_spin = QSpinBox()
        self.adversary_spin.setRange(0, 10)
        self.adversary_spin.setValue(0)
        self.adversary_spin.setToolTip("对抗智能体数量")
        scene_layout.addWidget(self.adversary_spin, 1, 1)
        
        layout.addWidget(scene_group)
        
        # 训练参数
        params_group = QGroupBox("训练参数")
        params_layout = QGridLayout(params_group)
        
        params_layout.addWidget(QLabel("Episodes:"), 0, 0)
        self.episodes_spin = QSpinBox()
        self.episodes_spin.setRange(100, 1000000)
        self.episodes_spin.setValue(5000)
        self.episodes_spin.setSingleStep(1000)
        self.episodes_spin.setToolTip("训练回合数\n简单场景: 5000+\n复杂场景: 50000+")
        params_layout.addWidget(self.episodes_spin, 0, 1)
        
        params_layout.addWidget(QLabel("回合长度:"), 1, 0)
        self.episode_len_spin = QSpinBox()
        self.episode_len_spin.setRange(10, 500)
        self.episode_len_spin.setValue(25)
        self.episode_len_spin.setToolTip("每回合最大步数")
        params_layout.addWidget(self.episode_len_spin, 1, 1)
        
        params_layout.addWidget(QLabel("学习率:"), 2, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 0.5)
        self.lr_spin.setValue(0.01)
        self.lr_spin.setDecimals(5)
        self.lr_spin.setSingleStep(0.001)
        self.lr_spin.setToolTip("梯度下降步长\n建议: 0.001-0.01")
        params_layout.addWidget(self.lr_spin, 2, 1)
        
        params_layout.addWidget(QLabel("折扣γ:"), 3, 0)
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.0, 0.9999)
        self.gamma_spin.setValue(0.95)
        self.gamma_spin.setDecimals(4)
        self.gamma_spin.setSingleStep(0.01)
        self.gamma_spin.setToolTip("未来奖励折扣率\n建议: 0.9-0.99")
        params_layout.addWidget(self.gamma_spin, 3, 1)
        
        layout.addWidget(params_group)
        
        # 可视化设置
        vis_group = QGroupBox("渲染速度")
        vis_layout = QVBoxLayout(vis_group)
        
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 100)
        self.speed_slider.setValue(30)
        self.speed_slider.setToolTip("调节观察模式下的渲染速度")
        vis_layout.addWidget(self.speed_slider)
        
        speed_labels = QHBoxLayout()
        speed_labels.addWidget(QLabel("慢"))
        speed_labels.addStretch()
        speed_labels.addWidget(QLabel("快"))
        vis_layout.addLayout(speed_labels)
        
        layout.addWidget(vis_group)
        layout.addStretch()
        
        return tab
    
    def _create_network_tab(self):
        """创建网络参数标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(8)
        
        # 网络结构
        struct_group = QGroupBox("网络结构")
        struct_layout = QGridLayout(struct_group)
        
        struct_layout.addWidget(QLabel("隐藏单元:"), 0, 0)
        self.units_spin = QSpinBox()
        self.units_spin.setRange(16, 1024)
        self.units_spin.setValue(64)
        self.units_spin.setSingleStep(32)
        self.units_spin.setToolTip("每层神经元数量\n越多表达能力越强\n建议: 64-256")
        struct_layout.addWidget(self.units_spin, 0, 1)
        
        struct_layout.addWidget(QLabel("网络层数:"), 1, 0)
        self.layers_spin = QSpinBox()
        self.layers_spin.setRange(2, 10)
        self.layers_spin.setValue(3)
        self.layers_spin.setToolTip("总层数(隐藏+输出)\n3=2隐藏层\n建议: 3-5")
        struct_layout.addWidget(self.layers_spin, 1, 1)
        
        layout.addWidget(struct_group)
        
        # 训练配置
        train_group = QGroupBox("训练配置")
        train_layout = QGridLayout(train_group)
        
        train_layout.addWidget(QLabel("批次大小:"), 0, 0)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(32, 8192)
        self.batch_spin.setValue(1024)
        self.batch_spin.setSingleStep(256)
        self.batch_spin.setToolTip("每次采样数据量\n越大越稳定\n建议: 512-2048")
        train_layout.addWidget(self.batch_spin, 0, 1)
        
        train_layout.addWidget(QLabel("经验池:"), 1, 0)
        self.buffer_spin = QSpinBox()
        self.buffer_spin.setRange(10000, 10000000)
        self.buffer_spin.setValue(1000000)
        self.buffer_spin.setSingleStep(100000)
        self.buffer_spin.setToolTip("经验回放池大小\n建议: 10万-100万")
        train_layout.addWidget(self.buffer_spin, 1, 1)
        
        layout.addWidget(train_group)
        
        # 参数说明
        help_group = QGroupBox("💡 参数说明")
        help_layout = QVBoxLayout(help_group)
        help_text = QLabel(
            "• 隐藏单元: 网络宽度，影响表达能力\n"
            "• 网络层数: 网络深度，影响复杂度\n"
            "• 批次大小: 采样量，影响训练稳定性\n"
            "• 经验池: 历史记忆容量"
        )
        help_text.setStyleSheet("color: #7f8c8d; font-size: 10px;")
        help_text.setWordWrap(True)
        help_layout.addWidget(help_text)
        
        layout.addWidget(help_group)
        layout.addStretch()
        
        return tab
    
    def _create_model_tab(self):
        """创建模型管理标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)
        
        # 保存模型
        save_group = QGroupBox("保存模型")
        save_layout = QVBoxLayout(save_group)
        
        self.save_btn = QPushButton("💾 保存当前模型")
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #9b59b6;
                font-size: 13px;
                padding: 10px;
            }
            QPushButton:hover { background-color: #8e44ad; }
            QPushButton:disabled { background-color: #bdc3c7; }
        """)
        self.save_btn.clicked.connect(self.save_model)
        self.save_btn.setEnabled(False)
        self.save_btn.setToolTip("暂停训练后可保存模型")
        save_layout.addWidget(self.save_btn)
        
        save_hint = QLabel("提示: 暂停训练后可保存")
        save_hint.setStyleSheet("color: #7f8c8d; font-size: 10px;")
        save_layout.addWidget(save_hint)
        
        layout.addWidget(save_group)
        
        # 加载模型
        load_group = QGroupBox("加载模型")
        load_layout = QVBoxLayout(load_group)
        
        self.load_btn = QPushButton("📂 加载并继续训练")
        self.load_btn.setStyleSheet("""
            QPushButton {
                background-color: #f39c12;
                font-size: 13px;
                padding: 10px;
            }
            QPushButton:hover { background-color: #d68910; }
        """)
        self.load_btn.clicked.connect(self.load_model_and_train)
        self.load_btn.setToolTip("加载保存的模型继续训练")
        load_layout.addWidget(self.load_btn)
        
        self.inference_btn = QPushButton("🔍 加载并推理演示")
        self.inference_btn.setStyleSheet("""
            QPushButton {
                background-color: #1abc9c;
                font-size: 13px;
                padding: 10px;
            }
            QPushButton:hover { background-color: #16a085; }
        """)
        self.inference_btn.clicked.connect(self.load_model_and_inference)
        self.inference_btn.setToolTip("加载模型进行推理演示")
        load_layout.addWidget(self.inference_btn)
        
        layout.addWidget(load_group)
        
        # 使用说明
        usage_group = QGroupBox("📖 使用流程")
        usage_layout = QVBoxLayout(usage_group)
        usage_text = QLabel(
            "训练流程:\n"
            "1. 设置参数 → 开始训练\n"
            "2. 暂停 → 保存模型\n"
            "3. 下次加载继续训练\n\n"
            "推理流程:\n"
            "1. 加载训练好的模型\n"
            "2. 自动开始可视化演示"
        )
        usage_text.setStyleSheet("color: #7f8c8d; font-size: 10px;")
        usage_text.setWordWrap(True)
        usage_layout.addWidget(usage_text)
        
        layout.addWidget(usage_group)
        layout.addStretch()
        
        return tab
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #3498db;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #3498db;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        layout.addStretch()
        return panel
        
    def create_visualization_panel(self):
        """创建可视化面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 使用分割器分隔环境渲染和曲线图
        splitter = QSplitter(Qt.Vertical)
        
        # 环境渲染
        env_group = QGroupBox("环境渲染")
        env_layout = QVBoxLayout(env_group)
        self.env_renderer = EnvironmentRenderer()
        env_layout.addWidget(self.env_renderer)
        splitter.addWidget(env_group)
        
        # 奖励曲线
        plot_group = QGroupBox("奖励曲线")
        plot_layout = QVBoxLayout(plot_group)
        self.reward_plot = RewardPlotCanvas()
        plot_layout.addWidget(self.reward_plot)
        splitter.addWidget(plot_group)
        
        splitter.setSizes([400, 300])
        layout.addWidget(splitter)
        
        return panel
    
    def trigger_watch_next(self):
        """触发观察下一轮"""
        self.watch_next_episode = True
        self.watch_btn.setEnabled(False)
        self.watch_status_label.setText("状态: 等待下一轮开始...")
        self.watch_status_label.setStyleSheet("color: #e67e22; font-weight: bold;")
        self.statusBar().showMessage("将在下一轮Episode开始时进行可视化观察...")
        
    def toggle_pause(self):
        """切换暂停状态"""
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_btn.setText("▶ 继续")
            self.save_btn.setEnabled(True)  # 暂停时允许保存
            self.statusBar().showMessage("训练已暂停 - 可以保存模型")
        else:
            self.pause_btn.setText("⏸ 暂停")
            self.save_btn.setEnabled(False)
            self.statusBar().showMessage("训练中 - Training...")
    
    def save_model(self):
        """保存模型权重"""
        if self.trainers is None:
            QMessageBox.warning(self, "警告", "没有可保存的模型！")
            return
        
        # 创建保存目录
        scenario = self.current_scenario
        save_dir = os.path.join(self.model_save_dir, scenario)
        os.makedirs(save_dir, exist_ok=True)
        
        # 弹出文件选择对话框
        default_name = f"{scenario}_ep{self.current_episode}"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存模型",
            os.path.join(save_dir, default_name),
            "PyTorch Model (*.pth);;All Files (*)"
        )
        
        if file_path:
            try:
                # 保存所有智能体的模型
                checkpoint = {
                    'scenario': scenario,
                    'episode': self.current_episode,
                    'step': self.current_step,
                    'all_rewards': self.all_rewards.copy() if self.all_rewards else [],
                    # 保存训练参数以便恢复
                    'params': {
                        'num_units': self.units_spin.value(),
                        'num_layers': self.layers_spin.value(),
                        'buffer_size': self.buffer_spin.value(),
                        'lr': self.lr_spin.value(),
                        'gamma': self.gamma_spin.value(),
                        'batch_size': self.batch_spin.value(),
                        'max_episode_len': self.episode_len_spin.value(),
                    },
                    'agents': []
                }
                
                for i, trainer in enumerate(self.trainers):
                    agent_data = {
                        'name': trainer.name,
                        'actor_state_dict': trainer.actor.state_dict(),
                        'critic_state_dict': trainer.critic.state_dict(),
                        'actor_target_state_dict': trainer.actor_target.state_dict(),
                        'critic_target_state_dict': trainer.critic_target.state_dict(),
                        'actor_optimizer_state_dict': trainer.actor_optimizer.state_dict(),
                        'critic_optimizer_state_dict': trainer.critic_optimizer.state_dict(),
                    }
                    checkpoint['agents'].append(agent_data)
                
                torch.save(checkpoint, file_path)
                
                QMessageBox.information(
                    self, 
                    "成功", 
                    f"模型已保存到:\n{file_path}\n\nEpisode: {self.current_episode}\n平均奖励: {np.mean(self.all_rewards[-100:]) if self.all_rewards else 0:.2f}"
                )
                self.statusBar().showMessage(f"模型已保存: {os.path.basename(file_path)}")
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存模型失败:\n{str(e)}")
    
    def load_model_and_train(self):
        """加载模型并继续训练"""
        self._load_model(inference_only=False)
    
    def load_model_and_inference(self):
        """加载模型进行推理"""
        self._load_model(inference_only=True)
    
    def _load_model(self, inference_only=False):
        """加载模型"""
        # 选择模型文件
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择模型文件",
            self.model_save_dir,
            "PyTorch Model (*.pth);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            checkpoint = torch.load(file_path, map_location=U.get_device(), weights_only=False)
            
            # 获取场景信息
            saved_scenario = checkpoint.get('scenario', 'simple_spread')
            saved_episode = checkpoint.get('episode', 0)
            saved_rewards = checkpoint.get('all_rewards', [])
            
            # 恢复训练参数（如果有保存）
            saved_params = checkpoint.get('params', {})
            if saved_params:
                if 'num_units' in saved_params:
                    self.units_spin.setValue(saved_params['num_units'])
                if 'num_layers' in saved_params:
                    self.layers_spin.setValue(saved_params['num_layers'])
                if 'buffer_size' in saved_params:
                    self.buffer_spin.setValue(saved_params['buffer_size'])
                if 'lr' in saved_params:
                    self.lr_spin.setValue(saved_params['lr'])
                if 'gamma' in saved_params:
                    self.gamma_spin.setValue(saved_params['gamma'])
                if 'batch_size' in saved_params:
                    self.batch_spin.setValue(saved_params['batch_size'])
                if 'max_episode_len' in saved_params:
                    self.episode_len_spin.setValue(saved_params['max_episode_len'])
            
            # 设置场景
            idx = self.scenario_combo.findText(saved_scenario)
            if idx >= 0:
                self.scenario_combo.setCurrentIndex(idx)
            
            self.current_scenario = saved_scenario
            self.current_episode = saved_episode
            self.all_rewards = saved_rewards
            
            # 更新奖励曲线
            if saved_rewards:
                self.reward_plot.update_plot(saved_rewards)
            
            # 存储checkpoint供训练线程使用
            self.loaded_checkpoint = checkpoint
            self.inference_mode = inference_only
            
            if inference_only:
                # 推理模式 - 直接开始观察
                self.start_inference()
            else:
                # 继续训练模式
                params_info = ""
                if saved_params:
                    params_info = f"\n\n参数: 层数={saved_params.get('num_layers', 3)}, 单元数={saved_params.get('num_units', 64)}"
                QMessageBox.information(
                    self,
                    "模型已加载",
                    f"场景: {saved_scenario}\nEpisode: {saved_episode}\n奖励历史: {len(saved_rewards)} episodes{params_info}\n\n点击'开始训练'继续训练。"
                )
                self.statusBar().showMessage(f"模型已加载 - 场景: {saved_scenario}, Episode: {saved_episode}")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载模型失败:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def start_inference(self):
        """开始推理模式"""
        self.is_training = True
        self.is_paused = False
        self.stop_event.clear()
        self.watch_next_episode = True  # 自动开始观察
        
        # 更新UI状态
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.watch_btn.setEnabled(True)
        self.scenario_combo.setEnabled(False)
        self.load_btn.setEnabled(False)
        self.inference_btn.setEnabled(False)
        
        # 在新线程中运行推理
        self.training_thread = Thread(target=self.inference_loop, daemon=True)
        self.training_thread.start()
        
        self.statusBar().showMessage("推理模式 - Inference Mode")
            
    def start_training(self):
        """开始训练"""
        self.is_training = True
        self.is_paused = False
        self.stop_event.clear()
        
        # 如果没有加载模型，重置奖励历史
        if not hasattr(self, 'loaded_checkpoint') or self.loaded_checkpoint is None:
            self.all_rewards = []
            self.current_episode = 0
        
        self.watch_next_episode = False
        self.is_watching = False
        
        # 更新UI状态
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.watch_btn.setEnabled(True)
        self.scenario_combo.setEnabled(False)
        self.load_btn.setEnabled(False)
        self.inference_btn.setEnabled(False)
        
        # 在新线程中运行训练
        self.training_thread = Thread(target=self.training_loop, daemon=True)
        self.training_thread.start()
        
        self.statusBar().showMessage("训练中 - Training...")
        
    def stop_training(self):
        """停止训练"""
        self.stop_event.set()
        self.is_training = False
        self.statusBar().showMessage("正在停止训练...")
        
    def on_training_finished(self):
        """训练结束回调"""
        self.is_training = False
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.watch_btn.setEnabled(False)
        self.save_btn.setEnabled(True if self.trainers else False)
        self.scenario_combo.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.inference_btn.setEnabled(True)
        self.watch_status_label.setText("状态: 训练已结束")
        self.watch_status_label.setStyleSheet("color: #7f8c8d; font-style: italic;")
        self.statusBar().showMessage("训练完成 - Training Finished (可保存模型)")
        
        # 清除加载的checkpoint
        self.loaded_checkpoint = None
        
    def on_update_frame(self, world):
        """更新渲染帧"""
        self.env_renderer.update_world(world)
        
    def on_update_stats(self, stats):
        """更新统计信息"""
        self.episode_label.setText(str(stats.get('episode', 0)))
        self.steps_label.setText(str(stats.get('steps', 0)))
        self.reward_label.setText(f"{stats.get('reward', 0):.2f}")
        self.agents_label.setText(str(stats.get('n_agents', 0)))
        
        progress = int(stats.get('progress', 0))
        self.progress_bar.setValue(progress)
        
        if 'rewards' in stats:
            self.reward_plot.update_plot(stats['rewards'])
        
        # 更新观察状态
        if stats.get('watching', False):
            self.watch_status_label.setText(f"状态: 正在观察 Episode {stats.get('episode', 0)}")
            self.watch_status_label.setStyleSheet("color: #27ae60; font-weight: bold;")
        elif not self.watch_next_episode:
            self.watch_btn.setEnabled(True)
            if self.is_training:
                self.watch_status_label.setText("状态: 待命 (点击观察按钮)")
                self.watch_status_label.setStyleSheet("color: #7f8c8d; font-style: italic;")
            
    def process_data_queue(self):
        """处理数据队列"""
        try:
            while True:
                data_type, data = self.data_queue.get_nowait()
                if data_type == 'frame':
                    self.signals.update_frame.emit(data)
                elif data_type == 'stats':
                    self.signals.update_stats.emit(data)
        except queue.Empty:
            pass
    
    def _create_env_and_trainers(self, args):
        """创建环境和训练器"""
        from multiagent.environment import MultiAgentEnv
        import multiagent.scenarios as scenarios
        
        scenario = args.scenario
        scenario_module = scenarios.load(scenario + ".py").Scenario()
        world = scenario_module.make_world()
        self.env = MultiAgentEnv(world, scenario_module.reset_world, 
                                 scenario_module.reward, scenario_module.observation)
        
        # 设置渲染器的环境引用和场景名称
        self.env_renderer.set_env(self.env, scenario)
        self.current_scenario = scenario
        
        # 创建训练器
        obs_shape_n = [self.env.observation_space[i].shape for i in range(self.env.n)]
        num_adversaries = min(self.env.n, args.num_adversaries)
        
        self.trainers = []
        for i in range(num_adversaries):
            self.trainers.append(MADDPGAgentTrainer(
                f"agent_{i}", None, obs_shape_n, self.env.action_space, i, args,
                local_q_func=(args.adv_policy == 'ddpg')))
        for i in range(num_adversaries, self.env.n):
            self.trainers.append(MADDPGAgentTrainer(
                f"agent_{i}", None, obs_shape_n, self.env.action_space, i, args,
                local_q_func=(args.good_policy == 'ddpg')))
        
        return num_adversaries
    
    def _load_checkpoint_to_trainers(self, checkpoint):
        """将checkpoint加载到训练器"""
        agents_data = checkpoint.get('agents', [])
        
        for i, agent_data in enumerate(agents_data):
            if i < len(self.trainers):
                trainer = self.trainers[i]
                trainer.actor.load_state_dict(agent_data['actor_state_dict'])
                trainer.critic.load_state_dict(agent_data['critic_state_dict'])
                trainer.actor_target.load_state_dict(agent_data['actor_target_state_dict'])
                trainer.critic_target.load_state_dict(agent_data['critic_target_state_dict'])
                trainer.actor_optimizer.load_state_dict(agent_data['actor_optimizer_state_dict'])
                trainer.critic_optimizer.load_state_dict(agent_data['critic_optimizer_state_dict'])
            
    def training_loop(self):
        """训练循环（在单独线程中运行）"""
        try:
            # 获取训练参数
            scenario = self.scenario_combo.currentText()
            num_episodes = self.episodes_spin.value()
            max_episode_len = self.episode_len_spin.value()
            num_adversaries = self.adversary_spin.value()
            lr = self.lr_spin.value()
            gamma = self.gamma_spin.value()
            batch_size = self.batch_spin.value()
            num_units = self.units_spin.value()
            num_layers = self.layers_spin.value()
            buffer_size = self.buffer_spin.value()
            
            # 创建参数对象
            class Args:
                pass
            args = Args()
            args.scenario = scenario
            args.num_episodes = num_episodes
            args.max_episode_len = max_episode_len
            args.num_adversaries = num_adversaries
            args.lr = lr
            args.batch_size = batch_size
            args.num_units = num_units
            args.num_layers = num_layers        # 网络层数
            args.buffer_size = buffer_size      # 经验池大小
            args.gamma = gamma                  # 折扣因子
            args.good_policy = 'maddpg'
            args.adv_policy = 'maddpg'
            
            # 创建环境和训练器
            self._create_env_and_trainers(args)
            
            # 如果有加载的checkpoint，恢复权重
            if hasattr(self, 'loaded_checkpoint') and self.loaded_checkpoint is not None:
                self._load_checkpoint_to_trainers(self.loaded_checkpoint)
                start_episode = self.loaded_checkpoint.get('episode', 0)
                all_rewards = self.loaded_checkpoint.get('all_rewards', []).copy()
                self.loaded_checkpoint = None  # 清除
            else:
                start_episode = 0
                all_rewards = []
            
            # 发送初始统计
            self.data_queue.put(('stats', {
                'episode': start_episode,
                'steps': 0,
                'reward': np.mean(all_rewards[-100:]) if all_rewards else 0,
                'n_agents': self.env.n,
                'progress': int((start_episode / num_episodes) * 100) if num_episodes > 0 else 0,
                'rewards': all_rewards.copy()
            }))
            
            # 训练循环
            episode_rewards = [0.0]
            obs_n = self.env.reset()
            episode_step = 0
            train_step = 0
            current_episode = start_episode + 1
            watching_this_episode = False
            
            while current_episode <= num_episodes and not self.stop_event.is_set():
                # 暂停检查
                while self.is_paused and not self.stop_event.is_set():
                    time.sleep(0.1)
                
                if self.stop_event.is_set():
                    break
                
                # 检查是否开始观察新的Episode
                if episode_step == 0 and self.watch_next_episode:
                    watching_this_episode = True
                    self.watch_next_episode = False
                    self.is_watching = True
                
                # 获取动作
                action_n = [agent.action(obs) for agent, obs in zip(self.trainers, obs_n)]
                
                # 环境步进
                new_obs_n, rew_n, done_n, info_n = self.env.step(action_n)
                episode_step += 1
                train_step += 1
                done = all(done_n)
                terminal = (episode_step >= args.max_episode_len)
                
                # 收集经验
                for i, agent in enumerate(self.trainers):
                    agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
                obs_n = new_obs_n
                
                # 累计奖励
                for rew in rew_n:
                    episode_rewards[-1] += rew
                
                # 可视化渲染 - 仅在观察模式下渲染
                if watching_this_episode:
                    speed = self.speed_slider.value()
                    delay = (101 - speed) / 500.0  # 转换为秒，更慢的延迟
                    time.sleep(delay)
                    
                    # 发送世界状态用于渲染
                    self.data_queue.put(('frame', self.env.world))
                    
                    # 发送统计（带有watching标志）
                    avg_reward = np.mean(all_rewards[-100:]) if all_rewards else episode_rewards[-1]
                    progress = int((current_episode / num_episodes) * 100)
                    self.data_queue.put(('stats', {
                        'episode': current_episode,
                        'steps': train_step,
                        'reward': avg_reward,
                        'n_agents': self.env.n,
                        'progress': progress,
                        'watching': True
                    }))
                
                # Episode结束
                if done or terminal:
                    all_rewards.append(episode_rewards[-1])
                    self.all_rewards = all_rewards.copy()
                    self.current_episode = current_episode
                    self.current_step = train_step
                    
                    # 如果正在观察，结束观察模式
                    if watching_this_episode:
                        watching_this_episode = False
                        self.is_watching = False
                    
                    # 更新统计
                    avg_reward = np.mean(all_rewards[-100:]) if all_rewards else 0
                    progress = int((current_episode / num_episodes) * 100)
                    
                    self.data_queue.put(('stats', {
                        'episode': current_episode,
                        'steps': train_step,
                        'reward': avg_reward,
                        'n_agents': self.env.n,
                        'progress': progress,
                        'rewards': all_rewards.copy(),
                        'watching': False
                    }))
                    
                    obs_n = self.env.reset()
                    episode_step = 0
                    episode_rewards.append(0)
                    current_episode += 1
                
                # 更新网络
                for agent in self.trainers:
                    agent.preupdate()
                for agent in self.trainers:
                    agent.update(self.trainers, train_step)
            
            # 训练结束
            self.signals.training_finished.emit()
            
        except Exception as e:
            print(f"Training error: {e}")
            import traceback
            traceback.print_exc()
            self.signals.training_finished.emit()
    
    def inference_loop(self):
        """推理循环（在单独线程中运行）"""
        try:
            # 获取参数
            scenario = self.scenario_combo.currentText()
            max_episode_len = self.episode_len_spin.value()
            num_adversaries = self.adversary_spin.value()
            num_units = self.units_spin.value()
            num_layers = self.layers_spin.value()
            
            # 创建参数对象
            class Args:
                pass
            args = Args()
            args.scenario = scenario
            args.num_episodes = 9999999  # 无限推理
            args.max_episode_len = max_episode_len
            args.num_adversaries = num_adversaries
            args.lr = 0.01
            args.batch_size = 1024
            args.num_units = num_units
            args.num_layers = num_layers
            args.buffer_size = 100000  # 推理不需要大buffer
            args.gamma = 0.95
            args.good_policy = 'maddpg'
            args.adv_policy = 'maddpg'
            
            # 创建环境和训练器
            self._create_env_and_trainers(args)
            
            # 加载checkpoint
            if hasattr(self, 'loaded_checkpoint') and self.loaded_checkpoint is not None:
                self._load_checkpoint_to_trainers(self.loaded_checkpoint)
                self.loaded_checkpoint = None
            
            # 推理循环
            current_episode = 1
            obs_n = self.env.reset()
            episode_step = 0
            episode_reward = 0
            
            self.statusBar().showMessage("推理模式 - 自动观察每一轮")
            
            while not self.stop_event.is_set():
                # 暂停检查
                while self.is_paused and not self.stop_event.is_set():
                    time.sleep(0.1)
                
                if self.stop_event.is_set():
                    break
                
                # 获取动作（不添加噪声）
                action_n = []
                for agent, obs in zip(self.trainers, obs_n):
                    # 推理时不添加探索噪声
                    with torch.no_grad():
                        action = agent.action(obs)
                    action_n.append(action)
                
                # 环境步进
                new_obs_n, rew_n, done_n, info_n = self.env.step(action_n)
                episode_step += 1
                done = all(done_n)
                terminal = (episode_step >= args.max_episode_len)
                
                # 累计奖励
                for rew in rew_n:
                    episode_reward += rew
                
                obs_n = new_obs_n
                
                # 渲染
                speed = self.speed_slider.value()
                delay = (101 - speed) / 300.0  # 推理模式稍快一些
                time.sleep(delay)
                
                # 发送世界状态用于渲染
                self.data_queue.put(('frame', self.env.world))
                
                # 发送统计
                self.data_queue.put(('stats', {
                    'episode': current_episode,
                    'steps': episode_step,
                    'reward': episode_reward,
                    'n_agents': self.env.n,
                    'progress': 0,  # 推理模式不显示进度
                    'watching': True
                }))
                
                # Episode结束
                if done or terminal:
                    obs_n = self.env.reset()
                    episode_step = 0
                    episode_reward = 0
                    current_episode += 1
            
            # 推理结束
            self.signals.training_finished.emit()
            
        except Exception as e:
            print(f"Inference error: {e}")
            import traceback
            traceback.print_exc()
            self.signals.training_finished.emit()
            
    def closeEvent(self, event):
        """窗口关闭事件"""
        self.stop_event.set()
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=2)
        event.accept()


def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用字体
    font = QFont("Microsoft YaHei", 9)
    app.setFont(font)
    
    window = MADDPGVisualizer()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
