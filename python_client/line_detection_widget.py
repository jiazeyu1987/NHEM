"""
Line Detection Widget - ROI1 图像显示与绿线交点可视化组件
使用 matplotlib 实现 ROI1 图像显示，支持交互式操作和绿线交点检测可视化。
"""

import logging
import base64
import io
import traceback
import requests
from typing import Dict, Any, Optional, Tuple, List, Callable
from enum import Enum
import numpy as np
import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from PIL import Image
import threading
import time
from datetime import datetime

# Import the line detection API client
import os
import sys

# Check if the API client file exists
api_client_file = os.path.join(os.path.dirname(__file__), 'line_detection_api_client.py')
api_file_exists = os.path.exists(api_client_file)

print(f"DEBUG: API client file exists: {api_file_exists}")
print(f"DEBUG: API client file path: {api_client_file}")
print(f"DEBUG: Current working directory: {os.getcwd()}")
print(f"DEBUG: Python path: {sys.path[:3]}")  # Show first 3 entries

LINE_DETECTION_API_AVAILABLE = False
LineDetectionAPIClient = None
LineDetectionAPIError = None
LineDetectionStatus = None

if api_file_exists:
    try:
        from line_detection_api_client import (
            LineDetectionAPIClient,
            LineDetectionAPIError,
            LineDetectionStatus,
            create_line_detection_client,
            toggle_line_detection,
            get_detection_status_simple
        )
        LINE_DETECTION_API_AVAILABLE = True
        print("SUCCESS: LineDetectionAPIClient imported successfully")
    except ImportError as e:
        print(f"WARNING: LineDetectionAPIClient import failed: {str(e)}")
        print(f"WARNING: ImportError type: {type(e).__name__}")
        print("WARNING: Available modules in line_detection_api_client:")

        # Try to inspect what's available in the module
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("line_detection_api_client", api_client_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                available_attrs = [attr for attr in dir(module) if not attr.startswith('_')]
                print(f"WARNING: Available attributes: {available_attrs}")
        except Exception as inspect_error:
            print(f"WARNING: Could not inspect module: {inspect_error}")

        LINE_DETECTION_API_AVAILABLE = False
        # Set safe defaults
        class LineDetectionAPIError(Exception):
            pass
        class LineDetectionStatus:
            pass
        def create_line_detection_client(*args, **kwargs):
            return None
        def toggle_line_detection(*args, **kwargs):
            return None
        def get_detection_status_simple(*args, **kwargs):
            return None
    except Exception as e:
        print(f"ERROR: Unexpected error during import: {str(e)}")
        print(f"ERROR: Error type: {type(e).__name__}")
        LINE_DETECTION_API_AVAILABLE = False
        # Set safe defaults
        class LineDetectionAPIError(Exception):
            pass
        class LineDetectionStatus:
            pass
        def create_line_detection_client(*args, **kwargs):
            return None
        def toggle_line_detection(*args, **kwargs):
            return None
        def get_detection_status_simple(*args, **kwargs):
            return None
else:
    print("ERROR: line_detection_api_client.py file not found")
    print(f"ERROR: Expected file at: {api_client_file}")
    print(f"ERROR: Files in current directory: {os.listdir('.')[:10]}")  # Show first 10 files

    # Set safe defaults
    class LineDetectionAPIError(Exception):
        pass
    class LineDetectionStatus:
        pass
    def create_line_detection_client(*args, **kwargs):
        return None
    def toggle_line_detection(*args, **kwargs):
        return None
    def get_detection_status_simple(*args, **kwargs):
        return None

print(f"FINAL STATUS: LINE_DETECTION_API_AVAILABLE = {LINE_DETECTION_API_AVAILABLE}")

# 设置matplotlib字体和样式
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["axes.unicode_minus"] = False

logger = logging.getLogger(__name__)


class StatusState(Enum):
    """线条相交点检测状态枚举"""
    DISABLED = "disabled"  # 未启用
    ENABLED_NO_DETECTION = "enabled_no_detection"  # 已启用但未识别
    DETECTION_SUCCESS = "detection_success"  # 检测成功
    DETECTION_ERROR = "detection_error"  # 检测错误


class ChineseStatusDisplay:
    """
    中文状态显示组件 - 线条相交点检测状态显示
    支持四种状态的颜色编码和实时更新
    """

    def __init__(self, parent_frame: ttk.Frame, config: Optional[Dict[str, Any]] = None):
        """
        初始化中文状态显示组件

        Args:
            parent_frame: 父级Tkinter框架
            config: 配置字典，包含颜色和字体设置
        """
        self.parent_frame = parent_frame
        self.config = config or {}

        # 状态配置
        self.status_colors = {
            StatusState.DISABLED: self.config.get('color_disabled', '#808080'),      # 灰色
            StatusState.ENABLED_NO_DETECTION: self.config.get('color_enabled', '#FFA500'),  # 黄色/橙色
            StatusState.DETECTION_SUCCESS: self.config.get('color_success', '#00AA00'),    # 绿色
            StatusState.DETECTION_ERROR: self.config.get('color_error', '#FF0000')        # 红色
        }

        # 中文字体配置
        self.font_family = self.config.get('font_family', 'Microsoft YaHei')
        self.font_size = self.config.get('font_size', 10)
        self.font_bold_size = self.config.get('font_bold_size', 10)

        self.chinese_font = (self.font_family, self.font_size, 'normal')
        self.chinese_font_bold = (self.font_family, self.font_bold_size, 'bold')
        self.timestamp_font = (self.font_family, 8, 'normal')

        # 当前状态
        self.current_state = StatusState.DISABLED
        self.current_intersection = None
        self.current_confidence = None
        self.current_error_msg = None
        self.last_update_time = None

        # 状态历史记录
        self.status_history = []  # 最近10次状态变化
        self.max_history_size = self.config.get('max_history_size', 10)

        # UI组件
        self.status_frame = None
        self.status_label = None
        self.details_label = None
        self.timestamp_label = None
        self.history_button = None

        # 创建UI
        self.setup_status_display()

        logger.info("ChineseStatusDisplay initialized")

    def setup_status_display(self):
        """设置状态显示UI"""
        # 创建主状态框架
        self.status_frame = ttk.LabelFrame(
            self.parent_frame,
            text="线条相交点检测状态",
            padding=10
        )
        self.status_frame.pack(fill='x', padx=5, pady=5)

        # 主状态标签
        self.status_label = ttk.Label(
            self.status_frame,
            text=self.format_status_text(StatusState.DISABLED),
            font=self.chinese_font_bold,
            foreground=self.status_colors[StatusState.DISABLED]
        )
        self.status_label.pack(anchor='w', pady=(0, 5))

        # 详细信息标签
        self.details_label = ttk.Label(
            self.status_frame,
            text="等待启用检测功能...",
            font=self.chinese_font,
            foreground='gray'
        )
        self.details_label.pack(anchor='w', pady=(0, 5))

        # 时间戳框架
        timestamp_frame = ttk.Frame(self.status_frame)
        timestamp_frame.pack(fill='x', pady=(5, 0))

        # 时间戳标签
        self.timestamp_label = ttk.Label(
            timestamp_frame,
            text="",
            font=self.timestamp_font,
            foreground='gray'
        )
        self.timestamp_label.pack(side='left')

        # 历史记录按钮
        self.history_button = ttk.Button(
            timestamp_frame,
            text="历史记录 (0)",
            command=self.show_status_history,
            width=12
        )
        self.history_button.pack(side='right')

        # 分隔符
        ttk.Separator(self.status_frame, orient='horizontal').pack(fill='x', pady=(10, 0))

        logger.info("Status display UI setup completed")

    def format_status_text(self, state: StatusState, **kwargs) -> str:
        """
        格式化中文状态文本

        Args:
            state: 状态枚举
            **kwargs: 额外参数（intersection, confidence, error_msg）

        Returns:
            格式化的中文状态文本
        """
        intersection = kwargs.get('intersection')
        confidence = kwargs.get('confidence')
        error_msg = kwargs.get('error_msg')

        if state == StatusState.DISABLED:
            return "线条相交点: 未启用"

        elif state == StatusState.ENABLED_NO_DETECTION:
            return "线条相交点: 已启用 - 未识别"

        elif state == StatusState.DETECTION_SUCCESS:
            if intersection and confidence is not None:
                x, y = intersection
                return f"线条相交点: 已识别 ({int(x)}, {int(y)}) 置信度: {confidence:.0%}"
            else:
                return "线条相交点: 已识别 - 数据不完整"

        elif state == StatusState.DETECTION_ERROR:
            if error_msg:
                # 限制错误消息长度
                truncated_error = error_msg[:50] + "..." if len(error_msg) > 50 else error_msg
                return f"线条相交点: 检测错误: {truncated_error}"
            else:
                return "线条相交点: 检测错误: 未知错误"

        else:
            return "线条相交点: 未知状态"

    def format_details_text(self, state: StatusState, **kwargs) -> str:
        """
        格式化详细信息文本

        Args:
            state: 状态枚举
            **kwargs: 额外参数

        Returns:
            格式化的详细信息文本
        """
        if state == StatusState.DISABLED:
            return "检测功能尚未启用，请点击控制面板中的"

        elif state == StatusState.ENABLED_NO_DETECTION:
            return "检测功能已启用，正在等待图像数据或执行检测操作..."

        elif state == StatusState.DETECTION_SUCCESS:
            intersection = kwargs.get('intersection')
            confidence = kwargs.get('confidence')

            if intersection and confidence is not None:
                x, y = intersection
                details = f"检测到线条交点位置: ({x:.2f}, {y:.2f})\n"
                details += f"检测置信度: {confidence:.2%} ({self._get_confidence_level(confidence)})"

                if confidence >= 0.7:
                    details += "\n✓ 检测结果可信度较高"
                elif confidence >= 0.4:
                    details += "\n⚠ 检测结果可信度中等，建议验证"
                else:
                    details += "\n✗ 检测结果可信度较低，需要重新检测"

                return details
            else:
                return "检测成功但数据不完整，请重新执行检测"

        elif state == StatusState.DETECTION_ERROR:
            error_msg = kwargs.get('error_msg', '未知错误')
            return f"检测过程中发生错误: {error_msg}\n请检查网络连接和图像数据，然后重试"

        else:
            return "状态信息不完整，请联系技术支持"

    def _get_confidence_level(self, confidence: float) -> str:
        """获取置信度等级描述"""
        if confidence >= 0.8:
            return "非常高"
        elif confidence >= 0.6:
            return "较高"
        elif confidence >= 0.4:
            return "中等"
        elif confidence >= 0.2:
            return "较低"
        else:
            return "非常低"

    def update_status(self,
                     state: StatusState,
                     intersection: Optional[Tuple[float, float]] = None,
                     confidence: Optional[float] = None,
                     error_msg: Optional[str] = None):
        """
        更新状态显示

        Args:
            state: 新状态
            intersection: 交点坐标 (x, y)
            confidence: 置信度 (0.0 - 1.0)
            error_msg: 错误消息
        """
        try:
            # 记录状态变化历史
            self._record_status_change(state, intersection, confidence, error_msg)

            # 更新当前状态
            self.current_state = state
            self.current_intersection = intersection
            self.current_confidence = confidence
            self.current_error_msg = error_msg
            self.last_update_time = datetime.now()

            # 更新UI显示
            self._update_display()

            # 立即刷新UI
            self.parent_frame.update()

            logger.info(f"Status updated to {state.value}: {self.format_status_text(state, intersection=intersection, confidence=confidence, error_msg=error_msg)}")

        except Exception as e:
            logger.error(f"Error updating status: {e}")

    def _update_display(self):
        """更新UI显示内容"""
        try:
            # 更新主状态文本
            status_text = self.format_status_text(
                self.current_state,
                intersection=self.current_intersection,
                confidence=self.current_confidence,
                error_msg=self.current_error_msg
            )

            self.status_label.config(
                text=status_text,
                foreground=self.status_colors[self.current_state]
            )

            # 更新详细信息文本
            details_text = self.format_details_text(
                self.current_state,
                intersection=self.current_intersection,
                confidence=self.current_confidence,
                error_msg=self.current_error_msg
            )

            self.details_label.config(text=details_text)

            # 更新时间戳
            if self.last_update_time:
                timestamp_str = self.last_update_time.strftime("%Y-%m-%d %H:%M:%S")
                self.timestamp_label.config(text=f"更新时间: {timestamp_str}")

            # 更新历史记录按钮
            self.history_button.config(text=f"历史记录 ({len(self.status_history)})")

        except Exception as e:
            logger.error(f"Error updating display: {e}")

    def _record_status_change(self,
                             state: StatusState,
                             intersection: Optional[Tuple[float, float]] = None,
                             confidence: Optional[float] = None,
                             error_msg: Optional[str] = None):
        """记录状态变化到历史"""
        history_entry = {
            'timestamp': datetime.now(),
            'state': state,
            'intersection': intersection,
            'confidence': confidence,
            'error_msg': error_msg,
            'status_text': self.format_status_text(state, intersection=intersection, confidence=confidence, error_msg=error_msg)
        }

        # 添加到历史记录
        self.status_history.append(history_entry)

        # 限制历史记录大小
        if len(self.status_history) > self.max_history_size:
            self.status_history.pop(0)

    def show_status_history(self):
        """显示状态历史记录对话框"""
        try:
            if not self.status_history:
                self._show_info_dialog("状态历史", "暂无状态变化记录")
                return

            # 创建历史记录窗口
            history_window = tk.Toplevel(self.parent_frame)
            history_window.title("线条相交点检测状态历史")
            history_window.geometry("600x400")
            history_window.resizable(True, True)

            # 创建滚动文本框
            frame = ttk.Frame(history_window)
            frame.pack(fill='both', expand=True, padx=10, pady=10)

            text_widget = tk.Text(frame, wrap=tk.WORD, font=self.chinese_font)
            scrollbar = ttk.Scrollbar(frame, orient="vertical", command=text_widget.yview)
            text_widget.configure(yscrollcommand=scrollbar.set)

            text_widget.pack(side='left', fill='both', expand=True)
            scrollbar.pack(side='right', fill='y')

            # 添加历史记录内容
            for i, entry in enumerate(reversed(self.status_history), 1):
                timestamp = entry['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
                status_text = entry['status_text']

                # 根据状态选择颜色标签
                color_map = {
                    StatusState.DISABLED: 'gray',
                    StatusState.ENABLED_NO_DETECTION: 'orange',
                    StatusState.DETECTION_SUCCESS: 'green',
                    StatusState.DETECTION_ERROR: 'red'
                }
                color = color_map.get(entry['state'], 'black')

                text_widget.insert(tk.END, f"{i}. [{timestamp}]\n", 'timestamp')
                text_widget.insert(tk.END, f"   {status_text}\n", ('status', color))

                # 添加详细信息
                if entry['intersection'] and entry['confidence'] is not None:
                    x, y = entry['intersection']
                    text_widget.insert(tk.END,
                        f"   详情: 坐标({x:.2f}, {y:.2f}), 置信度{entry['confidence']:.2%}\n", 'details')

                if entry['error_msg']:
                    text_widget.insert(tk.END, f"   错误: {entry['error_msg']}\n", 'error')

                text_widget.insert(tk.END, "\n")

            # 配置文本样式
            text_widget.tag_config('timestamp', font=('Arial', 9, 'bold'))
            text_widget.tag_config('details', font=('Arial', 9), foreground='gray')
            text_widget.tag_config('error', font=('Arial', 9), foreground='red')

            for color in ['gray', 'orange', 'green', 'red']:
                text_widget.tag_config(color, foreground=color)

            text_widget.config(state='disabled')

            # 添加关闭按钮
            button_frame = ttk.Frame(history_window)
            button_frame.pack(fill='x', padx=10, pady=(0, 10))

            ttk.Button(button_frame, text="关闭",
                      command=history_window.destroy).pack(side='right')
            ttk.Button(button_frame, text="清除历史",
                      command=self.clear_status_history).pack(side='right', padx=(0, 10))

        except Exception as e:
            logger.error(f"Error showing status history: {e}")
            self._show_error_dialog("错误", f"无法显示历史记录: {e}")

    def _show_info_dialog(self, title: str, message: str):
        """显示信息对话框"""
        try:
            dialog = tk.Toplevel(self.parent_frame)
            dialog.title(title)
            dialog.geometry("400x150")
            dialog.resizable(False, False)

            ttk.Label(dialog, text=message, font=self.chinese_font).pack(pady=20)
            ttk.Button(dialog, text="确定", command=dialog.destroy).pack(pady=10)
        except Exception as e:
            logger.error(f"Error showing info dialog: {e}")

    def _show_error_dialog(self, title: str, message: str):
        """显示错误对话框"""
        try:
            dialog = tk.Toplevel(self.parent_frame)
            dialog.title(title)
            dialog.geometry("400x150")
            dialog.resizable(False, False)

            ttk.Label(dialog, text=message, font=self.chinese_font,
                     foreground='red').pack(pady=20)
            ttk.Button(dialog, text="确定", command=dialog.destroy).pack(pady=10)
        except Exception as e:
            logger.error(f"Error showing error dialog: {e}")

    def get_current_status(self) -> Dict[str, Any]:
        """
        获取当前状态信息

        Returns:
            当前状态字典
        """
        return {
            'state': self.current_state,
            'state_text': self.format_status_text(
                self.current_state,
                intersection=self.current_intersection,
                confidence=self.current_confidence,
                error_msg=self.current_error_msg
            ),
            'intersection': self.current_intersection,
            'confidence': self.current_confidence,
            'error_msg': self.current_error_msg,
            'last_update_time': self.last_update_time.isoformat() if self.last_update_time else None,
            'color': self.status_colors[self.current_state]
        }

    def get_status_history(self) -> List[Dict[str, Any]]:
        """
        获取状态变化历史

        Returns:
            状态历史列表
        """
        return [
            {
                'timestamp': entry['timestamp'].isoformat(),
                'state': entry['state'].value,
                'intersection': entry['intersection'],
                'confidence': entry['confidence'],
                'error_msg': entry['error_msg'],
                'status_text': entry['status_text']
            }
            for entry in self.status_history
        ]

    def clear_status_history(self):
        """清除状态历史记录"""
        self.status_history.clear()
        if hasattr(self, 'history_button'):
            self.history_button.config(text="历史记录 (0)")
        logger.info("Status history cleared")

    def set_status_colors(self, colors: Dict[StatusState, str]):
        """
        自定义状态颜色

        Args:
            colors: 状态颜色字典
        """
        try:
            for state, color in colors.items():
                if state in self.status_colors:
                    self.status_colors[state] = color

            # 更新当前显示的颜色
            if hasattr(self, 'status_label'):
                self.status_label.config(foreground=self.status_colors[self.current_state])

            logger.info("Status colors updated")
        except Exception as e:
            logger.error(f"Error setting status colors: {e}")

    def set_font_config(self, font_family: str = None, font_size: int = None):
        """
        设置字体配置

        Args:
            font_family: 字体族名
            font_size: 字体大小
        """
        try:
            if font_family:
                self.font_family = font_family
                self.chinese_font = (font_family, self.font_size, 'normal')
                self.chinese_font_bold = (font_family, self.font_bold_size, 'bold')
                self.timestamp_font = (font_family, 8, 'normal')

            if font_size:
                self.font_size = font_size
                self.chinese_font = (self.font_family, font_size, 'normal')

            # 更新现有标签的字体
            if hasattr(self, 'status_label'):
                self.status_label.config(font=self.chinese_font_bold)
            if hasattr(self, 'details_label'):
                self.details_label.config(font=self.chinese_font)
            if hasattr(self, 'timestamp_label'):
                self.timestamp_label.config(font=self.timestamp_font)

            logger.info("Font configuration updated")
        except Exception as e:
            logger.error(f"Error setting font config: {e}")

    def animate_status_change(self, new_state: StatusState, **kwargs):
        """
        带动画效果的状态变化

        Args:
            new_state: 新状态
            **kwargs: 状态参数
        """
        try:
            if not hasattr(self, 'status_label'):
                self.update_status(new_state, **kwargs)
                return

            # 闪烁效果
            original_color = self.status_label.cget('foreground')

            for i in range(3):  # 闪烁3次
                self.status_label.config(foreground='white')
                self.parent_frame.update()
                self.parent_frame.after(100)

                self.status_label.config(foreground=original_color)
                self.parent_frame.update()
                self.parent_frame.after(100)

            # 更新到新状态
            self.update_status(new_state, **kwargs)

        except Exception as e:
            logger.error(f"Error animating status change: {e}")
            self.update_status(new_state, **kwargs)


class LineDetectionControls:
    """ROI1 绿线交点检测控制面板 - 支持中文界面和加载状态"""

    def __init__(self, parent_frame: ttk.Frame, callbacks: Optional[Dict[str, Callable]] = None):
        """
        初始化控制面板

        Args:
            parent_frame: 父级Tkinter框架
            callbacks: 回调函数字典，包含 'on_toggle', 'on_manual_detection' 等
        """
        self.parent_frame = parent_frame
        self.callbacks = callbacks or {}

        # 检测状态
        self.detection_enabled = False
        self.loading_states = {}  # 跟踪按钮加载状态

        # 按钮组件
        self.toggle_btn = None
        self.manual_btn = None
        self.refresh_btn = None
        self.clear_btn = None

        # 状态标签
        self.status_label = None

        # 设置支持中文的字体
        self.chinese_font = ('Microsoft YaHei', 10, 'normal')
        self.chinese_font_bold = ('Microsoft YaHei', 10, 'bold')

        # 创建控制面板
        self.setup_controls()

        logger.info("LineDetectionControls initialized")

    def setup_controls(self):
        """设置控制面板UI"""
        # 创建主控制框架
        self.control_frame = ttk.LabelFrame(self.parent_frame, text="检测控制", padding=10)
        self.control_frame.pack(fill='x', padx=5, pady=5)

        # 按钮行框架
        button_row1 = ttk.Frame(self.control_frame)
        button_row1.pack(fill='x', pady=(0, 5))

        # 启用/禁用检测切换按钮
        self.toggle_btn = ttk.Button(
            button_row1,
            text="启用检测",
            command=self.handle_toggle_click,
            width=15
        )
        self.toggle_btn.pack(side='left', padx=(0, 10))

        # 手动检测按钮
        self.manual_btn = ttk.Button(
            button_row1,
            text="手动检测",
            command=self.handle_manual_detection,
            width=15
        )
        self.manual_btn.pack(side='left', padx=(0, 10))

        # 立即刷新按钮
        self.refresh_btn = ttk.Button(
            button_row1,
            text="立即刷新",
            command=self.handle_refresh,
            width=15
        )
        self.refresh_btn.pack(side='left', padx=(0, 10))

        # 第二行按钮框架
        button_row2 = ttk.Frame(self.control_frame)
        button_row2.pack(fill='x', pady=(5, 0))

        # 清除覆盖层按钮
        self.clear_btn = ttk.Button(
            button_row2,
            text="清除覆盖层",
            command=self.handle_clear_overlays,
            width=15
        )
        self.clear_btn.pack(side='left', padx=(0, 10))

        # 重置视图按钮
        self.reset_btn = ttk.Button(
            button_row2,
            text="重置视图",
            command=self.handle_reset_view,
            width=15
        )
        self.reset_btn.pack(side='left', padx=(0, 10))

        # 保存截图按钮
        self.save_btn = ttk.Button(
            button_row2,
            text="保存截图",
            command=self.handle_save_screenshot,
            width=15
        )
        self.save_btn.pack(side='left', padx=(0, 10))

        # 状态显示框架
        status_frame = ttk.Frame(self.control_frame)
        status_frame.pack(fill='x', pady=(10, 0))

        # 状态标签
        self.status_label = ttk.Label(
            status_frame,
            text="检测状态: 未启用",
            font=self.chinese_font,
            foreground='gray'
        )
        self.status_label.pack(side='left')

        # 分隔符
        ttk.Separator(status_frame, orient='vertical').pack(side='left', fill='y', padx=10)

        # 检测计数标签
        self.count_label = ttk.Label(
            status_frame,
            text="检测次数: 0",
            font=self.chinese_font,
            foreground='gray'
        )
        self.count_label.pack(side='left')

        # 初始化按钮状态
        self.update_button_states()

        # 添加工具提示
        self.setup_tooltips()

    def setup_tooltips(self):
        """设置按钮工具提示"""
        try:
            # 创建工具提示
            self.tooltip_toggle = self.create_tooltip(self.toggle_btn, "启用或禁用自动绿线交点检测")
            self.tooltip_manual = self.create_tooltip(self.manual_btn, "立即执行一次检测操作")
            self.tooltip_refresh = self.create_tooltip(self.refresh_btn, "刷新当前图像和数据")
            self.tooltip_clear = self.create_tooltip(self.clear_btn, "清除所有检测覆盖层")
            self.tooltip_reset = self.create_tooltip(self.reset_btn, "重置图像视图到默认状态")
            self.tooltip_save = self.create_tooltip(self.save_btn, "保存当前截图到文件")
        except Exception as e:
            logger.warning(f"Failed to setup tooltips: {e}")

    def create_tooltip(self, widget, text):
        """创建工具提示"""
        try:
            def on_enter(event):
                tooltip = tk.Toplevel()
                tooltip.wm_overrideredirect(True)
                tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")

                label = tk.Label(tooltip, text=text, background="lightyellow",
                               relief="solid", borderwidth=1, font=self.chinese_font)
                label.pack()

                widget.tooltip = tooltip

            def on_leave(event):
                if hasattr(widget, 'tooltip'):
                    widget.tooltip.destroy()
                    del widget.tooltip

            widget.bind("<Enter>", on_enter)
            widget.bind("<Leave>", on_leave)
        except Exception as e:
            # 如果工具提示创建失败，静默继续
            logger.warning(f"Failed to create tooltip for {text}: {e}")

    def set_loading_state(self, button_name: str, loading: bool):
        """
        设置按钮加载状态

        Args:
            button_name: 按钮名称 ('toggle', 'manual', 'refresh')
            loading: 是否处于加载状态
        """
        self.loading_states[button_name] = loading

        if button_name == 'toggle' and self.toggle_btn:
            if loading:
                self.toggle_btn.config(text="处理中...", state="disabled")
            else:
                self.toggle_btn.config(state="normal")
                self.update_button_states()  # 更新按钮文本

        elif button_name == 'manual' and self.manual_btn:
            if loading:
                self.manual_btn.config(text="检测中...", state="disabled")
            else:
                self.manual_btn.config(text="手动检测", state="normal")

        elif button_name == 'refresh' and self.refresh_btn:
            if loading:
                self.refresh_btn.config(text="刷新中...", state="disabled")
            else:
                self.refresh_btn.config(text="立即刷新", state="normal")

        # 更新状态标签
        self.update_status_display()

    def update_button_states(self):
        """根据当前检测状态更新按钮显示"""
        if self.toggle_btn and not self.loading_states.get('toggle', False):
            if self.detection_enabled:
                self.toggle_btn.config(text="禁用检测")
            else:
                self.toggle_btn.config(text="启用检测")

    def update_status_display(self):
        """更新状态显示"""
        if self.status_label:
            if any(self.loading_states.values()):
                self.status_label.config(text="检测状态: 处理中...", foreground='orange')
            elif self.detection_enabled:
                self.status_label.config(text="检测状态: 已启用", foreground='green')
            else:
                self.status_label.config(text="检测状态: 未启用", foreground='red')

    def handle_toggle_click(self):
        """处理启用/禁用检测按钮点击"""
        if self.loading_states.get('toggle', False):
            return  # 防止重复点击

        # 在后台线程中处理以避免UI冻结
        thread = threading.Thread(target=self._process_toggle, daemon=True)
        thread.start()

    def _process_toggle(self):
        """在后台线程中处理切换操作"""
        try:
            # 设置加载状态
            self.parent_frame.after(0, lambda: self.set_loading_state('toggle', True))

            # 模拟API调用延迟
            time.sleep(0.5)

            # 切换检测状态
            self.detection_enabled = not self.detection_enabled

            # 调用回调函数
            if 'on_toggle' in self.callbacks:
                self.parent_frame.after(0, lambda: self.callbacks['on_toggle'](self.detection_enabled))

            # 更新检测计数
            if hasattr(self, 'detection_count'):
                self.detection_count += 1
            else:
                self.detection_count = 1

            # 重置加载状态
            self.parent_frame.after(0, lambda: self.set_loading_state('toggle', False))

            # 更新计数显示
            self.parent_frame.after(0, lambda: self._update_count_display())

            logger.info(f"Detection toggled: {'enabled' if self.detection_enabled else 'disabled'}")

        except Exception as e:
            logger.error(f"Error in toggle operation: {e}")
            # 重置加载状态
            self.parent_frame.after(0, lambda: self.set_loading_state('toggle', False))

    def handle_manual_detection(self):
        """处理手动检测按钮点击"""
        if self.loading_states.get('manual', False):
            return  # 防止重复点击

        # 在后台线程中处理
        thread = threading.Thread(target=self._process_manual_detection, daemon=True)
        thread.start()

    def _process_manual_detection(self):
        """在后台线程中处理手动检测"""
        try:
            # 设置加载状态
            self.parent_frame.after(0, lambda: self.set_loading_state('manual', True))

            # 调用回调函数
            if 'on_manual_detection' in self.callbacks:
                self.parent_frame.after(0, self.callbacks['on_manual_detection'])

            # 模拟处理延迟
            time.sleep(0.8)

            # 重置加载状态
            self.parent_frame.after(0, lambda: self.set_loading_state('manual', False))

            logger.info("Manual detection completed")

        except Exception as e:
            logger.error(f"Error in manual detection: {e}")
            # 重置加载状态
            self.parent_frame.after(0, lambda: self.set_loading_state('manual', False))

    def handle_refresh(self):
        """处理刷新按钮点击"""
        if self.loading_states.get('refresh', False):
            return

        thread = threading.Thread(target=self._process_refresh, daemon=True)
        thread.start()

    def _process_refresh(self):
        """在后台线程中处理刷新操作"""
        try:
            self.parent_frame.after(0, lambda: self.set_loading_state('refresh', True))

            # 调用回调函数
            if 'on_refresh' in self.callbacks:
                self.parent_frame.after(0, self.callbacks['on_refresh'])

            time.sleep(0.3)  # 模拟刷新延迟

            self.parent_frame.after(0, lambda: self.set_loading_state('refresh', False))

            logger.info("Refresh completed")

        except Exception as e:
            logger.error(f"Error in refresh: {e}")
            self.parent_frame.after(0, lambda: self.set_loading_state('refresh', False))

    def handle_clear_overlays(self):
        """处理清除覆盖层按钮点击"""
        try:
            if 'on_clear_overlays' in self.callbacks:
                self.callbacks['on_clear_overlays']()
            logger.info("Overlays cleared")
        except Exception as e:
            logger.error(f"Error clearing overlays: {e}")

    def handle_reset_view(self):
        """处理重置视图按钮点击"""
        try:
            if 'on_reset_view' in self.callbacks:
                self.callbacks['on_reset_view']()
            logger.info("View reset")
        except Exception as e:
            logger.error(f"Error resetting view: {e}")

    def handle_save_screenshot(self):
        """处理保存截图按钮点击"""
        try:
            if 'on_save_screenshot' in self.callbacks:
                self.callbacks['on_save_screenshot']()
            logger.info("Screenshot saved")
        except Exception as e:
            logger.error(f"Error saving screenshot: {e}")

    def _update_count_display(self):
        """更新检测计数显示"""
        if hasattr(self, 'count_label') and hasattr(self, 'detection_count'):
            self.count_label.config(text=f"检测次数: {self.detection_count}")

    def set_detection_enabled(self, enabled: bool):
        """
        从外部设置检测状态

        Args:
            enabled: 检测是否启用
        """
        self.detection_enabled = enabled
        self.update_button_states()
        self.update_status_display()

    def get_control_state(self) -> Dict[str, Any]:
        """获取控制面板状态信息"""
        return {
            'detection_enabled': self.detection_enabled,
            'loading_states': self.loading_states.copy(),
            'detection_count': getattr(self, 'detection_count', 0)
        }

    def set_callbacks(self, callbacks: Dict[str, Callable]):
        """设置或更新回调函数"""
        self.callbacks.update(callbacks)

    def enable_all_buttons(self, enabled: bool = True):
        """启用或禁用所有按钮"""
        buttons = [
            (self.toggle_btn, 'toggle'),
            (self.manual_btn, 'manual'),
            (self.refresh_btn, 'refresh'),
            (self.clear_btn, 'clear'),
            (self.reset_btn, 'reset'),
            (self.save_btn, 'save')
        ]

        for btn, name in buttons:
            if btn and not self.loading_states.get(name, False):  # 不在加载中的按钮
                btn.config(state="normal" if enabled else "disabled")


class LineDetectionWidget:
    """ROI1 绿线交点检测可视化组件"""

    def __init__(self, parent_frame, config: Optional[Dict[str, Any]] = None):
        """
        初始化LineDetectionWidget

        Args:
            parent_frame: 父级Tkinter框架
            config: 配置字典，包含显示参数和样式设置
        """
        self.parent_frame = parent_frame
        self.config = config or {}

        # 组件配置
        self.figsize = self.config.get('figsize', (8, 6))
        self.dpi = self.config.get('dpi', 100)
        self.enable_toolbar = self.config.get('enable_toolbar', True)
        self.dark_theme = self.config.get('dark_theme', True)

        # API客户端配置
        self.api_base_url = self.config.get('api_base_url', 'http://localhost:8421')
        self.api_password = self.config.get('api_password', '31415')
        self.api_timeout = self.config.get('api_timeout', 10)
        self.enable_api_integration = self.config.get('enable_api_integration', True)

        # API客户端实例
        self.api_client: Optional[LineDetectionAPIClient] = None

        # Log API integration state for debugging
        print(f"WIDGET DEBUG: enable_api_integration = {self.enable_api_integration}")
        print(f"WIDGET DEBUG: LINE_DETECTION_API_AVAILABLE = {LINE_DETECTION_API_AVAILABLE}")
        print(f"WIDGET DEBUG: api_base_url = {self.api_base_url}")
        print(f"WIDGET DEBUG: api_password = {'*' * len(self.api_password)}")  # Hide password

        # 图像数据
        self.current_roi1_data = None
        self.image_shape = None
        print(f"WIDGET INIT DEBUG: image_shape initialized to None")
        self.pixel_coordinates = []

        # matplotlib 组件
        self.fig = None
        self.ax = None
        self.canvas = None
        self.toolbar = None

        # 交互状态
        self.mouse_pressed = False
        self.last_mouse_pos = None
        self.coordinate_callback = None

        # 显示状态
        self.image_displayed = False
        self.grid_enabled = True
        self.crosshair_enabled = True

        # 可视化元素存储
        self.overlay_elements = {
            'detected_lines': [],
            'intersection_circles': [],
            'intersection_crosshairs': [],
            'confidence_texts': [],
            'line_patches': []
        }

        # 可视化配置
        self.visualization_config = {
            'intersection_outer_radius': 6,  # 外圆半径（红色）
            'intersection_inner_radius': 3,  # 内圆半径（橙色）
            'crosshair_length': 8,  # 十字线延伸长度
            'line_alpha': 0.8,  # 线条透明度
            'line_width': 2.0,  # 线条宽度
            'text_alpha': 0.8,  # 文本透明度
            'high_confidence_threshold': 0.7,  # 高置信度阈值
            'medium_confidence_threshold': 0.4,  # 中等置信度阈值
            'z_order_lines': 4,  # 线条层级
            'z_order_intersections': 5,  # 交点层级
            'z_order_text': 6  # 文本层级
        }

        # 控制面板配置
        self.enable_control_panel = self.config.get('enable_control_panel', True)
        self.control_panel = None

        # 中文状态显示配置
        self.enable_chinese_status = self.config.get('enable_chinese_status', True)
        self.chinese_status_display = None

        # 初始化组件
        self.setup_widget()

        # 初始化API客户端（如果启用）
        print("WIDGET DEBUG: Starting API client initialization")
        print(f"WIDGET DEBUG: enable_api_integration = {self.enable_api_integration}")
        print(f"WIDGET DEBUG: LINE_DETECTION_API_AVAILABLE = {LINE_DETECTION_API_AVAILABLE}")

        if self.enable_api_integration and LINE_DETECTION_API_AVAILABLE:
            print("WIDGET DEBUG: Both conditions met, calling initialize_api_client()")
            self.initialize_api_client()
            print(f"WIDGET DEBUG: API client initialization completed. api_client = {self.api_client}")
        elif self.enable_api_integration:
            print("WIDGET DEBUG: API integration enabled but LineDetectionAPIClient not available")
            logger.warning("API integration enabled but LineDetectionAPIClient not available")
            self.enable_api_integration = False
            print("WIDGET DEBUG: Disabled API integration due to unavailability")
        else:
            print("WIDGET DEBUG: API integration not enabled or not available")
            if not self.enable_api_integration:
                print("WIDGET DEBUG: Reason: enable_api_integration is False")
            if not LINE_DETECTION_API_AVAILABLE:
                print("WIDGET DEBUG: Reason: LINE_DETECTION_API_AVAILABLE is False")

        # 初始化错误处理系统（Task 32）
        self._setup_error_handling()

        logger.info("LineDetectionWidget initialized")

    def setup_widget(self):
        """设置完整的Widget结构"""
        # 创建主框架
        self.main_frame = ttk.Frame(self.parent_frame)
        self.main_frame.pack(fill='both', expand=True)

        # 创建中文状态显示（如果启用）
        if self.enable_chinese_status:
            self.setup_chinese_status_display()

        # 创建控制面板（如果启用）
        if self.enable_control_panel:
            self.setup_control_panel()

        # 创建工具栏框架（如果启用）
        if self.enable_toolbar:
            self.toolbar_frame = ttk.Frame(self.main_frame)
            self.toolbar_frame.pack(side='top', fill='x', padx=2, pady=2)

        # 创建matplotlib画布框架
        self.canvas_frame = ttk.Frame(self.main_frame)
        self.canvas_frame.pack(side='top', fill='both', expand=True)

        # 创建状态栏框架
        self.status_frame = ttk.Frame(self.main_frame)
        self.status_frame.pack(side='bottom', fill='x', padx=2, pady=2)

        # 设置matplotlib组件
        self.setup_canvas()

        # 设置工具栏（如果启用）
        if self.enable_toolbar:
            self.setup_toolbar()

        # 设置状态栏
        self.setup_statusbar()

    def get_figure_size(self) -> Tuple[float, float]:
        """
        获取适当的图形尺寸

        Returns:
            图形尺寸(宽度, 高度)英寸
        """
        # 根据父容器大小计算合适的图形尺寸
        try:
            parent_width = self.parent_frame.winfo_width()
            parent_height = self.parent_frame.winfo_height()

            if parent_width > 1 and parent_height > 1:  # 有效尺寸
                # 转换为英寸（考虑DPI）
                width_inches = min(parent_width / self.dpi * 0.9, 12.0)
                height_inches = min(parent_height / self.dpi * 0.8, 8.0)
                return (width_inches, height_inches)
        except:
            pass

        return self.figsize

    def setup_canvas(self):
        """创建matplotlib图形和画布"""
        # 创建图形和轴
        figsize = self.get_figure_size()
        self.fig = Figure(figsize=figsize, dpi=self.dpi, facecolor='#1e1e1e' if self.dark_theme else 'white')

        # 创建主轴
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("ROI1 - Green Line Intersection Detection",
                         fontsize=12, fontweight='bold',
                         color='white' if self.dark_theme else 'black')

        # 设置轴标签
        self.ax.set_xlabel("X Coordinate (pixels)", color='white' if self.dark_theme else 'black')
        self.ax.set_ylabel("Y Coordinate (pixels)", color='white' if self.dark_theme else 'black')

        # 应用深色主题
        if self.dark_theme:
            self.fig.patch.set_facecolor('#1e1e1e')
            self.ax.set_facecolor('#2d2d2d')
            self.ax.spines['bottom'].set_color('white')
            self.ax.spines['top'].set_color('white')
            self.ax.spines['left'].set_color('white')
            self.ax.spines['right'].set_color('white')
            self.ax.tick_params(colors='white')
            self.ax.xaxis.label.set_color('white')
            self.ax.yaxis.label.set_color('white')
        else:
            self.fig.patch.set_facecolor('white')
            self.ax.set_facecolor('white')

        # 启用网格
        self.ax.grid(True, alpha=0.3, color='gray' if self.dark_theme else 'lightgray')

        # 设置初始轴范围（ROI1典型尺寸）
        self.ax.set_xlim(0, 200)
        self.ax.set_ylim(200, 0)  # Y轴反向，图像坐标系

        # 创建Tkinter画布
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        # 绑定鼠标事件
        self.setup_mouse_events()

        # 调整布局
        self.fig.tight_layout()

    def setup_toolbar(self):
        """设置matplotlib工具栏"""
        if self.enable_toolbar and self.fig:
            # 创建自定义工具栏
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
            self.toolbar.update()

            # 添加自定义按钮
            ttk.Separator(self.toolbar_frame, orient='vertical').pack(side='left', padx=5, fill='y')

            # 重置视图按钮
            reset_btn = ttk.Button(self.toolbar_frame, text="Reset View",
                                 command=self.reset_view)
            reset_btn.pack(side='left', padx=2)

            # 切换网格按钮
            self.grid_btn = ttk.Button(self.toolbar_frame, text="Toggle Grid",
                                     command=self.toggle_grid)
            self.grid_btn.pack(side='left', padx=2)

            # 切换十字线按钮
            self.crosshair_btn = ttk.Button(self.toolbar_frame, text="Toggle Crosshair",
                                           command=self.toggle_crosshair)
            self.crosshair_btn.pack(side='left', padx=2)

    def setup_statusbar(self):
        """设置状态栏"""
        # 确保status_frame存在
        if not hasattr(self, 'status_frame') or self.status_frame is None:
            print("WARNING: status_frame not found, creating a new one")
            self.status_frame = ttk.Frame(self.main_frame)
            self.status_frame.pack(side='bottom', fill='x', padx=2, pady=2)

        # 坐标显示标签
        self.coord_label = ttk.Label(self.status_frame, text="X: -- , Y: --",
                                    font=('Courier', 10))
        self.coord_label.pack(side='left', padx=10)

        # 图像信息标签
        self.info_label = ttk.Label(self.status_frame, text="No image loaded",
                                   font=('Arial', 9))
        self.info_label.pack(side='left', padx=10)

        # 分隔符
        ttk.Separator(self.status_frame, orient='vertical').pack(side='left', fill='y', padx=10)

        # 缩放级别显示
        self.zoom_label = ttk.Label(self.status_frame, text="Zoom: 100%",
                                   font=('Arial', 9))
        self.zoom_label.pack(side='left', padx=10)

    def setup_chinese_status_display(self):
        """设置中文状态显示组件"""
        try:
            # 创建状态显示的配置
            status_config = {
                'color_disabled': self.config.get('color_disabled', '#808080'),
                'color_enabled': self.config.get('color_enabled', '#FFA500'),
                'color_success': self.config.get('color_success', '#00AA00'),
                'color_error': self.config.get('color_error', '#FF0000'),
                'font_family': self.config.get('font_family', 'Microsoft YaHei'),
                'font_size': self.config.get('font_size', 10),
                'font_bold_size': self.config.get('font_bold_size', 10),
                'max_history_size': self.config.get('max_history_size', 10)
            }

            # 创建中文状态显示组件
            self.chinese_status_display = ChineseStatusDisplay(self.main_frame, status_config)

            logger.info("Chinese status display setup completed")

        except Exception as e:
            logger.error(f"Error setting up Chinese status display: {e}")

    def setup_control_panel(self):
        """设置中文控制面板"""
        try:
            # 创建控制面板的回调函数
            callbacks = {
                'on_toggle': self._on_detection_toggle,
                'on_manual_detection': self._on_manual_detection,
                'on_refresh': self._on_refresh,
                'on_clear_overlays': self.clear_overlays,
                'on_reset_view': self.reset_view,
                'on_save_screenshot': self._on_save_screenshot
            }

            # 创建控制面板
            self.control_panel = LineDetectionControls(self.main_frame, callbacks)

            logger.info("Control panel setup completed")

        except Exception as e:
            logger.error(f"Error setting up control panel: {e}")

    def _on_detection_toggle(self, enabled: bool):
        """处理检测切换回调"""
        def process_toggle():
            try:
                logger.info(f"Detection toggled to: {'enabled' if enabled else 'disabled'}")

                # 更新状态为处理中
                if self.chinese_status_display:
                    self.chinese_status_display.update_status(StatusState.ENABLED_NO_DETECTION)

                # 如果启用了API集成，使用API客户端
                if self.enable_api_integration and self.api_client:
                    try:
                        if enabled:
                            result = self.api_client.enable_line_detection()
                        else:
                            result = self.api_client.disable_line_detection()

                        if result.get('success', False):
                            # 成功，更新状态
                            if self.chinese_status_display:
                                if enabled:
                                    self.chinese_status_display.update_status(StatusState.ENABLED_NO_DETECTION)
                                else:
                                    self.chinese_status_display.update_status(StatusState.DISABLED)
                            logger.info(f"Detection {'enabled' if enabled else 'disabled'} via API")
                        else:
                            # API返回失败
                            error_msg = result.get('error', 'API operation failed')
                            raise Exception(error_msg)

                    except LineDetectionAPIError as e:
                        logger.error(f"API error toggling detection: {e}")
                        if self.chinese_status_display:
                            self.chinese_status_display.update_status(
                                StatusState.DETECTION_ERROR, error_msg=str(e)
                            )
                        return
                    except Exception as e:
                        logger.error(f"Unexpected error toggling detection: {e}")
                        if self.chinese_status_display:
                            self.chinese_status_display.update_status(
                                StatusState.DETECTION_ERROR, error_msg=str(e)
                            )
                        return
                else:
                    # 离线模式或API集成未启用，仅更新UI状态
                    logger.warning("API integration not available, operating in offline mode")

                    # 更新中文状态显示
                    if self.chinese_status_display:
                        if enabled:
                            self.chinese_status_display.update_status(StatusState.ENABLED_NO_DETECTION)
                        else:
                            self.chinese_status_display.update_status(StatusState.DISABLED)

            except Exception as e:
                logger.error(f"Error in detection toggle callback: {e}")
                # 如果出错，更新为错误状态
                if self.chinese_status_display:
                    self.chinese_status_display.update_status(StatusState.DETECTION_ERROR, error_msg=str(e))

        # 在后台线程中处理以避免UI冻结
        thread = threading.Thread(target=process_toggle, daemon=True)
        thread.start()

    def _on_manual_detection(self):
        """处理手动检测回调"""
        def process_manual_detection():
            try:
                logger.info("Manual detection triggered")

                # 更新状态为处理中
                if self.chinese_status_display:
                    self.chinese_status_display.update_status(StatusState.ENABLED_NO_DETECTION)

                # 如果启用了API集成，使用API客户端进行手动检测
                print(f"MANUAL_DETECTION_DEBUG: enable_api_integration = {self.enable_api_integration}")
                print(f"MANUAL_DETECTION_DEBUG: api_client = {self.api_client}")
                print(f"MANUAL_DETECTION_DEBUG: Condition result = {self.enable_api_integration and self.api_client}")

                if self.enable_api_integration and self.api_client:
                    print("MANUAL_DETECTION_DEBUG: Using API client for manual detection")
                    try:
                        # 获取当前ROI数据用于检测
                        roi_data = None
                        if self.image_shape:
                            # 如果有显示的图像，使用图像的边界作为ROI
                            roi_coords = {
                                'x1': 0,
                                'y1': 0,
                                'x2': self.image_shape[1],
                                'y2': self.image_shape[0]
                            }
                        else:
                            roi_coords = None

                        print(f"MANUAL_DETECTION_DEBUG: Calling API client manual_detection with roi_coords = {roi_coords}")

                        # 执行手动检测
                        detection_result = self.api_client.manual_detection(
                            roi_coordinates=roi_coords,
                            force_refresh=True
                        )
                        print(f"MANUAL_DETECTION_DEBUG: API detection result = {detection_result}")

                        if detection_result.get('success', False):
                            # 处理检测结果
                            lines = detection_result.get('lines', [])
                            intersections = detection_result.get('intersections', [])

                            # 更新可视化
                            self.update_visualization({
                                'lines': lines,
                                'intersections': intersections
                            })

                            # 更新状态
                            if intersections:
                                # 选择最高置信度的交点更新状态
                                best_intersection = max(intersections, key=lambda x: x.get('confidence', 0))
                                point = best_intersection.get('point')
                                confidence = best_intersection.get('confidence', 1.0)

                                if self.chinese_status_display:
                                    self.chinese_status_display.update_status(
                                        StatusState.DETECTION_SUCCESS,
                                        intersection=tuple(point) if point else None,
                                        confidence=confidence
                                    )

                                logger.info(f"Manual detection successful: {len(lines)} lines, {len(intersections)} intersections")
                            else:
                                # 没有检测到交点
                                if self.chinese_status_display:
                                    self.chinese_status_display.update_status(StatusState.ENABLED_NO_DETECTION)
                                logger.info("Manual detection completed: no intersections found")
                        else:
                            # 检测失败
                            error_msg = detection_result.get('error', 'Manual detection failed')
                            raise Exception(error_msg)

                    except LineDetectionAPIError as e:
                        logger.error(f"API error in manual detection: {e}")
                        if self.chinese_status_display:
                            self.chinese_status_display.update_status(
                                StatusState.DETECTION_ERROR, error_msg=str(e)
                            )
                        return
                    except Exception as e:
                        logger.error(f"Unexpected error in manual detection: {e}")
                        if self.chinese_status_display:
                            self.chinese_status_display.update_status(
                                StatusState.DETECTION_ERROR, error_msg=str(e)
                            )
                        return
                else:
                    # 离线模式，提供模拟检测功能
                    print("MANUAL_DETECTION_DEBUG: Entering offline mode - API integration not available")

                    # 详细诊断信息
                    if not self.enable_api_integration:
                        error_reason = "API integration is disabled in configuration"
                        print(f"MANUAL_DETECTION_DEBUG: Reason: {error_reason}")
                    elif not self.api_client:
                        error_reason = "API client is None (initialization failed)"
                        print(f"MANUAL_DETECTION_DEBUG: Reason: {error_reason}")
                    else:
                        error_reason = "Unknown reason"
                        print(f"MANUAL_DETECTION_DEBUG: Reason: {error_reason}")

                    logger.warning(f"API integration not available, manual detection in offline mode. Reason: {error_reason}")

                    # 更新状态显示为离线模式
                    if self.chinese_status_display:
                        error_msg = f"API不可用，使用离线模式 ({error_reason})"
                        print(f"MANUAL_DETECTION_DEBUG: Updating status with error_msg: {error_msg}")
                        self.chinese_status_display.update_status(
                            StatusState.DETECTION_ERROR,
                            error_msg=error_msg
                        )

                    # 执行模拟检测
                    print("MANUAL_DETECTION_DEBUG: Calling _simulate_manual_detection()")
                    self._simulate_manual_detection()
                    print("MANUAL_DETECTION_DEBUG: _simulate_manual_detection() completed")

            except Exception as e:
                logger.error(f"Error in manual detection callback: {e}")
                # 如果出错，更新为错误状态
                if self.chinese_status_display:
                    self.chinese_status_display.update_status(StatusState.DETECTION_ERROR, error_msg=str(e))

        # 在后台线程中处理以避免UI冻结
        thread = threading.Thread(target=process_manual_detection, daemon=True)
        thread.start()

    def _simulate_manual_detection(self):
        """模拟手动检测功能（离线模式）"""
        print("SIMULATE_DETECTION_DEBUG: Starting simulated manual detection")

        # 详细检查状态
        print(f"SIMULATE_DETECTION_DEBUG: image_shape = {self.image_shape}")
        print(f"SIMULATE_DETECTION_DEBUG: hasattr('_last_roi1_data') = {hasattr(self, '_last_roi1_data')}")
        if hasattr(self, '_last_roi1_data'):
            print(f"SIMULATE_DETECTION_DEBUG: _last_roi1_data = {self._last_roi1_data[:50] if self._last_roi1_data else 'None'}...")
        print(f"SIMULATE_DETECTION_DEBUG: current_roi1_data = {type(self.current_roi1_data)}")
        print(f"SIMULATE_DETECTION_DEBUG: image_displayed = {getattr(self, 'image_displayed', 'Not set')}")

        try:
            import random

            # 首先更新状态为检测中
            if self.chinese_status_display:
                print("SIMULATE_DETECTION_DEBUG: Updating status to ENABLED_NO_DETECTION")
                self.chinese_status_display.update_status(StatusState.ENABLED_NO_DETECTION)

            # 模拟检测延迟
            time.sleep(0.5)

            # 清除现有的可视化
            if self.ax:
                print("SIMULATE_DETECTION_DEBUG: Clearing canvas")
                self.clear_canvas()

                # 重新显示ROI1图像
                if hasattr(self, '_last_roi1_data') and self._last_roi1_data:
                    print("SIMULATE_DETECTION_DEBUG: Updating ROI1 image with last data")
                    self.update_roi1_image(self._last_roi1_data)
                    print(f"SIMULATE_DETECTION_DEBUG: After update, image_shape = {self.image_shape}")
                else:
                    print("SIMULATE_DETECTION_DEBUG: No ROI1 data available")

            # 生成模拟检测结果
            if self.image_shape:
                # 生成随机的线条和交点
                lines = []
                intersections = []

                # 生成1-2条随机线条
                num_lines = random.randint(1, 2)
                for i in range(num_lines):
                    x1 = random.randint(50, self.image_shape[1] - 50)
                    y1 = random.randint(50, self.image_shape[0] - 50)
                    x2 = random.randint(50, self.image_shape[1] - 50)
                    y2 = random.randint(50, self.image_shape[0] - 50)
                    lines.append([(x1, y1), (x2, y2)])

                # 如果有两条线，计算交点
                if num_lines == 2 and len(lines) == 2:
                    try:
                        # 简单的交点计算（演示用）
                        intersection_x = random.randint(100, self.image_shape[1] - 100)
                        intersection_y = random.randint(100, self.image_shape[0] - 100)
                        confidence = random.uniform(0.6, 0.95)
                        intersections.append({
                            'x': intersection_x,
                            'y': intersection_y,
                            'confidence': confidence
                        })
                    except Exception:
                        pass

                # 更新可视化
                if self.ax and lines:
                    self.render_detected_lines(lines)

                if self.ax and intersections:
                    for intersection in intersections:
                        x = intersection['x']
                        y = intersection['y']
                        confidence = intersection['confidence']
                        self.render_intersection_point(
                            [(x, y)], confidence
                        )

                if self.canvas:
                    self.canvas.draw()

                # 更新状态显示
                if self.chinese_status_display:
                    if intersections:
                        intersection = intersections[0]
                        x, y = int(intersection['x']), int(intersection['y'])
                        confidence = intersection['confidence'] * 100
                        print(f"SIMULATE_DETECTION_DEBUG: Updating status to DETECTION_SUCCESS at ({x}, {y}) confidence {confidence:.1f}%")
                        self.chinese_status_display.update_status(
                            StatusState.DETECTION_SUCCESS,
                            intersection=(x, y),
                            confidence=confidence
                        )
                        logger.info(f"Simulated manual detection: intersection found at ({x}, {y}) with confidence {confidence:.1f}%")
                        print(f"SIMULATE_DETECTION_DEBUG: ✅ Simulated detection completed successfully!")
                    else:
                        print("SIMULATE_DETECTION_DEBUG: No intersections found, updating to ENABLED_NO_DETECTION")
                        self.chinese_status_display.update_status(StatusState.ENABLED_NO_DETECTION)
                        logger.info("Simulated manual detection: no intersections found")
                        print(f"SIMULATE_DETECTION_DEBUG: ✅ Simulated detection completed (no intersections)")
            else:
                # 没有图像数据
                print("SIMULATE_DETECTION_DEBUG: No image shape available")
                if self.chinese_status_display:
                    error_msg = "离线模式：无图像数据，请先获取ROI图像"
                    print(f"SIMULATE_DETECTION_DEBUG: Updating status with error_msg: {error_msg}")
                    self.chinese_status_display.update_status(
                        StatusState.DETECTION_ERROR,
                        error_msg=error_msg
                    )

        except Exception as e:
            logger.error(f"Error in simulated manual detection: {e}")
            if self.chinese_status_display:
                self.chinese_status_display.update_status(
                    StatusState.DETECTION_ERROR,
                    error_msg=f"模拟检测失败: {str(e)}"
                )

    def _on_refresh(self):
        """处理刷新回调"""
        def process_refresh():
            try:
                logger.info("Refresh triggered")

                # 如果启用了API集成，获取最新的ROI数据和检测状态
                if self.enable_api_integration and self.api_client:
                    try:
                        # 获取最新的ROI数据
                        roi_data = self.api_client.get_current_roi_data(dual_roi=True)

                        if roi_data and roi_data.get('type') == 'dual_realtime_data':
                            dual_roi_data = roi_data.get('dual_roi_data', {})
                            roi1_data = dual_roi_data.get('roi1_data', {})

                            # 更新ROI1图像显示
                            if roi1_data and 'pixels' in roi1_data:
                                self.parent_frame.after(0, lambda: self.update_roi1_image(roi1_data['pixels']))

                        # 获取检测状态
                        status_data = self.api_client.get_detection_status()
                        if status_data.get('enabled', False):
                            # 如果检测已启用，执行一次手动检测获取最新结果
                            detection_result = self.api_client.manual_detection(force_refresh=True)

                            if detection_result.get('success', False):
                                lines = detection_result.get('lines', [])
                                intersections = detection_result.get('intersections', [])

                                # 更新可视化
                                self.parent_frame.after(0, lambda: self.update_visualization({
                                    'lines': lines,
                                    'intersections': intersections
                                }))

                                # 更新状态显示
                                if intersections:
                                    best_intersection = max(intersections, key=lambda x: x.get('confidence', 0))
                                    point = best_intersection.get('point')
                                    confidence = best_intersection.get('confidence', 1.0)

                                    self.parent_frame.after(0, lambda: self.update_intersection_status(
                                        StatusState.DETECTION_SUCCESS,
                                        intersection=tuple(point) if point else None,
                                        confidence=confidence
                                    ))
                                else:
                                    self.parent_frame.after(0, lambda: self.update_intersection_status(
                                        StatusState.ENABLED_NO_DETECTION
                                    ))

                        logger.info("Refresh completed successfully")

                    except LineDetectionAPIError as e:
                        logger.error(f"API error during refresh: {e}")
                        self.parent_frame.after(0, lambda: self.update_intersection_status(
                            StatusState.DETECTION_ERROR, error_msg=str(e)
                        ))
                    except Exception as e:
                        logger.error(f"Unexpected error during refresh: {e}")
                        self.parent_frame.after(0, lambda: self.update_intersection_status(
                            StatusState.DETECTION_ERROR, error_msg=str(e)
                        ))
                else:
                    # 离线模式，仅更新时间戳
                    logger.warning("API integration not available, refresh in offline mode")
                    # 保持当前状态不变，只更新时间戳（由状态显示组件自动处理）

            except Exception as e:
                logger.error(f"Error in refresh callback: {e}")
                # 如果出错，更新为错误状态
                if self.chinese_status_display:
                    self.chinese_status_display.update_status(StatusState.DETECTION_ERROR, error_msg=str(e))

        # 在后台线程中处理以避免UI冻结
        thread = threading.Thread(target=process_refresh, daemon=True)
        thread.start()

    def _on_save_screenshot(self):
        """处理保存截图回调"""
        try:
            # 生成文件名（使用时间戳）
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"roi1_detection_{timestamp}.png"

            # 调用保存视图方法
            success = self.save_current_view(filename)

            if success:
                logger.info(f"Screenshot saved as: {filename}")
            else:
                logger.error("Failed to save screenshot")

        except Exception as e:
            logger.error(f"Error saving screenshot: {e}")

    def setup_mouse_events(self):
        """设置鼠标事件处理"""
        if self.canvas:
            # 鼠标移动事件
            self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

            # 鼠标按下事件
            self.canvas.mpl_connect('button_press_event', self.on_mouse_press)

            # 鼠标释放事件
            self.canvas.mpl_connect('button_release_event', self.on_mouse_release)

            # 鼠标滚轮缩放事件
            self.canvas.mpl_connect('scroll_event', self.on_mouse_scroll)

            # 轴范围变化事件 - 使用现代matplotlib API
            if hasattr(self, 'ax') and self.ax:
                self.ax.callbacks.connect('xlim_changed', self.on_xlim_changed)
                self.ax.callbacks.connect('ylim_changed', self.on_ylim_changed)
            else:
                # 如果轴不存在，使用画布级别的事件（虽然限制变化事件较少）
                print("WARNING: ax not available for limit change callbacks")

    def update_roi1_image(self, roi1_data: str):
        """
        更新ROI1图像显示

        Args:
            roi1_data: Base64编码的ROI1图像数据
        """
        try:
            if not roi1_data or not isinstance(roi1_data, str):
                logger.warning("Invalid ROI1 data received")
                return

            # 解析base64图像数据
            if roi1_data.startswith("data:image/"):
                # 提取base64部分
                header, encoded = roi1_data.split(",", 1)
                image_bytes = base64.b64decode(encoded)
            else:
                # 直接base64数据
                image_bytes = base64.b64decode(roi1_data)

            # 使用PIL打开图像
            image = Image.open(io.BytesIO(image_bytes))

            # 转换为numpy数组
            self.current_roi1_data = np.array(image)
            self.image_shape = self.current_roi1_data.shape

            # 保存最后接收的图像数据供离线模式使用
            self._last_roi1_data = roi1_data  # 保存原始base64数据
            print(f"IMAGE_DEBUG: Saved last ROI1 data, image_shape = {self.image_shape}")

            # 清除当前内容并显示新图像
            self.clear_canvas()

            # 显示图像
            if len(self.image_shape) == 3:
                # 彩色图像
                self.ax.imshow(self.current_roi1_data, extent=[0, self.image_shape[1],
                                                              self.image_shape[0], 0])
            else:
                # 灰度图像
                self.ax.imshow(self.current_roi1_data, cmap='gray',
                             extent=[0, self.image_shape[1], self.image_shape[0], 0])

            # 更新轴范围
            self.ax.set_xlim(0, self.image_shape[1])
            self.ax.set_ylim(self.image_shape[0], 0)

            # 更新状态栏
            self.info_label.config(text=f"Image: {self.image_shape[1]}x{self.image_shape[0]} pixels")

            # 重绘画布
            self.canvas.draw()

            self.image_displayed = True
            logger.info(f"ROI1 image updated: {self.image_shape}")

        except Exception as e:
            logger.error(f"Error updating ROI1 image: {e}")
            self.info_label.config(text=f"Error loading image: {str(e)}")

    def add_intersection_points(self, points: List[Tuple[float, float]],
                              color: str = 'lime', size: int = 8,
                              label: str = "Intersection"):
        """
        添加交点标记

        Args:
            points: 交点坐标列表 [(x1, y1), (x2, y2), ...]
            color: 标记颜色
            size: 标记大小
            label: 标记标签
        """
        try:
            if not points or not self.image_displayed:
                return

            # 分离x和y坐标
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]

            # 绘制交点
            self.ax.scatter(x_coords, y_coords, c=color, s=size**2,
                          marker='o', alpha=0.8, edgecolors='white',
                          linewidth=1, label=label, zorder=5)

            # 添加坐标标签（可选）
            if len(points) <= 10:  # 限制显示数量以避免混乱
                for i, (x, y) in enumerate(points):
                    self.ax.annotate(f"({int(x)}, {int(y)})",
                                   (x, y), xytext=(5, 5),
                                   textcoords='offset points',
                                   fontsize=8, color=color,
                                   bbox=dict(boxstyle='round,pad=0.3',
                                           facecolor='black', alpha=0.7),
                                   zorder=6)

            # 更新图例
            if label:
                self.ax.legend(loc='upper right', fontsize=8)

            # 重绘画布
            self.canvas.draw()

            logger.info(f"Added {len(points)} intersection points with color {color}")

        except Exception as e:
            logger.error(f"Error adding intersection points: {e}")

    def add_detection_lines(self, lines: List[Dict[str, Any]],
                           color: str = 'lime', linewidth: float = 2.0,
                           alpha: float = 0.8):
        """
        添加检测到的线条

        Args:
            lines: 线条列表，每个线条包含起点和终点
            color: 线条颜色
            linewidth: 线条宽度
            alpha: 透明度
        """
        try:
            if not lines or not self.image_displayed:
                return

            for line in lines:
                start_point = line.get('start', [0, 0])
                end_point = line.get('end', [0, 0])

                # 绘制线条
                self.ax.plot([start_point[0], end_point[0]],
                           [start_point[1], end_point[1]],
                           color=color, linewidth=linewidth,
                           alpha=alpha, zorder=4)

            # 重绘画布
            self.canvas.draw()

            logger.info(f"Added {len(lines)} detection lines")

        except Exception as e:
            logger.error(f"Error adding detection lines: {e}")

    def clear_canvas(self):
        """清除当前显示内容"""
        try:
            if self.ax:
                self.ax.clear()

                # 清除所有覆盖层元素
                self.overlay_elements = {
                    'detected_lines': [],
                    'intersection_circles': [],
                    'intersection_crosshairs': [],
                    'confidence_texts': [],
                    'line_patches': []
                }

                # 重新设置标题和标签
                self.ax.set_title("ROI1 - Green Line Intersection Detection",
                                 fontsize=12, fontweight='bold',
                                 color='white' if self.dark_theme else 'black')
                self.ax.set_xlabel("X Coordinate (pixels)",
                                 color='white' if self.dark_theme else 'black')
                self.ax.set_ylabel("Y Coordinate (pixels)",
                                 color='white' if self.dark_theme else 'black')

                # 重新应用主题
                if self.dark_theme:
                    self.ax.set_facecolor('#2d2d2d')
                    self.ax.spines['bottom'].set_color('white')
                    self.ax.spines['top'].set_color('white')
                    self.ax.spines['left'].set_color('white')
                    self.ax.spines['right'].set_color('white')
                    self.ax.tick_params(colors='white')
                else:
                    self.ax.set_facecolor('white')

                # 重新启用网格
                if self.grid_enabled:
                    self.ax.grid(True, alpha=0.3,
                               color='gray' if self.dark_theme else 'lightgray')

                # 重置图像显示状态
                self.image_displayed = False

        except Exception as e:
            logger.error(f"Error clearing canvas: {e}")

    def reset_view(self):
        """重置视图到默认状态"""
        try:
            if self.image_displayed and self.image_shape:
                # 重置到完整图像范围
                self.ax.set_xlim(0, self.image_shape[1])
                self.ax.set_ylim(self.image_shape[0], 0)
            else:
                # 重置到默认范围
                self.ax.set_xlim(0, 200)
                self.ax.set_ylim(200, 0)

            self.canvas.draw()
            logger.info("View reset to default")

        except Exception as e:
            logger.error(f"Error resetting view: {e}")

    def toggle_grid(self):
        """切换网格显示"""
        try:
            self.grid_enabled = not self.grid_enabled
            self.ax.grid(self.grid_enabled, alpha=0.3,
                       color='gray' if self.dark_theme else 'lightgray')
            self.canvas.draw()
            logger.info(f"Grid {'enabled' if self.grid_enabled else 'disabled'}")

        except Exception as e:
            logger.error(f"Error toggling grid: {e}")

    def toggle_crosshair(self):
        """切换十字线显示"""
        self.crosshair_enabled = not self.crosshair_enabled
        logger.info(f"Crosshair {'enabled' if self.crosshair_enabled else 'disabled'}")

    def get_mouse_coordinates(self, event) -> Optional[Tuple[float, float]]:
        """
        获取鼠标坐标

        Args:
            event: matplotlib鼠标事件

        Returns:
            坐标元组(x, y)或None
        """
        try:
            if event.inaxes != self.ax:
                return None

            # 获取数据坐标
            x, y = event.xdata, event.ydata

            if x is not None and y is not None:
                return (float(x), float(y))

            return None

        except Exception as e:
            logger.error(f"Error getting mouse coordinates: {e}")
            return None

    def on_mouse_move(self, event):
        """鼠标移动事件处理"""
        try:
            coords = self.get_mouse_coordinates(event)

            # 确保coord_label存在
            if not hasattr(self, 'coord_label') or self.coord_label is None:
                return  # 如果coord_label不存在，直接返回

            if coords:
                x, y = coords
                self.coord_label.config(text=f"X: {int(x)} , Y: {int(y)}")

                # 如果启用十字线，更新显示
                if self.crosshair_enabled and self.image_displayed:
                    # 这里可以添加十字线绘制逻辑
                    pass

                # 调用回调函数（如果设置）
                if self.coordinate_callback:
                    self.coordinate_callback(coords)
            else:
                self.coord_label.config(text="X: -- , Y: --")

        except Exception as e:
            logger.error(f"Error handling mouse move: {e}")

    def on_mouse_press(self, event):
        """鼠标按下事件处理"""
        try:
            if event.inaxes == self.ax:
                self.mouse_pressed = True
                coords = self.get_mouse_coordinates(event)
                if coords:
                    self.last_mouse_pos = coords

        except Exception as e:
            logger.error(f"Error handling mouse press: {e}")

    def on_mouse_release(self, event):
        """鼠标释放事件处理"""
        try:
            self.mouse_pressed = False
            self.last_mouse_pos = None

        except Exception as e:
            logger.error(f"Error handling mouse release: {e}")

    def on_mouse_scroll(self, event):
        """鼠标滚轮缩放事件处理"""
        try:
            if event.inaxes == self.ax:
                # 获取当前轴范围
                xlim = self.ax.get_xlim()
                ylim = self.ax.get_ylim()

                # 计算缩放因子
                scale_factor = 1.1 if event.button == 'up' else 0.9

                # 获取鼠标位置
                x_center = event.xdata if event.xdata else (xlim[0] + xlim[1]) / 2
                y_center = event.ydata if event.ydata else (ylim[0] + ylim[1]) / 2

                # 计算新范围
                x_range = (xlim[1] - xlim[0]) * scale_factor
                y_range = (ylim[1] - ylim[0]) * scale_factor

                new_xlim = [x_center - x_range / 2, x_center + x_range / 2]
                new_ylim = [y_center - y_range / 2, y_center + y_range / 2]

                # 设置新范围
                self.ax.set_xlim(new_xlim)
                self.ax.set_ylim(new_ylim)

                self.canvas.draw()

        except Exception as e:
            logger.error(f"Error handling mouse scroll: {e}")

    def on_xlim_changed(self, axes):
        """X轴范围变化事件"""
        try:
            xlim = axes.get_xlim()
            zoom_level = 200 / (xlim[1] - xlim[0]) * 100
            self.zoom_label.config(text=f"Zoom: {zoom_level:.0f}%")
        except:
            pass

    def on_ylim_changed(self, axes):
        """Y轴范围变化事件"""
        pass  # 已经由on_xlim_changed处理

    def set_coordinate_callback(self, callback):
        """设置坐标更新回调函数"""
        self.coordinate_callback = callback

    def save_current_view(self, filename: str, dpi: int = 150):
        """
        保存当前视图为图像文件

        Args:
            filename: 保存文件名
            dpi: 保存分辨率
        """
        try:
            if self.fig:
                self.fig.savefig(filename, dpi=dpi, bbox_inches='tight',
                               facecolor=self.fig.get_facecolor())
                logger.info(f"Current view saved to {filename}")
                return True
        except Exception as e:
            logger.error(f"Error saving view: {e}")
        return False

    def get_widget_info(self) -> Dict[str, Any]:
        """获取组件状态信息"""
        info = {
            'image_loaded': self.image_displayed,
            'image_shape': self.image_shape,
            'grid_enabled': self.grid_enabled,
            'crosshair_enabled': self.crosshair_enabled,
            'canvas_size': self.canvas.get_width_height() if self.canvas else None,
            'current_xlim': list(self.ax.get_xlim()) if self.ax else None,
            'current_ylim': list(self.ax.get_ylim()) if self.ax else None,
        }

        # 添加中文状态显示信息
        if self.chinese_status_display:
            info['chinese_status'] = {
                'enabled': True,
                'current_status': self.chinese_status_display.get_current_status(),
                'history_count': len(self.chinese_status_display.status_history),
                'status_colors': {
                    state.value: color for state, color in self.chinese_status_display.status_colors.items()
                }
            }
        else:
            info['chinese_status'] = {'enabled': False}

        # 添加控制面板信息
        if self.control_panel:
            info['control_panel'] = self.control_panel.get_control_state()
        else:
            info['control_panel'] = {'enabled': False}

        return info

    def update_config(self, new_config: Dict[str, Any]):
        """更新组件配置"""
        try:
            self.config.update(new_config)

            # 更新主题
            if 'dark_theme' in new_config:
                self.dark_theme = new_config['dark_theme']
                # 重新设置canvas以应用新主题
                if self.fig:
                    self.setup_canvas()

            # 更新其他设置
            if 'enable_toolbar' in new_config:
                self.enable_toolbar = new_config['enable_toolbar']

            logger.info("Configuration updated")

        except Exception as e:
            logger.error(f"Error updating config: {e}")

    # ============ Task 21: Line Overlay Rendering and Intersection Point Visualization ============

    def render_detected_lines(self, lines_data: List[Dict[str, Any]]):
        """
        渲染检测到的绿线覆盖层

        Args:
            lines_data: 线条数据列表，每个线条包含起点、终点和置信度信息
                      格式: [{'start': [x1, y1], 'end': [x2, y2], 'confidence': c}, ...]
        """
        try:
            if not lines_data or not self.image_displayed or not self.ax:
                logger.warning("Cannot render lines: no image displayed or no lines data")
                return

            # 清除之前的线条覆盖层
            self.clear_line_overlays()

            config = self.visualization_config

            for i, line_data in enumerate(lines_data):
                start_point = line_data.get('start', [0, 0])
                end_point = line_data.get('end', [0, 0])
                confidence = line_data.get('confidence', 1.0)

                # 验证坐标有效性
                if not self._validate_coordinates(start_point, end_point):
                    logger.warning(f"Invalid coordinates for line {i}: {start_point} -> {end_point}")
                    continue

                # 根据置信度调整颜色
                line_color = self._get_confidence_color(confidence)

                # 根据置信度调整线条透明度
                alpha = config['line_alpha'] * (0.5 + 0.5 * confidence)

                # 绘制线条
                line_plot = self.ax.plot([start_point[0], end_point[0]],
                                       [start_point[1], end_point[1]],
                                       color=line_color,
                                       linewidth=config['line_width'],
                                       alpha=alpha,
                                       zorder=config['z_order_lines'],
                                       label=f'Line {i+1}' if confidence > 0.5 else None)[0]

                # 存储线条引用
                self.overlay_elements['detected_lines'].append(line_plot)
                self.overlay_elements['line_patches'].append({
                    'line': line_plot,
                    'confidence': confidence,
                    'start': start_point,
                    'end': end_point
                })

            # 添加图例（仅高置信度线条）
            if any(line['confidence'] > 0.5 for line in self.overlay_elements['line_patches']):
                self.ax.legend(loc='upper right', fontsize=8, framealpha=0.8)

            # 重绘画布
            self.canvas.draw()

            logger.info(f"Rendered {len(lines_data)} detected lines")

        except Exception as e:
            logger.error(f"Error rendering detected lines: {e}")

    def render_intersection_point(self, intersection: Tuple[float, float], confidence: float):
        """
        绘制交点标记（双圆圈和十字线）

        Args:
            intersection: 交点坐标 (x, y)
            confidence: 交点置信度 (0.0 - 1.0)
        """
        try:
            if not intersection or not self.image_displayed or not self.ax:
                logger.warning("Cannot render intersection: no image displayed or invalid intersection")
                return

            x, y = intersection
            config = self.visualization_config

            # 验证坐标有效性
            if not self._validate_point_coordinates(x, y):
                logger.warning(f"Invalid intersection coordinates: {intersection}")
                return

            # 根据置信度选择颜色
            base_color = self._get_confidence_color(confidence)

            # 绘制外圆（红色，仅边框）
            outer_circle = patches.Circle((x, y),
                                        radius=config['intersection_outer_radius'],
                                        fill=False,
                                        edgecolor='red',
                                        linewidth=2,
                                        zorder=config['z_order_intersections'])
            self.ax.add_patch(outer_circle)

            # 绘制内圆（橙色，填充）
            inner_circle = patches.Circle((x, y),
                                        radius=config['intersection_inner_radius'],
                                        fill=True,
                                        facecolor='orange',
                                        edgecolor=base_color,
                                        linewidth=1,
                                        alpha=config['text_alpha'],
                                        zorder=config['z_order_intersections'] + 1)
            self.ax.add_patch(inner_circle)

            # 绘制十字线（延伸8像素）
            crosshair_length = config['crosshair_length']
            crosshair_color = base_color

            # 水平十字线
            h_crosshair = self.ax.plot([x - crosshair_length, x + crosshair_length], [y, y],
                                     color=crosshair_color, linewidth=1, alpha=0.8,
                                     zorder=config['z_order_intersections'])[0]

            # 垂直十字线
            v_crosshair = self.ax.plot([x, x], [y - crosshair_length, y + crosshair_length],
                                     color=crosshair_color, linewidth=1, alpha=0.8,
                                     zorder=config['z_order_intersections'])[0]

            # 存储元素引用
            self.overlay_elements['intersection_circles'].extend([outer_circle, inner_circle])
            self.overlay_elements['intersection_crosshairs'].extend([h_crosshair, v_crosshair])

            # 添加坐标和置信度文本
            self.add_confidence_text(x, y, confidence)

            # 重绘画布
            self.canvas.draw()

            logger.info(f"Rendered intersection point at ({x:.1f}, {y:.1f}) with confidence {confidence:.2f}")

        except Exception as e:
            logger.error(f"Error rendering intersection point: {e}")

    def add_confidence_text(self, x: float, y: float, confidence: float):
        """
        添加坐标和置信度文本显示

        Args:
            x: X坐标
            y: Y坐标
            confidence: 置信度值 (0.0 - 1.0)
        """
        try:
            if not self.ax:
                return

            config = self.visualization_config

            # 格式化文本内容
            coord_text = f"({int(x)}, {int(y)})"
            conf_text = f"c:{confidence:.2f}"
            full_text = f"{coord_text}\n{conf_text}"

            # 根据置信度选择文本颜色
            text_color = self._get_confidence_color(confidence)

            # 创建半透明黑色背景框
            bbox_props = dict(boxstyle='round,pad=0.5',
                            facecolor='black',
                            alpha=config['text_alpha'] * 0.7,
                            edgecolor=text_color,
                            linewidth=1)

            # 添加文本（位置稍微偏移以避免遮挡交点）
            text_offset = 15  # 像素偏移
            text_obj = self.ax.text(x + text_offset, y - text_offset, full_text,
                                  fontsize=8,
                                  color=text_color,
                                  bbox=bbox_props,
                                  verticalalignment='top',
                                  horizontalalignment='left',
                                  zorder=config['z_order_text'])

            # 存储文本引用
            self.overlay_elements['confidence_texts'].append(text_obj)

        except Exception as e:
            logger.error(f"Error adding confidence text: {e}")

    def clear_overlays(self):
        """清除所有覆盖层元素"""
        try:
            if not self.ax:
                return

            # 清除检测线条
            for line_plot in self.overlay_elements['detected_lines']:
                if line_plot in self.ax.lines:
                    line_plot.remove()

            # 清除交点圆圈
            for circle in self.overlay_elements['intersection_circles']:
                if circle in self.ax.patches:
                    circle.remove()

            # 清除十字线
            for crosshair in self.overlay_elements['intersection_crosshairs']:
                if crosshair in self.ax.lines:
                    crosshair.remove()

            # 清除文本
            for text_obj in self.overlay_elements['confidence_texts']:
                if text_obj in self.ax.texts:
                    text_obj.remove()

            # 清空存储
            self.overlay_elements = {
                'detected_lines': [],
                'intersection_circles': [],
                'intersection_crosshairs': [],
                'confidence_texts': [],
                'line_patches': []
            }

            # 清除图例
            if self.ax.get_legend():
                self.ax.get_legend().remove()

            # 重绘画布
            self.canvas.draw()

            logger.info("All overlay elements cleared")

        except Exception as e:
            logger.error(f"Error clearing overlays: {e}")

    def clear_line_overlays(self):
        """仅清除线条覆盖层"""
        try:
            if not self.ax:
                return

            # 清除检测线条
            for line_plot in self.overlay_elements['detected_lines']:
                if line_plot in self.ax.lines:
                    line_plot.remove()

            # 清空相关存储
            self.overlay_elements['detected_lines'] = []
            self.overlay_elements['line_patches'] = []

            logger.info("Line overlays cleared")

        except Exception as e:
            logger.error(f"Error clearing line overlays: {e}")

    def clear_intersection_overlays(self):
        """仅清除交点覆盖层"""
        try:
            if not self.ax:
                return

            # 清除交点圆圈
            for circle in self.overlay_elements['intersection_circles']:
                if circle in self.ax.patches:
                    circle.remove()

            # 清除十字线
            for crosshair in self.overlay_elements['intersection_crosshairs']:
                if crosshair in self.ax.lines:
                    crosshair.remove()

            # 清除文本
            for text_obj in self.overlay_elements['confidence_texts']:
                if text_obj in self.ax.texts:
                    text_obj.remove()

            # 清空相关存储
            self.overlay_elements['intersection_circles'] = []
            self.overlay_elements['intersection_crosshairs'] = []
            self.overlay_elements['confidence_texts'] = []

            logger.info("Intersection overlays cleared")

        except Exception as e:
            logger.error(f"Error clearing intersection overlays: {e}")

    def update_visualization(self, detection_result: Dict[str, Any]):
        """
        更新所有可视化元素

        Args:
            detection_result: 检测结果字典
                格式: {
                    'lines': [{'start': [x1, y1], 'end': [x2, y2], 'confidence': c}, ...],
                    'intersections': [{'point': [x, y], 'confidence': c}, ...]
                }
        """
        try:
            if not detection_result:
                logger.warning("No detection result to visualize")
                # 更新状态为无检测结果
                if self.chinese_status_display:
                    self.chinese_status_display.update_status(StatusState.ENABLED_NO_DETECTION)
                return

            # 清除之前的覆盖层
            self.clear_overlays()

            # 处理线条数据
            lines_data = detection_result.get('lines', [])
            if lines_data:
                self.render_detected_lines(lines_data)

            # 处理交点数据并更新状态
            intersections = detection_result.get('intersections', [])
            if intersections:
                # 选择最高置信度的交点用于状态显示
                best_intersection = max(intersections, key=lambda x: x.get('confidence', 0))
                point = best_intersection.get('point')
                confidence = best_intersection.get('confidence', 1.0)

                if point:
                    # 渲染所有交点
                    for intersection_data in intersections:
                        pt = intersection_data.get('point')
                        conf = intersection_data.get('confidence', 1.0)
                        if pt:
                            self.render_intersection_point(pt, conf)

                    # 更新中文状态显示为成功
                    if self.chinese_status_display:
                        self.chinese_status_display.update_status(
                            StatusState.DETECTION_SUCCESS,
                            intersection=tuple(point),
                            confidence=confidence
                        )
                else:
                    # 交点数据无效
                    if self.chinese_status_display:
                        self.chinese_status_display.update_status(
                            StatusState.DETECTION_ERROR,
                            error_msg="交点坐标无效"
                        )
            else:
                # 没有检测到交点
                if self.chinese_status_display:
                    self.chinese_status_display.update_status(StatusState.ENABLED_NO_DETECTION)

            logger.info(f"Visualization updated: {len(lines_data)} lines, {len(intersections)} intersections")

        except Exception as e:
            logger.error(f"Error updating visualization: {e}")
            # 更新状态为错误
            if self.chinese_status_display:
                self.chinese_status_display.update_status(StatusState.DETECTION_ERROR, error_msg=str(e))

    def update_multiple_intersections(self, intersections: List[Dict[str, Any]], max_display: int = 5):
        """
        更新多个交点显示，支持优先级处理

        Args:
            intersections: 交点列表，格式: [{'point': [x, y], 'confidence': c}, ...]
            max_display: 最大显示数量
        """
        try:
            if not intersections:
                self.clear_intersection_overlays()
                return

            # 清除之前的交点覆盖层
            self.clear_intersection_overlays()

            # 按置信度排序，优先显示高置信度交点
            sorted_intersections = sorted(intersections,
                                        key=lambda x: x.get('confidence', 0),
                                        reverse=True)

            # 限制显示数量
            display_intersections = sorted_intersections[:max_display]

            # 渲染每个交点
            for i, intersection_data in enumerate(display_intersections):
                point = intersection_data.get('point')
                confidence = intersection_data.get('confidence', 1.0)

                if point:
                    # 为排名较低的交点调整透明度
                    alpha_scale = 1.0 - (i * 0.15)  # 每个后续交点降低15%透明度
                    original_alpha = self.visualization_config['text_alpha']
                    self.visualization_config['text_alpha'] = original_alpha * alpha_scale

                    self.render_intersection_point(point, confidence)

                    # 恢复原始透明度
                    self.visualization_config['text_alpha'] = original_alpha

            logger.info(f"Displayed {len(display_intersections)} intersections (max: {max_display})")

        except Exception as e:
            logger.error(f"Error updating multiple intersections: {e}")

    def _validate_coordinates(self, start: List[float], end: List[float]) -> bool:
        """验证线条坐标的有效性"""
        try:
            if (len(start) >= 2 and len(end) >= 2 and
                all(isinstance(coord, (int, float)) for coord in start + end) and
                start[0] >= 0 and start[1] >= 0 and end[0] >= 0 and end[1] >= 0):

                # 如果有图像尺寸，进一步验证
                if self.image_shape:
                    return (start[0] < self.image_shape[1] and start[1] < self.image_shape[0] and
                           end[0] < self.image_shape[1] and end[1] < self.image_shape[0])
                return True
        except:
            pass
        return False

    def _validate_point_coordinates(self, x: float, y: float) -> bool:
        """验证点坐标的有效性"""
        try:
            if isinstance(x, (int, float)) and isinstance(y, (int, float)) and x >= 0 and y >= 0:
                # 如果有图像尺寸，进一步验证
                if self.image_shape:
                    return x < self.image_shape[1] and y < self.image_shape[0]
                return True
        except:
            pass
        return False

    def _get_confidence_color(self, confidence: float) -> str:
        """
        根据置信度返回对应颜色

        Args:
            confidence: 置信度值 (0.0 - 1.0)

        Returns:
            颜色字符串
        """
        config = self.visualization_config

        if confidence >= config['high_confidence_threshold']:
            return 'red'  # 高置信度 - 红色
        elif confidence >= config['medium_confidence_threshold']:
            return 'orange'  # 中等置信度 - 橙色
        else:
            return 'yellow'  # 低置信度 - 黄色

    def get_visualization_info(self) -> Dict[str, Any]:
        """获取当前可视化状态信息"""
        return {
            'overlay_count': {
                'lines': len(self.overlay_elements['detected_lines']),
                'intersections': len(self.overlay_elements['intersection_circles']) // 2,  # 每个交点2个圆
                'crosshairs': len(self.overlay_elements['intersection_crosshairs']) // 2,  # 每个交点2条线
                'texts': len(self.overlay_elements['confidence_texts'])
            },
            'visualization_config': self.visualization_config,
            'image_displayed': self.image_displayed
        }

    def set_visualization_config(self, new_config: Dict[str, Any]):
        """更新可视化配置"""
        try:
            # 验证配置参数
            valid_keys = self.visualization_config.keys()
            for key, value in new_config.items():
                if key in valid_keys:
                    self.visualization_config[key] = value
                else:
                    logger.warning(f"Invalid visualization config key: {key}")

            logger.info("Visualization configuration updated")

        except Exception as e:
            logger.error(f"Error updating visualization config: {e}")

    # ============ Task 23: Chinese Status Display Interface Methods ============

    def get_chinese_status_display(self) -> Optional[ChineseStatusDisplay]:
        """
        获取中文状态显示组件实例

        Returns:
            ChineseStatusDisplay实例或None
        """
        return self.chinese_status_display

    def update_intersection_status(self,
                                  state: StatusState,
                                  intersection: Optional[Tuple[float, float]] = None,
                                  confidence: Optional[float] = None,
                                  error_msg: Optional[str] = None,
                                  animate: bool = True):
        """
        更新线条相交点检测状态

        Args:
            state: 状态枚举值
            intersection: 交点坐标 (x, y)
            confidence: 置信度 (0.0 - 1.0)
            error_msg: 错误消息
            animate: 是否使用动画效果
        """
        try:
            if not self.chinese_status_display:
                logger.warning("Chinese status display not available")
                return

            if animate:
                self.chinese_status_display.animate_status_change(
                    state,
                    intersection=intersection,
                    confidence=confidence,
                    error_msg=error_msg
                )
            else:
                self.chinese_status_display.update_status(
                    state,
                    intersection=intersection,
                    confidence=confidence,
                    error_msg=error_msg
                )

            logger.info(f"Intersection status updated: {state.value}")

        except Exception as e:
            logger.error(f"Error updating intersection status: {e}")

    def update_detection_success(self, intersection: Tuple[float, float], confidence: float):
        """
        更新检测成功状态

        Args:
            intersection: 交点坐标 (x, y)
            confidence: 置信度 (0.0 - 1.0)
        """
        self.update_intersection_status(
            StatusState.DETECTION_SUCCESS,
            intersection=intersection,
            confidence=confidence
        )

    def update_detection_error(self, error_message: str):
        """
        更新检测错误状态

        Args:
            error_message: 错误消息
        """
        self.update_intersection_status(
            StatusState.DETECTION_ERROR,
            error_msg=error_message
        )

    def set_detection_enabled_status(self, enabled: bool):
        """
        设置检测启用状态

        Args:
            enabled: 是否启用检测
        """
        state = StatusState.ENABLED_NO_DETECTION if enabled else StatusState.DISABLED
        self.update_intersection_status(state)

    def get_intersection_status(self) -> Dict[str, Any]:
        """
        获取当前线条相交点检测状态

        Returns:
            状态信息字典
        """
        if self.chinese_status_display:
            return self.chinese_status_display.get_current_status()
        return {
            'state': StatusState.DISABLED,
            'state_text': 'Status display not available',
            'intersection': None,
            'confidence': None,
            'error_msg': None,
            'last_update_time': None,
            'color': '#808080'
        }

    def get_status_history(self) -> List[Dict[str, Any]]:
        """
        获取状态变化历史

        Returns:
            状态历史列表
        """
        if self.chinese_status_display:
            return self.chinese_status_display.get_status_history()
        return []

    def clear_status_history(self):
        """清除状态变化历史"""
        if self.chinese_status_display:
            self.chinese_status_display.clear_status_history()

    def show_status_history_dialog(self):
        """显示状态历史对话框"""
        if self.chinese_status_display:
            self.chinese_status_display.show_status_history()

    def set_status_colors(self, colors: Dict[str, str]):
        """
        自定义状态颜色

        Args:
            colors: 颜色字典，键为状态名称，值为颜色代码
        """
        try:
            if not self.chinese_status_display:
                return

            # 转换字符串键到状态枚举
            status_colors = {}
            for state_name, color in colors.items():
                try:
                    state = StatusState(state_name)
                    status_colors[state] = color
                except ValueError:
                    logger.warning(f"Invalid state name: {state_name}")

            if status_colors:
                self.chinese_status_display.set_status_colors(status_colors)

        except Exception as e:
            logger.error(f"Error setting status colors: {e}")

    def set_status_font_config(self, font_family: str = None, font_size: int = None):
        """
        设置状态显示字体配置

        Args:
            font_family: 字体族名
            font_size: 字体大小
        """
        if self.chinese_status_display:
            self.chinese_status_display.set_font_config(font_family, font_size)

    def reset_status_to_disabled(self):
        """重置状态为未启用"""
        self.update_intersection_status(StatusState.DISABLED)

    # ============ Control Panel Integration Methods ============

    def get_control_panel(self) -> Optional[LineDetectionControls]:
        """
        获取控制面板实例

        Returns:
            LineDetectionControls实例或None
        """
        return self.control_panel

    def set_detection_enabled(self, enabled: bool):
        """
        设置检测启用状态

        Args:
            enabled: 是否启用检测
        """
        if self.control_panel:
            self.control_panel.set_detection_enabled(enabled)

    def get_detection_state(self) -> Dict[str, Any]:
        """
        获取检测状态信息

        Returns:
            检测状态字典
        """
        if self.control_panel:
            return self.control_panel.get_control_state()
        return {'detection_enabled': False, 'loading_states': {}, 'detection_count': 0}

    def set_control_callbacks(self, callbacks: Dict[str, Callable]):
        """
        设置控制面板回调函数

        Args:
            callbacks: 回调函数字典
        """
        if self.control_panel:
            self.control_panel.set_callbacks(callbacks)

    def enable_control_buttons(self, enabled: bool = True):
        """
        启用或禁用所有控制按钮

        Args:
            enabled: 是否启用按钮
        """
        if self.control_panel:
            self.control_panel.enable_all_buttons(enabled)

    def set_control_loading_state(self, button_name: str, loading: bool):
        """
        设置控制按钮的加载状态

        Args:
            button_name: 按钮名称 ('toggle', 'manual', 'refresh')
            loading: 是否处于加载状态
        """
        if self.control_panel:
            self.control_panel.set_loading_state(button_name, loading)

    def update_detection_status(self, status_text: str, is_processing: bool = False):
        """
        更新检测状态显示（用于外部更新）

        Args:
            status_text: 状态文本
            is_processing: 是否正在处理
        """
        if self.control_panel and self.control_panel.status_label:
            # 根据处理状态选择颜色
            if is_processing:
                color = 'orange'
            elif "启用" in status_text or "enabled" in status_text.lower():
                color = 'green'
            elif "禁用" in status_text or "disabled" in status_text.lower():
                color = 'red'
            else:
                color = 'gray'

            self.control_panel.status_label.config(
                text=f"检测状态: {status_text}",
                foreground=color
            )

    # ============ Task 24: API Client Integration Methods ============

    def update_line_intersection_data(self, line_intersection_result):
        """
        更新交点检测数据（包含ROI数据）

        Args:
            line_intersection_result: 包含交点检测结果的字典，可能包含ROI数据
        """
        try:
            print(f"LINE_WIDGET_DEBUG: Received line intersection result: {type(line_intersection_result)}")

            # 检查结果类型和内容
            if isinstance(line_intersection_result, dict):
                print(f"LINE_WIDGET_DEBUG: Result keys: {list(line_intersection_result.keys())}")

                # 检查是否有ROI数据
                roi1_data = None
                if 'roi1_data' in line_intersection_result:
                    roi1_data = line_intersection_result['roi1_data']
                    print(f"LINE_WIDGET_DEBUG: Found roi1_data: {type(roi1_data)}")
                    print(f"LINE_WIDGET_DEBUG: roi1_data keys: {list(roi1_data.keys()) if roi1_data else 'None'}")

                elif 'dual_roi_data' in line_intersection_result:
                    dual_roi_data = line_intersection_result['dual_roi_data']
                    if dual_roi_data and 'roi1_data' in dual_roi_data:
                        roi1_data = dual_roi1_data['roi1_data']
                        print(f"LINE_WIDGET_DEBUG: Found roi1_data in dual_roi_data: {type(roi1_data)}")

                # 更新ROI1图像
                if roi1_data and 'pixels' in roi1_data:
                    print("LINE_WIDGET_DEBUG: Updating ROI1 image with new data")
                    self.update_roi1_image(roi1_data['pixels'])
                else:
                    print("LINE_WIDGET_DEBUG: No ROI1 pixels data found in result")

                # 更新可视化（如果有交点）
                if 'intersection' in line_intersection_result:
                    intersection = line_intersection_result['intersection']
                    if intersection and len(intersection) >= 2:
                        print(f"LINE_WIDGET_DEBUG: Updating intersection point: {intersection}")
                        x, y = intersection[0], intersection[1]
                        confidence = line_intersection_result.get('confidence', 0.0)

                        # 渲染交点
                        if hasattr(self, 'render_intersection_point'):
                            self.render_intersection_point(
                                [(x, y)], confidence
                            )
                        if self.canvas:
                            self.canvas.draw()

                        # 更新状态显示
                        if self.chinese_status_display:
                            self.chinese_status_display.update_status(
                                StatusState.DETETECTION_SUCCESS,
                                intersection=(int(x), int(y)),
                                confidence=confidence * 100
                            )

                # 更新线条
                if 'lines' in line_intersection_result:
                    lines = line_intersection_result['lines']
                    print(f"LINE_WIDGET_DEBUG: Updating {len(lines)} lines")

                    if lines and hasattr(self, 'render_detected_lines'):
                        self.render_detected_lines(lines)
                        if self.canvas:
                            self.canvas.draw()

                print("LINE_WIDGET_DEBUG: Line intersection data updated successfully")

            else:
                print(f"LINE_WIDGET_DEBUG: Invalid line_intersection_result type: {type(line_intersection_result)}")

        except Exception as e:
            print(f"LINE_WIDGET_DEBUG: Error updating line intersection data: {e}")
            if self.chinese_status_display:
                self.chinese_status_display.update_status(
                    StatusState.DETECTION_ERROR,
                    error_msg=f"更新交点数据失败: {str(e)}"
                )
            logger.error(f"Error updating line intersection data: {e}")

    def initialize_api_client(self):
        """
        初始化API客户端

        根据配置创建和配置线条检测API客户端
        """
        print("API_CLIENT_DEBUG: initialize_api_client() called")
        print(f"API_CLIENT_DEBUG: self.enable_api_integration = {self.enable_api_integration}")
        print(f"API_CLIENT_DEBUG: LINE_DETECTION_API_AVAILABLE = {LINE_DETECTION_API_AVAILABLE}")
        print(f"API_CLIENT_DEBUG: LineDetectionAPIClient class = {LineDetectionAPIClient}")

        try:
            if not self.enable_api_integration:
                print("API_CLIENT_DEBUG: API integration disabled, returning")
                logger.info("API integration disabled")
                return

            if not LINE_DETECTION_API_AVAILABLE:
                print("API_CLIENT_DEBUG: LineDetectionAPIClient not available, setting api_client = None")
                logger.error("LineDetectionAPIClient not available")
                self.api_client = None
                return

            if LineDetectionAPIClient is None:
                print("API_CLIENT_DEBUG: LineDetectionAPIClient class is None, cannot create instance")
                logger.error("LineDetectionAPIClient class is None")
                self.api_client = None
                return

            print("API_CLIENT_DEBUG: All checks passed, initializing line detection API client")
            logger.info("Initializing line detection API client")

            # 创建API客户端
            print("API_CLIENT_DEBUG: Creating LineDetectionAPIClient instance")
            print(f"API_CLIENT_DEBUG: base_url = {self.api_base_url}")
            print(f"API_CLIENT_DEBUG: timeout = {self.api_timeout}")

            if LineDetectionAPIClient:
                try:
                    print("API_CLIENT_DEBUG: Instantiating LineDetectionAPIClient...")
                    self.api_client = LineDetectionAPIClient(
                        base_url=self.api_base_url,
                        password=self.api_password,
                        timeout=self.api_timeout
                    )
                    print(f"API_CLIENT_DEBUG: LineDetectionAPIClient instance created: {self.api_client}")
                except Exception as e:
                    print(f"API_CLIENT_DEBUG: Exception creating LineDetectionAPIClient: {e}")
                    logger.error(f"Exception creating LineDetectionAPIClient: {e}")
                    self.api_client = None
                    return
            else:
                print("API_CLIENT_DEBUG: LineDetectionAPIClient class is None, cannot create instance")
                logger.error("LineDetectionAPIClient class not available")
                self.api_client = None
                return

            # 测试连接
            try:
                print(f"API_CLIENT_DEBUG: Attempting health check to {self.api_base_url}")
                health_result = self.api_client.health_check()
                print(f"API_CLIENT_DEBUG: Health check result: {health_result}")

                # 接受'ok'和'healthy'作为有效状态
                status = health_result.get('status', 'unknown')
                if status in ['ok', 'healthy']:
                    print(f"API_CLIENT_DEBUG: API client connection successful (status: {status})")
                    logger.info(f"API client connection successful (status: {status})")

                    # 获取当前检测状态并同步到UI
                    self._sync_detection_status_from_api()
                else:
                    print(f"API_CLIENT_DEBUG: API health check returned invalid status: {status}")
                    logger.warning(f"API health check returned: {status}")
                    self.api_client = None  # 禁用API客户端
            except Exception as e:
                print(f"API_CLIENT_DEBUG: Health check exception: {e}")
                logger.error(f"API client health check failed: {e}")
                self.api_client = None  # 禁用API客户端

        except Exception as e:
            logger.error(f"Error initializing API client: {e}")
            self.api_client = None

    def _sync_detection_status_from_api(self):
        """
        从API同步检测状态到UI
        """
        try:
            if not self.api_client:
                return

            # 获取当前检测状态
            status_data = self.api_client.get_detection_status()
            enabled = status_data.get('enabled', False)

            # 更新控制面板状态
            if self.control_panel:
                self.control_panel.set_detection_enabled(enabled)

            # 更新中文状态显示
            if self.chinese_status_display:
                if enabled:
                    self.chinese_status_display.update_status(StatusState.ENABLED_NO_DETECTION)
                else:
                    self.chinese_status_display.update_status(StatusState.DISABLED)

            logger.info(f"Detection status synced from API: {'enabled' if enabled else 'disabled'}")

        except Exception as e:
            logger.error(f"Error syncing detection status: {e}")

    def get_api_client(self) -> Optional[LineDetectionAPIClient]:
        """
        获取API客户端实例

        Returns:
            LineDetectionAPIClient实例或None
        """
        return self.api_client

    def is_api_integration_available(self) -> bool:
        """
        检查API集成是否可用

        Returns:
            API集成是否可用
        """
        return self.enable_api_integration and self.api_client is not None

    def get_enhanced_realtime_data_with_line_detection(self, count: int = 100) -> Optional[Dict[str, Any]]:
        """
        获取包含线条检测数据的增强实时数据

        Args:
            count: 数据点数量

        Returns:
            增强数据字典或None
        """
        try:
            if not self.is_api_integration_available():
                logger.warning("API integration not available for enhanced data")
                return None

            return self.api_client.get_enhanced_realtime_data(
                count=count,
                include_line_intersection=True
            )

        except Exception as e:
            logger.error(f"Error getting enhanced realtime data: {e}")
            return None

    def update_line_detection_config(self, config_updates: Dict[str, Any]) -> bool:
        """
        更新线条检测配置

        Args:
            config_updates: 配置更新字典

        Returns:
            更新是否成功
        """
        try:
            if not self.is_api_integration_available():
                logger.warning("API integration not available for config update")
                return False

            result = self.api_client.update_line_detection_config(config_updates)
            return result.get('success', False)

        except Exception as e:
            logger.error(f"Error updating line detection config: {e}")
            return False

    def get_line_detection_config(self) -> Optional[Dict[str, Any]]:
        """
        获取当前线条检测配置

        Returns:
            配置字典或None
        """
        try:
            if not self.is_api_integration_available():
                logger.warning("API integration not available for config retrieval")
                return None

            return self.api_client.get_line_detection_config()

        except Exception as e:
            logger.error(f"Error getting line detection config: {e}")
            return None

    def set_api_config(self, base_url: str = None, password: str = None, timeout: int = None):
        """
        更新API配置

        Args:
            base_url: API基础URL
            password: 认证密码
            timeout: 请求超时时间
        """
        try:
            # 更新配置
            if base_url is not None:
                self.api_base_url = base_url
            if password is not None:
                self.api_password = password
            if timeout is not None:
                self.api_timeout = timeout

            # 重新初始化API客户端
            if self.enable_api_integration:
                # 关闭现有客户端
                if self.api_client:
                    self.api_client.close()

                # 创建新客户端
                self.initialize_api_client()

                logger.info("API configuration updated and client reinitialized")

        except Exception as e:
            logger.error(f"Error updating API config: {e}")

    def get_api_statistics(self) -> Dict[str, Any]:
        """
        获取API客户端统计信息

        Returns:
            统计信息字典
        """
        try:
            if self.api_client:
                return self.api_client.get_statistics()
            else:
                return {'error': 'API client not initialized'}

        except Exception as e:
            logger.error(f"Error getting API statistics: {e}")
            return {'error': str(e)}

    def reset_api_statistics(self):
        """重置API客户端统计信息"""
        try:
            if self.api_client:
                self.api_client.reset_statistics()
                logger.info("API statistics reset")

        except Exception as e:
            logger.error(f"Error resetting API statistics: {e}")

    def cleanup_api_client(self):
        """清理API客户端资源"""
        try:
            if self.api_client:
                self.api_client.close()
                self.api_client = None
                logger.info("API client cleaned up")

        except Exception as e:
            logger.error(f"Error cleaning up API client: {e}")

    def get_widget_api_status(self) -> Dict[str, Any]:
        """
        获取组件的API集成状态信息

        Returns:
            状态信息字典
        """
        status = {
            'api_integration_enabled': self.enable_api_integration,
            'api_client_available': self.api_client is not None,
            'api_base_url': self.api_base_url,
            'api_timeout': self.api_timeout,
            'api_configured': bool(self.api_base_url and self.api_password)
        }

        if self.api_client:
            stats = self.get_api_statistics()
            status.update({
                'api_statistics': stats,
                'connection_healthy': stats.get('error_count', 0) == 0,
                'total_requests': stats.get('request_count', 0),
                'error_rate': stats.get('success_rate', 1.0)
            })

        return status

    def cleanup(self):
        """
        清理组件资源

        清理matplotlib组件、API客户端和其他资源
        """
        try:
            logger.info("Cleaning up LineDetectionWidget resources")

            # 清理API客户端
            self.cleanup_api_client()

            # 清理matplotlib组件
            if hasattr(self, 'canvas') and self.canvas:
                try:
                    self.canvas.get_tk_widget().destroy()
                except:
                    pass

            if hasattr(self, 'fig') and self.fig:
                try:
                    import matplotlib.pyplot as plt
                    plt.close(self.fig)
                except:
                    pass

            # 清理其他资源
            self.current_roi1_data = None
            self.overlay_elements = {
                'detected_lines': [],
                'intersection_circles': [],
                'intersection_crosshairs': [],
                'confidence_texts': [],
                'line_patches': []
            }

            logger.info("LineDetectionWidget cleanup completed")

        except Exception as e:
            logger.error(f"Error during widget cleanup: {e}")


# ============ Task 32: Client-Side Error Handling and User Feedback Mechanisms ============

class ErrorSeverity(Enum):
    """错误严重程度枚举"""
    INFO = "info"          # 信息提示
    WARNING = "warning"    # 警告
    ERROR = "error"        # 错误
    CRITICAL = "critical"  # 严重错误


class ErrorCategory(Enum):
    """错误类别枚举"""
    NETWORK = "network"        # 网络连接错误
    API = "api"               # API调用错误
    AUTHENTICATION = "auth"   # 认证错误
    DATA_PARSING = "parsing"  # 数据解析错误
    CONFIGURATION = "config"  # 配置错误
    PROCESSING = "processing" # 处理错误
    MEMORY = "memory"         # 内存错误
    TIMEOUT = "timeout"       # 超时错误
    USER_INPUT = "input"      # 用户输入错误
    UNKNOWN = "unknown"       # 未知错误


class ClientErrorHandler:
    """
    客户端错误处理器 - 提供用户友好的错误处理和反馈机制

    功能包括：
    - 错误分类和严重程度评估
    - 用户友好的错误消息翻译
    - 错误恢复建议和操作指导
    - 网络连接状态监控
    - 错误历史记录和统计分析
    - 错误报告和反馈收集
    """

    def __init__(self, parent_widget, config: Optional[Dict[str, Any]] = None):
        """
        初始化客户端错误处理器

        Args:
            parent_widget: 父级Tkinter组件
            config: 配置字典
        """
        self.parent_widget = parent_widget
        self.config = config or {}

        # 错误历史记录
        self.error_history = []
        self.max_history_size = self.config.get('max_error_history', 100)
        self.error_statistics = {
            'total_errors': 0,
            'by_category': {},
            'by_severity': {},
            'by_hour': {},
            'recent_errors': []
        }

        # 网络连接监控
        self.network_status = {
            'connected': False,
            'last_check': None,
            'response_time': None,
            'quality': 'unknown'
        }

        # 错误消息翻译字典
        self.error_translations = {
            # 网络错误
            'ConnectionError': {
                'zh': '网络连接失败，请检查网络设置和服务器状态',
                'en': 'Network connection failed, please check network settings and server status',
                'actions': ['检查网络连接', '验证服务器地址', '尝试重新连接']
            },
            'TimeoutError': {
                'zh': '请求超时，服务器响应时间过长',
                'en': 'Request timeout, server response time too long',
                'actions': ['检查网络速度', '稍后重试', '联系技术支持']
            },
            'HTTPError': {
                'zh': 'HTTP错误，服务器返回异常状态码',
                'en': 'HTTP error, server returned abnormal status code',
                'actions': ['检查服务器状态', '验证请求参数', '联系管理员']
            },

            # API错误
            'LineDetectionAPIError': {
                'zh': '线条检测API错误',
                'en': 'Line detection API error',
                'actions': ['检查API配置', '验证认证信息', '重试操作']
            },
            'AuthenticationError': {
                'zh': '认证失败，请检查用户名和密码',
                'en': 'Authentication failed, please check username and password',
                'actions': ['检查密码设置', '联系管理员获取凭证', '重新配置API']
            },
            'PermissionError': {
                'zh': '权限不足，无法执行此操作',
                'en': 'Insufficient permissions to perform this operation',
                'actions': ['检查用户权限', '联系管理员', '使用有权限的账户']
            },

            # 数据错误
            'JSONDecodeError': {
                'zh': '数据格式错误，无法解析服务器响应',
                'en': 'Data format error, unable to parse server response',
                'actions': ['刷新数据', '检查数据源', '联系技术支持']
            },
            'ValueError': {
                'zh': '数据值错误，参数不在有效范围内',
                'en': 'Data value error, parameter not in valid range',
                'actions': ['检查输入参数', '使用默认值', '查看使用说明']
            },
            'KeyError': {
                'zh': '数据结构错误，缺少必要字段',
                'en': 'Data structure error, missing required fields',
                'actions': ['更新数据格式', '检查API版本', '联系开发人员']
            },

            # 配置错误
            'ConfigurationError': {
                'zh': '配置错误，请检查设置文件',
                'en': 'Configuration error, please check settings file',
                'actions': ['检查配置文件', '重置为默认设置', '查看配置文档']
            },
            'FileNotFoundError': {
                'zh': '文件未找到，请检查文件路径',
                'en': 'File not found, please check file path',
                'actions': ['检查文件路径', '确认文件存在', '重新选择文件']
            },

            # 系统错误
            'MemoryError': {
                'zh': '内存不足，请释放内存后重试',
                'en': 'Insufficient memory, please free memory and retry',
                'actions': ['关闭其他程序', '减少数据量', '重启应用程序']
            },
            'RuntimeError': {
                'zh': '运行时错误，系统遇到意外情况',
                'en': 'Runtime error, system encountered unexpected situation',
                'actions': ['重启应用程序', '检查系统资源', '联系技术支持']
            },

            # 用户输入错误
            'ValidationError': {
                'zh': '输入验证失败，请检查输入内容',
                'en': 'Input validation failed, please check input content',
                'actions': ['检查输入格式', '使用有效范围值', '参考输入示例']
            }
        }

        # 错误恢复建议
        self.recovery_guidance = {
            ErrorCategory.NETWORK: [
                '检查网络连接状态',
                '验证服务器地址和端口',
                '尝试ping服务器地址',
                '检查防火墙设置',
                '联系网络管理员'
            ],
            ErrorCategory.API: [
                '检查API配置参数',
                '验证认证信息',
                '查看API文档',
                '检查API版本兼容性',
                '联系API提供方'
            ],
            ErrorCategory.AUTHENTICATION: [
                '检查用户名和密码',
                '确认账户状态',
                '重置密码',
                '联系管理员',
                '检查权限设置'
            ],
            ErrorCategory.DATA_PARSING: [
                '刷新数据源',
                '检查数据格式',
                '验证数据完整性',
                '尝试重新获取数据',
                '联系数据提供方'
            ],
            ErrorCategory.CONFIGURATION: [
                '检查配置文件语法',
                '重置为默认配置',
                '查看配置文档',
                '验证配置参数',
                '联系技术支持'
            ],
            ErrorCategory.PROCESSING: [
                '减少处理数据量',
                '检查系统资源',
                '重启处理流程',
                '更新软件版本',
                '联系开发人员'
            ],
            ErrorCategory.MEMORY: [
                '关闭其他应用程序',
                '减少数据缓存大小',
                '重启应用程序',
                '增加系统内存',
                '优化内存使用'
            ],
            ErrorCategory.TIMEOUT: [
                '增加超时时间设置',
                '检查网络速度',
                '减少单次请求数据量',
                '尝试分批处理',
                '优化网络环境'
            ]
        }

        # UI组件
        self.error_dialog = None
        self.status_bar_error = None
        self.network_status_label = None

        # 错误通知配置
        self.enable_sound_notifications = self.config.get('enable_sound', False)
        self.enable_visual_notifications = self.config.get('enable_visual', True)
        self.enable_auto_recovery = self.config.get('enable_auto_recovery', True)

        # 网络监控线程
        self.network_monitor_thread = None
        self.network_monitor_running = False

        logger.info("ClientErrorHandler initialized")

    def _translate_technical_error(self, error: Exception) -> str:
        """
        将技术错误消息翻译为用户友好的描述

        Args:
            error: 异常对象

        Returns:
            str: 用户友好的错误消息
        """
        try:
            error_name = type(error).__name__
            error_message = str(error)

            # 获取当前语言设置（默认中文）
            language = self.config.get('language', 'zh')

            # 查找翻译
            if error_name in self.error_translations:
                translation = self.error_translations[error_name]
                user_message = translation.get(language, translation.get('en', error_message))
            else:
                # 如果没有找到特定翻译，使用通用处理
                user_message = self._generate_generic_error_message(error_name, error_message, language)

            # 添加具体错误细节（如果有用的话）
            if 'password' in error_message.lower() or 'auth' in error_message.lower():
                user_message += "\n\n请检查认证配置中的密码设置。"
            elif 'connection' in error_message.lower():
                user_message += "\n\n请确认服务器地址和网络连接状态。"
            elif 'timeout' in error_message.lower():
                user_message += "\n\n服务器响应时间过长，请稍后重试。"

            return user_message

        except Exception as e:
            logger.error(f"Error translating technical error: {e}")
            return f"发生错误: {str(error)}\n请查看日志了解详细信息。"


class ClientErrorNotifier:
    """
    客户端错误通知器 - 提供多模态错误通知和反馈

    功能包括：
    - 视觉错误通知（弹窗、状态栏、图标）
    - 音频错误通知（可配置的提示音）
    - 错误状态指示器
    - 错误历史查看界面
    - 用户反馈收集
    """

    def __init__(self, parent_widget, error_handler: ClientErrorHandler):
        """
        初始化错误通知器

        Args:
            parent_widget: 父级Tkinter组件
            error_handler: 错误处理器实例
        """
        self.parent_widget = parent_widget
        self.error_handler = error_handler

        # 通知配置
        self.notification_config = {
            'popup_duration': 5000,  # 弹窗显示时长（毫秒）
            'fade_duration': 500,    # 淡入淡出时长
            'max_visible_popups': 3, # 最大同时显示弹窗数
            'position': 'top-right', # 弹窗位置
            'enable_sound': False,   # 是否启用声音
            'enable_vibration': False # 是否启用震动（如果支持）
        }

        # 当前通知列表
        self.active_notifications = []
        self.notification_queue = []

        # 错误图标和颜色
        self.error_icons = {
            ErrorSeverity.INFO: "ℹ️",
            ErrorSeverity.WARNING: "⚠️",
            ErrorSeverity.ERROR: "❌",
            ErrorSeverity.CRITICAL: "🔴"
        }

        self.error_colors = {
            ErrorSeverity.INFO: "#2196F3",      # 蓝色
            ErrorSeverity.WARNING: "#FF9800",   # 橙色
            ErrorSeverity.ERROR: "#F44336",     # 红色
            ErrorSeverity.CRITICAL: "#B71C1C"   # 深红色
        }

        logger.info("ClientErrorNotifier initialized")

    def show_error_notification(self, severity: ErrorSeverity, message: str,
                                actions: List[str] = None, context: str = ""):
        """
        显示错误通知

        Args:
            severity: 错误严重程度
            message: 错误消息
            actions: 建议操作列表
            context: 错误上下文
        """
        try:
            # 如果启用视觉通知，创建通知弹窗
            if self.enable_visual_notifications:
                self._create_notification_popup(severity, message, actions)

            # 如果启用声音通知，播放提示音
            if self.enable_sound_notifications:
                self._play_error_sound(severity)

            # 记录通知
            logger.info(f"Error notification displayed: {severity.value} - {message}")

        except Exception as e:
            logger.error(f"Error showing notification: {e}")
            # 如果通知显示失败，使用备用方案
            self._show_fallback_notification(message)

    def _create_notification_popup(self, severity: ErrorSeverity, message: str,
                                  actions: List[str] = None):
        """创建通知弹窗"""
        try:
            # 检查是否超过最大显示数量
            if len(self.active_notifications) >= self.notification_config['max_visible_popups']:
                # 移除最旧的通知
                oldest_notification = self.active_notifications.pop(0)
                try:
                    oldest_notification.destroy()
                except:
                    pass

            # 创建通知窗口
            notification = tk.Toplevel(self.parent_widget)
            notification.title("错误通知")
            notification.geometry("400x150")
            notification.overrideredirect(True)  # 无边框窗口

            # 设置背景颜色
            bg_color = self.error_colors.get(severity, "#FFFFFF")
            notification.configure(bg=bg_color)

            # 确定位置
            screen_width = notification.winfo_screenwidth()
            screen_height = notification.winfo_screenheight()
            x = screen_width - 420  # 右边距20px
            y = 100 + len(self.active_notifications) * 160  # 垂直偏移

            notification.geometry(f"+{x}+{y}")

            # 通知内容
            main_frame = tk.Frame(notification, bg=bg_color, padx=15, pady=10)
            main_frame.pack(fill='both', expand=True)

            # 图标和标题
            title_frame = tk.Frame(main_frame, bg=bg_color)
            title_frame.pack(fill='x', pady=(0, 5))

            icon = self.error_icons.get(severity, "❓")
            tk.Label(title_frame, text=icon, font=('Arial', 16), bg=bg_color).pack(side='left', padx=(0, 10))

            title_text = severity.value.upper()
            tk.Label(title_frame, text=title_text, font=('Microsoft YaHei', 12, 'bold'),
                    bg=bg_color, fg='white').pack(side='left')

            # 消息内容
            message_label = tk.Label(main_frame, text=message, wraplength=350,
                                   font=('Microsoft YaHei', 9), bg=bg_color, fg='white')
            message_label.pack(fill='x', pady=(5, 10))

            # 关闭按钮
            close_btn = tk.Button(main_frame, text="✕", font=('Arial', 10, 'bold'),
                                 bg=bg_color, fg='white', bd=0, relief='flat',
                                 command=notification.destroy)
            close_btn.place(relx=1.0, rely=0.0, x=-10, y=5)

            # 自动关闭
            auto_close_time = self.notification_config['popup_duration'] // 1000
            notification.after(auto_close_time * 1000, notification.destroy)

            # 添加到活动通知列表
            self.active_notifications.append(notification)

        except Exception as e:
            logger.error(f"Error creating notification popup: {e}")

    def _play_error_sound(self, severity: ErrorSeverity):
        """播放错误提示音"""
        try:
            # 这里可以实现声音播放功能
            # 例如使用 winsound（Windows）或其他跨平台音频库
            pass
        except Exception as e:
            logger.error(f"Error playing sound: {e}")

    def _show_fallback_notification(self, message: str):
        """显示备用通知"""
        try:
            import tkinter.messagebox as messagebox
            messagebox.showwarning("提示", message)
        except:
            print(f"NOTIFICATION: {message}")

    def show_pattern_warning(self, error_type: str, count: int):
        """显示错误模式警告"""
        try:
            warning_message = f"检测到重复错误模式：\n错误类型 {error_type} 在短时间内发生了 {count} 次。\n\n建议查看错误详情或联系技术支持。"

            self.show_error_notification(
                ErrorSeverity.WARNING,
                warning_message,
                ['查看错误历史', '联系技术支持', '重置配置'],
                "重复错误检测"
            )

        except Exception as e:
            logger.error(f"Error showing pattern warning: {e}")

    def clear_all_notifications(self):
        """清除所有活动通知"""
        try:
            for notification in self.active_notifications:
                try:
                    notification.destroy()
                except:
                    pass
            self.active_notifications.clear()
        except Exception as e:
            logger.error(f"Error clearing notifications: {e}")


def handle_client_error(self, error: Exception, context: str = "") -> bool:
    """
    处理客户端错误并提供用户友好的反馈

    Args:
        error: 异常对象
        context: 错误发生的上下文信息

    Returns:
        bool: 错误是否已成功处理
    """
    try:
        # 分类错误
        if hasattr(self, 'error_handler'):
            error_category = self.error_handler._classify_error(error)
            error_severity = self.error_handler._assess_error_severity(error, error_category)
        else:
            # 备用分类逻辑
            error_category = self._classify_error_for_widget(error)
            error_severity = self._assess_error_severity_for_widget(error, error_category)

        # 记录错误
        error_record = {
            'timestamp': datetime.now(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'category': error_category.value,
            'severity': error_severity.value,
            'context': context,
            'stack_trace': traceback.format_exc() if traceback else None,
            'user_action_suggested': [],
            'recovery_attempted': False,
            'recovery_successful': False
        }

        # 添加到错误历史
        self._add_to_error_history(error_record)

        # 更新错误统计
        self._update_error_statistics(error_record)

        # 获取用户友好的错误消息
        user_message = self.translate_technical_error(error)

        # 获取恢复建议
        if hasattr(self, 'error_handler'):
            recovery_actions = self.error_handler._get_recovery_actions(error_category, error)
        else:
            recovery_actions = self.provide_error_guidance(type(error).__name__)

        # 显示用户友好的错误通知
        if hasattr(self, 'error_notifier'):
            self.error_notifier.show_error_notification(
                error_severity, user_message, recovery_actions, context
            )
        else:
            # 如果没有通知器，使用基本错误显示
            self._show_basic_error_dialog(error_severity, user_message, recovery_actions)

        # 记录到日志
        logger.error(f"Client error handled: {error_record}")

        # 尝试自动恢复（如果启用）
        if self.enable_auto_recovery and error_severity != ErrorSeverity.CRITICAL:
            recovery_result = self._attempt_auto_recovery(error_category, error)
            error_record['recovery_attempted'] = True
            error_record['recovery_successful'] = recovery_result

        return True

    except Exception as handler_error:
        logger.critical(f"Error in error handler itself: {handler_error}")
        return False


def show_error_dialog(self, error_type: str, message: str, actions: List[str] = None) -> None:
    """
    显示详细错误对话框，包含建议操作

    Args:
        error_type: 错误类型
        message: 错误消息
        actions: 建议操作列表
    """
    try:
        if self.error_dialog:
            # 如果已有错误对话框，先关闭
            try:
                self.error_dialog.destroy()
            except:
                pass

        # 创建错误对话框
        self.error_dialog = tk.Toplevel(self.parent_widget)
        self.error_dialog.title("错误详情")
        self.error_dialog.geometry("600x400")
        self.error_dialog.resizable(True, True)

        # 设置对话框为模态
        self.error_dialog.transient(self.parent_widget)
        self.error_dialog.grab_set()

        # 主框架
        main_frame = ttk.Frame(self.error_dialog, padding="20")
        main_frame.pack(fill='both', expand=True)

        # 错误图标和标题
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill='x', pady=(0, 20))

        # 根据错误类型选择图标
        severity_icon = self._get_error_icon(error_type)
        ttk.Label(title_frame, text=severity_icon, font=('Arial', 16)).pack(side='left', padx=(0, 10))

        ttk.Label(title_frame, text="错误详情", font=('Microsoft YaHei', 14, 'bold')).pack(side='left')

        # 错误信息区域
        info_frame = ttk.LabelFrame(main_frame, text="错误信息", padding="15")
        info_frame.pack(fill='x', pady=(0, 15))

        # 错误类型
        ttk.Label(info_frame, text=f"错误类型: {error_type}", font=('Microsoft YaHei', 10, 'bold')).pack(anchor='w')

        # 错误消息
        message_text = tk.Text(info_frame, height=6, wrap='word', font=('Microsoft YaHei', 9))
        message_text.pack(fill='x', pady=(10, 0))
        message_text.insert('1.0', message)
        message_text.config(state='disabled')

        # 建议操作区域
        if actions:
            actions_frame = ttk.LabelFrame(main_frame, text="建议操作", padding="15")
            actions_frame.pack(fill='both', expand=True, pady=(0, 15))

            # 创建可滚动的操作列表
            actions_canvas = tk.Canvas(actions_frame, height=120)
            scrollbar = ttk.Scrollbar(actions_frame, orient="vertical", command=actions_canvas.yview)
            scrollable_frame = ttk.Frame(actions_canvas)

            scrollable_frame.bind(
                "<Configure>",
                lambda e: actions_canvas.configure(scrollregion=actions_canvas.bbox("all"))
            )

            actions_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            actions_canvas.configure(yscrollcommand=scrollbar.set)

            # 添加操作项
            for i, action in enumerate(actions):
                action_frame = ttk.Frame(scrollable_frame)
                action_frame.pack(fill='x', pady=2)

                # 复选框（可选执行）
                action_var = tk.BooleanVar()
                action_cb = ttk.Checkbutton(action_frame, text=action, variable=action_var)
                action_cb.pack(side='left')

            actions_canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")

        # 按钮区域
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x', pady=(10, 0))

        # 按钮定义
        buttons = []

        # 重试按钮
        retry_btn = ttk.Button(button_frame, text="重试", command=lambda: self._retry_last_action())
        retry_btn.pack(side='left', padx=(0, 10))
        buttons.append(retry_btn)

        # 查看详情按钮
        details_btn = ttk.Button(button_frame, text="查看详情", command=lambda: self._show_error_details())
        details_btn.pack(side='left', padx=(0, 10))
        buttons.append(details_btn)

        # 报告问题按钮
        report_btn = ttk.Button(button_frame, text="报告问题", command=lambda: self._report_error_issue())
        report_btn.pack(side='left', padx=(0, 10))
        buttons.append(report_btn)

        # 关闭按钮
        close_btn = ttk.Button(button_frame, text="关闭", command=self.error_dialog.destroy)
        close_btn.pack(side='right')
        buttons.append(close_btn)

        # 居中显示对话框
        self._center_dialog(self.error_dialog)

        logger.info(f"Error dialog displayed: {error_type}")

    except Exception as e:
        logger.error(f"Error showing error dialog: {e}")
        # 如果错误对话框显示失败，使用简单的消息框
        self._show_fallback_error_message(message)


def check_network_connectivity(self) -> bool:
    """
    检查网络连接状态

    Returns:
        bool: 网络连接是否正常
    """
    try:
        import requests
        import socket

        # 记录检查开始时间
        start_time = time.time()

        # 检查基本网络连接
        try:
            # 尝试连接到配置的API服务器
            if hasattr(self, 'api_client') and self.api_client:
                test_url = f"{self.api_client.base_url}/health"
            else:
                test_url = "http://localhost:8421/health"

            response = requests.get(test_url, timeout=5)

            # 计算响应时间
            response_time = (time.time() - start_time) * 1000  # 毫秒

            # 更新网络状态
            self.network_status.update({
                'connected': response.status_code == 200,
                'last_check': datetime.now(),
                'response_time': response_time,
                'quality': self._assess_network_quality(response_time, response.status_code)
            })

            # 如果有网络状态标签，更新显示
            if self.network_status_label:
                self._update_network_status_display()

            return response.status_code == 200

        except requests.exceptions.ConnectionError:
            self.network_status.update({
                'connected': False,
                'last_check': datetime.now(),
                'response_time': None,
                'quality': 'poor'
            })
            return False

        except requests.exceptions.Timeout:
            self.network_status.update({
                'connected': False,
                'last_check': datetime.now(),
                'response_time': None,
                'quality': 'poor'
            })
            return False

    except Exception as e:
        logger.error(f"Error checking network connectivity: {e}")
        self.network_status['connected'] = False
        return False


def translate_technical_error(self, error: Exception) -> str:
    """
    将技术错误消息翻译为用户友好的描述

    Args:
        error: 异常对象

    Returns:
        str: 用户友好的错误消息
    """
    try:
        error_name = type(error).__name__
        error_message = str(error)

        # 获取当前语言设置（默认中文）
        language = self.config.get('language', 'zh')

        # 查找翻译
        if error_name in self.error_translations:
            translation = self.error_translations[error_name]
            user_message = translation.get(language, translation.get('en', error_message))
        else:
            # 如果没有找到特定翻译，使用通用处理
            user_message = self._generate_generic_error_message(error_name, error_message, language)

        # 添加具体错误细节（如果有用的话）
        if 'password' in error_message.lower() or 'auth' in error_message.lower():
            user_message += "\n\n请检查认证配置中的密码设置。"
        elif 'connection' in error_message.lower():
            user_message += "\n\n请确认服务器地址和网络连接状态。"
        elif 'timeout' in error_message.lower():
            user_message += "\n\n服务器响应时间过长，请稍后重试。"

        return user_message

    except Exception as e:
        logger.error(f"Error translating technical error: {e}")
        return f"发生错误: {str(error)}\n请查看日志了解详细信息。"


def log_user_error(self, error: Exception, user_action: str = "") -> None:
    """
    记录用户操作错误，用于分析和改进

    Args:
        error: 异常对象
        user_action: 用户执行的操作描述
    """
    try:
        user_error_record = {
            'timestamp': datetime.now(),
            'user_action': user_action,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'user_context': self._get_user_context(),
            'application_state': self._get_application_state(),
            'system_info': self._get_system_info()
        }

        # 添加到用户错误历史
        if not hasattr(self, 'user_error_history'):
            self.user_error_history = []

        self.user_error_history.append(user_error_record)

        # 限制历史记录大小
        max_user_errors = self.config.get('max_user_error_history', 50)
        if len(self.user_error_history) > max_user_errors:
            self.user_error_history = self.user_error_history[-max_user_errors:]

        logger.info(f"User error logged: {user_action} -> {type(error).__name__}")

        # 如果启用错误分析，可以进行实时分析
        if self.config.get('enable_error_analysis', False):
            self._analyze_user_error_pattern(user_error_record)

    except Exception as e:
        logger.error(f"Error logging user error: {e}")


def provide_error_guidance(self, error_type: str) -> List[str]:
    """
    根据错误类型提供具体的恢复指导

    Args:
        error_type: 错误类型

    Returns:
        List[str]: 恢复指导步骤列表
    """
    try:
        # 将错误类型映射到错误类别
        error_category = self._map_error_type_to_category(error_type)

        # 获取基础恢复指导
        base_guidance = self.recovery_guidance.get(error_category, [
            "重启应用程序",
            "检查网络连接",
            "联系技术支持"
        ])

        # 根据具体错误类型添加特定指导
        specific_guidance = []

        if error_type == 'ConnectionError':
            specific_guidance.extend([
                "检查服务器是否运行",
                "验证服务器地址和端口",
                "尝试在浏览器中访问服务器URL"
            ])
        elif error_type == 'TimeoutError':
            specific_guidance.extend([
                "增加操作超时时间设置",
                "减少单次处理的数据量",
                "检查网络延迟和带宽"
            ])
        elif error_type == 'AuthenticationError':
            specific_guidance.extend([
                "检查用户名和密码是否正确",
                "确认账户未被锁定",
                "尝试重置密码"
            ])
        elif error_type == 'MemoryError':
            specific_guidance.extend([
                "关闭其他占用内存的程序",
                "减少数据处理缓存大小",
                "考虑增加系统内存"
            ])
        elif error_type == 'ConfigurationError':
            specific_guidance.extend([
                "检查配置文件语法",
                "重置为默认配置",
                "参考配置文档说明"
            ])

        # 合并指导步骤
        full_guidance = base_guidance + specific_guidance

        # 去重并限制数量
        unique_guidance = list(dict.fromkeys(full_guidance))  # 保持顺序的去重
        return unique_guidance[:8]  # 最多返回8条指导

    except Exception as e:
        logger.error(f"Error providing guidance for {error_type}: {e}")
        return ["重启应用程序", "联系技术支持"]


# 将错误处理方法添加到LineDetectionWidget类
def _setup_error_handling(self):
    """为LineDetectionWidget设置错误处理系统"""
    try:
        # 初始化错误处理器
        if not hasattr(self, 'error_handler'):
            self.error_handler = ClientErrorHandler(self.main_frame, self.config)

        # 初始化错误通知器
        if not hasattr(self, 'error_notifier'):
            self.error_notifier = ClientErrorNotifier(self.main_frame, self.error_handler)

        # 设置网络监控
        self._setup_network_monitoring()

        # 创建状态栏错误显示
        self._create_error_status_display()

        # 重写现有方法以包含错误处理
        self._wrap_methods_with_error_handling()

        logger.info("Error handling system setup completed")

    except Exception as e:
        logger.error(f"Error setting up error handling: {e}")


def _wrap_methods_with_error_handling(self):
    """为关键方法包装错误处理"""
    methods_to_wrap = [
        'update_roi1_image',
        'render_detected_lines',
        'render_intersection_point',
        'handle_toggle_detection',
        'handle_manual_detection',
        'handle_refresh'
    ]

    for method_name in methods_to_wrap:
        if hasattr(self, method_name):
            original_method = getattr(self, method_name)

            def create_wrapper(method):
                def wrapper(*args, **kwargs):
                    try:
                        return method(*args, **kwargs)
                    except Exception as e:
                        self.handle_client_error(e, f"在 {method.__name__} 方法中")
                        return False
                return wrapper

            wrapped_method = create_wrapper(original_method)
            setattr(self, method_name, wrapped_method)


# 将方法添加到LineDetectionWidget类
LineDetectionWidget._setup_error_handling = _setup_error_handling
LineDetectionWidget._wrap_methods_with_error_handling = _wrap_methods_with_error_handling

# 为LineDetectionWidget类添加错误处理方法
LineDetectionWidget.handle_client_error = handle_client_error
LineDetectionWidget.show_error_dialog = show_error_dialog
LineDetectionWidget.check_network_connectivity = check_network_connectivity
LineDetectionWidget.translate_technical_error = translate_technical_error
LineDetectionWidget.log_user_error = log_user_error
LineDetectionWidget.provide_error_guidance = provide_error_guidance

logger.info("Task 32: Client-side error handling and user feedback mechanisms implemented")


# ============ Task 32: Helper Methods for Error Handling System ============

def _classify_error(self, error: Exception) -> ErrorCategory:
    """分类错误类型"""
    error_name = type(error).__name__

    if 'Connection' in error_name or 'Network' in error_name:
        return ErrorCategory.NETWORK
    elif 'Timeout' in error_name:
        return ErrorCategory.TIMEOUT
    elif 'Auth' in error_name or 'Permission' in error_name or 'Login' in error_name:
        return ErrorCategory.AUTHENTICATION
    elif 'API' in error_name or 'LineDetection' in error_name:
        return ErrorCategory.API
    elif 'JSON' in error_name or 'Parse' in error_name or 'Decode' in error_name:
        return ErrorCategory.DATA_PARSING
    elif 'Config' in error_name or 'FileNotFound' in error_name:
        return ErrorCategory.CONFIGURATION
    elif 'Memory' in error_name:
        return ErrorCategory.MEMORY
    elif 'Value' in error_name or 'Type' in error_name:
        return ErrorCategory.USER_INPUT
    elif 'Runtime' in error_name or 'Process' in error_name:
        return ErrorCategory.PROCESSING
    else:
        return ErrorCategory.UNKNOWN


def _assess_error_severity(self, error: Exception, category: ErrorCategory) -> ErrorSeverity:
    """评估错误严重程度"""
    error_name = type(error).__name__
    error_message = str(error).lower()

    # 关键词映射
    critical_keywords = ['critical', 'fatal', 'severe', 'crash', 'corrupt', 'invalid']
    error_keywords = ['error', 'fail', 'exception', 'timeout', 'disconnect']
    warning_keywords = ['warning', 'deprecated', 'retry', 'temporary']

    if any(keyword in error_message for keyword in critical_keywords):
        return ErrorSeverity.CRITICAL
    elif any(keyword in error_message for keyword in error_keywords):
        return ErrorSeverity.ERROR
    elif any(keyword in error_message for keyword in warning_keywords):
        return ErrorSeverity.WARNING
    elif category in [ErrorCategory.MEMORY, ErrorCategory.AUTHENTICATION]:
        return ErrorSeverity.ERROR
    elif category in [ErrorCategory.NETWORK, ErrorCategory.TIMEOUT]:
        return ErrorSeverity.WARNING
    else:
        return ErrorSeverity.INFO


def _add_to_error_history(self, error_record: Dict[str, Any]):
    """添加错误到历史记录"""
    if hasattr(self, 'error_handler'):
        self.error_handler.error_history.append(error_record)

        # 限制历史记录大小
        max_size = self.error_handler.max_history_size
        if len(self.error_handler.error_history) > max_size:
            self.error_handler.error_history = self.error_handler.error_history[-max_size:]


def _update_error_statistics(self, error_record: Dict[str, Any]):
    """更新错误统计信息"""
    if hasattr(self, 'error_handler'):
        stats = self.error_handler.error_statistics

        # 总错误数
        stats['total_errors'] += 1

        # 按类别统计
        category = error_record['category']
        stats['by_category'][category] = stats['by_category'].get(category, 0) + 1

        # 按严重程度统计
        severity = error_record['severity']
        stats['by_severity'][severity] = stats['by_severity'].get(severity, 0) + 1

        # 按小时统计
        hour = error_record['timestamp'].hour
        hour_key = f"{hour:02d}:00"
        stats['by_hour'][hour_key] = stats['by_hour'].get(hour_key, 0) + 1

        # 最近错误
        stats['recent_errors'].append(error_record)
        if len(stats['recent_errors']) > 10:
            stats['recent_errors'] = stats['recent_errors'][-10:]


def _get_recovery_actions(self, category: ErrorCategory, error: Exception) -> List[str]:
    """获取错误恢复建议"""
    if hasattr(self, 'error_handler'):
        # 获取基础恢复指导
        base_actions = self.error_handler.recovery_guidance.get(category, [])

        # 根据具体错误类型添加特定操作
        error_name = type(error).__name__
        error_message = str(error).lower()

        specific_actions = []

        if 'password' in error_message:
            specific_actions.extend(['检查密码配置', '联系管理员获取正确密码'])
        elif 'timeout' in error_message:
            specific_actions.extend(['增加超时设置', '检查网络连接速度'])
        elif 'connection' in error_message:
            specific_actions.extend(['检查服务器地址', '验证网络连接'])
        elif 'permission' in error_message:
            specific_actions.extend(['检查用户权限', '使用管理员账户'])
        elif 'memory' in error_message:
            specific_actions.extend(['释放系统内存', '减少数据处理量'])

        # 合并并去重
        all_actions = base_actions + specific_actions
        return list(dict.fromkeys(all_actions))[:8]  # 最多返回8个操作

    return ['重启应用程序', '联系技术支持']


def _get_error_icon(self, error_type: str) -> str:
    """根据错误类型获取图标"""
    if 'Error' in error_type or 'Exception' in error_type:
        return "❌"
    elif 'Warning' in error_type or 'Timeout' in error_type:
        return "⚠️"
    elif 'Info' in error_type:
        return "ℹ️"
    elif 'Critical' in error_type or 'Fatal' in error_type:
        return "🔴"
    else:
        return "❓"


def _assess_network_quality(self, response_time: float, status_code: int) -> str:
    """评估网络质量"""
    if status_code != 200:
        return 'poor'
    elif response_time < 200:
        return 'excellent'
    elif response_time < 500:
        return 'good'
    elif response_time < 1500:
        return 'fair'
    else:
        return 'poor'


def _generate_generic_error_message(self, error_name: str, error_message: str, language: str) -> str:
    """生成通用错误消息"""
    if language == 'zh':
        return f"发生了 {error_name} 类型的错误。\n\n错误详情：{error_message}\n\n请检查系统状态并重试操作。"
    else:
        return f"A {error_name} error occurred.\n\nError details: {error_message}\n\nPlease check system status and retry the operation."


def _get_user_context(self) -> Dict[str, Any]:
    """获取用户上下文信息"""
    context = {
        'current_time': datetime.now().isoformat(),
        'user_session': getattr(self, 'session_id', 'unknown'),
        'active_features': []
    }

    # 添加当前活动功能
    if hasattr(self, 'detection_enabled') and self.detection_enabled:
        context['active_features'].append('line_detection')

    if hasattr(self, 'image_displayed') and self.image_displayed:
        context['active_features'].append('image_display')

    if hasattr(self, 'api_client') and self.api_client:
        context['active_features'].append('api_connected')

    return context


def _get_application_state(self) -> Dict[str, Any]:
    """获取应用程序状态"""
    state = {
        'widget_initialized': hasattr(self, 'main_frame'),
        'api_client_available': hasattr(self, 'api_client') and self.api_client is not None,
        'error_handler_available': hasattr(self, 'error_handler'),
        'current_config_size': len(getattr(self, 'config', {})),
        'memory_usage': 'unknown'  # 可以添加内存使用情况
    }

    # 添加图像状态
    if hasattr(self, 'current_roi1_data'):
        state['has_roi_data'] = self.current_roi1_data is not None
        state['roi_data_size'] = len(self.current_roi1_data) if self.current_roi1_data else 0

    return state


def _get_system_info(self) -> Dict[str, Any]:
    """获取系统信息"""
    import platform
    import sys

    return {
        'platform': platform.system(),
        'platform_version': platform.version(),
        'python_version': sys.version,
        'architecture': platform.architecture()[0],
        'processor': platform.processor()
    }


def _show_basic_error_dialog(self, severity: ErrorSeverity, message: str, actions: List[str]):
    """显示基本错误对话框（当通知器不可用时）"""
    try:
        dialog = tk.Toplevel(self.parent_widget)
        dialog.title("错误")
        dialog.geometry("400x200")

        # 错误消息
        ttk.Label(dialog, text=message, wraplength=350).pack(pady=20)

        # 按钮
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)

        ttk.Button(button_frame, text="确定", command=dialog.destroy).pack()

        # 居中显示
        dialog.transient(self.parent_widget)
        dialog.grab_set()

    except Exception as e:
        logger.error(f"Error showing basic error dialog: {e}")
        # 最后的备用方案：打印到控制台
        print(f"ERROR: {message}")


def _retry_last_action(self):
    """重试最后一次操作"""
    if hasattr(self, '_last_failed_action'):
        try:
            action_func = self._last_failed_action.get('function')
            action_args = self._last_failed_action.get('args', [])
            action_kwargs = self._last_failed_action.get('kwargs', {})

            if action_func:
                action_func(*action_args, **action_kwargs)
                logger.info("Last action retried successfully")
        except Exception as e:
            logger.error(f"Error retrying last action: {e}")
            self.handle_client_error(e, "重试最后一次操作时发生错误")


def _show_error_details(self):
    """显示错误详情"""
    if hasattr(self, 'error_handler') and self.error_handler.error_history:
        last_error = self.error_handler.error_history[-1]

        details_dialog = tk.Toplevel(self.parent_widget)
        details_dialog.title("错误详情")
        details_dialog.geometry("600x500")

        # 创建滚动文本框
        text_frame = ttk.Frame(details_dialog)
        text_frame.pack(fill='both', expand=True, padx=10, pady=10)

        text_widget = tk.Text(text_frame, wrap='word')
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)

        # 添加错误详情
        details_text = f"""
时间戳: {last_error['timestamp']}
错误类型: {last_error['error_type']}
错误消息: {last_error['error_message']}
错误类别: {last_error['category']}
严重程度: {last_error['severity']}
上下文: {last_error.get('context', '')}

堆栈跟踪:
{last_error.get('stack_trace', '无堆栈跟踪信息')}
        """

        text_widget.insert('1.0', details_text)
        text_widget.config(state='disabled')

        text_widget.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 关闭按钮
        ttk.Button(details_dialog, text="关闭", command=details_dialog.destroy).pack(pady=10)


def _report_error_issue(self):
    """报告错误问题"""
    try:
        # 创建错误报告对话框
        report_dialog = tk.Toplevel(self.parent_widget)
        report_dialog.title("报告问题")
        report_dialog.geometry("500x400")

        # 主框架
        main_frame = ttk.Frame(report_dialog, padding="20")
        main_frame.pack(fill='both', expand=True)

        # 说明文本
        ttk.Label(main_frame, text="请描述您遇到的问题：", font=('Microsoft YaHei', 10, 'bold')).pack(anchor='w')

        # 问题描述文本框
        problem_text = tk.Text(main_frame, height=8, wrap='word', font=('Microsoft YaHei', 9))
        problem_text.pack(fill='both', expand=True, pady=(10, 0))

        # 预填充一些信息
        if hasattr(self, 'error_handler') and self.error_handler.error_history:
            last_error = self.error_handler.error_history[-1]
            preset_text = f"""
错误类型: {last_error['error_type']}
错误消息: {last_error['error_message']}
发生时间: {last_error['timestamp']}

请在此处描述问题的具体情况和重现步骤...
            """
            problem_text.insert('1.0', preset_text.strip())

        # 按钮框架
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x', pady=(10, 0))

        def send_report():
            problem_description = problem_text.get('1.0', 'end-1c')
            logger.info(f"User error report: {problem_description}")

            # 显示感谢消息
            tk.messagebox.showinfo("感谢", "问题报告已收到，我们会尽快处理。")
            report_dialog.destroy()

        ttk.Button(button_frame, text="发送报告", command=send_report).pack(side='left')
        ttk.Button(button_frame, text="取消", command=report_dialog.destroy).pack(side='right')

    except Exception as e:
        logger.error(f"Error creating error report dialog: {e}")


def _center_dialog(self, dialog):
    """居中显示对话框"""
    try:
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
    except Exception as e:
        logger.error(f"Error centering dialog: {e}")


def _show_fallback_error_message(self, message: str):
    """显示备用错误消息"""
    try:
        import tkinter.messagebox as messagebox
        messagebox.showerror("错误", message)
    except:
        # 最后的备用方案
        print(f"FALLBACK ERROR: {message}")


def _setup_network_monitoring(self):
    """设置网络监控"""
    if hasattr(self, 'error_handler'):
        # 创建网络状态标签
        if hasattr(self, 'main_frame'):
            self.network_status_label = ttk.Label(
                self.main_frame,
                text="网络状态: 检查中...",
                font=('Microsoft YaHei', 8)
            )
            # 可以将标签添加到适当的位置

        # 启动网络监控线程
        if not self.error_handler.network_monitor_running:
            self.error_handler.network_monitor_running = True
            self._start_network_monitor()


def _start_network_monitor(self):
    """启动网络监控线程"""
    def monitor_network():
        while getattr(self.error_handler, 'network_monitor_running', False):
            try:
                connectivity = self.check_network_connectivity()

                # 更新UI（在主线程中）
                if hasattr(self, 'network_status_label'):
                    status_text = f"网络: {'已连接' if connectivity else '未连接'}"
                    if hasattr(self.error_handler, 'network_status'):
                        response_time = self.error_handler.network_status.get('response_time')
                        quality = self.error_handler.network_status.get('quality', 'unknown')

                        if response_time:
                            status_text += f" ({response_time:.0f}ms, {quality})"

                    self.parent_widget.after(0, lambda: self.network_status_label.config(text=status_text))

            except Exception as e:
                logger.error(f"Error in network monitor: {e}")

            # 每30秒检查一次
            time.sleep(30)

    if hasattr(self, 'error_handler'):
        self.error_handler.network_monitor_thread = threading.Thread(
            target=monitor_network,
            daemon=True
        )
        self.error_handler.network_monitor_thread.start()


def _create_error_status_display(self):
    """创建状态栏错误显示"""
    if hasattr(self, 'main_frame'):
        # 创建状态栏
        self.status_bar = ttk.Frame(self.main_frame)
        self.status_bar.pack(side='bottom', fill='x')

        # 错误状态标签
        self.status_bar_error = ttk.Label(
            self.status_bar,
            text="",
            font=('Microsoft YaHei', 8),
            foreground='red'
        )
        self.status_bar_error.pack(side='left', padx=5)

        # 网络状态标签（如果还没有创建）
        if not hasattr(self, 'network_status_label'):
            self.network_status_label = ttk.Label(
                self.status_bar,
                text="网络: 检查中...",
                font=('Microsoft YaHei', 8)
            )
            self.network_status_label.pack(side='right', padx=5)


def _attempt_auto_recovery(self, category: ErrorCategory, error: Exception) -> bool:
    """尝试自动恢复"""
    try:
        if category == ErrorCategory.NETWORK:
            # 尝试重新连接
            return self.check_network_connectivity()

        elif category == ErrorCategory.TIMEOUT:
            # 增加超时时间并重试
            if hasattr(self, 'api_client'):
                original_timeout = getattr(self.api_client, 'timeout', 10)
                self.api_client.timeout = original_timeout * 1.5
                return True

        elif category == ErrorCategory.MEMORY:
            # 尝试清理内存
            import gc
            gc.collect()

            # 清理缓存数据
            if hasattr(self, 'current_roi1_data'):
                self.current_roi1_data = None

            return True

        # 其他类型的错误暂时不进行自动恢复
        return False

    except Exception as e:
        logger.error(f"Error during auto recovery: {e}")
        return False


def _analyze_user_error_pattern(self, error_record: Dict[str, Any]):
    """分析用户错误模式"""
    try:
        # 这里可以实现错误模式分析
        # 例如：检测重复错误、错误趋势、用户行为模式等

        error_type = error_record['error_type']
        user_action = error_record.get('user_action', '')

        # 检测是否有重复的相同错误
        if hasattr(self, 'user_error_history') and len(self.user_error_history) > 1:
            recent_errors = self.user_error_history[-5:]  # 检查最近5个错误
            same_type_errors = [e for e in recent_errors if e['error_type'] == error_type]

            if len(same_type_errors) >= 3:
                logger.warning(f"Detected repeated error pattern: {error_type} occurred {len(same_type_errors)} times")

                # 可以触发特殊的用户指导或预防措施
                if hasattr(self, 'error_notifier'):
                    self.error_notifier.show_pattern_warning(error_type, len(same_type_errors))

    except Exception as e:
        logger.error(f"Error analyzing user error pattern: {e}")


def _classify_error_for_widget(self, error: Exception) -> ErrorCategory:
    """Widget级别的错误分类（备用方法）"""
    error_name = type(error).__name__

    if 'Connection' in error_name or 'Network' in error_name:
        return ErrorCategory.NETWORK
    elif 'Timeout' in error_name:
        return ErrorCategory.TIMEOUT
    elif 'Auth' in error_name or 'Permission' in error_name:
        return ErrorCategory.AUTHENTICATION
    elif 'API' in error_name or 'LineDetection' in error_name:
        return ErrorCategory.API
    elif 'JSON' in error_name or 'Parse' in error_name:
        return ErrorCategory.DATA_PARSING
    elif 'Config' in error_name or 'FileNotFound' in error_name:
        return ErrorCategory.CONFIGURATION
    elif 'Memory' in error_name:
        return ErrorCategory.MEMORY
    elif 'Value' in error_name:
        return ErrorCategory.USER_INPUT
    else:
        return ErrorCategory.UNKNOWN


def _assess_error_severity_for_widget(self, error: Exception, category: ErrorCategory) -> ErrorSeverity:
    """Widget级别的错误严重程度评估（备用方法）"""
    error_message = str(error).lower()

    if any(keyword in error_message for keyword in ['critical', 'fatal', 'severe']):
        return ErrorSeverity.CRITICAL
    elif any(keyword in error_message for keyword in ['error', 'fail', 'exception']):
        return ErrorSeverity.ERROR
    elif any(keyword in error_message for keyword in ['warning', 'timeout']):
        return ErrorSeverity.WARNING
    else:
        return ErrorSeverity.INFO


def _map_error_type_to_category(self, error_type: str) -> ErrorCategory:
    """将错误类型映射到错误类别"""
    if 'Connection' in error_type or 'Network' in error_type:
        return ErrorCategory.NETWORK
    elif 'Timeout' in error_type:
        return ErrorCategory.TIMEOUT
    elif 'Auth' in error_type or 'Permission' in error_type:
        return ErrorCategory.AUTHENTICATION
    elif 'API' in error_type:
        return ErrorCategory.API
    elif 'JSON' in error_type or 'Parse' in error_type:
        return ErrorCategory.DATA_PARSING
    elif 'Config' in error_type:
        return ErrorCategory.CONFIGURATION
    elif 'Memory' in error_type:
        return ErrorCategory.MEMORY
    else:
        return ErrorCategory.UNKNOWN


def _update_network_status_display(self):
    """更新网络状态显示"""
    if hasattr(self, 'network_status_label') and hasattr(self, 'error_handler'):
        network_status = self.error_handler.network_status
        if network_status['connected']:
            status_text = "网络: 已连接"
            response_time = network_status.get('response_time')
            quality = network_status.get('quality', 'unknown')
            if response_time:
                status_text += f" ({response_time:.0f}ms, {quality})"
        else:
            status_text = "网络: 未连接"

        self.network_status_label.config(text=status_text)


def _show_basic_error_dialog(self, severity: ErrorSeverity, message: str, actions: List[str]):
    """显示基本错误对话框（Widget级别）"""
    try:
        dialog = tk.Toplevel(self.parent_widget if hasattr(self, 'parent_widget') else self)
        dialog.title("错误")
        dialog.geometry("400x200")

        # 错误消息
        ttk.Label(dialog, text=message, wraplength=350).pack(pady=20)

        # 按钮
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)

        ttk.Button(button_frame, text="确定", command=dialog.destroy).pack()

        # 居中显示
        dialog.transient(self.parent_widget if hasattr(self, 'parent_widget') else self)
        dialog.grab_set()

    except Exception as e:
        logger.error(f"Error showing basic error dialog: {e}")
        print(f"ERROR: {message}")


def _add_to_error_history(self, error_record: Dict[str, Any]):
    """添加错误到历史记录（Widget级别）"""
    if hasattr(self, 'error_handler'):
        self.error_handler.error_history.append(error_record)

        # 限制历史记录大小
        max_size = self.error_handler.max_history_size
        if len(self.error_handler.error_history) > max_size:
            self.error_handler.error_history = self.error_handler.error_history[-max_size:]


def _update_error_statistics(self, error_record: Dict[str, Any]):
    """更新错误统计信息（Widget级别）"""
    if hasattr(self, 'error_handler'):
        stats = self.error_handler.error_statistics

        # 总错误数
        stats['total_errors'] += 1

        # 按类别统计
        category = error_record['category']
        stats['by_category'][category] = stats['by_category'].get(category, 0) + 1

        # 按严重程度统计
        severity = error_record['severity']
        stats['by_severity'][severity] = stats['by_severity'].get(severity, 0) + 1

        # 最近错误
        stats['recent_errors'].append(error_record)
        if len(stats['recent_errors']) > 10:
            stats['recent_errors'] = stats['recent_errors'][-10:]


# 将辅助方法添加到相应的类中
ClientErrorHandler._classify_error = _classify_error
ClientErrorHandler._assess_error_severity = _assess_error_severity
ClientErrorHandler._add_to_error_history = _add_to_error_history
ClientErrorHandler._update_error_statistics = _update_error_statistics
ClientErrorHandler._get_recovery_actions = _get_recovery_actions
ClientErrorHandler._get_error_icon = _get_error_icon
ClientErrorHandler._assess_network_quality = _assess_network_quality
ClientErrorHandler._generate_generic_error_message = _generate_generic_error_message
ClientErrorHandler._get_user_context = _get_user_context
ClientErrorHandler._get_application_state = _get_application_state
ClientErrorHandler._get_system_info = _get_system_info
ClientErrorHandler._show_basic_error_dialog = _show_basic_error_dialog
ClientErrorHandler._retry_last_action = _retry_last_action
ClientErrorHandler._show_error_details = _show_error_details
ClientErrorHandler._report_error_issue = _report_error_issue
ClientErrorHandler._center_dialog = _center_dialog
ClientErrorHandler._show_fallback_error_message = _show_fallback_error_message
ClientErrorHandler._setup_network_monitoring = _setup_network_monitoring
ClientErrorHandler._start_network_monitor = _start_network_monitor
ClientErrorHandler._create_error_status_display = _create_error_status_display
ClientErrorHandler._attempt_auto_recovery = _attempt_auto_recovery
ClientErrorHandler._analyze_user_error_pattern = _analyze_user_error_pattern
# ClientErrorHandler._translate_technical_error = ClientErrorHandler._translate_technical_error.__func__

# 为LineDetectionWidget添加备用错误处理方法
LineDetectionWidget._classify_error_for_widget = _classify_error_for_widget
LineDetectionWidget._assess_error_severity_for_widget = _assess_error_severity_for_widget
LineDetectionWidget._map_error_type_to_category = _map_error_type_to_category
LineDetectionWidget._update_network_status_display = _update_network_status_display
LineDetectionWidget._show_basic_error_dialog = _show_basic_error_dialog

logger.info("Task 32: Error handling helper methods implemented")