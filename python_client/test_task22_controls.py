#!/usr/bin/env python3
"""
Task 22 Test: Chinese Control Buttons with Loading States
Tests the new LineDetectionControls functionality with proper Chinese UI
"""

import tkinter as tk
from tkinter import ttk
import logging
import time
import threading

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the widget
from line_detection_widget import LineDetectionWidget, LineDetectionControls


class Task22TestApp:
    """Task 22 测试应用"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Task 22 Test: Chinese Control Buttons with Loading States")
        self.root.geometry("1200x800")

        # 设置中文字体
        self.chinese_font = ('Microsoft YaHei', 12, 'normal')
        self.chinese_font_bold = ('Microsoft YaHei', 12, 'bold')

        self.setup_ui()

        logger.info("Task 22 Test App initialized")

    def setup_ui(self):
        """设置用户界面"""
        # 主标题
        title_label = tk.Label(
            self.root,
            text="Task 22 测试: 中文控制按钮与加载状态",
            font=self.chinese_font_bold,
            fg='blue'
        )
        title_label.pack(pady=10)

        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=5)

        # 左侧：独立控制面板测试
        left_frame = ttk.LabelFrame(main_frame, text="独立控制面板测试", padding=10)
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))

        # 创建独立控制面板
        self.setup_standalone_control_panel(left_frame)

        # 右侧：集成控制面板测试
        right_frame = ttk.LabelFrame(main_frame, text="集成控制面板测试", padding=10)
        right_frame.pack(side='right', fill='both', expand=True, padx=(5, 0))

        # 创建集成控制面板的LineDetectionWidget
        self.setup_integrated_control_panel(right_frame)

        # 底部：测试控制
        bottom_frame = ttk.Frame(self.root)
        bottom_frame.pack(fill='x', padx=10, pady=5)

        self.setup_test_controls(bottom_frame)

    def setup_standalone_control_panel(self, parent):
        """设置独立控制面板测试"""
        # 创建回调函数
        callbacks = {
            'on_toggle': self.on_standalone_toggle,
            'on_manual_detection': self.on_standalone_manual_detection,
            'on_refresh': self.on_standalone_refresh,
            'on_clear_overlays': self.on_standalone_clear_overlays,
            'on_reset_view': self.on_standalone_reset_view,
            'on_save_screenshot': self.on_standalone_save_screenshot
        }

        # 创建控制面板
        self.standalone_control = LineDetectionControls(parent, callbacks)

        # 添加状态显示
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill='x', pady=10)

        self.standalone_status = tk.Label(
            status_frame,
            text="状态: 等待操作",
            font=self.chinese_font,
            fg='gray'
        )
        self.standalone_status.pack(side='left')

    def setup_integrated_control_panel(self, parent):
        """设置集成控制面板测试"""
        # 创建配置
        config = {
            'enable_control_panel': True,
            'enable_toolbar': True,
            'dark_theme': True
        }

        # 创建LineDetectionWidget
        self.integrated_widget = LineDetectionWidget(parent, config)

        # 获取控制面板引用
        self.integrated_control = self.integrated_widget.get_control_panel()

        # 设置回调
        if self.integrated_control:
            callbacks = {
                'on_toggle': self.on_integrated_toggle,
                'on_manual_detection': self.on_integrated_manual_detection,
                'on_refresh': self.on_integrated_refresh
            }
            self.integrated_control.set_callbacks(callbacks)

    def setup_test_controls(self, parent):
        """设置测试控制按钮"""
        ttk.Label(parent, text="测试控制:", font=self.chinese_font_bold).pack(side='left', padx=(0, 10))

        # 测试状态切换
        ttk.Button(
            parent,
            text="外部启用检测",
            command=self.test_external_enable
        ).pack(side='left', padx=5)

        ttk.Button(
            parent,
            text="外部禁用检测",
            command=self.test_external_disable
        ).pack(side='left', padx=5)

        ttk.Button(
            text="测试加载状态",
            command=self.test_loading_states
        ).pack(side='left', padx=5)

        ttk.Button(
            parent,
            text="退出测试",
            command=self.root.quit
        ).pack(side='right', padx=5)

    # ============ Standalone Control Panel Callbacks ============

    def on_standalone_toggle(self, enabled):
        """独立控制面板切换回调"""
        status = "启用" if enabled else "禁用"
        self.standalone_status.config(
            text=f"状态: 检测已{status}",
            fg='green' if enabled else 'red'
        )
        logger.info(f"Standalone control: Detection {status}")

    def on_standalone_manual_detection(self):
        """独立控制面板手动检测回调"""
        self.standalone_status.config(text="状态: 执行手动检测...", fg='orange')
        self.root.after(1000, lambda: self.standalone_status.config(text="状态: 手动检测完成", fg='blue'))
        logger.info("Standalone control: Manual detection triggered")

    def on_standalone_refresh(self):
        """独立控制面板刷新回调"""
        self.standalone_status.config(text="状态: 刷新数据...", fg='orange')
        self.root.after(500, lambda: self.standalone_status.config(text="状态: 刷新完成", fg='blue'))
        logger.info("Standalone control: Refresh triggered")

    def on_standalone_clear_overlays(self):
        """独立控制面板清除覆盖层回调"""
        self.standalone_status.config(text="状态: 覆盖层已清除", fg='blue')
        logger.info("Standalone control: Overlays cleared")

    def on_standalone_reset_view(self):
        """独立控制面板重置视图回调"""
        self.standalone_status.config(text="状态: 视图已重置", fg='blue')
        logger.info("Standalone control: View reset")

    def on_standalone_save_screenshot(self):
        """独立控制面板保存截图回调"""
        self.standalone_status.config(text="状态: 截图已保存", fg='blue')
        logger.info("Standalone control: Screenshot saved")

    # ============ Integrated Control Panel Callbacks ============

    def on_integrated_toggle(self, enabled):
        """集成控制面板切换回调"""
        logger.info(f"Integrated control: Detection {'enabled' if enabled else 'disabled'}")

    def on_integrated_manual_detection(self):
        """集成控制面板手动检测回调"""
        logger.info("Integrated control: Manual detection triggered")

    def on_integrated_refresh(self):
        """集成控制面板刷新回调"""
        logger.info("Integrated control: Refresh triggered")

    # ============ Test Control Methods ============

    def test_external_enable(self):
        """测试外部启用检测"""
        if self.integrated_control:
            self.integrated_control.set_detection_enabled(True)
            self.integrated_widget.update_detection_status("已启用", False)

    def test_external_disable(self):
        """测试外部禁用检测"""
        if self.integrated_control:
            self.integrated_control.set_detection_enabled(False)
            self.integrated_widget.update_detection_status("已禁用", False)

    def test_loading_states(self):
        """测试加载状态"""
        if self.integrated_control:
            # 测试不同按钮的加载状态
            buttons = ['toggle', 'manual', 'refresh']
            messages = ['处理中...', '检测中...', '刷新中...']

            for i, (button, message) in enumerate(zip(buttons, messages)):
                # 设置加载状态
                self.integrated_control.set_loading_state(button, True)
                self.integrated_widget.update_detection_status(message, True)

                # 1秒后恢复正常
                self.root.after(1000 + i*200, lambda b=button: self.integrated_control.set_loading_state(b, False))

            # 最后重置状态
            self.root.after(2000, lambda: self.integrated_widget.update_detection_status("测试完成", False))

    def run(self):
        """运行应用"""
        logger.info("Starting Task 22 Test App...")
        self.root.mainloop()


if __name__ == "__main__":
    try:
        app = Task22TestApp()
        app.run()
        logger.info("Task 22 Test completed successfully")
    except Exception as e:
        logger.error(f"Error in Task 22 Test: {e}")
        import traceback
        traceback.print_exc()