#!/usr/bin/env python3
"""
Simple test for Chinese control buttons functionality
"""

import tkinter as tk
from tkinter import ttk
import threading
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleChineseControls:
    """Simple Chinese control buttons test"""

    def __init__(self, parent_frame):
        self.parent_frame = parent_frame

        # 状态
        self.detection_enabled = False
        self.loading_states = {}

        # 中文字体
        self.chinese_font = ('Microsoft YaHei', 10, 'normal')

        # 创建UI
        self.setup_ui()

    def setup_ui(self):
        """设置用户界面"""
        # 主控制框架
        control_frame = ttk.LabelFrame(self.parent_frame, text="检测控制", padding=10)
        control_frame.pack(fill='x', padx=5, pady=5)

        # 按钮行
        button_row = ttk.Frame(control_frame)
        button_row.pack(fill='x', pady=(0, 5))

        # 启用/禁用检测按钮
        self.toggle_btn = ttk.Button(
            button_row,
            text="启用检测",
            command=self.handle_toggle_click,
            width=15
        )
        self.toggle_btn.pack(side='left', padx=(0, 10))

        # 手动检测按钮
        self.manual_btn = ttk.Button(
            button_row,
            text="手动检测",
            command=self.handle_manual_detection,
            width=15
        )
        self.manual_btn.pack(side='left', padx=(0, 10))

        # 立即刷新按钮
        self.refresh_btn = ttk.Button(
            button_row,
            text="立即刷新",
            command=self.handle_refresh,
            width=15
        )
        self.refresh_btn.pack(side='left', padx=(0, 10))

        # 状态标签
        self.status_label = ttk.Label(
            control_frame,
            text="检测状态: 未启用",
            font=self.chinese_font,
            foreground='gray'
        )
        self.status_label.pack(pady=(10, 0))

    def handle_toggle_click(self):
        """处理启用/禁用检测点击"""
        if self.loading_states.get('toggle', False):
            return

        # 后台处理
        thread = threading.Thread(target=self._process_toggle, daemon=True)
        thread.start()

    def _process_toggle(self):
        """后台处理切换"""
        try:
            # 设置加载状态
            self.parent_frame.after(0, lambda: self.set_loading_state('toggle', True))

            # 模拟API调用
            time.sleep(0.5)

            # 切换状态
            self.detection_enabled = not self.detection_enabled

            # 重置加载状态
            self.parent_frame.after(0, lambda: self.set_loading_state('toggle', False))

            logger.info(f"Detection toggled: {'enabled' if self.detection_enabled else 'disabled'}")

        except Exception as e:
            logger.error(f"Error in toggle: {e}")
            self.parent_frame.after(0, lambda: self.set_loading_state('toggle', False))

    def handle_manual_detection(self):
        """处理手动检测"""
        if self.loading_states.get('manual', False):
            return

        thread = threading.Thread(target=self._process_manual_detection, daemon=True)
        thread.start()

    def _process_manual_detection(self):
        """后台处理手动检测"""
        try:
            self.parent_frame.after(0, lambda: self.set_loading_state('manual', True))

            # 模拟检测
            time.sleep(0.8)

            self.parent_frame.after(0, lambda: self.set_loading_state('manual', False))

            logger.info("Manual detection completed")

        except Exception as e:
            logger.error(f"Error in manual detection: {e}")
            self.parent_frame.after(0, lambda: self.set_loading_state('manual', False))

    def handle_refresh(self):
        """处理刷新"""
        if self.loading_states.get('refresh', False):
            return

        thread = threading.Thread(target=self._process_refresh, daemon=True)
        thread.start()

    def _process_refresh(self):
        """后台处理刷新"""
        try:
            self.parent_frame.after(0, lambda: self.set_loading_state('refresh', True))

            time.sleep(0.3)

            self.parent_frame.after(0, lambda: self.set_loading_state('refresh', False))

            logger.info("Refresh completed")

        except Exception as e:
            logger.error(f"Error in refresh: {e}")
            self.parent_frame.after(0, lambda: self.set_loading_state('refresh', False))

    def set_loading_state(self, button_name: str, loading: bool):
        """设置按钮加载状态"""
        self.loading_states[button_name] = loading

        if button_name == 'toggle' and self.toggle_btn:
            if loading:
                self.toggle_btn.config(text="处理中...", state="disabled")
            else:
                self.toggle_btn.config(state="normal")
                self.update_button_text()

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

        self.update_status_display()

    def update_button_text(self):
        """更新按钮文本"""
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

    def set_detection_enabled(self, enabled: bool):
        """外部设置检测状态"""
        self.detection_enabled = enabled
        self.update_button_text()
        self.update_status_display()


def main():
    """主函数"""
    root = tk.Tk()
    root.title("Task 22 Chinese Controls Test")
    root.geometry("600x400")

    # 标题
    title = tk.Label(root, text="Task 22: 中文控制按钮与加载状态测试",
                     font=('Microsoft YaHei', 14, 'bold'), fg='blue')
    title.pack(pady=20)

    # 创建控制面板
    controls = SimpleChineseControls(root)

    # 测试按钮
    test_frame = ttk.Frame(root)
    test_frame.pack(pady=20)

    ttk.Button(test_frame, text="外部启用",
               command=lambda: controls.set_detection_enabled(True)).pack(side='left', padx=5)
    ttk.Button(test_frame, text="外部禁用",
               command=lambda: controls.set_detection_enabled(False)).pack(side='left', padx=5)
    ttk.Button(test_frame, text="退出", command=root.quit).pack(side='left', padx=20)

    logger.info("Simple Chinese Controls Test started")
    root.mainloop()
    logger.info("Simple Chinese Controls Test completed")


if __name__ == "__main__":
    main()