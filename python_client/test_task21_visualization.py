#!/usr/bin/env python3
"""
Test script for Task 21 - Line Overlay Rendering and Intersection Point Visualization
Demonstrates the new visualization features of LineDetectionWidget
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import base64
import io
from PIL import Image
import matplotlib.pyplot as plt

from line_detection_widget import LineDetectionWidget


def create_test_image_data(width=200, height=150):
    """创建测试图像数据（模拟ROI1图像）"""
    # 创建一个简单的测试图像
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # 添加一些背景噪声
    noise = np.random.randint(0, 50, (height, width, 3))
    image = image + noise

    # 添加一些绿色线条（模拟要检测的线）
    # 垂直线
    image[20:130, 50:52, 1] = 200  # G通道
    image[20:130, 50:52, 0] = 0    # R通道
    image[20:130, 50:52, 2] = 0    # B通道

    # 斜线
    for i in range(100):
        y = 25 + i
        x = 80 + int(i * 0.8)
        if y < height and x < width:
            image[y, x:x+2, 1] = 200  # G通道
            image[y, x:x+2, 0] = 0    # R通道
            image[y, x:x+2, 2] = 0    # B通道

    # 转换为base64
    pil_image = Image.fromarray(image)
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    image_bytes = buffer.getvalue()
    encoded_image = base64.b64encode(image_bytes).decode('utf-8')

    return f"data:image/png;base64,{encoded_image}"


class Task21DemoApp:
    """Task 21 功能演示应用"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Task 21 - Line Overlay Rendering & Intersection Visualization Demo")
        self.root.geometry("900x700")

        # 创建主框架
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # 创建标题
        title_label = ttk.Label(self.main_frame,
                               text="ROI1 Green Line Detection Visualization Demo",
                               font=('Arial', 14, 'bold'))
        title_label.pack(pady=(0, 10))

        # 创建水平分割
        self.create_horizontal_paned()

        # 创建控制面板
        self.create_control_panel()

        # 初始化演示数据
        self.init_demo_data()

        logger.info("Task 21 Demo App initialized")

    def create_horizontal_paned(self):
        """创建水平分割面板"""
        # 创建分割面板
        self.paned_window = ttk.PanedWindow(self.main_frame, orient='horizontal')
        self.paned_window.pack(fill='both', expand=True)

        # 左侧：可视化控件框架
        self.viz_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(self.viz_frame, weight=3)

        # 右侧：控制和信息框架
        self.control_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(self.control_frame, weight=1)

    def create_control_panel(self):
        """创建控制面板"""
        # 可视化控件
        viz_label = ttk.Label(self.viz_frame, text="Visualization:", font=('Arial', 12, 'bold'))
        viz_label.pack(pady=(0, 5))

        # 创建LineDetectionWidget
        config = {
            'figsize': (8, 6),
            'dpi': 100,
            'enable_toolbar': True,
            'dark_theme': True
        }

        self.line_widget = LineDetectionWidget(self.viz_frame, config)

        # 控制按钮框架
        control_label = ttk.Label(self.control_frame, text="Demo Controls:", font=('Arial', 12, 'bold'))
        control_label.pack(pady=(0, 10))

        # 加载测试图像
        load_btn = ttk.Button(self.control_frame, text="Load Test Image",
                             command=self.load_test_image)
        load_btn.pack(pady=5, fill='x', padx=10)

        # 演示单条线
        single_line_btn = ttk.Button(self.control_frame, text="Render Single Line",
                                    command=self.demo_single_line)
        single_line_btn.pack(pady=5, fill='x', padx=10)

        # 演示多条线
        multi_line_btn = ttk.Button(self.control_frame, text="Render Multiple Lines",
                                   command=self.demo_multiple_lines)
        multi_line_btn.pack(pady=5, fill='x', padx=10)

        # 演示单个交点
        single_intersection_btn = ttk.Button(self.control_frame, text="Render Single Intersection",
                                           command=self.demo_single_intersection)
        single_intersection_btn.pack(pady=5, fill='x', padx=10)

        # 演示多个交点
        multi_intersection_btn = ttk.Button(self.control_frame, text="Render Multiple Intersections",
                                          command=self.demo_multiple_intersections)
        multi_intersection_btn.pack(pady=5, fill='x', padx=10)

        # 演示完整检测
        full_detection_btn = ttk.Button(self.control_frame, text="Full Detection Demo",
                                      command=self.demo_full_detection)
        full_detection_btn.pack(pady=5, fill='x', padx=10)

        # 清除覆盖层
        clear_btn = ttk.Button(self.control_frame, text="Clear Overlays",
                              command=self.clear_overlays)
        clear_btn.pack(pady=5, fill='x', padx=10)

        # 分隔符
        ttk.Separator(self.control_frame, orient='horizontal').pack(fill='x', pady=20, padx=10)

        # 配置控制
        config_label = ttk.Label(self.control_frame, text="Visualization Config:", font=('Arial', 11, 'bold'))
        config_label.pack(pady=(0, 10))

        # 线条宽度调整
        ttk.Label(self.control_frame, text="Line Width:").pack()
        self.line_width_var = tk.DoubleVar(value=2.0)
        line_width_scale = ttk.Scale(self.control_frame, from_=0.5, to=5.0,
                                    variable=self.line_width_var, orient='horizontal',
                                    command=self.update_line_width)
        line_width_scale.pack(fill='x', padx=10, pady=5)

        # 透明度调整
        ttk.Label(self.control_frame, text="Line Alpha:").pack()
        self.line_alpha_var = tk.DoubleVar(value=0.8)
        alpha_scale = ttk.Scale(self.control_frame, from_=0.1, to=1.0,
                               variable=self.line_alpha_var, orient='horizontal',
                               command=self.update_alpha)
        alpha_scale.pack(fill='x', padx=10, pady=5)

        # 分隔符
        ttk.Separator(self.control_frame, orient='horizontal').pack(fill='x', pady=20, padx=10)

        # 信息显示
        info_label = ttk.Label(self.control_frame, text="Visualization Info:", font=('Arial', 11, 'bold'))
        info_label.pack(pady=(0, 10))

        self.info_text = tk.Text(self.control_frame, height=10, width=30, font=('Courier', 9))
        self.info_text.pack(fill='both', expand=True, padx=10, pady=5)

        # 更新信息按钮
        update_info_btn = ttk.Button(self.control_frame, text="Update Info",
                                    command=self.update_info)
        update_info_btn.pack(pady=5)

    def init_demo_data(self):
        """初始化演示数据"""
        # 生成测试线条数据
        self.demo_lines = [
            {
                'start': [50, 30],
                'end': [50, 120],
                'confidence': 0.95
            },
            {
                'start': [80, 25],
                'end': [150, 85],
                'confidence': 0.78
            },
            {
                'start': [30, 80],
                'end': [120, 40],
                'confidence': 0.62
            },
            {
                'start': [160, 30],
                'end': [180, 110],
                'confidence': 0.35
            }
        ]

        # 生成测试交点数据
        self.demo_intersections = [
            {
                'point': [50, 50],
                'confidence': 0.92
            },
            {
                'point': [95, 65],
                'confidence': 0.78
            },
            {
                'point': [110, 50],
                'confidence': 0.45
            }
        ]

    def load_test_image(self):
        """加载测试图像"""
        try:
            # 生成测试图像数据
            test_image_data = create_test_image_data()

            # 更新到可视化控件
            self.line_widget.update_roi1_image(test_image_data)

            logger.info("Test image loaded successfully")

        except Exception as e:
            logger.error(f"Error loading test image: {e}")

    def demo_single_line(self):
        """演示单条线渲染"""
        single_line = [self.demo_lines[0]]  # 高置信度线
        self.line_widget.render_detected_lines(single_line)
        logger.info("Rendered single line demo")

    def demo_multiple_lines(self):
        """演示多条线渲染"""
        self.line_widget.render_detected_lines(self.demo_lines)
        logger.info("Rendered multiple lines demo")

    def demo_single_intersection(self):
        """演示单个交点渲染"""
        intersection = self.demo_intersections[0]['point']
        confidence = self.demo_intersections[0]['confidence']
        self.line_widget.render_intersection_point(intersection, confidence)
        logger.info("Rendered single intersection demo")

    def demo_multiple_intersections(self):
        """演示多个交点渲染"""
        self.line_widget.update_multiple_intersections(self.demo_intersections)
        logger.info("Rendered multiple intersections demo")

    def demo_full_detection(self):
        """演示完整检测可视化"""
        detection_result = {
            'lines': self.demo_lines,
            'intersections': self.demo_intersections
        }
        self.line_widget.update_visualization(detection_result)
        logger.info("Rendered full detection demo")

    def clear_overlays(self):
        """清除所有覆盖层"""
        self.line_widget.clear_overlays()
        logger.info("Cleared all overlays")

    def update_line_width(self, value):
        """更新线条宽度"""
        new_width = float(value)
        self.line_widget.set_visualization_config({'line_width': new_width})

    def update_alpha(self, value):
        """更新透明度"""
        new_alpha = float(value)
        self.line_widget.set_visualization_config({'line_alpha': new_alpha})

    def update_info(self):
        """更新信息显示"""
        try:
            info = self.line_widget.get_visualization_info()

            # 格式化信息文本
            text_lines = []
            text_lines.append("=== Visualization Status ===")
            text_lines.append(f"Image Displayed: {info['image_displayed']}")
            text_lines.append("")
            text_lines.append("=== Overlay Counts ===")
            counts = info['overlay_count']
            text_lines.append(f"Lines: {counts['lines']}")
            text_lines.append(f"Intersections: {counts['intersections']}")
            text_lines.append(f"Crosshairs: {counts['crosshairs']}")
            text_lines.append(f"Text Labels: {counts['texts']}")
            text_lines.append("")
            text_lines.append("=== Configuration ===")
            config = info['visualization_config']
            text_lines.append(f"Outer Radius: {config['intersection_outer_radius']}px")
            text_lines.append(f"Inner Radius: {config['intersection_inner_radius']}px")
            text_lines.append(f"Crosshair Length: {config['crosshair_length']}px")
            text_lines.append(f"Line Width: {config['line_width']:.1f}")
            text_lines.append(f"Line Alpha: {config['line_alpha']:.2f}")
            text_lines.append(f"High Confidence: >{config['high_confidence_threshold']}")
            text_lines.append(f"Medium Confidence: >{config['medium_confidence_threshold']}")

            # 更新文本显示
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(1.0, '\n'.join(text_lines))

        except Exception as e:
            logger.error(f"Error updating info: {e}")

    def run(self):
        """运行应用"""
        # 加载测试图像
        self.root.after(100, self.load_test_image)

        # 启动主循环
        self.root.mainloop()


if __name__ == "__main__":
    import logging

    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # 创建并运行演示应用
    app = Task21DemoApp()
    app.run()