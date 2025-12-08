"""
Line Detection API Client Integration Example
演示如何使用线条检测API客户端集成的完整示例
"""

import tkinter as tk
from tkinter import ttk, messagebox
import logging
import threading
import time

# 导入线条检测组件
from line_detection_widget import LineDetectionWidget, StatusState
from line_detection_api_client import LineDetectionAPIClient, LineDetectionAPIError

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LineDetectionAPIExample:
    """线条检测API集成示例应用"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Line Detection API Integration Example")
        self.root.geometry("1200x800")

        # 默认API配置
        self.api_base_url = "http://localhost:8421"
        self.api_password = "31415"
        self.api_timeout = 10

        # 创建主界面
        self.setup_ui()

        logger.info("LineDetectionAPIExample initialized")

    def setup_ui(self):
        """设置用户界面"""
        # 顶部配置面板
        config_frame = ttk.LabelFrame(self.root, text="API Configuration", padding=10)
        config_frame.pack(fill='x', padx=10, pady=5)

        # API配置输入
        ttk.Label(config_frame, text="Base URL:").grid(row=0, column=0, sticky='w', padx=5)
        self.url_var = tk.StringVar(value=self.api_base_url)
        ttk.Entry(config_frame, textvariable=self.url_var, width=30).grid(row=0, column=1, padx=5)

        ttk.Label(config_frame, text="Password:").grid(row=0, column=2, sticky='w', padx=5)
        self.password_var = tk.StringVar(value=self.api_password)
        ttk.Entry(config_frame, textvariable=self.password_var, width=15, show="*").grid(row=0, column=3, padx=5)

        ttk.Label(config_frame, text="Timeout:").grid(row=0, column=4, sticky='w', padx=5)
        self.timeout_var = tk.StringVar(value=str(self.api_timeout))
        ttk.Entry(config_frame, textvariable=self.timeout_var, width=8).grid(row=0, column=5, padx=5)

        # 配置按钮
        ttk.Button(config_frame, text="Apply Config", command=self.apply_api_config).grid(row=0, column=6, padx=10)
        ttk.Button(config_frame, text="Test Connection", command=self.test_connection).grid(row=0, column=7, padx=5)

        # API状态面板
        status_frame = ttk.LabelFrame(self.root, text="API Status", padding=10)
        status_frame.pack(fill='x', padx=10, pady=5)

        self.api_status_label = ttk.Label(status_frame, text="API Status: Unknown", font=('Arial', 10))
        self.api_status_label.pack(anchor='w')

        self.stats_label = ttk.Label(status_frame, text="Statistics: Not available", font=('Arial', 9))
        self.stats_label.pack(anchor='w')

        # 按钮面板
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill='x', padx=10, pady=5)

        ttk.Button(button_frame, text="Toggle Detection", command=self.toggle_detection).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Manual Detection", command=self.manual_detection).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Get Status", command=self.get_status).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Get Config", command=self.get_config).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Enhanced Data", command=self.get_enhanced_data).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Reset Stats", command=self.reset_stats).pack(side='left', padx=5)

        # 主检测组件区域
        widget_frame = ttk.LabelFrame(self.root, text="Line Detection Widget", padding=5)
        widget_frame.pack(fill='both', expand=True, padx=10, pady=5)

        # 创建线条检测组件配置
        widget_config = {
            'api_base_url': self.api_base_url,
            'api_password': self.api_password,
            'api_timeout': self.api_timeout,
            'enable_api_integration': True,
            'enable_control_panel': True,
            'enable_chinese_status': True,
            'dark_theme': True,
            'enable_toolbar': True,
            'figsize': (10, 6),
            'dpi': 100
        }

        # 创建线条检测组件
        self.line_widget = LineDetectionWidget(widget_frame, config=widget_config)
        self.line_widget.main_frame.pack(fill='both', expand=True)

        # 定期更新状态
        self.update_status()

    def apply_api_config(self):
        """应用API配置"""
        try:
            base_url = self.url_var.get().strip()
            password = self.password_var.get().strip()
            timeout_str = self.timeout_var.get().strip()

            if not base_url:
                raise ValueError("Base URL is required")
            if not password:
                raise ValueError("Password is required")

            try:
                timeout = int(timeout_str)
                if timeout <= 0:
                    raise ValueError("Timeout must be positive")
            except ValueError:
                raise ValueError("Timeout must be a positive integer")

            # 更新本地配置
            self.api_base_url = base_url
            self.api_password = password
            self.api_timeout = timeout

            # 更新组件API配置
            self.line_widget.set_api_config(base_url, password, timeout)

            messagebox.showinfo("Success", "API configuration updated successfully")
            logger.info(f"API configuration updated: {base_url}")

        except Exception as e:
            messagebox.showerror("Configuration Error", f"Failed to apply configuration: {str(e)}")
            logger.error(f"Configuration error: {e}")

    def test_connection(self):
        """测试API连接"""
        def test():
            try:
                self.api_status_label.config(text="API Status: Testing connection...", foreground='blue')

                if self.line_widget.is_api_integration_available():
                    # 执行健康检查
                    health_result = self.line_widget.api_client.health_check()
                    if health_result.get('status') == 'healthy':
                        self.root.after(0, lambda: self.api_status_label.config(
                            text="API Status: Connected and healthy", foreground='green'
                        ))
                        logger.info("API connection test successful")
                    else:
                        self.root.after(0, lambda: self.api_status_label.config(
                            text=f"API Status: Unhealthy - {health_result.get('status', 'unknown')}",
                            foreground='orange'
                        ))
                else:
                    self.root.after(0, lambda: self.api_status_label.config(
                        text="API Status: Not available", foreground='red'
                    ))

            except Exception as e:
                self.root.after(0, lambda: self.api_status_label.config(
                    text=f"API Status: Connection failed - {str(e)}", foreground='red'
                ))
                logger.error(f"Connection test failed: {e}")

        # 在后台线程中测试
        threading.Thread(target=test, daemon=True).start()

    def toggle_detection(self):
        """切换检测状态"""
        if not self.line_widget.is_api_integration_available():
            messagebox.showwarning("API Unavailable", "API integration is not available")
            return

        # 获取当前状态
        try:
            status_data = self.line_widget.get_line_detection_status()
            current_enabled = status_data.get('enabled', False)
        except:
            current_enabled = False

        # 切换状态
        new_enabled = not current_enabled

        def toggle():
            try:
                self.api_status_label.config(
                    text=f"API Status: {'Enabling' if new_enabled else 'Disabling'} detection...",
                    foreground='blue'
                )

                if new_enabled:
                    result = self.line_widget.api_client.enable_line_detection()
                else:
                    result = self.line_widget.api_client.disable_line_detection()

                if result.get('success', False):
                    action = "enabled" if new_enabled else "disabled"
                    self.root.after(0, lambda: self.api_status_label.config(
                        text=f"API Status: Detection {action} successfully", foreground='green'
                    ))
                    logger.info(f"Detection {action} successfully")
                else:
                    error_msg = result.get('error', 'Unknown error')
                    raise Exception(error_msg)

            except Exception as e:
                self.root.after(0, lambda: self.api_status_label.config(
                    text=f"API Status: Toggle failed - {str(e)}", foreground='red'
                ))
                logger.error(f"Detection toggle failed: {e}")

        # 在后台线程中执行
        threading.Thread(target=toggle, daemon=True).start()

    def manual_detection(self):
        """执行手动检测"""
        if not self.line_widget.is_api_integration_available():
            messagebox.showwarning("API Unavailable", "API integration is not available")
            return

        def detect():
            try:
                self.api_status_label.config(text="API Status: Executing manual detection...", foreground='blue')

                result = self.line_widget.api_client.manual_detection(force_refresh=True)

                if result.get('success', False):
                    lines = result.get('lines', [])
                    intersections = result.get('intersections', [])
                    processing_time = result.get('processing_time_ms', 0)

                    # 更新可视化
                    self.root.after(0, lambda: self.line_widget.update_visualization({
                        'lines': lines,
                        'intersections': intersections
                    }))

                    # 更新状态
                    if intersections:
                        best_intersection = max(intersections, key=lambda x: x.get('confidence', 0))
                        point = best_intersection.get('point')
                        confidence = best_intersection.get('confidence', 1.0)

                        self.root.after(0, lambda: self.line_widget.update_intersection_status(
                            StatusState.DETECTION_SUCCESS,
                            intersection=tuple(point) if point else None,
                            confidence=confidence
                        ))
                    else:
                        self.root.after(0, lambda: self.line_widget.update_intersection_status(
                            StatusState.ENABLED_NO_DETECTION
                        ))

                    self.root.after(0, lambda: self.api_status_label.config(
                        text=f"API Status: Detection completed - {len(lines)} lines, {len(intersections)} intersections ({processing_time}ms)",
                        foreground='green'
                    ))
                    logger.info(f"Manual detection completed: {len(lines)} lines, {len(intersections)} intersections")
                else:
                    error_msg = result.get('error', 'Manual detection failed')
                    raise Exception(error_msg)

            except Exception as e:
                self.root.after(0, lambda: self.api_status_label.config(
                    text=f"API Status: Manual detection failed - {str(e)}", foreground='red'
                ))
                self.root.after(0, lambda: self.line_widget.update_intersection_status(
                    StatusState.DETECTION_ERROR, error_msg=str(e)
                ))
                logger.error(f"Manual detection failed: {e}")

        # 在后台线程中执行
        threading.Thread(target=detect, daemon=True).start()

    def get_status(self):
        """获取检测状态"""
        if not self.line_widget.is_api_integration_available():
            messagebox.showwarning("API Unavailable", "API integration is not available")
            return

        def get_status():
            try:
                status_data = self.line_widget.get_line_detection_status()

                # 格式化状态信息
                enabled = status_data.get('enabled', False)
                status = status_data.get('status', 'unknown')
                detection_count = status_data.get('detection_count', 0)
                error_count = status_data.get('error_count', 0)
                last_detection_time = status_data.get('last_detection_time')
                last_error = status_data.get('last_error')

                info = f"Enabled: {enabled}\n"
                info += f"Status: {status}\n"
                info += f"Detection Count: {detection_count}\n"
                info += f"Error Count: {error_count}\n"
                if last_detection_time:
                    info += f"Last Detection: {last_detection_time}\n"
                if last_error:
                    info += f"Last Error: {last_error}"

                self.root.after(0, lambda: messagebox.showinfo("Detection Status", info))
                logger.info("Detection status retrieved successfully")

            except Exception as e:
                error_msg = f"Failed to get status: {str(e)}"
                self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
                logger.error(f"Get status failed: {e}")

        threading.Thread(target=get_status, daemon=True).start()

    def get_config(self):
        """获取检测配置"""
        if not self.line_widget.is_api_integration_available():
            messagebox.showwarning("API Unavailable", "API integration is not available")
            return

        def get_config():
            try:
                config_data = self.line_widget.get_line_detection_config()

                if config_data:
                    # 格式化配置信息
                    info = f"Enabled: {config_data.get('enabled', False)}\n"
                    info += f"HSV Green Lower: {config_data.get('hsv_green_lower', [])}\n"
                    info += f"HSV Green Upper: {config_data.get('hsv_green_upper', [])}\n"
                    info += f"Canny Low Threshold: {config_data.get('canny_low_threshold', 0)}\n"
                    info += f"Canny High Threshold: {config_data.get('canny_high_threshold', 0)}\n"
                    info += f"Hough Threshold: {config_data.get('hough_threshold', 0)}\n"
                    info += f"Min Line Length: {config_data.get('hough_min_line_length', 0)}\n"
                    info += f"Max Line Gap: {config_data.get('hough_max_line_gap', 0)}\n"
                    info += f"Min Confidence: {config_data.get('min_confidence', 0)}\n"
                    info += f"Processing Mode: {config_data.get('roi_processing_mode', 'unknown')}\n"
                    info += f"Cache Timeout: {config_data.get('cache_timeout_ms', 0)}ms\n"
                    info += f"Max Processing Time: {config_data.get('max_processing_time_ms', 0)}ms"

                    self.root.after(0, lambda: messagebox.showinfo("Detection Configuration", info))
                    logger.info("Detection configuration retrieved successfully")
                else:
                    self.root.after(0, lambda: messagebox.showwarning("No Data", "No configuration data available"))

            except Exception as e:
                error_msg = f"Failed to get config: {str(e)}"
                self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
                logger.error(f"Get config failed: {e}")

        threading.Thread(target=get_config, daemon=True).start()

    def get_enhanced_data(self):
        """获取增强实时数据"""
        if not self.line_widget.is_api_integration_available():
            messagebox.showwarning("API Unavailable", "API integration is not available")
            return

        def get_data():
            try:
                enhanced_data = self.line_widget.get_enhanced_realtime_data_with_line_detection(count=50)

                if enhanced_data:
                    data_points = enhanced_data.get('data_points', [])
                    line_data = enhanced_data.get('line_intersection_data', {})

                    info = f"Data Points: {len(data_points)}\n"
                    info += f"Line Data Available: {bool(line_data)}\n"
                    if line_data:
                        info += f"Lines: {len(line_data.get('lines', []))}\n"
                        info += f"Intersections: {len(line_data.get('intersections', []))}\n"

                    self.root.after(0, lambda: messagebox.showinfo("Enhanced Realtime Data", info))
                    logger.info(f"Enhanced data retrieved: {len(data_points)} points")
                else:
                    self.root.after(0, lambda: messagebox.showwarning("No Data", "No enhanced data available"))

            except Exception as e:
                error_msg = f"Failed to get enhanced data: {str(e)}"
                self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
                logger.error(f"Get enhanced data failed: {e}")

        threading.Thread(target=get_data, daemon=True).start()

    def reset_stats(self):
        """重置API统计信息"""
        if not self.line_widget.is_api_integration_available():
            messagebox.showwarning("API Unavailable", "API integration is not available")
            return

        try:
            self.line_widget.reset_api_statistics()
            messagebox.showinfo("Success", "API statistics reset successfully")
            logger.info("API statistics reset")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to reset statistics: {str(e)}")
            logger.error(f"Reset stats failed: {e}")

    def update_status(self):
        """定期更新状态显示"""
        try:
            if self.line_widget.is_api_integration_available():
                stats = self.line_widget.get_api_statistics()
                if stats and 'error' not in stats:
                    request_count = stats.get('request_count', 0)
                    error_count = stats.get('error_count', 0)
                    success_rate = stats.get('success_rate', 1.0)

                    self.stats_label.config(
                        text=f"Statistics: {request_count} requests, {error_count} errors, {success_rate:.1%} success rate"
                    )
                else:
                    self.stats_label.config(text="Statistics: Not available")
            else:
                self.stats_label.config(text="Statistics: API not available")

        except Exception as e:
            logger.error(f"Error updating status: {e}")

        # 每5秒更新一次
        self.root.after(5000, self.update_status)

    def run(self):
        """运行应用"""
        try:
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            self.root.mainloop()
        except KeyboardInterrupt:
            logger.info("Application interrupted by user")

    def on_closing(self):
        """应用关闭时的清理工作"""
        try:
            logger.info("Cleaning up application resources")

            # 清理线条检测组件
            if hasattr(self, 'line_widget'):
                self.line_widget.cleanup()

            # 销毁窗口
            self.root.destroy()

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            self.root.destroy()


def main():
    """主函数"""
    try:
        app = LineDetectionAPIExample()
        app.run()
    except Exception as e:
        logger.error(f"Application error: {e}")
        messagebox.showerror("Application Error", f"Failed to start application: {str(e)}")


if __name__ == "__main__":
    main()