"""
基于HTTP的Python客户端实时绘图
使用HTTP轮询获取实时数据，实现与Web前端相同的实时曲线绘制
"""

import json
import logging
import threading
import time
import tkinter as tk
import os
from tkinter import messagebox, ttk, scrolledtext, StringVar
import requests
from typing import Dict, Any, Optional
from PIL import Image, ImageTk
import base64
import io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
try:
    from local_config_loader import LocalConfigLoader
    LOCAL_CONFIG_LOADER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: LocalConfigLoader import failed: {e}")
    LOCAL_CONFIG_LOADER_AVAILABLE = False
    LocalConfigLoader = None

try:
    from line_detection_config_manager import LineDetectionConfigManager
    LINE_DETECTION_CONFIG_AVAILABLE = True
except ImportError as e:
    print(f"Warning: LineDetectionConfigManager import failed: {e}")
    LINE_DETECTION_CONFIG_AVAILABLE = False
    LineDetectionConfigManager = None
from enum import Enum

from realtime_plotter import RealtimePlotter
from line_detection_widget import LineDetectionWidget

# 设置logger
logger = logging.getLogger(__name__)


class LineDetectionState(Enum):
    """绿线交点检测状态枚举"""
    DISABLED = "disabled"           # 检测未启用
    ENABLING = "enabling"           # 正在启用（过渡状态）
    ENABLED = "enabled"             # 检测已启用
    DISABLING = "disabling"         # 正在禁用（过渡状态）
    ERROR = "error"                 # 错误状态需要干预


class LineDetectionConfig:
    """绿线交点检测配置管理"""

    def __init__(self):
        self.enabled = False  # 检测是否启用
        self.auto_start = False  # 应用启动时自动启用
        self.auto_recovery = True  # 连接中断后自动恢复
        self.sync_interval = 5.0  # 状态同步间隔（秒）
        self.timeout = 10.0  # 操作超时时间（秒）
        self.retry_count = 3  # 重试次数
        self.retry_delay = 1.0  # 重试延迟（秒）

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'enabled': self.enabled,
            'auto_start': self.auto_start,
            'auto_recovery': self.auto_recovery,
            'sync_interval': self.sync_interval,
            'timeout': self.timeout,
            'retry_count': self.retry_count,
            'retry_delay': self.retry_delay
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LineDetectionConfig':
        """从字典创建配置"""
        config = cls()
        config.enabled = data.get('enabled', False)
        config.auto_start = data.get('auto_start', False)
        config.auto_recovery = data.get('auto_recovery', True)
        config.sync_interval = data.get('sync_interval', 5.0)
        config.timeout = data.get('timeout', 10.0)
        config.retry_count = data.get('retry_count', 3)
        config.retry_delay = data.get('retry_delay', 1.0)
        return config


class HTTPRealtimeClient:
    """基于HTTP的实时客户端"""

    def __init__(self, base_url: str = "http://localhost:8421", password: str = "31415"):
        self.base_url = base_url
        self.password = password
        self.session = requests.Session()

        # 状态变量
        self.connected = False
        self.detection_running = False
        self.polling_running = False
        self.polling_thread: Optional[threading.Thread] = None

        # 绿线交点检测状态管理
        self.line_detection_state = LineDetectionState.DISABLED
        self.line_detection_config = LineDetectionConfig()
        self.line_detection_lock = threading.RLock()  # 线程安全锁

        # 状态管理变量
        self.line_detection_state_callbacks = []  # 状态变化回调
        self.last_state_sync_time = 0
        self.state_recovery_in_progress = False
        self.state_sync_thread: Optional[threading.Thread] = None
        self.state_sync_running = False

        # 数据更新控制
        self.polling_interval = 0.05  # 50ms (20 FPS)
        self.data_count = 0
        self.last_update_time = 0

        # 双ROI模式
        self.dual_roi_mode = True  # 默认启用双ROI模式

        # 增强数据获取配置
        self.include_line_intersection = True  # 默认启用绿线交点检测数据获取
        self.enhanced_data_enabled = True  # 默认启用增强数据获取
        self.fallback_on_error = True  # 出错时回退到标准数据获取

        # 性能监控
        self.enhanced_fetch_count = 0
        self.enhanced_fetch_errors = 0
        self.last_fetch_time = 0
        self.avg_fetch_time = 0.05

        # 绘图器
        self.plotter: Optional[RealtimePlotter] = None

        # ROI更新回调
        self.roi_update_callback: Optional[callable] = None

        # 绿线检测配置管理器
        if LINE_DETECTION_CONFIG_AVAILABLE and LineDetectionConfigManager:
            try:
                self.line_detection_config_manager = LineDetectionConfigManager()
                self.line_detection_config_loaded = False
            except Exception as e:
                logger.warning(f"Failed to initialize LineDetectionConfigManager: {str(e)}")
                self.line_detection_config_manager = None
                self.line_detection_config_loaded = False
        else:
            self.line_detection_config_manager = None
            self.line_detection_config_loaded = False
            logger.warning("LineDetectionConfigManager not available, line detection configuration disabled")

        # 绿线交点数据回调
        self.line_intersection_callback: Optional[callable] = None

        logger.info(f"HTTPRealtimeClient initialized for {base_url}")
        logger.info(f"Enhanced data fetching: enabled={self.enhanced_data_enabled}, line_intersection={self.include_line_intersection}")

    def set_roi_update_callback(self, callback: callable):
        """设置ROI更新回调函数"""
        self.roi_update_callback = callback

    def set_line_intersection_callback(self, callback: callable):
        """设置绿线交点检测数据回调函数"""
        self.line_intersection_callback = callback

    def set_enhanced_data_config(self, include_line_intersection: bool = None,
                                enhanced_data_enabled: bool = None,
                                fallback_on_error: bool = None):
        """设置增强数据获取配置"""
        if include_line_intersection is not None:
            self.include_line_intersection = include_line_intersection
        if enhanced_data_enabled is not None:
            self.enhanced_data_enabled = enhanced_data_enabled
        if fallback_on_error is not None:
            self.fallback_on_error = fallback_on_error

        logger.info(f"Enhanced data config updated: enhanced={self.enhanced_data_enabled}, "
                   f"line_intersection={self.include_line_intersection}, fallback={self.fallback_on_error}")

    def get_enhanced_data_stats(self) -> Dict[str, Any]:
        """获取增强数据获取性能统计"""
        return {
            "enhanced_fetch_count": self.enhanced_fetch_count,
            "enhanced_fetch_errors": self.enhanced_fetch_errors,
            "error_rate": self.enhanced_fetch_errors / max(1, self.enhanced_fetch_count),
            "avg_fetch_time": self.avg_fetch_time,
            "include_line_intersection": self.include_line_intersection,
            "enhanced_data_enabled": self.enhanced_data_enabled
        }

    def test_connection(self) -> bool:
        """测试服务器连接"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info("Server connection successful")
                # 连接成功后初始化绿线交点检测状态
                if not hasattr(self, '_line_detection_initialized'):
                    self.initialize_line_detection_state()
                    self._line_detection_initialized = True
                return True
            else:
                logger.error(f"Server returned status code: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            self._handle_connection_lost()
            return False

    def get_system_status(self) -> Optional[Dict[str, Any]]:
        """获取系统状态"""
        try:
            response = self.session.get(f"{self.base_url}/status", timeout=5)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return None

    def get_realtime_data(self) -> Optional[Dict[str, Any]]:
        """获取实时数据"""
        try:
            response = self.session.get(f"{self.base_url}/data/realtime?count=1", timeout=3)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logger.error(f"Failed to get realtime data: {e}")
            return None

    def get_dual_roi_data(self) -> Optional[Dict[str, Any]]:
        """获取双ROI实时数据"""
        try:
            response = self.session.get(f"{self.base_url}/data/dual-realtime?count=1", timeout=3)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logger.error(f"Failed to get dual ROI data: {e}")
            return None

    def get_enhanced_realtime_data(self, include_line_intersection: bool = None) -> Optional[Dict[str, Any]]:
        """获取增强的实时数据（包含绿线交点检测数据）"""
        fetch_start_time = time.time()

        try:
            # 使用实例配置作为默认值
            if include_line_intersection is None:
                include_line_intersection = self.include_line_intersection

            # 构建请求参数
            params = {"count": 1}
            if include_line_intersection:
                params["include_line_intersection"] = "true"

            # 使用增强端点获取数据
            response = self.session.get(
                f"{self.base_url}/data/realtime/enhanced",
                params=params,
                timeout=5  # 增加超时时间以适应可能的数据处理时间
            )

            self.enhanced_fetch_count += 1

            if response.status_code == 200:
                fetch_time = time.time() - fetch_start_time
                self._update_fetch_performance(fetch_time)

                logger.debug(f"Enhanced realtime data fetched successfully in {fetch_time:.3f}s")
                return response.json()
            else:
                self.enhanced_fetch_errors += 1
                logger.warning(f"Enhanced data endpoint returned status {response.status_code}")
                return None
        except Exception as e:
            self.enhanced_fetch_errors += 1
            logger.error(f"Failed to get enhanced realtime data: {e}")
            return None

    def get_enhanced_dual_roi_data(self, include_line_intersection: bool = None) -> Optional[Dict[str, Any]]:
        """获取增强的双ROI实时数据（包含绿线交点检测数据）"""
        fetch_start_time = time.time()

        try:
            # 使用实例配置作为默认值
            if include_line_intersection is None:
                include_line_intersection = self.include_line_intersection

            # 构建请求参数
            params = {"count": 1}
            if include_line_intersection:
                params["include_line_intersection"] = "true"

            # 使用增强双ROI端点获取数据 - 注意：后端没有dual-realtime/enhanced端点
            # 使用dual-realtime端点并在客户端处理line_intersection数据
            response = self.session.get(
                f"{self.base_url}/data/dual-realtime",
                params=params,
                timeout=5  # 增加超时时间以适应可能的数据处理时间
            )

            self.enhanced_fetch_count += 1

            if response.status_code == 200:
                fetch_time = time.time() - fetch_start_time
                self._update_fetch_performance(fetch_time)

                logger.debug(f"Enhanced dual ROI data fetched successfully in {fetch_time:.3f}s")
                return response.json()
            else:
                self.enhanced_fetch_errors += 1
                logger.warning(f"Enhanced dual ROI data endpoint returned status {response.status_code}")
                return None
        except Exception as e:
            self.enhanced_fetch_errors += 1
            logger.error(f"Failed to get enhanced dual ROI data: {e}")
            return None

    def _update_fetch_performance(self, fetch_time: float):
        """更新数据获取性能统计"""
        self.last_fetch_time = fetch_time
        # 使用指数移动平均计算平均获取时间
        alpha = 0.1  # 平滑因子
        self.avg_fetch_time = alpha * fetch_time + (1 - alpha) * self.avg_fetch_time

    def _should_use_enhanced_data(self) -> bool:
        """判断是否应该使用增强数据获取"""
        # 只有在检测运行且启用增强数据时才使用增强端点
        return self.enhanced_data_enabled and self.detection_running

    def send_control_command(self, command: str) -> Optional[Dict[str, Any]]:
        """发送控制命令"""
        try:
            data = {
                "command": command,
                "password": self.password
            }
            response = self.session.post(f"{self.base_url}/control", data=data, timeout=5)

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Control command failed: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Failed to send control command: {e}")
            return None

    def start_polling(self):
        """开始数据轮询"""
        if self.polling_running:
            logger.warning("Polling is already running")
            return

        self.polling_running = True
        self.polling_thread = threading.Thread(target=self._polling_loop, daemon=True)
        self.polling_thread.start()
        logger.info("Started data polling")

    def stop_polling(self):
        """停止数据轮询"""
        if not self.polling_running:
            return

        self.polling_running = False
        if self.polling_thread and self.polling_thread.is_alive():
            self.polling_thread.join(timeout=2)

        # 停止连接时清理绿线交点检测状态
        self._handle_connection_lost()

        logger.info("Stopped data polling")

    def cleanup(self):
        """清理客户端资源"""
        try:
            logger.info("🧹 Cleaning up HTTPRealtimeClient resources...")

            # 停止数据轮询
            self.stop_polling()

            # 清理绿线交点检测状态
            self.cleanup_line_detection_state()

            logger.info("✅ HTTPRealtimeClient cleanup completed")

        except Exception as e:
            logger.error(f"❌ Error during HTTPRealtimeClient cleanup: {e}")

    def _polling_loop(self):
        """轮询循环"""
        previous_connection_state = self.connected

        while self.polling_running:
            try:
                # 检测连接状态变化
                current_connection_state = self.test_connection()
                if current_connection_state != previous_connection_state:
                    if current_connection_state and not previous_connection_state:
                        # 连接恢复
                        self._handle_connection_restored()
                        logger.info("🔄 Connection restored, recovering line detection state")
                    elif not current_connection_state and previous_connection_state:
                        # 连接丢失
                        self._handle_connection_lost()
                        logger.warning("⚠️ Connection lost, handling line detection state recovery")

                    previous_connection_state = current_connection_state

                data = None
                data_type = None
                use_enhanced = self._should_use_enhanced_data()

                # 选择数据获取方式
                if use_enhanced:
                    # 尝试使用增强数据获取
                    if self.dual_roi_mode:
                        data = self.get_enhanced_dual_roi_data()
                        data_type = "dual_realtime_data"  # 后端返回的类型是dual_realtime_data
                    else:
                        data = self.get_enhanced_realtime_data()
                        data_type = "enhanced_realtime_data"

                    # 如果增强数据获取失败且启用了回退机制，使用标准数据获取
                    if data is None and self.fallback_on_error:
                        logger.debug("Enhanced data fetch failed, falling back to standard endpoint")
                        if self.dual_roi_mode:
                            data = self.get_dual_roi_data()
                            data_type = "dual_realtime_data"
                        else:
                            data = self.get_realtime_data()
                            data_type = "realtime_data"
                else:
                    # 使用标准数据获取
                    if self.dual_roi_mode:
                        data = self.get_dual_roi_data()
                        data_type = "dual_realtime_data"
                    else:
                        data = self.get_realtime_data()
                        data_type = "realtime_data"

                if data and data.get("type") in [data_type, data_type.replace("enhanced_", "")]:
                    # 处理增强数据中的绿线交点检测结果
                    if "enhanced" in data_type and self.include_line_intersection:
                        self._process_line_intersection_data(data)

                    # 更新绘图器（确保数据格式兼容）
                    if self.plotter:
                        self.plotter.update_data(data)

                    # 处理ROI更新
                    if self.dual_roi_mode and data_type in ["dual_realtime_data", "enhanced_dual_realtime_data"] and self.roi_update_callback:
                        try:
                            self.roi_update_callback(data)
                        except Exception as e:
                            logger.error(f"Error in ROI update callback: {e}")

                    # 对于dual ROI数据，也要触发line intersection回调以传递ROI数据给LineDetectionWidget
                    if self.dual_roi_mode and data_type in ["dual_realtime_data", "enhanced_dual_realtime_data"] and self.line_intersection_callback:
                        try:
                            # 将整个dual_roi_data传递给LineDetectionWidget
                            dual_roi_data = data.get("dual_roi_data", {})
                            self.line_intersection_callback(dual_roi_data)
                        except Exception as e:
                            logger.error(f"Error in line intersection callback for ROI data: {e}")

                    self.data_count += 1
                    self.last_update_time = time.time()

                # 等待下一次轮询
                time.sleep(self.polling_interval)

            except Exception as e:
                logger.error(f"Error in polling loop: {e}")
                time.sleep(1)  # 出错时等待1秒后重试

    def _process_line_intersection_data(self, data: Dict[str, Any]):
        """处理绿线交点检测数据"""
        try:
            # 检查是否有绿线交点检测结果
            line_intersection_result = data.get("line_intersection_result")
            if line_intersection_result and self.line_intersection_callback:
                logger.debug("Processing line intersection data")
                self.line_intersection_callback(line_intersection_result)

            # 可以在这里添加额外的绿线交点数据处理逻辑
            if line_intersection_result:
                logger.debug(f"Line intersection status: {line_intersection_result.get('status', 'unknown')}")

        except Exception as e:
            logger.error(f"Error processing line intersection data: {e}")

    def start_detection(self) -> bool:
        """开始检测"""
        response = self.send_control_command("start_detection")
        if response and response.get("status") == "success":
            self.detection_running = True
            logger.info("Detection started successfully")
            return True
        else:
            logger.error("Failed to start detection")
            return False

    def stop_detection(self) -> bool:
        """停止检测"""
        response = self.send_control_command("stop_detection")
        if response and response.get("status") == "success":
            self.detection_running = False
            logger.info("Detection stopped successfully")
            return True
        else:
            logger.error("Failed to stop detection")
            return False

    def get_status(self) -> Dict[str, Any]:
        """获取客户端状态"""
        with self.line_detection_lock:
            status = {
                "connected": self.connected,
                "detection_running": self.detection_running,
                "polling_running": self.polling_running,
                "data_count": self.data_count,
                "base_url": self.base_url,
                "polling_interval": self.polling_interval,
                "dual_roi_mode": self.dual_roi_mode,
                "line_detection": {
                    "state": self.line_detection_state.value,
                    "config": self.line_detection_config.to_dict(),
                    "last_sync_time": self.last_state_sync_time,
                    "recovery_in_progress": self.state_recovery_in_progress
                },
                "enhanced_data": {
                    "enabled": self.enhanced_data_enabled,
                    "include_line_intersection": self.include_line_intersection,
                    "fallback_on_error": self.fallback_on_error,
                    "stats": self.get_enhanced_data_stats()
                }
            }
        return status

    # ================== 绿线交点检测状态管理方法 ==================

    def initialize_line_detection_state(self) -> bool:
        """初始化绿线交点检测状态"""
        try:
            with self.line_detection_lock:
                logger.info("Initializing line detection state...")

                # 加载本地状态配置
                if self._load_line_detection_state():
                    logger.info("✅ Line detection state loaded from local config")
                else:
                    logger.info("📝 Using default line detection state")

                # 如果配置了自动启动，尝试启用检测
                if self.line_detection_config.auto_start:
                    logger.info("🚀 Auto-start enabled, attempting to enable line detection...")
                    # 这里不立即启用，等待连接建立后再处理
                    pass

                # 启动状态同步线程
                self._start_state_sync_thread()

                logger.info("✅ Line detection state initialized successfully")
                return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize line detection state: {e}")
            self.set_line_detection_state(LineDetectionState.ERROR)
            return False

    def set_line_detection_state(self, new_state: LineDetectionState,
                                error_msg: str = None, notify_callbacks: bool = True) -> bool:
        """设置绿线交点检测状态"""
        try:
            with self.line_detection_lock:
                old_state = self.line_detection_state

                # 检查状态转换是否合法
                if not self._is_valid_state_transition(old_state, new_state):
                    logger.warning(f"⚠️ Invalid state transition: {old_state.value} → {new_state.value}")
                    return False

                # 记录状态变化
                logger.info(f"🔄 Line detection state transition: {old_state.value} → {new_state.value}")
                if error_msg:
                    logger.error(f"❌ State change error: {error_msg}")

                self.line_detection_state = new_state

                # 如果状态变化涉及启用/禁用，更新配置
                if new_state == LineDetectionState.ENABLED:
                    self.line_detection_config.enabled = True
                elif new_state == LineDetectionState.DISABLED:
                    self.line_detection_config.enabled = False

                # 保存状态到配置
                self._save_line_detection_state()

                # 通知回调函数
                if notify_callbacks:
                    self._notify_state_change_callbacks(old_state, new_state, error_msg)

                return True

        except Exception as e:
            logger.error(f"❌ Failed to set line detection state: {e}")
            return False

    def get_line_detection_state(self) -> LineDetectionState:
        """获取当前绿线交点检测状态"""
        with self.line_detection_lock:
            return self.line_detection_state

    def sync_line_detection_state(self) -> bool:
        """与后端同步绿线交点检测状态"""
        try:
            with self.line_detection_lock:
                if not self.connected:
                    logger.debug("Skipping state sync: not connected to server")
                    return False

                logger.debug("🔄 Syncing line detection state with backend...")

                # 查询后端状态
                backend_status = self._get_backend_line_detection_status()

                if backend_status is None:
                    logger.warning("⚠️ Failed to get backend status")
                    return False

                backend_enabled = backend_status.get('enabled', False)
                backend_state = LineDetectionState.ENABLED if backend_enabled else LineDetectionState.DISABLED

                # 根据后端状态更新本地状态
                current_state = self.line_detection_state
                if current_state not in [LineDetectionState.ENABLING, LineDetectionState.DISABLING]:
                    if backend_state != current_state:
                        logger.info(f"🔄 Syncing state with backend: {current_state.value} → {backend_state.value}")
                        self.set_line_detection_state(backend_state, notify_callbacks=False)

                self.last_state_sync_time = time.time()
                return True

        except Exception as e:
            logger.error(f"❌ Failed to sync line detection state: {e}")
            return False

    def enable_line_detection(self) -> bool:
        """启用绿线交点检测"""
        try:
            with self.line_detection_lock:
                if self.line_detection_state == LineDetectionState.ENABLED:
                    logger.info("Line detection is already enabled")
                    return True

                if self.line_detection_state in [LineDetectionState.ENABLING]:
                    logger.info("Line detection is already being enabled")
                    return True

                logger.info("🚀 Enabling line detection...")
                self.set_line_detection_state(LineDetectionState.ENABLING)

                # 发送启用请求到后端
                success = self._send_line_detection_enable_request()

                if success:
                    self.set_line_detection_state(LineDetectionState.ENABLED)
                    logger.info("✅ Line detection enabled successfully")
                    return True
                else:
                    self.set_line_detection_state(LineDetectionState.ERROR, "Failed to enable detection")
                    logger.error("❌ Failed to enable line detection")
                    return False

        except Exception as e:
            error_msg = f"Exception while enabling line detection: {str(e)}"
            self.set_line_detection_state(LineDetectionState.ERROR, error_msg)
            logger.error(f"❌ {error_msg}")
            return False

    def disable_line_detection(self) -> bool:
        """禁用绿线交点检测"""
        try:
            with self.line_detection_lock:
                if self.line_detection_state == LineDetectionState.DISABLED:
                    logger.info("Line detection is already disabled")
                    return True

                if self.line_detection_state in [LineDetectionState.DISABLING]:
                    logger.info("Line detection is already being disabled")
                    return True

                logger.info("🛑 Disabling line detection...")
                self.set_line_detection_state(LineDetectionState.DISABLING)

                # 发送禁用请求到后端
                success = self._send_line_detection_disable_request()

                if success:
                    self.set_line_detection_state(LineDetectionState.DISABLED)
                    logger.info("✅ Line detection disabled successfully")
                    return True
                else:
                    self.set_line_detection_state(LineDetectionState.ERROR, "Failed to disable detection")
                    logger.error("❌ Failed to disable line detection")
                    return False

        except Exception as e:
            error_msg = f"Exception while disabling line detection: {str(e)}"
            self.set_line_detection_state(LineDetectionState.ERROR, error_msg)
            logger.error(f"❌ {error_msg}")
            return False

    def add_line_detection_state_callback(self, callback: callable):
        """添加状态变化回调函数"""
        with self.line_detection_lock:
            if callback not in self.line_detection_state_callbacks:
                self.line_detection_state_callbacks.append(callback)
                logger.debug(f"Added line detection state callback: {callback}")

    def remove_line_detection_state_callback(self, callback: callable):
        """移除状态变化回调函数"""
        with self.line_detection_lock:
            if callback in self.line_detection_state_callbacks:
                self.line_detection_state_callbacks.remove(callback)
                logger.debug(f"Removed line detection state callback: {callback}")

    def cleanup_line_detection_state(self):
        """清理绿线交点检测状态管理资源"""
        try:
            with self.line_detection_lock:
                logger.info("🧹 Cleaning up line detection state management...")

                # 停止状态同步线程
                self._stop_state_sync_thread()

                # 如果检测正在运行，尝试禁用
                if self.line_detection_state == LineDetectionState.ENABLED:
                    try:
                        self._send_line_detection_disable_request()
                    except Exception as e:
                        logger.warning(f"Failed to disable detection during cleanup: {e}")

                # 保存最终状态
                self._save_line_detection_state()

                # 清理回调函数
                self.line_detection_state_callbacks.clear()

                # 重置状态
                self.line_detection_state = LineDetectionState.DISABLED
                self.state_recovery_in_progress = False

                logger.info("✅ Line detection state management cleaned up successfully")

        except Exception as e:
            logger.error(f"❌ Error during line detection state cleanup: {e}")

    # ================== 私有状态管理方法 ==================

    def _is_valid_state_transition(self, old_state: LineDetectionState, new_state: LineDetectionState) -> bool:
        """检查状态转换是否合法"""
        valid_transitions = {
            LineDetectionState.DISABLED: [LineDetectionState.ENABLING, LineDetectionState.ERROR],
            LineDetectionState.ENABLING: [LineDetectionState.ENABLED, LineDetectionState.DISABLED, LineDetectionState.ERROR],
            LineDetectionState.ENABLED: [LineDetectionState.DISABLING, LineDetectionState.ERROR],
            LineDetectionState.DISABLING: [LineDetectionState.DISABLED, LineDetectionState.ENABLED, LineDetectionState.ERROR],
            LineDetectionState.ERROR: [LineDetectionState.DISABLED, LineDetectionState.ENABLING]
        }
        return new_state in valid_transitions.get(old_state, [])

    def _get_backend_line_detection_status(self) -> Optional[Dict[str, Any]]:
        """获取后端绿线交点检测状态"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/roi/line-intersection/status",
                timeout=self.line_detection_config.timeout
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Backend status request failed: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Failed to get backend line detection status: {e}")
            return None

    def _send_line_detection_enable_request(self) -> bool:
        """发送启用绿线交点检测请求"""
        try:
            for attempt in range(self.line_detection_config.retry_count):
                try:
                    response = self.session.post(
                        f"{self.base_url}/api/roi/line-intersection/enable",
                        data={"password": self.password},
                        timeout=self.line_detection_config.timeout
                    )

                    if response.status_code == 200:
                        result = response.json()
                        if result.get('success', True):
                            return True
                        else:
                            error_msg = result.get('error', 'Unknown error')
                            logger.warning(f"Enable request failed: {error_msg}")

                    logger.warning(f"Enable request attempt {attempt + 1} failed: {response.status_code}")

                except Exception as e:
                    logger.warning(f"Enable request attempt {attempt + 1} exception: {e}")

                if attempt < self.line_detection_config.retry_count - 1:
                    time.sleep(self.line_detection_config.retry_delay)

            return False

        except Exception as e:
            logger.error(f"Exception in enable request: {e}")
            return False

    def _send_line_detection_disable_request(self) -> bool:
        """发送禁用绿线交点检测请求"""
        try:
            for attempt in range(self.line_detection_config.retry_count):
                try:
                    response = self.session.post(
                        f"{self.base_url}/api/roi/line-intersection/disable",
                        data={"password": self.password},
                        timeout=self.line_detection_config.timeout
                    )

                    if response.status_code == 200:
                        result = response.json()
                        if result.get('success', True):
                            return True
                        else:
                            error_msg = result.get('error', 'Unknown error')
                            logger.warning(f"Disable request failed: {error_msg}")

                    logger.warning(f"Disable request attempt {attempt + 1} failed: {response.status_code}")

                except Exception as e:
                    logger.warning(f"Disable request attempt {attempt + 1} exception: {e}")

                if attempt < self.line_detection_config.retry_count - 1:
                    time.sleep(self.line_detection_config.retry_delay)

            return False

        except Exception as e:
            logger.error(f"Exception in disable request: {e}")
            return False

    def _notify_state_change_callbacks(self, old_state: LineDetectionState,
                                     new_state: LineDetectionState, error_msg: str = None):
        """通知状态变化回调函数"""
        try:
            for callback in self.line_detection_state_callbacks:
                try:
                    callback(old_state, new_state, error_msg)
                except Exception as e:
                    logger.error(f"Error in state change callback: {e}")
        except Exception as e:
            logger.error(f"Error notifying state change callbacks: {e}")

    def _save_line_detection_state(self) -> bool:
        """保存绿线交点检测状态到本地配置"""
        try:
            config_file = "line_detection_state.json"

            state_data = {
                "state": self.line_detection_state.value,
                "config": self.line_detection_config.to_dict(),
                "last_saved": time.time()
            }

            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2, ensure_ascii=False)

            logger.debug("Line detection state saved to local config")
            return True

        except Exception as e:
            logger.error(f"Failed to save line detection state: {e}")
            return False

    def _load_line_detection_state(self) -> bool:
        """从本地配置加载绿线交点检测状态"""
        try:
            config_file = "line_detection_state.json"

            if not os.path.exists(config_file):
                logger.debug("No local line detection state config found")
                return False

            with open(config_file, 'r', encoding='utf-8') as f:
                state_data = json.load(f)

            # 加载状态
            state_str = state_data.get('state', 'disabled')
            self.line_detection_state = LineDetectionState(state_str)

            # 加载配置
            config_data = state_data.get('config', {})
            self.line_detection_config = LineDetectionConfig.from_dict(config_data)

            saved_time = state_data.get('last_saved', 0)
            logger.debug(f"Line detection state loaded from local config (saved: {time.ctime(saved_time)})")
            return True

        except Exception as e:
            logger.error(f"Failed to load line detection state: {e}")
            return False

    def _start_state_sync_thread(self):
        """启动状态同步线程"""
        if self.state_sync_running:
            logger.debug("State sync thread is already running")
            return

        self.state_sync_running = True
        self.state_sync_thread = threading.Thread(target=self._state_sync_loop, daemon=True)
        self.state_sync_thread.start()
        logger.debug("State sync thread started")

    def _stop_state_sync_thread(self):
        """停止状态同步线程"""
        if not self.state_sync_running:
            return

        self.state_sync_running = False

        if self.state_sync_thread and self.state_sync_thread.is_alive():
            self.state_sync_thread.join(timeout=2)

        logger.debug("State sync thread stopped")

    def _state_sync_loop(self):
        """状态同步循环"""
        while self.state_sync_running:
            try:
                # 检查是否需要同步
                current_time = time.time()
                time_since_last_sync = current_time - self.last_state_sync_time

                if time_since_last_sync >= self.line_detection_config.sync_interval:
                    self.sync_line_detection_state()

                # 睡眠一小段时间
                time.sleep(1.0)

            except Exception as e:
                logger.error(f"Error in state sync loop: {e}")
                time.sleep(5.0)  # 出错时等待更长时间

    def _handle_connection_lost(self):
        """处理连接丢失事件"""
        try:
            with self.line_detection_lock:
                if not self.line_detection_config.auto_recovery:
                    logger.info("Auto-recovery disabled for line detection")
                    return

                if self.state_recovery_in_progress:
                    logger.debug("State recovery already in progress")
                    return

                logger.info("🔄 Connection lost, starting line detection state recovery...")
                self.state_recovery_in_progress = True

                # 将状态设为错误，等待连接恢复后处理
                if self.line_detection_state in [LineDetectionState.ENABLED, LineDetectionState.ENABLING]:
                    self.set_line_detection_state(LineDetectionState.ERROR, "Connection lost")

        except Exception as e:
            logger.error(f"Error handling connection lost: {e}")

    def _handle_connection_restored(self):
        """处理连接恢复事件"""
        try:
            with self.line_detection_lock:
                if not self.state_recovery_in_progress:
                    logger.debug("No state recovery needed")
                    return

                logger.info("🔄 Connection restored, recovering line detection state...")

                # 如果之前是启用状态，尝试恢复
                if self.line_detection_config.enabled:
                    logger.info("🔄 Attempting to recover line detection...")
                    success = self.enable_line_detection()

                    if success:
                        logger.info("✅ Line detection state recovered successfully")
                    else:
                        logger.warning("⚠️ Line detection state recovery failed")
                else:
                    logger.info("🔄 Syncing line detection state with backend...")
                    self.sync_line_detection_state()

                self.state_recovery_in_progress = False

        except Exception as e:
            logger.error(f"Error handling connection restored: {e}")
            self.state_recovery_in_progress = False

    def _load_line_detection_config(self):
        """加载绿线检测配置"""
        try:
            logger.info("🔄 正在加载绿线检测配置...")

            # 检查配置管理器是否可用
            if self.line_detection_config_manager is None:
                logger.error("❌ 绿线检测配置管理器未初始化")
                return False

            # 加载配置
            success, message, config_data = self.line_detection_config_manager.load_config()

            if success:
                self.line_detection_config_loaded = True
                line_detection_config = self.line_detection_config_manager.get_line_detection_config()

                # 更新绿线检测配置对象
                self.line_detection_config.enabled = line_detection_config.get("enabled", False)
                self.line_detection_config.auto_start = line_detection_config.get("auto_start", False)
                self.line_detection_config.auto_recovery = line_detection_config.get("auto_recovery", True)
                self.line_detection_config.sync_interval = line_detection_config.get("sync_interval", 5.0)
                self.line_detection_config.timeout = line_detection_config.get("timeout", 10.0)
                self.line_detection_config.retry_count = line_detection_config.get("retry_count", 3)
                self.line_detection_config.retry_delay = line_detection_config.get("retry_delay", 1.0)

                logger.info("✅ 绿线检测配置加载完成")
                return True

            else:
                logger.warning(f"⚠️ 绿线检测配置加载失败: {message}")
                return False

        except Exception as e:
            logger.error(f"❌ 绿线检测配置加载异常: {str(e)}")
            return False


class HTTPRealtimeClientUI(tk.Tk):
    """基于HTTP的Python客户端UI"""

    def __init__(self):
        super().__init__()
        self.title("NHEM Python Client - HTTP + Real-time Plotting")
        self.geometry("1200x800")

        # HTTP客户端
        self.http_client: HTTPRealtimeClient = None

        # 状态变量
        self.connected = False

        # UI模式状态
        self.compact_mode = False
        self.normal_geometry = "1200x800"
        self.compact_geometry = "900x500"

        # UI组件引用
        self.conn_frame = None
        self.info_frame = None
        self.btn_clear = None
        self.btn_save = None
        self.btn_capture = None

        # ROI图像缓存
        self._last_image = None

        # Line Detection Widget
        self.line_detection_widget = None
        self.show_line_detection = True  # Configuration option for show/hide

        # 构建UI
        self._build_widgets()
        self._setup_plotter()

        # 加载绿线检测配置
        if self.http_client and hasattr(self.http_client, '_load_line_detection_config'):
            try:
                self.http_client._load_line_detection_config()
            except Exception as e:
                self._log(f"绿线检测配置加载失败: {str(e)}", "WARNING")
        else:
            self._log("绿线检测配置管理器不可用，跳过配置加载", "INFO")

        # 绑定关闭事件
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

        # 启动状态更新循环
        self._start_status_update()

    def _build_widgets(self):
        """构建UI组件"""
        # 顶部连接配置
        self.conn_frame = ttk.LabelFrame(self, text="HTTP连接配置")
        self.conn_frame.pack(fill="x", padx=8, pady=4)

        ttk.Label(self.conn_frame, text="后端URL:").grid(row=0, column=0, sticky="e", padx=4, pady=2)
        self.entry_base_url = ttk.Entry(self.conn_frame, width=40)
        self.entry_base_url.grid(row=0, column=1, sticky="w", padx=4, pady=2)
        self.entry_base_url.insert(0, "http://localhost:8421")

        ttk.Label(self.conn_frame, text="密码:").grid(row=0, column=2, sticky="e", padx=4, pady=2)
        self.entry_password = ttk.Entry(self.conn_frame, width=12, show="*")
        self.entry_password.grid(row=0, column=3, sticky="w", padx=4, pady=2)
        self.entry_password.insert(0, "31415")

        # 连接按钮
        self.btn_connect = ttk.Button(self.conn_frame, text="连接", command=self._toggle_connection)
        self.btn_connect.grid(row=0, column=4, padx=8, pady=2)

        # 连接状态指示器
        self.status_var = tk.StringVar(value="未连接")
        self.status_label = ttk.Label(self.conn_frame, textvariable=self.status_var, foreground="red")
        self.status_label.grid(row=0, column=5, padx=4, pady=2)

        # 控制面板
        control_frame = ttk.LabelFrame(self, text="控制面板")
        control_frame.pack(fill="x", padx=8, pady=4)

        # 核心按钮（始终显示）
        self.btn_start = ttk.Button(control_frame, text="开始检测", command=self._start_detection, state="disabled")
        self.btn_start.pack(side="left", padx=8, pady=4)

        self.btn_stop = ttk.Button(control_frame, text="停止检测", command=self._stop_detection, state="disabled")
        self.btn_stop.pack(side="left", padx=8, pady=4)

        # UI模式切换按钮
        self.btn_ui_toggle = ttk.Button(control_frame, text="缩小", command=self._toggle_ui_mode)
        self.btn_ui_toggle.pack(side="right", padx=8, pady=4)

        # 绿线检测切换按钮
        self.btn_line_detection_toggle = ttk.Button(control_frame, text="启用检测", command=self._toggle_line_detection)
        self.btn_line_detection_toggle.pack(side="right", padx=8, pady=4)

        # 附加按钮（在紧凑模式下隐藏）
        self.btn_clear = ttk.Button(control_frame, text="清除数据", command=self._clear_data, state="disabled")
        self.btn_clear.pack(side="left", padx=8, pady=4)

        self.btn_save = ttk.Button(control_frame, text="保存截图", command=self._save_screenshot, state="disabled")
        self.btn_save.pack(side="left", padx=8, pady=4)

        self.btn_capture = ttk.Button(control_frame, text="截取曲线", command=self._capture_curve, state="disabled")
        self.btn_capture.pack(side="left", padx=8, pady=4)

        # 主框架 - 使用Notebook创建标签页界面
        main_frame = ttk.Frame(self)
        main_frame.pack(fill="both", expand=True, padx=8, pady=4)

        # 创建Notebook用于标签页
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill="both", expand=True)

        # 标签页1: 实时监控 (原有功能)
        self.monitoring_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.monitoring_frame, text="实时监控")

        # 标签页2: 绿线交点检测 (LineDetectionWidget)
        if self.show_line_detection:
            self.line_detection_frame = ttk.Frame(self.notebook)
            self.notebook.add(self.line_detection_frame, text="绿线交点检测")

        # 在监控标签页中构建原有布局
        # 左侧信息面板
        self.info_frame = ttk.LabelFrame(self.monitoring_frame, text="实时信息")
        self.info_frame.pack(side="left", fill="y", padx=(0, 8))

        # 状态信息
        status_info = ttk.Frame(self.info_frame)
        status_info.pack(fill="x", padx=8, pady=4)

        ttk.Label(status_info, text="数据点数:").grid(row=0, column=0, sticky="w", pady=2)
        self.data_count_label = ttk.Label(status_info, text="0")
        self.data_count_label.grid(row=0, column=1, sticky="w", padx=(8, 0), pady=2)

        ttk.Label(status_info, text="更新FPS:").grid(row=1, column=0, sticky="w", pady=2)
        self.fps_label = ttk.Label(status_info, text="0")
        self.fps_label.grid(row=1, column=1, sticky="w", padx=(8, 0), pady=2)

        ttk.Label(status_info, text="检测状态:").grid(row=2, column=0, sticky="w", pady=2)
        self.detection_status_label = ttk.Label(status_info, text="未运行")
        self.detection_status_label.grid(row=2, column=1, sticky="w", padx=(8, 0), pady=2)

        ttk.Label(status_info, text="连接状态:").grid(row=3, column=0, sticky="w", pady=2)
        self.connection_status_label = ttk.Label(status_info, text="未连接")
        self.connection_status_label.grid(row=3, column=1, sticky="w", padx=(8, 0), pady=2)

        ttk.Label(status_info, text="轮询状态:").grid(row=4, column=0, sticky="w", pady=2)
        self.polling_status_label = ttk.Label(status_info, text="未轮询")
        self.polling_status_label.grid(row=4, column=1, sticky="w", padx=(8, 0), pady=2)

        # 分隔线
        ttk.Separator(self.info_frame, orient="horizontal").pack(fill="x", pady=8)

        # 参数设置面板
        config_frame = ttk.LabelFrame(self.info_frame, text="参数设置")
        config_frame.pack(fill="x", padx=8, pady=4)

        # ROI设置子面板
        roi_config_frame = ttk.LabelFrame(config_frame, text="ROI配置")
        roi_config_frame.pack(fill="x", padx=8, pady=4)

        # ROI坐标设置
        roi_coords = ttk.Frame(roi_config_frame)
        roi_coords.pack(fill="x", padx=8, pady=2)

        ttk.Label(roi_coords, text="X1:").grid(row=0, column=0, sticky="w")
        self.roi_x1_var = tk.StringVar(value="0")
        ttk.Entry(roi_coords, textvariable=self.roi_x1_var, width=8).grid(row=0, column=1, padx=2)

        ttk.Label(roi_coords, text="Y1:").grid(row=0, column=2, sticky="w", padx=(10,0))
        self.roi_y1_var = tk.StringVar(value="0")
        ttk.Entry(roi_coords, textvariable=self.roi_y1_var, width=8).grid(row=0, column=3, padx=2)

        ttk.Label(roi_coords, text="X2:").grid(row=1, column=0, sticky="w")
        self.roi_x2_var = tk.StringVar(value="200")
        ttk.Entry(roi_coords, textvariable=self.roi_x2_var, width=8).grid(row=1, column=1, padx=2)

        ttk.Label(roi_coords, text="Y2:").grid(row=1, column=2, sticky="w", padx=(10,0))
        self.roi_y2_var = tk.StringVar(value="150")
        ttk.Entry(roi_coords, textvariable=self.roi_y2_var, width=8).grid(row=1, column=3, padx=2)

        # ROI帧率设置
        roi_fps_frame = ttk.Frame(roi_config_frame)
        roi_fps_frame.pack(fill="x", padx=8, pady=2)

        ttk.Label(roi_fps_frame, text="ROI帧率:").pack(side="left")
        self.roi_fps_var = tk.StringVar(value="2")
        fps_spinbox = ttk.Spinbox(roi_fps_frame, from_=1, to=60, textvariable=self.roi_fps_var, width=8)
        fps_spinbox.pack(side="left", padx=(8, 4))
        ttk.Label(roi_fps_frame, text="FPS").pack(side="left")

        # 波峰检测设置子面板
        peak_config_frame = ttk.LabelFrame(config_frame, text="波峰检测设置")
        peak_config_frame.pack(fill="x", padx=8, pady=4)

        # 绝对阈值
        threshold_frame = ttk.Frame(peak_config_frame)
        threshold_frame.pack(fill="x", padx=8, pady=2)

        ttk.Label(threshold_frame, text="绝对阈值:").pack(side="left")
        self.peak_threshold_var = tk.StringVar(value="105.0")
        ttk.Entry(threshold_frame, textvariable=self.peak_threshold_var, width=10).pack(side="left", padx=(8, 4))
        ttk.Label(threshold_frame, text="灰度值").pack(side="left")

        # 边界帧数
        margin_frame = ttk.Frame(peak_config_frame)
        margin_frame.pack(fill="x", padx=8, pady=2)

        ttk.Label(margin_frame, text="边界帧数:").pack(side="left")
        self.peak_margin_var = tk.StringVar(value="5")
        ttk.Spinbox(margin_frame, from_=1, to=20, textvariable=self.peak_margin_var, width=8).pack(side="left", padx=(8, 4))
        ttk.Label(margin_frame, text="帧").pack(side="left")

        # 差值阈值
        diff_frame = ttk.Frame(peak_config_frame)
        diff_frame.pack(fill="x", padx=8, pady=2)

        ttk.Label(diff_frame, text="差值阈值:").pack(side="left")
        self.peak_diff_var = tk.StringVar(value="2.1")
        ttk.Entry(diff_frame, textvariable=self.peak_diff_var, width=10).pack(side="left", padx=(8, 4))

        # 应用配置按钮
        config_buttons = ttk.Frame(config_frame)
        config_buttons.pack(fill="x", padx=8, pady=4)

        ttk.Button(config_buttons, text="应用ROI配置", command=self._apply_roi_config).pack(side="left", padx=4)
        ttk.Button(config_buttons, text="应用波峰配置", command=self._apply_peak_config).pack(side="left", padx=4)
        ttk.Button(config_buttons, text="保存配置", command=self._save_config).pack(side="left", padx=4)
        ttk.Button(config_buttons, text="加载配置", command=self._load_config).pack(side="left", padx=4)

        # 绿线检测配置管理按钮
        ttk.Separator(config_buttons, orient="vertical").pack(side="left", fill="y", padx=4)
        ttk.Button(config_buttons, text="备份绿线配置", command=self._backup_line_detection_config).pack(side="left", padx=4)
        ttk.Button(config_buttons, text="导出绿线配置", command=self._export_line_detection_config_dialog).pack(side="left", padx=4)
        ttk.Button(config_buttons, text="重载绿线配置", command=self._reload_line_detection_config).pack(side="left", padx=4)

        # ROI截图显示面板
        roi_frame = ttk.LabelFrame(self.info_frame, text="ROI Screenshot")
        roi_frame.pack(fill="x", padx=8, pady=4)

        # 创建ROI双显示容器
        roi_container = ttk.Frame(roi_frame)
        roi_container.pack(fill="x", pady=4)

        # 左侧ROI显示
        self.roi_label_left = ttk.Label(roi_container, text="Waiting for ROI data...",
                                        relief="sunken", background="white")
        self.roi_label_left.pack(side="left", fill="both", expand=True, padx=(0, 2))

        # 分隔符
        separator_label = ttk.Label(roi_container, text="|",
                                   font=("Arial", 16, "bold"),
                                   foreground="gray")
        separator_label.pack(side="left", padx=2)

        # 右侧ROI显示
        self.roi_label_right = ttk.Label(roi_container, text="Waiting for ROI data...",
                                         relief="sunken", background="white")
        self.roi_label_right.pack(side="left", fill="both", expand=True, padx=(2, 0))

        # 保持对原始标签的引用（向后兼容）
        self.roi_label = self.roi_label_left

        # ROI信息
        roi_info = ttk.Frame(roi_frame)
        roi_info.pack(fill="x", padx=4, pady=2)

        ttk.Label(roi_info, text="分辨率:").pack(side="left")
        self.roi_resolution_label = ttk.Label(roi_info, text="N/A")
        self.roi_resolution_label.pack(side="left", padx=(8, 16))

        ttk.Label(roi_info, text="灰度值:").pack(side="left")
        self.roi_gray_value_label = ttk.Label(roi_info, text="N/A")
        self.roi_gray_value_label.pack(side="left", padx=(8, 16))

        # 日志面板
        log_frame = ttk.LabelFrame(self.info_frame, text="日志")
        log_frame.pack(fill="both", expand=True, padx=8, pady=4)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, width=40)
        self.log_text.pack(fill="both", expand=True, padx=4, pady=4)

        # 右侧图表区域 (在监控标签页内)
        right_frame = ttk.Frame(self.monitoring_frame)
        right_frame.pack(side="right", fill="both", expand=True)

        # 上方截取曲线显示框架 - 放在实时图表上方
        captured_frame = ttk.LabelFrame(right_frame, text="Captured Curve")
        captured_frame.pack(fill="both", expand=False, pady=(0, 4))
        # 设置截取曲线框架的固定高度
        captured_frame.configure(height=300)

        # 创建截取曲线显示区域 - 使用Frame包装以支持高度设置
        self.captured_wrapper = ttk.Frame(captured_frame)
        self.captured_wrapper.pack(fill="both", expand=True, padx=4, pady=4)
        self.captured_wrapper.pack_propagate(False)  # 防止子组件改变父容器大小
        self.captured_wrapper.configure(height=300)  # 设置最小高度

        # 在Frame内部创建Label
        self.captured_label = ttk.Label(self.captured_wrapper, text="No captured curve yet. Click '截取曲线' to capture data.",
                                      relief="sunken", background="white")
        self.captured_label.pack(fill="both", expand=True)

        # 下方实时图表框架
        plot_frame = ttk.LabelFrame(right_frame, text="Real-time Charts")
        plot_frame.pack(fill="both", expand=True, pady=(4, 0))

        self.plot_frame = plot_frame

        # 截取信息
        capture_info = ttk.Frame(captured_frame)
        capture_info.pack(fill="x", padx=4, pady=2)

        ttk.Label(capture_info, text="数据点数:").pack(side="left")
        self.captured_count_label = ttk.Label(capture_info, text="N/A")
        self.captured_count_label.pack(side="left", padx=(8, 16))

        ttk.Label(capture_info, text="数据源:").pack(side="left")
        self.captured_source_label = ttk.Label(capture_info, text="N/A")
        self.captured_source_label.pack(side="left", padx=(8, 16))

        # 清除截取按钮
        self.btn_clear_capture = ttk.Button(capture_info, text="清除截取", command=self._clear_capture, state="disabled")
        self.btn_clear_capture.pack(side="right", padx=4)

    def _setup_plotter(self):
        """设置绘图器"""
        try:
            import matplotlib.pyplot as plt
            self.plotter = RealtimePlotter(master=self.plot_frame, figsize=(10, 6))
            self.plotter.setup_plot()
            self.plotter.setup_canvas()

            # 启动动画
            self.plotter.start_animation(interval=50)  # 20 FPS

            # 自动启动连接和数据收集
            self.after(1000, self.auto_connect_and_start)

        except ImportError:
            no_mpl_label = ttk.Label(self.plot_frame, text="matplotlib未安装，无法显示图表")
            no_mpl_label.pack(expand=True)
            self.plotter = None

        # 设置LineDetectionWidget
        self._setup_line_detection_widget()

    def _setup_line_detection_widget(self):
        """设置LineDetectionWidget"""
        try:
            if self.show_line_detection and hasattr(self, 'line_detection_frame'):
                # LineDetectionWidget配置
                line_detection_config = {
                    'figure_size': (12, 8),
                    'update_interval': 100,  # 100ms更新间隔
                    'enable_toolbar': True,
                    'enable_interactive': True,
                    'initial_view_mode': 'full'  # 'full', 'roi_only', 'zoom'
                }

                # 创建LineDetectionWidget实例
                self.line_detection_widget = LineDetectionWidget(
                    self.line_detection_frame,
                    config=line_detection_config
                )

                # 设置ROI数据回调 - 这是关键！
                if self.http_client:
                    self.http_client.set_line_intersection_callback(
                        self._handle_line_intersection_update
                    )
                    print("HTTP_CLIENT_DEBUG: Set line_intersection callback for LineDetectionWidget")

                # 打包LineDetectionWidget
                self.line_detection_widget.pack(fill="both", expand=True, padx=8, pady=8)

                self._log("✅ LineDetectionWidget初始化成功")
                logger.info("LineDetectionWidget initialized successfully")

            else:
                logger.info("LineDetectionWidget disabled in configuration")

        except Exception as e:
            error_msg = f"LineDetectionWidget初始化失败: {str(e)}"
            self._log(error_msg, "ERROR")
            logger.error(f"Failed to initialize LineDetectionWidget: {e}")

            # 显示错误信息在LineDetection框架中
            if hasattr(self, 'line_detection_frame'):
                error_label = ttk.Label(
                    self.line_detection_frame,
                    text=f"绿线交点检测组件初始化失败:\n{str(e)}",
                    foreground="red",
                    justify="center"
                )
                error_label.pack(expand=True)

    def auto_connect_and_start(self):
        """自动连接并启动数据收集"""
        try:
            # 更新状态显示
            self.status_var.set("Connecting...")
            self.status_label.config(foreground="blue")
            self._log("Auto-connecting to server...")

            # 首先加载本地配置（无需服务器连接）
            self._log("🔄 自动加载本地配置文件...")
            local_config_loaded = self._load_local_config()

            # 使用输入框中的URL和密码
            base_url = self.entry_base_url.get()
            password = self.entry_password.get()

            # 创建HTTP客户端
            self.http_client = HTTPRealtimeClient(base_url=base_url, password=password)

            # 加载绿线检测配置
            if hasattr(self.http_client, '_load_line_detection_config'):
                try:
                    self.http_client._load_line_detection_config()
                except Exception as e:
                    self._log(f"绿线检测配置加载失败: {str(e)}", "WARNING")

            # 注册绿线交点检测状态变化回调
            self.http_client.add_line_detection_state_callback(self._on_line_detection_state_changed)

            # 设置绿线交点检测回调
            if hasattr(self, 'line_detection_widget') and self.line_detection_widget:
                self.http_client.set_line_intersection_callback(
                    self._handle_line_intersection_update
                )

            # 应用本地配置中的增强数据设置
            self._apply_enhanced_data_from_client_config()

            # 测试连接
            if self.http_client.test_connection():
                self.connected = True
                self._update_connection_status()
                self._log("Auto-connection successful!")

                # 如果本地配置加载失败，尝试从服务器加载配置
                if not local_config_loaded:
                    self._log("🔄 本地配置加载失败，尝试从服务器加载配置...")
                    config_loaded = self._auto_load_config()
                    if config_loaded:
                        self._log("✅ 服务器配置加载成功，将应用配置参数")
                    else:
                        self._log("⚠️ 服务器配置加载失败，使用默认值")
                else:
                    self._log("✅ 本地配置加载成功，已应用到UI界面")

                # 配置ROI（使用当前UI中的值或默认值）
                self._log("Configuring ROI...")
                session = self.http_client.session

                # 从UI获取ROI参数
                try:
                    roi_x1 = int(self.roi_x1_var.get())
                    roi_y1 = int(self.roi_y1_var.get())
                    roi_x2 = int(self.roi_x2_var.get())
                    roi_y2 = int(self.roi_y2_var.get())
                except ValueError:
                    # 如果UI值无效，使用默认值
                    roi_x1, roi_y1 = 0, 0
                    roi_x2, roi_y2 = 200, 150

                roi_data = {
                    "x1": roi_x1,
                    "y1": roi_y1,
                    "x2": roi_x2,
                    "y2": roi_y2,
                    "password": password
                }
                response = session.post(f"{self.http_client.base_url}/roi/config", data=roi_data, timeout=5)

                if response.status_code == 200:
                    self._log(f"ROI configuration successful: ({roi_x1}, {roi_y1}) → ({roi_x2}, {roi_y2})")
                else:
                    self._log(f"ROI configuration failed: {response.status_code}")

                # 启动检测
                self._log("Starting detection...")
                if self.http_client.start_detection():
                    self._log("Detection started successfully!")

                    # 启动数据轮询
                    self.http_client.start_polling()

                    # 设置绘图器到HTTP客户端
                    self.http_client.plotter = self.plotter

                    # 启动ROI截图更新
                    self.after(2000, self.start_roi_updates)  # 2秒后开始更新ROI截图

                    # 更新按钮状态
                    self.btn_connect.config(text="Disconnect")
                    self._update_detection_status()

                    self._log("Auto-setup complete! Data collection started.")
                    self._log("ROI screenshot updates started (2 FPS).")

                else:
                    self._log("Failed to start detection")

            else:
                raise Exception("Server connection failed")

        except Exception as e:
            self._log(f"Auto-connection failed: {str(e)}", "ERROR")
            self.status_var.set("Auto-connect failed")
            self.status_label.config(foreground="red")

    def _toggle_connection(self):
        """切换连接状态"""
        if not self.connected:
            self._connect()
        else:
            self._disconnect()

    def _connect(self):
        """连接到服务器"""
        try:
            base_url = self.entry_base_url.get()
            password = self.entry_password.get()

            # 创建HTTP客户端
            self.http_client = HTTPRealtimeClient(base_url=base_url, password=password)

            # 加载绿线检测配置
            if hasattr(self.http_client, '_load_line_detection_config'):
                try:
                    self.http_client._load_line_detection_config()
                except Exception as e:
                    self._log(f"绿线检测配置加载失败: {str(e)}", "WARNING")

            # 注册绿线交点检测状态变化回调
            self.http_client.add_line_detection_state_callback(self._on_line_detection_state_changed)

            # 设置绿线交点检测回调
            if hasattr(self, 'line_detection_widget') and self.line_detection_widget:
                self.http_client.set_line_intersection_callback(
                    self._handle_line_intersection_update
                )

            # 应用本地配置中的增强数据设置
            self._apply_enhanced_data_from_client_config()

            # 测试连接
            if self.http_client.test_connection():
                self.connected = True
                self._update_connection_status()

                # 启动数据轮询
                self.http_client.start_polling()

                self._log("连接成功！")
                messagebox.showinfo("连接成功", "已连接到NHEM服务器")
            else:
                raise Exception("服务器连接测试失败")

        except Exception as e:
            messagebox.showerror("连接错误", f"连接失败: {str(e)}")
            self._log(f"连接失败: {str(e)}", "ERROR")

    def _disconnect(self):
        """断开连接"""
        if self.http_client:
            self.http_client.stop_polling()
            # 清理绿线交点检测状态管理
            self.http_client.cleanup_line_detection_state()
            self.http_client = None

        self.connected = False
        self._update_connection_status()

    def _update_connection_status(self):
        """更新连接状态显示"""
        if self.connected:
            self.status_var.set("已连接")
            self.status_label.config(foreground="green")
            self.connection_status_label.config(text="已连接", foreground="green")
            self.polling_status_label.config(text="轮询中", foreground="blue")
            self.btn_connect.config(text="断开连接", state="normal")
            self.btn_start.config(state="normal")
            self.btn_clear.config(state="normal")
            self.btn_save.config(state="normal" if self.plotter else "disabled")
            self.btn_capture.config(state="normal")
            self.btn_clear_capture.config(state="normal")
        else:
            self.status_var.set("未连接")
            self.status_label.config(foreground="red")
            self.connection_status_label.config(text="未连接", foreground="red")
            self.polling_status_label.config(text="未轮询", foreground="red")
            self.btn_connect.config(text="连接", state="normal")
            self.btn_start.config(state="disabled")
            self.btn_stop.config(state="disabled")
            self.btn_clear.config(state="disabled")
            self.btn_save.config(state="disabled")
            self.btn_capture.config(state="disabled")
            self.btn_clear_capture.config(state="disabled")

    def _start_detection(self):
        """开始检测"""
        if self.http_client:
            if self.http_client.start_detection():
                self._update_detection_status()
                self._log("开始检测命令发送成功")
            else:
                messagebox.showerror("错误", "开始检测失败")
                self._log("开始检测失败", "ERROR")

    def _stop_detection(self):
        """停止检测"""
        if self.http_client:
            if self.http_client.stop_detection():
                self._update_detection_status()
                self._log("停止检测命令发送成功")
            else:
                messagebox.showerror("错误", "停止检测失败")
                self._log("停止检测失败", "ERROR")

    def _update_detection_status(self):
        """更新检测状态"""
        if self.http_client and self.http_client.detection_running:
            self.detection_status_label.config(text="运行中", foreground="green")
            self.btn_start.config(state="disabled")
            self.btn_stop.config(state="normal")
        else:
            self.detection_status_label.config(text="未运行", foreground="red")
            self.btn_start.config(state="normal")
            self.btn_stop.config(state="disabled")

    def _clear_data(self):
        """清除数据"""
        if self.plotter:
            self.plotter.clear_data()
            if self.http_client:
                self.http_client.data_count = 0
                self.data_count_label.config(text="0")
                self.fps_label.config(text="0")
            self._log("数据已清除")

    def _save_screenshot(self):
        """保存截图"""
        if self.plotter:
            from tkinter import filedialog
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
            )
            if filename:
                self.plotter.save_screenshot(filename)
                self._log(f"截图已保存: {filename}")
                messagebox.showinfo("成功", f"截图已保存到: {filename}")

    def _start_status_update(self):
        """启动状态更新循环"""
        def update_status():
            try:
                if self.connected and self.http_client:
                    # 更新信息显示
                    self.data_count_label.config(text=str(self.http_client.data_count))

                    # 更新检测状态
                    self._update_detection_status()

                    # 更新FPS（如果有绘图器）
                    if self.plotter:
                        stats = self.plotter.get_statistics()
                        self.fps_label.config(text=f"{stats['fps']:.1f}")

                # 每秒更新一次
                self.after(1000, update_status)
            except Exception as e:
                self._log(f"状态更新错误: {str(e)}", "ERROR")
                self.after(5000, update_status)  # 出错时5秒后重试

        self.after(1000, update_status)

    def start_roi_updates(self):
        """开始ROI截图更新"""
        if self.connected and self.http_client:
            # 设置ROI更新回调
            self.http_client.set_roi_update_callback(self._handle_roi_update_callback)
            # 不再需要独立的ROI更新调度，现在由主轮询驱动
            logger.info("ROI update callback configured, using main polling loop")

    def _handle_roi_update_callback(self, data):
        """处理来自主轮询的双ROI数据更新"""
        try:
            logger.debug("ROI update callback received data")

            if not data or data.get("type") != "dual_realtime_data":
                logger.debug(f"Skipping non-dual ROI data: type={data.get('type') if data else 'None'}")
                return

            dual_roi_data = data.get("dual_roi_data", {})
            roi1_data = dual_roi_data.get("roi1_data", {})
            roi2_data = dual_roi_data.get("roi2_data", {})

            # 验证数据结构
            if not roi1_data or not roi2_data:
                logger.error("Missing ROI data in dual ROI response")
                self._update_roi_displays_error("Missing ROI data in response")
                return

            if "pixels" not in roi1_data or "pixels" not in roi2_data:
                logger.error("Missing pixel data in dual ROI response")
                self._update_roi_displays_error("Missing pixel data in response")
                return

            # 更新双ROI显示
            logger.debug("Updating dual ROI displays...")
            self._update_dual_roi_displays(roi1_data, roi2_data)

            # 更新ROI信息（显示ROI2的灰度值，因为ROI2用于峰值检测）
            roi1_width = roi1_data.get("width", 0)
            roi1_height = roi1_data.get("height", 0)
            roi2_gray_value = roi2_data.get("gray_value", 0)

            # 增强ROI2显示逻辑，区分正常和错误状态
            roi2_pixels = roi2_data.get("pixels", "")

            if roi2_pixels.startswith("roi2_"):
                # ROI2提取失败或错误状态
                if roi2_pixels == "roi2_capture_failed":
                    display_text = "ROI2: 截取失败"
                    color = "red"
                elif roi2_pixels == "roi2_extract_failed":
                    display_text = "ROI2: 提取失败"
                    color = "orange"
                elif roi2_pixels == "roi2_capture_error":
                    display_text = "ROI2: 错误"
                    color = "red"
                else:
                    display_text = f"ROI2: 异常({roi2_gray_value:.1f})"
                    color = "orange"
                logger.debug(f"ROI2 in error state: {roi2_pixels}, gray={roi2_gray_value:.1f}")
            elif roi2_gray_value == 0.0:
                # ROI2灰度值为0，可能是有效数据或回退数据
                display_text = f"ROI2: {roi2_gray_value:.1f}"
                color = "orange"
                logger.debug(f"ROI2 gray value is 0.0: pixels_type={'text' if roi2_pixels.startswith('roi') else 'image'}")
            else:
                # ROI2数据正常
                display_text = f"ROI2: {roi2_gray_value:.1f}"
                color = "green"
                logger.debug(f"ROI2 data normal: gray={roi2_gray_value:.1f}")

            # 显示ROI1灰度值信息，帮助诊断ROI2问题
            roi1_gray_value = roi1_data.get("gray_value", 0)
            roi1_info = f"ROI1: {roi1_width}x{roi1_height}"
            if roi1_gray_value > 0:
                roi1_info += f" (灰度:{roi1_gray_value:.1f})"
            self.roi_resolution_label.config(text=roi1_info)
            self.roi_gray_value_label.config(text=display_text, foreground=color)

        except Exception as e:
            logger.error(f"❌ Error in ROI update callback: {e}")
            import traceback
            logger.error(f"Callback traceback: {traceback.format_exc()}")
            self._update_roi_displays_error(f"Callback error: {str(e)}")

    def update_roi_screenshot(self):
        """更新ROI截图显示（单ROI模式 - 向后兼容）"""
        if not self.connected or not self.http_client or self.http_client.dual_roi_mode:
            # 双ROI模式不需要独立更新，由回调处理
            return

        try:
            # 双ROI模式跳过，只处理单ROI模式（原有逻辑）

            # 单ROI模式（原有逻辑）
            response = self.http_client.session.get(f"{self.http_client.base_url}/data/realtime?count=1", timeout=3)
            if response.status_code == 200:
                data = response.json()
                if data.get("type") == "realtime_data":
                    roi_data = data.get("roi_data", {})

                    if roi_data and "pixels" in roi_data:
                        # 更新ROI截图
                        base64_image = roi_data["pixels"]
                        if base64_image.startswith("data:image/png;base64,"):
                            # 提取base64数据
                            base64_data = base64_image.split("data:image/png;base64,")[1]

                            # 将base64转换为PhotoImage
                            image_data = base64.b64decode(base64_data)
                            image = Image.open(io.BytesIO(image_data))

                            # 调整图像大小以适应显示区域
                            image = image.resize((200, 150), Image.Resampling.LANCZOS)

                            # 保存PIL Image对象供后续使用
                            self._last_image = image

                            # 创建PhotoImage对象
                            photo = ImageTk.PhotoImage(image)

                            # 更新双ROI标签显示
                            self._update_roi_displays(photo)

                            # 更新ROI信息
                            width = roi_data.get("width", 0)
                            height = roi_data.get("height", 0)
                            gray_value = roi_data.get("gray_value", 0)

                            self.roi_resolution_label.config(text=f"{width}x{height}")
                            self.roi_gray_value_label.config(text=f"{gray_value:.1f}")

                        else:
                            self._update_roi_displays_error("Invalid ROI data format")
                    else:
                        self._update_roi_displays_error("No ROI data available")
                        self.roi_resolution_label.config(text="N/A")
                        self.roi_gray_value_label.config(text="N/A")
                else:
                    self._update_roi_displays_error("Invalid data type")
            else:
                self._update_roi_displays_error("Failed to get ROI data")

        except Exception as e:
            self._update_roi_displays_error(f"Error: {str(e)}")
            print(f"ROI update error: {e}")

        # 每500ms更新一次 (2 FPS)
        if self.connected:
            self.after(500, self.update_roi_screenshot)

    def _update_roi_displays(self, photo):
        """更新左右两个ROI显示"""
        # 更新左侧ROI显示
        self.roi_label_left.config(image=photo, text="")
        self.roi_label_left.image = photo  # 保持引用避免垃圾回收

        # 创建右侧的PhotoImage副本以确保两个widget都能正常显示
        # PhotoImage对象需要在每个widget中保持独立的引用
        if hasattr(self, '_last_image'):
            # 重用上次的PIL Image对象来创建新的PhotoImage
            right_photo = ImageTk.PhotoImage(self._last_image)
        else:
            # 如果没有保存的Image对象，使用当前photo创建副本
            # 这种情况下，我们需要从photo重建PIL Image
            try:
                # 获取原始图像数据并创建新的PhotoImage
                right_photo = ImageTk.PhotoImage(photo)
            except:
                # 如果无法创建副本，就使用同一个photo（可能会有显示问题）
                right_photo = photo

        # 更新右侧ROI显示
        self.roi_label_right.config(image=right_photo, text="")
        self.roi_label_right.right_image = right_photo  # 保持右侧引用

    def _update_dual_roi_displays(self, roi1_data, roi2_data):
        """更新双ROI显示：ROI1在左侧，ROI2在右侧"""
        try:
            logger.debug("Processing dual ROI display updates...")

            # 处理ROI1（大区域）- 左侧显示
            roi1_base64 = roi1_data["pixels"]
            if roi1_base64.startswith("data:image/png;base64,"):
                logger.debug("Processing ROI1 image...")
                roi1_base64_data = roi1_base64.split("data:image/png;base64,")[1]
                roi1_image_data = base64.b64decode(roi1_base64_data)
                roi1_image = Image.open(io.BytesIO(roi1_image_data))

                # 调整ROI1图像大小
                try:
                    roi1_resized = roi1_image.resize((250, 188), Image.Resampling.LANCZOS)
                except AttributeError:
                    # 兼容旧版本PIL
                    roi1_resized = roi1_image.resize((250, 188), Image.LANCZOS)
                roi1_photo = ImageTk.PhotoImage(roi1_resized)

                # 更新左侧ROI显示
                self.roi_label_left.config(image=roi1_photo, text="ROI1 (Large)")
                self.roi_label_left.image = roi1_photo
                logger.debug("✅ ROI1 display updated successfully")
            else:
                logger.warning("ROI1: Invalid base64 format")
                self.roi_label_left.config(text="ROI1: Invalid data format", image="")

            # 处理ROI2（50x50中心区域）- 右侧显示
            roi2_base64 = roi2_data["pixels"]
            if roi2_base64.startswith("data:image/png;base64,"):
                logger.debug("Processing ROI2 image...")
                roi2_base64_data = roi2_base64.split("data:image/png;base64,")[1]

                # 添加Base64解码调试日志
                roi2_image_data = base64.b64decode(roi2_base64_data)
                roi2_data_size = len(roi2_image_data)
                logger.debug(f"ROI2 base64 decoded: size={roi2_data_size} bytes")

                roi2_image = Image.open(io.BytesIO(roi2_image_data))
                roi2_original_size = roi2_image.size
                roi2_mode = roi2_image.mode
                logger.debug(f"ROI2 image loaded: size={roi2_original_size}, mode={roi2_mode}")

                # 检查ROI2图像内容
                roi2_pixel_stats = list(roi2_image.getextrema())
                logger.debug(f"ROI2 pixel stats (RGB): {roi2_pixel_stats}")

                # 检查是否为灰度图像
                if roi2_mode == 'L':
                    min_val, max_val = roi2_image.getextrema()
                    logger.debug(f"ROI2 grayscale range: {min_val} - {max_val}")
                    if max_val == 0:
                        logger.warning("ROI2 image appears to be all black (grayscale)")

                # 调整ROI2图像大小
                try:
                    roi2_resized = roi2_image.resize((250, 188), Image.Resampling.LANCZOS)
                except AttributeError:
                    # 兼容旧版本PIL
                    roi2_resized = roi2_image.resize((250, 188), Image.LANCZOS)

                roi2_resized_size = roi2_resized.size
                logger.debug(f"ROI2 resized to: {roi2_resized_size}")

                # 检查调整大小后的图像
                if roi2_resized.mode == 'L':
                    min_val, max_val = roi2_resized.getextrema()
                    logger.debug(f"ROI2 resized range: {min_val} - {max_val}")
                    if max_val == 0:
                        logger.warning("ROI2 resized image appears to be all black")

                roi2_photo = ImageTk.PhotoImage(roi2_resized)

                # 更新右侧ROI显示
                self.roi_label_right.config(image=roi2_photo, text="ROI2 (50x50)")
                self.roi_label_right.image = roi2_photo  # 保持一致的引用命名
                logger.debug("✅ ROI2 display updated successfully")
            else:
                logger.warning("ROI2: Invalid base64 format")
                self.roi_label_right.config(text="ROI2: Invalid data format", image="")

            # 更新ROI信息（显示ROI2的灰度值，因为ROI2用于峰值检测）
            roi1_width = roi1_data.get("width", 0)
            roi1_height = roi1_data.get("height", 0)
            roi2_gray_value = roi2_data.get("gray_value", 0)

            # 增强ROI2显示逻辑（与上方逻辑保持一致）
            roi2_pixels = roi2_data.get("pixels", "")

            if roi2_pixels.startswith("roi2_"):
                # ROI2提取失败或错误状态
                if roi2_pixels == "roi2_capture_failed":
                    display_text = "ROI2: 截取失败"
                    color = "red"
                elif roi2_pixels == "roi2_extract_failed":
                    display_text = "ROI2: 提取失败"
                    color = "orange"
                elif roi2_pixels == "roi2_capture_error":
                    display_text = "ROI2: 错误"
                    color = "red"
                else:
                    display_text = f"ROI2: 异常({roi2_gray_value:.1f})"
                    color = "orange"
            elif roi2_gray_value == 0.0:
                display_text = f"ROI2: {roi2_gray_value:.1f}"
                color = "orange"
            else:
                display_text = f"ROI2: {roi2_gray_value:.1f}"
                color = "green"

            # 显示ROI1灰度值信息，帮助诊断ROI2问题
            roi1_gray_value = roi1_data.get("gray_value", 0)
            roi1_info = f"ROI1: {roi1_width}x{roi1_height}"
            if roi1_gray_value > 0:
                roi1_info += f" (灰度:{roi1_gray_value:.1f})"
            self.roi_resolution_label.config(text=roi1_info)
            self.roi_gray_value_label.config(text=display_text, foreground=color)

            logger.debug(f"✅ Dual ROI info updated: ROI1={roi1_width}x{roi1_height}, ROI2 gray={roi2_gray_value:.1f}, status={color}")

        except Exception as e:
            logger.error(f"❌ Error updating dual ROI displays: {e}")
            import traceback
            logger.error(f"Dual ROI display traceback: {traceback.format_exc()}")
            self._update_roi_displays_error("Dual ROI display error")

    def _update_roi_displays_error(self, error_message):
        """更新ROI显示错误状态"""
        # 更新左侧ROI显示
        self.roi_label_left.config(text=error_message, image="")
        if hasattr(self.roi_label_left, 'image'):
            self.roi_label_left.image = None

        # 更新右侧ROI显示
        self.roi_label_right.config(text=error_message, image="")
        if hasattr(self.roi_label_right, 'image'):
            self.roi_label_right.image = None

    def _capture_curve(self):
        """截取曲线数据"""
        if not self.connected or not self.http_client:
            messagebox.showerror("错误", "请先连接到服务器")
            return

        try:
            self._log("Starting curve capture...")
            self.btn_capture.config(state="disabled", text="截取中...")

            # 使用ROI窗口截取API获取带波峰检测的数据，强制刷新缓存
            response = self.http_client.session.get(
                f"{self.http_client.base_url}/data/roi-window-capture-with-peaks?count=100&force_refresh=true",
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    # 获取截取的数据 - 适配服务器返回的数据结构
                    captured_data = data.get("series", [])
                    peak_results = data.get("peak_detection_results", {})

                    # 将波峰数据转换为客户端期望的格式
                    peaks = []
                    green_peaks = peak_results.get("green_peaks", [])
                    red_peaks = peak_results.get("red_peaks", [])

                    # 转换波峰数据格式
                    for peak_info in green_peaks:
                        if len(peak_info) >= 2 and peak_info[0] < len(captured_data):
                            peaks.append({
                                't': captured_data[peak_info[0]]['t'],
                                'value': captured_data[peak_info[0]]['gray_value'],
                                'peak_color': 'green'
                            })

                    for peak_info in red_peaks:
                        if len(peak_info) >= 2 and peak_info[0] < len(captured_data):
                            peaks.append({
                                't': captured_data[peak_info[0]]['t'],
                                'value': captured_data[peak_info[0]]['gray_value'],
                                'peak_color': 'red'
                            })

                    if captured_data:
                        # 添加调试信息验证修复效果
                        times = [point.get("t", 0) for point in captured_data]
                        values = [point.get("gray_value", point.get("value", 0)) for point in captured_data]

                        if times and values:
                            time_range = max(times) - min(times) if len(times) > 1 else 0
                            value_range = max(values) - min(values) if len(values) > 1 else 0
                            self._log(f"DEBUG: Time range: {time_range:.3f}s, Value range: {value_range:.2f}")
                            self._log(f"DEBUG: Time span: [{min(times):.3f}, {max(times):.3f}], Value span: [{min(values):.2f}, {max(values):.2f}]")

                        self._log(f"Curve capture successful! Got {len(captured_data)} data points with {len(peaks)} peaks")
                        self._display_captured_curve(captured_data, peaks, peak_results)

                        # 更新截取信息
                        self.captured_count_label.config(text=str(len(captured_data)))
                        self.captured_source_label.config(text="ROI数据")

                        # 启用清除按钮
                        self.btn_clear_capture.config(state="normal")

                        # 成功日志记录（不显示弹框）
                        self._log(f"✅ 曲线截取成功！数据点数: {len(captured_data)}, 波峰数: {len(peaks)}")
                    else:
                        raise Exception("No captured data received")
                else:
                    raise Exception(data.get("error", "Unknown error"))
            else:
                raise Exception(f"Server error: {response.status_code}")

        except Exception as e:
            self._log(f"Curve capture failed: {str(e)}", "ERROR")
            messagebox.showerror("截取失败", f"曲线截取失败: {str(e)}")
        finally:
            self.btn_capture.config(state="normal", text="截取曲线")

    def _display_captured_curve(self, data_points, peaks, peak_results=None):
        """显示截取的曲线"""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            import numpy as np

            # 首先清理之前的画布 - 修复第二次截取无法显示的关键问题
            self._clear_capture()

            # 创建新图表 - 使用与主图表相同的大小
            fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
            fig.patch.set_facecolor('white')

            # 打印窗口大小信息
            self._log(f"截取曲线图表尺寸信息:")
            self._log(f"  - 图表尺寸: 12 x 8 英寸")
            self._log(f"  - DPI设置: 100")
            self._log(f"  - 像素尺寸: {fig.get_figwidth() * fig.dpi:.0f} x {fig.get_figheight() * fig.dpi:.0f} 像素")
            self._log(f"  - 容器高度: 300 像素 (最小)")
            self._log(f"  - 位置: 实时图表上方")

            # 提取时间和数值 - 适配服务器返回的数据格式
            times = [point.get("t", 0) for point in data_points]
            values = [point.get("gray_value", point.get("value", 0)) for point in data_points]

            self._log(f"DEBUG: Preparing to display curve with {len(times)} points")
            self._log(f"DEBUG: Data validation - times count: {len(times)}, values count: {len(values)}")

            # 验证数据完整性
            if len(times) != len(values):
                raise ValueError(f"Data length mismatch: {len(times)} times vs {len(values)} values")

            if not times or not values:
                raise ValueError("No valid data points to display")

            # 验证数据范围
            if len(times) > 0 and len(values) > 0:
                # 绘制曲线
                ax.plot(times, values, 'b-', linewidth=2, label='Captured Signal')

                # 添加基于真实波峰检测的区间高亮
                if peak_results and len(times) > 0:
                    green_peaks = peak_results.get("green_peaks", [])
                    red_peaks = peak_results.get("red_peaks", [])

                    self._log(f"DEBUG: Peak results - Green peaks: {len(green_peaks)}, Red peaks: {len(red_peaks)}")

                    # 绘制绿色波峰区间（稳定HEM事件）
                    for i, (start_frame, end_frame) in enumerate(green_peaks):
                        if start_frame < len(times) and end_frame < len(times):
                            start_time = times[start_frame]
                            end_time = times[end_frame]
                            ax.axvspan(start_time, end_time, alpha=0.2, color='green',
                                      label='Stable HEM' if i == 0 else None)
                            self._log(f"DEBUG: Green peak {i+1}: frames {start_frame}-{end_frame}, time {start_time:.3f}-{end_time:.3f}")

                    # 绘制红色波峰区间（不稳定HEM事件）
                    for i, (start_frame, end_frame) in enumerate(red_peaks):
                        if start_frame < len(times) and end_frame < len(times):
                            start_time = times[start_frame]
                            end_time = times[end_frame]
                            ax.axvspan(start_time, end_time, alpha=0.2, color='red',
                                      label='Unstable HEM' if i == 0 else None)
                            self._log(f"DEBUG: Red peak {i+1}: frames {start_frame}-{end_frame}, time {start_time:.3f}-{end_time:.3f}")

                # 强制设置Y轴范围，确保小的灰度变化能够清晰显示
                min_val = min(values)
                max_val = max(values)
                value_range = max_val - min_val

                if value_range < 10:  # 如果数据范围太小，强制扩展显示范围
                    center = (min_val + max_val) / 2
                    expanded_range = 5  # 至少显示5的范围
                    ax.set_ylim(center - expanded_range/2, center + expanded_range/2)
                else:
                    # 否则使用正常范围并稍微扩展
                    padding = value_range * 0.1
                    ax.set_ylim(min_val - padding, max_val + padding)

                # 绘制基线
                if values:
                    baseline = np.mean(values)
                    baseline_line = [baseline] * len(times)
                    ax.plot(times, baseline_line, 'r--', linewidth=1, alpha=0.6, label=f'Baseline={baseline:.1f}')

                # 标记波峰
                if peaks:
                    peak_times = [peak.get("t", 0) for peak in peaks]
                    peak_values = [peak.get("value", 0) for peak in peaks]
                    peak_colors = []

                    # 根据波峰颜色分类
                    for peak in peaks:
                        if peak.get("peak_color") == "green":
                            peak_colors.append('green')
                        elif peak.get("peak_color") == "red":
                            peak_colors.append('red')
                        else:
                            peak_colors.append('orange')

                    # 绘制波峰点
                    for i, (t, v, color) in enumerate(zip(peak_times, peak_values, peak_colors)):
                        ax.scatter([t], [v], c=color, s=50, zorder=5)

                ax.set_title("Captured Curve with Peak Detection", fontsize=12, fontweight='bold')
                ax.set_xlabel("Time (seconds)")
                ax.set_ylabel("Signal Value")
                ax.grid(True, alpha=0.3)
                ax.legend()

                # 自动调整坐标轴
                ax.set_xlim(min(times) - 0.1, max(times) + 0.1)
                if values:
                    ax.set_ylim(min(values) - 2, max(values) + 2)

                plt.tight_layout()

                # 清理标签内容并嵌入新的canvas
                self.captured_label.config(text="")

                # 创建并嵌入canvas - 添加验证
                self._log("DEBUG: Creating FigureCanvasTkAgg...")
                canvas = FigureCanvasTkAgg(fig, master=self.captured_wrapper)

                # 验证canvas创建是否成功
                if canvas is None:
                    raise RuntimeError("Failed to create matplotlib canvas")

                # 绘制图表
                self._log("DEBUG: Drawing canvas...")
                canvas.draw()

                # 获取widget并验证
                widget = canvas.get_tk_widget()
                if widget is None:
                    raise RuntimeError("Failed to get tkinter widget from canvas")

                # 嵌入widget
                self._log("DEBUG: Packing canvas widget...")
                widget.pack(fill='both', expand=True)

                # 验证widget是否正确嵌入
                self.after(100, lambda: self._verify_canvas_display(canvas, fig))

                # 保存引用
                self.captured_canvas = canvas
                self.captured_fig = fig

                self._log(f"DEBUG: Canvas created and embedded successfully")

        except Exception as e:
            self._log(f"Error displaying captured curve: {str(e)}", "ERROR")
            self.captured_label.config(text=f"显示错误: {str(e)}", image="")

    def _verify_canvas_display(self, canvas, fig):
        """验证canvas是否正确显示"""
        try:
            if canvas is None:
                self._log("ERROR: Canvas is None after creation", "ERROR")
                return

            widget = canvas.get_tk_widget()
            if widget is None:
                self._log("ERROR: Widget is None after canvas creation", "ERROR")
                return

            # 检查widget是否可见
            try:
                if widget.winfo_viewable():
                    self._log("DEBUG: Canvas widget is visible and properly displayed")
                else:
                    self._log("WARNING: Canvas widget is not visible", "WARNING")
            except Exception as e:
                self._log(f"DEBUG: Could not verify widget visibility: {e}")

            # 检查widget尺寸
            try:
                width = widget.winfo_width()
                height = widget.winfo_height()
                self._log(f"DEBUG: Canvas widget size: {width}x{height}")
            except Exception as e:
                self._log(f"DEBUG: Could not get widget size: {e}")

        except Exception as e:
            self._log(f"Error in canvas verification: {str(e)}", "ERROR")

    def _clear_capture(self):
        """清除截取的曲线"""
        try:
            import matplotlib.pyplot as plt

            self._log("DEBUG: Clearing previous captured curve...")

            # 清除canvas - 修复关键：确保彻底清理
            if hasattr(self, 'captured_canvas') and self.captured_canvas is not None:
                try:
                    # 获取canvas的tkinter widget并销毁
                    widget = self.captured_canvas.get_tk_widget()
                    if widget.winfo_exists():
                        widget.destroy()
                except Exception as e:
                    self._log(f"DEBUG: Error destroying canvas widget: {e}")
                finally:
                    self.captured_canvas = None

            # 清除matplotlib图形对象
            if hasattr(self, 'captured_fig') and self.captured_fig is not None:
                try:
                    plt.close(self.captured_fig)
                except Exception as e:
                    self._log(f"DEBUG: Error closing figure: {e}")
                finally:
                    self.captured_fig = None

            # 清除标签的所有子组件 - 确保彻底清理
            for widget in self.captured_label.winfo_children():
                try:
                    widget.destroy()
                except Exception as e:
                    self._log(f"DEBUG: Error destroying child widget: {e}")

            # 重置标签状态
            self.captured_label.config(text="No captured curve yet. Click '截取曲线' to capture data.", image="")
            self.captured_label.image = None

            # 重置信息标签
            self.captured_count_label.config(text="N/A")
            self.captured_source_label.config(text="N/A")

            self._log("DEBUG: Capture cleared successfully")

        except Exception as e:
            self._log(f"Error in _clear_capture: {str(e)}", "ERROR")

        # 禁用清除按钮
        self.btn_clear_capture.config(state="disabled")
        self._log("Captured curve cleared")

    def _log(self, message: str, level: str = "INFO"):
        """添加日志"""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}\n"

        self.log_text.insert("end", log_entry)
        self.log_text.see("end")  # 自动滚动到底部

    def _apply_roi_config(self):
        """应用ROI配置"""
        if not self.connected or not self.http_client:
            messagebox.showerror("错误", "请先连接到服务器")
            return

        try:
            self._log("应用ROI配置...")

            # 获取配置值
            x1 = int(self.roi_x1_var.get())
            y1 = int(self.roi_y1_var.get())
            x2 = int(self.roi_x2_var.get())
            y2 = int(self.roi_y2_var.get())

            # 验证ROI坐标
            if x2 <= x1 or y2 <= y1:
                messagebox.showerror("错误", "ROI坐标无效：X2必须大于X1，Y2必须大于Y1")
                return

            # 发送ROI配置请求
            roi_data = {
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "password": self.http_client.password
            }

            response = self.http_client.session.post(
                f"{self.http_client.base_url}/roi/config",
                data=roi_data,
                timeout=5
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    self._log(f"✅ ROI配置应用成功: X={x1},{x2}, Y={y1},{y2}")
                else:
                    messagebox.showerror("配置失败", f"ROI配置失败: {result.get('error', '未知错误')}")
            else:
                messagebox.showerror("配置失败", f"服务器错误: {response.status_code}")

        except ValueError as e:
            messagebox.showerror("输入错误", f"参数格式错误: {str(e)}")
        except Exception as e:
            self._log(f"ROI配置应用失败: {str(e)}", "ERROR")
            messagebox.showerror("配置失败", f"ROI配置应用失败: {str(e)}")

    def _apply_peak_config(self):
        """应用波峰检测配置"""
        if not self.connected or not self.http_client:
            messagebox.showerror("错误", "请先连接到服务器")
            return

        try:
            self._log("应用波峰检测配置...")

            # 获取配置值
            threshold = float(self.peak_threshold_var.get())
            margin_frames = int(self.peak_margin_var.get())
            diff_threshold = float(self.peak_diff_var.get())

            # 验证参数范围
            if not (50 <= threshold <= 255):
                messagebox.showerror("错误", "绝对阈值必须在50-255之间")
                return
            if not (1 <= margin_frames <= 20):
                messagebox.showerror("错误", "边界帧数必须在1-20之间")
                return
            if not (0.1 <= diff_threshold <= 10.0):
                messagebox.showerror("错误", "差值阈值必须在0.1-10.0之间")
                return

            # 发送波峰检测配置请求
            peak_data = {
                "threshold": threshold,
                "margin_frames": margin_frames,
                "difference_threshold": diff_threshold,
                "password": self.http_client.password
            }

            response = self.http_client.session.post(
                f"{self.http_client.base_url}/peak-detection/config",
                data=peak_data,
                timeout=5
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    self._log(f"✅ 波峰检测配置应用成功: 阈值={threshold}, 边界={margin_frames}, 差值={diff_threshold}")
                else:
                    messagebox.showerror("配置失败", f"波峰检测配置失败: {result.get('error', '未知错误')}")
            else:
                messagebox.showerror("配置失败", f"服务器错误: {response.status_code}")

        except ValueError as e:
            messagebox.showerror("输入错误", f"参数格式错误: {str(e)}")
        except Exception as e:
            self._log(f"波峰检测配置应用失败: {str(e)}", "ERROR")
            messagebox.showerror("配置失败", f"波峰检测配置应用失败: {str(e)}")

    def _save_config(self):
        """保存配置到后端fem_config.json"""
        try:
            config_updates = {
                "roi_capture": {
                    "default_config": {
                        "x1": int(self.roi_x1_var.get()),
                        "y1": int(self.roi_y1_var.get()),
                        "x2": int(self.roi_x2_var.get()),
                        "y2": int(self.roi_y2_var.get())
                    },
                    "frame_rate": float(self.roi_fps_var.get())
                },
                "peak_detection": {
                    "threshold": float(self.peak_threshold_var.get()),
                    "margin_frames": int(self.peak_margin_var.get()),
                    "difference_threshold": float(self.peak_diff_var.get())
                },
                "line_detection": {
                    "enabled": self.show_line_detection,
                    "auto_start": False,
                    "update_interval": 100
                }
            }

            # 使用后端API保存配置
            config_data = json.dumps(config_updates, ensure_ascii=False)

            response = self.http_client.session.post(
                f"{self.http_client.base_url}/config",
                params={
                    "config_data": config_data,
                    "password": self.http_client.password
                },
                timeout=5
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("success", True):
                    self._log("✅ 配置已保存到服务器 fem_config.json")
                    messagebox.showinfo("保存成功", "配置已保存到服务器 fem_config.json")
                else:
                    error_msg = result.get("error", "保存失败")
                    self._log(f"配置保存失败: {error_msg}", "ERROR")
                    messagebox.showerror("保存失败", f"配置保存失败: {error_msg}")
            else:
                error_text = response.text
                self._log(f"配置保存失败: HTTP {response.status_code} - {error_text}", "ERROR")
                messagebox.showerror("保存失败", f"配置保存失败: {response.status_code}")

        except Exception as e:
            self._log(f"配置保存失败: {str(e)}", "ERROR")
            messagebox.showerror("保存失败", f"配置保存失败: {str(e)}")

    def _load_config(self):
        """从后端fem_config.json加载配置"""
        try:
            if not self.connected or not self.http_client:
                messagebox.showerror("错误", "请先连接到服务器")
                return

            # 使用后端API获取配置
            response = self.http_client.session.get(
                f"{self.http_client.base_url}/config",
                params={
                    "password": self.http_client.password
                },
                timeout=5
            )

            if response.status_code == 200:
                result = response.json()
                if "config" in result:
                    config = result["config"]

                    # 加载ROI配置
                    if "roi_capture" in config:
                        roi_config = config["roi_capture"]
                        default_config = roi_config.get("default_config", {})
                        self.roi_x1_var.set(str(default_config.get("x1", 0)))
                        self.roi_y1_var.set(str(default_config.get("y1", 0)))
                        self.roi_x2_var.set(str(default_config.get("x2", 200)))
                        self.roi_y2_var.set(str(default_config.get("y2", 150)))
                        self.roi_fps_var.set(str(roi_config.get("frame_rate", 2)))

                    # 加载波峰检测配置
                    if "peak_detection" in config:
                        peak_config = config["peak_detection"]
                        self.peak_threshold_var.set(str(peak_config.get("threshold", 105.0)))
                        self.peak_margin_var.set(str(peak_config.get("margin_frames", 5)))
                        self.peak_diff_var.set(str(peak_config.get("difference_threshold", 2.1)))

                    self._log("✅ 配置已从服务器 fem_config.json 加载")
                    messagebox.showinfo("加载成功", "配置已从服务器 fem_config.json 加载")
                else:
                    error_msg = result.get("error", "获取配置失败")
                    self._log(f"加载配置失败: {error_msg}", "ERROR")
                    messagebox.showerror("加载失败", f"加载配置失败: {error_msg}")
            else:
                error_text = response.text
                self._log(f"加载配置失败: HTTP {response.status_code} - {error_text}", "ERROR")
                messagebox.showerror("加载失败", f"加载配置失败: {response.status_code}")

        except Exception as e:
            self._log(f"配置加载失败: {str(e)}", "ERROR")
            messagebox.showerror("加载失败", f"配置加载失败: {str(e)}")

        # 限制日志行数
        lines = int(self.log_text.index("end-1c").split(".")[0])
        if lines > 1000:
            self.log_text.delete("1.0", "100.0")

    def _apply_server_config(self, config_dict):
        """应用从服务器加载的配置到UI字段"""
        try:
            if not config_dict:
                self._log("服务器配置为空，使用默认值")
                return False

            config_applied = False
            missing_fields = []

            # 应用ROI配置
            if "roi_capture" in config_dict:
                roi_config = config_dict["roi_capture"]

                # 应用ROI坐标
                if "default_config" in roi_config:
                    default_config = roi_config["default_config"]
                    self.roi_x1_var.set(str(default_config.get("x1", 0)))
                    self.roi_y1_var.set(str(default_config.get("y1", 0)))
                    self.roi_x2_var.set(str(default_config.get("x2", 200)))
                    self.roi_y2_var.set(str(default_config.get("y2", 150)))
                    config_applied = True

                # 应用ROI帧率
                if "frame_rate" in roi_config:
                    self.roi_fps_var.set(str(roi_config["frame_rate"]))
                    config_applied = True
            else:
                missing_fields.append("roi_capture")

            # 应用波峰检测配置
            if "peak_detection" in config_dict:
                peak_config = config_dict["peak_detection"]

                self.peak_threshold_var.set(str(peak_config.get("threshold", 105.0)))
                self.peak_margin_var.set(str(peak_config.get("margin_frames", 5)))
                self.peak_diff_var.set(str(peak_config.get("difference_threshold", 2.1)))
                config_applied = True
            else:
                missing_fields.append("peak_detection")

            # 应用绿线检测配置
            if "line_detection" in config_dict:
                line_config = config_dict["line_detection"]
                line_detection_enabled = line_config.get("enabled", True)

                # 更新显示状态但不强制创建标签页（因为窗口已经构建）
                if self.show_line_detection != line_detection_enabled:
                    self.show_line_detection = line_detection_enabled
                    # 更新按钮文本
                    if hasattr(self, 'btn_line_detection_toggle'):
                        if self.show_line_detection:
                            self.btn_line_detection_toggle.config(text="隐藏绿线检测")
                        else:
                            self.btn_line_detection_toggle.config(text="显示绿线检测")

                config_applied = True
            else:
                missing_fields.append("line_detection")

            # 同步绿线检测配置与后端
            if hasattr(self, 'http_client') and self.http_client and self.http_client.line_detection_config_loaded:
                self.http_client._sync_line_detection_config_with_backend(config_dict)

            # 应用增强数据配置
            if "enhanced_data" in config_dict:
                enhanced_config = config_dict["enhanced_data"]

                # 如果HTTP客户端已创建，应用配置到客户端
                if hasattr(self, 'http_client') and self.http_client:
                    self.http_client.set_enhanced_data_config(
                        include_line_intersection=enhanced_config.get("include_line_intersection", True),
                        enhanced_data_enabled=enhanced_config.get("enabled", True),
                        fallback_on_error=enhanced_config.get("fallback_on_error", True)
                    )

                config_applied = True
                self._log("✅ 增强数据配置已应用")
            else:
                missing_fields.append("enhanced_data")

            if config_applied:
                self._log("✅ 成功应用服务器配置到UI")
                if missing_fields:
                    self._log(f"⚠️ 缺少配置字段: {', '.join(missing_fields)}")
                return True
            else:
                self._log("⚠️ 配置格式不符合预期，使用默认值")
                return False

        except Exception as e:
            self._log(f"❌ 应用服务器配置失败: {str(e)}", "ERROR")
            return False

    def _load_local_config(self):
        """从本地配置文件加载配置"""
        try:
            self._log("🔄 正在加载本地配置文件...")

            # 检查本地配置加载器是否可用
            if not LOCAL_CONFIG_LOADER_AVAILABLE or LocalConfigLoader is None:
                self._log("❌ 本地配置加载器不可用", "WARNING")
                return False

            # 创建本地配置加载器
            config_loader = LocalConfigLoader()

            # 加载配置
            success, message, config_data = config_loader.load_config()

            if success:
                self._log(f"✅ {message}")

                # 应用配置到UI字段
                if self._apply_server_config(config_data):
                    self._log("🎯 本地配置已成功应用到UI界面")
                    return True
                else:
                    self._log("⚠️ 本地配置应用失败，使用默认值")
                    return False
            else:
                self._log(f"❌ 本地配置加载失败: {message}")
                return False

        except Exception as e:
            self._log(f"❌ 本地配置加载异常: {str(e)}", "ERROR")
            return False

    def _auto_load_config(self):
        """自动从服务器加载配置"""
        try:
            if not self.connected or not self.http_client:
                self._log("⚠️ 服务器未连接，跳过自动配置加载")
                return False

            self._log("🔄 自动加载服务器配置...")

            # 向服务器请求配置
            response = self.http_client.session.get(
                f"{self.http_client.base_url}/config",
                params={"password": self.http_client.password},
                timeout=5
            )

            if response.status_code == 200:
                result = response.json()
                if "config" in result:
                    config = result["config"]
                    success = self._apply_server_config(config)
                    if success:
                        self._log("🎯 自动配置加载完成")
                        return True
                    else:
                        self._log("⚠️ 自动配置加载失败，使用默认值")
                        return False
                else:
                    error_msg = result.get("error", "获取配置失败")
                    self._log(f"❌ 自动配置加载失败: {error_msg}", "ERROR")
                    return False
            else:
                self._log(f"❌ 获取配置失败: HTTP {response.status_code}", "ERROR")
                return False

        except Exception as e:
            self._log(f"❌ 自动配置加载异常: {str(e)}", "ERROR")
            return False

    def _load_line_detection_config(self):
        """加载绿线检测配置"""
        try:
            self._log("🔄 正在加载绿线检测配置...")

            # 检查配置管理器是否可用
            if self.line_detection_config_manager is None:
                self._log("❌ 绿线检测配置管理器未初始化", "ERROR")
                return False

            # 加载配置
            success, message, config_data = self.line_detection_config_manager.load_config()

            if success:
                self.line_detection_config_loaded = True
                line_detection_config = self.line_detection_config_manager.get_line_detection_config()

                # 更新绿线检测配置对象
                self.line_detection_config.enabled = line_detection_config.get("enabled", False)
                self.line_detection_config.auto_start = line_detection_config.get("auto_start", False)

                # 获取性能配置
                performance_config = line_detection_config.get("performance", {})
                self.line_detection_config.timeout = performance_config.get("processing_timeout_ms", 300) / 1000.0
                self.line_detection_config.retry_count = performance_config.get("max_retries", 3)
                self.line_detection_config.retry_delay = performance_config.get("retry_delay_ms", 100) / 1000.0

                # 获取同步配置
                sync_config = line_detection_config.get("synchronization", {})
                self.line_detection_config.sync_interval = sync_config.get("sync_interval_ms", 1000) / 1000.0

                self._log(f"✅ 绿线检测配置加载成功")
                self._log(f"   - 检测启用: {self.line_detection_config.enabled}")
                self._log(f"   - 自动启动: {self.line_detection_config.auto_start}")
                self._log(f"   - 超时时间: {self.line_detection_config.timeout:.1f}秒")
                self._log(f"   - 同步间隔: {self.line_detection_config.sync_interval:.1f}秒")

                # 如果配置了自动启动，则启用绿线检测
                if self.line_detection_config.auto_start and self.connected:
                    self._log("🚀 配置自动启动绿线检测...")
                    self._start_line_detection_state_sync()

                return True
            else:
                self._log(f"❌ 绿线检测配置加载失败: {message}")
                self.line_detection_config_loaded = False
                return False

        except Exception as e:
            self._log(f"❌ 绿线检测配置加载异常: {str(e)}", "ERROR")
            self.line_detection_config_loaded = False
            return False

    def _sync_line_detection_config_with_backend(self, backend_config: Dict[str, Any]):
        """同步绿线检测配置与后端"""
        try:
            if not self.line_detection_config_loaded:
                self._log("⚠️ 绿线检测配置未加载，跳过后端同步")
                return False

            self._log("🔄 正在同步绿线检测配置与后端...")

            success, message = self.line_detection_config_manager.sync_with_backend(backend_config)

            if success:
                self._log(f"✅ {message}")

                # 重新加载配置以获取同步后的设置
                line_detection_config = self.line_detection_config_manager.get_line_detection_config()

                # 更新运行时配置
                detection_config = line_detection_config.get("detection", {})

                self._log("🎯 同步完成，参数已更新")
                return True
            else:
                self._log(f"❌ 同步失败: {message}")
                return False

        except Exception as e:
            self._log(f"❌ 同步绿线检测配置异常: {str(e)}", "ERROR")
            return False

    def _create_line_detection_config_backup(self):
        """创建绿线检测配置备份"""
        try:
            if not self.line_detection_config_loaded:
                self._log("⚠️ 绿线检测配置未加载，跳过备份创建")
                return False

            success, message = self.line_detection_config_manager.create_backup()

            if success:
                self._log(f"✅ 绿线检测配置备份创建成功")
                return True
            else:
                self._log(f"❌ 备份创建失败: {message}")
                return False

        except Exception as e:
            self._log(f"❌ 创建配置备份异常: {str(e)}", "ERROR")
            return False

    def _export_line_detection_config(self, export_path: str = None):
        """导出绿线检测配置"""
        try:
            if not self.line_detection_config_loaded:
                self._log("⚠️ 绿线检测配置未加载，无法导出")
                return False

            if not export_path:
                # 默认导出路径
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                export_path = f"./exports/line_detection_config_{timestamp}.json"

            # 确保导出目录存在
            import os
            os.makedirs(os.path.dirname(export_path), exist_ok=True)

            success, message = self.line_detection_config_manager.export_config(export_path)

            if success:
                self._log(f"✅ 绿线检测配置导出成功: {export_path}")
                return True
            else:
                self._log(f"❌ 配置导出失败: {message}")
                return False

        except Exception as e:
            self._log(f"❌ 导出配置异常: {str(e)}", "ERROR")
            return False

    def _get_line_detection_ui_config(self) -> Dict[str, Any]:
        """获取绿线检测UI配置"""
        try:
            if not self.line_detection_config_loaded:
                return {}

            line_detection_config = self.line_detection_config_manager.get_line_detection_config()
            ui_config = line_detection_config.get("ui", {})

            return {
                "enable_widget": ui_config.get("enable_widget", True),
                "show_control_panel": ui_config.get("show_control_panel", True),
                "show_statistics_panel": ui_config.get("show_statistics_panel", True),
                "show_debug_panel": ui_config.get("show_debug_panel", False),
                "display_colors": ui_config.get("display_colors", {}),
                "font_settings": ui_config.get("font_settings", {}),
                "layout": ui_config.get("layout", {}),
                "animation": ui_config.get("animation", {})
            }

        except Exception as e:
            self._log(f"❌ 获取UI配置异常: {str(e)}", "ERROR")
            return {}

    def _get_line_detection_detection_config(self) -> Dict[str, Any]:
        """获取绿线检测算法配置"""
        try:
            if not self.line_detection_config_loaded:
                return {}

            line_detection_config = self.line_detection_config_manager.get_line_detection_config()
            return line_detection_config.get("detection", {})

        except Exception as e:
            self._log(f"❌ 获取检测配置异常: {str(e)}", "ERROR")
            return {}

    def _toggle_line_detection(self):
        """切换绿线检测标签页显示"""
        try:
            current_visible = self.show_line_detection

            if current_visible:
                # 隐藏绿线检测标签页
                if hasattr(self, 'line_detection_frame') and self.line_detection_frame in self.notebook.children.values():
                    # 获取当前索引
                    current_index = self.notebook.index(self.notebook.select())
                    # 移除标签页
                    self.notebook.forget(self.line_detection_frame)
                    self.show_line_detection = False
                    self.btn_line_detection_toggle.config(text="显示绿线检测")
                    self._log("绿线检测标签页已隐藏")

                    # 如果当前在绿线检测标签页，切换到监控标签页
                    if hasattr(self, 'line_detection_frame'):
                        try:
                            self.notebook.select(self.monitoring_frame)
                        except:
                            pass
            else:
                # 显示绿线检测标签页
                self.show_line_detection = True
                self.line_detection_frame = ttk.Frame(self.notebook)
                self.notebook.add(self.line_detection_frame, text="绿线交点检测")
                self.btn_line_detection_toggle.config(text="隐藏绿线检测")
                self._log("绿线检测标签页已显示")

                # 重新初始化LineDetectionWidget
                self._setup_line_detection_widget()

        except Exception as e:
            error_msg = f"切换绿线检测显示失败: {str(e)}"
            self._log(error_msg, "ERROR")
            logger.error(f"Failed to toggle line detection: {e}")

    def _on_line_detection_state_changed(self, old_state: LineDetectionState,
                                    new_state: LineDetectionState, error_msg: str = None):
        """处理绿线交点检测状态变化回调"""
        try:
            # 更新按钮文本和状态
            if hasattr(self, 'btn_line_detection_toggle'):
                if new_state == LineDetectionState.ENABLED:
                    self.btn_line_detection_toggle.config(text="禁用检测", state="normal")
                    self._log("✅ 绿线交点检测已启用")
                elif new_state == LineDetectionState.DISABLED:
                    self.btn_line_detection_toggle.config(text="启用检测", state="normal")
                    self._log("⏹️ 绿线交点检测已禁用")
                elif new_state == LineDetectionState.ENABLING:
                    self.btn_line_detection_toggle.config(text="启用中...", state="disabled")
                    self._log("🔄 正在启用绿线交点检测...")
                elif new_state == LineDetectionState.DISABLING:
                    self.btn_line_detection_toggle.config(text="禁用中...", state="disabled")
                    self._log("🔄 正在禁用绿线交点检测...")
                elif new_state == LineDetectionState.ERROR:
                    self.btn_line_detection_toggle.config(text="启用检测", state="normal")
                    error_text = f"绿线交点检测错误: {error_msg}" if error_msg else "绿线交点检测发生错误"
                    self._log(f"❌ {error_text}", "ERROR")

        except Exception as e:
            logger.error(f"Error handling line detection state change: {e}")

    def _toggle_line_detection(self):
        """切换绿线交点检测状态"""
        try:
            if not self.http_client:
                messagebox.showerror("错误", "请先连接到服务器")
                return

            current_state = self.http_client.get_line_detection_state()

            if current_state in [LineDetectionState.DISABLED, LineDetectionState.ERROR]:
                # 尝试启用检测
                self._log("🚀 正在启用绿线交点检测...")
                success = self.http_client.enable_line_detection()

                if success:
                    self._log("✅ 绿线交点检测启用成功")
                else:
                    self._log("❌ 绿线交点检测启用失败", "ERROR")
                    messagebox.showerror("错误", "绿线交点检测启用失败")

            elif current_state == LineDetectionState.ENABLED:
                # 尝试禁用检测
                self._log("🛑 正在禁用绿线交点检测...")
                success = self.http_client.disable_line_detection()

                if success:
                    self._log("✅ 绿线交点检测禁用成功")
                else:
                    self._log("❌ 绿线交点检测禁用失败", "ERROR")
                    messagebox.showerror("错误", "绿线交点检测禁用失败")

            else:
                # 正在转换中，提示用户等待
                self._log("⏳ 绿线交点检测状态正在转换中，请稍候...", "INFO")
                messagebox.showinfo("提示", "绿线交点检测状态正在转换中，请稍候...")

        except Exception as e:
            error_msg = f"切换绿线交点检测状态时发生错误: {str(e)}"
            self._log(error_msg, "ERROR")
            messagebox.showerror("错误", error_msg)

    def _on_closing(self):
        """窗口关闭事件"""
        try:
            # 断开连接
            self._disconnect()

            # 停止绘图动画
            if self.plotter:
                self.plotter.stop_animation()

            # 清理LineDetectionWidget
            if self.line_detection_widget:
                try:
                    self.line_detection_widget.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up LineDetectionWidget: {e}")

            # 销毁窗口
            self.destroy()

        except Exception as e:
            print(f"Error during cleanup: {e}")
            self.destroy()

    def _toggle_ui_mode(self):
        """切换UI模式（紧凑/完整）"""
        self.compact_mode = not self.compact_mode

        if self.compact_mode:
            # 切换到紧凑模式
            self.geometry(self.compact_geometry)
            self.btn_ui_toggle.config(text="放大")

            # 隐藏非必要组件
            if self.conn_frame:
                self.conn_frame.pack_forget()
            if self.info_frame:
                self.info_frame.pack_forget()
            if self.btn_clear:
                self.btn_clear.pack_forget()
            if self.btn_save:
                self.btn_save.pack_forget()
            if self.btn_capture:
                self.btn_capture.pack_forget()
            if hasattr(self, 'btn_line_detection_toggle'):
                self.btn_line_detection_toggle.pack_forget()

            # 简化状态文本
            if hasattr(self, 'status_var') and self.status_var:
                current_text = self.status_var.get()
                if "已连接" in current_text:
                    self.status_var.set("运行中")
                else:
                    self.status_var.set("就绪")

        else:
            # 切换到完整模式
            self.geometry(self.normal_geometry)
            self.btn_ui_toggle.config(text="缩小")

            # 重新显示所有组件
            if self.conn_frame:
                self.conn_frame.pack(fill="x", padx=8, pady=4, before=self.winfo_children()[1])
            if self.info_frame:
                # 找到主框架并重新添加info_frame
                for child in self.winfo_children():
                    if isinstance(child, ttk.Frame) and len(child.winfo_children()) > 0:
                        # 检查是否是主框架（包含图表）
                        for grandchild in child.winfo_children():
                            if hasattr(grandchild, 'figure'):  # matplotlib canvas
                                self.info_frame.pack(side="left", fill="y", padx=(0, 8), before=grandchild)
                                break
                        break

            if self.btn_clear:
                self.btn_clear.pack(side="left", padx=8, pady=4, after=self.btn_stop)
            if self.btn_save:
                self.btn_save.pack(side="left", padx=8, pady=4, after=self.btn_clear)
            if self.btn_capture:
                self.btn_capture.pack(side="left", padx=8, pady=4, after=self.btn_save)
            if hasattr(self, 'btn_line_detection_toggle'):
                self.btn_line_detection_toggle.pack(side="right", padx=8, pady=4)

            # 恢复详细状态文本
            if hasattr(self, 'status_var') and self.status_var:
                current_text = self.status_var.get()
                if "运行中" in current_text:
                    self.status_var.set("已连接")
                elif "就绪" in current_text:
                    self.status_var.set("未连接")

        # 重新布局和绘制
        self.update_idletasks()

    def _apply_enhanced_data_from_client_config(self):
        """从客户端配置文件应用增强数据设置"""
        try:
            # 加载客户端配置文件
            config_file = "http_client_config.json"
            if not os.path.exists(config_file):
                self._log("⚠️ 客户端配置文件不存在，使用默认增强数据设置")
                return

            with open(config_file, 'r', encoding='utf-8') as f:
                client_config = json.load(f)

            enhanced_config = client_config.get("enhanced_data", {})
            if enhanced_config and hasattr(self, 'http_client') and self.http_client:
                self.http_client.set_enhanced_data_config(
                    include_line_intersection=enhanced_config.get("include_line_intersection", True),
                    enhanced_data_enabled=enhanced_config.get("enabled", True),
                    fallback_on_error=enhanced_config.get("fallback_on_error", True)
                )
                self._log("✅ 客户端增强数据配置已应用")

        except Exception as e:
            self._log(f"❌ 应用客户端增强数据配置失败: {str(e)}", "ERROR")

    def _handle_line_intersection_update(self, line_intersection_result):
        """处理绿线交点检测结果更新"""
        try:
            logger.debug(f"Received line intersection update: {type(line_intersection_result)}")

            # 更新LineDetectionWidget（如果存在）
            if hasattr(self, 'line_detection_widget') and self.line_detection_widget:
                self.line_detection_widget.update_line_intersection_data(line_intersection_result)

            # 可以在这里添加其他绿线交点数据处理逻辑
            # 例如：状态显示、日志记录等

            if isinstance(line_intersection_result, dict):
                status = line_intersection_result.get('status', 'unknown')
                logger.debug(f"Line intersection status: {status}")

        except Exception as e:
            logger.error(f"Error handling line intersection update: {e}")

    def _backup_line_detection_config(self):
        """备份绿线检测配置"""
        try:
            if not self.http_client or not self.http_client.line_detection_config_loaded:
                messagebox.showerror("错误", "绿线检测配置未加载")
                return

            success = self.http_client._create_line_detection_config_backup()
            if success:
                messagebox.showinfo("成功", "绿线检测配置备份已创建")
            else:
                messagebox.showerror("错误", "配置备份创建失败")

        except Exception as e:
            messagebox.showerror("错误", f"备份配置时发生错误: {str(e)}")

    def _export_line_detection_config_dialog(self):
        """导出绿线检测配置对话框"""
        try:
            if not self.http_client or not self.http_client.line_detection_config_loaded:
                messagebox.showerror("错误", "绿线检测配置未加载")
                return

            from tkinter import filedialog
            from datetime import datetime

            # 默认文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"line_detection_config_{timestamp}.json"

            # 文件选择对话框
            file_path = filedialog.asksaveasfilename(
                title="导出绿线检测配置",
                defaultextension=".json",
                filetypes=[
                    ("JSON文件", "*.json"),
                    ("YAML文件", "*.yaml"),
                    ("CSV文件", "*.csv"),
                    ("所有文件", "*.*")
                ],
                initialfile=default_filename,
                initialdir="./exports/"
            )

            if file_path:  # 用户选择了文件
                success = self.http_client._export_line_detection_config(file_path)
                if success:
                    messagebox.showinfo("成功", f"配置已导出到: {file_path}")
                else:
                    messagebox.showerror("错误", "配置导出失败")

        except Exception as e:
            messagebox.showerror("错误", f"导出配置时发生错误: {str(e)}")

    def _reload_line_detection_config(self):
        """重新加载绿线检测配置"""
        try:
            if not self.http_client:
                messagebox.showerror("错误", "HTTP客户端未初始化")
                return

            # 重新加载配置
            if hasattr(self.http_client, '_load_line_detection_config'):
                success = self.http_client._load_line_detection_config()
            else:
                messagebox.showerror("错误", "绿线检测配置加载功能不可用")
                return
            if success:
                # 更新UI显示
                if hasattr(self, 'line_detection_widget') and self.line_detection_widget:
                    ui_config = self.http_client._get_line_detection_ui_config()
                    if ui_config.get("enable_widget", True):
                        self.line_detection_widget.apply_ui_config(ui_config)

                messagebox.showinfo("成功", "绿线检测配置已重新加载")
            else:
                messagebox.showerror("错误", "配置重新加载失败")

        except Exception as e:
            messagebox.showerror("错误", f"重新加载配置时发生错误: {str(e)}")


def main():
    """主函数"""
    app = HTTPRealtimeClientUI()
    app.mainloop()


if __name__ == "__main__":
    main()