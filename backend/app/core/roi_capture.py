"""
ROI截图服务模块
提供屏幕截图和ROI区域截取功能
"""

import base64
import io
import logging
import threading
import time
from typing import Optional, Tuple, Union

# 启用PIL导入
from PIL import Image, ImageGrab

# 导入OpenCV和numpy用于绿线检测
import cv2
import numpy as np

from ..models import RoiConfig, RoiData, LineIntersectionPoint

# 导入绿线检测算法
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../doc/ROI'))
from green_detector import detect_green_intersection


class RoiCaptureService:
    """ROI截图服务类"""

    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)
        # 从配置获取ROI帧率（配置现在会从JSON文件加载）
        from ..config import settings
        self._settings = settings
        self._frame_rate = settings.roi_frame_rate
        self._cache_interval = settings.roi_update_interval  # 缓存间隔

        # ROI截图缓存机制
        self._last_capture_time = 0.0
        self._cached_roi_data: Optional[RoiData] = None
        self._last_roi_config: Optional[RoiConfig] = None

        # 绿线交点缓存机制
        self._last_intersection_point: Optional[LineIntersectionPoint] = None
        self._intersection_cache_valid = False

        # ROI2专用缓存机制 - 解决闪烁问题
        self._cached_roi2_data: Optional[RoiData] = None
        self._last_roi2_config: Optional[RoiConfig] = None
        self._last_roi2_capture_time = 0.0
        self._roi2_cache_valid = False
        self._roi2_cache_duration = 1.0  # ROI2缓存时间（秒），比ROI1更长避免闪烁

        # ROI2坐标平滑机制已移除 - 使用真实检测数据
        # self._last_roi2_x: Optional[int] = None  # 移除历史坐标跟踪
        # self._last_roi2_y: Optional[int] = None  # 移除历史坐标跟踪
        # self._roi2_smoothing_factor = 0.8       # 移除平滑因子

        # 定时器方案 - 替代复杂缓存机制
        self._roi_timer_thread: Optional[threading.Thread] = None
        self._stop_timer_event = threading.Event()
        self._latest_roi1_data: Optional[RoiData] = None
        self._latest_roi2_data: Optional[RoiData] = None
        self._roi_lock = threading.RLock()  # 线程安全的数据访问
        self._use_timer_mode = True  # 默认启用定时器模式

        self._logger.info("ROI Capture Service initialized with JSON config: frame_rate=%d, update_interval=%.1f, roi2_cache=%.1f, timer_mode=%s",
                         self._frame_rate, self._cache_interval, self._roi2_cache_duration, self._use_timer_mode)

    def clear_cache(self):
        """
        清除ROI截图缓存，强制下次截图时重新捕获
        """
        self._cached_roi_data = None
        self._last_roi_config = None
        self._last_capture_time = 0.0

        # 清除ROI2缓存
        self._cached_roi2_data = None
        self._last_roi2_config = None
        self._last_roi2_capture_time = 0.0
        self._roi2_cache_valid = False

        # 坐标平滑机制已移除 - 无需清除历史坐标缓存

        # 清除日志跟踪变量
        if hasattr(self, '_last_logged_roi2_x'):
            delattr(self, '_last_logged_roi2_x')
        if hasattr(self, '_last_logged_roi2_y'):
            delattr(self, '_last_logged_roi2_y')

        self._logger.debug("ROI and ROI2 cache cleared - next capture will be forced")

    def _update_intersection_cache(self, intersection_point: Optional[LineIntersectionPoint]):
        """更新绿线交点缓存"""
        if intersection_point is not None:
            self._last_intersection_point = intersection_point
            self._intersection_cache_valid = True
            self._logger.debug(f"Intersection cache updated: ROI({intersection_point.roi_x}, {intersection_point.roi_y}) Screen({intersection_point.x}, {intersection_point.y})")
        else:
            # 检测失败，保持现有缓存
            self._logger.debug("No intersection detected, keeping existing cache")

    def _invalidate_intersection_cache(self):
        """清除绿线交点缓存"""
        self._last_intersection_point = None
        self._intersection_cache_valid = False
        self._logger.debug("Intersection cache invalidated")

    def get_last_intersection_point(self) -> Optional[LineIntersectionPoint]:
        """获取最后缓存的交点坐标"""
        return self._last_intersection_point if self._intersection_cache_valid else None

    def _is_roi2_cache_valid(self, roi_config: RoiConfig, current_time: float) -> bool:
        """检查ROI2缓存是否有效"""
        if not self._roi2_cache_valid or self._cached_roi2_data is None:
            return False

        # 检查时间有效性
        time_valid = (current_time - self._last_roi2_capture_time) < self._roi2_cache_duration
        if not time_valid:
            self._logger.debug("ROI2 cache expired")
            return False

        # 检查配置变化
        config_unchanged = (self._last_roi2_config is not None and
                           self._roi_config_changed(roi_config, self._last_roi2_config) == False)
        if not config_unchanged:
            self._logger.debug("ROI2 cache invalid due to config change")
            return False

        return True

    def _update_roi2_cache(self, roi_config: RoiConfig, roi2_data: RoiData, current_time: float):
        """更新ROI2缓存"""
        self._cached_roi2_data = roi2_data
        self._last_roi2_config = roi_config
        self._last_roi2_capture_time = current_time
        self._roi2_cache_valid = True
        # 移除缓存更新日志，避免刷屏

    def _smooth_roi2_coordinates(self, new_x: int, new_y: int) -> Tuple[int, int]:
        """ROI2坐标平滑已移除 - 直接返回真实检测数据"""
        # 平滑算法已移除，直接返回原始坐标以获得真实检测数据
        # 这消除了5-6帧的收敛延迟，ROI2将立即响应交点变化
        return new_x, new_y

    def capture_screen(self) -> Optional[Image.Image]:
        """
        截取整个屏幕

        Returns:
            PIL.Image: 屏幕截图，失败返回None
        """
        try:
            screenshot = ImageGrab.grab()
            self._logger.debug("Screen captured successfully, size: %s", screenshot.size)
            return screenshot
        except Exception as e:
            self._logger.error("Failed to capture screen: %s", str(e))
            return None

    def capture_roi(self, roi_config: RoiConfig) -> Optional[RoiData]:
        """
        截取指定ROI区域（带缓存机制）

        Args:
            roi_config: ROI配置

        Returns:
            RoiData: ROI数据，失败返回None
        """
        try:
            # 验证ROI坐标
            if not roi_config.validate_coordinates():
                self._logger.error("Invalid ROI coordinates: %s", roi_config)
                return None

            current_time = time.time()

            # 简化的缓存机制：只基于时间间隔和配置变化
            time_valid = current_time - self._last_capture_time < self._cache_interval
            config_unchanged = (self._last_roi_config is not None and
                               self._roi_config_changed(roi_config, self._last_roi_config) == False)

            # 只有在缓存有效且配置未变化时才使用缓存
            if (self._cached_roi_data is not None and time_valid and config_unchanged):
                self._logger.debug(f"Using cached ROI data (age: {current_time - self._last_capture_time:.3f}s)")
                return self._cached_roi_data
            else:
                self._logger.debug(f"Forcing new ROI capture - time_valid: {time_valid}, config_unchanged: {config_unchanged}")

            # 执行真实的截图操作
            roi_result = self._capture_roi_internal(roi_config)

            # 更新缓存和状态
            if roi_result is not None:
                roi_data, roi_image = roi_result
                self._cached_roi_data = roi_data
                self._last_roi_config = roi_config
                self._last_capture_time = current_time
                self._logger.debug("ROI captured successfully (gray_value=%.2f)", roi_data.gray_value)

                # 集成历史存储 - 保存ROI帧到DataStore
                try:
                    from ..core.data_store import data_store
                    # 获取当前主信号帧数
                    _, main_frame_count, _, _, _, _ = data_store.get_status_snapshot()

                    # 添加ROI历史帧
                    roi_frame = data_store.add_roi_frame(
                        gray_value=roi_data.gray_value,
                        roi_config=roi_config,
                        frame_count=main_frame_count,
                        capture_duration=self._cache_interval
                    )

                    # 减少日志频率 - 每50帧记录一次，并改为debug级别
                    if roi_frame.index % 50 == 0:
                        self._logger.debug("ROI frame added to history: index=%d, gray_value=%.2f, main_frame=%d",
                                           roi_frame.index, roi_frame.gray_value, main_frame_count)

                except Exception as e:
                    self._logger.error("Failed to add ROI frame to history: %s", str(e))

            return roi_data

        except Exception as e:
            self._logger.error("Failed to capture ROI: %s", str(e))
            return None

    def _roi_config_changed(self, current: RoiConfig, cached: RoiConfig) -> bool:
        """检查ROI配置是否发生变化"""
        changed = (current.x1 != cached.x1 or current.y1 != cached.y1 or
                  current.x2 != cached.x2 or current.y2 != cached.y2)

        if changed:
            self._invalidate_intersection_cache()  # ROI变化时清除交点缓存

        return changed

    def _capture_roi_internal(self, roi_config: RoiConfig) -> Optional[Tuple[RoiData, Image.Image]]:
        """执行实际的ROI截图操作"""
        # 首先截取整个屏幕
        screen = self.capture_screen()
        if screen is None:
            self._logger.error("Failed to capture screen for ROI")
            return None

        # 检查ROI是否在屏幕范围内
        screen_width, screen_height = screen.size
        if (roi_config.x2 > screen_width or roi_config.y2 > screen_height or
            roi_config.x1 < 0 or roi_config.y1 < 0):
            self._logger.warning(
                "ROI coordinates exceed screen bounds. Screen: %dx%d, ROI: (%d,%d)->(%d,%d)",
                screen_width, screen_height,
                roi_config.x1, roi_config.y1, roi_config.x2, roi_config.y2
            )
            # 自动调整到屏幕范围内
            x1 = max(0, min(roi_config.x1, screen_width - 1))
            y1 = max(0, min(roi_config.y1, screen_height - 1))
            x2 = max(x1 + 1, min(roi_config.x2, screen_width))
            y2 = max(y1 + 1, min(roi_config.y2, screen_height))
        else:
            x1, y1, x2, y2 = roi_config.x1, roi_config.y1, roi_config.x2, roi_config.y2

        # 截取ROI区域
        roi_image = screen.crop((x1, y1, x2, y2))

        # 计算ROI平均灰度值
        gray_roi = roi_image.convert('L')
        # 简化计算：使用PIL的直方图来计算平均值
        histogram = gray_roi.histogram()
        total_pixels = roi_config.width * roi_config.height
        total_sum = sum(i * count for i, count in enumerate(histogram))
        average_gray = float(total_sum / total_pixels) if total_pixels > 0 else 0.0

        # 集成绿线交点检测 - 使用原始ROI数据
        intersection_point = None
        detection_start_time = time.time()
        try:
            # 性能优化: 检查ROI尺寸，如果过大则进行智能降采样
            roi_width, roi_height = roi_image.size
            max_detection_size = 1500  # 最大检测尺寸，超过则降采样

            detection_image = roi_image
            scale_factor = 1.0

            if roi_width > max_detection_size or roi_height > max_detection_size:
                # 计算缩放比例
                scale_factor = min(max_detection_size / roi_width, max_detection_size / roi_height)
                new_width = int(roi_width * scale_factor)
                new_height = int(roi_height * scale_factor)
                detection_image = roi_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                self._logger.debug(f"ROI detection scaled from {roi_width}x{roi_height} to {new_width}x{new_height}")

            # 将PIL图像转换为OpenCV格式进行检测
            roi_cv_image = cv2.cvtColor(np.array(detection_image), cv2.COLOR_RGB2BGR)

            # 调用绿线检测算法 - 使用(可能缩放后的)ROI数据
            intersection = detect_green_intersection(roi_cv_image)

            if intersection is not None:
                roi_x, roi_y = intersection

                # 如果进行了缩放，需要将坐标转换回原始ROI尺寸
                if scale_factor != 1.0:
                    roi_x = int(roi_x / scale_factor)
                    roi_y = int(roi_y / scale_factor)

                # 转换为屏幕坐标
                screen_x = roi_config.x1 + roi_x
                screen_y = roi_config.y1 + roi_y

                # 打印到控制台 - 显示屏幕坐标和ROI内坐标
                #print(f"ROI1 Green Line Intersection: Screen({screen_x}, {screen_y}) ROI({roi_x}, {roi_y})")

                # 创建交点数据模型 - 包含屏幕坐标和ROI内坐标
                intersection_point = LineIntersectionPoint(
                    x=screen_x,
                    y=screen_y,
                    roi_x=roi_x,
                    roi_y=roi_y,
                    confidence=1.0
                )
                #self._logger.debug(f"Green line intersection detected at Screen({screen_x}, {screen_y}) ROI({roi_x}, {roi_y})")

                # 更新交点缓存
                self._update_intersection_cache(intersection_point)
            else:
                #print("ROI1: No green line intersection detected")
                #self._logger.debug("No green line intersection detected")

                # 检测失败时，保持现有缓存，记录日志
                if self._intersection_cache_valid:
                    self._logger.debug(f"Using cached intersection point: ROI({self._last_intersection_point.roi_x}, {self._last_intersection_point.roi_y})")
                else:
                    self._logger.debug("No cached intersection point available")

        except Exception as e:
            self._logger.error(f"Green line detection failed: {str(e)}")
            print("ROI1: Green line detection error")

        # 性能监控
        detection_time = (time.time() - detection_start_time) * 1000  # 转换为毫秒
        self._logger.debug(f"Green line detection completed in {detection_time:.2f}ms")

        # 如果检测时间超过250ms（4 FPS目标），记录警告
        if detection_time > 250:
            self._logger.warning(f"Green line detection took {detection_time:.2f}ms - may affect 4 FPS target")

        # 调整ROI图像大小到标准尺寸（200x150）- 用于显示
        try:
            roi_resized = roi_image.resize((200, 150), Image.Resampling.LANCZOS)
        except AttributeError:
            # 兼容旧版本PIL
            roi_resized = roi_image.resize((200, 150), Image.LANCZOS)

        # 转换为base64
        buffer = io.BytesIO()
        roi_resized.save(buffer, format='PNG')
        roi_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        roi_data = RoiData(
            width=roi_config.width,
            height=roi_config.height,
            pixels=f"data:image/png;base64,{roi_base64}",
            gray_value=average_gray,
            format="base64",
            intersection=intersection_point
        )

        self._logger.debug(
            "ROI captured successfully: size=%dx%d, gray_value=%.2f, base64_length=%d",
            roi_config.width, roi_config.height, average_gray, len(roi_base64)
        )

        return roi_data, roi_image

    # 定时器方案方法
    def start_roi_timer(self):
        """启动ROI定时器"""
        if not self._use_timer_mode:
            self._logger.debug("Timer mode is disabled, using cache mode")
            return

        if self._roi_timer_thread and self._roi_timer_thread.is_alive():
            self._logger.debug("ROI timer is already running")
            return

        self._stop_timer_event.clear()
        self._roi_timer_thread = threading.Thread(
            target=self._roi_timer_loop,
            daemon=True
        )
        self._roi_timer_thread.start()
        self._logger.info("ROI timer started at %d FPS", self._frame_rate)

    def stop_roi_timer(self):
        """停止ROI定时器"""
        if not self._use_timer_mode:
            return

        self._stop_timer_event.set()
        if self._roi_timer_thread and self._roi_timer_thread.is_alive():
            self._roi_timer_thread.join(timeout=2.0)
        self._logger.info("ROI timer stopped")

    def get_latest_roi_data(self) -> Tuple[Optional[RoiData], Optional[RoiData]]:
        """获取最新的ROI数据（线程安全）"""
        with self._roi_lock:
            return self._latest_roi1_data, self._latest_roi2_data

    def _roi_timer_loop(self):
        """ROI定时器循环"""
        interval = 1.0 / self._frame_rate if self._frame_rate > 0 else 1.0
        self._logger.info("ROI timer loop started with interval: %.3fs (%d FPS)", interval, self._frame_rate)

        while not self._stop_timer_event.is_set():
            start_time = time.perf_counter()

            try:
                # 执行双ROI捕获（定时器模式，无缓存机制）
                roi1_data, roi2_data = self._capture_dual_roi_timer()

                # 线程安全地更新最新数据
                with self._roi_lock:
                    self._latest_roi1_data = roi1_data
                    self._latest_roi2_data = roi2_data

                if roi1_data and roi2_data:
                    self._logger.debug("ROI timer update: ROI1=%.2f, ROI2=%.2f",
                                      roi1_data.gray_value, roi2_data.gray_value)

            except Exception as e:
                self._logger.error("ROI timer loop error: %s", str(e))
                # 发生错误时不更新数据，继续循环

            # 计算等待时间，确保精确的频率控制
            elapsed = time.perf_counter() - start_time
            sleep_time = max(0, interval - elapsed)

            # 使用可中断的sleep
            if self._stop_timer_event.wait(sleep_time):
                break

        self._logger.info("ROI timer loop stopped")

    def _capture_dual_roi_timer(self) -> Tuple[Optional[RoiData], Optional[RoiData]]:
        """定时器模式下的双ROI捕获（无缓存机制）"""
        try:
            # 获取当前ROI配置
            from ..core.data_store import data_store
            roi_config = data_store.get_roi_config()

            if not roi_config or not roi_config.validate_coordinates():
                self._logger.warning("Invalid ROI configuration in timer mode")
                return None, None

            # 直接执行ROI1捕获
            roi1_result = self._capture_roi_internal(roi_config)
            if roi1_result is None:
                self._logger.error("ROI1 capture failed in timer mode")
                return None, None

            roi1_data, roi1_image = roi1_result

            # 直接从ROI1提取ROI2（使用原始图像）
            roi2_data = self._extract_roi2_from_roi1(
                roi_config, roi1_data, roi1_data.intersection, roi1_image
            )

            return roi1_data, roi2_data

        except Exception as e:
            self._logger.error("Dual ROI capture failed in timer mode: %s", str(e))
            return None, None

    def capture_dual_roi(self, roi_config: RoiConfig) -> Tuple[Optional[RoiData], Optional[RoiData]]:
        """
        截取双ROI区域：ROI1为原始配置区域，ROI2为从ROI1中心截取的50x50区域

        Args:
            roi_config: ROI1配置

        Returns:
            Tuple[Optional[RoiData], Optional[RoiData]]: (ROI1数据, ROI2数据)，失败返回None
        """
        # 定时器模式：直接返回定时器中的最新数据
        if self._use_timer_mode:
            return self.get_latest_roi_data()

        # 缓存模式：原有的复杂缓存逻辑
        try:
            # 验证ROI1坐标
            if not roi_config.validate_coordinates():
                self._logger.error("Invalid ROI1 coordinates: %s", roi_config)
                return None, None

            # 检查ROI1是否足够大以包含50x50的ROI2
            if roi_config.width < 50 or roi_config.height < 50:
                self._logger.error("ROI1 too small for dual ROI: size=%dx%d, minimum 50x50 required",
                                 roi_config.width, roi_config.height)
                return None, None

            current_time = time.time()

            # ROI1缓存检查（原有逻辑）
            roi1_time_valid = current_time - self._last_capture_time < self._cache_interval
            roi1_config_unchanged = (self._last_roi_config is not None and
                                   self._roi_config_changed(roi_config, self._last_roi_config) == False)

            # ROI2缓存检查（新增独立缓存逻辑）
            roi2_cache_valid = self._is_roi2_cache_valid(roi_config, current_time)

            # 如果ROI1和ROI2缓存都有效，直接返回缓存数据
            if (self._cached_roi_data is not None and roi1_time_valid and roi1_config_unchanged and roi2_cache_valid):
                roi1_data = self._cached_roi_data
                roi2_data = self._cached_roi2_data
                self._logger.debug(f"Using cached dual ROI data (ROI1 age: {current_time - self._last_capture_time:.3f}s, "
                                 f"ROI2 age: {current_time - self._last_roi2_capture_time:.3f}s)")
                return roi1_data, roi2_data

            # 如果只有ROI1缓存有效，检查ROI2是否需要更新
            elif (self._cached_roi_data is not None and roi1_time_valid and roi1_config_unchanged):
                roi1_data = self._cached_roi_data

                # ROI2缓存逻辑：检查是否需要更新ROI2
                roi2_time_valid = (self._last_roi2_capture_time > 0 and
                                  current_time - self._last_roi2_capture_time < self._roi2_cache_duration)

                if roi2_cache_valid or roi2_time_valid:
                    # ROI2缓存有效，直接返回
                    roi2_data = self._cached_roi2_data if self._cached_roi2_data else None
                    self._logger.debug(f"Using cached ROI1 and ROI2 (ROI2 age: {current_time - self._last_roi2_capture_time:.3f}s)")
                    return roi1_data, roi2_data
                else:
                    # ROI2缓存过期，需要重新提取
                    self._logger.debug(f"ROI2 cache expired, extracting from cached ROI1")
                    try:
                        roi2_data = self._extract_roi2_from_roi1(roi_config, roi1_data, roi1_data.intersection)
                        # 更新ROI2缓存
                        self._update_roi2_cache(roi_config, roi2_data, current_time)
                        return roi1_data, roi2_data
                    except Exception as e:
                        self._logger.error(f"Failed to extract ROI2 from cached ROI1: {e}")
                        return roi1_data, None

            # 如果缓存都无效或配置变化，执行完整捕获
            else:
                self._logger.debug(f"Performing full dual ROI capture - ROI1_valid: {roi1_time_valid}, "
                                 f"ROI1_config_unchanged: {roi1_config_unchanged}, ROI2_valid: {roi2_cache_valid}")

            # 执行真实双ROI截图操作
            roi1_data, roi2_data = self._capture_dual_roi_internal(roi_config)

            # 更新缓存和状态（ROI1和ROI2都缓存）
            if roi1_data is not None:
                # 更新ROI1缓存（原有逻辑）
                self._cached_roi_data = roi1_data
                self._last_roi_config = roi_config
                self._last_capture_time = current_time

                # 更新ROI2缓存（新增逻辑）
                if roi2_data is not None:
                    self._update_roi2_cache(roi_config, roi2_data, current_time)

                self._logger.debug("Dual ROI captured successfully (ROI1 gray=%.2f, ROI2 gray=%.2f)",
                                 roi1_data.gray_value, roi2_data.gray_value if roi2_data else 0.0)

                # 集成历史存储 - 保存ROI2帧到DataStore（用于峰值检测）
                try:
                    from ..core.data_store import data_store
                    # 获取当前主信号帧数
                    _, main_frame_count, _, _, _, _ = data_store.get_status_snapshot()

                    # 添加ROI2历史帧（用于峰值检测）
                    if roi2_data:
                        # 从ROI1交点信息计算ROI2坐标
                        roi2_x1 = roi1_data.intersection.roi_x if roi1_data.intersection and roi1_data.intersection.roi_x is not None else None
                        roi2_y1 = roi1_data.intersection.roi_y if roi1_data.intersection and roi1_data.intersection.roi_y is not None else None

                        if roi2_x1 is not None and roi2_y1 is not None:
                            roi2_config = self._calculate_roi2_screen_coordinates(roi_config, roi2_x1, roi2_y1)
                        else:
                            # Fallback to center-based calculation
                            roi1_center_x = roi_config.width // 2
                            roi1_center_y = roi_config.height // 2
                            roi2_size = 50
                            roi2_x1_fallback = roi1_center_x - roi2_size // 2
                            roi2_y1_fallback = roi1_center_y - roi2_size // 2
                            roi2_config = self._calculate_roi2_screen_coordinates(roi_config, roi2_x1_fallback, roi2_y1_fallback)

                        roi2_frame = data_store.add_roi_frame(
                            gray_value=roi2_data.gray_value,
                            roi_config=roi2_config,
                            frame_count=main_frame_count,
                            capture_duration=self._cache_interval
                        )

                        # 减少日志频率 - 每50帧记录一次
                        if roi2_frame.index % 50 == 0:
                            self._logger.debug("ROI2 frame added to history: index=%d, gray_value=%.2f, main_frame=%d",
                                               roi2_frame.index, roi2_data.gray_value, main_frame_count)

                except Exception as e:
                    self._logger.error("Failed to add ROI2 frame to history: %s", str(e))

            return roi1_data, roi2_data

        except Exception as e:
            self._logger.error("Failed to capture dual ROI: %s", str(e))
            return None, None

    def _extract_roi2_from_roi1(self, roi1_config: RoiConfig, roi1_data: RoiData,
                                intersection_point: Optional[LineIntersectionPoint] = None,
                                roi1_original_image: Optional[Image.Image] = None) -> Optional[RoiData]:
        """从ROI1原始图像中提取ROI2（50x50区域，基于绿线交点或中心点）"""
        try:
            self._logger.debug(f"Extracting ROI2 from original ROI1 image: ROI1 config=({roi1_config.x1},{roi1_config.y1})->({roi1_config.x2},{roi1_config.y2}), "
                             f"size={roi1_config.width}x{roi1_config.height}")

            # ROI2固定尺寸: 50x50
            roi2_size = 50
            half_size = roi2_size // 2

            # 确定ROI2中心点（使用ROI内坐标）
            if intersection_point is not None and intersection_point.roi_x is not None and intersection_point.roi_y is not None:
                # 使用交点坐标
                center_x = intersection_point.roi_x
                center_y = intersection_point.roi_y
                source = "intersection"
                self._logger.debug(f"Using intersection point for ROI2: ROI({center_x}, {center_y})")
            elif self._intersection_cache_valid and self._last_intersection_point is not None:
                # 使用缓存的交点坐标
                center_x = self._last_intersection_point.roi_x
                center_y = self._last_intersection_point.roi_y
                source = "cached_intersection"
                self._logger.debug(f"Using cached intersection point for ROI2: ROI({center_x}, {center_y})")
            else:
                # Fallback: 使用ROI1中心点
                center_x = roi1_config.width // 2
                center_y = roi1_config.height // 2
                source = "center"
                self._logger.debug(f"Using ROI1 center for ROI2: ({center_x}, {center_y})")

            # 平滑算法已移除 - 直接使用检测到的真实坐标
            # ROI2将立即响应交点变化，无收敛延迟
            self._logger.debug(f"ROI2 using raw detection coordinates: ({center_x}, {center_y}), source: {source}")

            # ROI2的起始和结束坐标（在ROI1内）
            roi2_x1 = center_x - half_size
            roi2_y1 = center_y - half_size
            roi2_x2 = center_x + half_size
            roi2_y2 = center_y + half_size

            # 边界检查 - 如果超出ROI1边界，返回错误状态
            if (roi2_x1 < 0 or roi2_y1 < 0 or
                roi2_x2 > roi1_config.width or roi2_y2 > roi1_config.height):
                self._logger.warning(f"ROI2 {roi2_size}x{roi2_size} region at ({center_x}, {center_y}) exceeds ROI1 bounds: {roi1_config.width}x{roi1_config.height}")
                self._logger.warning(f"ROI2 coordinates: ({roi2_x1}, {roi2_y1}) → ({roi2_x2}, {roi2_y2})")
                raise ValueError("ROI2 region exceeds ROI1 boundaries")

            # ROI2坐标验证已通过边界检查，确保是精确的50x50尺寸
            self._logger.debug(f"ROI2 ROI coordinates calculated: ({roi2_x1},{roi2_y1})->({roi2_x2},{roi2_y2}) "
                             f"size={(roi2_x2-roi2_x1)}x{(roi2_y2-roi2_y1)}")

            # 使用原始ROI1图像而不是解码base64数据
            if roi1_original_image is None:
                self._logger.warning("Original ROI1 image not provided, falling back to base64 decoded image")
                # 回退到原来的逻辑
                import base64
                from io import BytesIO

                # 提取base64数据
                if roi1_data.pixels.startswith('data:image/png;base64,'):
                    base64_data = roi1_data.pixels.replace('data:image/png;base64,', '')
                else:
                    base64_data = roi1_data.pixels

                # 解码图像
                image_data = base64.b64decode(base64_data)
                roi1_image = Image.open(BytesIO(image_data))
            else:
                roi1_image = roi1_original_image
                self._logger.debug(f"Using original ROI1 image: size={roi1_image.size}, mode={roi1_image.mode}")

            # 从原始图像中直接截取ROI2区域（使用ROI内坐标）
            # 由于roi1_image是ROI1区域的原始图像，可以直接使用ROI内坐标
            roi2_image = roi1_image.crop((roi2_x1, roi2_y1, roi2_x2, roi2_y2))

            # 添加ROI2图像调试日志
            roi2_actual_size = roi2_image.size
            roi2_mode = roi2_image.mode
            self._logger.debug(f"ROI2 image cropped from original: size={roi2_actual_size}, mode={roi2_mode}")

            # 计算ROI2平均灰度值
            gray_roi2 = roi2_image.convert('L')
            histogram = gray_roi2.histogram()
            total_pixels = roi2_size * roi2_size  # 固定50x50 = 2500像素
            total_sum = sum(i * count for i, count in enumerate(histogram))
            average_gray = float(total_sum / total_pixels) if total_pixels > 0 else 0.0

            # 检查ROI2图像质量
            non_zero_pixels = sum(1 for p in gray_roi2.getdata() if p > 0)
            if non_zero_pixels == 0:
                self._logger.warning(f"ROI2 image is completely black: {total_pixels} pixels, average_gray={average_gray:.2f}")
            else:
                self._logger.debug(f"ROI2 image quality: {non_zero_pixels}/{total_pixels} non-zero pixels, average_gray={average_gray:.2f}")

            # 调整ROI2图像大小到标准尺寸（200x150用于显示）
            try:
                roi2_resized = roi2_image.resize((200, 150), Image.Resampling.LANCZOS)
            except AttributeError:
                roi2_resized = roi2_image.resize((200, 150), Image.LANCZOS)

            # 转换为base64
            import base64
            from io import BytesIO
            buffer = BytesIO()
            roi2_resized.save(buffer, format='PNG')
            roi2_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            # 添加ROI2编码调试日志
            roi2_base64_length = len(roi2_base64)
            self._logger.debug(f"ROI2 encoded: resized_size={roi2_resized.size}, base64_length={roi2_base64_length}")

            roi2_data = RoiData(
                width=roi2_size,  # 固定50像素宽度
                height=roi2_size,  # 固定50像素高度
                pixels=f"data:image/png;base64,{roi2_base64}",
                gray_value=average_gray,
                format="base64"
            )

            self._logger.debug("ROI2 extracted from original ROI1 image: size=%dx%d, gray_value=%.2f, non_zero_ratio=%.2f%%",
                             roi2_size, roi2_size, average_gray, (non_zero_pixels/total_pixels)*100)

            # ROI2状态监控日志（仅在坐标发生很大变化时才输出，正常情况下静默）
            coord_changed = (hasattr(self, '_last_logged_roi2_x') and
                           (abs(self._last_logged_roi2_x - center_x) > 20 or
                            abs(self._last_logged_roi2_y - center_y) > 20))

            if coord_changed or not hasattr(self, '_last_logged_roi2_x'):
                self._logger.debug(f"ROI2 Position Changed - ROI({center_x}, {center_y}) -> Screen({roi1_config.x1 + center_x}, {roi1_config.y1 + center_y}), "
                                 f"Quality: {average_gray:.1f}, Source: {source}")
                self._last_logged_roi2_x = center_x
                self._last_logged_roi2_y = center_y
            # 正常情况下不再输出ROI2状态日志，避免刷屏

            return roi2_data

        except Exception as e:
            self._logger.error("Failed to extract ROI2 from ROI1: %s", str(e))
            return None

    def _capture_dual_roi_internal(self, roi_config: RoiConfig) -> Tuple[Optional[RoiData], Optional[RoiData]]:
        """执行实际的双ROI截图操作 - 统一从ROI1图像中提取ROI2"""
        dual_roi_start_time = time.time()
        self._logger.debug("Starting unified dual ROI capture - always extract ROI2 from ROI1 image")

        # 首先捕获ROI1（包含原始图像）
        roi1_result = self._capture_roi_internal(roi_config)
        if roi1_result is None:
            self._logger.error("Failed to capture ROI1, cannot extract ROI2")
            return None, None

        roi1_data, roi1_image = roi1_result

        # 然后从ROI1原始图像中提取ROI2 - 基于交点坐标
        roi2_extraction_start_time = time.time()
        try:
            roi2_data = self._extract_roi2_from_roi1(roi_config, roi1_data, roi1_data.intersection, roi1_image)

            # 记录ROI2提取成功信息
            roi2_extraction_time = (time.time() - roi2_extraction_start_time) * 1000
            self._logger.debug(f"ROI2 extraction completed in {roi2_extraction_time:.2f}ms")

            # 根据交点状态记录详细信息
            if roi1_data.intersection and roi1_data.intersection.roi_x is not None:
                self._logger.debug(f"ROI2 extracted based on intersection: ROI({roi1_data.intersection.roi_x}, {roi1_data.intersection.roi_y})")
            elif self._intersection_cache_valid:
                self._logger.debug(f"ROI2 extracted based on cached intersection: ROI({self._last_intersection_point.roi_x}, {self._last_intersection_point.roi_y})")
            else:
                self._logger.debug("ROI2 extracted based on ROI1 center (no intersection available)")

        except ValueError as e:
            # ROI2截取失败（如超出边界），创建错误状态
            roi2_extraction_time = (time.time() - roi2_extraction_start_time) * 1000
            self._logger.error(f"ROI2 extraction failed after {roi2_extraction_time:.2f}ms: {e}")

            roi2_data = RoiData(
                width=50,
                height=50,
                pixels="roi2_extraction_failed",
                gray_value=roi1_data.gray_value,  # 使用ROI1灰度值作为fallback
                format="text",
                intersection=None
            )
        if roi2_data is None:
            self._logger.error("Failed to extract ROI2 from ROI1 image")
            # 仍然返回ROI1数据，让系统可以继续工作
            return roi1_data, None

        # 记录整体性能指标
        dual_roi_total_time = (time.time() - dual_roi_start_time) * 1000
        self._logger.debug(
            "Unified dual ROI capture successful: ROI1=%.2f, ROI2=%.2f, total_time=%.2fms",
            roi1_data.gray_value, roi2_data.gray_value, dual_roi_total_time
        )

        # 性能监控 - 如果总时间过长，发出警告
        if dual_roi_total_time > 300:  # 300ms阈值（比单独检测更宽松）
            self._logger.warning(f"Dual ROI capture took {dual_roi_total_time:.2f}ms - may affect performance")

        return roi1_data, roi2_data

    def _calculate_roi2_screen_coordinates(self, roi1_config: RoiConfig, roi2_x1: int, roi2_y1: int) -> RoiConfig:
        """计算ROI2在屏幕上的实际坐标"""
        roi2_size = 50

        # ROI2坐标在屏幕上的实际位置
        screen_x1 = roi1_config.x1 + roi2_x1
        screen_y1 = roi1_config.y1 + roi2_y1
        screen_x2 = screen_x1 + roi2_size
        screen_y2 = screen_y1 + roi2_size

        return RoiConfig(x1=screen_x1, y1=screen_y1, x2=screen_x2, y2=screen_y2)

    def get_roi_frame_rate(self) -> int:
        """获取当前ROI帧率设置"""
        return self._frame_rate

    def set_roi_frame_rate(self, frame_rate: int) -> bool:
        """
        动态设置ROI帧率

        Args:
            frame_rate: 新的帧率 (1-60 FPS)

        Returns:
            bool: 设置是否成功
        """
        if 1 <= frame_rate <= 60:
            self._frame_rate = frame_rate
            self._cache_interval = 1.0 / frame_rate
            self._logger.info("ROI frame rate updated to %d FPS, cache interval: %.3f seconds",
                              frame_rate, self._cache_interval)

            # 保存到JSON配置文件
            try:
                from .config_manager import get_config_manager
                config_manager = get_config_manager()

                # 更新ROI帧率配置
                updates = {"frame_rate": frame_rate}
                success = config_manager.update_config(updates, section="roi_capture")
                if success:
                    config_manager.save_config()
                    self._logger.info("ROI frame rate %d saved to JSON configuration file", frame_rate)
                else:
                    self._logger.warning("Failed to save ROI frame rate to JSON configuration file")

            except Exception as e:
                self._logger.error("Error saving ROI frame rate to JSON: %s", str(e))

            return True
        else:
            self._logger.error("Invalid frame rate: %d (must be 1-60)", frame_rate)
            return False

    def get_screen_resolution(self) -> Tuple[int, int]:
        """
        获取屏幕分辨率

        Returns:
            Tuple[int, int]: (宽度, 高度)
        """
        try:
            screen = self.capture_screen()
            if screen:
                return screen.size
            return (1920, 1080)  # 默认分辨率
        except Exception:
            return (1920, 1080)  # 默认分辨率

    def validate_roi_coordinates(self, roi_config: RoiConfig) -> Tuple[bool, str]:
        """
        验证ROI坐标是否有效

        Args:
            roi_config: ROI配置

        Returns:
            Tuple[bool, str]: (是否有效, 错误信息)
        """
        try:
            # 基本坐标验证
            if not roi_config.validate_coordinates():
                return False, "Invalid coordinates: x1 must be < x2 and y1 must be < y2"

            # 获取屏幕分辨率
            screen_width, screen_height = self.get_screen_resolution()

            # 检查坐标范围
            if roi_config.x1 < 0 or roi_config.y1 < 0:
                return False, "Coordinates cannot be negative"

            if roi_config.x2 > screen_width or roi_config.y2 > screen_height:
                return False, f"Coordinates exceed screen resolution ({screen_width}x{screen_height})"

            # 检查ROI大小
            if roi_config.width < 10 or roi_config.height < 10:
                return False, "ROI size too small (minimum 10x10)"

            if roi_config.width > 1000 or roi_config.height > 1000:
                return False, "ROI size too large (maximum 1000x1000)"

            return True, "Valid ROI coordinates"

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def reload_config(self) -> bool:
        """
        从JSON配置文件重新加载ROI配置

        Returns:
            bool: 重新加载是否成功
        """
        try:
            # 重新加载settings对象（这会从JSON文件读取最新配置）
            from ..config import AppConfig
            new_settings = AppConfig.reload_from_json()

            if new_settings:
                # 更新本地配置
                old_frame_rate = self._frame_rate
                old_interval = self._cache_interval

                self._settings = new_settings
                self._frame_rate = new_settings.roi_frame_rate
                self._cache_interval = new_settings.roi_update_interval

                self._logger.info(
                    "ROI config reloaded from JSON: frame_rate %d->%d, interval %.1f->%.1f",
                    old_frame_rate, self._frame_rate, old_interval, self._cache_interval
                )
                return True
            else:
                self._logger.error("Failed to reload ROI config from JSON")
                return False

        except Exception as e:
            self._logger.error("Error reloading ROI config: %s", str(e))
            return False


# 单例ROI截图服务
roi_capture_service = RoiCaptureService()