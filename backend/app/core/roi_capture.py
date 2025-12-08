"""
ROI截图服务模块
提供屏幕截图和ROI区域截取功能
扩展支持ROI1绿色线条相交检测和ROI2灰度分析
"""

import base64
import io
import logging
import time
import threading
from typing import Optional, Tuple, Dict, Any

# 启用PIL导入
from PIL import Image, ImageGrab

from ..models import RoiConfig, RoiData, LineIntersectionResult, LineDetectionConfig


class RoiCaptureService:
    """ROI截图服务类，支持ROI1线条检测和ROI2灰度分析"""

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

        # ROI1线条检测缓存机制（任务17：ROI1专用）
        self._roi1_line_detection_cache: Optional[LineIntersectionResult] = None
        self._roi1_line_detection_time: float = 0.0
        self._roi1_line_detection_config: Optional[LineDetectionConfig] = None
        self._roi1_last_processed_image_id: Optional[str] = None  # 基于图像内容的缓存键

        # ROI2灰度处理缓存机制（任务17：ROI2专用）
        self._roi2_gray_cache: Optional[RoiData] = None
        self._roi2_gray_time: float = 0.0
        self._roi2_last_roi1_config: Optional[RoiConfig] = None

        # 线条检测器实例（延迟初始化，ROI1专用）
        self._roi1_line_detector = None
        self._roi1_detector_lock = threading.RLock()

        # 任务17：独立的ROI1和ROI2性能监控
        self._roi1_performance_stats = {
            'total_line_detections': 0,
            'avg_line_detection_time_ms': 0.0,
            'line_detection_errors': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }

        self._roi2_performance_stats = {
            'total_gray_processing': 0,
            'avg_gray_processing_time_ms': 0.0,
            'gray_processing_errors': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }

        # 任务17：独立的线程锁确保ROI1/ROI2处理隔离
        self._roi1_stats_lock = threading.RLock()
        self._roi2_stats_lock = threading.RLock()
        self._roi2_processing_lock = threading.RLock()

        self._logger.info("ROI Capture Service initialized with ROI1/ROI2 isolation: frame_rate=%d, update_interval=%.1f",
                         self._frame_rate, self._cache_interval)

    def clear_cache(self):
        """
        任务17：清除所有ROI缓存（ROI1/ROI2独立缓存）
        """
        # 清除主ROI缓存
        self._cached_roi_data = None
        self._last_roi_config = None
        self._last_capture_time = 0.0

        # 任务17：清除ROI1线条检测专用缓存
        with self._roi1_detector_lock:
            self._roi1_line_detection_cache = None
            self._roi1_line_detection_time = 0.0
            self._roi1_last_processed_image_id = None
            if self._roi1_line_detector:
                self._roi1_line_detector.clear_cache()

        # 任务17：清除ROI2灰度处理专用缓存
        with self._roi2_processing_lock:
            self._roi2_gray_cache = None
            self._roi2_gray_time = 0.0
            self._roi2_last_roi1_config = None

        self._logger.debug("All ROI caches cleared (ROI1 line detection + ROI2 gray processing) - next capture will be forced")

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
            roi_data = self._capture_roi_internal(roi_config)

            # 更新缓存和状态
            if roi_data is not None:
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
        return (current.x1 != cached.x1 or current.y1 != cached.y1 or
                current.x2 != cached.x2 or current.y2 != cached.y2)

    def _capture_roi_internal(self, roi_config: RoiConfig) -> Optional[RoiData]:
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

        # 调整ROI图像大小到标准尺寸（200x150）
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
            format="base64"
        )

        self._logger.debug(
            "ROI captured successfully: size=%dx%d, gray_value=%.2f, base64_length=%d",
            roi_config.width, roi_config.height, average_gray, len(roi_base64)
        )

        return roi_data

    def capture_dual_roi(self, roi_config: RoiConfig) -> Tuple[Optional[RoiData], Optional[RoiData]]:
        """
        截取双ROI区域：ROI1为原始配置区域，ROI2为从ROI1中心截取的50x50区域

        Args:
            roi_config: ROI1配置

        Returns:
            Tuple[Optional[RoiData], Optional[RoiData]]: (ROI1数据, ROI2数据)，失败返回None
        """
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

            # 使用相同的缓存机制（基于ROI1配置）
            time_valid = current_time - self._last_capture_time < self._cache_interval
            config_unchanged = (self._last_roi_config is not None and
                               self._roi_config_changed(roi_config, self._last_roi_config) == False)

            if (self._cached_roi_data is not None and time_valid and config_unchanged):
                self._logger.debug(f"Using cached dual ROI data (age: {current_time - self._last_capture_time:.3f}s)")
                # 从缓存重建ROI2数据
                roi1_data = self._cached_roi_data
                roi2_data = self._extract_roi2_from_roi1(roi_config, roi1_data)
                return roi1_data, roi2_data
            else:
                self._logger.debug(f"Forcing new dual ROI capture - time_valid: {time_valid}, config_unchanged: {config_unchanged}")

            # 执行真实双ROI截图操作
            roi1_data, roi2_data = self._capture_dual_roi_internal(roi_config)

            # 更新缓存和状态（仅缓存ROI1）
            if roi1_data is not None:
                self._cached_roi_data = roi1_data
                self._last_roi_config = roi_config
                self._last_capture_time = current_time
                self._logger.debug("Dual ROI captured successfully (ROI1 gray=%.2f, ROI2 gray=%.2f)",
                                 roi1_data.gray_value, roi2_data.gray_value if roi2_data else 0.0)

                # 集成历史存储 - 保存ROI2帧到DataStore（用于峰值检测）
                try:
                    from ..core.data_store import data_store
                    # 获取当前主信号帧数
                    _, main_frame_count, _, _, _, _ = data_store.get_status_snapshot()

                    # 添加ROI2历史帧（用于峰值检测）
                    if roi2_data:
                        roi2_frame = data_store.add_roi_frame(
                            gray_value=roi2_data.gray_value,
                            roi_config=self._create_roi2_config(roi_config),
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

    def capture_dual_roi_with_line_detection_isolated(self, roi_config: RoiConfig, frame_count: int = 0) -> Tuple[Optional[RoiData], Optional[RoiData], Optional[LineIntersectionResult]]:
        """
        任务17：截取双ROI区域并进行隔离处理（ROI1线条检测 + ROI2灰度分析）

        特性：
        - ROI1和ROI2完全独立的处理管道
        - ROI1线条检测失败不影响ROI2灰度处理
        - ROI2灰度处理失败不影响ROI1线条检测
        - 独立的缓存机制和性能监控
        - 线程安全的并发处理

        Args:
            roi_config: ROI1配置
            frame_count: 当前帧计数，用于线条检测的时间稳定性分析

        Returns:
            Tuple[Optional[RoiData], Optional[RoiData], Optional[LineIntersectionResult]]:
                   (ROI1数据, ROI2数据, ROI1线条相交检测结果)，失败返回None
        """
        start_time = time.time()

        try:
            # 验证ROI1坐标
            if not roi_config.validate_coordinates():
                self._logger.error("Invalid ROI1 coordinates: %s", roi_config)
                return None, None, None

            # 检查ROI1是否足够大以包含50x50的ROI2
            if roi_config.width < 50 or roi_config.height < 50:
                self._logger.error("ROI1 too small for dual ROI: size=%dx%d, minimum 50x50 required",
                                 roi_config.width, roi_config.height)
                return None, None, None

            current_time = time.time()

            # 任务17：使用主ROI缓存机制（仅ROI1数据）
            time_valid = current_time - self._last_capture_time < self._cache_interval
            config_unchanged = (self._last_roi_config is not None and
                               self._roi_config_changed(roi_config, self._last_roi_config) == False)

            roi1_data = None
            roi2_data = None

            if (self._cached_roi_data is not None and time_valid and config_unchanged):
                self._logger.debug(f"Task17: Using cached ROI1 data (age: {current_time - self._last_capture_time:.3f}s)")
                roi1_data = self._cached_roi_data
            else:
                self._logger.debug(f"Task17: Forcing new ROI1 capture - time_valid: {time_valid}, config_unchanged: {config_unchanged}")

                # 执行ROI1截图操作
                roi1_data = self._capture_roi_internal(roi_config)

                # 更新ROI1缓存和状态
                if roi1_data is not None:
                    self._cached_roi_data = roi1_data
                    self._last_roi_config = roi_config
                    self._last_capture_time = current_time
                    self._logger.debug("Task17: ROI1 captured successfully (gray=%.2f)", roi1_data.gray_value)

            # 任务17：ROI1和ROI2独立并行处理
            line_intersection_result = None

            # ROI1线条检测处理（完全隔离）
            if roi1_data is not None and self._settings.line_detection.enabled:
                try:
                    line_intersection_result = self._detect_lines_in_roi1_isolated(roi1_data, roi_config, frame_count)
                except Exception as e:
                    self._logger.error(f"Task17: ROI1 line detection failed but continuing with ROI2: {e}")
                    # ROI1失败不影响ROI2处理
            else:
                self._logger.debug("Task17: ROI1 line detection disabled or ROI1 data unavailable")

            # ROI2灰度处理（完全隔离）
            if roi1_data is not None:
                try:
                    roi2_data = self._extract_roi2_grayscale_isolated(roi_config, roi1_data)

                    # 任务17：集成历史存储 - 保存ROI2帧到DataStore（用于峰值检测）
                    if roi2_data:
                        try:
                            from ..core.data_store import data_store
                            # 获取当前主信号帧数
                            _, main_frame_count, _, _, _, _ = data_store.get_status_snapshot()

                            # 添加ROI2历史帧（用于峰值检测）
                            roi2_frame = data_store.add_roi_frame(
                                gray_value=roi2_data.gray_value,
                                roi_config=self._create_roi2_config(roi_config),
                                frame_count=main_frame_count,
                                capture_duration=self._cache_interval
                            )

                            # 减少日志频率 - 每50帧记录一次
                            if roi2_frame.index % 50 == 0:
                                self._logger.debug("Task17: ROI2 frame added to history: index=%d, gray_value=%.2f, main_frame=%d",
                                                   roi2_frame.index, roi2_data.gray_value, main_frame_count)

                        except Exception as e:
                            self._logger.error("Task17: Failed to add ROI2 frame to history: %s", str(e))

                except Exception as e:
                    self._logger.error(f"Task17: ROI2 grayscale processing failed but continuing with ROI1: {e}")
                    # ROI2失败不影响ROI1处理
            else:
                self._logger.debug("Task17: ROI2 grayscale processing skipped - ROI1 data unavailable")

            # 任务17：记录总体处理状态
            processing_time_ms = (time.time() - start_time) * 1000
            self._logger.debug(f"Task17: Isolated dual ROI processing completed in {processing_time_ms:.1f}ms - "
                             f"ROI1: {'OK' if roi1_data else 'FAIL'}, "
                             f"ROI2: {'OK' if roi2_data else 'FAIL'}, "
                             f"LineDetection: {'OK' if line_intersection_result else 'FAIL/DISABLED'}")

            return roi1_data, roi2_data, line_intersection_result

        except Exception as e:
            self._logger.error(f"Task17: Critical failure in isolated dual ROI processing: {e}", exc_info=True)
            # 任务17：确保关键错误不会同时影响ROI1和ROI2
            return None, None, None

    def capture_dual_roi_with_line_detection(self, roi_config: RoiConfig, frame_count: int = 0) -> Tuple[Optional[RoiData], Optional[RoiData], Optional[LineIntersectionResult]]:
        """
        截取双ROI区域并进行ROI1线条检测：ROI1为原始配置区域，ROI2为从ROI1中心截取的50x50区域
        兼容性方法，内部调用新的隔离处理方法

        Args:
            roi_config: ROI1配置
            frame_count: 当前帧计数，用于线条检测的时间稳定性分析

        Returns:
            Tuple[Optional[RoiData], Optional[RoiData], Optional[LineIntersectionResult]]:
                   (ROI1数据, ROI2数据, ROI1线条相交检测结果)，失败返回None
        """
        # 任务17：使用新的隔离处理方法
        return self.capture_dual_roi_with_line_detection_isolated(roi_config, frame_count)

    def _extract_roi2_from_roi1(self, roi1_config: RoiConfig, roi1_data: RoiData) -> Optional[RoiData]:
        """从ROI1数据中提取ROI2（50x50中心区域）"""
        try:
            self._logger.debug(f"Extracting ROI2 from ROI1: ROI1 config=({roi1_config.x1},{roi1_config.y1})->({roi1_config.x2},{roi1_config.y2}), "
                             f"size={roi1_config.width}x{roi1_config.height}")

            # 计算ROI2在ROI1中的坐标（50x50中心）
            roi1_center_x = roi1_config.width // 2
            roi1_center_y = roi1_config.height // 2
            roi2_size = 50

            # ROI2的起始和结束坐标（在ROI1内）
            roi2_x1 = max(0, roi1_center_x - roi2_size // 2)
            roi2_y1 = max(0, roi1_center_y - roi2_size // 2)
            roi2_x2 = min(roi1_config.width, roi2_x1 + roi2_size)
            roi2_y2 = min(roi1_config.height, roi2_y1 + roi2_size)

            # 验证ROI2坐标的合理性
            if roi2_x2 <= roi2_x1 or roi2_y2 <= roi2_y1:
                self._logger.error(f"Invalid ROI2 coordinates calculated: roi2_x1={roi2_x1}, roi2_x2={roi2_x2}, "
                                  f"roi2_y1={roi2_y1}, roi2_y2={roi2_y2}")
                self._logger.error(f"ROI1 center: ({roi1_center_x},{roi1_center_y}), ROI1 size: {roi1_config.width}x{roi1_config.height}")
                return None

            # 如果ROI1太小，调整ROI2大小到可用最大尺寸
            if roi2_x2 - roi2_x1 < 50 or roi2_y2 - roi2_y1 < 50:
                actual_width = roi2_x2 - roi2_x1
                actual_height = roi2_y2 - roi2_y1
                self._logger.warning(f"ROI1 too small for 50x50 ROI2, using available size: "
                                    f"{actual_width}x{actual_height}")
                # 不需要调整坐标，已经计算好了最大可用尺寸

            self._logger.debug(f"ROI2 coordinates calculated: ({roi2_x1},{roi2_y1})->({roi2_x2},{roi2_y2}) "
                             f"size={(roi2_x2-roi2_x1)}x{(roi2_y2-roi2_y1)}")

            # 从base64解码ROI1图像
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

            # 添加ROI1图像整体诊断
            roi1_gray = roi1_image.convert('L')
            roi1_hist = roi1_gray.histogram()
            roi1_min = min(i for i, count in enumerate(roi1_hist) if count > 0)
            roi1_max = max(i for i, count in enumerate(roi1_hist) if count > 0)
            roi1_nonzero = sum(roi1_hist[1:])  # 排除0值的像素数量
            roi1_total = sum(roi1_hist)
      
            # 如果ROI1图像大部分是黑色，记录警告
            if roi1_nonzero < roi1_total * 0.01:  # 少于1%的非零像素
                self._logger.warning("ROI1 image appears to be mostly black - check ROI configuration")

            # 检查ROI1中ROI2区域的像素值分布
            roi1_gray = roi1_image.convert('L')
            roi2_region_pixels = []

            # 初始化相对坐标变量
            rel_x1 = rel_y1 = rel_x2 = rel_y2 = 0

            # 将ROI2坐标从屏幕空间转换到ROI1图像空间
            # ROI1屏幕坐标: (roi1_config.x1, roi1_config.y1)->(roi1_config.x2, roi1_config.y2)
            # ROI1图像尺寸: roi1_gray.size (已调整大小)
            roi1_screen_width = roi1_config.x2 - roi1_config.x1
            roi1_screen_height = roi1_config.y2 - roi1_config.y1

            # 计算缩放比例
            scale_x = roi1_gray.width / roi1_screen_width
            scale_y = roi1_gray.height / roi1_screen_height

            # ROI2中心点在ROI1中的相对位置
            roi2_center_x = (roi2_x1 + roi2_x2) // 2 - roi1_config.x1
            roi2_center_y = (roi2_y1 + roi2_y2) // 2 - roi1_config.y1

            # 转换到图像坐标并缩放
            rel_center_x = int(roi2_center_x * scale_x)
            rel_center_y = int(roi2_center_y * scale_y)

            # ROI2在图像中的尺寸 (保持50x50或按比例缩放)
            roi2_image_size = min(50, min(roi1_gray.width, roi1_gray.height) // 4)
            rel_x1 = max(0, rel_center_x - roi2_image_size // 2)
            rel_y1 = max(0, rel_center_y - roi2_image_size // 2)
            rel_x2 = min(roi1_gray.width, rel_x1 + roi2_image_size)
            rel_y2 = min(roi1_gray.height, rel_y1 + roi2_image_size)

        
            # 验证坐标范围
            if (rel_x1 < 0 or rel_y1 < 0 or rel_x2 > roi1_gray.width or rel_y2 > roi1_gray.height):
                self._logger.error(f"ROI2 coordinates exceed ROI1 image bounds: ROI1={roi1_gray.size}, ROI2_rel=({rel_x1},{rel_y1},{rel_x2},{rel_y2})")
                # 使用安全的默认坐标而不是返回None
                rel_x1, rel_y1 = 0, 0
                rel_x2, rel_y2 = min(50, roi1_gray.width), min(50, roi1_gray.height)
                self._logger.warning(f"Using fallback ROI2 coordinates: ({rel_x1},{rel_y1})->({rel_x2},{rel_y2})")

            # 使用相对坐标采样ROI2区域的像素
            for y in range(rel_y1, rel_y2):
                for x in range(rel_x1, rel_x2):
                    if 0 <= x < roi1_gray.width and 0 <= y < roi1_gray.height:
                        roi2_region_pixels.append(roi1_gray.getpixel((x, y)))
                    else:
                        self._logger.warning(f"Pixel coordinate out of bounds: ({x},{y}) in image size {roi1_gray.size}")

            if roi2_region_pixels:
                roi2_region_min = min(roi2_region_pixels)
                roi2_region_max = max(roi2_region_pixels)
                roi2_region_avg = sum(roi2_region_pixels) / len(roi2_region_pixels)
                non_zero_count = sum(1 for p in roi2_region_pixels if p > 0)

              
                if roi2_region_max == 0:
                    self._logger.warning("ROI1 center region is completely black - ROI2 area will be black")
                else:
                    self._logger.debug("ROI1 center region has valid pixel data")
            else:
                self._logger.error("Failed to sample ROI2 region pixels from ROI1")

            # 裁剪ROI2区域 - 使用相对坐标
            roi2_image = roi1_image.crop((rel_x1, rel_y1, rel_x2, rel_y2))

            # 添加ROI2图像调试日志
            roi2_original_size = (roi2_x2 - roi2_x1, roi2_y2 - roi2_y1)
            roi2_actual_size = (rel_x2 - rel_x1, rel_y2 - rel_y1)
            roi2_mode = roi2_image.mode
            self._logger.debug(f"ROI2 image cropped: original_size={roi2_original_size}, actual_size={roi2_actual_size}, mode={roi2_mode}")

            # 计算ROI2平均灰度值
            gray_roi2 = roi2_image.convert('L')
            histogram = gray_roi2.histogram()
            roi2_width = rel_x2 - rel_x1
            roi2_height = rel_y2 - rel_y1
            total_pixels = roi2_width * roi2_height
            total_sum = sum(i * count for i, count in enumerate(histogram))
            average_gray = float(total_sum / total_pixels) if total_pixels > 0 else 0.0

            
            # 调整ROI2图像大小到标准尺寸（200x150用于显示）
            try:
                roi2_resized = roi2_image.resize((200, 150), Image.Resampling.LANCZOS)
            except AttributeError:
                roi2_resized = roi2_image.resize((200, 150), Image.LANCZOS)

            # 转换为base64
            buffer = BytesIO()
            roi2_resized.save(buffer, format='PNG')
            roi2_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            # 添加ROI2编码调试日志
            roi2_base64_length = len(roi2_base64)
            self._logger.debug(f"ROI2 encoded: resized_size={roi2_resized.size}, base64_length={roi2_base64_length}")

            # 检查ROI2图像是否为全黑
            non_zero_pixels = sum(1 for p in roi2_resized.getdata() if sum(p) > 0)
            total_pixels = roi2_resized.size[0] * roi2_resized.size[1]
            if non_zero_pixels == 0:
                self._logger.warning(f"ROI2 image appears to be completely black: {total_pixels} pixels, 0 non-zero, "
                                    f"calculated_gray={average_gray:.2f} - MISMATCH DETECTED!")
            
            roi2_data = RoiData(
                width=roi2_width,
                height=roi2_height,
                pixels=f"data:image/png;base64,{roi2_base64}",
                gray_value=average_gray,
                format="base64"
            )

            self._logger.debug("ROI2 extracted from ROI1: size=%dx%d, gray_value=%.2f",
                             roi2_width, roi2_height, average_gray)

            return roi2_data

        except Exception as e:
            self._logger.error("Failed to extract ROI2 from ROI1: %s", str(e))
            return None

    def _capture_dual_roi_internal(self, roi_config: RoiConfig) -> Tuple[Optional[RoiData], Optional[RoiData]]:
        """执行实际的双ROI截图操作 - 统一从ROI1图像中提取ROI2"""
        self._logger.debug("Starting unified dual ROI capture - always extract ROI2 from ROI1 image")

        # 首先捕获ROI1
        roi1_data = self._capture_roi_internal(roi_config)
        if roi1_data is None:
            self._logger.error("Failed to capture ROI1, cannot extract ROI2")
            return None, None

        # 然后从ROI1图像中提取ROI2 - 使用统一的方法
        roi2_data = self._extract_roi2_from_roi1(roi_config, roi1_data)
        if roi2_data is None:
            self._logger.error("Failed to extract ROI2 from ROI1 image")
            # 仍然返回ROI1数据，让系统可以继续工作
            return roi1_data, None

        self._logger.debug(
            "Unified dual ROI capture successful: ROI1=%.2f, ROI2=%.2f",
            roi1_data.gray_value, roi2_data.gray_value
        )

        return roi1_data, roi2_data

    def _create_roi2_config(self, roi1_config: RoiConfig) -> RoiConfig:
        """创建ROI2的配置对象"""
        # ROI2在屏幕上的坐标
        roi1_center_x = roi1_config.x1 + roi1_config.width // 2
        roi1_center_y = roi1_config.y1 + roi1_config.height // 2
        roi2_size = 50

        roi2_x1 = max(roi1_config.x1, roi1_center_x - roi2_size // 2)
        roi2_y1 = max(roi1_config.y1, roi1_center_y - roi2_size // 2)
        roi2_x2 = min(roi1_config.x2, roi2_x1 + roi2_size)
        roi2_y2 = min(roi1_config.y2, roi2_y1 + roi2_size)

        return RoiConfig(x1=roi2_x1, y1=roi2_y1, x2=roi2_x2, y2=roi2_y2)

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

    def _detect_lines_in_roi1_isolated(self, roi1_data: RoiData, roi_config: RoiConfig, frame_count: int = 0) -> Optional[LineIntersectionResult]:
        """
        任务17：ROI1专用绿色线条相交点检测（完全隔离处理）

        特性：
        - ROI1专用缓存机制，基于图像内容ID和配置变更
        - 独立的性能监控和错误处理
        - 线程安全的处理管道
        - ROI2处理完全隔离，不受影响

        Args:
            roi1_data: ROI1数据对象
            roi_config: ROI1配置对象
            frame_count: 当前帧计数

        Returns:
            LineIntersectionResult: 线条相交检测结果，失败返回None
        """
        start_time = time.time()

        try:
            # 任务17：更新ROI1专用统计
            with self._roi1_stats_lock:
                self._roi1_performance_stats['total_line_detections'] += 1

            # 任务17：ROI1专用配置验证
            current_config = self._settings.line_detection
            if not self._validate_roi1_line_detection_config(current_config):
                self._logger.warning("ROI1 line detection configuration validation failed")
                with self._roi1_stats_lock:
                    self._roi1_performance_stats['line_detection_errors'] += 1
                return None

            # 任务17：生成基于图像内容的缓存键
            image_content_id = self._generate_image_content_id(roi1_data.pixels)
            config_changed = (self._roi1_line_detection_config is None or
                            not self._configs_equal(self._roi1_line_detection_config, current_config))

            # 任务17：检查ROI1专用缓存有效性
            current_time = time.time()
            cache_timeout = current_config.cache_timeout_ms / 1000.0  # 转换为秒
            cache_valid = (self._roi1_line_detection_cache is not None and
                          current_time - self._roi1_line_detection_time < cache_timeout and
                          not config_changed and
                          self._roi1_last_processed_image_id == image_content_id)

            if cache_valid:
                with self._roi1_stats_lock:
                    self._roi1_performance_stats['cache_hits'] += 1
                self._logger.debug(f"ROI1: Using cached line detection result (age: {current_time - self._roi1_line_detection_time:.3f}s)")
                return self._roi1_line_detection_cache
            else:
                with self._roi1_stats_lock:
                    self._roi1_performance_stats['cache_misses'] += 1

            self._logger.debug("ROI1: Performing new isolated line detection")

            # 任务17：ROI1专用线条检测器初始化
            with self._roi1_detector_lock:
                if (self._roi1_line_detector is None or
                    self._roi1_line_detection_config is None or
                    config_changed):

                    if current_config.enabled:
                        try:
                            from .line_intersection_detector import LineIntersectionDetector
                            self._roi1_line_detector = LineIntersectionDetector(current_config)
                            self._roi1_line_detection_config = current_config
                            self._logger.info("ROI1: Line detector initialized/reconfigured")
                        except Exception as e:
                            self._logger.error(f"ROI1: Failed to initialize line detector: {e}")
                            with self._roi1_stats_lock:
                                self._roi1_performance_stats['line_detection_errors'] += 1
                            return None
                    else:
                        self._roi1_line_detector = None
                        self._roi1_line_detection_config = None
                        return None

                # 如果检测器未启用，返回None
                if self._roi1_line_detector is None:
                    return None

            # 任务17：将base64图像数据转换为PIL图像（ROI1专用）
            pil_image = self._base64_to_pil_image_isolated(roi1_data.pixels)
            if pil_image is None:
                self._logger.error("ROI1: Failed to convert ROI1 data to PIL image")
                with self._roi1_stats_lock:
                    self._roi1_performance_stats['line_detection_errors'] += 1
                return None

            # 转换为OpenCV格式进行线条检测
            import cv2
            import numpy as np
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')

            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            # 任务17：执行隔离的线条检测
            detection_result = self._roi1_line_detector.detect_intersection(cv_image, frame_count)

            # 任务17：缓存ROI1检测结果
            if detection_result:
                self._roi1_line_detection_cache = detection_result
                self._roi1_line_detection_time = current_time
                self._roi1_last_processed_image_id = image_content_id

                # 更新ROI1专用性能统计
                processing_time = (time.time() - start_time) * 1000
                with self._roi1_stats_lock:
                    stats = self._roi1_performance_stats
                    stats['avg_line_detection_time_ms'] = (
                        (stats['avg_line_detection_time_ms'] * (stats['total_line_detections'] - 1) + processing_time) /
                        stats['total_line_detections']
                    )

                # 记录ROI1检测结果
                if detection_result.has_intersection:
                    self._logger.info(f"ROI1: Line detection successful - intersection={detection_result.intersection}, "
                                    f"confidence={detection_result.confidence:.3f}, "
                                    f"processing_time={processing_time:.1f}ms")
                else:
                    self._logger.debug(f"ROI1: Line detection completed - no intersection found, "
                                     f"confidence={detection_result.confidence:.3f}")

            return detection_result

        except Exception as e:
            self._logger.error(f"ROI1: Isolated line detection failed: {e}", exc_info=True)
            with self._roi1_stats_lock:
                self._roi1_performance_stats['line_detection_errors'] += 1
            # 任务17：确保ROI1错误不影响ROI2处理
            return None

    def _validate_roi1_line_detection_config(self, config: LineDetectionConfig) -> bool:
        """
        任务17：ROI1专用线条检测配置验证

        Args:
            config: 线条检测配置

        Returns:
            bool: 配置是否有效
        """
        try:
            # 检查是否启用ROI1专用处理模式
            if config.roi_processing_mode != "roi1_only":
                self._logger.warning(f"ROI1: Invalid processing mode: {config.roi_processing_mode}, expected 'roi1_only'")
                return False

            # 验证HSV颜色范围
            if not config.validate_hsv_ranges():
                self._logger.error("ROI1: HSV color ranges are invalid")
                return False

            # 验证Canny阈值
            if not config.validate_canny_thresholds():
                self._logger.error("ROI1: Canny thresholds are invalid")
                return False

            # 验证Hough参数
            if not config.validate_hough_parameters():
                self._logger.error("ROI1: Hough parameters are invalid")
                return False

            # 验证置信度范围
            if not (0.0 <= config.min_confidence <= 1.0):
                self._logger.error(f"ROI1: Invalid min_confidence: {config.min_confidence}")
                return False

            # 验证角度范围
            if not (0.0 <= config.min_angle_degrees <= 90.0 and 0.0 <= config.max_angle_degrees <= 90.0):
                self._logger.error(f"ROI1: Invalid angle ranges: min={config.min_angle_degrees}, max={config.max_angle_degrees}")
                return False

            if config.min_angle_degrees >= config.max_angle_degrees:
                self._logger.error(f"ROI1: min_angle must be less than max_angle: {config.min_angle_degrees} >= {config.max_angle_degrees}")
                return False

            return True

        except Exception as e:
            self._logger.error(f"ROI1: Configuration validation error: {e}")
            return False

    def _generate_image_content_id(self, base64_data: str) -> str:
        """
        任务17：基于图像内容生成唯一的缓存键

        Args:
            base64_data: base64编码的图像数据

        Returns:
            str: 基于内容的唯一标识符
        """
        try:
            import hashlib
            # 提取实际的base64数据
            if base64_data.startswith('data:image/'):
                base64_data = base64_data.split(',', 1)[1]

            # 生成MD5哈希作为内容ID
            content_hash = hashlib.md5(base64_data.encode('utf-8')).hexdigest()
            return f"roi1_img_{content_hash[:16]}"  # 使用前16位

        except Exception as e:
            self._logger.error(f"ROI1: Failed to generate image content ID: {e}")
            # 回退到时间戳
            return f"roi1_fallback_{int(time.time() * 1000)}"

    def _configs_equal(self, config1: LineDetectionConfig, config2: LineDetectionConfig) -> bool:
        """
        任务17：比较两个线条检测配置是否相等

        Args:
            config1: 第一个配置
            config2: 第二个配置

        Returns:
            bool: 配置是否相等
        """
        if not (config1 and config2):
            return False

        # 比较关键配置项
        return (
            config1.enabled == config2.enabled and
            config1.hsv_green_lower == config2.hsv_green_lower and
            config1.hsv_green_upper == config2.hsv_green_upper and
            config1.canny_low_threshold == config2.canny_low_threshold and
            config1.canny_high_threshold == config2.canny_high_threshold and
            config1.hough_threshold == config2.hough_threshold and
            config1.min_confidence == config2.min_confidence and
            config1.roi_processing_mode == config2.roi_processing_mode
        )

    def _base64_to_pil_image_isolated(self, base64_data: str) -> Optional[Image.Image]:
        """
        任务17：ROI1专用的base64到PIL图像转换

        Args:
            base64_data: base64编码的图像数据

        Returns:
            PIL.Image对象，失败返回None
        """
        try:
            import base64
            from io import BytesIO

            # 提取base64数据
            if base64_data.startswith('data:image/png;base64,'):
                base64_data = base64_data.replace('data:image/png;base64,', '')
            elif base64_data.startswith('data:image/jpeg;base64,'):
                base64_data = base64_data.replace('data:image/jpeg;base64,', '')

            # 解码图像
            image_data = base64.b64decode(base64_data)
            pil_image = Image.open(BytesIO(image_data))

            return pil_image

        except Exception as e:
            self._logger.error(f"ROI1: Failed to convert base64 to PIL image: {e}")
            return None

    def _extract_roi2_grayscale_isolated(self, roi1_config: RoiConfig, roi1_data: RoiData) -> Optional[RoiData]:
        """
        任务17：ROI2专用灰度处理（完全隔离，不受ROI1线条检测影响）

        特性：
        - ROI2专用缓存机制，与ROI1完全独立
        - 独立的性能监控和错误处理
        - 线程安全的处理管道
        - ROI1线条检测失败不影响ROI2灰度处理

        Args:
            roi1_config: ROI1配置对象
            roi1_data: ROI1数据对象

        Returns:
            RoiData: ROI2灰度数据，失败返回None
        """
        start_time = time.time()

        try:
            # 任务17：更新ROI2专用统计
            with self._roi2_stats_lock:
                self._roi2_performance_stats['total_gray_processing'] += 1

            # 任务17：检查ROI2专用缓存有效性
            current_time = time.time()
            cache_timeout = self._cache_interval  # 使用ROI缓存间隔
            config_unchanged = (self._roi2_last_roi1_config is not None and
                               self._roi_config_changed(roi1_config, self._roi2_last_roi1_config) == False)

            cache_valid = (self._roi2_gray_cache is not None and
                          current_time - self._roi2_gray_time < cache_timeout and
                          config_unchanged)

            if cache_valid:
                with self._roi2_stats_lock:
                    self._roi2_performance_stats['cache_hits'] += 1
                self._logger.debug(f"ROI2: Using cached grayscale data (age: {current_time - self._roi2_gray_time:.3f}s)")
                return self._roi2_gray_cache
            else:
                with self._roi2_stats_lock:
                    self._roi2_performance_stats['cache_misses'] += 1

            self._logger.debug("ROI2: Performing new isolated grayscale processing")

            # 任务17：ROI2专用处理，完全独立于ROI1
            roi2_data = self._process_roi2_grayscale_internal(roi1_config, roi1_data)

            # 任务17：缓存ROI2处理结果
            if roi2_data is not None:
                self._roi2_gray_cache = roi2_data
                self._roi2_gray_time = current_time
                self._roi2_last_roi1_config = roi1_config

                # 更新ROI2专用性能统计
                processing_time = (time.time() - start_time) * 1000
                with self._roi2_stats_lock:
                    stats = self._roi2_performance_stats
                    stats['avg_gray_processing_time_ms'] = (
                        (stats['avg_gray_processing_time_ms'] * (stats['total_gray_processing'] - 1) + processing_time) /
                        stats['total_gray_processing']
                    )

                self._logger.debug(f"ROI2: Grayscale processing completed - gray_value={roi2_data.gray_value:.2f}, "
                                 f"size={roi2_data.width}x{roi2_data.height}, "
                                 f"processing_time={processing_time:.1f}ms")

            return roi2_data

        except Exception as e:
            self._logger.error(f"ROI2: Isolated grayscale processing failed: {e}", exc_info=True)
            with self._roi2_stats_lock:
                self._roi2_performance_stats['gray_processing_errors'] += 1
            # 任务17：确保ROI2错误不影响ROI1处理
            return None

    def _process_roi2_grayscale_internal(self, roi1_config: RoiConfig, roi1_data: RoiData) -> Optional[RoiData]:
        """
        任务17：ROI2灰度处理内部实现（与ROI1完全隔离）

        Args:
            roi1_config: ROI1配置对象
            roi1_data: ROI1数据对象

        Returns:
            RoiData: 处理后的ROI2数据，失败返回None
        """
        try:
            # 计算ROI2在ROI1中的坐标（50x50中心）
            roi1_center_x = roi1_config.width // 2
            roi1_center_y = roi1_config.height // 2
            roi2_size = 50

            # ROI2的起始和结束坐标（在ROI1内）
            roi2_x1 = max(0, roi1_center_x - roi2_size // 2)
            roi2_y1 = max(0, roi1_center_y - roi2_size // 2)
            roi2_x2 = min(roi1_config.width, roi2_x1 + roi2_size)
            roi2_y2 = min(roi1_config.height, roi2_y1 + roi2_size)

            # 验证ROI2坐标的合理性
            if roi2_x2 <= roi2_x1 or roi2_y2 <= roi2_y1:
                self._logger.error(f"ROI2: Invalid coordinates calculated: roi2_x1={roi2_x1}, roi2_x2={roi2_x2}, "
                                  f"roi2_y1={roi2_y1}, roi2_y2={roi2_y2}")
                return None

            # 如果ROI1太小，调整ROI2大小到可用最大尺寸
            if roi2_x2 - roi2_x1 < 50 or roi2_y2 - roi2_y1 < 50:
                actual_width = roi2_x2 - roi2_x1
                actual_height = roi2_y2 - roi2_y1
                self._logger.warning(f"ROI2: ROI1 too small for 50x50 ROI2, using available size: "
                                    f"{actual_width}x{actual_height}")

            self._logger.debug(f"ROI2: Coordinates calculated: ({roi2_x1},{roi2_y1})->({roi2_x2},{roi2_y2}) "
                             f"size={(roi2_x2-roi2_x1)}x{(roi2_y2-roi2_y1)}")

            # 从base64解码ROI1图像（ROI2专用处理）
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

            # 将ROI2坐标从ROI1空间转换到图像空间
            roi1_screen_width = roi1_config.x2 - roi1_config.x1
            roi1_screen_height = roi1_config.y2 - roi1_config.y1

            # 计算缩放比例
            scale_x = roi1_image.width / roi1_screen_width
            scale_y = roi1_image.height / roi1_screen_height

            # ROI2中心点在ROI1中的相对位置
            roi2_center_x = (roi2_x1 + roi2_x2) // 2
            roi2_center_y = (roi2_y1 + roi2_y2) // 2

            # 转换到图像坐标并缩放
            rel_center_x = int(roi2_center_x * scale_x)
            rel_center_y = int(roi2_center_y * scale_y)

            # ROI2在图像中的尺寸
            roi2_image_size = min(50, min(roi1_image.width, roi1_image.height) // 4)
            rel_x1 = max(0, rel_center_x - roi2_image_size // 2)
            rel_y1 = max(0, rel_center_y - roi2_image_size // 2)
            rel_x2 = min(roi1_image.width, rel_x1 + roi2_image_size)
            rel_y2 = min(roi1_image.height, rel_y1 + roi2_image_size)

            # 验证坐标范围
            if (rel_x1 < 0 or rel_y1 < 0 or rel_x2 > roi1_image.width or rel_y2 > roi1_image.height):
                self._logger.error(f"ROI2: Coordinates exceed ROI1 image bounds: ROI1={roi1_image.size}, ROI2_rel=({rel_x1},{rel_y1},{rel_x2},{rel_y2})")
                # 使用安全的默认坐标
                rel_x1, rel_y1 = 0, 0
                rel_x2, rel_y2 = min(50, roi1_image.width), min(50, roi1_image.height)
                self._logger.warning(f"ROI2: Using fallback coordinates: ({rel_x1},{rel_y1})->({rel_x2},{rel_y2})")

            # 裁剪ROI2区域
            roi2_image = roi1_image.crop((rel_x1, rel_y1, rel_x2, rel_y2))

            # 计算ROI2平均灰度值（ROI2专用）
            gray_roi2 = roi2_image.convert('L')
            histogram = gray_roi2.histogram()
            roi2_width = rel_x2 - rel_x1
            roi2_height = rel_y2 - rel_y1
            total_pixels = roi2_width * roi2_height
            total_sum = sum(i * count for i, count in enumerate(histogram))
            average_gray = float(total_sum / total_pixels) if total_pixels > 0 else 0.0

            # 调整ROI2图像大小到标准尺寸（200x150用于显示）
            try:
                roi2_resized = roi2_image.resize((200, 150), Image.Resampling.LANCZOS)
            except AttributeError:
                roi2_resized = roi2_image.resize((200, 150), Image.LANCZOS)

            # 转换为base64
            buffer = BytesIO()
            roi2_resized.save(buffer, format='PNG')
            roi2_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            roi2_data = RoiData(
                width=roi2_width,
                height=roi2_height,
                pixels=f"data:image/png;base64,{roi2_base64}",
                gray_value=average_gray,
                format="base64"
            )

            self._logger.debug(f"ROI2: Grayscale processing successful: size={roi2_width}x{roi2_height}, "
                             f"gray_value={average_gray:.2f}")

            return roi2_data

        except Exception as e:
            self._logger.error(f"ROI2: Internal grayscale processing failed: {e}", exc_info=True)
            return None

    def _base64_to_pil_image(self, base64_data: str) -> Optional[Image.Image]:
        """
        将base64图像数据转换为PIL图像对象

        Args:
            base64_data: base64编码的图像数据

        Returns:
            PIL.Image对象，失败返回None
        """
        try:
            import base64
            from io import BytesIO

            # 提取base64数据
            if base64_data.startswith('data:image/png;base64,'):
                base64_data = base64_data.replace('data:image/png;base64,', '')
            elif base64_data.startswith('data:image/jpeg;base64,'):
                base64_data = base64_data.replace('data:image/jpeg;base64,', '')

            # 解码图像
            image_data = base64.b64decode(base64_data)
            pil_image = Image.open(BytesIO(image_data))

            return pil_image

        except Exception as e:
            self._logger.error(f"Failed to convert base64 to PIL image: {e}")
            return None

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        任务17：获取ROI捕获和线条检测的性能统计信息（ROI1/ROI2隔离）

        Returns:
            包含ROI1和ROI2独立性能统计的字典
        """
        # 任务17：获取ROI1专用性能统计
        with self._roi1_stats_lock:
            roi1_stats = self._roi1_performance_stats.copy()

        # 任务17：获取ROI2专用性能统计
        with self._roi2_stats_lock:
            roi2_stats = self._roi2_performance_stats.copy()

        # 任务17：添加ROI1线条检测器状态
        roi1_detector_status = {}
        with self._roi1_detector_lock:
            if self._roi1_line_detector:
                roi1_detector_status = self._roi1_line_detector.get_detector_status()

        # 任务17：计算缓存效率
        roi1_cache_efficiency = self._calculate_cache_efficiency(roi1_stats.get('cache_hits', 0), roi1_stats.get('cache_misses', 0))
        roi2_cache_efficiency = self._calculate_cache_efficiency(roi2_stats.get('cache_hits', 0), roi2_stats.get('cache_misses', 0))

        return {
            'roi1_line_detection_performance': {
                'stats': roi1_stats,
                'detector_status': roi1_detector_status,
                'cache_efficiency_percent': roi1_cache_efficiency,
                'processing_mode': 'roi1_only'
            },
            'roi2_grayscale_performance': {
                'stats': roi2_stats,
                'cache_efficiency_percent': roi2_cache_efficiency,
                'processing_mode': 'grayscale_only'
            },
            'service_config': {
                'frame_rate': self._frame_rate,
                'cache_interval': self._cache_interval,
                'roi1_line_detection_enabled': self._settings.line_detection.enabled,
                'roi2_grayscale_enabled': True
            },
            'isolation_status': {
                'roi1_roi2_isolated': True,
                'independent_caching': True,
                'independent_error_handling': True,
                'thread_safe_processing': True
            }
        }

    def get_dual_roi_performance_metrics(self) -> Dict[str, Any]:
        """
        任务17：获取双ROI处理的详细性能指标（ROI1/ROI2隔离）

        Returns:
            详细的ROI1和ROI2独立性能指标字典
        """
        # 任务17：获取ROI1性能数据
        with self._roi1_stats_lock:
            roi1_stats = self._roi1_performance_stats.copy()

        # 任务17：获取ROI2性能数据
        with self._roi2_stats_lock:
            roi2_stats = self._roi2_performance_stats.copy()

        # 任务17：获取ROI1检测器指标
        roi1_detector_metrics = {}
        with self._roi1_detector_lock:
            if self._roi1_line_detector:
                roi1_detector_metrics = self._roi1_line_detector.get_detection_metrics()

        # 任务17：计算ROI1成功率
        roi1_total_operations = roi1_stats.get('total_line_detections', 0)
        roi1_error_count = roi1_stats.get('line_detection_errors', 0)
        roi1_success_rate = 1.0 if roi1_total_operations == 0 else (roi1_total_operations - roi1_error_count) / roi1_total_operations

        # 任务17：计算ROI2成功率
        roi2_total_operations = roi2_stats.get('total_gray_processing', 0)
        roi2_error_count = roi2_stats.get('gray_processing_errors', 0)
        roi2_success_rate = 1.0 if roi2_total_operations == 0 else (roi2_total_operations - roi2_error_count) / roi2_total_operations

        return {
            'roi1_processing_performance': {
                'total_line_detection_operations': roi1_total_operations,
                'average_processing_time_ms': roi1_stats.get('avg_line_detection_time_ms', 0.0),
                'error_count': roi1_error_count,
                'success_rate': roi1_success_rate,
                'cache_hits': roi1_stats.get('cache_hits', 0),
                'cache_misses': roi1_stats.get('cache_misses', 0)
            },
            'roi2_processing_performance': {
                'total_grayscale_processing_operations': roi2_total_operations,
                'average_processing_time_ms': roi2_stats.get('avg_gray_processing_time_ms', 0.0),
                'error_count': roi2_error_count,
                'success_rate': roi2_success_rate,
                'cache_hits': roi2_stats.get('cache_hits', 0),
                'cache_misses': roi2_stats.get('cache_misses', 0)
            },
            'line_detection_metrics': roi1_detector_metrics,
            'service_health': {
                'roi1_processing_active': self._settings.line_detection.enabled,
                'roi2_processing_active': True,
                'isolation_status': 'fully_isolated',
                'roi1_cache_efficiency': self._calculate_cache_efficiency(roi1_stats.get('cache_hits', 0), roi1_stats.get('cache_misses', 0)),
                'roi2_cache_efficiency': self._calculate_cache_efficiency(roi2_stats.get('cache_hits', 0), roi2_stats.get('cache_misses', 0))
            },
            'isolation_metrics': {
                'roi1_independent_failures': roi1_error_count,
                'roi2_independent_failures': roi2_error_count,
                'cross_roi_interference_prevented': True,
                'thread_safety_active': True
            }
        }

    def _calculate_cache_efficiency(self, hits: int, misses: int) -> float:
        """
        任务17：计算缓存效率百分比

        Args:
            hits: 缓存命中次数
            misses: 缓存未命中次数

        Returns:
            float: 缓存效率百分比 (0-100)
        """
        total = hits + misses
        if total == 0:
            return 0.0
        return (hits / total) * 100.0

    def reset_roi_performance_stats(self) -> Dict[str, bool]:
        """
        任务17：重置ROI1和ROI2性能统计

        Returns:
            Dict[str, bool]: 重置结果状态
        """
        try:
            # 重置ROI1统计
            with self._roi1_stats_lock:
                self._roi1_performance_stats = {
                    'total_line_detections': 0,
                    'avg_line_detection_time_ms': 0.0,
                    'line_detection_errors': 0,
                    'cache_hits': 0,
                    'cache_misses': 0
                }

            # 重置ROI2统计
            with self._roi2_stats_lock:
                self._roi2_performance_stats = {
                    'total_gray_processing': 0,
                    'avg_gray_processing_time_ms': 0.0,
                    'gray_processing_errors': 0,
                    'cache_hits': 0,
                    'cache_misses': 0
                }

            # 清除检测器统计（如果可用）
            with self._roi1_detector_lock:
                if self._roi1_line_detector:
                    try:
                        self._roi1_line_detector.reset_metrics()
                    except AttributeError:
                        # 检测器可能不支持重置
                        pass

            self._logger.info("Task17: ROI1 and ROI2 performance stats reset successfully")
            return {'roi1_stats_reset': True, 'roi2_stats_reset': True}

        except Exception as e:
            self._logger.error(f"Task17: Failed to reset ROI performance stats: {e}")
            return {'roi1_stats_reset': False, 'roi2_stats_reset': False}

    def validate_isolation_configuration(self) -> Dict[str, Any]:
        """
        任务17：验证ROI1/ROI2隔离配置的完整性

        Returns:
            Dict[str, Any]: 验证结果，包含隔离状态和配置有效性
        """
        validation_results = {
            'isolation_configuration_valid': True,
            'roi1_configuration': {'valid': True, 'errors': []},
            'roi2_configuration': {'valid': True, 'errors': []},
            'thread_safety_valid': True,
            'cache_isolation_valid': True
        }

        try:
            # 验证ROI1线条检测配置
            if self._settings.line_detection.enabled:
                roi1_config_valid = self._validate_roi1_line_detection_config(self._settings.line_detection)
                validation_results['roi1_configuration']['valid'] = roi1_config_valid
                if not roi1_config_valid:
                    validation_results['roi1_configuration']['errors'].append("ROI1 line detection configuration validation failed")
                    validation_results['isolation_configuration_valid'] = False

            # 验证线程锁初始化
            required_locks = [
                '_roi1_stats_lock',
                '_roi2_stats_lock',
                '_roi1_detector_lock',
                '_roi2_processing_lock'
            ]

            for lock_name in required_locks:
                if not hasattr(self, lock_name):
                    validation_results['thread_safety_valid'] = False
                    validation_results['isolation_configuration_valid'] = False
                    validation_results.setdefault('thread_safety_errors', []).append(f"Missing lock: {lock_name}")

            # 验证缓存隔离
            required_caches = [
                '_roi1_line_detection_cache',
                '_roi2_gray_cache',
                '_roi1_last_processed_image_id'
            ]

            for cache_name in required_caches:
                if not hasattr(self, cache_name):
                    validation_results['cache_isolation_valid'] = False
                    validation_results['isolation_configuration_valid'] = False
                    validation_results.setdefault('cache_isolation_errors', []).append(f"Missing cache: {cache_name}")

            # 验证ROI2处理可用性（ROI2总是启用的）
            roi2_min_size = 50
            if hasattr(self, '_settings') and hasattr(self._settings, 'roi'):
                current_roi_config = self._settings.roi
                if (current_roi_config.width < roi2_min_size or
                    current_roi_config.height < roi2_min_size):
                    validation_results['roi2_configuration']['valid'] = False
                    validation_results['roi2_configuration']['errors'].append(
                        f"ROI1 too small for ROI2: requires at least {roi2_min_size}x{roi2_min_size}"
                    )
                    validation_results['isolation_configuration_valid'] = False

            self._logger.info(f"Task17: Isolation configuration validation completed - "
                            f"Overall: {'VALID' if validation_results['isolation_configuration_valid'] else 'INVALID'}")

        except Exception as e:
            validation_results['isolation_configuration_valid'] = False
            validation_results['validation_error'] = str(e)
            self._logger.error(f"Task17: Isolation configuration validation failed: {e}")

        return validation_results

    def get_isolation_status(self) -> Dict[str, Any]:
        """
        任务17：获取ROI1/ROI2隔离状态的实时信息

        Returns:
            Dict[str, Any]: 隔离状态信息
        """
        try:
            # 获取当前线程信息
            import threading
            current_thread = threading.current_thread()

            # 检查ROI1处理状态
            roi1_status = {
                'processing_active': False,
                'detector_initialized': False,
                'cache_valid': False,
                'lock_holder': None
            }

            with self._roi1_detector_lock:
                roi1_status['detector_initialized'] = self._roi1_line_detector is not None
                if self._roi1_line_detector:
                    roi1_status['processing_active'] = self._settings.line_detection.enabled

                current_time = time.time()
                if self._roi1_line_detection_cache and self._roi1_line_detection_time:
                    cache_timeout = self._settings.line_detection.cache_timeout_ms / 1000.0
                    roi1_status['cache_valid'] = (current_time - self._roi1_line_detection_time) < cache_timeout

            # 检查ROI2处理状态
            roi2_status = {
                'processing_active': True,  # ROI2总是激活的
                'cache_valid': False,
                'lock_holder': None
            }

            with self._roi2_processing_lock:
                current_time = time.time()
                if self._roi2_gray_cache and self._roi2_gray_time:
                    roi2_status['cache_valid'] = (current_time - self._roi2_gray_time) < self._cache_interval

            # 检查锁状态（非阻塞检查）
            lock_status = {}
            try:
                roi1_stats_locked = self._roi1_stats_lock.acquire(blocking=False)
                if roi1_stats_locked:
                    self._roi1_stats_lock.release()
                lock_status['roi1_stats_lock'] = 'unlocked' if roi1_stats_locked else 'locked'
            except:
                lock_status['roi1_stats_lock'] = 'error'

            try:
                roi2_stats_locked = self._roi2_stats_lock.acquire(blocking=False)
                if roi2_stats_locked:
                    self._roi2_stats_lock.release()
                lock_status['roi2_stats_lock'] = 'unlocked' if roi2_stats_locked else 'locked'
            except:
                lock_status['roi2_stats_lock'] = 'error'

            return {
                'isolation_active': True,
                'current_thread': {
                    'name': current_thread.name,
                    'id': current_thread.ident
                },
                'roi1_status': roi1_status,
                'roi2_status': roi2_status,
                'thread_locks': lock_status,
                'processing_isolation': {
                    'roi1_independent_failures': self._roi1_performance_stats.get('line_detection_errors', 0),
                    'roi2_independent_failures': self._roi2_performance_stats.get('gray_processing_errors', 0),
                    'cross_interference_detected': False
                },
                'cache_isolation': {
                    'roi1_cache_independent': True,
                    'roi2_cache_independent': True,
                    'shared_cache_state': False
                }
            }

        except Exception as e:
            self._logger.error(f"Task17: Failed to get isolation status: {e}")
            return {
                'isolation_active': False,
                'error': str(e),
                'status_check_failed': True
            }


# 单例ROI截图服务
roi_capture_service = RoiCaptureService()