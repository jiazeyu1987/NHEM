"""
ROI截图服务模块
提供屏幕截图和ROI区域截取功能
"""

import base64
import io
import logging
import time
from datetime import datetime
from typing import Optional, Tuple

# 启用PIL导入
from PIL import Image, ImageGrab

from ..models import RoiConfig, RoiData, DualRoiFrame, DualRoiConfig, DualRoiMode


class RoiCaptureService:
    """ROI截图服务类"""

    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)
        # 从配置获取ROI帧率（配置现在会从JSON文件加载）
        from ..config import settings
        self._settings = settings
        self._frame_rate = settings.roi_frame_rate
        self._cache_interval = settings.roi_update_interval  # 缓存间隔

        # 大ROI帧率控制（独立于主ROI帧率）
        self._large_roi_frame_rate = 5  # 默认5 FPS
        self._large_roi_cache_interval = 1.0 / self._large_roi_frame_rate  # 大ROI缓存间隔
        self._last_large_roi_capture_time = 0.0
        self._cached_large_roi_data: Optional[RoiData] = None
        self._last_large_roi_config: Optional[RoiConfig] = None

        # ROI截图缓存机制
        self._last_capture_time = 0.0
        self._cached_roi_data: Optional[RoiData] = None
        self._last_roi_config: Optional[RoiConfig] = None

        self._logger.info("ROI Capture Service initialized with JSON config: frame_rate=%d, update_interval=%.1f",
                         self._frame_rate, self._cache_interval)
        self._logger.info("Large ROI frame rate initialized: frame_rate=%d, update_interval=%.1f",
                         self._large_roi_frame_rate, self._large_roi_cache_interval)

    def clear_cache(self):
        """
        清除ROI截图缓存，强制下次截图时重新捕获
        """
        self._cached_roi_data = None
        self._last_roi_config = None
        self._last_capture_time = 0.0

        # 清除大ROI缓存
        self._cached_large_roi_data = None
        self._last_large_roi_config = None
        self._last_large_roi_capture_time = 0.0

        self._logger.debug("ROI cache cleared - next capture will be forced")

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

    def capture_roi(self, roi_config: RoiConfig, failure_count: int = 0) -> Optional[RoiData]:
        """
        截取指定ROI区域（带缓存机制和增强的错误处理）

        Args:
            roi_config: ROI配置
            failure_count: 当前连续失败次数

        Returns:
            RoiData: ROI数据，失败返回None
        """
        capture_start_time = time.time()

        try:
            # 使用新的ROI边界验证方法
            is_valid, adjusted_config, validation_msg = self.validate_roi_bounds(roi_config, auto_adjust=True)

            if not is_valid:
                # 验证失败，尝试处理失败情况
                self._logger.error("ROI validation failed: %s", validation_msg)
                return self.handle_capture_failure(roi_config, failure_count)

            # 如果配置被调整了，使用调整后的配置
            effective_config = adjusted_config if adjusted_config else roi_config

            current_time = time.time()

            # 简化的缓存机制：只基于时间间隔和配置变化
            time_valid = current_time - self._last_capture_time < self._cache_interval
            config_unchanged = (self._last_roi_config is not None and
                               self._roi_config_changed(effective_config, self._last_roi_config) == False)

            # 只有在缓存有效且配置未变化时才使用缓存
            if (self._cached_roi_data is not None and time_valid and config_unchanged):
                capture_duration = time.time() - capture_start_time
                self.log_capture_metrics(effective_config, capture_duration, success=True, cache_hit=True)

                self._logger.debug(f"Using cached ROI data (age: {current_time - self._last_capture_time:.3f}s)")
                return self._cached_roi_data
            else:
                self._logger.debug(f"Forcing new ROI capture - time_valid: {time_valid}, config_unchanged: {config_unchanged}")

            # 执行真实的截图操作
            roi_data = self._capture_roi_internal(effective_config)

            # 更新缓存和状态
            if roi_data is not None:
                self._cached_roi_data = roi_data
                self._last_roi_config = effective_config
                self._last_capture_time = current_time
                self._logger.debug("ROI captured successfully (gray_value=%.2f)", roi_data.gray_value)

                # 记录成功指标
                capture_duration = time.time() - capture_start_time
                self.log_capture_metrics(effective_config, capture_duration, success=True)

                # 集成历史存储 - 保存ROI帧到DataStore
                try:
                    from ..core.data_store import data_store
                    # 获取当前主信号帧数
                    _, main_frame_count, _, _, _, _ = data_store.get_status_snapshot()

                    # 添加ROI历史帧
                    roi_frame = data_store.add_roi_frame(
                        gray_value=roi_data.gray_value,
                        roi_config=effective_config,
                        frame_count=main_frame_count,
                        capture_duration=self._cache_interval
                    )

                    # 减少日志频率 - 每50帧记录一次，并改为debug级别
                    if roi_frame.index % 50 == 0:
                        self._logger.debug("ROI frame added to history: index=%d, gray_value=%.2f, main_frame=%d",
                                           roi_frame.index, roi_frame.gray_value, main_frame_count)

                except Exception as e:
                    self._logger.error("Failed to add ROI frame to history: %s", str(e))

            else:
                # 捕获失败，记录失败指标并尝试处理
                capture_duration = time.time() - capture_start_time
                self.log_capture_metrics(effective_config, capture_duration, success=False,
                                       failure_reason="Internal capture failed")

                return self.handle_capture_failure(effective_config, failure_count)

            return roi_data

        except Exception as e:
            # 异常处理
            capture_duration = time.time() - capture_start_time
            self.log_capture_metrics(roi_config, capture_duration, success=False,
                                   failure_reason=f"Exception: {str(e)}")

            self._logger.error("Failed to capture ROI: %s", str(e))
            return self.handle_capture_failure(roi_config, failure_count)

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

    def get_large_roi_frame_rate(self) -> int:
        """获取当前大ROI帧率设置"""
        return self._large_roi_frame_rate

    def set_large_roi_frame_rate(self, frame_rate: int) -> bool:
        """
        动态设置大ROI帧率（独立于主ROI帧率）

        Args:
            frame_rate: 新的帧率 (1-30 FPS，推荐默认5 FPS)

        Returns:
            bool: 设置是否成功
        """
        if 1 <= frame_rate <= 30:
            self._large_roi_frame_rate = frame_rate
            self._large_roi_cache_interval = 1.0 / frame_rate
            self._logger.info("Large ROI frame rate updated to %d FPS, cache interval: %.3f seconds",
                              frame_rate, self._large_roi_cache_interval)

            # 清除大ROI缓存，强制下次使用新帧率重新捕获
            self._cached_large_roi_data = None
            self._last_large_roi_config = None
            self._last_large_roi_capture_time = 0.0

            return True
        else:
            self._logger.error("Invalid large ROI frame rate: %d (must be 1-30)", frame_rate)
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

    def extract_small_roi(self, large_roi_image: Image.Image, small_roi_size: int = 50) -> Optional[Image.Image]:
        """
        从大ROI图像中心提取小ROI区域（50x50）

        Args:
            large_roi_image: 大ROI图像
            small_roi_size: 小ROI尺寸，默认50x50

        Returns:
            Optional[Image.Image]: 小ROI图像，失败返回None
        """
        try:
            if large_roi_image is None:
                self._logger.error("Large ROI image is None")
                return None

            # 获取大ROI图像尺寸
            large_width, large_height = large_roi_image.size

            # 验证大ROI尺寸是否足够包含小ROI
            if large_width < small_roi_size or large_height < small_roi_size:
                self._logger.error(
                    "Large ROI image too small for small ROI extraction: %dx%d < %dx%d",
                    large_width, large_height, small_roi_size, small_roi_size
                )
                return None

            # 计算中心点
            center_x = large_width // 2
            center_y = large_height // 2

            # 计算小ROI的边界（确保不超出大ROI边界）
            half_size = small_roi_size // 2
            x1 = max(0, center_x - half_size)
            y1 = max(0, center_y - half_size)
            x2 = min(large_width, x1 + small_roi_size)
            y2 = min(large_height, y1 + small_roi_size)

            # 如果边界计算导致尺寸不对，调整起始点
            if x2 - x1 < small_roi_size:
                x1 = max(0, x2 - small_roi_size)
            if y2 - y1 < small_roi_size:
                y1 = max(0, y2 - small_roi_size)

            # 截取中心小ROI区域
            small_roi_image = large_roi_image.crop((x1, y1, x1 + small_roi_size, y1 + small_roi_size))

            self._logger.debug(
                "Small ROI extracted from center: %dx%d from large ROI %dx%d, coordinates: (%d,%d)->(%d,%d)",
                small_roi_size, small_roi_size, large_width, large_height, x1, y1, x1 + small_roi_size, y1 + small_roi_size
            )

            return small_roi_image

        except Exception as e:
            self._logger.error("Failed to extract small ROI from center: %s", str(e))
            return None

    def capture_dual_roi(self, dual_config: DualRoiConfig, frame_count: int = 0,
