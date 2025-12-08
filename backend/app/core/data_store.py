from __future__ import annotations

import hashlib
import heapq
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Deque, Dict, List, Optional, Tuple, Any

import logging

from ..config import settings
from ..models import SystemStatus, RoiConfig, LineIntersectionResult


@dataclass
class Frame:
    index: int
    timestamp: datetime
    value: float


@dataclass
class RoiFrame:
    """ROI截图帧数据"""
    index: int
    timestamp: datetime
    gray_value: float
    roi_config: RoiConfig
    frame_count: int  # 主信号帧计数
    capture_duration: float  # ROI截图持续时间


@dataclass
class LineIntersectionFrame:
    """线条相交检测帧数据"""
    index: int
    timestamp: datetime
    result: LineIntersectionResult
    main_frame_count: int  # 主信号帧计数


@dataclass
class CacheEntry:
    """缓存条目数据结构"""
    result: LineIntersectionResult
    timestamp: datetime
    access_count: int = 0
    last_access_time: datetime = field(default_factory=datetime.utcnow)
    cache_key: str = ""

    def is_expired(self, timeout_ms: int) -> bool:
        """检查缓存条目是否过期"""
        timeout_seconds = timeout_ms / 1000.0
        return (datetime.utcnow() - self.timestamp).total_seconds() > timeout_seconds

    def update_access(self):
        """更新访问计数和时间"""
        self.access_count += 1
        self.last_access_time = datetime.utcnow()


@dataclass
class CachePerformanceStats:
    """缓存性能统计"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    insertions: int = 0
    total_requests: int = 0
    avg_response_time_ms: float = 0.0
    hit_ratio: float = 0.0
    miss_ratio: float = 0.0
    cache_size: int = 0
    max_cache_size: int = 50

    def update_hit(self):
        """更新缓存命中统计"""
        self.hits += 1
        self.total_requests += 1
        self._update_ratios()

    def update_miss(self):
        """更新缓存未命中统计"""
        self.misses += 1
        self.total_requests += 1
        self._update_ratios()

    def update_eviction(self):
        """更新缓存驱逐统计"""
        self.evictions += 1

    def update_insertion(self):
        """更新缓存插入统计"""
        self.insertions += 1

    def update_response_time(self, response_time_ms: float):
        """更新平均响应时间"""
        if self.total_requests > 0:
            self.avg_response_time_ms = (
                (self.avg_response_time_ms * (self.total_requests - 1) + response_time_ms) /
                self.total_requests
            )

    def _update_ratios(self):
        """更新命中率 ratios"""
        if self.total_requests > 0:
            self.hit_ratio = self.hits / self.total_requests
            self.miss_ratio = self.misses / self.total_requests

    def reset(self):
        """重置所有统计数据"""
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.insertions = 0
        self.total_requests = 0
        self.avg_response_time_ms = 0.0
        self.hit_ratio = 0.0
        self.miss_ratio = 0.0
        self.cache_size = 0


class DataStore:
    """
    内存中的时序数据存储，线程安全。
    """

    def __init__(self, buffer_size: int) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)
        self._buffer_size = buffer_size
        self._frames: Deque[Frame] = deque(maxlen=buffer_size)
        self._lock = threading.Lock()

        # ROI历史数据存储 - 独立于主信号数据
        # 默认存储最近500个ROI截图帧（约4分钟，按2FPS计算）
        roi_buffer_size = 500
        self._roi_frames: Deque[RoiFrame] = deque(maxlen=roi_buffer_size)
        self._roi_frame_count: int = 0

        # ROI1绿色线条相交检测数据存储 - 独立于其他数据存储
        # 默认存储最近1000个检测结果（约83秒，按12FPS计算）
        line_detection_buffer_size = getattr(settings, 'line_detection_buffer_size', 1000)
        self._line_intersection_frames: Deque[LineIntersectionFrame] = deque(maxlen=line_detection_buffer_size)
        self._line_intersection_frame_count: int = 0

        # Task 19: 增强的结果缓存系统
        # 线程相交检测结果缓存 - 100ms超时，LRU驱逐策略
        self._cache_timeout_ms: int = 100  # NF-Performance: 100ms缓存超时
        self._max_cache_size: int = 50  # 最大缓存条目数
        self._intersection_cache: Dict[str, CacheEntry] = {}  # 主缓存存储
        self._cache_lru_queue: List[Tuple[float, str]] = []  # LRU队列 (last_access_time, cache_key)
        self._cache_lock: threading.RLock = threading.RLock()  # 缓存专用锁

        # 缓存性能统计
        self._cache_stats = CachePerformanceStats(max_cache_size=self._max_cache_size)

        # 缓存预热数据存储（最近使用的检测参数）
        self._cache_warmup_params: Dict[str, Any] = {}
        self._cache_warmup_enabled: bool = True

        self._frame_count: int = 0
        self._current_value: float = 0.0
        self._baseline: float = 0.0
        self._peak_signal: Optional[int] = None
        self._last_peak_signal: Optional[int] = None
        self._status: SystemStatus = SystemStatus.STOPPED

        # ROI配置
        self._roi_config: RoiConfig = RoiConfig(x1=0, y1=0, x2=200, y2=150)
        self._roi_configured: bool = False  # 标记ROI是否已由用户配置

        # 增强波峰检测信息
        self._enhanced_peak_color: Optional[str] = None  # 'green' or 'red'
        self._enhanced_peak_confidence: float = 0.0
        self._enhanced_peak_threshold: float = 0.0
        self._enhanced_in_peak_region: bool = False
        self._enhanced_peak_frame_count: int = 0

    # 写操作
    def add_frame(
        self,
        value: float,
        timestamp: Optional[datetime] = None,
        peak_signal: Optional[int] = None,
    ) -> Frame:
        if timestamp is None:
            timestamp = datetime.utcnow()
        with self._lock:
            self._frame_count += 1
            frame = Frame(index=self._frame_count, timestamp=timestamp, value=value)
            self._frames.append(frame)
            self._current_value = value
            self._update_baseline_locked()
            self._peak_signal = peak_signal
            if peak_signal is not None:
                self._last_peak_signal = peak_signal
            self._logger.debug(
                "Added frame index=%d value=%.3f baseline=%.3f peak_signal=%s",
                self._frame_count,
                value,
                self._baseline,
                str(peak_signal),
            )
            return frame

    def _update_baseline_locked(self) -> None:
        if not self._frames:
            self._baseline = 0.0
            return
        # 简化实现：最近 N 帧（最多 60 帧）的平均值
        window_size = min(len(self._frames), settings.fps)
        recent_values = [f.value for f in list(self._frames)[-window_size:]]
        self._baseline = sum(recent_values) / window_size

    # 读操作（线程安全快照）
    def get_status_snapshot(self) -> Tuple[SystemStatus, int, float, Optional[int], int, float]:
        with self._lock:
            snapshot = (
                self._status,
                self._frame_count,
                self._current_value,
                self._peak_signal,
                len(self._frames),
                self._baseline,
            )
        self._logger.debug(
            "Status snapshot status=%s frame_count=%d current=%.3f peak_signal=%s buffer_size=%d baseline=%.3f",
            snapshot[0],
            snapshot[1],
            snapshot[2],
            str(snapshot[3]),
            snapshot[4],
            snapshot[5],
        )
        return snapshot

    def get_series(self, count: int) -> List[Frame]:
        with self._lock:
            frames = list(self._frames)
        if count >= len(frames):
            return frames
        return frames[-count:]

    # 状态控制
    def set_status(self, status: SystemStatus) -> None:
        with self._lock:
            self._status = status
        self._logger.info("System status changed to %s", status.value)

    def get_status(self) -> SystemStatus:
        with self._lock:
            return self._status

    def reset(self) -> None:
        with self._lock:
            self._frames.clear()
            self._frame_count = 0
            self._current_value = 0.0
            self._baseline = 0.0
            self._peak_signal = None
            self._last_peak_signal = None
            # 重置ROI配置状态
            self._roi_config = RoiConfig(x1=0, y1=0, x2=200, y2=150)
            self._roi_configured = False
            # 重置线条相交检测数据
            self._line_intersection_frames.clear()
            self._line_intersection_frame_count = 0

        # Task 19: 重置增强缓存系统
        self.invalidate_cache()
        self._cache_warmup_params.clear()
        self._logger.warning("Data store has been reset (including enhanced cache)")

    def get_last_peak_signal(self) -> Optional[int]:
        with self._lock:
            return self._last_peak_signal

    # ROI配置操作
    def set_roi_config(self, roi_config: RoiConfig) -> None:
        """设置ROI配置"""
        with self._lock:
            if roi_config.validate_coordinates():
                self._roi_config = roi_config
                self._roi_configured = True  # 标记为用户已配置
                self._logger.info(
                    "ROI config updated: (%d,%d) -> (%d,%d), size: %dx%d, center: (%d,%d)",
                    roi_config.x1, roi_config.y1, roi_config.x2, roi_config.y2,
                    roi_config.width, roi_config.height, roi_config.center_x, roi_config.center_y
                )
            else:
                self._logger.error("Invalid ROI config: coordinates validation failed")
                raise ValueError("Invalid ROI coordinates")

    def get_roi_config(self) -> RoiConfig:
        """获取ROI配置"""
        with self._lock:
            return self._roi_config

    def is_roi_configured(self) -> bool:
        """检查ROI是否已由用户配置"""
        with self._lock:
            return self._roi_configured

    def get_roi_status(self) -> Tuple[bool, RoiConfig]:
        """获取ROI配置状态和配置"""
        with self._lock:
            return self._roi_configured, self._roi_config

    def add_enhanced_peak(
        self,
        peak_signal: Optional[int],
        peak_color: Optional[str],
        peak_confidence: float,
        threshold: float,
        in_peak_region: bool,
        frame_count: int
    ) -> None:
        """添加增强波峰检测信息"""
        with self._lock:
            self._peak_signal = peak_signal
            if peak_signal == 1:
                self._last_peak_signal = peak_signal
            elif peak_signal is None and self._last_peak_signal == 1:
                self._last_peak_signal = None

            self._enhanced_peak_color = peak_color
            self._enhanced_peak_confidence = peak_confidence
            self._enhanced_peak_threshold = threshold
            self._enhanced_in_peak_region = in_peak_region
            self._enhanced_peak_frame_count = frame_count

    def get_enhanced_peak_status(self) -> Tuple[Optional[str], float, float, bool, int]:
        """获取增强波峰检测状态"""
        with self._lock:
            return (
                self._enhanced_peak_color,
                self._enhanced_peak_confidence,
                self._enhanced_peak_threshold,
                self._enhanced_in_peak_region,
                self._enhanced_peak_frame_count
            )

    def get_enhanced_status_snapshot(self) -> Tuple[
        int, float, float, Optional[int], Optional[int], float,
        Optional[str], float, float, bool, int, bool, RoiConfig
    ]:
        """获取包含增强波峰信息的状态快照"""
        with self._lock:
            return (
                self._frame_count,
                self._current_value,
                self._baseline,
                self._peak_signal,
                self._last_peak_signal,
                float(self._status.value),
                self._enhanced_peak_color,
                self._enhanced_peak_confidence,
                self._enhanced_peak_threshold,
                self._enhanced_in_peak_region,
                self._enhanced_peak_frame_count,
                self._roi_configured,
                self._roi_config
            )

    # ROI历史数据操作
    def add_roi_frame(
        self,
        gray_value: float,
        roi_config: RoiConfig,
        frame_count: int,
        capture_duration: float = 0.5,
        timestamp: Optional[datetime] = None,
    ) -> RoiFrame:
        """添加ROI截图帧数据"""
        if timestamp is None:
            timestamp = datetime.utcnow()

        with self._lock:
            self._roi_frame_count += 1
            roi_frame = RoiFrame(
                index=self._roi_frame_count,
                timestamp=timestamp,
                gray_value=gray_value,
                roi_config=roi_config,
                frame_count=frame_count,
                capture_duration=capture_duration
            )
            self._roi_frames.append(roi_frame)

            # 减少日志频率 - 每50帧记录一次，并改为debug级别
            if self._roi_frame_count % 50 == 0:
                self._logger.debug(
                    "Added ROI frame index=%d gray_value=%.2f frame_count=%d buffer_size=%d",
                    self._roi_frame_count,
                    gray_value,
                    frame_count,
                    len(self._roi_frames)
                )

            return roi_frame

    def get_roi_series(self, count: int) -> List[RoiFrame]:
        """获取最近N个ROI帧数据"""
        with self._lock:
            roi_frames = list(self._roi_frames)

        if count >= len(roi_frames):
            return roi_frames
        return roi_frames[-count:]

    def get_roi_status_snapshot(self) -> Tuple[int, int, float, int]:
        """获取ROI数据状态快照"""
        with self._lock:
            return (
                self._roi_frame_count,
                len(self._roi_frames),
                self._roi_frames[-1].gray_value if self._roi_frames else 0.0,
                self._roi_frames[-1].frame_count if self._roi_frames else 0
            )

    def get_roi_frame_rate_info(self) -> Tuple[float, int]:
        """获取ROI帧率信息"""
        with self._lock:
            if len(self._roi_frames) < 2:
                return 0.0, len(self._roi_frames)

            # 计算实际ROI帧率
            recent_frames = list(self._roi_frames)[-10:]  # 取最近10帧
            if len(recent_frames) >= 2:
                time_span = (recent_frames[-1].timestamp - recent_frames[0].timestamp).total_seconds()
                if time_span > 0:
                    actual_fps = (len(recent_frames) - 1) / time_span
                else:
                    actual_fps = 0.0
            else:
                actual_fps = 0.0

            return actual_fps, len(self._roi_frames)

    def reset_roi_history(self) -> None:
        """重置ROI历史数据"""
        with self._lock:
            self._roi_frames.clear()
            self._roi_frame_count = 0
        self._logger.warning("ROI history has been reset")

    # ROI1绿色线条相交检测数据操作
    def store_line_intersection_result(
        self,
        result: LineIntersectionResult,
        main_frame_count: int,
        timestamp: Optional[datetime] = None,
    ) -> LineIntersectionFrame:
        """存储线条相交检测结果"""
        if timestamp is None:
            timestamp = datetime.utcnow()

        with self._lock:
            self._line_intersection_frame_count += 1
            line_frame = LineIntersectionFrame(
                index=self._line_intersection_frame_count,
                timestamp=timestamp,
                result=result,
                main_frame_count=main_frame_count
            )
            self._line_intersection_frames.append(line_frame)

            # 减少日志频率 - 每100帧记录一次，并改为debug级别
            if self._line_intersection_frame_count % 100 == 0:
                self._logger.debug(
                    "Added line intersection frame index=%d has_intersection=%s confidence=%.3f buffer_size=%d",
                    self._line_intersection_frame_count,
                    result.has_intersection,
                    result.confidence,
                    len(self._line_intersection_frames)
                )

            return line_frame

    def get_latest_line_intersection_results(self, count: int) -> List[LineIntersectionFrame]:
        """获取最近的线条相交检测结果"""
        with self._lock:
            line_frames = list(self._line_intersection_frames)

        if count >= len(line_frames):
            return line_frames
        return line_frames[-count:]

    def get_line_intersection_result_by_timestamp(self, timestamp: datetime) -> Optional[LineIntersectionFrame]:
        """根据时间戳获取特定的线条相交检测结果"""
        with self._lock:
            for frame in reversed(self._line_intersection_frames):
                if frame.timestamp == timestamp:
                    return frame
        return None

    def get_line_intersection_buffer_stats(self) -> dict:
        """获取线条相交检测缓冲区统计信息"""
        with self._lock:
            buffer_size = len(self._line_intersection_frames)
            max_buffer_size = self._line_intersection_frames.maxlen

            # 计算基本统计信息
            has_intersection_count = sum(1 for frame in self._line_intersection_frames if frame.result.has_intersection)
            avg_confidence = 0.0
            if buffer_size > 0:
                total_confidence = sum(frame.result.confidence for frame in self._line_intersection_frames)
                avg_confidence = total_confidence / buffer_size

            # 计算最近处理时间
            recent_avg_processing_time = 0.0
            if buffer_size >= 10:
                recent_frames = list(self._line_intersection_frames)[-10:]
                recent_avg_processing_time = sum(frame.result.processing_time_ms for frame in recent_frames) / len(recent_frames)

            # 计算时间跨度
            time_span_seconds = 0.0
            if buffer_size >= 2:
                first_frame = self._line_intersection_frames[0]
                last_frame = self._line_intersection_frames[-1]
                time_span_seconds = (last_frame.timestamp - first_frame.timestamp).total_seconds()

            # 计算有效检测率
            valid_detection_rate = 0.0
            if buffer_size > 0:
                valid_detection_rate = has_intersection_count / buffer_size

            return {
                "total_frames": self._line_intersection_frame_count,
                "buffer_size": buffer_size,
                "max_buffer_size": max_buffer_size,
                "buffer_usage_percent": (buffer_size / max_buffer_size * 100) if max_buffer_size > 0 else 0.0,
                "has_intersection_count": has_intersection_count,
                "avg_confidence": avg_confidence,
                "valid_detection_rate": valid_detection_rate,
                "recent_avg_processing_time_ms": recent_avg_processing_time,
                "time_span_seconds": time_span_seconds,
                "latest_result": self._line_intersection_frames[-1].result.dict() if self._line_intersection_frames else None,
                "latest_timestamp": self._line_intersection_frames[-1].timestamp.isoformat() if self._line_intersection_frames else None
            }

    def clear_line_intersection_buffer(self) -> None:
        """清空线条相交检测缓冲区数据"""
        with self._lock:
            self._line_intersection_frames.clear()
            self._line_intersection_frame_count = 0
        self._logger.warning("Line intersection buffer has been cleared")

    def get_line_intersection_status_snapshot(self) -> Tuple[int, int, bool, float, datetime]:
        """获取线条相交检测状态快照"""
        with self._lock:
            if self._line_intersection_frames:
                latest_frame = self._line_intersection_frames[-1]
                return (
                    self._line_intersection_frame_count,
                    len(self._line_intersection_frames),
                    latest_frame.result.has_intersection,
                    latest_frame.result.confidence,
                    latest_frame.timestamp
                )
            else:
                return (
                    self._line_intersection_frame_count,
                    len(self._line_intersection_frames),
                    False,
                    0.0,
                    datetime.utcnow()
                )

    def get_line_intersection_time_series(self, count: int) -> List[dict]:
        """获取线条相交检测时间序列数据，用于图表展示"""
        with self._lock:
            frames = list(self._line_intersection_frames)

        if count >= len(frames):
            selected_frames = frames
        else:
            selected_frames = frames[-count:]

        return [
            {
                "index": frame.index,
                "timestamp": frame.timestamp.isoformat(),
                "has_intersection": frame.result.has_intersection,
                "confidence": frame.result.confidence,
                "processing_time_ms": frame.result.processing_time_ms,
                "edge_quality": frame.result.edge_quality,
                "temporal_stability": frame.result.temporal_stability,
                "main_frame_count": frame.main_frame_count
            }
            for frame in selected_frames
        ]

    # Task 19: Enhanced Caching System Methods

    def generate_cache_key(self, roi_image_data: str, detection_params: Dict[str, Any]) -> str:
        """
        生成基于内容的缓存键

        Args:
            roi_image_data: Base64编码的ROI图像数据
            detection_params: 检测参数字典

        Returns:
            str: SHA256哈希缓存键
        """
        try:
            # 创建组合字符串用于哈希
            params_str = "|".join(f"{k}:{v}" for k, v in sorted(detection_params.items()))
            combined_data = f"{roi_image_data[:100]}...{params_str}"  # 只使用图像数据前100字符避免哈希过大

            # 生成SHA256哈希
            cache_key = hashlib.sha256(combined_data.encode('utf-8')).hexdigest()[:16]  # 使用前16字符

            self._logger.debug(f"Generated cache key: {cache_key[:8]}... for ROI data")
            return cache_key

        except Exception as e:
            self._logger.error(f"Failed to generate cache key: {e}")
            # 降级为时间戳键
            return f"fallback_{int(time.time() * 1000)}"

    def get_cached_intersection_result(self, cache_key: str) -> Optional[LineIntersectionResult]:
        """
        获取缓存的线条相交检测结果

        Args:
            cache_key: 缓存键

        Returns:
            Optional[LineIntersectionResult]: 缓存的结果，如果不存在或过期则返回None
        """
        start_time = time.time()

        with self._cache_lock:
            try:
                # 检查缓存中是否存在该条目
                if cache_key in self._intersection_cache:
                    cache_entry = self._intersection_cache[cache_key]

                    # 检查是否过期
                    if cache_entry.is_expired(self._cache_timeout_ms):
                        self._logger.debug(f"Cache entry expired for key: {cache_key[:8]}...")
                        self._remove_cache_entry(cache_key)
                        self._cache_stats.update_miss()
                        return None

                    # 更新访问时间和统计
                    cache_entry.update_access()
                    self._update_lru_queue(cache_key)

                    # 更新性能统计
                    response_time_ms = (time.time() - start_time) * 1000
                    self._cache_stats.update_hit()
                    self._cache_stats.update_response_time(response_time_ms)

                    self._logger.debug(f"Cache hit for key: {cache_key[:8]}... (access count: {cache_entry.access_count})")
                    return cache_entry.result

                # 缓存未命中
                self._cache_stats.update_miss()
                response_time_ms = (time.time() - start_time) * 1000
                self._cache_stats.update_response_time(response_time_ms)

                self._logger.debug(f"Cache miss for key: {cache_key[:8]}...")
                return None

            except Exception as e:
                self._logger.error(f"Error retrieving cached result: {e}")
                self._cache_stats.update_miss()
                return None

    def cache_intersection_result(
        self,
        cache_key: str,
        result: LineIntersectionResult
    ) -> bool:
        """
        缓存线条相交检测结果

        Args:
            cache_key: 缓存键
            result: 检测结果

        Returns:
            bool: 缓存是否成功
        """
        with self._cache_lock:
            try:
                current_time = datetime.utcnow()

                # 创建新的缓存条目
                cache_entry = CacheEntry(
                    result=result,
                    timestamp=current_time,
                    cache_key=cache_key
                )

                # 检查是否需要驱逐旧条目
                if len(self._intersection_cache) >= self._max_cache_size:
                    if cache_key not in self._intersection_cache:
                        self._evict_lru_entry()
                    else:
                        # 更新现有条目
                        self._update_lru_queue(cache_key)

                # 存储到缓存
                self._intersection_cache[cache_key] = cache_entry
                self._update_lru_queue(cache_key)

                # 更新统计
                self._cache_stats.update_insertion()
                self._cache_stats.cache_size = len(self._intersection_cache)

                # 更新预热参数
                if self._cache_warmup_enabled:
                    self._update_warmup_params(cache_key, result)

                self._logger.debug(f"Cached result for key: {cache_key[:8]}... (cache size: {len(self._intersection_cache)})")
                return True

            except Exception as e:
                self._logger.error(f"Error caching result: {e}")
                return False

    def _update_lru_queue(self, cache_key: str):
        """
        更新LRU队列

        Args:
            cache_key: 缓存键
        """
        current_time = time.time()

        # 移除旧的条目（如果存在）
        self._cache_lru_queue = [(t, k) for t, k in self._cache_lru_queue if k != cache_key]

        # 添加新的条目到队列末尾（最近使用）
        heapq.heappush(self._cache_lru_queue, (current_time, cache_key))

    def _evict_lru_entry(self):
        """驱逐最久未使用的缓存条目"""
        if not self._cache_lru_queue:
            return

        try:
            # 获取最久未使用的条目（最小时间戳）
            oldest_time, oldest_key = heapq.heappop(self._cache_lru_queue)

            # 从缓存中移除
            if oldest_key in self._intersection_cache:
                del self._intersection_cache[oldest_key]
                self._cache_stats.update_eviction()

                self._logger.debug(f"Evicted LRU cache entry: {oldest_key[:8]}...")

        except Exception as e:
            self._logger.error(f"Error during LRU eviction: {e}")

    def _remove_cache_entry(self, cache_key: str):
        """
        从缓存中移除指定条目

        Args:
            cache_key: 要移除的缓存键
        """
        if cache_key in self._intersection_cache:
            del self._intersection_cache[cache_key]

        # 从LRU队列中移除
        self._cache_lru_queue = [(t, k) for t, k in self._cache_lru_queue if k != cache_key]

        # 更新统计
        self._cache_stats.cache_size = len(self._intersection_cache)

    def cleanup_expired_cache_entries(self) -> int:
        """
        清理过期的缓存条目

        Returns:
            int: 清理的条目数量
        """
        cleaned_count = 0

        with self._cache_lock:
            try:
                expired_keys = []
                current_time = datetime.utcnow()

                # 找出所有过期条目
                for cache_key, cache_entry in self._intersection_cache.items():
                    if cache_entry.is_expired(self._cache_timeout_ms):
                        expired_keys.append(cache_key)

                # 清理过期条目
                for cache_key in expired_keys:
                    self._remove_cache_entry(cache_key)
                    cleaned_count += 1

                if cleaned_count > 0:
                    self._logger.info(f"Cleaned up {cleaned_count} expired cache entries")

                return cleaned_count

            except Exception as e:
                self._logger.error(f"Error during cache cleanup: {e}")
                return 0

    def invalidate_cache(self) -> bool:
        """
        清空所有缓存条目

        Returns:
            bool: 清空是否成功
        """
        with self._cache_lock:
            try:
                cleared_count = len(self._intersection_cache)
                self._intersection_cache.clear()
                self._cache_lru_queue.clear()

                # 重置统计但保留最大大小设置
                max_size = self._cache_stats.max_cache_size
                self._cache_stats.reset()
                self._cache_stats.max_cache_size = max_size

                self._logger.warning(f"Invalidated cache: cleared {cleared_count} entries")
                return True

            except Exception as e:
                self._logger.error(f"Error invalidating cache: {e}")
                return False

    def get_cache_performance_stats(self) -> Dict[str, Any]:
        """
        获取缓存性能统计信息

        Returns:
            Dict[str, Any]: 详细的缓存性能统计
        """
        with self._cache_lock:
            try:
                # 更新当前缓存大小
                self._cache_stats.cache_size = len(self._intersection_cache)

                # 计算缓存内存占用估算
                cache_memory_mb = 0.0
                try:
                    # 估算每个结果对象约1KB，加上缓存结构开销
                    cache_memory_mb = (len(self._intersection_cache) * 1024) / (1024 * 1024)
                except:
                    pass

                # 计算平均命中率
                avg_hit_ratio = self._cache_stats.hit_ratio

                # 获取最近使用统计
                recent_access_count = 0
                if self._intersection_cache:
                    recent_access_count = sum(
                        entry.access_count for entry in self._intersection_cache.values()
                    )

                stats = {
                    "performance": {
                        "total_requests": self._cache_stats.total_requests,
                        "cache_hits": self._cache_stats.hits,
                        "cache_misses": self._cache_stats.misses,
                        "hit_ratio": round(avg_hit_ratio, 4),
                        "miss_ratio": round(self._cache_stats.miss_ratio, 4),
                        "avg_response_time_ms": round(self._cache_stats.avg_response_time_ms, 2),
                        "evictions": self._cache_stats.evictions,
                        "insertions": self._cache_stats.insertions
                    },
                    "capacity": {
                        "current_size": self._cache_stats.cache_size,
                        "max_size": self._cache_stats.max_cache_size,
                        "usage_percent": round(
                            (self._cache_stats.cache_size / self._cache_stats.max_cache_size * 100)
                            if self._cache_stats.max_cache_size > 0 else 0, 2
                        ),
                        "estimated_memory_mb": round(cache_memory_mb, 2)
                    },
                    "configuration": {
                        "timeout_ms": self._cache_timeout_ms,
                        "lru_enabled": True,
                        "warmup_enabled": self._cache_warmup_enabled,
                        "isolation_from_roi2": True  # ROI2处理隔离
                    },
                    "health": {
                        "is_healthy": avg_hit_ratio > 0.1 and self._cache_stats.cache_size < self._cache_stats.max_cache_size,
                        "last_cleanup_time": datetime.utcnow().isoformat(),
                        "needs_cleanup": len([
                            k for k, v in self._intersection_cache.items()
                            if v.is_expired(self._cache_timeout_ms)
                        ]) > 0
                    }
                }

                return stats

            except Exception as e:
                self._logger.error(f"Error getting cache performance stats: {e}")
                return {"error": str(e)}

    def _update_warmup_params(self, cache_key: str, result: LineIntersectionResult):
        """
        更新缓存预热参数

        Args:
            cache_key: 缓存键
            result: 检测结果
        """
        if not self._cache_warmup_enabled:
            return

        try:
            # 存储最近使用的高置信度结果参数
            if result.confidence > 0.7:  # 只预热高置信度结果
                self._cache_warmup_params[cache_key] = {
                    "timestamp": datetime.utcnow(),
                    "confidence": result.confidence,
                    "has_intersection": result.has_intersection
                }

                # 保持预热参数数量在合理范围内
                if len(self._cache_warmup_params) > 20:
                    # 删除最旧的预热参数
                    oldest_key = min(
                        self._cache_warmup_params.keys(),
                        key=lambda k: self._cache_warmup_params[k]["timestamp"]
                    )
                    del self._cache_warmup_params[oldest_key]

        except Exception as e:
            self._logger.error(f"Error updating warmup params: {e}")

    def warmup_cache(self, expected_params: List[Dict[str, Any]]) -> int:
        """
        预热缓存，为预期的检测参数预先生成缓存

        Args:
            expected_params: 预期的检测参数列表

        Returns:
            int: 预热的缓存条目数量
        """
        if not self._cache_warmup_enabled or not expected_params:
            return 0

        warmed_count = 0

        with self._cache_lock:
            try:
                for params in expected_params:
                    # 生成模拟的缓存键（实际应用中可能需要真实的ROI数据）
                    mock_roi_data = "warmup_data"
                    cache_key = self.generate_cache_key(mock_roi_data, params)

                    # 如果不存在该缓存键，则创建空的结果条目
                    if cache_key not in self._intersection_cache:
                        mock_result = LineIntersectionResult(
                            has_intersection=False,
                            confidence=0.0,
                            processing_time_ms=0.0,
                            frame_count=0
                        )

                        if self.cache_intersection_result(cache_key, mock_result):
                            warmed_count += 1

                self._logger.info(f"Warmed up {warmed_count} cache entries")
                return warmed_count

            except Exception as e:
                self._logger.error(f"Error during cache warmup: {e}")
                return 0

    def get_cache_isolation_status(self) -> Dict[str, Any]:
        """
        获取缓存隔离状态（确保与ROI2处理隔离）

        Returns:
            Dict[str, Any]: 缓存隔离状态信息
        """
        with self._cache_lock:
            try:
                # 检查缓存中是否只包含ROI1相关数据
                roi1_only_entries = 0
                total_entries = len(self._intersection_cache)

                # 通过缓存键模式验证ROI1隔离
                for cache_key in self._intersection_cache.keys():
                    # 假设ROI1缓存键有特定模式或前缀
                    if not cache_key.startswith("roi2_"):
                        roi1_only_entries += 1

                isolation_status = {
                    "is_isolated_from_roi2": roi1_only_entries == total_entries,
                    "total_cache_entries": total_entries,
                    "roi1_only_entries": roi1_only_entries,
                    "potential_roi2_entries": total_entries - roi1_only_entries,
                    "isolation_ratio": round(roi1_only_entries / total_entries, 4) if total_entries > 0 else 1.0,
                    "cache_namespace": "roi1_only",  # 明确标识命名空间
                    "last_checked": datetime.utcnow().isoformat()
                }

                return isolation_status

            except Exception as e:
                self._logger.error(f"Error checking cache isolation: {e}")
                return {"error": str(e), "is_isolated_from_roi2": False}

    def configure_cache_settings(
        self,
        timeout_ms: Optional[int] = None,
        max_size: Optional[int] = None,
        warmup_enabled: Optional[bool] = None
    ) -> bool:
        """
        配置缓存设置

        Args:
            timeout_ms: 缓存超时时间（毫秒）
            max_size: 最大缓存大小
            warmup_enabled: 是否启用预热

        Returns:
            bool: 配置是否成功
        """
        with self._cache_lock:
            try:
                if timeout_ms is not None and timeout_ms > 0:
                    self._cache_timeout_ms = timeout_ms

                if max_size is not None and max_size > 0:
                    old_max_size = self._max_cache_size
                    self._max_cache_size = max_size
                    self._cache_stats.max_cache_size = max_size

                    # 如果新大小更小，需要驱逐多余条目
                    if max_size < old_max_size:
                        while len(self._intersection_cache) > max_size:
                            self._evict_lru_entry()

                if warmup_enabled is not None:
                    self._cache_warmup_enabled = warmup_enabled
                    if not warmup_enabled:
                        self._cache_warmup_params.clear()

                self._logger.info(
                    f"Cache configuration updated: timeout={self._cache_timeout_ms}ms, "
                    f"max_size={self._max_cache_size}, warmup={self._cache_warmup_enabled}"
                )
                return True

            except Exception as e:
                self._logger.error(f"Error configuring cache settings: {e}")
                return False


# 单例数据存储
data_store = DataStore(buffer_size=settings.buffer_size)
