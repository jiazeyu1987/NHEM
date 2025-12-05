# 波形截取功能后端逻辑文档

## 概述

波形截取功能的后端实现基于 FastAPI 框架，提供三个核心API端点来支持不同类型的数据截取需求。该功能主要用于从实时数据流中提取指定长度的历史数据，支持主信号数据和ROI灰度数据，并可选择性地进行波峰检测分析。

## API端点架构

### 1. 主信号数据截取端点

#### 端点定义
```python
@router.get("/data/window-capture", response_model=WindowCaptureResponse)
async def window_capture(
    count: int = Query(100, ge=50, le=200, description="窗口大小：50-200帧")
) -> WindowCaptureResponse:
    """截取指定帧数的历史数据窗口"""
```

#### 处理逻辑
```python
async def window_capture(count: int) -> WindowCaptureResponse:
    logger.info("🖼️ Window capture requested: count=%d", count)

    # 1. 从数据存储中获取指定数量的历史帧
    frames = data_store.get_series(count)
    if not frames:
        logger.warning("Window capture failed: no data available")
        raise HTTPException(status_code=404, detail="No data available for capture")

    # 2. 获取当前状态信息
    status, current_frame_count, current_value, peak_signal, buffer_size, baseline = data_store.get_status_snapshot()

    # 3. 计算帧范围
    frame_count = len(frames)
    start_frame = max(0, current_frame_count - frame_count)
    end_frame = current_frame_count - 1

    # 4. 转换为API响应格式
    series = []
    for frame in frames:
        series.append(TimeSeriesPoint(
            t=frame.timestamp.strftime('%H:%M:%S.%f')[:-3],
            value=frame.value
        ))

    # 5. 计算元数据
    duration = frame_count / settings.data_fps if frame_count > 0 else 0.0
    values = [frame.value for frame in frames]
    value_range = [min(values), max(values)] if values else [0.0, 0.0]

    capture_metadata = {
        "duration": duration,
        "fps": settings.data_fps,
        "value_range": value_range,
        "baseline": baseline,
        "start_frame": start_frame,
        "end_frame": end_frame
    }

    logger.info("✅ Window capture successful: frames=%d, range=(%d,%d), duration=%.3fs",
               frame_count, start_frame, end_frame, duration)

    return WindowCaptureResponse(
        timestamp=datetime.utcnow(),
        window_size=count,
        frame_range=(start_frame, end_frame),
        series=series,
        capture_metadata=capture_metadata
    )
```

### 2. ROI数据截取端点

#### 端点定义
```python
@router.get("/data/roi-window-capture", response_model=RoiWindowCaptureResponse)
async def roi_window_capture(
    count: int = Query(100, ge=50, le=500, description="ROI窗口大小：50-500帧")
) -> RoiWindowCaptureResponse:
    """截取指定帧数的ROI灰度分析历史数据窗口"""
```

#### 处理逻辑
```python
async def roi_window_capture(count: int) -> RoiWindowCaptureResponse:
    logger.info("🖼️ ROI window capture requested: count=%d", count)

    # 1. 从数据存储中获取指定数量的ROI历史帧
    roi_frames = data_store.get_roi_series(count)
    if not roi_frames:
        logger.warning("ROI window capture failed: no ROI data available")
        raise HTTPException(status_code=404, detail="No ROI data available for capture")

    # 2. 获取当前状态信息
    _, current_main_frame_count, _, _, _, _ = data_store.get_status_snapshot()
    roi_count, roi_buffer_size, last_gray_value, last_main_frame_count = data_store.get_roi_status_snapshot()

    # 3. 计算帧范围
    roi_start_frame = max(0, roi_count - len(roi_frames))
    roi_end_frame = roi_count - 1

    # 4. 转换为API响应格式
    series = []
    main_frame_start = None
    main_frame_end = None

    for roi_frame in roi_frames:
        series.append(RoiTimeSeriesPoint(
            t=roi_frame.timestamp.strftime('%H:%M:%S.%f')[:-3],
            gray_value=roi_frame.gray_value,
            roi_index=roi_frame.index
        ))

        # 记录主信号帧范围
        if main_frame_start is None:
            main_frame_start = roi_frame.frame_count
        main_frame_end = roi_frame.frame_count

    # 5. 获取ROI配置信息
    roi_config = {}
    if roi_frames:
        last_roi_frame = roi_frames[-1]
        roi_config = {
            "x1": last_roi_frame.roi_config.x1,
            "y1": last_roi_frame.roi_config.y1,
            "x2": last_roi_frame.roi_config.x2,
            "y2": last_roi_frame.roi_config.y2,
            "width": last_roi_frame.roi_config.width,
            "height": last_roi_frame.roi_config.height
        }

    # 6. 计算元数据
    duration = sum(roi_frame.capture_duration for roi_frame in roi_frames)
    gray_values = [roi_frame.gray_value for roi_frame in roi_frames]
    gray_range = [min(gray_values), max(gray_values)] if gray_values else [0.0, 0.0]

    capture_metadata = {
        "capture_duration": duration,
        "roi_frame_start": roi_start_frame,
        "roi_frame_end": roi_end_frame,
        "main_frame_start": main_frame_start,
        "main_frame_end": main_frame_end,
        "gray_range": gray_range,
        "last_gray_value": last_gray_value,
        "roi_buffer_size": roi_buffer_size
    }

    # 7. 获取ROI帧率信息
    actual_fps, available_frames = data_store.get_roi_frame_rate_info()
    capture_metadata["actual_roi_fps"] = actual_fps
    capture_metadata["available_roi_frames"] = available_frames

    logger.info("✅ ROI window capture successful: frames=%d, roi_range=(%d,%d), main_range=(%d,%d), duration=%.3fs",
               len(series), roi_start_frame, roi_end_frame,
               capture_metadata["main_frame_start"], capture_metadata["main_frame_end"],
               capture_metadata["capture_duration"])

    return RoiWindowCaptureResponse(
        timestamp=datetime.utcnow(),
        window_size=count,
        roi_frame_range=(roi_start_frame, roi_end_frame),
        main_frame_range=(capture_metadata["main_frame_start"], capture_metadata["main_frame_end"]),
        series=series,
        roi_config=roi_config,
        capture_metadata=capture_metadata
    )
```

### 3. ROI数据截取带波峰检测端点

#### 端点定义
```python
@router.get("/data/roi-window-capture-with-peaks", response_model=RoiWindowCaptureWithPeaksResponse)
async def roi_window_capture_with_peaks(
    count: int = Query(100, ge=50, le=500, description="ROI窗口大小：50-500帧"),
    threshold: Optional[float] = Query(None, ge=0.0, le=200.0, description="波峰检测阈值：0-200（留空使用配置值）"),
    margin_frames: Optional[int] = Query(None, ge=1, le=20, description="边界扩展帧数：1-20（留空使用配置值）"),
    difference_threshold: Optional[float] = Query(None, ge=0.1, le=10.0, description="帧差值阈值：0.1-10.0（留空使用配置值）"),
    force_refresh: bool = Query(False, description="强制刷新缓存，获取最新数据")
) -> RoiWindowCaptureWithPeaksResponse:
    """截取指定帧数的ROI灰度分析历史数据窗口并进行波峰检测分析"""
```

#### 处理逻辑
```python
async def roi_window_capture_with_peaks(
    count, threshold, margin_frames, difference_threshold, force_refresh
) -> RoiWindowCaptureWithPeaksResponse:

    # 1. 使用默认值处理
    if threshold is None:
        threshold = settings.peak_threshold
    if margin_frames is None:
        margin_frames = settings.peak_margin_frames
    if difference_threshold is None:
        difference_threshold = settings.peak_difference_threshold

    logger.info("🔍 ROI window capture with peak detection requested: count=%d, threshold=%.1f, margin=%d, diff=%.2f, force_refresh=%s",
                count, threshold, margin_frames, difference_threshold, force_refresh)

    # 2. 强制刷新处理
    if force_refresh:
        roi_capture_service.clear_cache()
        logger.info("🔄 ROI cache cleared due to force_refresh=True")

    # 3. 获取ROI数据
    roi_frames = data_store.get_roi_series(count)
    if not roi_frames:
        logger.warning("ROI window capture with peaks failed: no ROI data available")
        raise HTTPException(status_code=404, detail="No ROI data available for capture")

    # 4. 数据转换 (与roi_window_capture相同)
    series = []
    for roi_frame in roi_frames:
        series.append(RoiTimeSeriesPoint(
            t=roi_frame.timestamp.strftime('%H:%M:%S.%f')[:-3],
            gray_value=roi_frame.gray_value,
            roi_index=roi_frame.index
        ))

    # 5. 波峰检测分析
    peak_detection_results = {}
    peak_detection_params = {
        "threshold": threshold,
        "margin_frames": margin_frames,
        "difference_threshold": difference_threshold,
        "data_count": len(roi_frames),
        "algorithm_version": "enhanced_v2.0"
    }

    try:
        # 提取灰度值数据
        gray_values = [frame.gray_value for frame in roi_frames]

        # 使用增强的波峰检测器
        enhanced_detector = EnhancedPeakDetector()
        peaks_info = enhanced_detector.detect_peaks_enhanced(
            data=gray_values,
            threshold=threshold,
            margin_frames=margin_frames,
            difference_threshold=difference_threshold
        )

        # 格式化波峰检测结果
        detected_peaks = []
        for i, peak_info in enumerate(peaks_info["peaks"]):
            if i < len(roi_frames):  # 确保索引有效
                roi_frame = roi_frames[i]
                detected_peaks.append({
                    "index": i,
                    "gray_value": roi_frame.gray_value,
                    "main_frame": roi_frame.frame_count,
                    "roi_frame": roi_frame.index,
                    "type": peak_info.get("type", "unknown"),
                    "confidence": peak_info.get("confidence", 0.0),
                    "score": peak_info.get("score", 0.0),
                    "threshold": threshold,
                    "in_peak_region": peak_info.get("in_peak_region", False),
                    "frame_count": roi_frame.frame_count
                })

        # 生成波峰检测摘要
        detection_summary = {
            "total_peaks": len(detected_peaks),
            "green_peaks": len([p for p in detected_peaks if p["type"] == "green"]),
            "red_peaks": len([p for p in detected_peaks if p["type"] == "red"]),
            "peak_indices": [p["index"] for p in detected_peaks],
            "average_confidence": sum(p["confidence"] for p in detected_peaks) / len(detected_peaks) if detected_peaks else 0.0,
            "detection_rate": len(detected_peaks) / len(gray_values) if gray_values else 0.0
        }

        peak_detection_results = {
            "peaks": detected_peaks,
            "detection_summary": detection_summary,
            "raw_analysis": peaks_info
        }

        logger.info("🎯 Peak detection completed: %d peaks detected (%d green, %d red)",
                   detection_summary["total_peaks"],
                   detection_summary["green_peaks"],
                   detection_summary["red_peaks"])

    except Exception as e:
        logger.error("❌ Peak detection failed: %s", str(e))
        peak_detection_results = {"error": str(e)}
        peak_detection_params["error"] = True

    # 6. 构建响应元数据 (与roi_window_capture相同)
    roi_count, roi_buffer_size, last_gray_value, last_main_frame_count = data_store.get_roi_status_snapshot()
    roi_start_frame = max(0, roi_count - len(roi_frames))
    roi_end_frame = roi_count - 1

    roi_config = {}
    main_frame_start = None
    main_frame_end = None

    if roi_frames:
        last_roi_frame = roi_frames[-1]
        roi_config = {
            "x1": last_roi_frame.roi_config.x1,
            "y1": last_roi_frame.roi_config.y1,
            "x2": last_roi_frame.roi_config.x2,
            "y2": last_roi_frame.roi_config.y2,
            "width": last_roi_frame.roi_config.width,
            "height": last_roi_frame.roi_config.height
        }
        main_frame_start = min(roi_frame.frame_count for roi_frame in roi_frames)
        main_frame_end = max(roi_frame.frame_count for roi_frame in roi_frames)

    duration = sum(roi_frame.capture_duration for roi_frame in roi_frames)
    gray_values = [roi_frame.gray_value for roi_frame in roi_frames]
    gray_range = [min(gray_values), max(gray_values)] if gray_values else [0.0, 0.0]

    capture_metadata = {
        "capture_duration": duration,
        "roi_frame_start": roi_start_frame,
        "roi_frame_end": roi_end_frame,
        "main_frame_start": main_frame_start,
        "main_frame_end": main_frame_end,
        "gray_range": gray_range,
        "last_gray_value": last_gray_value,
        "roi_buffer_size": roi_buffer_size
    }

    actual_fps, available_frames = data_store.get_roi_frame_rate_info()
    capture_metadata["actual_roi_fps"] = actual_fps
    capture_metadata["available_roi_frames"] = available_frames

    logger.info("✅ ROI window capture with peaks successful: frames=%d, roi_range=(%d,%d), main_range=(%d,%d), duration=%.3fs",
               len(series), roi_start_frame, roi_end_frame,
               capture_metadata["main_frame_start"], capture_metadata["main_frame_end"],
               capture_metadata["capture_duration"])

    return RoiWindowCaptureWithPeaksResponse(
        timestamp=datetime.utcnow(),
        window_size=count,
        roi_frame_range=(roi_start_frame, roi_end_frame),
        main_frame_range=(capture_metadata["main_frame_start"], capture_metadata["main_frame_end"]),
        series=series,
        roi_config=roi_config,
        capture_metadata=capture_metadata,
        peak_detection_results=peak_detection_results,
        peak_detection_params=peak_detection_params
    )
```

## 数据存储层实现

### DataStore 类的核心方法

#### 主信号数据获取
```python
class DataStore:
    def get_series(self, count: int) -> List[Frame]:
        """
        获取最近N帧主信号数据

        Args:
            count: 需要获取的帧数

        Returns:
            List[Frame]: 帧数据列表
        """
        with self._lock:
            frames = list(self._frames)

        if count >= len(frames):
            return frames
        return frames[-count:]
```

#### ROI数据获取
```python
def get_roi_series(self, count: int) -> List[RoiFrame]:
    """
    获取最近N帧ROI数据

    Args:
        count: 需要获取的ROI帧数

    Returns:
        List[RoiFrame]: ROI帧数据列表
    """
    with self._lock:
        roi_frames = list(self._roi_frames)

    if count >= len(roi_frames):
        return roi_frames
    return roi_frames[-count:]
```

### 数据结构定义

#### Frame 结构
```python
@dataclass
class Frame:
    """主信号数据帧"""
    index: int              # 帧索引
    timestamp: datetime     # 时间戳
    value: float           # 信号值
```

#### RoiFrame 结构
```python
@dataclass
class RoiFrame:
    """ROI截图帧数据"""
    index: int                     # ROI帧索引
    timestamp: datetime            # 时间戳
    gray_value: float              # ROI区域平均灰度值
    roi_config: RoiConfig          # ROI配置信息
    frame_count: int               # 对应的主信号帧计数
    capture_duration: float        # ROI截图持续时间
```

## 响应模型定义

### WindowCaptureResponse
```python
class WindowCaptureResponse(BaseModel):
    """窗口截取响应模型"""
    type: str = "window_capture"
    timestamp: datetime
    window_size: int                          # 请求的窗口大小
    frame_range: Tuple[int, int]             # 实际帧范围
    series: List[TimeSeriesPoint]            # 时间序列数据点
    capture_metadata: Dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    message: str = "Window data captured successfully"
```

### RoiWindowCaptureResponse
```python
class RoiWindowCaptureResponse(BaseModel):
    """ROI窗口截取响应模型"""
    type: str = "roi_window_capture"
    timestamp: datetime
    window_size: int                          # 请求的窗口大小
    roi_frame_range: Tuple[int, int]         # ROI帧范围
    main_frame_range: Tuple[int, int]        # 对应主信号帧范围
    series: List[RoiTimeSeriesPoint]         # ROI时间序列数据点
    roi_config: Dict[str, Any]               # ROI配置信息
    capture_metadata: Dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    message: str = "ROI window data captured successfully"
```

### RoiWindowCaptureWithPeaksResponse
```python
class RoiWindowCaptureWithPeaksResponse(BaseModel):
    """ROI窗口截取带波峰检测响应模型"""
    type: str = "roi_window_capture_with_peaks"
    timestamp: datetime
    window_size: int
    roi_frame_range: Tuple[int, int]
    main_frame_range: Tuple[int, int]
    series: List[RoiTimeSeriesPoint]
    roi_config: Dict[str, Any]
    capture_metadata: Dict[str, Any] = Field(default_factory=dict)

    # 波峰检测结果
    peak_detection_results: Dict[str, Any] = Field(default_factory=dict)
    # 波峰检测参数
    peak_detection_params: Dict[str, Any] = Field(default_factory=dict)

    success: bool = True
    message: str = "ROI window data captured with peak detection analysis"
```

## 波峰检测集成

### EnhancedPeakDetector 集成
```python
from ..core.enhanced_peak_detector import EnhancedPeakDetector

# 在roi_window_capture_with_peaks中使用
enhanced_detector = EnhancedPeakDetector()
peaks_info = enhanced_detector.detect_peaks_enhanced(
    data=gray_values,
    threshold=threshold,
    margin_frames=margin_frames,
    difference_threshold=difference_threshold
)
```

### 波峰检测结果格式化
```python
# 将原始检测结果转换为API响应格式
detected_peaks = []
for i, peak_info in enumerate(peaks_info["peaks"]):
    if i < len(roi_frames):
        roi_frame = roi_frames[i]
        detected_peaks.append({
            "index": i,
            "gray_value": roi_frame.gray_value,
            "main_frame": roi_frame.frame_count,
            "roi_frame": roi_frame.index,
            "type": peak_info.get("type", "unknown"),
            "confidence": peak_info.get("confidence", 0.0),
            "score": peak_info.get("score", 0.0),
            "threshold": threshold,
            "in_peak_region": peak_info.get("in_peak_region", False),
            "frame_count": roi_frame.frame_count
        })
```

## 错误处理机制

### 数据不可用错误
```python
frames = data_store.get_series(count)
if not frames:
    logger.warning("Window capture failed: no data available")
    raise HTTPException(status_code=404, detail="No data available for capture")
```

### ROI数据不可用错误
```python
roi_frames = data_store.get_roi_series(count)
if not roi_frames:
    logger.warning("ROI window capture failed: no ROI data available")
    raise HTTPException(status_code=404, detail="No ROI data available for capture")
```

### 波峰检测错误处理
```python
try:
    # 波峰检测逻辑
    peaks_info = enhanced_detector.detect_peaks_enhanced(...)
except Exception as e:
    logger.error("❌ Peak detection failed: %s", str(e))
    peak_detection_results = {"error": str(e)}
    peak_detection_params["error"] = True
```

## 性能优化策略

### 1. 内存管理
- 使用循环缓冲区限制内存使用
- 线程安全的数据访问
- 及时清理过期数据

### 2. 数据访问优化
- 使用锁机制保证线程安全
- 最小化锁持有时间
- 批量数据操作

### 3. 缓存机制
- ROI帧数据缓存
- 波峰检测结果缓存
- 配置信息缓存

## 日志记录

### 请求日志
```python
logger.info("🖼️ Window capture requested: count=%d", count)
logger.info("🖼️ ROI window capture requested: count=%d", count)
logger.info("🔍 ROI window capture with peak detection requested: count=%d, threshold=%.1f, margin=%d, diff=%.2f, force_refresh=%s",
            count, threshold, margin_frames, difference_threshold, force_refresh)
```

### 成功日志
```python
logger.info("✅ Window capture successful: frames=%d, range=(%d,%d), duration=%.3fs",
           frame_count, start_frame, end_frame, duration)
logger.info("✅ ROI window capture successful: frames=%d, roi_range=(%d,%d), main_range=(%d,%d), duration=%.3fs",
           len(series), roi_start_frame, roi_end_frame,
           capture_metadata["main_frame_start"], capture_metadata["main_frame_end"],
           capture_metadata["capture_duration"])
```

### 错误日志
```python
logger.warning("Window capture failed: no data available")
logger.warning("ROI window capture failed: no ROI data available")
logger.error("❌ Peak detection failed: %s", str(e))
```

## 配置参数

### 默认配置值
```python
# 从settings获取默认值
if threshold is None:
    threshold = settings.peak_threshold
if margin_frames is None:
    margin_frames = settings.peak_margin_frames
if difference_threshold is None:
    difference_threshold = settings.peak_difference_threshold
```

### 参数验证
```python
count: int = Query(100, ge=50, le=200, description="窗口大小：50-200帧")
threshold: Optional[float] = Query(None, ge=0.0, le=200.0, description="波峰检测阈值：0-200")
margin_frames: Optional[int] = Query(None, ge=1, le=20, description="边界扩展帧数：1-20")
difference_threshold: Optional[float] = Query(None, ge=0.1, le=10.0, description="帧差值阈值：0.1-10.0")
```

## 扩展性设计

### 1. 新数据源支持
- 可扩展的数据存储接口
- 统一的数据访问模式
- 配置化的数据源选择

### 2. 新分析算法支持
- 插件化的分析算法
- 可配置的算法参数
- 标准化的结果格式

### 3. 新导出格式支持
- 多种数据格式支持
- 可配置的元数据
- 标准化的响应结构