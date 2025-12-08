from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from dataclasses import dataclass
from typing import Deque
from collections import deque


class SystemStatus(str, Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


class LineDetectionConfig(BaseSettings):
    """
    ROI1绿色线条相交检测配置模型
    支持从JSON配置文件和环境变量(NHEM_LINE_DETECTION_*)加载
    """
    # 基础开关配置
    enabled: bool = Field(False, description="是否启用ROI1绿色线条相交检测")

    # HSV颜色空间绿色阈值配置
    hsv_green_lower: Tuple[int, int, int] = Field(
        (40, 50, 50),
        description="HSV绿色颜色下限阈值 (H, S, V)"
    )
    hsv_green_upper: Tuple[int, int, int] = Field(
        (80, 255, 255),
        description="HSV绿色颜色上限阈值 (H, S, V)"
    )

    # Canny边缘检测阈值配置
    canny_low_threshold: int = Field(
        25,
        ge=0, le=255,
        description="Canny边缘检测低阈值"
    )
    canny_high_threshold: int = Field(
        80,
        ge=0, le=255,
        description="Canny边缘检测高阈值"
    )

    # Hough直线变换参数配置
    hough_threshold: int = Field(
        50,
        ge=1,
        description="Hough直线变换投票阈值"
    )
    hough_min_line_length: int = Field(
        15,
        ge=1,
        description="检测直线最小长度(像素)"
    )
    hough_max_line_gap: int = Field(
        8,
        ge=0,
        description="检测直线最大间隙(像素)"
    )

    # 置信度相关配置
    min_confidence: float = Field(
        0.4,
        ge=0.0, le=1.0,
        description="最小置信度阈值"
    )

    # 处理模式配置
    roi_processing_mode: str = Field(
        "roi1_only",
        description="ROI处理模式: 'roi1_only' 仅处理ROI1"
    )

    # 性能优化配置
    cache_timeout_ms: int = Field(
        100,
        ge=0,
        description="结果缓存超时时间(毫秒)"
    )
    max_processing_time_ms: int = Field(
        300,
        ge=50,
        description="最大处理时间限制(毫秒)"
    )

    # 线条过滤配置
    min_angle_degrees: float = Field(
        10.0,
        ge=0.0, le=90.0,
        description="过滤水平线的最小角度(度)"
    )
    max_angle_degrees: float = Field(
        80.0,
        ge=0.0, le=90.0,
        description="过滤垂直线的最大角度(度)"
    )

    # 平行线检测配置
    parallel_threshold: float = Field(
        0.01,
        ge=0.0001, le=1.0,
        description="平行线检测的分母阈值"
    )

    class Config:
        env_prefix = "NHEM_LINE_DETECTION_"
        case_sensitive = False

    def validate_hsv_ranges(self) -> bool:
        """
        验证HSV颜色范围的有效性

        Returns:
            bool: 颜色范围是否有效
        """
        h1, s1, v1 = self.hsv_green_lower
        h2, s2, v2 = self.hsv_green_upper

        return (
            0 <= h1 <= 179 and 0 <= h2 <= 179 and  # OpenCV H范围: 0-179
            0 <= s1 <= 255 and 0 <= s2 <= 255 and  # S,V范围: 0-255
            0 <= v1 <= 255 and 0 <= v2 <= 255 and
            h1 < h2 and s1 < s2 and v1 < v2  # 下限小于上限
        )

    def validate_canny_thresholds(self) -> bool:
        """
        验证Canny阈值的有效性

        Returns:
            bool: Canny阈值是否有效
        """
        return 0 <= self.canny_low_threshold < self.canny_high_threshold <= 255

    def validate_hough_parameters(self) -> bool:
        """
        验证Hough参数的有效性

        Returns:
            bool: Hough参数是否有效
        """
        return (
            self.hough_threshold > 0 and
            self.hough_min_line_length > 0 and
            self.hough_max_line_gap >= 0 and
            self.hough_min_line_length > self.hough_max_line_gap
        )


class HealthResponse(BaseModel):
    status: str = "ok"
    system: str = "NHEM API Server"
    version: str = "1.0.0"


class StatusResponse(BaseModel):
    status: SystemStatus
    frame_count: int
    current_value: float
    peak_signal: Optional[int] = Field(
        None, description="1: HE peak, 0: non-HE peak, null: no peak"
    )
    buffer_size: int
    baseline: float
    timestamp: datetime


class TimeSeriesPoint(BaseModel):
    t: float
    value: float


class RoiData(BaseModel):
    width: int
    height: int
    pixels: str
    gray_value: float
    format: str = "base64"


class RealtimeDataResponse(BaseModel):
    type: str = "realtime_data"
    timestamp: datetime
    frame_count: int
    series: List[TimeSeriesPoint]
    roi_data: RoiData
    peak_signal: Optional[int]
    enhanced_peak: Optional[EnhancedPeakSignal] = None
    baseline: float


class DualRoiDataResponse(BaseModel):
    """双ROI数据响应模型"""
    roi1_data: RoiData  # 大ROI区域数据
    roi2_data: RoiData  # 50x50中心ROI数据
    roi1_config: RoiConfig  # ROI1配置
    roi2_config: RoiConfig  # ROI2配置（从ROI1中心计算）


class DualRealtimeDataResponse(BaseModel):
    """双ROI实时数据响应模型"""
    type: str = "dual_realtime_data"
    timestamp: datetime
    frame_count: int
    series: List[TimeSeriesPoint]
    dual_roi_data: DualRoiDataResponse
    peak_signal: Optional[int]
    enhanced_peak: Optional[EnhancedPeakSignal] = None
    baseline: float


class EnhancedRealtimeDataResponse(DualRealtimeDataResponse):
    """
    增强的双ROI实时数据响应模型
    扩展基础响应以包含ROI1线条相交检测结果
    """
    line_intersection: Optional[LineIntersectionResult] = Field(
        None,
        description="ROI1绿色线条相交检测结果，仅在启用检测时包含"
    )


class BaseSuccessResponse(BaseModel):
    type: str
    timestamp: datetime
    success: bool = True
    data: Dict[str, Any]


class ErrorDetails(BaseModel):
    parameter: Optional[str] = None
    value: Optional[Any] = None
    constraint: Optional[str] = None


class ErrorResponse(BaseModel):
    type: str = "error"
    timestamp: datetime
    error_code: str
    error_message: str
    details: Optional[ErrorDetails] = None


class PeakSignalResponse(BaseModel):
    type: str = "peak_signal"
    timestamp: datetime
    signal: Optional[int]
    has_peak: bool
    current_value: float
    frame_count: int


class ControlStatusResponse(BaseModel):
    type: str = "status"
    timestamp: datetime
    server_status: SystemStatus
    connected_clients: int
    last_peak_signal: Optional[int]


class ControlCommandStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"


class ControlCommandResponse(BaseModel):
    """
    控制类命令（start/stop/pause/resume）的响应结构。
    """

    type: str = "control_response"
    timestamp: datetime
    command: str
    status: ControlCommandStatus
    message: str


class AnalyzeEvent(BaseModel):
    t: float
    type: str
    score: float


class AnalyzeSeriesPoint(BaseModel):
    t: float
    value: float
    ref: float
    std: float
    high: float
    orange: float


class RoiConfig(BaseModel):
    """ROI配置模型"""
    x1: int = Field(0, ge=0, description="ROI左上角X坐标")
    y1: int = Field(0, ge=0, description="ROI左上角Y坐标")
    x2: int = Field(100, ge=0, description="ROI右下角X坐标")
    y2: int = Field(100, ge=0, description="ROI右下角Y坐标")

    @property
    def center_x(self) -> int:
        """ROI中心X坐标"""
        return (self.x1 + self.x2) // 2

    @property
    def center_y(self) -> int:
        """ROI中心Y坐标"""
        return (self.y1 + self.y2) // 2

    @property
    def width(self) -> int:
        """ROI宽度"""
        return abs(self.x2 - self.x1)

    @property
    def height(self) -> int:
        """ROI高度"""
        return abs(self.y2 - self.y1)

    def validate_coordinates(self) -> bool:
        """验证坐标有效性"""
        return self.x1 < self.x2 and self.y1 < self.y2 and self.width > 0 and self.height > 0


class RoiConfigResponse(BaseModel):
    """ROI配置响应模型"""
    type: str = "roi_config"
    timestamp: datetime
    config: RoiConfig
    success: bool = True


class RoiCaptureResponse(BaseModel):
    """ROI截图响应模型"""
    type: str = "roi_capture"
    timestamp: datetime
    success: bool = True
    roi_data: RoiData
    config: RoiConfig
    message: str = "ROI capture successful"


class RoiFrameRateResponse(BaseModel):
    """ROI帧率设置响应模型"""
    type: str = "roi_frame_rate"
    timestamp: datetime
    frame_rate: int
    success: bool = True
    message: str = "ROI frame rate updated successfully"


class DataFpsResponse(BaseModel):
    """数据生成频率设置响应模型"""
    type: str = "data_fps"
    timestamp: datetime
    fps: int
    success: bool = True
    message: str = "Data generation FPS updated successfully"


class PeakRegionData(BaseModel):
    """波峰区域数据模型"""
    start_frame: int
    end_frame: int
    peak_frame: int
    max_value: float
    color: str  # 'green' or 'red'
    confidence: float
    difference: float


class PeakDetectionConfigResponse(BaseModel):
    """波峰检测配置响应模型"""
    type: str = "peak_detection_config"
    timestamp: datetime
    threshold: float
    margin_frames: int
    difference_threshold: float
    min_region_length: int
    success: bool = True
    message: str = "Peak detection configuration retrieved successfully"


class EnhancedPeakSignal(BaseModel):
    """增强波峰信号模型"""
    signal: Optional[int]  # 1 for peak, None for no peak
    color: Optional[str]  # 'green' or 'red'
    confidence: float
    threshold: float
    in_peak_region: bool
    frame_count: int


class LineIntersectionResult(BaseModel):
    """
    ROI1绿色线条相交检测结果模型
    包含检测状态、相交点坐标、置信度等信息
    Task 31: Enhanced with medical-grade error handling fields
    """
    has_intersection: bool = Field(
        False,
        description="是否检测到有效的线条相交点"
    )
    intersection: Optional[Tuple[float, float]] = Field(
        None,
        description="相交点坐标 (x, y)，相对于ROI1左上角"
    )
    intersection_x: Optional[float] = Field(
        None,
        description="相交点X坐标（单独字段，便于向后兼容）"
    )
    intersection_y: Optional[float] = Field(
        None,
        description="相交点Y坐标（单独字段，便于向后兼容）"
    )
    confidence: float = Field(
        0.0,
        ge=0.0, le=1.0,
        description="检测置信度 (0.0-1.0)"
    )
    confidence_score: float = Field(
        0.0,
        ge=0.0, le=1.0,
        description="置信度评分（别名，保持向后兼容）"
    )
    line_count: int = Field(
        0,
        ge=0,
        description="检测到的线条数量"
    )
    detected_lines: List[Tuple[Tuple[int, int, int, int], float]] = Field(
        default_factory=list,
        description="检测到的线条列表，格式为 [((x1, y1, x2, y2), confidence)]"
    )
    processing_time_ms: float = Field(
        0.0,
        ge=0.0,
        description="处理耗时(毫秒)"
    )
    error_message: Optional[str] = Field(
        None,
        description="错误信息，仅在检测失败时设置"
    )
    error_details: Optional[Dict[str, Any]] = Field(
        None,
        description="Task 31: 详细错误信息，包含错误代码、分类和恢复信息"
    )
    edge_quality: float = Field(
        0.0,
        ge=0.0, le=1.0,
        description="边缘检测质量评分"
    )
    temporal_stability: float = Field(
        0.0,
        ge=0.0, le=1.0,
        description="时间稳定性评分"
    )
    frame_count: int = Field(
        0,
        ge=0,
        description="处理时的帧计数"
    )

    def __init__(self, **data):
        """Initialize with backward compatibility for coordinate fields"""
        super().__init__(**data)

        # Handle backward compatibility between intersection tuple and individual fields
        if self.intersection is not None and len(self.intersection) == 2:
            if self.intersection_x is None:
                self.intersection_x = self.intersection[0]
            if self.intersection_y is None:
                self.intersection_y = self.intersection[1]
        elif self.intersection_x is not None and self.intersection_y is not None:
            self.intersection = (self.intersection_x, self.intersection_y)

        # Handle backward compatibility for confidence fields
        if self.confidence_score == 0.0 and self.confidence > 0.0:
            self.confidence_score = self.confidence
        elif self.confidence == 0.0 and self.confidence_score > 0.0:
            self.confidence = self.confidence_score

        # Handle backward compatibility for line count
        if self.line_count == 0 and self.detected_lines:
            self.line_count = len(self.detected_lines)

    def is_high_confidence(self) -> bool:
        """
        判断是否为高置信度检测结果

        Returns:
            bool: 高置信度返回True (>0.7)
        """
        return self.confidence > 0.7

    def is_medium_confidence(self) -> bool:
        """
        判断是否为中等置信度检测结果

        Returns:
            bool: 中等置信度返回True (0.4-0.7)
        """
        return 0.4 <= self.confidence <= 0.7

    def get_confidence_level(self) -> str:
        """
        获取置信度等级描述

        Returns:
            str: 置信度等级 ('high', 'medium', 'low')
        """
        if self.is_high_confidence():
            return "high"
        elif self.is_medium_confidence():
            return "medium"
        else:
            return "low"

    class Config:
        """Pydantic配置类"""
        json_encoders = {
            # 处理numpy数组的JSON序列化
            # 如果将来需要支持numpy类型，可以在这里添加编码器
            # 例如：np.ndarray: lambda v: v.tolist()
        }
        schema_extra = {
            "example": {
                "has_intersection": True,
                "intersection": [75.5, 42.3],
                "confidence": 0.85,
                "detected_lines": [([10, 20, 100, 80], 0.9), ([30, 60, 120, 40], 0.8)],
                "processing_time_ms": 45.2,
                "error_message": None,
                "edge_quality": 0.92,
                "temporal_stability": 0.88,
                "frame_count": 1234
            }
        }


# Task 29: Performance monitoring dataclasses
@dataclass
class StageTimingMetrics:
    """处理阶段时间度量数据"""
    stage_name: str
    duration_ms: float
    timestamp: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """线条检测性能指标"""
    # 总体处理时间
    total_processing_time_ms: float = 0.0

    # 各阶段处理时间
    hsv_conversion_time_ms: float = 0.0
    morphological_operations_time_ms: float = 0.0
    canny_edge_detection_time_ms: float = 0.0
    hough_transform_time_ms: float = 0.0
    line_filtering_time_ms: float = 0.0
    intersection_calculation_time_ms: float = 0.0
    confidence_calculation_time_ms: float = 0.0

    # 内存使用情况
    memory_usage_mb: float = 0.0
    peak_memory_usage_mb: float = 0.0
    memory_efficiency_score: float = 0.0

    # 缓存效率
    cache_hit_count: int = 0
    cache_miss_count: int = 0
    cache_hit_rate: float = 0.0

    # 错误统计
    error_count: int = 0
    timeout_count: int = 0
    success_count: int = 0
    success_rate: float = 0.0

    # 算法效率指标
    detected_lines_count: int = 0
    filtered_lines_count: int = 0
    processing_fps: float = 0.0

    # 时间戳
    timestamp: float = 0.0
    frame_count: int = 0

    def calculate_total_processing_time(self) -> float:
        """计算总处理时间"""
        return (self.hsv_conversion_time_ms +
                self.morphological_operations_time_ms +
                self.canny_edge_detection_time_ms +
                self.hough_transform_time_ms +
                self.line_filtering_time_ms +
                self.intersection_calculation_time_ms +
                self.confidence_calculation_time_ms)


@dataclass
class PerformanceStats:
    """性能统计数据（滑动窗口统计）"""
    # 时间窗口统计
    window_size_10_avg: float = 0.0
    window_size_100_avg: float = 0.0
    window_size_1000_avg: float = 0.0

    # 百分位数统计
    percentile_50th: float = 0.0
    percentile_90th: float = 0.0
    percentile_95th: float = 0.0
    percentile_99th: float = 0.0

    # 性能趋势分析
    performance_trend: str = "stable"  # "improving", "degrading", "stable"
    trend_confidence: float = 0.0

    # 瓶颈识别
    bottleneck_stage: Optional[str] = None
    bottleneck_percentage: float = 0.0

    # 性能合规性
    medical_grade_compliance: bool = True
    processing_time_compliance: bool = True
    memory_usage_compliance: bool = True
    algorithm_efficiency_compliance: bool = True


@dataclass
class PerformanceAlert:
    """性能警报数据"""
    alert_type: str  # "timeout", "memory", "performance", "error_rate"
    severity: str    # "critical", "warning", "info"
    message: str
    current_value: float
    threshold_value: float
    timestamp: float
    recommendation: Optional[str] = None


class PerformanceMonitoringConfig(BaseModel):
    """性能监控配置"""
    # 时间阈值设置（毫秒）
    max_total_processing_time_ms: float = Field(300.0, description="最大总处理时间")
    max_hsv_conversion_time_ms: float = Field(50.0, description="HSV转换最大时间")
    max_edge_detection_time_ms: float = Field(100.0, description="边缘检测最大时间")
    max_hough_transform_time_ms: float = Field(100.0, description="Hough变换最大时间")
    max_intersection_calculation_time_ms: float = Field(50.0, description="交点计算最大时间")

    # 内存阈值设置（MB）
    max_memory_usage_mb: float = Field(50.0, description="最大内存使用量")
    cache_hit_rate_threshold: float = Field(0.8, description="缓存命中率阈值")

    # 性能监控设置
    enable_real_time_monitoring: bool = Field(True, description="启用实时监控")
    enable_performance_alerts: bool = Field(True, description="启用性能警报")
    enable_trend_analysis: bool = Field(True, description="启用趋势分析")

    # 统计窗口大小
    sliding_window_size_small: int = Field(10, description="小滑动窗口大小")
    sliding_window_size_medium: int = Field(100, description="中等滑动窗口大小")
    sliding_window_size_large: int = Field(1000, description="大滑动窗口大小")

    # 警报设置
    alert_cooldown_seconds: int = Field(60, description="警报冷却时间（秒）")
    max_alerts_per_hour: int = Field(10, description="每小时最大警报数")

    class Config:
        env_prefix = "NHEM_PERFORMANCE_"


class WindowCaptureResponse(BaseModel):
    """窗口截取响应模型"""
    type: str = "window_capture"
    timestamp: datetime
    window_size: int
    frame_range: Tuple[int, int]
    series: List[TimeSeriesPoint]
    capture_metadata: Dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    message: str = "Window data captured successfully"


class RoiTimeSeriesPoint(BaseModel):
    """ROI时间序列数据点"""
    t: float
    gray_value: float
    roi_index: int


class RoiWindowCaptureResponse(BaseModel):
    """ROI窗口截取响应模型"""
    type: str = "roi_window_capture"
    timestamp: datetime
    window_size: int
    roi_frame_range: Tuple[int, int]
    main_frame_range: Tuple[int, int]
    series: List[RoiTimeSeriesPoint]
    roi_config: Dict[str, Any]
    capture_metadata: Dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    message: str = "ROI window data captured successfully"


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


class AnalyzeResponse(BaseModel):
    has_hem: bool
    events: List[AnalyzeEvent]
    baseline: float
    series: List[AnalyzeSeriesPoint]
    realtime: bool
    peak_signal: Optional[int]
    enhanced_peak: Optional[EnhancedPeakSignal] = None
    peak_regions: List[PeakRegionData] = []
    frame_count: int


class ManualLineDetectionRequest(BaseModel):
    """
    手动线条相交检测请求模型

    支持两种输入模式：
    1. ROI坐标模式：提供ROI坐标，系统自动截图并检测
    2. 图像数据模式：直接提供base64编码的图像数据进行检测
    """
    # 认证信息
    password: str = Field(..., description="管理密码，用于身份验证")

    # 输入模式选择（互斥）
    roi_coordinates: Optional[RoiConfig] = Field(
        None,
        description="ROI坐标配置，与image_data互斥"
    )
    image_data: Optional[str] = Field(
        None,
        description="Base64编码的图像数据，与roi_coordinates互斥"
    )

    # 检测参数（可选，不提供则使用系统默认值）
    detection_params: Optional[LineDetectionConfig] = Field(
        None,
        description="自定义检测参数，不提供则使用系统配置"
    )

    # 处理选项
    force_refresh: bool = Field(
        False,
        description="强制刷新缓存，不使用历史结果"
    )
    include_debug_info: bool = Field(
        False,
        description="是否包含调试信息（检测到的线条详情等）"
    )


class ManualLineDetectionResponse(BaseModel):
    """
    手动线条相交检测响应模型
    """
    # 基础响应信息
    success: bool = Field(..., description="检测是否成功执行")
    timestamp: datetime = Field(..., description="检测时间戳")
    message: str = Field(..., description="响应消息")

    # 检测结果
    result: Optional[LineIntersectionResult] = Field(
        None,
        description="线条相交检测结果，仅在检测成功时设置"
    )

    # 处理信息
    processing_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="处理信息（输入模式、耗时、缓存状态等）"
    )

    # 调试信息
    debug_info: Optional[Dict[str, Any]] = Field(
        None,
        description="调试信息，仅在请求包含时返回"
    )

    # 错误信息
    error_details: Optional[ErrorDetails] = Field(
        None,
        description="错误详情，仅在检测失败时设置"
    )
