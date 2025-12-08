from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class SystemStatus(str, Enum):
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


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


class LineIntersectionPoint(BaseModel):
    """绿线交点坐标模型"""
    x: int = Field(description="屏幕X坐标")
    y: int = Field(description="屏幕Y坐标")
    roi_x: Optional[int] = Field(default=None, description="ROI内X坐标")
    roi_y: Optional[int] = Field(default=None, description="ROI内Y坐标")
    confidence: float = Field(default=1.0, description="检测置信度")


class RoiData(BaseModel):
    width: int
    height: int
    pixels: str
    gray_value: float
    format: str = "base64"
    intersection: Optional[LineIntersectionPoint] = Field(default=None, description="绿线交点坐标")


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


class Roi2ExtensionParams(BaseModel):
    """ROI2扩展参数模型"""
    left: int = Field(default=20, ge=0, description="交点向左扩展像素数")
    right: int = Field(default=30, ge=0, description="交点向右扩展像素数")
    top: int = Field(default=15, ge=0, description="交点向上扩展像素数")
    bottom: int = Field(default=35, ge=0, description="交点向下扩展像素数")


class Roi2SizeConstraints(BaseModel):
    """ROI2尺寸约束模型"""
    min_width: int = Field(default=25, ge=10, description="ROI2最小宽度")
    min_height: int = Field(default=25, ge=10, description="ROI2最小高度")
    max_width: int = Field(default=150, le=300, description="ROI2最大宽度")
    max_height: int = Field(default=150, le=300, description="ROI2最大高度")


class Roi2Config(BaseModel):
    """ROI2配置模型"""
    enabled: bool = Field(default=True, description="是否启用ROI2")
    default_width: int = Field(default=50, ge=10, le=200, description="ROI2默认宽度")
    default_height: int = Field(default=50, ge=10, le=200, description="ROI2默认高度")
    dynamic_sizing: bool = Field(default=True, description="是否启用动态尺寸调整")
    adaptive_mode: str = Field(default="extension_based", description="自适应模式: extension_based, fixed, golden_ratio")
    extension_params: Roi2ExtensionParams = Field(default_factory=Roi2ExtensionParams, description="扩展参数")
    size_constraints: Roi2SizeConstraints = Field(default_factory=Roi2SizeConstraints, description="尺寸约束")
    fallback_mode: str = Field(default="center_based", description="备用模式: center_based, fixed_size")

    @property
    def extension_total_width(self) -> int:
        """ROI2扩展后的总宽度"""
        return self.extension_params.left + self.extension_params.right

    @property
    def extension_total_height(self) -> int:
        """ROI2扩展后的总高度"""
        return self.extension_params.top + self.extension_params.bottom

    def validate_config(self) -> bool:
        """验证ROI2配置有效性"""
        # 检查尺寸约束
        if (self.size_constraints.min_width > self.size_constraints.max_width or
            self.size_constraints.min_height > self.size_constraints.max_height):
            return False

        # 检查默认尺寸是否在约束范围内
        if not (self.size_constraints.min_width <= self.default_width <= self.size_constraints.max_width):
            return False

        if not (self.size_constraints.min_height <= self.default_height <= self.size_constraints.max_height):
            return False

        # 检查自适应模式
        valid_modes = ["extension_based", "fixed", "golden_ratio"]
        if self.adaptive_mode not in valid_modes:
            return False

        return True


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


class Roi2ConfigResponse(BaseModel):
    """ROI2配置响应模型"""
    type: str = "roi2_config"
    timestamp: datetime
    config: Roi2Config
    success: bool = True
    message: str = "ROI2 configuration updated successfully"


class Roi2RegionInfo(BaseModel):
    """ROI2区域信息模型"""
    x1: int = Field(description="ROI2左上角X坐标(相对于ROI1)")
    y1: int = Field(description="ROI2左上角Y坐标(相对于ROI1)")
    x2: int = Field(description="ROI2右下角X坐标(相对于ROI1)")
    y2: int = Field(description="ROI2右下角Y坐标(相对于ROI1)")
    width: int = Field(description="ROI2实际宽度")
    height: int = Field(description="ROI2实际高度")
    center_x: int = Field(description="ROI2中心X坐标")
    center_y: int = Field(description="ROI2中心Y坐标")
    source: str = Field(description="ROI2区域来源: intersection, cached_intersection, center, extension_based")
    screen_x1: Optional[int] = Field(default=None, description="ROI2屏幕左上角X坐标")
    screen_y1: Optional[int] = Field(default=None, description="ROI2屏幕左上角Y坐标")
    screen_x2: Optional[int] = Field(default=None, description="ROI2屏幕右下角X坐标")
    screen_y2: Optional[int] = Field(default=None, description="ROI2屏幕右下角Y坐标")


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
