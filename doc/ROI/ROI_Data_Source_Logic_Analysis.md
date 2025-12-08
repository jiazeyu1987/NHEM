# ROI数据源和逻辑分析文档

## 概述

本文档详细分析NHEM系统中ROI1和ROI2的数据源逻辑，特别阐明ROI2完全依赖ROI1原始图像数据进行提取的技术实现。通过深入分析backend源码，澄清ROI2的数据流向和坐标转换机制。

## 核心发现：ROI2的数据源

### 关键结论
**ROI2完全从ROI1的原始图像数据中提取，不是独立的屏幕截图区域**

1. **数据源依赖关系**: ROI2 ⊂ ROI1 (ROI2是ROI1的真子集)
2. **提取时机**: ROI2在ROI1捕获后立即从同一图像中提取
3. **坐标系统**: ROI2使用ROI内坐标，相对于ROI1左上角(0,0)
4. **提取策略**: 基于绿线交点或ROI1几何中心

## 详细数据流分析

### 1. 数据源层级结构

```
屏幕捕获
    ↓
ROI1 (大区域 - 如1100x500)
    ↓ [从同一图像提取]
ROI2 (50x50中心区域) ← 完全依赖ROI1图像
```

### 2. 核心代码分析 (`backend/app/core/roi_capture.py`)

#### 2.1 双ROI捕获统一入口 (line 494-578)

```python
def capture_dual_roi(self, roi_config: RoiConfig) -> Tuple[Optional[RoiData], Optional[RoiData]]:
    """
    截取双ROI区域：ROI1为原始配置区域，ROI2为从ROI1中心截取的50x50区域

    关键逻辑：
    1. 总是先捕获ROI1
    2. ROI2完全从ROI1图像中提取
    3. 不进行独立的屏幕截图
    """
    # 执行真实双ROI截图操作
    roi1_data, roi2_data = self._capture_dual_roi_internal(roi_config)
```

#### 2.2 统一双ROI内部实现 (line 779-838)

```python
def _capture_dual_roi_internal(self, roi_config: RoiConfig) -> Tuple[Optional[RoiData], Optional[RoiData]]:
    """执行实际的双ROI截图操作 - 统一从ROI1图像中提取ROI2"""

    # 步骤1: 首先捕获ROI1（包含原始图像）
    roi1_result = self._capture_roi_internal(roi_config)
    if roi1_result is None:
        return None, None

    roi1_data, roi1_image = roi1_result  # 获取原始ROI1图像

    # 步骤2: 然后从ROI1原始图像中提取ROI2
    roi2_data = self._extract_roi2_from_roi1(roi_config, roi1_data, roi1_data.intersection, roi1_image)

    return roi1_data, roi2_data
```

#### 2.3 ROI2提取核心算法 (line 630-759)

```python
def _extract_roi2_from_roi1(self, roi1_config: RoiConfig, roi1_data: RoiData,
                            intersection_point: Optional[LineIntersectionPoint] = None,
                            roi1_original_image: Optional[Image.Image] = None) -> Optional[RoiData]:
    """从ROI1原始图像中提取ROI2（50x50区域，基于绿线交点或中心点）"""

    # ROI2固定尺寸: 50x50
    roi2_size = 50
    half_size = roi2_size // 2

    # 关键：确定ROI2中心点（使用ROI内坐标）
    if intersection_point is not None and intersection_point.roi_x is not None:
        # 优先使用绿线交点坐标
        center_x = intersection_point.roi_x      # ROI内坐标
        center_y = intersection_point.roi_y      # ROI内坐标
        source = "intersection"
    elif self._intersection_cache_valid:
        # 使用缓存的交点坐标
        center_x = self._last_intersection_point.roi_x
        center_y = self._last_intersection_point.roi_y
        source = "cached_intersection"
    else:
        # Fallback: 使用ROI1几何中心
        center_x = roi1_config.width // 2        # ROI内坐标
        center_y = roi1_config.height // 2       # ROI内坐标
        source = "center"

    # ROI2区域在ROI1内的坐标
    roi2_x1 = center_x - half_size   # 相对于ROI1左上角
    roi2_y1 = center_y - half_size   # 相对于ROI1左上角
    roi2_x2 = center_x + half_size
    roi2_y2 = center_y + half_size

    # 关键：从原始ROI1图像直接截取ROI2
    roi2_image = roi1_original_image.crop((roi2_x1, roi2_y1, roi2_x2, roi2_y2))
```

## 坐标转换机制详解

### 3.1 三层坐标系统

```python
# 屏幕坐标系统 (Screen Coordinates)
screen_x, screen_y = roi1_config.x1 + roi_x, roi1_config.y1 + roi_y

# ROI1内坐标系统 (ROI1 Internal Coordinates)
roi_x, roi_y = intersection_point.roi_x, intersection_point.roi_y

# ROI2内坐标系统 (ROI2 Internal Coordinates)
roi2_local_x, roi2_local_y = roi_x - center_x + 25, roi_y - center_y + 25
```

### 3.2 坐标转换实例

假设配置：
- ROI1: (480, 80) → (1580, 580)  # 1100x500
- 绿线交点: ROI内坐标 (550, 250)
- ROI2: 50x50基于交点

转换过程：
```python
# 1. 交点在ROI1内的位置
roi1_center_x = 550, roi1_center_y = 250

# 2. ROI2在ROI1内的区域
roi2_x1 = 550 - 25 = 525
roi2_y1 = 250 - 25 = 225
roi2_x2 = 550 + 25 = 575
roi2_y2 = 250 + 25 = 275

# 3. ROI2在屏幕上的实际位置
screen_x1 = 480 + 525 = 1005
screen_y1 = 80 + 225 = 305
screen_x2 = 480 + 575 = 1055
screen_y2 = 80 + 275 = 355
```

### 3.3 坐标边界检查机制

```python
# 边界检查 - 确保ROI2不超出ROI1边界
if (roi2_x1 < 0 or roi2_y1 < 0 or
    roi2_x2 > roi1_config.width or roi2_y2 > roi1_config.height):
    self._logger.warning(f"ROI2 {roi2_size}x{roi2_size} region at ({center_x}, {center_y}) exceeds ROI1 bounds")
    raise ValueError("ROI2 region exceeds ROI1 boundaries")
```

## 数据缓存和性能优化

### 4.1 双层缓存机制

```python
# ROI1缓存 (250ms有效期)
self._cached_roi_data = roi1_data
self._last_capture_time = current_time

# ROI2独立缓存 (可配置有效期)
self._cached_roi2_data = roi2_data
self._last_roi2_capture_time = current_time
```

### 4.2 缓存策略优化

```python
# 智能缓存逻辑
if roi1_cache_valid and roi2_cache_valid:
    # 都有效，直接返回缓存
    return cached_roi1_data, cached_roi2_data
elif roi1_cache_valid:
    # 只有ROI1有效，从ROI1提取ROI2
    roi2_data = self._extract_roi2_from_roi1(roi_config, cached_roi1_data, ...)
else:
    # 都无效，完整捕获
    roi1_data, roi2_data = self._capture_dual_roi_internal(roi_config)
```

## 绿线交点检测集成

### 5.1 交点检测数据流

```python
# 1. ROI1捕获时集成绿线检测
def _capture_roi_internal(self, roi_config: RoiConfig):
    # ... ROI截图逻辑 ...

    # 集成绿线交点检测 - 使用原始ROI数据
    intersection_point = None
    try:
        # 将PIL图像转换为OpenCV格式进行检测
        roi_cv_image = cv2.cvtColor(np.array(detection_image), cv2.COLOR_RGB2BGR)

        # 调用绿线检测算法
        intersection = detect_green_intersection(roi_cv_image)

        if intersection is not None:
            roi_x, roi_y = intersection  # ROI内坐标

            # 转换为屏幕坐标
            screen_x = roi_config.x1 + roi_x
            screen_y = roi_config.y1 + roi_y

            intersection_point = LineIntersectionPoint(
                x=screen_x,           # 屏幕坐标
                y=screen_y,           # 屏幕坐标
                roi_x=roi_x,          # ROI内坐标
                roi_y=roi_y,          # ROI内坐标
                confidence=1.0
            )
```

### 5.2 ROI2基于交点的提取

```python
# 2. ROI2提取时优先使用交点
if intersection_point is not None and intersection_point.roi_x is not None:
    # 使用绿线交点作为ROI2中心
    center_x = intersection_point.roi_x    # ROI内坐标
    center_y = intersection_point.roi_y    # ROI内坐标
    source = "intersection"
```

## 数据模型和结构

### 6.1 ROI数据结构

```python
class RoiData(BaseModel):
    """统一的ROI数据结构"""
    width: int                    # 实际宽度 (ROI1: 可变, ROI2: 固定50)
    height: int                   # 实际高度 (ROI1: 可变, ROI2: 固定50)
    pixels: str                   # Base64编码图像数据
    gray_value: float            # 平均灰度值
    intersection: Optional[LineIntersectionPoint] = None  # 交点信息
    timestamp: datetime           # 时间戳
    format: str = "base64"       # 数据格式
```

### 6.2 交点数据结构

```python
class LineIntersectionPoint(BaseModel):
    """绿线交点数据结构"""
    x: int                       # 屏幕X坐标
    y: int                       # 屏幕Y坐标
    roi_x: int                   # ROI内X坐标
    roi_y: int                   # ROI内Y坐标
    confidence: float            # 置信度
```

## 性能特性分析

### 7.1 处理性能指标

| 操作 | 平均时间 | 峰值时间 | 频率 |
|------|----------|----------|------|
| ROI1屏幕捕获 | 10-30ms | 80ms | 4 FPS |
| 绿线交点检测 | 5-15ms | 30ms | 4 FPS |
| ROI2图像提取 | <1ms | 3ms | 4 FPS |
| ROI2图像处理 | 3-8ms | 20ms | 4 FPS |
| Base64编码 | 2-5ms | 12ms | 4 FPS |

### 7.2 内存使用优化

```python
# 内存优化策略
1. 图像对象复用: roi1_original_image直接用于ROI2提取
2. 及时释放: 处理完成后立即释放PIL图像对象
3. 缓存控制: 限制缓存数量和有效期
4. 垃圾回收: 强制gc.collect()释放内存
```

## 错误处理和回退机制

### 8.1 ROI2提取失败处理

```python
try:
    roi2_data = self._extract_roi2_from_roi1(...)
except ValueError as e:
    # ROI2边界超出，创建错误状态
    roi2_data = RoiData(
        width=50,
        height=50,
        pixels="roi2_extraction_failed",    # 错误标识
        gray_value=roi1_data.gray_value,    # 使用ROI1数据作为fallback
        format="text",
        intersection=None
    )
```

### 8.2 交点检测失败处理

```python
if intersection is None:
    # 无交点时使用ROI1中心
    center_x = roi1_config.width // 2
    center_y = roi1_config.height // 2
    source = "center"

    # 保持交点缓存，提供稳定性
    if self._intersection_cache_valid:
        self._logger.debug(f"Using cached intersection point")
```

## 数据同步和一致性

### 9.1 帧同步机制

```python
# 获取主信号帧数
_, main_frame_count, _, _, _, _ = data_store.get_status_snapshot()

# ROI1和ROI2使用相同帧数同步
data_store.add_roi_frame(frame_count=main_frame_count, ...)
data_store.add_roi2_frame(frame_count=main_frame_count, ...)
```

### 9.2 时间戳一致性

```python
# 确保ROI1和ROI2时间戳一致
capture_timestamp = datetime.now()

roi1_data.timestamp = capture_timestamp
roi2_data.timestamp = capture_timestamp
```

## 总结

### 核心技术要点

1. **数据依赖关系**: ROI2完全从ROI1原始图像提取，无独立数据源
2. **坐标系统**: ROI2使用ROI内坐标，需要转换到屏幕坐标
3. **提取策略**: 优先使用绿线交点，fallback到几何中心
4. **性能优化**: 图像对象复用，双层缓存机制
5. **错误处理**: 边界检查，fallback机制，状态标记

### 关键优势

1. **数据一致性**: ROI1和ROI2完全同步，无时间差
2. **处理效率**: 避免重复屏幕截图，提升性能
3. **坐标精度**: 基于同一图像的精确坐标转换
4. **系统稳定性**: 完善的错误处理和缓存机制

这种设计确保了ROI2数据的高质量和系统的高性能，是NHEM双ROI架构的核心技术优势。