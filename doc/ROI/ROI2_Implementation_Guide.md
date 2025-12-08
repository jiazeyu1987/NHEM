# ROI2 实现指南

## 概述

本指南面向开发人员，详细介绍如何在 NHEM 系统中实现、调试和扩展 ROI2 功能。包含代码实现细节、调试技巧和扩展示例。

## 核心类和接口

### 1. RoiCaptureService 类

ROI2 的核心服务类，负责所有 ROI2 相关的操作。

```python
class RoiCaptureService:
    """ROI截图服务类"""

    def __init__(self):
        # ROI2 配置
        self._roi2_config: Roi2Config = self._load_roi2_config()

        # 缓存机制
        self._cached_roi2_data: Optional[RoiData] = None
        self._roi2_cache_duration = 1.0  # ROI2 缓存时间

        # 定时器模式
        self._roi_timer_thread: Optional[threading.Thread] = None
        self._use_timer_mode = True

        # 历史记录
        self._roi2_region_history: list = []
```

### 2. 核心方法详解

#### capture_dual_roi()
双 ROI 捕获的主入口方法。

```python
def capture_dual_roi(self, roi_config: RoiConfig) -> Tuple[Optional[RoiData], Optional[RoiData]]:
    """
    截取双 ROI 区域
    Args:
        roi_config: ROI1 配置
    Returns:
        Tuple[Optional[RoiData], Optional[RoiData]]: (ROI1数据, ROI2数据)
    """
```

**实现逻辑**:
1. 检查是否使用定时器模式
2. 验证 ROI1 配置和尺寸
3. 检查缓存有效性
4. 执行双 ROI 捕获或从缓存获取
5. 更新缓存和历史记录

#### _extract_roi2_from_roi1()
从 ROI1 原始图像中提取 ROI2 的核心方法。

```python
def _extract_roi2_from_roi1(self, roi1_config: RoiConfig, roi1_data: RoiData,
                            intersection_point: Optional[LineIntersectionPoint] = None,
                            roi1_original_image: Optional[Image.Image] = None) -> Optional[RoiData]:
```

**关键实现细节**:
- 使用原始图像而非 base64 解码，提高性能
- 智能区域计算和边界约束
- 图像质量监控和错误处理
- 标准化输出尺寸 (200x150)

#### _calculate_adaptive_roi2_region()
ROI2 自适应区域计算的核心算法。

```python
def _calculate_adaptive_roi2_region(self, intersection_point: Optional[LineIntersectionPoint],
                                   roi1_config: RoiConfig) -> Roi2RegionInfo:
```

**算法流程**:
1. 检查 ROI2 是否启用
2. 根据自适应模式选择计算策略
3. 确定中心点（交点 → 缓存交点 → ROI1 中心）
4. 应用扩展参数或固定尺寸
5. 智能边界约束和尺寸调整
6. 计算屏幕坐标和返回区域信息

## 坐标系统详解

### 1. 坐标系统定义

```python
# 屏幕坐标系统 (原点在屏幕左上角)
screen_x, screen_y

# ROI1 坐标系统 (原点在 ROI1 左上角)
roi1_x, roi1_y

# ROI2 坐标系统 (相对于 ROI1)
roi2_x1, roi2_y1, roi2_x2, roi2_y2
```

### 2. 坐标变换实现

```python
def transform_roi2_to_screen_coordinates(roi1_config, roi2_region):
    """将 ROI2 坐标转换为屏幕坐标"""
    screen_x1 = roi1_config.x1 + roi2_region.x1
    screen_y1 = roi1_config.y1 + roi2_region.y1
    screen_x2 = roi1_config.x1 + roi2_region.x2
    screen_y2 = roi1_config.y1 + roi2_region.y2
    return screen_x1, screen_y1, screen_x2, screen_y2
```

### 3. 边界约束算法

```python
def _apply_boundary_constraints(self, x1: int, y1: int, x2: int, y2: int,
                               roi1_config: RoiConfig, source: str) -> Tuple[int, int, int, int]:
    """
    应用智能边界约束
    1. 基础边界检查
    2. 最小尺寸约束 - 智能扩展
    3. 最大尺寸约束 - 从中心收缩
    4. 最终验证和安全回退
    """
    # 第1步: 基础边界检查
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(roi1_config.width, x2)
    y2 = min(roi1_config.height, y2)

    current_width = x2 - x1
    current_height = y2 - y1

    # 第2步: 最小尺寸约束
    if current_width < self._roi2_config.size_constraints.min_width:
        needed_width = self._roi2_config.size_constraints.min_width
        # 智能扩展逻辑
        available_right = roi1_config.width - x2
        available_left = x1

        if available_right >= needed_width - current_width:
            x2 = x2 + (needed_width - current_width)
        elif available_left + available_right >= needed_width - current_width:
            expand_left = min(available_left, (needed_width - current_width) // 2)
            x1 = x1 - expand_left
            x2 = x2 + (needed_width - current_width - expand_left)
        else:
            x1 = 0
            x2 = min(roi1_config.width, needed_width)

    # 第3步: 最大尺寸约束 (类似处理)
    # 第4步: 最终验证

    return x1, y1, x2, y2
```

## 缓存机制实现

### 1. ROI2 缓存策略

```python
def _is_roi2_cache_valid(self, roi_config: RoiConfig, current_time: float) -> bool:
    """检查 ROI2 缓存是否有效"""
    if not self._roi2_cache_valid or self._cached_roi2_data is None:
        return False

    # 时间有效性检查
    time_valid = (current_time - self._last_roi2_capture_time) < self._roi2_cache_duration
    if not time_valid:
        return False

    # 配置变化检查
    config_unchanged = (self._last_roi2_config is not None and
                       self._roi_config_changed(roi_config, self._last_roi2_config) == False)

    return config_unchanged
```

### 2. 定时器模式实现

```python
def _roi_timer_loop(self):
    """ROI 定时器循环"""
    interval = 1.0 / self._frame_rate if self._frame_rate > 0 else 1.0

    while not self._stop_timer_event.is_set():
        start_time = time.perf_counter()

        try:
            # 执行双 ROI 捕获
            roi1_data, roi2_data = self._capture_dual_roi_timer()

            # 线程安全地更新最新数据
            with self._roi_lock:
                self._latest_roi1_data = roi1_data
                self._latest_roi2_data = roi2_data

        except Exception as e:
            self._logger.error("ROI timer loop error: %s", str(e))

        # 精确的频率控制
        elapsed = time.perf_counter() - start_time
        sleep_time = max(0, interval - elapsed)

        if self._stop_timer_event.wait(sleep_time):
            break
```

## 图像处理流程

### 1. ROI2 图像提取

```python
def _extract_roi2_from_roi1(self, roi1_config, roi1_data, intersection_point, roi1_original_image):
    """ROI2 图像提取的完整流程"""

    # 1. 获取 ROI2 区域信息
    roi2_region = self._calculate_adaptive_roi2_region(intersection_point, roi1_config)

    # 2. 图像截取 (使用原始图像)
    roi2_image = roi1_original_image.crop((
        roi2_region.x1, roi2_region.y1,
        roi2_region.x2, roi2_region.y2
    ))

    # 3. 图像质量检查
    gray_roi2 = roi2_image.convert('L')
    non_zero_pixels = sum(1 for p in gray_roi2.getdata() if p > 0)
    total_pixels = roi2_region.width * roi2_region.height

    if non_zero_pixels == 0:
        self._logger.warning(f"ROI2 image is completely black: {total_pixels} pixels")

    # 4. 灰度值计算
    histogram = gray_roi2.histogram()
    total_sum = sum(i * count for i, count in enumerate(histogram))
    average_gray = float(total_sum / total_pixels) if total_pixels > 0 else 0.0

    # 5. 尺寸标准化 (用于显示)
    roi2_resized = roi2_image.resize((200, 150), Image.Resampling.LANCZOS)

    # 6. Base64 编码
    buffer = io.BytesIO()
    roi2_resized.save(buffer, format='PNG')
    roi2_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # 7. 构造返回数据
    roi2_data = RoiData(
        width=roi2_region.width,
        height=roi2_region.height,
        pixels=f"data:image/png;base64,{roi2_base64}",
        gray_value=average_gray,
        format="base64",
        intersection=intersection_point
    )

    return roi2_data
```

### 2. 绿线交点集成

```python
def _capture_roi_internal(self, roi_config: RoiConfig):
    """ROI 捕获内部实现，集成绿线检测"""

    # ... ROI1 截图逻辑 ...

    # 绿线交点检测
    intersection_point = None
    try:
        # 性能优化: 智能降采样
        roi_width, roi_height = roi_image.size
        max_detection_size = 1500

        detection_image = roi_image
        scale_factor = 1.0

        if roi_width > max_detection_size or roi_height > max_detection_size:
            scale_factor = min(max_detection_size / roi_width, max_detection_size / roi_height)
            new_width = int(roi_width * scale_factor)
            new_height = int(roi_height * scale_factor)
            detection_image = roi_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # OpenCV 格式转换和检测
        roi_cv_image = cv2.cvtColor(np.array(detection_image), cv2.COLOR_RGB2BGR)
        intersection = detect_green_intersection(roi_cv_image)

        if intersection is not None:
            roi_x, roi_y = intersection

            # 坐标缩放回原始尺寸
            if scale_factor != 1.0:
                roi_x = int(roi_x / scale_factor)
                roi_y = int(roi_y / scale_factor)

            # 转换为屏幕坐标
            screen_x = roi_config.x1 + roi_x
            screen_y = roi_config.y1 + roi_y

            intersection_point = LineIntersectionPoint(
                x=screen_x, y=screen_y,
                roi_x=roi_x, roi_y=roi_y,
                confidence=1.0
            )

            # 更新交点缓存
            self._update_intersection_cache(intersection_point)

    except Exception as e:
        self._logger.error(f"Green line detection failed: {str(e)}")
        # 检测失败时保持现有缓存

    return roi_data, roi_image
```

## 配置管理实现

### 1. 配置加载

```python
def _load_roi2_config(self) -> Roi2Config:
    """从配置文件加载 ROI2 配置"""
    try:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'fem_config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)

        roi2_config_data = config_data.get('roi_capture', {}).get('roi2_config', {})

        if not roi2_config_data:
            self._logger.warning("ROI2 config not found, using default values")
            return Roi2Config()

        # 创建扩展参数对象
        extension_data = roi2_config_data.get('extension_params', {})
        extension_params = Roi2ExtensionParams(**extension_data)

        # 创建尺寸约束对象
        constraints_data = roi2_config_data.get('size_constraints', {})
        size_constraints = Roi2SizeConstraints(**constraints_data)

        # 创建 ROI2 配置对象
        roi2_config = Roi2Config(
            enabled=roi2_config_data.get('enabled', True),
            default_width=roi2_config_data.get('default_width', 50),
            default_height=roi2_config_data.get('default_height', 50),
            dynamic_sizing=roi2_config_data.get('dynamic_sizing', True),
            adaptive_mode=roi2_config_data.get('adaptive_mode', 'extension_based'),
            extension_params=extension_params,
            size_constraints=size_constraints,
            fallback_mode=roi2_config_data.get('fallback_mode', 'center_based')
        )

        if roi2_config.validate_config():
            self._logger.info(f"ROI2 config loaded successfully: {roi2_config}")
            return roi2_config
        else:
            self._logger.error("Invalid ROI2 config, using default values")
            return Roi2Config()

    except Exception as e:
        self._logger.error(f"Failed to load ROI2 config: {e}, using default values")
        return Roi2Config()
```

### 2. 配置更新

```python
def update_roi2_config(self, config: Roi2Config) -> bool:
    """更新 ROI2 配置"""
    try:
        if not config.validate_config():
            self._logger.error("Invalid ROI2 configuration")
            return False

        self._roi2_config = config

        # 清除 ROI2 缓存以应用新配置
        self._invalidate_roi2_cache()

        self._logger.info(f"ROI2 config updated: {config}")
        return True

    except Exception as e:
        self._logger.error(f"Failed to update ROI2 config: {e}")
        return False
```

## 错误处理和调试

### 1. 分层错误处理

```python
def _extract_roi2_from_roi1(self, ...):
    try:
        # 主要逻辑
        roi2_image = roi1_original_image.crop(...)

    except ValueError as e:
        # ROI2 截取失败（如超出边界）
        self._logger.error(f"ROI2 extraction failed: {e}")

        # 创建错误状态的 ROI2 数据
        roi2_data = RoiData(
            width=50,
            height=50,
            pixels="roi2_extraction_failed",
            gray_value=roi1_data.gray_value,
            format="text",
            intersection=None
        )
        return roi2_data

    except Exception as e:
        # 其他未预期错误
        self._logger.error(f"Unexpected error in ROI2 extraction: {e}")
        return None
```

### 2. 调试日志策略

```python
def _extract_roi2_from_roi1(self, ...):
    # ROI2 状态监控日志（避免刷屏）
    coord_changed = (hasattr(self, '_last_logged_roi2_center_x') and
                   (abs(self._last_logged_roi2_center_x - roi2_region.center_x) > 20 or
                    abs(self._last_logged_roi2_center_y - roi2_region.center_y) > 20))

    if coord_changed or not hasattr(self, '_last_logged_roi2_center_x'):
        self._logger.info(f"ROI2 Position Changed - ROI({roi2_region.center_x}, {roi2_region.center_y}) "
                        f"-> Screen({roi2_region.screen_x1 + roi2_region.width//2}, "
                        f"{roi2_region.screen_y1 + roi2_region.height//2}), "
                        f"Quality: {average_gray:.1f}, Source: {roi2_region.source}")
        self._last_logged_roi2_center_x = roi2_region.center_x
        self._last_logged_roi2_center_y = roi2_region.center_y
```

### 3. 性能监控

```python
def _capture_dual_roi_internal(self, roi_config: RoiConfig):
    dual_roi_start_time = time.time()

    # ... ROI1 捕获 ...

    roi2_extraction_start_time = time.time()
    try:
        roi2_data = self._extract_roi2_from_roi1(...)
        roi2_extraction_time = (time.time() - roi2_extraction_start_time) * 1000
        self._logger.debug(f"ROI2 extraction completed in {roi2_extraction_time:.2f}ms")

    except Exception as e:
        roi2_extraction_time = (time.time() - roi2_extraction_start_time) * 1000
        self._logger.error(f"ROI2 extraction failed after {roi2_extraction_time:.2f}ms: {e}")

    # 整体性能监控
    dual_roi_total_time = (time.time() - dual_roi_start_time) * 1000
    if dual_roi_total_time > 300:  # 性能警告阈值
        self._logger.warning(f"Dual ROI capture took {dual_roi_total_time:.2f}ms - may affect performance")
```

## API 接口实现

### 1. 双 ROI 数据接口

```python
@router.get("/dual-roi", response_model=DualRoiDataResponse)
async def get_dual_roi_data():
    """获取双 ROI 数据"""
    try:
        # 获取当前 ROI 配置
        roi_config = data_store.get_roi_config()

        if not roi_config or not roi_config.validate_coordinates():
            raise HTTPException(status_code=400, detail="Invalid ROI configuration")

        # 捕获双 ROI 数据
        roi1_data, roi2_data = roi_capture_service.capture_dual_roi(roi_config)

        if roi1_data is None:
            raise HTTPException(status_code=500, detail="Failed to capture ROI data")

        return DualRoiDataResponse(
            roi1_data=roi1_data,
            roi2_data=roi2_data,
            timestamp=datetime.now(),
            success=True,
            message="Dual ROI data captured successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error capturing dual ROI data: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
```

### 2. ROI2 配置接口

```python
@router.get("/roi2/config", response_model=Roi2ConfigResponse)
async def get_roi2_config():
    """获取 ROI2 配置"""
    try:
        roi2_config = roi_capture_service.get_roi2_config()
        return Roi2ConfigResponse(
            config=roi2_config,
            timestamp=datetime.now(),
            success=True
        )
    except Exception as e:
        logger.error(f"Error getting ROI2 config: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get ROI2 configuration")

@router.post("/roi2/config", response_model=Roi2ConfigResponse)
async def update_roi2_config(config: Roi2Config):
    """更新 ROI2 配置"""
    try:
        success = roi_capture_service.update_roi2_config(config)

        if success:
            return Roi2ConfigResponse(
                config=config,
                timestamp=datetime.now(),
                success=True,
                message="ROI2 configuration updated successfully"
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid ROI2 configuration")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating ROI2 config: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update ROI2 configuration")
```

## 扩展指南

### 1. 添加新的自适应模式

```python
def _calculate_adaptive_roi2_region(self, intersection_point, roi1_config):
    """扩展示例：添加新的自适应模式"""

    if self._roi2_config.adaptive_mode == "adaptive_circle":
        # 新的圆形自适应模式
        if intersection_point and intersection_point.roi_x is not None:
            center_x = intersection_point.roi_x
            center_y = intersection_point.roi_y
            source = "intersection"
        else:
            center_x = roi1_config.width // 2
            center_y = roi1_config.height // 2
            source = "center"

        # 基于目标大小计算圆形半径
        radius = min(roi1_config.width, roi1_config.height) // 8

        roi2_x1 = max(0, center_x - radius)
        roi2_y1 = max(0, center_y - radius)
        roi2_x2 = min(roi1_config.width, center_x + radius)
        roi2_y2 = min(roi1_config.height, center_y + radius)

    # ... 其他模式 ...
```

### 2. 添加新的图像处理功能

```python
def _extract_roi2_from_roi1(self, roi1_config, roi1_data, intersection_point, roi1_original_image):
    """扩展示例：添加图像增强功能"""

    # 原有的 ROI2 提取逻辑
    roi2_image = roi1_original_image.crop(...)

    # 新增：图像增强
    if self._roi2_config.enable_image_enhancement:
        # 对比度增强
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(roi2_image)
        roi2_image = enhancer.enhance(1.2)

        # 锐化
        enhancer = ImageEnhance.Sharpness(roi2_image)
        roi2_image = enhancer.enhance(1.1)

    # 新增：边缘检测（用于调试）
    if self._roi2_config.enable_edge_detection:
        import cv2
        roi2_cv = cv2.cvtColor(np.array(roi2_image), cv2.COLOR_RGB2BGR)
        edges = cv2.Canny(roi2_cv, 50, 150)
        # 可以保存边缘检测结果用于调试

    # 继续原有的处理流程...
```

### 3. 添加新的性能监控

```python
class RoiCaptureService:
    def __init__(self):
        # 新增：性能监控
        self._performance_metrics = {
            'roi2_extraction_times': [],
            'cache_hit_rates': {'roi1': 0, 'roi2': 0},
            'total_captures': 0,
            'failed_extractions': 0
        }

    def _extract_roi2_from_roi1(self, ...):
        extraction_start = time.time()

        try:
            roi2_data = self._extract_roi2_from_roi1_impl(...)

            # 记录成功指标
            extraction_time = time.time() - extraction_start
            self._performance_metrics['roi2_extraction_times'].append(extraction_time)
            self._performance_metrics['total_captures'] += 1

            # 保持最近 100 次记录
            if len(self._performance_metrics['roi2_extraction_times']) > 100:
                self._performance_metrics['roi2_extraction_times'].pop(0)

            return roi2_data

        except Exception as e:
            self._performance_metrics['failed_extractions'] += 1
            raise

    def get_performance_metrics(self):
        """获取性能指标"""
        times = self._performance_metrics['roi2_extraction_times']
        return {
            'average_extraction_time': sum(times) / len(times) if times else 0,
            'success_rate': 1 - (self._performance_metrics['failed_extractions'] /
                               max(1, self._performance_metrics['total_captures'])),
            'total_captures': self._performance_metrics['total_captures']
        }
```

## 测试策略

### 1. 单元测试示例

```python
import unittest
from unittest.mock import Mock, patch
from app.core.roi_capture import RoiCaptureService
from app.models import RoiConfig, Roi2Config

class TestROI2Extraction(unittest.TestCase):

    def setUp(self):
        self.roi_capture_service = RoiCaptureService()
        self.roi1_config = RoiConfig(x1=100, y1=100, x2=200, y2=200)
        self.roi1_data = Mock()
        self.roi1_data.intersection = Mock()
        self.roi1_data.intersection.roi_x = 50
        self.roi1_data.intersection.roi_y = 50

        # 模拟图像
        from PIL import Image
        self.roi1_image = Image.new('RGB', (100, 100), color='white')

    def test_extension_based_mode(self):
        """测试基于扩展的模式"""
        self.roi_capture_service._roi2_config.adaptive_mode = "extension_based"

        region = self.roi_capture_service._calculate_adaptive_roi2_region(
            self.roi1_data.intersection, self.roi1_config
        )

        # 验证区域计算
        self.assertEqual(region.x1, 30)  # 50 - 20
        self.assertEqual(region.x2, 80)  # 50 + 30
        self.assertEqual(region.y1, 35)  # 50 - 15
        self.assertEqual(region.y2, 85)  # 50 + 35
        self.assertEqual(region.source, "intersection")

    def test_boundary_constraints(self):
        """测试边界约束"""
        # 创建超出边界的 ROI
        invalid_region = Mock()
        invalid_region.x1 = -10
        invalid_region.y1 = -5
        invalid_region.x2 = 120
        invalid_region.y2 = 110

        x1, y1, x2, y2 = self.roi_capture_service._apply_boundary_constraints(
            -10, -5, 120, 110, self.roi1_config, "test"
        )

        # 验证边界约束生效
        self.assertEqual(x1, 0)
        self.assertEqual(y1, 0)
        self.assertEqual(x2, 100)
        self.assertEqual(y2, 100)
```

### 2. 集成测试示例

```python
class TestROI2Integration(unittest.TestCase):

    @patch('app.core.roi_capture.ImageGrab.grab')
    def test_dual_roi_capture_flow(self, mock_grab):
        """测试完整的双 ROI 捕获流程"""
        # 模拟屏幕截图
        from PIL import Image
        mock_screenshot = Image.new('RGB', (1920, 1080), color='white')
        mock_grab.return_value = mock_screenshot

        roi_capture_service = RoiCaptureService()
        roi_config = RoiConfig(x1=100, y1=100, x2=200, y2=200)

        # 执行双 ROI 捕获
        roi1_data, roi2_data = roi_capture_service.capture_dual_roi(roi_config)

        # 验证结果
        self.assertIsNotNone(roi1_data)
        self.assertIsNotNone(roi2_data)
        self.assertIsInstance(roi1_data.gray_value, float)
        self.assertIsInstance(roi2_data.gray_value, float)
```

---

*文档生成时间: 2024-12-08*
*代码版本: NHEM backend before_green_interaction*
*核心文件: app/core/roi_capture.py, app/models.py*