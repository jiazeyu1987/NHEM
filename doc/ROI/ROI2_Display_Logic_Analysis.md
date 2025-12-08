# ROI2 显示逻辑分析报告

## 概述

ROI2 是 NHEM 系统中的第二级区域感兴趣(Region of Interest)系统，作为 ROI1 的智能子区域存在。ROI2 从 ROI1 的原始图像中实时提取，具有自适应尺寸调整、基于绿线交点定位和智能边界约束等高级功能。

## 核心架构

### 1. 双 ROI 系统架构

```
屏幕坐标系统
┌─────────────────────────────────────┐
│           ROI1 (大区域)              │
│  ┌───────────────────────────────┐  │
│  │          ROI2 (子区域)        │  │
│  │  ┌─────┐                     │  │
│  │  │绿线 │                     │  │
│  │  │交点 │←ROI2中心             │  │
│  │  └─────┘                     │  │
│  │                               │  │
│  └───────────────────────────────┘  │
│                                   │
└─────────────────────────────────────┘
```

### 2. 主要组件关系

- **RoiCaptureService**: 核心服务类，管理 ROI1 和 ROI2 的捕获逻辑
- **Roi2Config**: ROI2 配置管理，包含自适应模式和约束参数
- **Roi2RegionInfo**: ROI2 区域信息，包含坐标、尺寸和来源信息
- **LineIntersectionPoint**: 绿线交点信息，作为 ROI2 定位的基准

## ROI2 配置系统

### 1. 配置层次结构

```json
{
  "roi_capture": {
    "roi2_config": {
      "enabled": true,
      "default_width": 50,
      "default_height": 50,
      "dynamic_sizing": true,
      "adaptive_mode": "extension_based",
      "extension_params": {
        "left": 20,
        "right": 30,
        "top": 15,
        "bottom": 35
      },
      "size_constraints": {
        "min_width": 25,
        "min_height": 25,
        "max_width": 150,
        "max_height": 150
      },
      "fallback_mode": "center_based"
    }
  }
}
```

### 2. 自适应模式详解

#### extension_based 模式（默认）
- 基于绿线交点的智能扩展模式
- 以交点为中心，向四个方向扩展指定像素数
- 扩展参数: left=20px, right=30px, top=15px, bottom=35px
- 最终尺寸: 50x50 (20+30, 15+35)

#### fixed 模式
- 使用固定尺寸和位置
- 始终使用 ROI1 中心作为 ROI2 中心
- 尺寸: default_width × default_height (50×50)

#### golden_ratio 模式
- 使用黄金比例 (1:1.618) 计算尺寸
- base_size = min(ROI1.width, ROI1.height) // 6
- width = int(base_size * 1.618), height = int(base_size)

### 3. 尺寸约束系统

```python
min_width: 25px, min_height: 25px     # 最小尺寸保证
max_width: 150px, max_height: 150px   # 最大尺寸限制
default_width: 50px, default_height: 50px  # 默认尺寸
```

## ROI2 坐标变换逻辑

### 1. 坐标系统定义

- **屏幕坐标**: 绝对屏幕坐标系统，原点在屏幕左上角
- **ROI1 坐标**: 相对于 ROI1 左上角的坐标系统
- **ROI2 坐标**: 相对于 ROI1 的坐标，用于图像截取

### 2. 坐标变换公式

```python
# ROI2 在 ROI1 内的坐标
roi2_x1 = center_x - extension_params.left
roi2_y1 = center_y - extension_params.top
roi2_x2 = center_x + extension_params.right
roi2_y2 = center_y + extension_params.bottom

# ROI2 在屏幕上的绝对坐标
screen_x1 = roi1_config.x1 + roi2_x1
screen_y1 = roi1_config.y1 + roi2_y1
screen_x2 = roi1_config.x1 + roi2_x2
screen_y2 = roi1_config.y1 + roi2_y2
```

### 3. 中心点选择优先级

```python
1. 实时绿线交点 (intersection_point.roi_x, intersection_point.roi_y)
2. 缓存的绿线交点 (cached_intersection_point.roi_x, cached_intersection_point.roi_y)
3. ROI1 中心点 (roi1.width // 2, roi1.height // 2)
```

## ROI2 显示流程

### 1. 完整处理流程

```python
def capture_dual_roi():
    # 步骤1: 捕获 ROI1 原始图像
    roi1_data, roi1_image = capture_roi_internal(roi1_config)

    # 步骤2: 检测绿线交点
    intersection_point = detect_green_intersection(roi1_image)

    # 步骤3: 计算 ROI2 区域
    roi2_region = calculate_adaptive_roi2_region(intersection_point, roi1_config)

    # 步骤4: 从 ROI1 图像中截取 ROI2
    roi2_image = roi1_image.crop((roi2_region.x1, roi2_region.y1,
                                roi2_region.x2, roi2_region.y2))

    # 步骤5: 生成 ROI2 数据
    roi2_data = process_roi2_image(roi2_image, roi2_region)

    return roi1_data, roi2_data
```

### 2. 智能边界约束

```python
def apply_boundary_constraints(x1, y1, x2, y2, roi1_config, source):
    # 1. 基础边界检查 - 确保不超出 ROI1 范围
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(roi1_config.width, x2)
    y2 = min(roi1_config.height, y2)

    # 2. 最小尺寸约束 - 智能扩展
    if current_width < min_width:
        # 优先向右/下扩展，空间不足时向左/上扩展

    # 3. 最大尺寸约束 - 从中心收缩
    if current_width > max_width:
        # 保持中心点，向内收缩

    # 4. 最终验证 - 确保有效性
    if x1 >= x2 or y1 >= y2:
        # 使用安全的默认区域
```

### 3. 缓存和性能优化

#### 缓存机制
- **ROI1 缓存**: 0.5秒更新间隔，避免频繁截图
- **ROI2 缓存**: 1.0秒更新间隔，减少闪烁
- **交点缓存**: 检测失败时使用最后有效的交点

#### 定时器模式
- 独立线程以 1-4 FPS 运行
- 线程安全的数据更新
- 与主数据处理解耦

## ROI2 图像处理

### 1. 图像提取和处理

```python
def extract_roi2_from_roi1(roi1_config, roi1_data, intersection_point, roi1_image):
    # 使用原始 ROI1 图像而非解码的 base64
    roi2_image = roi1_image.crop((roi2_x1, roi2_y1, roi2_x2, roi2_y2))

    # 计算平均灰度值
    gray_roi2 = roi2_image.convert('L')
    histogram = gray_roi2.histogram()
    average_gray = calculate_average_gray(histogram)

    # 调整到标准显示尺寸 (200x150)
    roi2_resized = roi2_image.resize((200, 150), Image.Resampling.LANCZOS)

    # 编码为 base64
    roi2_base64 = base64.b64encode(roi2_resized.tobytes()).decode('utf-8')

    return RoiData(...)
```

### 2. 图像质量监控

```python
# 检查 ROI2 图像质量
non_zero_pixels = sum(1 for p in gray_roi2.getdata() if p > 0)
if non_zero_pixels == 0:
    logger.warning("ROI2 image is completely black")
else:
    quality_ratio = (non_zero_pixels / total_pixels) * 100
    logger.debug(f"ROI2 quality: {quality_ratio:.2f}% non-zero pixels")
```

## 数据存储和历史管理

### 1. ROI2 数据存储

```python
# 将 ROI2 帧添加到历史数据
roi2_frame = data_store.add_roi_frame(
    gray_value=roi2_data.gray_value,
    roi_config=roi2_screen_config,  # 屏幕坐标配置
    frame_count=main_frame_count,
    capture_duration=cache_interval
)
```

### 2. 历史记录管理

```python
# ROI2 区域历史记录
self._roi2_region_history.append({
    'timestamp': time.time(),
    'region_info': roi2_region,
    'intersection_point': intersection_point
})

# 保持不超过 100 条记录
if len(self._roi2_region_history) > 100:
    self._roi2_region_history.pop(0)
```

## 错误处理和回退机制

### 1. 分层错误处理

```python
try:
    roi2_data = extract_roi2_from_roi1(...)
except ValueError as e:
    # ROI2 截取失败（超出边界）
    roi2_data = RoiData(
        width=50,
        height=50,
        pixels="roi2_extraction_failed",
        gray_value=roi1_data.gray_value,  # 使用 ROI1 灰度值
        format="text",
        intersection=None
    )
except Exception as e:
    # 其他错误，返回 None
    roi2_data = None
```

### 2. 回退策略

1. **交点检测失败**: 使用缓存的交点 → ROI1 中心
2. **ROI2 提取失败**: 使用 ROI1 灰度值和错误状态
3. **配置验证失败**: 使用默认 ROI2 配置
4. **图像质量问题**: 记录警告但继续处理

## 性能特性

### 1. 时间性能

- **ROI1 截图**: ~50-100ms
- **绿线检测**: ~100-200ms (可优化)
- **ROI2 提取**: ~10-20ms
- **总处理时间**: ~200-300ms

### 2. 内存性能

- **图像缓存**: ROI1 + ROI2 原始图像
- **历史记录**: 100 条 ROI2 区域记录
- **配置缓存**: ROI2 配置对象

### 3. 优化策略

- **降采样**: ROI1 尺寸 > 1500px 时智能缩放
- **定时器模式**: 减少重复计算
- **缓存机制**: 避免不必要的截图操作
- **线程安全**: 使用锁保护共享数据

## API 集成

### 1. 核心 API 端点

```python
# 双 ROI 数据获取
@router.get("/dual-roi")
async def get_dual_roi_data():
    roi1_data, roi2_data = roi_capture_service.capture_dual_roi(roi_config)
    return DualRoiDataResponse(roi1_data=roi1_data, roi2_data=roi2_data)

# ROI2 配置管理
@router.get("/roi2/config")
async def get_roi2_config():
    return Roi2ConfigResponse(config=roi_capture_service.get_roi2_config())

@router.post("/roi2/config")
async def update_roi2_config(config: Roi2Config):
    success = roi_capture_service.update_roi2_config(config)
    return Roi2ConfigResponse(config=config, success=success)
```

### 2. 实时数据流

- **WebSocket 推送**: ROI2 数据实时推送到前端
- **定时更新**: 1-4 FPS 的更新频率
- **状态同步**: ROI2 配置变更的实时同步

## 总结

ROI2 系统是 NHEM 的核心创新功能之一，提供了:

1. **智能定位**: 基于绿线交点的精确定位
2. **自适应调整**: 多种自适应模式满足不同需求
3. **性能优化**: 缓存和定时器机制保证流畅性
4. **错误恢复**: 完善的回退机制确保系统稳定性
5. **可配置性**: 丰富的配置选项支持灵活定制

ROI2 的显示逻辑体现了现代计算机视觉技术与实时信号处理的完美结合，为 HEM 检测提供了更高精度的分析能力。

---

*文档生成时间: 2024-12-08*
*代码版本: NHEM backend before_green_interaction*
*分析文件: app/core/roi_capture.py, app/models.py*