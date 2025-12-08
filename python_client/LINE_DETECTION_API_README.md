# Line Detection API Client Integration

本文档描述了线条检测API客户端集成功能，包括API客户端类、集成方法和使用示例。

## 概述

线条检测API客户端提供了与NHEM后端线条检测API的完整集成，支持：

- 启用/禁用线条检测
- 手动检测请求
- 配置管理
- 状态查询
- 增强实时数据获取
- 错误处理和重试机制

## 核心组件

### 1. LineDetectionAPIClient

主要的API客户端类，提供所有线条检测相关的API访问功能。

**主要特性：**
- 自动重试机制（最多3次）
- 请求超时控制
- 会话管理（连接池）
- 统计信息跟踪
- 错误处理和日志记录

**基本用法：**

```python
from line_detection_api_client import LineDetectionAPIClient

# 创建API客户端
client = LineDetectionAPIClient(
    base_url="http://localhost:8421",
    password="31415",
    timeout=10
)

try:
    # 启用线条检测
    result = client.enable_line_detection()
    if result.get('success', False):
        print("Line detection enabled successfully")

    # 执行手动检测
    detection_result = client.manual_detection(force_refresh=True)
    if detection_result.get('success', False):
        lines = detection_result.get('lines', [])
        intersections = detection_result.get('intersections', [])
        print(f"Found {len(lines)} lines and {len(intersections)} intersections")

    # 获取检测状态
    status = client.get_detection_status()
    print(f"Detection enabled: {status.get('enabled', False)}")

finally:
    # 清理资源
    client.close()
```

### 2. LineDetectionWidget集成

线条检测组件已集成API客户端功能，可通过配置启用：

```python
from line_detection_widget import LineDetectionWidget

# 创建组件配置
config = {
    'api_base_url': 'http://localhost:8421',
    'api_password': '31415',
    'api_timeout': 10,
    'enable_api_integration': True,  # 启用API集成
    'enable_control_panel': True,
    'enable_chinese_status': True
}

# 创建组件
widget = LineDetectionWidget(parent_frame, config=config)
```

## API方法详解

### 检测控制方法

#### enable_line_detection()
启用线条检测功能。

**返回：**
```python
{
    'success': True,
    'message': 'Line detection enabled successfully'
}
```

#### disable_line_detection()
禁用线条检测功能。

**返回：**
```python
{
    'success': True,
    'message': 'Line detection disabled successfully'
}
```

#### manual_detection(roi_coordinates=None, image_data=None, force_refresh=False)
执行手动线条检测。

**参数：**
- `roi_coordinates`: ROI坐标字典 {'x1': int, 'y1': int, 'x2': int, 'y2': int}
- `image_data`: Base64编码的图像数据
- `force_refresh`: 是否强制刷新缓存

**返回：**
```python
{
    'success': True,
    'lines': [
        {'start': [x1, y1], 'end': [x2, y2], 'confidence': 0.85},
        ...
    ],
    'intersections': [
        {'point': [x, y], 'confidence': 0.92},
        ...
    ],
    'processing_time_ms': 150,
    'detection_confidence': 0.89,
    'roi_info': {'width': 200, 'height': 150}
}
```

### 状态和配置方法

#### get_detection_status()
获取线条检测状态。

**返回：**
```python
{
    'enabled': True,
    'status': 'active',
    'last_detection_time': '2025-12-08T10:30:00Z',
    'detection_count': 25,
    'error_count': 1,
    'last_error': None,
    'config': {...}
}
```

#### get_line_detection_config()
获取当前检测配置。

#### update_line_detection_config(config_updates)
更新检测配置。

**配置参数：**
- `hsv_green_lower`: HSV绿色下限 [H, S, V]
- `hsv_green_upper`: HSV绿色上限 [H, S, V]
- `canny_low_threshold`: Canny边缘检测下限
- `canny_high_threshold`: Canny边缘检测上限
- `hough_threshold`: 霍夫变换阈值
- `min_confidence`: 最小置信度
- `roi_processing_mode`: ROI处理模式 ('roi1_only', 'dual_roi', 'auto')

### 数据获取方法

#### get_enhanced_realtime_data(count=100, include_line_intersection=True)
获取包含线条检测数据的增强实时数据。

**返回：**
```python
{
    'type': 'enhanced_realtime_data',
    'timestamp': '2025-12-08T10:30:00Z',
    'data_points': [...],
    'line_intersection_data': {
        'lines': [...],
        'intersections': [...],
        'last_update': '2025-12-08T10:30:00Z'
    },
    'roi_data': {...},
    'processing_info': {...}
}
```

#### get_current_roi_data(dual_roi=False)
获取当前ROI数据。

#### health_check()
执行健康检查。

## 错误处理

### LineDetectionAPIError

API客户端专用异常类，包含状态码和响应数据：

```python
try:
    result = client.enable_line_detection()
except LineDetectionAPIError as e:
    print(f"API Error: {e}")
    print(f"Status Code: {e.status_code}")
    print(f"Response Data: {e.response_data}")
```

### 常见错误类型

1. **连接错误**: 服务器不可达或网络问题
2. **超时错误**: 请求超过指定时间
3. **认证错误**: 密码错误或权限不足
4. **服务器错误**: 后端内部错误（5xx）
5. **客户端错误**: 请求参数错误（4xx）

### 重试机制

- 自动重试服务器错误（5xx）
- 最多重试3次
- 指数退避延迟
- 不重试客户端错误（4xx）

## 统计和监控

### 获取统计信息

```python
stats = client.get_statistics()
print(f"Total requests: {stats['request_count']}")
print(f"Error count: {stats['error_count']}")
print(f"Success rate: {stats['success_rate']:.2%}")
```

### 重置统计信息

```python
client.reset_statistics()
```

## 配置选项

### API客户端配置

```python
client = LineDetectionAPIClient(
    base_url="http://localhost:8421",    # 后端服务器URL
    password="31415",                    # 认证密码
    timeout=10                          # 请求超时时间（秒）
)
```

### 组件集成配置

```python
widget_config = {
    'api_base_url': 'http://localhost:8421',
    'api_password': '31415',
    'api_timeout': 10,
    'enable_api_integration': True,      # 启用API集成
    'enable_control_panel': True,        # 启用控制面板
    'enable_chinese_status': True,       # 启用中文状态显示
    'dark_theme': True,                  # 深色主题
    'enable_toolbar': True,              # 启用工具栏
    'figsize': (10, 6),                 # 图像尺寸
    'dpi': 100                          # 分辨率
}
```

## 使用示例

### 完整示例应用

运行完整示例应用：

```bash
python line_detection_api_example.py
```

该示例展示了：
- API配置管理
- 连接测试
- 检测控制
- 状态监控
- 错误处理

### 基本使用模式

```python
# 1. 配置API客户端
widget = LineDetectionWidget(parent, config={
    'enable_api_integration': True,
    'api_base_url': 'http://localhost:8421',
    'api_password': '31415'
})

# 2. 检查API集成状态
if widget.is_api_integration_available():
    print("API integration is available")

# 3. 控制检测
widget.set_detection_enabled(True)  # 启用检测

# 4. 手动检测
widget._on_manual_detection()  # 触发手动检测

# 5. 获取状态
status = widget.get_widget_api_status()
print(f"API Status: {status}")

# 6. 清理资源
widget.cleanup()
```

## 最佳实践

### 1. 错误处理

```python
try:
    result = client.manual_detection()
    if result.get('success', False):
        # 处理成功结果
        lines = result.get('lines', [])
        intersections = result.get('intersections', [])
    else:
        # 处理API返回的错误
        error_msg = result.get('error', 'Unknown error')
        logger.error(f"Detection failed: {error_msg}")
except LineDetectionAPIError as e:
    # 处理网络或服务器错误
    logger.error(f"API error: {e}")
except Exception as e:
    # 处理其他意外错误
    logger.error(f"Unexpected error: {e}")
```

### 2. 资源管理

```python
# 使用上下文管理器
with LineDetectionAPIClient(base_url, password) as client:
    result = client.manual_detection()
    # 自动清理资源

# 或者手动清理
client = LineDetectionAPIClient(base_url, password)
try:
    result = client.manual_detection()
finally:
    client.close()
```

### 3. 异步操作

```python
import threading

def perform_detection():
    try:
        result = client.manual_detection()
        # 在主线程中更新UI
        root.after(0, lambda: update_ui(result))
    except Exception as e:
        root.after(0, lambda: show_error(e))

# 在后台线程中执行
threading.Thread(target=perform_detection, daemon=True).start()
```

### 4. 配置验证

```python
def validate_config(config):
    required_keys = ['api_base_url', 'api_password']
    for key in required_keys:
        if not config.get(key):
            raise ValueError(f"Missing required config: {key}")

    # 验证URL格式
    if not config['api_base_url'].startswith(('http://', 'https://')):
        raise ValueError("Invalid API base URL format")

# 使用前验证配置
try:
    validate_config(widget_config)
    widget = LineDetectionWidget(parent, config=widget_config)
except ValueError as e:
    print(f"Configuration error: {e}")
```

## 故障排除

### 常见问题

1. **连接被拒绝**
   - 检查后端服务器是否运行
   - 验证URL和端口是否正确
   - 检查防火墙设置

2. **认证失败**
   - 验证密码是否正确
   - 检查密码中是否包含特殊字符

3. **超时错误**
   - 增加超时时间
   - 检查网络连接稳定性
   - 验证服务器响应性能

4. **检测无结果**
   - 检查ROI配置是否正确
   - 验证图像数据质量
   - 调整检测参数

### 调试模式

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 启用详细日志记录
client = LineDetectionAPIClient(base_url, password)
# 现在会显示详细的请求和响应信息
```

## 更新日志

### v1.0.0
- 初始版本发布
- 完整的API客户端功能
- 线条检测组件集成
- 错误处理和重试机制
- 统计和监控功能
- 示例应用和文档