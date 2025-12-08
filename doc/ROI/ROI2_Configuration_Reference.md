# ROI2 配置参考手册

## 概述

ROI2 配置系统提供了丰富的参数选项，支持不同的自适应模式、尺寸约束和行为控制。本文档详细说明了所有配置参数的含义、默认值和使用方法。

## 配置文件结构

ROI2 配置存储在 `backend/app/fem_config.json` 文件的 `roi_capture.roi2_config` 节点下。

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

## 核心配置参数

### 1. 基础设置

#### enabled
- **类型**: `boolean`
- **默认值**: `true`
- **描述**: 是否启用 ROI2 功能
- **说明**: 当设置为 `false` 时，ROI2 不会进行捕获和处理

#### default_width
- **类型**: `integer`
- **默认值**: `50`
- **范围**: `10-200`
- **描述**: ROI2 的默认宽度（像素）
- **说明**: 在固定模式或作为基础尺寸时使用

#### default_height
- **类型**: `integer`
- **默认值**: `50`
- **范围**: `10-200`
- **描述**: ROI2 的默认高度（像素）
- **说明**: 在固定模式或作为基础尺寸时使用

#### dynamic_sizing
- **类型**: `boolean`
- **默认值**: `true`
- **描述**: 是否启用动态尺寸调整
- **说明**: 启用后 ROI2 会根据自适应模式和约束条件调整尺寸

### 2. 自适应模式 (adaptive_mode)

#### extension_based (推荐)
- **描述**: 基于绿线交点的智能扩展模式
- **行为**: 以绿线交点为中心，向四个方向扩展指定像素数
- **适用场景**: 需要精确跟随目标移动的场景
- **依赖参数**: `extension_params`

```python
# 计算公式
roi2_x1 = intersection_x - extension_params.left
roi2_y1 = intersection_y - extension_params.top
roi2_x2 = intersection_x + extension_params.right
roi2_y2 = intersection_y + extension_params.bottom
```

#### fixed
- **描述**: 固定尺寸和位置模式
- **行为**: 始终使用 ROI1 中心作为 ROI2 中心，尺寸固定
- **适用场景**: 目标位置相对固定的场景
- **依赖参数**: `default_width`, `default_height`

```python
# 计算公式
center_x = roi1.width // 2
center_y = roi1.height // 2
roi2_x1 = center_x - default_width // 2
roi2_y1 = center_y - default_height // 2
```

#### golden_ratio
- **描述**: 黄金比例模式
- **行为**: 使用黄金比例 (1:1.618) 计算最优尺寸
- **适用场景**: 对视觉美感有要求的场景
- **计算公式**:
```python
base_size = min(roi1.width, roi1.height) // 6
roi2_width = int(base_size * 1.618)
roi2_height = int(base_size)
```

### 3. 扩展参数 (extension_params)

#### left
- **类型**: `integer`
- **默认值**: `20`
- **范围**: `0-100`
- **描述**: 从中心点向左扩展的像素数
- **单位**: 像素

#### right
- **类型**: `integer`
- **默认值**: `30`
- **范围**: `0-100`
- **描述**: 从中心点向右扩展的像素数
- **单位**: 像素

#### top
- **类型**: `integer`
- **默认值**: `15`
- **范围**: `0-100`
- **描述**: 从中心点向上扩展的像素数
- **单位**: 像素

#### bottom
- **类型**: `integer`
- **默认值**: `35`
- **范围**: `0-100`
- **描述**: 从中心点向下扩展的像素数
- **单位**: 像素

**扩展参数示例**:
```json
{
  "extension_params": {
    "left": 20,   // 左侧扩展 20px
    "right": 30,  // 右侧扩展 30px
    "top": 15,    // 上方扩展 15px
    "bottom": 35  // 下方扩展 35px
  }
}
```

**结果尺寸**: 50×50 (20+30, 15+35)

### 4. 尺寸约束 (size_constraints)

#### min_width
- **类型**: `integer`
- **默认值**: `25`
- **范围**: `10-100`
- **描述**: ROI2 的最小宽度限制
- **说明**: 当计算结果小于此值时，会智能扩展到此尺寸

#### min_height
- **类型**: `integer`
- **默认值**: `25`
- **范围**: `10-100`
- **描述**: ROI2 的最小高度限制
- **说明**: 当计算结果小于此值时，会智能扩展到此尺寸

#### max_width
- **类型**: `integer`
- **默认值**: `150`
- **范围**: `50-300`
- **描述**: ROI2 的最大宽度限制
- **说明**: 当计算结果超过此值时，会从中心收缩到此尺寸

#### max_height
- **类型**: `integer`
- **默认值**: `150`
- **范围**: `50-300`
- **描述**: ROI2 的最大高度限制
- **说明**: 当计算结果超过此值时，会从中心收缩到此尺寸

### 5. 回退模式 (fallback_mode)

#### center_based
- **描述**: 基于 ROI1 中心的回退模式
- **行为**: 当绿线交点不可用时，使用 ROI1 的几何中心
- **适用场景**: 通用场景，推荐使用

#### fixed_size
- **描述**: 固定尺寸回退模式
- **行为**: 使用固定的 ROI2 尺寸和位置
- **适用场景**: 对位置稳定性要求极高的场景

## 配置验证规则

### 1. 基础验证
```python
# enabled 必须是布尔值
isinstance(enabled, bool)

# 尺寸必须在合理范围内
10 <= default_width <= 200
10 <= default_height <= 200
```

### 2. 扩展参数验证
```python
# 扩展参数必须非负
extension_params.left >= 0
extension_params.right >= 0
extension_params.top >= 0
extension_params.bottom >= 0
```

### 3. 约束参数验证
```python
# 最小值必须小于等于最大值
size_constraints.min_width <= size_constraints.max_width
size_constraints.min_height <= size_constraints.max_height

# 默认尺寸必须在约束范围内
size_constraints.min_width <= default_width <= size_constraints.max_width
size_constraints.min_height <= default_height <= size_constraints.max_height
```

### 4. 模式验证
```python
# 自适应模式必须是预定义值之一
adaptive_mode in ["extension_based", "fixed", "golden_ratio"]

# 回退模式必须是预定义值之一
fallback_mode in ["center_based", "fixed_size"]
```

## 配置使用示例

### 1. 高精度跟踪配置
适用于需要精确跟随移动目标的场景：

```json
{
  "roi2_config": {
    "enabled": true,
    "dynamic_sizing": true,
    "adaptive_mode": "extension_based",
    "extension_params": {
      "left": 15,
      "right": 15,
      "top": 15,
      "bottom": 15
    },
    "size_constraints": {
      "min_width": 20,
      "min_height": 20,
      "max_width": 60,
      "max_height": 60
    },
    "fallback_mode": "center_based"
  }
}
```

### 2. 稳定性优先配置
适用于对位置稳定性要求高的场景：

```json
{
  "roi2_config": {
    "enabled": true,
    "default_width": 80,
    "default_height": 80,
    "dynamic_sizing": false,
    "adaptive_mode": "fixed",
    "size_constraints": {
      "min_width": 60,
      "min_height": 60,
      "max_width": 100,
      "max_height": 100
    },
    "fallback_mode": "fixed_size"
  }
}
```

### 3. 大范围监控配置
适用于需要监控较大区域的场景：

```json
{
  "roi2_config": {
    "enabled": true,
    "dynamic_sizing": true,
    "adaptive_mode": "extension_based",
    "extension_params": {
      "left": 40,
      "right": 40,
      "top": 30,
      "bottom": 30
    },
    "size_constraints": {
      "min_width": 50,
      "min_height": 40,
      "max_width": 120,
      "max_height": 100
    },
    "fallback_mode": "center_based"
  }
}
```

### 4. 最小化资源占用配置
适用于资源受限的环境：

```json
{
  "roi2_config": {
    "enabled": true,
    "default_width": 30,
    "default_height": 30,
    "dynamic_sizing": false,
    "adaptive_mode": "fixed",
    "size_constraints": {
      "min_width": 25,
      "min_height": 25,
      "max_width": 35,
      "max_height": 35
    },
    "fallback_mode": "center_based"
  }
}
```

## API 配置接口

### 1. 获取当前配置
```http
GET /roi2/config
```

**响应示例**:
```json
{
  "type": "roi2_config",
  "timestamp": "2024-12-08T10:30:00Z",
  "config": {
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
  },
  "success": true
}
```

### 2. 更新配置
```http
POST /roi2/config
Content-Type: application/json
```

**请求体示例**:
```json
{
  "enabled": true,
  "adaptive_mode": "extension_based",
  "extension_params": {
    "left": 25,
    "right": 25,
    "top": 20,
    "bottom": 20
  }
}
```

**响应示例**:
```json
{
  "type": "roi2_config",
  "timestamp": "2024-12-08T10:30:00Z",
  "config": {
    // 更新后的完整配置
  },
  "success": true,
  "message": "ROI2 configuration updated successfully"
}
```

## 配置最佳实践

### 1. 性能优化
- **合理的尺寸范围**: 避免过大的 ROI2 尺寸，推荐 25-100px
- **适当的扩展参数**: 避免不对称的扩展，保持 ROI2 的稳定性
- **约束范围匹配**: 确保默认尺寸在约束范围内

### 2. 稳定性考虑
- **最小尺寸保证**: 设置合理的最小尺寸，避免 ROI2 过小
- **回退模式选择**: 根据应用场景选择合适的回退模式
- **边界约束**: 充分利用边界约束避免 ROI2 超出 ROI1 范围

### 3. 调试建议
- **渐进式调整**: 逐步调整参数，观察效果
- **日志监控**: 关注 ROI2 区域变化和错误日志
- **性能监控**: 监控 ROI2 处理时间和资源占用

## 常见问题解答

### Q1: ROI2 为什么会闪烁？
**A**: 可能是缓存时间设置过短或扩展参数不合理。建议:
- 增加 ROI2 缓存时间到 1.0-2.0 秒
- 使用更对称的扩展参数
- 启用定时器模式

### Q2: ROI2 超出 ROI1 边界怎么办？
**A**: 系统会自动应用边界约束:
- 智能收缩到边界内
- 保持中心点位置
- 记录警告日志

### Q3: 如何平衡精度和性能？
**A**: 建议配置:
- 尺寸范围: 30-80px
- 缓存时间: 1.0 秒
- 启用定时器模式
- 合理的扩展参数

### Q4: 绿线检测失败时 ROI2 如何表现？
**A**: 系统有完整的回退机制:
1. 使用缓存的交点
2. 使用 ROI1 中心
3. 应用固定尺寸
4. 记录状态日志

---

*文档生成时间: 2024-12-08*
*代码版本: NHEM backend before_green_interaction*
*配置文件: backend/app/fem_config.json*