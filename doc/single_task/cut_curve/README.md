# 波形截取功能文档

## 概述

波形截取功能是NHEM系统中的核心功能之一，允许用户从实时数据流中截取指定长度的时间窗口数据，支持主信号数据和ROI灰度数据两种数据源，并可选地集成了先进的波峰检测算法。

## 功能特性

### 核心功能
- ✅ **主信号数据截取** - 从主数据流中截取指定帧数的信号数据
- ✅ **ROI数据截取** - 从ROI分析数据中截取灰度时间序列
- ✅ **波峰检测集成** - 在ROI数据截取时自动进行波峰检测分析
- ✅ **可配置窗口大小** - 支持50-500帧的可调节截取窗口
- ✅ **实时图表显示** - 截取后立即显示子波形图表
- ✅ **智能颜色编码** - 根据波峰检测结果自动确定图表颜色

### 高级特性
- ✅ **多算法波峰检测** - 集成阈值法、形态学法、斜率法等多种算法
- ✅ **波峰分类系统** - 绿色波峰(稳定)、红色波峰(不稳定)
- ✅ **质量评分机制** - 基于幅度、形态、稳定性的综合评分
- ✅ **缓存优化** - ROI数据缓存机制提升性能
- ✅ **错误恢复** - 完善的错误处理和用户提示

## 文档结构

本目录包含波形截取功能的完整技术文档：

### 📊 [data_flow_diagram.md](./data_flow_diagram.md)
**数据流架构文档**
- 完整的前后端数据流时序图
- API调用链路详细分析
- 数据转换和处理流程
- 错误处理和性能指标

### 🎨 [frontend_logic.md](./frontend_logic.md)
**前端逻辑文档**
- WaveformCapture类详细设计
- UI组件和事件处理机制
- SubWaveformChart图表组件实现
- 前端性能优化策略

### ⚙️ [backend_logic.md](./backend_logic.md)
**后端逻辑文档**
- FastAPI端点实现详解
- 数据存储层架构设计
- 波峰检测算法集成
- 错误处理和日志机制

## 技术架构

### 前端架构
```
WaveformCapture 类
├── 配置管理 (captureSettings)
├── UI组件控制 (elements)
├── API调用服务 (ApiService)
├── 数据处理 (displayCapturedData)
└── 图表渲染 (SubWaveformChart)
```

### 后端架构
```
FastAPI 端点层
├── /data/window-capture (主信号截取)
├── /data/roi-window-capture (ROI截取)
└── /data/roi-window-capture-with-peaks (ROI+波峰检测)

数据存储层 (DataStore)
├── get_series() - 主信号数据
├── get_roi_series() - ROI数据
└── 线程安全缓冲机制

波峰检测层 (EnhancedPeakDetector)
├── 多算法检测引擎
├── 波峰分类系统
└── 质量评分机制
```

## API接口

### 主信号数据截取
```http
GET /data/window-capture?count=100
```

**参数说明:**
- `count`: 截取帧数 (50-200)

**响应格式:**
```json
{
    "type": "window_capture",
    "timestamp": "2025-12-05T12:00:00Z",
    "window_size": 100,
    "frame_range": [800, 899],
    "series": [
        {"t": "17.78", "value": 123.45}
    ],
    "capture_metadata": {
        "duration": 2.22,
        "fps": 45.0,
        "value_range": [45.2, 167.8]
    }
}
```

### ROI数据截取
```http
GET /data/roi-window-capture?count=100
```

**参数说明:**
- `count`: 截取帧数 (50-500)

### ROI数据截取带波峰检测
```http
GET /data/roi-window-capture-with-peaks?count=100&threshold=110.5&margin_frames=6&difference_threshold=2.3
```

**参数说明:**
- `count`: 截取帧数 (50-500)
- `threshold`: 波峰检测阈值 (0-200)
- `margin_frames`: 边界扩展帧数 (1-20)
- `difference_threshold`: 帧差值阈值 (0.1-10.0)
- `force_refresh`: 强制刷新缓存 (true/false)

## 使用场景

### 1. 实时监控分析
- 截取特定时间段的数据进行详细分析
- 对比不同时间段的波形特征
- 监控系统运行状态变化

### 2. 波峰检测研究
- 分析HEM事件的发生规律
- 研究波峰的形态特征
- 评估检测算法的准确性

### 3. 数据导出和报告
- 生成分析报告的数据源
- 导出特定时间段的原始数据
- 制作演示材料

### 4. 系统调试和优化
- 分析系统性能瓶颈
- 验证数据处理算法
- 优化参数配置

## 性能指标

### 响应性能
- **API响应时间**: < 100ms
- **前端渲染时间**: < 50ms
- **数据处理延迟**: < 20ms

### 内存使用
- **主信号缓冲**: 100帧 × Frame结构
- **ROI缓冲**: 500帧 × RoiFrame结构
- **前端图表内存**: ~2MB

### 数据精度
- **时间戳精度**: 毫秒级
- **数值精度**: 浮点数
- **帧索引连续性**: 保证

## 配置参数

### 前端配置
```javascript
captureSettings = {
    windowSize: 100,        // 截取窗口大小
    dataSource: 'roi'       // 数据源类型
}
```

### 后端配置
```python
# 主信号处理
data_fps: 45               # 数据帧率
buffer_size: 100           # 缓冲区大小

# ROI处理
roi_fps: 5                 # ROI帧率
roi_buffer_size: 500       # ROI缓冲区大小

# 波峰检测
peak_threshold: 110.5              # 检测阈值
peak_margin_frames: 6              # 边界帧数
peak_difference_threshold: 2.3     # 帧差阈值
```

## 错误处理

### 常见错误类型
1. **数据不可用** - 系统未启动或无历史数据
2. **ROI未配置** - 尝试截取ROI数据但未配置ROI区域
3. **参数超范围** - 请求参数超出允许范围
4. **网络连接失败** - 前后端通信异常
5. **数据处理异常** - 波峰检测算法失败

### 错误恢复机制
- 自动重试机制
- 用户友好的错误提示
- 详细的日志记录
- 优雅的降级处理

## 开发指南

### 前端开发
- 参考 `frontend_logic.md` 了解详细的实现逻辑
- 使用 `WaveformCapture` 类进行功能扩展
- 遵循ES6+现代JavaScript规范
- 注意Canvas渲染性能优化

### 后端开发
- 参考 `backend_logic.md` 了解API设计
- 使用Pydantic模型进行数据验证
- 遵循FastAPI最佳实践
- 注意线程安全和性能优化

### 测试建议
- 单元测试覆盖核心算法
- 集成测试验证API功能
- 性能测试确保响应时间
- 用户体验测试验证交互流程

## 版本历史

### v1.0.0 (2025-12-05)
- ✅ 基础波形截取功能
- ✅ 主信号和ROI数据源支持
- ✅ 波峰检测算法集成
- ✅ 前端图表显示功能

### 未来计划
- 📋 数据导出功能增强
- 📋 更多图表类型支持
- 📋 批量截取功能
- 📋 历史数据对比分析

## 相关文档

- [NHEM系统架构文档](../../code_structure/architecture_overview.md)
- [实时曲线渲染文档](../realtime_curve/)
- [API接口文档](../../code_structure/api_endpoints.md)
- [数据存储设计](../../code_structure/data_flow.md)

---

*本文档最后更新: 2025-12-05*