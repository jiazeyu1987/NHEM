# NHEM Backend

## 项目简介
NHEM (New HEM Monitor) 后端服务，基于 FastAPI 的实时 HEM (高回声事件) 检测系统，移植自 NewFEM 项目。

## 系统架构
- **FastAPI 框架**: 提供高性能的 RESTful API 服务
- **实时数据处理**: 45 FPS 数据生成和峰值检测
- **线程安全**: 使用循环缓冲区进行内存存储
- **双协议支持**: HTTP API + WebSocket
- **无数据库设计**: 使用内存存储，重启后数据清空

## 文件说明
- `run.py` - 主启动脚本
- `requirements.txt` - Python 依赖包列表
- `app/` - 应用核心目录
  - `config.py` - 配置管理（支持环境变量）
  - `logging_config.py` - 日志系统配置
  - `models.py` - API 数据模型定义
  - `api/routes.py` - FastAPI 路由定义
  - `core/` - 核心处理模块
  - `utils/` - 工具函数

## 技术栈
- **Web框架**: FastAPI 0.115.0
- **ASGI服务器**: Uvicorn 0.30.6
- **数据处理**: NumPy >=1.21.0
- **图像处理**: Pillow >=8.0.0
- **数据验证**: Pydantic >=2.0.0
- **多线程**: Python threading 模块

## 快速开始

### 1. 环境要求
- Python 3.8+
- 8421端口可用（或修改配置）

### 2. 安装依赖
```bash
cd D:\ProjectPackage\NHEM\backend
pip install -r requirements.txt
```

### 3. 启动服务
```bash
python run.py
```

服务将在 http://localhost:8421 启动

## API 端点

### 系统管理
- `GET /health` - 健康检查
- `GET /status` - 系统状态和指标

### 实时数据
- `GET /data/realtime?count=N` - 获取实时数据（N个数据点）
  - 返回JSON格式的时间序列数据
  - 包含ROI图像数据、峰值信号、基线值

### 控制命令
- `POST /control` - 执行系统控制命令
  - `start_detection` - 开始检测
  - `stop_detection` - 停止检测
  - `pause_detection` - 暂停检测
  - `resume_detection` - 恢复检测
  - 需要密码认证（默认：31415）

### 视频分析
- `POST /analyze` - 视频分析接口
- `GET /data/waveform-with-peaks` - 生成带波峰标注的波形图像
- `GET /data/roi-window-capture` - ROI窗口截取

### 配置管理
- `POST /config/apply` - 应用配置更改
- `GET /config/peak-detection` - 获取峰值检测配置

## 配置说明

### 环境变量
支持以下环境变量（前缀：`NHEM_`）：

```bash
NHEM_HOST=0.0.0.0              # 服务器地址
NHEM_API_PORT=8421             # HTTP API端口
NHEM_SOCKET_PORT=30415         # WebSocket端口
NHEM_LOG_LEVEL=INFO            # 日志级别（DEBUG/INFO/WARNING/ERROR）
NHEM_BUFFER_SIZE=100           # 数据缓冲区大小
NHEM_FPS=45                    # 数据生成帧率
NHEM_PASSWORD=31415            # 控制命令密码
NHEM_ENABLE_CORS=True          # CORS支持
```

### JSON配置文件
配置文件位置：`app/fem_config.json`

```json
{
  "server": {
    "host": "0.0.0.0",
    "api_port": 8421,
    "socket_port": 30415,
    "enable_cors": true
  },
  "data_processing": {
    "fps": 45,
    "buffer_size": 100,
    "max_frame_count": 10000
  },
  "roi_capture": {
    "frame_rate": 5,
    "update_interval": 0.5,
    "default_config": {
      "x1": 10, "y1": 20,
      "x2": 210, "y2": 170
    }
  },
  "peak_detection": {
    "threshold": 110.5,
    "margin_frames": 6,
    "difference_threshold": 2.3,
    "min_region_length": 4
  }
}
```

## 核心功能

### 实时数据处理
- **数据生成器**: 45 FPS 生成模拟信号（120 ± 10）
- **峰值检测**: 智能识别超过阈值的信号峰值
- **线程安全**: 使用锁机制保护共享数据
- **循环缓冲区**: 默认保存最近100个数据点

### ROI (感兴趣区域) 处理
- **区域配置**: 可配置矩形ROI区域
- **实时截图**: 5 FPS 截取ROI区域图像
- **灰度值提取**: 计算ROI区域平均灰度值
- **Base64编码**: 返回Base64格式的图像数据

### WebSocket 支持
- **实时推送**: 支持WebSocket实时数据推送
- **双协议**: 同时支持HTTP REST和WebSocket
- **向后兼容**: 支持传统客户端连接

## 日志系统
- **文件日志**: 自动按时间戳创建日志文件
- **控制台输出**: INFO级别及以上显示在控制台
- **结构化日志**: 使用标准化的日志格式
- **日志过滤**: 过滤掉过于频繁的调试信息

## 开发说明

### 数据流处理
1. 数据处理器在后台线程运行
2. 生成模拟信号或处理ROI数据
3. 应用峰值检测算法
4. 数据存储在循环缓冲区中
5. API端点提供实时数据访问

### 启动流程
1. 初始化配置和日志系统
2. 启动FastAPI应用
3. 等待前端控制命令启动数据处理
4. 手动通过前端UI点击"开始分析"

### 安全特性
- **密码认证**: 控制命令需要密码验证
- **CORS配置**: 可配置跨域访问策略
- **数据验证**: 使用Pydantic进行严格的输入验证

## 性能特性
- **高并发**: FastAPI异步处理支持
- **内存效率**: 使用循环缓冲区避免内存泄漏
- **实时响应**: 20-45 FPS数据处理和API响应
- **线程安全**: 全面的线程安全保护机制

## 故障排除

### 常见问题
1. **端口占用**: 确保8421端口未被其他程序占用
2. **依赖缺失**: 运行 `pip install -r requirements.txt`
3. **权限问题**: 确保有日志目录写入权限
4. **CORS错误**: 检查allowed_origins配置

### 调试模式
设置环境变量启用详细日志：
```bash
NHEM_LOG_LEVEL=DEBUG python run.py
```

## 移植说明

### 从 NewFEM 移植的更改
1. **项目名称**: "NewFEM" → "NHEM"
2. **环境变量前缀**: "NEWFEM_" → "NHEM_"
3. **日志文件名**: "newfem_" → "nhem_"
4. **API版本**: "3.0.0" → "1.0.0"
5. **默认FPS**: 60 → 45（与前端一致）

### 原有功能保持
- 所有核心功能完全保留
- API接口兼容性保持
- 性能优化保持
- 实时处理能力保持

## 版本信息
- **当前版本**: 1.0.0
- **移植时间**: 2025-12-05
- **基于项目**: NewFEM 3.0.0

---
*移植自 NewFEM 项目，专为 NHEM 系统定制*