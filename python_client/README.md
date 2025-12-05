# NHEM Python Client

## 项目简介
NHEM (New HEM Monitor) Python客户端，提供实时HEM（高回声事件）检测和监控功能的图形界面，移植自NewFEM项目。

## 功能特性
- ✅ 实时数据监控和可视化（20 FPS更新率）
- ✅ 基于matplotlib的交互式图表
- ✅ Tkinter GUI界面，支持多种操作模式
- ✅ HTTP API与后端服务器通信
- ✅ 配置文件管理和参数同步
- ✅ 波峰检测和控制功能
- ✅ Y轴范围：0-200（与前端和后端统一）

## 文件说明
- `run_realtime_client.py` - 主启动脚本（推荐使用）
- `http_realtime_client.py` - 完整的GUI应用程序
- `realtime_plotter.py` - 实时绘图组件
- `simple_http_client.py` - 简化版客户端
- `client.py` - 命令行API客户端
- `local_config_loader.py` - 配置文件加载器
- `http_client_config.json` - 默认配置文件
- `__init__.py` - Python包初始化

## 技术栈
- **GUI框架**: Tkinter (Python标准库)
- **数据可视化**: Matplotlib 3.3.0+
- **HTTP通信**: Requests 2.25.0+
- **图像处理**: Pillow 8.0.0+
- **数值计算**: NumPy 1.20.0+
- **并发处理**: Threading

## 快速开始

### 1. 环境要求
- Python 3.8+
- matplotlib >= 3.3.0
- requests >= 2.25.0
- Pillow >= 8.0.0
- numpy >= 1.20.0

### 2. 安装依赖
```bash
cd D:\ProjectPackage\NHEM\python_client
pip install matplotlib requests pillow numpy
```

### 3. 启动客户端

#### 完整GUI版本（推荐）
```bash
python run_realtime_client.py
```

#### 简化GUI版本
```bash
python simple_http_client.py
```

#### 命令行版本
```bash
# 查看帮助
python client.py --help

# 获取系统状态
python client.py status

# 启动检测
python client.py start-detection --password 31415

# 停止检测
python client.py stop-detection --password 31415
```

## 使用说明

### 完整GUI版本 (run_realtime_client.py)
这是功能最完整的版本，提供：

1. **实时数据图表**:
   - 动态更新的数据曲线
   - Y轴范围：0-200
   - 可自定义图表参数

2. **系统控制面板**:
   - 开始/停止检测
   - 暂停/恢复处理
   - 实时状态显示

3. **ROI配置**:
   - 可视化ROI区域设置
   - 实时ROI数据采集
   - 灰度值显示

4. **参数配置**:
   - 波峰检测参数
   - 显示参数调整
   - 配置持久化保存

### 简化GUI版本 (simple_http_client.py)
轻量级版本，包含：
- 基础实时图表
- 简化的控制按钮
- 自动连接功能
- 紧凑模式支持

### 命令行版本 (client.py)
适合脚本和自动化使用：
- 完整的API命令封装
- 参数化的命令行接口
- 批量操作支持
- 脚本友好的输出格式

## 配置管理

### 配置文件加载
客户端会自动尝试从以下位置加载配置：
1. `../backend/app/fem_config.json` (相对路径)
2. `../../backend/app/fem_config.json` (相对路径)
3. `D:/ProjectPackage/NHEM/backend/app/fem_config.json` (绝对路径)
4. 当前目录的 `fem_config.json`

### 配置文件格式
```json
{
  "server": {
    "host": "localhost",
    "api_port": 8421
  },
  "roi_capture": {
    "frame_rate": 5,
    "default_config": {
      "x1": 10, "y1": 20,
      "x2": 210, "y2": 170
    }
  },
  "peak_detection": {
    "threshold": 110.5,
    "margin_frames": 6,
    "difference_threshold": 2.3
  },
  "display": {
    "fps": 45,
    "buffer_size": 100,
    "y_axis_min": 0,
    "y_axis_max": 200
  }
}
```

### 环境变量
支持以下环境变量：
```bash
NHEM_BASE_URL=http://localhost:8421    # 服务器地址
NHEM_PASSWORD=31415                    # 控制密码
```

## API接口

客户端与NHEM后端通过HTTP API通信：

### 系统管理
- `GET /health` - 健康检查
- `GET /status` - 系统状态

### 数据获取
- `GET /data/realtime?count=N` - 获取实时数据
- `GET /data/roi-window-capture` - ROI窗口数据

### 控制命令
- `POST /control` - 执行控制命令
  - `start_detection` - 开始检测
  - `stop_detection` - 停止检测
  - `pause_detection` - 暂停检测
  - `resume_detection` - 恢复检测

### 配置管理
- `GET /config/peak-detection` - 获取峰值检测配置
- `POST /config/apply` - 应用配置更改

## 操作说明

### 基本操作流程
1. **启动后端服务器**: 确保NHEM后端在 http://localhost:8421 运行
2. **启动Python客户端**: 运行 `python run_realtime_client.py`
3. **连接服务器**: 点击连接按钮或自动连接
4. **配置ROI** (可选): 设置感兴趣区域
5. **开始检测**: 点击"开始分析"按钮
6. **监控数据**: 观察实时图表和状态信息

### 高级功能
- **波形截取**: 截取特定时间段的数据
- **参数调整**: 实时修改检测参数
- **数据导出**: 保存图表和数据
- **批量操作**: 使用命令行工具进行批量处理

## 故障排除

### 常见问题

1. **连接失败**:
   - 检查NHEM后端是否在运行
   - 确认端口8421未被占用
   - 检查防火墙设置

2. **配置加载失败**:
   - 确认配置文件路径正确
   - 检查配置文件JSON格式
   - 验证文件读取权限

3. **图表显示问题**:
   - 安装matplotlib依赖
   - 检查Python环境
   - 确认数据格式正确

4. **导入错误**:
   - 安装所有依赖包
   - 检查Python版本兼容性
   - 确认文件路径正确

### 调试模式
客户端提供详细的调试信息：
```bash
# 启用详细日志
python run_realtime_client.py 2>&1 | tee client.log

# 测试API连接
python client.py --base-url http://localhost:8421 status
```

## 开发说明

### 扩展功能
- 添加新的UI控件：修改 `http_realtime_client.py`
- 自定义图表样式：修改 `realtime_plotter.py`
- 扩展API功能：修改 `client.py`
- 配置新参数：修改 `local_config_loader.py`

### 代码结构
```
run_realtime_client.py     # 入口点
├── http_realtime_client   # 主GUI应用
│   ├── realtime_plotter   # 绘图组件
│   └── local_config_loader # 配置管理
├── simple_http_client     # 简化版GUI
└── client.py              # 命令行工具
```

## 移植说明

### 从 NewFEM 移植的更改
1. **项目名称**: "NewFEM" → "NHEM"
2. **配置路径**: "backends/" → "backend/"
3. **UI标题**: 更新所有窗口标题
4. **环境变量**: "NEWFEM_" → "NHEM_"
5. **文档更新**: README.md中的项目信息

### 原有功能保持
- 所有核心功能完全保留
- API兼容性保持
- 实时性能保持
- 配置管理保持

## 版本信息
- **当前版本**: 1.0.0
- **移植时间**: 2025-12-05
- **基于项目**: NewFEM Python Client

---
*移植自 NewFEM 项目，专为 NHEM 系统定制*