# NHEM Frontend

## 项目简介
NHEM (New HEM Monitor) 前端应用，移植自 NewFEM 项目。这是一个基于 Van JavaScript 的单页应用，提供实时 HEM (高回声事件) 检测和监控功能。

## 文件说明
- `index.html` - 主前端应用（包含所有 HTML、CSS、JavaScript 代码）
- `config.json` - 前端配置文件
- `README.md` - 本说明文件

## 功能特性
- ✅ 实时图表显示（20 FPS 更新）
- ✅ VS Code 风格的 UI 界面
- ✅ ROI（感兴趣区域）处理和显示
- ✅ 波峰检测和控制
- ✅ 响应式设计
- ✅ Mock 模式支持（离线开发）
- ✅ Y 轴范围：0-200

## 技术栈
- **前端**: HTML5, CSS3, Vanilla JavaScript (ES6+)
- **图表**: HTML5 Canvas
- **通信**: RESTful API (HTTP/JSON)
- **UI**: VS Code 风格暗色主题

## 配置说明

### 服务器配置 (config.json)
```json
{
  "server": {
    "host": "localhost",    // 服务器地址
    "api_port": 8421,       // API端口
    "enable_cors": true     // CORS支持
  }
}
```

### 显示配置
```json
{
  "display": {
    "fps": 45,             // 数据生成帧率
    "buffer_size": 100,    // 缓冲区大小
    "y_axis_min": 0,       // Y轴最小值
    "y_axis_max": 200      // Y轴最大值
  }
}
```

### ROI 配置
```json
{
  "roi_capture": {
    "frame_rate": 5,                           // ROI帧率
    "update_interval": 0.5,                    // 更新间隔
    "default_config": {
      "x1": 10, "y1": 20,                     // ROI左上角坐标
      "x2": 210, "y2": 170                    // ROI右下角坐标
    }
  }
}
```

### 波峰检测配置
```json
{
  "peak_detection": {
    "threshold": 110.5,                        // 检测阈值 (0-200)
    "margin_frames": 6,                       // 边界帧数
    "difference_threshold": 2.3,               // 帧差阈值
    "min_region_length": 4                    // 最小区域长度
  }
}
```

## 使用方法

### 1. 直接在浏览器中打开
```bash
# 直接双击 index.html 或在浏览器中打开
```

### 2. 使用 HTTP 服务器（推荐）
```bash
# Python 3
cd D:\ProjectPackage\NHEM\front
python -m http.server 3000

# Node.js
npx http-server -p 3000

# 然后访问 http://localhost:3000
```

## API 接口
前端应用需要连接到支持以下 API 端点的后端服务器：

- `GET /health` - 健康检查
- `GET /status` - 系统状态
- `GET /data/realtime?count=N` - 实时数据
- `POST /control` - 控制命令（开始/停止检测）

## 移植说明

### 从 NewFEM 移植的更改
1. 项目名称从 "NewFEM" 改为 "NHEM"
2. 简化配置文件，只保留前端相关配置
3. 修改服务器主机为 localhost
4. 添加显示配置部分
5. Y 轴范围统一为 0-200

### 原有功能保持
- 所有核心功能完全保留
- UI 界面和交互逻辑不变
- API 兼容性保持
- 实时性能优化保持

## 开发说明

### Mock 模式
前端支持 Mock 模式，可以在没有后端服务器的情况下进行开发和测试：
1. 打开前端页面
2. 点击 "Mock Mode" 开关
3. 系统将使用模拟数据进行展示

### 调试功能
- 浏览器开发者工具可查看网络请求和控制台日志
- 支持鼠标悬停查看数据点详细信息
- 实时显示系统状态和连接信息

## 浏览器兼容性
- Chrome 60+
- Firefox 55+
- Safari 12+
- Edge 79+

## 注意事项
1. 确保 Y 轴范围设置为 0-200 以保持与后端一致
2. 服务器端口配置需要与实际后端匹配
3. CORS 需要在后端正确配置以支持跨域请求

---
*移植自 NewFEM 项目，移植时间: 2025-12-05*