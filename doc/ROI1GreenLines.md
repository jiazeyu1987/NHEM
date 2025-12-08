# ROI1绿色线条相交点识别需求文档

## 需求概述

在NHEM系统中，需要在ROI1区域识别两条不相交的绿色线条，并计算它们的虚拟交点（延长线交点）。ROI2区域不执行线条检测，仅用于灰度分析。

## 功能需求

### 1. 检测范围
- **ROI1区域**：执行完整的绿色线条相交点检测
  - 绿色线条识别和提取
  - 直线检测和延长线交点计算
  - 虚拟交点坐标计算
  - Canvas可视化显示
  - 实时信息栏状态显示

- **ROI2区域**：不执行线条检测
  - 仅用于灰度值分析
  - 不显示线条检测结果
  - 不参与相交点计算

### 2. 实时信息栏显示要求
当在ROI1检测到绿色线条的虚拟交点时，实时信息栏应显示：

- **识别成功**：`线条相交点: 已识别 (x, y) 置信度: XX%`
- **启用未识别**：`线条相交点: 已启用 - 未识别`
- **未启用**：`线条相交点: 未启用`
- **检测错误**：`线条相交点: 检测错误: [错误信息]`

### 3. 图像处理方法

#### 3.1 颜色阈值分割
```python
import cv2
import numpy as np

# 1. 提取绿色区域
hsv = cv2.cvtColor(roi1_image, cv2.COLOR_BGR2HSV)
green_mask = cv2.inRange(hsv, (40, 50, 50), (80, 255, 255))
```

**技术要求**：
- 使用HSV色彩空间进行绿色提取
- 绿色阈值范围：H(40-80), S(50-255), V(50-255)
- 形态学操作去除噪声
- 仅在ROI1区域执行

#### 3.2 边缘检测与直线检测
```python
# 2. 霍夫直线检测
edges = cv2.Canny(green_mask, 25, 80)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=15, maxLineGap=8)
```

**技术要求**：
- Canny边缘检测，低阈值25，高阈值80
- 霍夫直线变换检测直线段
- 参数优化：距离分辨率1px，角度分辨率1度
- 最小线段长度15px，最大间隙8px
- 过滤掉水平线和垂直线，保留有意义的斜线

#### 3.3 虚拟交点计算
```python
# 3. 计算两条线的交点
def calculate_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    # 计算分母
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    # 避免除零
    if abs(denom) < 1e-10:
        return None  # 平行线

    # 计算交点（可以在线段延长线上）
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)

    return (x, y)
```

**技术要求**：
- 计算线条延长线的虚拟交点
- 支持不相交线条的交点计算
- 精确的数学计算，避免数值不稳定
- 基于线条长度的置信度评估

#### 3.4 轮廓分析法（备选方案）
```python
# 对绿色区域进行形态学处理
kernel = np.ones((3, 3), np.uint8)
processed = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

# 找到轮廓
contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

**技术要求**：
- 形态学处理优化轮廓提取
- 轮廓面积过滤去除小噪声
- 计算轮廓质心作为线条特征点
- 作为霍夫变换的补充验证方法

### 4. 性能要求

#### 4.1 实时性
- **处理频率**：与ROI捕获同步（4 FPS）
- **处理时间**：单帧处理时间 < 300ms
- **缓存机制**：避免重复处理相同图像
- **异步处理**：不阻塞主数据流程

#### 4.2 准确性
- **坐标精度**：像素级精度，误差 < 5px
- **检测成功率**：在清晰绿色线条下 > 90%
- **置信度评估**：基于线条长度和检测质量的置信度计算
- **错误处理**：完善的异常处理和回退机制

### 5. 用户界面要求

#### 5.1 Python客户端实现（run_realtime_client.py）
所有前端功能代码都在Python客户端中实现，不是在Web前端实现：

- **matplotlib Canvas可视化（ROI1）**：
  - 在ROI1图像上绘制检测到的线条
  - 显示虚拟交点标记（圆圈 + 十字线）
  - 坐标信息显示：`(x, y) c:confidence`
  - 置信度颜色编码：高置信度红色，低置信度橙色

- **Tkinter实时信息栏**：
  - 实时显示识别状态和坐标信息
  - 状态颜色编码：绿色（成功）、黄色（未识别）、红色（错误）、灰色（未启用）
  - 置信度百分比显示
  - 错误信息提示

- **控制面板**：
  - ROI1专用线条检测启用/禁用控制
  - 手动刷新检测按钮
  - 检测参数配置（可选）

#### 5.2 Web前端（front/index.html）
Web前端不实现线条检测相关功能，仅用于：
- 系统状态监控
- 基础ROI配置
- 数据查看

### 6. 技术实现架构

#### 6.1 后端实现
```python
class LineIntersectionDetector:
    def __init__(self):
        self.hsv_green_lower = (40, 50, 50)
        self.hsv_green_upper = (80, 255, 255)
        self.canny_low_threshold = 25
        self.canny_high_threshold = 80

    def detect_intersection(self, roi1_image):
        # 仅处理ROI1图像
        hsv = cv2.cvtColor(roi1_image, cv2.COLOR_RGB2HSV)
        green_mask = cv2.inRange(hsv, self.hsv_green_lower, self.hsv_green_upper)

        # 形态学处理
        kernel = np.ones((3, 3), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

        # 边缘检测
        edges = cv2.Canny(green_mask, self.canny_low_threshold, self.canny_high_threshold)

        # 霍夫直线检测
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50,
                               minLineLength=15, maxLineGap=8)

        # 计算虚拟交点
        intersection = self.calculate_best_intersection(lines)

        return intersection
```

#### 6.2 Python客户端集成（run_realtime_client.py）
```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class LineDetectionDisplay:
    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        self.setup_ui()

    def setup_ui(self):
        # 创建matplotlib画布用于ROI1显示
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.parent_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # 创建状态显示标签
        self.status_label = tk.Label(self.parent_frame, text="线条相交点: 未启用",
                                   font=('Arial', 12), fg='gray')
        self.status_label.pack(side=tk.BOTTOM, pady=5)

        # 创建控制按钮
        button_frame = tk.Frame(self.parent_frame)
        button_frame.pack(side=tk.BOTTOM, pady=5)

        self.toggle_btn = tk.Button(button_frame, text="启用检测",
                                  command=self.toggle_detection)
        self.toggle_btn.pack(side=tk.LEFT, padx=5)

        self.refresh_btn = tk.Button(button_frame, text="手动检测",
                                   command=self.manual_detection)
        self.refresh_btn.pack(side=tk.LEFT, padx=5)

    def update_display(self, roi1_data, line_intersection):
        # 清空画布
        self.ax.clear()

        # 显示ROI1图像
        if roi1_data:
            # 解码ROI1图像并显示
            roi_image = self.decode_roi_image(roi1_data)
            self.ax.imshow(roi_image, cmap='gray')

            # 绘制线条检测结果
            if line_intersection and line_intersection.intersection:
                self.draw_intersection(line_intersection)

                # 更新状态显示
                x, y = line_intersection.intersection
                confidence = line_intersection.confidence * 100
                status_text = f"线条相交点: 已识别 ({int(x)}, {int(y)}) 置信度: {confidence:.1f}%"
                self.status_label.config(text=status_text, fg='green')
            else:
                status_text = "线条相交点: 已启用 - 未识别"
                self.status_label.config(text=status_text, fg='orange')
        else:
            self.status_label.config(text="线条相交点: 未启用", fg='gray')

        self.canvas.draw()

    def draw_intersection(self, line_intersection):
        x, y = line_intersection.intersection
        confidence = line_intersection.confidence

        # 根据置信度选择颜色
        color = 'red' if confidence > 0.7 else 'orange'

        # 绘制交点标记
        # 外圈
        circle = patches.Circle((x, y), 6, fill=False, edgecolor=color, linewidth=2)
        self.ax.add_patch(circle)

        # 内圈
        circle_inner = patches.Circle((x, y), 3, fill=True, facecolor=color, edgecolor=color)
        self.ax.add_patch(circle_inner)

        # 十字线
        self.ax.plot([x-8, x+8], [y, y], color=color, linewidth=2)
        self.ax.plot([x, x], [y-8, y+8], color=color, linewidth=2)

        # 坐标文本
        text = f"({int(x)}, {int(y)})\nc:{confidence:.2f}"
        self.ax.text(x+10, y-10, text, color='white', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))

# 在主客户端类中集成
class RealtimeClient:
    def __init__(self):
        # ... 其他初始化代码
        self.line_detection_display = LineDetectionDisplay(self.main_frame)
        self.line_detection_enabled = False
        self.last_line_intersection = None

    def update_realtime_data(self, data):
        # ... 其他数据更新代码

        # 更新线条检测显示
        if 'line_intersection' in data and self.line_detection_enabled:
            self.last_line_intersection = data['line_intersection']
            self.line_detection_display.update_display(
                data.get('roi1_data'),
                self.last_line_intersection
            )
        else:
            self.line_detection_display.update_display(None, None)

    def toggle_detection(self):
        if self.line_detection_enabled:
            # 禁用检测
            response = self.api_call('POST', '/api/roi/line-intersection/disable')
            if response.get('success'):
                self.line_detection_enabled = False
                self.line_detection_display.toggle_btn.config(text="启用检测")
        else:
            # 启用检测
            response = self.api_call('POST', '/api/roi/line-intersection/enable')
            if response.get('success'):
                self.line_detection_enabled = True
                self.line_detection_display.toggle_btn.config(text="禁用检测")
```

#### 6.3 实时数据获取集成
```python
class RealtimeClient:
    def fetch_enhanced_data(self):
        """获取包含线条检测的增强实时数据"""
        params = {
            'count': 100,
            'include_line_intersection': str(self.line_detection_enabled).lower()
        }

        response = requests.get(
            f"{self.server_url}/data/realtime/enhanced",
            params=params
        )

        if response.status_code == 200:
            data = response.json()
            self.update_realtime_data(data)
            return data
        else:
            print(f"获取数据失败: {response.status_code}")
            return None

    def main_loop(self):
        """主循环，定期获取和更新数据"""
        while self.running:
            try:
                data = self.fetch_enhanced_data()

                # 控制更新频率
                time.sleep(0.05)  # 20 FPS

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"主循环错误: {e}")
                time.sleep(1)  # 错误时等待1秒
```

### 7. 配置参数

#### 7.1 检测参数
```json
{
  "line_detection": {
    "enabled": false,
    "hsv_green_lower": [40, 50, 50],
    "hsv_green_upper": [80, 255, 255],
    "canny_low_threshold": 25,
    "canny_high_threshold": 80,
    "hough_threshold": 50,
    "hough_min_line_length": 15,
    "hough_max_line_gap": 8,
    "min_lines_for_intersection": 2,
    "roi_processing_mode": "roi1_only"
  }
}
```

#### 7.2 ROI配置
```json
{
  "roi_capture": {
    "default_config": {
      "x1": 1080,
      "y1": 80,
      "x2": 1920,
      "y2": 1080
    },
    "line_detection_mode": "roi1_only"
  }
}
```

### 8. 测试要求

#### 8.1 功能测试
- 绿色线条识别准确率测试
- 虚拟交点计算精度测试
- 不同光照条件下的鲁棒性测试
- ROI1/ROI2分离处理验证

#### 8.2 性能测试
- 处理时间测试（目标 < 300ms）
- 内存使用测试
- 长时间运行稳定性测试

#### 8.3 边界条件测试
- 无绿色线条图像处理
- 单条线条图像处理
- 多条线条图像处理
- 噪声干扰图像处理

### 9. 错误处理

#### 9.1 检测失败处理
- 无线条检测到时返回null
- 平行线处理（无交点）
- 检测参数异常时的回退机制
- 图像格式错误的处理

#### 9.2 用户反馈
- 检测状态的清晰显示
- 错误信息的友好提示
- 操作指导（如"请确保ROI1中有绿色线条"）

### 10. 实施优先级

#### 10.1 核心功能（P0）
- ROI1绿色线条检测算法
- 虚拟交点计算
- 实时信息栏状态显示

#### 10.2 重要功能（P1）
- Canvas可视化显示
- 用户控制界面
- 配置参数管理

#### 10.3 优化功能（P2）
- 性能优化和缓存
- 错误处理完善
- 用户体验优化

---

**文档版本**：1.0
**创建时间**：2025-12-07
**最后更新**：2025-12-07
**负责人**：NHEM开发团队