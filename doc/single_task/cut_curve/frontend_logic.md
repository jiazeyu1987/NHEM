# 波形截取功能前端逻辑文档

## 概述

波形截取功能的前端实现位于 `front/index.html` 中，主要由 `WaveformCapture` 类负责管理。该功能允许用户从实时数据流中截取指定长度的波形数据，并支持两种数据源：主信号数据和ROI灰度数据。

## 核心类结构

### WaveformCapture 类

```javascript
class WaveformCapture {
    constructor() {
        // 截取数据缓存
        this.capturedData = null;

        // 截取设置
        this.captureSettings = {
            windowSize: 100,        // 窗口大小 (帧数)
            dataSource: 'roi'       // 数据源: 'roi' | 'main'
        };

        // UI元素引用
        this.elements = {
            // 截取控制面板
            windowSlider: document.getElementById('capture-window-slider'),
            windowSize: document.getElementById('capture-window-size'),
            captureBtn: document.getElementById('capture-btn'),
            statusIndicator: document.getElementById('capture-status-indicator'),
            statusText: document.getElementById('capture-status-text'),

            // 截取信息显示
            infoPanel: document.getElementById('capture-info'),
            frameCount: document.getElementById('captured-frame-count'),
            frameRange: document.getElementById('captured-frame-range'),
            duration: document.getElementById('captured-duration'),
            dataType: document.getElementById('captured-data-type')
        };
    }
}
```

## 主要功能模块

### 1. 初始化和事件绑定

#### 初始化流程
```javascript
init() {
    this.syncDisplayWithSettings();    // 同步显示设置
    this.cacheElements();             // 缓存DOM元素
    this.initEventListeners();        // 初始化事件监听
}
```

#### 事件监听器设置
```javascript
initEventListeners() {
    // 窗口大小滑块变化
    this.elements.windowSlider.addEventListener('input', (e) => {
        this.captureSettings.windowSize = parseInt(e.target.value);
        this.elements.windowSize.textContent = `${this.captureSettings.windowSize} 帧`;
    });

    // 数据源切换 (通过单选按钮)
    document.querySelectorAll('input[name="capture-data-source"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            this.captureSettings.dataSource = e.target.value;
        });
    });

    // 截取按钮点击
    this.elements.captureBtn.addEventListener('click', () => {
        this.captureWaveform();
    });

    // 清除按钮点击
    document.getElementById('clear-capture-btn').addEventListener('click', () => {
        this.clearCapture();
    });
}
```

### 2. 波形截取核心逻辑

#### captureWaveform() 方法
```javascript
async captureWaveform() {
    try {
        const dataSource = this.captureSettings.dataSource;

        // ROI数据源验证
        if (dataSource === 'roi') {
            const roiStatus = RoiManager.getRoiStatus();
            if (roiStatus !== 'configured') {
                alert('请先配置 ROI 区域并运行检测，再进行截取。');
                return;
            }
        }

        // 更新UI状态
        this.updateStatus('connecting', '截取中...');
        this.elements.captureBtn.disabled = true;

        let response;

        // 根据数据源选择API
        if (dataSource === 'roi') {
            // 使用带波峰检测的ROI截取API
            response = await ApiService.captureRoiWindowWithPeaks(
                this.captureSettings.windowSize,
                // 波峰检测参数从全局配置获取
                globalSettings.peakDetection.threshold,
                globalSettings.peakDetection.marginFrames,
                globalSettings.peakDetection.differenceThreshold
            );
        } else {
            // 主信号数据截取
            response = await ApiService.captureWindow(this.captureSettings.windowSize);
        }

        // 处理成功响应
        if (response.success) {
            this.capturedData = response;
            this.displayCapturedData(response);
            this.updateStatus('connected', '截取成功');
            this.showSuccessAnimation();
        } else {
            throw new Error(response?.error || '截取失败');
        }

    } catch (error) {
        console.error('波形截取失败:', error);
        this.updateStatus('disconnected', '截取失败');
        alert('波形截取失败: ' + error.message);
    } finally {
        this.elements.captureBtn.disabled = false;
    }
}
```

### 3. API调用服务

#### ApiService 方法
```javascript
// 窗口截取API
async captureWindow(count = 100) {
    try {
        const response = await fetch(`${appState.serverUrl}/data/window-capture?count=${count}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return await response.json();
    } catch (e) {
        console.error("Window capture error:", e);
        throw e;
    }
}

// ROI窗口截取API
async captureRoiWindow(count = 100) {
    try {
        const response = await fetch(`${appState.serverUrl}/data/roi-window-capture?count=${count}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return await response.json();
    } catch (e) {
        console.error("ROI window capture error:", e);
        throw e;
    }
}

// ROI窗口截取带波峰检测API
async captureRoiWindowWithPeaks(count = 100, threshold = null, marginFrames = null, differenceThreshold = null) {
    try {
        const url = `${appState.serverUrl}/data/roi-window-capture-with-peaks?count=${count}&threshold=${threshold}&margin_frames=${marginFrames}&difference_threshold=${differenceThreshold}`;
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return await response.json();
    } catch (e) {
        console.error("ROI window capture with peaks error:", e);
        throw e;
    }
}
```

### 4. 数据显示和处理

#### displayCapturedData() 方法
```javascript
displayCapturedData(data) {
    const dataSource = this.captureSettings.dataSource;

    // 显示截取信息面板 - 添加动画
    this.elements.infoPanel.style.display = 'block';
    setTimeout(() => {
        this.elements.infoPanel.classList.add('show');
    }, 50);

    if (dataSource === 'roi') {
        // ROI数据显示
        this.displayRoiCapturedData(data);
    } else {
        // 主信号数据显示
        this.displayMainCapturedData(data);
    }

    // 显示子波形图表
    this.showSubWaveform(data);
}
```

#### ROI数据显示
```javascript
displayRoiCapturedData(data) {
    // 基本统计信息
    const frameCount = data.series.length;
    const duration = data.capture_metadata?.capture_duration || 0;
    const roiRange = data.roi_frame_range;
    const mainRange = data.main_frame_range;

    // 更新UI显示
    this.elements.frameCount.textContent = frameCount;
    this.elements.frameRange.textContent = `${mainRange[0]}-${mainRange[1]}`;
    this.elements.duration.textContent = `${duration.toFixed(2)}s`;
    this.elements.dataType.textContent = 'ROI灰度数据';

    // 波峰检测结果显示
    if (data.peak_detection_results && data.peak_detection_results.peaks) {
        const peaks = data.peak_detection_results.peaks;
        const greenPeaks = peaks.filter(p => p.type === 'green').length;
        const redPeaks = peaks.filter(p => p.type === 'red').length;

        console.log(`截取到 ${greenPeaks} 个绿色波峰, ${redPeaks} 个红色波峰`);
    }
}
```

#### 主信号数据显示
```javascript
displayMainCapturedData(data) {
    // 基本统计信息
    const frameCount = data.series.length;
    const duration = data.capture_metadata?.duration || 0;
    const range = data.frame_range;

    // 数值范围分析
    const values = data.series.map(point => point.value);
    const minValue = Math.min(...values);
    const maxValue = Math.max(...values);
    const valueRange = maxValue - minValue;

    // 更新UI显示
    this.elements.frameCount.textContent = frameCount;
    this.elements.frameRange.textContent = `${range[0]}-${range[1]}`;
    this.elements.duration.textContent = `${duration.toFixed(2)}s`;
    this.elements.dataType.textContent = '主信号数据';

    // 数值范围警告
    if (valueRange < 1.0) {
        console.warn(`WARNING - Small value range (${valueRange.toFixed(1)}) may appear as a flat line!`);
    }
}
```

### 5. 子波形图表显示

#### showSubWaveform() 方法
```javascript
showSubWaveform(data) {
    const dataSource = this.captureSettings.dataSource;
    let series;

    // 提取数据序列
    if (dataSource === 'roi') {
        series = data.series.map(point => ({
            t: point.t,
            value: point.gray_value
        }));
    } else {
        series = data.series;
    }

    // 确定子波形颜色 (基于波峰检测)
    let subWaveColor = '#3498db'; // 默认蓝色
    if (dataSource === 'roi' && data.peak_detection_results && data.peak_detection_results.peaks) {
        subWaveColor = this.determineSubWaveColor(data.peak_detection_results.peaks);
    }

    // 创建或更新子波形图表
    if (this.subWaveformChart) {
        this.subWaveformChart.updateSeries([{
            name: '截取波形',
            data: series,
            color: subWaveColor
        }]);
    } else {
        this.subWaveformChart = new SubWaveformChart('sub-waveform-chart', {
            series: [{
                name: '截取波形',
                data: series,
                color: subWaveColor
            }],
            showPeaks: dataSource === 'roi',
            peaks: dataSource === 'roi' ? data.peak_detection_results?.peaks || [] : []
        });
    }

    // 显示图表容器
    const container = document.getElementById('sub-waveform-container');
    container.style.display = 'block';
    setTimeout(() => {
        container.classList.add('show');
    }, 50);
}
```

#### 子波形颜色确定
```javascript
determineSubWaveColor(peaks) {
    if (!peaks || peaks.length === 0) return '#3498db'; // 默认蓝色

    // 检查距离截取点最近的波峰颜色
    const capturePoint = Math.floor(this.captureSettings.windowSize / 2);
    let closestPeak = null;
    let minDistance = Infinity;

    // 寻找最近的绿色波峰
    for (const peak of peaks) {
        const distance = Math.abs(peak.index - capturePoint);
        if (peak.type === 'green' && distance < minDistance) {
            minDistance = distance;
            closestPeak = peak;
        }
    }

    // 如果没有绿色波峰，寻找最近的红色波峰
    if (!closestPeak) {
        for (const peak of peaks) {
            const distance = Math.abs(peak.index - capturePoint);
            if (peak.type === 'red' && distance < minDistance) {
                minDistance = distance;
                closestPeak = peak;
            }
        }
    }

    // 返回对应的颜色
    if (closestPeak) {
        return closestPeak.type === 'green' ? '#27ae60' : '#e74c3c'; // 绿色或红色
    }

    return '#3498db'; // 默认蓝色
}
```

### 6. UI状态管理

#### 状态更新方法
```javascript
updateStatus(status, text) {
    this.elements.statusIndicator.className = `indicator ${status}`;
    this.elements.statusText.textContent = text;
}
```

#### 成功动画
```javascript
showSuccessAnimation() {
    this.elements.captureBtn.classList.add('success');
    setTimeout(() => {
        this.elements.captureBtn.classList.remove('success');
    }, 1000);
}
```

#### 清除截取
```javascript
clearCapture() {
    this.capturedData = null;

    // 隐藏信息面板
    this.elements.infoPanel.classList.remove('show');
    setTimeout(() => {
        this.elements.infoPanel.style.display = 'none';
    }, 300);

    // 隐藏子波形图表
    const container = document.getElementById('sub-waveform-container');
    container.classList.remove('show');
    setTimeout(() => {
        container.style.display = 'none';
    }, 300);

    // 清除图表
    if (this.subWaveformChart) {
        this.subWaveformChart.destroy();
        this.subWaveformChart = null;
    }

    // 重置状态
    this.updateStatus('disconnected', '等待截取');
}
```

### 7. SubWaveformChart 类

#### 子波形图表组件
```javascript
class SubWaveformChart {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.container.appendChild(this.canvas);

        this.options = {
            width: 600,
            height: 200,
            backgroundColor: '#2d2d30',
            gridColor: '#3e3e42',
            textColor: '#cccccc',
            ...options
        };

        this.series = options.series || [];
        this.showPeaks = options.showPeaks || false;
        this.peaks = options.peaks || [];

        this.init();
    }

    init() {
        this.resizeCanvas();
        this.bindEvents();
        this.render();
    }

    resizeCanvas() {
        const rect = this.container.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = this.options.height;
    }

    bindEvents() {
        // 鼠标悬停显示数据点信息
        this.canvas.addEventListener('mousemove', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            this.showTooltip(x, y, e);
        });

        this.canvas.addEventListener('mouseleave', () => {
            this.hideTooltip();
        });
    }

    render() {
        const { width, height } = this.canvas;
        const { series } = this;

        if (!series || series.length === 0) return;

        const data = series[0].data;
        if (!data || data.length === 0) return;

        // 清除画布
        this.ctx.fillStyle = this.options.backgroundColor;
        this.ctx.fillRect(0, 0, width, height);

        // 计算数据范围
        const values = data.map(d => d.value);
        const minValue = Math.min(...values);
        const maxValue = Math.max(...values);
        const valueRange = maxValue - minValue || 1;

        const padding = 40;
        const chartWidth = width - padding * 2;
        const chartHeight = height - padding * 2;

        // 绘制网格
        this.drawGrid(padding, chartWidth, chartHeight);

        // 绘制坐标轴
        this.drawAxes(padding, chartWidth, chartHeight, minValue, maxValue);

        // 绘制数据线
        this.drawDataLine(data, padding, chartWidth, chartHeight, minValue, valueRange);

        // 绘制波峰标记 (如果启用)
        if (this.showPeaks && this.peaks.length > 0) {
            this.drawPeakMarkers(this.peaks, data, padding, chartWidth, chartHeight, minValue, valueRange);
        }
    }

    drawDataLine(data, padding, chartWidth, chartHeight, minValue, valueRange) {
        if (data.length === 0) return;

        this.ctx.strokeStyle = this.series[0].color || '#3498db';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();

        data.forEach((point, index) => {
            const x = padding + (index / (data.length - 1)) * chartWidth;
            const y = padding + chartHeight - ((point.value - minValue) / valueRange) * chartHeight;

            if (index === 0) {
                this.ctx.moveTo(x, y);
            } else {
                this.ctx.lineTo(x, y);
            }
        });

        this.ctx.stroke();
    }

    drawPeakMarkers(peaks, data, padding, chartWidth, chartHeight, minValue, valueRange) {
        peaks.forEach(peak => {
            if (peak.index < data.length) {
                const point = data[peak.index];
                const x = padding + (peak.index / (data.length - 1)) * chartWidth;
                const y = padding + chartHeight - ((point.value - minValue) / valueRange) * chartHeight;

                // 根据波峰类型选择颜色
                this.ctx.fillStyle = peak.type === 'green' ? '#27ae60' : '#e74c3c';

                // 绘制圆点标记
                this.ctx.beginPath();
                this.ctx.arc(x, y, 4, 0, Math.PI * 2);
                this.ctx.fill();

                // 绘制外圈
                this.ctx.strokeStyle = this.ctx.fillStyle;
                this.ctx.lineWidth = 2;
                this.ctx.beginPath();
                this.ctx.arc(x, y, 6, 0, Math.PI * 2);
                this.ctx.stroke();
            }
        });
    }

    // ... 其他绘图方法

    updateSeries(newSeries) {
        this.series = newSeries;
        if (newSeries[0]) {
            this.peaks = newSeries[0].peaks || [];
        }
        this.render();
    }

    destroy() {
        if (this.canvas && this.canvas.parentNode) {
            this.canvas.parentNode.removeChild(this.canvas);
        }
    }
}
```

## 错误处理

### 常见错误场景

1. **ROI未配置错误**
```javascript
if (dataSource === 'roi') {
    const roiStatus = RoiManager.getRoiStatus();
    if (roiStatus !== 'configured') {
        alert('请先配置 ROI 区域并运行检测，再进行截取。');
        return;
    }
}
```

2. **网络请求失败**
```javascript
try {
    response = await ApiService.captureRoiWindowWithPeaks(...);
} catch (error) {
    console.error('波形截取失败:', error);
    this.updateStatus('disconnected', '截取失败');
    alert('波形截取失败: ' + error.message);
}
```

3. **API响应错误**
```javascript
if (response.success) {
    // 处理成功响应
} else {
    throw new Error(response?.error || '截取失败');
}
```

## 性能优化

### 1. Canvas渲染优化
- 使用 `requestAnimationFrame` 进行动画
- 避免频繁的重绘操作
- 合理设置图表尺寸

### 2. 内存管理
- 及时清除不再使用的图表实例
- 避免重复创建DOM元素
- 合理使用事件监听器

### 3. 网络请求优化
- 实施请求防抖机制
- 显示加载状态提升用户体验
- 合理的错误重试机制

## 配置参数

### 可配置项
```javascript
captureSettings = {
    windowSize: 100,        // 截取窗口大小 (50-200帧)
    dataSource: 'roi'       // 数据源 ('roi' | 'main')
}
```

### API参数
```javascript
// ROI截取参数
{
    count: 100,                           // 截取帧数
    threshold: 110.5,                     // 波峰检测阈值
    margin_frames: 6,                     // 边界帧数
    difference_threshold: 2.3             // 帧差阈值
}
```

## 扩展性设计

### 1. 数据源扩展
- 支持添加新的数据源类型
- 统一的数据源接口设计
- 配置化的数据源选择

### 2. 图表功能扩展
- 支持多种图表类型
- 自定义颜色方案
- 交互功能增强

### 3. 导出功能
- 支持数据导出为JSON/CSV
- 图表导出为图片
- 分析报告生成