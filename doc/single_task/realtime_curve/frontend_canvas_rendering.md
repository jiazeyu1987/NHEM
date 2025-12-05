# Frontend Canvas-Based Real-time Curve Rendering Documentation

## Overview

This document covers the Canvas-based real-time curve rendering implementation in the NHEM frontend, focusing on the complete rendering pipeline from data reception to visual display.

## Architecture Overview

### File Location
- **Main File**: `D:\ProjectPackage\NHEM\front\index.html`
- **Canvas Element**: `<canvas id="waveform-chart"></canvas>` (Lines ~2000-4000)
- **Rendering Class**: `WaveformChart` (Lines ~1087-1310)
- **Update System**: Real-time polling at 50ms intervals (20 FPS)

### Core Rendering Pipeline

```
Backend API Response → Data Processing → Canvas Rendering → UI Updates
     ↓ (HTTP)          ↓ (JavaScript)      ↓ (Canvas 2D)    ↓ (DOM)
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ /data/realtime │ │   WaveformChart │ │   Canvas 2D     │ │   Chart UI      │
│ JSON Response  │ │   .draw()       │ │   Rendering     │ │   Controls      │
└─────────────────┘ └─────────────────┘ └─────────────────┘ └─────────────────┘
```

## Canvas Element Setup

### HTML Canvas Configuration

```html
<!-- Main chart canvas (around line 2000) -->
<div class="chart-container">
    <canvas id="waveform-chart" width="800" height="400"></canvas>
</div>
```

### Canvas Initialization

```javascript
// Canvas setup in WaveformChart constructor
constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d', {
        alpha: false,  // Disable alpha for better performance
        willReadFrequently: false  // Optimize for read-back operations
    });

    // Canvas dimensions
    this.width = 800;
    this.height = 400;

    // Performance optimization
    this.imageData = null;  // For double buffering if needed
    this.lastRenderTime = 0;
    this.frameCount = 0;
}
```

## Real-time Data Processing

### Data Reception and Processing

```javascript
// Real-time data update manager (around lines 3000-3500)
class RealtimeDataManager {
    constructor() {
        this.updateInterval = 50;  // 20 FPS (50ms intervals)
        this.maxDataPoints = 500;  // Maximum data points in buffer
        this.isRunning = false;
        this.errorCount = 0;
        this.maxRetries = 5;
    }

    async startUpdates() {
        if (this.isRunning) return;

        this.isRunning = true;
        console.log('Starting real-time updates at 20 FPS');

        while (this.isRunning) {
            const startTime = performance.now();

            try {
                // Fetch data from backend
                const data = await this.fetchRealtimeData();

                // Process and validate data
                if (this.validateData(data)) {
                    this.processData(data);
                    this.updateChart(data);
                    this.errorCount = 0;  // Reset error count on success
                }

            } catch (error) {
                this.handleError(error);
            }

            // Maintain precise timing
            const elapsed = performance.now() - startTime;
            const sleepTime = Math.max(0, this.updateInterval - elapsed);
            await this.sleep(sleepTime);
        }
    }

    async fetchRealtimeData() {
        const response = await fetch('/data/realtime?count=100', {
            method: 'GET',
            headers: {
                'Accept': 'application/json',
                'Cache-Control': 'no-cache'
            },
            signal: AbortSignal.timeout(5000)  // 5-second timeout
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        return await response.json();
    }
}
```

### Data Validation and Processing

```javascript
// Data validation and transformation
processData(data) {
    // Validate response structure
    if (!data || !data.series || !Array.isArray(data.series)) {
        throw new Error('Invalid data structure received');
    }

    // Transform data for chart rendering
    const transformedData = data.series.map(point => ({
        x: point.t,                    // Time value
        y: point.value,                // Signal value
        peak: point.peak_signal || null,  // Peak indicator
        timestamp: data.timestamp
    }));

    // Update application state
    appState.chartData = transformedData;
    appState.lastUpdate = Date.now();

    // Handle ROI data if available
    if (data.roi_data && data.roi_data.gray_value > 0) {
        appState.roiData = {
            grayValue: data.roi_data.gray_value,
            timestamp: data.timestamp,
            image: data.roi_data.pixels
        };
    }

    // Handle enhanced peak data
    if (data.enhanced_peak) {
        appState.enhancedPeak = data.enhanced_peak;
    }
}
```

## Canvas Rendering Implementation

### Main Drawing Function

```javascript
// Core rendering function (lines ~1087-1310)
class WaveformChart {
    draw() {
        const startTime = performance.now();

        const { width, height } = this;
        const { showGrid, showBaseline, showPoints, zoom } = appState.chartState;
        const data = appState.chartData;

        // 1. Clear canvas with VS Code dark theme background
        this.ctx.fillStyle = '#1e1e1e';
        this.ctx.fillRect(0, 0, width, height);

        // 2. Set up coordinate system
        this.setupCoordinateSystem();

        // 3. Draw grid system
        if (showGrid) {
            this.drawGrid();
        }

        // 4. Draw baseline if enabled
        if (showBaseline && appState.baseline !== undefined) {
            this.drawBaseline();
        }

        // 5. Draw main waveform
        if (data && data.length > 0) {
            this.drawWaveform(data);

            // 6. Draw peak indicators
            if (showPoints) {
                this.drawPeakIndicators(data);
            }
        }

        // 7. Draw enhanced peak annotations
        if (appState.enhancedPeak) {
            this.drawEnhancedPeakAnnotations();
        }

        // 8. Performance monitoring
        this.updatePerformanceStats(startTime);
    }

    setupCoordinateSystem() {
        // Fixed Y-axis range (0-200) centered at 100
        this.centerY = this.height / 2;
        this.scaleY = (this.height / 200) * appState.chartState.zoom;

        // X-axis mapping based on data points
        this.stepX = this.width / Math.max(appState.maxDataPoints - 1, 1);

        // Y-axis coordinate transformation (inverted for typical charts)
        this.mapY = (value) => {
            // Clamp value to valid range (0-200)
            const clampedValue = Math.max(0, Math.min(200, value));
            // Map to canvas coordinates (inverted Y-axis)
            return this.centerY - (clampedValue - 100) * this.scaleY;
        };

        // X-axis coordinate transformation
        this.mapX = (dataPointIndex) => {
            return dataPointIndex * this.stepX;
        };
    }
}
```

### Grid System Rendering

```javascript
drawGrid() {
    const { width, height } = this;
    const { mapY } = this;

    this.ctx.strokeStyle = '#404040';  # Subtle grid color
    this.ctx.lineWidth = 1;
    this.ctx.setLineDash([]);

    // Major grid lines (every 40 units)
    this.ctx.strokeStyle = '#404040';
    this.ctx.font = '10px Consolas, monospace';
    this.ctx.fillStyle = '#cccccc';

    for (let value = 0; value <= 200; value += 40) {
        const y = mapY(value);

        // Draw horizontal grid line
        this.ctx.beginPath();
        this.ctx.moveTo(0, y);
        this.ctx.lineTo(width, y);
        this.ctx.stroke();

        // Draw Y-axis labels
        this.ctx.fillText(value.toString(), 5, y + 3);
    }

    // Minor grid lines (every 20 units, dashed)
    this.ctx.strokeStyle = '#303030';
    this.ctx.setLineDash([2, 4]);

    for (let value = 20; value < 200; value += 20) {
        if (value % 40 !== 0) {  // Skip major lines
            const y = mapY(value);
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(width, y);
            this.ctx.stroke();
        }
    }

    // Reset line dash
    this.ctx.setLineDash([]);

    // Draw axis labels
    this.drawAxisLabels();
}

drawAxisLabels() {
    const { width, height } = this;
    const { mapY } = this;

    // Y-axis label
    this.ctx.save();
    this.ctx.translate(15, height / 2);
    this.ctx.rotate(-Math.PI / 2);
    this.ctx.fillStyle = '#cccccc';
    this.ctx.font = '12px Consolas, monospace';
    this.ctx.textAlign = 'center';
    this.ctx.fillText('Value', 0, 0);
    this.ctx.restore();

    // X-axis label
    this.ctx.fillStyle = '#cccccc';
    this.ctx.font = '12px Consolas, monospace';
    this.ctx.textAlign = 'center';
    this.ctx.fillText('Time (relative seconds)', width / 2, height - 5);
}
```

### Waveform Drawing

```javascript
drawWaveform(data) {
    const { width, height } = this;
    const { mapX, mapY } = this;

    // Apply antialiasing for smooth curves
    this.ctx.imageSmoothingEnabled = true;
    this.ctx.imageSmoothingQuality = 'high';

    // Main waveform styling
    this.ctx.strokeStyle = '#4ec9b0';  # VS Code success color
    this.ctx.lineWidth = 2;
    this.ctx.lineCap = 'round';
    this.ctx.lineJoin = 'round';

    // Create gradient for visual appeal
    const gradient = this.ctx.createLinearGradient(0, 0, width, 0);
    gradient.addColorStop(0, '#4ec9b0');
    gradient.addColorStop(0.5, '#569cd6');
    gradient.addColorStop(1, '#4ec9b0');
    this.ctx.strokeStyle = gradient;

    // Draw smooth curve through all data points
    this.ctx.beginPath();

    data.forEach((point, index) => {
        const x = mapX(index);
        const y = mapY(point.y);

        if (index === 0) {
            this.ctx.moveTo(x, y);
        } else {
            // Use quadratic curves for smoother appearance
            const prevPoint = data[index - 1];
            const prevX = mapX(index - 1);
            const prevY = mapY(prevPoint.y);

            // Calculate control point for smooth curve
            const controlX = (prevX + x) / 2;
            const controlY = (prevY + y) / 2;

            this.ctx.quadraticCurveTo(controlX, controlY, x, y);
        }
    });

    this.ctx.stroke();

    // Add subtle glow effect
    this.ctx.shadowColor = '#4ec9b0';
    this.ctx.shadowBlur = 10;
    this.ctx.stroke();
    this.ctx.shadowBlur = 0;

    // Reset image smoothing for other elements
    this.ctx.imageSmoothingEnabled = false;
}
```

### Peak Indicator Rendering

```javascript
drawPeakIndicators(data) {
    const { mapX, mapY } = this;

    data.forEach((point, index) => {
        // Check if this point is a peak
        if (point.peak === 1 || this.isPeakPoint(point, data, index)) {
            const x = mapX(index);
            const y = mapY(point.y);

            // Draw peak indicator circle
            this.ctx.fillStyle = '#ffffff';
            this.ctx.strokeStyle = '#ff6b6b';
            this.ctx.lineWidth = 2;

            this.ctx.beginPath();
            this.ctx.arc(x, y, 4, 0, Math.PI * 2);
            this.ctx.fill();
            this.ctx.stroke();

            // Add peak label
            this.ctx.fillStyle = '#ffffff';
            this.ctx.font = '10px Consolas, monospace';
            this.ctx.fillText('PEAK', x + 8, y - 8);
        }
    });
}

isPeakPoint(point, data, index) {
    // Peak detection logic for frontend (fallback)
    if (index === 0 || index === data.length - 1) {
        return false;
    }

    const prevPoint = data[index - 1];
    const nextPoint = data[index + 1];

    // Local peak detection (value higher than neighbors)
    return point.y > prevPoint.y && point.y > nextPoint.y && point.y > (appState.baseline + 10);
}
```

### Enhanced Peak Annotations

```javascript
drawEnhancedPeakAnnotations() {
    const peak = appState.enhancedPeak;
    if (!peak || !peak.signal) return;

    const { mapY } = this;
    const annotationX = this.width - 200;
    const annotationY = 50;

    // Draw annotation background
    this.ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
    this.ctx.fillRect(annotationX - 10, annotationY - 10, 190, 60);

    // Draw annotation border
    const borderColor = peak.color === 'green' ? '#4ec9b0' : '#ff6b6b';
    this.ctx.strokeStyle = borderColor;
    this.ctx.lineWidth = 2;
    this.ctx.strokeRect(annotationX - 10, annotationY - 10, 190, 60);

    // Draw annotation text
    this.ctx.fillStyle = '#ffffff';
    this.ctx.font = '11px Consolas, monospace';

    let yOffset = annotationY + 5;
    this.ctx.fillText(`Peak Detected`, annotationX, yOffset);
    yOffset += 15;
    this.ctx.fillText(`Color: ${peak.color}`, annotationX, yOffset);
    yOffset += 15;
    this.ctx.fillText(`Confidence: ${(peak.confidence * 100).toFixed(1)}%`, annotationX, yOffset);
    yOffset += 15;
    this.ctx.fillText(`Threshold: ${peak.threshold.toFixed(1)}`, annotationX, yOffset);
}
```

## Performance Optimization

### Frame Rate Control

```javascript
// Frame rate management system
class FrameRateController {
    constructor(targetFPS = 20) {
        this.targetFPS = targetFPS;
        this.targetFrameTime = 1000 / targetFPS;  // 50ms for 20 FPS
        this.lastFrameTime = 0;
        this.frameCount = 0;
        this.actualFPS = 0;
        this.fpsUpdateInterval = 1000;  // Update FPS every second
        this.lastFPSUpdate = 0;
    }

    shouldRender() {
        const now = performance.now();
        const timeSinceLastFrame = now - this.lastFrameTime;

        if (timeSinceLastFrame >= this.targetFrameTime) {
            this.lastFrameTime = now;
            this.frameCount++;
            this.updateFPSCounter(now);
            return true;
        }

        return false;
    }

    updateFPSCounter(now) {
        if (now - this.lastFPSUpdate >= this.fpsUpdateInterval) {
            this.actualFPS = this.frameCount;
            this.frameCount = 0;
            this.lastFPSUpdate = now;

            // Log performance warnings
            if (this.actualFPS < this.targetFPS * 0.8) {
                console.warn(`Low FPS detected: ${this.actualFPS} (target: ${this.targetFPS})`);
            }
        }
    }
}
```

### Memory Management

```javascript
// Data buffer management
class DataBuffer {
    constructor(maxSize = 500) {
        this.maxSize = maxSize;
        this.data = [];
        this.observers = [];
    }

    addDataPoint(point) {
        this.data.push(point);

        // Maintain buffer size to prevent memory leaks
        if (this.data.length > this.maxSize) {
            this.data.shift();
        }

        // Notify observers of data change
        this.notifyObservers();
    }

    clear() {
        this.data = [];
        this.notifyObservers();
    }

    getRecentData(count) {
        const startIndex = Math.max(0, this.data.length - count);
        return this.data.slice(startIndex);
    }
}
```

### Canvas Optimization

```javascript
// Canvas performance optimizations
class CanvasOptimizer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d', {
            alpha: false,  # Disable alpha channel for better performance
            willReadFrequently: false,  # Optimize if not reading pixels
            desynchronized: false  # Sync with display for smoother animation
        });

        // Performance monitoring
        this.renderTimes = [];
        this.maxRenderTimes = 60;  # Keep last 60 render times
    }

    optimizeForPerformance() {
        // Set up canvas for optimal rendering
        this.ctx.imageSmoothingEnabled = true;
        this.ctx.imageSmoothingQuality = 'high';

        // Use hardware acceleration hints
        this.canvas.style.willChange = 'transform';

        // Optimize compositing
        this.ctx.globalCompositeOperation = 'source-over';

        // Pre-allocate path objects if needed
        this.path = new Path2D();
    }

    measureRenderTime(renderFunction) {
        const startTime = performance.now();

        renderFunction();

        const endTime = performance.now();
        const renderTime = endTime - startTime;

        // Store render time for performance analysis
        this.renderTimes.push(renderTime);
        if (this.renderTimes.length > this.maxRenderTimes) {
            this.renderTimes.shift();
        }

        return renderTime;
    }

    getAverageRenderTime() {
        if (this.renderTimes.length === 0) return 0;

        const sum = this.renderTimes.reduce((a, b) => a + b, 0);
        return sum / this.renderTimes.length;
    }
}
```

## Visual Styling and Themes

### VS Code Theme Integration

```css
/* VS Code-style dark theme colors */
:root {
    --bg-primary: #1e1e1e;
    --bg-secondary: #252526;
    --text-primary: #cccccc;
    --text-secondary: #969696;
    --accent-success: #4ec9b0;
    --accent-warning: #dcdcaa;
    --accent-error: #f48771;
    --accent-info: #569cd6;
    --border-color: #404040;
    --grid-major: #404040;
    --grid-minor: #303030;
}

.chart-container {
    background-color: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    padding: 16px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
}

#waveform-chart {
    display: block;
    background-color: var(--bg-primary);
    border-radius: 2px;
}
```

### Color Schemes

```javascript
// Color palette configuration
const chartColors = {
    primary: '#4ec9b0',      # VS Code success color
    secondary: '#569cd6',    # VS Code info color
    accent: '#dcdcaa',       # VS Code warning color
    error: '#f48771',        # VS Code error color
    background: '#1e1e1e',   # VS Code background
    grid: '#404040',         # Subtle grid
    text: '#cccccc',         # Primary text
    peak: '#ffffff',         # Peak indicators
    peakBorder: '#ff6b6b'    # Peak border
};

// Dynamic color based on peak type
function getPeakColor(peakType) {
    switch (peakType) {
        case 'green': return '#4ec9b0';
        case 'red': return '#ff6b6b';
        case 'yellow': return '#dcdcaa';
        default: return '#ffffff';
    }
}
```

## Error Handling and Resilience

### Data Error Recovery

```javascript
class ErrorRecoveryManager {
    constructor(chart) {
        this.chart = chart;
        this.consecutiveErrors = 0;
        this.maxConsecutiveErrors = 5;
        this.fallbackData = this.generateFallbackData();
    }

    async handleDataError(error) {
        this.consecutiveErrors++;

        console.error(`Chart data error ${this.consecutiveErrors}/${this.maxConsecutiveErrors}:`, error);

        if (this.consecutiveErrors >= this.maxConsecutiveErrors) {
            // Switch to fallback data
            this.useFallbackData();
            this.showErrorMessage('连接到服务器失败，显示模拟数据');
        } else {
            // Exponential backoff
            const delay = Math.min(1000 * Math.pow(2, this.consecutiveErrors), 10000);
            setTimeout(() => this.retryDataFetch(), delay);
        }
    }

    generateFallbackData() {
        // Generate synthetic data for offline/demo mode
        const data = [];
        for (let i = 0; i < 100; i++) {
            data.push({
                x: i * 0.05,
                y: 100 + 10 * Math.sin(i * 0.1),
                peak: Math.random() > 0.95 ? 1 : null
            });
        }
        return data;
    }

    useFallbackData() {
        appState.chartData = this.fallbackData;
        appState.fallbackMode = true;
        this.chart.draw();
    }
}
```

### Canvas Error Handling

```javascript
// Safe rendering with error boundaries
class SafeChartRenderer {
    constructor(chart) {
        this.chart = chart;
        this.errorCount = 0;
        this.maxErrors = 3;
    }

    safeDraw() {
        try {
            this.chart.draw();
            this.errorCount = 0;  // Reset error count on success
        } catch (error) {
            this.errorCount++;
            console.error(`Canvas rendering error ${this.errorCount}/${this.maxErrors}:`, error);

            if (this.errorCount >= this.maxErrors) {
                // Disable rendering to prevent continuous errors
                this.disableRendering();
                this.showCriticalError('图表渲染失败，请刷新页面');
            } else {
                // Try to clear and redraw
                this.clearAndRedraw();
            }
        }
    }

    clearAndRedraw() {
        try {
            // Clear canvas
            this.chart.ctx.clearRect(0, 0, this.chart.width, this.chart.height);

            // Draw error message
            this.chart.ctx.fillStyle = '#f48771';
            this.chart.ctx.font = '14px Consolas, monospace';
            this.chart.ctx.textAlign = 'center';
            this.chart.ctx.fillText('渲染错误，正在重试...', this.chart.width / 2, this.chart.height / 2);
        } catch (error) {
            console.error('Even clear and redraw failed:', error);
        }
    }
}
```

## User Interaction Features

### Interactive Controls

```javascript
// Chart interaction handlers
class ChartInteractions {
    constructor(chart) {
        this.chart = chart;
        this.setupEventListeners();
        this.interactionState = {
            isDragging: false,
            dragStartX: 0,
            dragStartY: 0,
            isZooming: false
        };
    }

    setupEventListeners() {
        // Mouse events
        this.chart.canvas.addEventListener('mousedown', this.handleMouseDown.bind(this));
        this.chart.canvas.addEventListener('mousemove', this.handleMouseMove.bind(this));
        this.chart.canvas.addEventListener('mouseup', this.handleMouseUp.bind(this));
        this.chart.canvas.addEventListener('wheel', this.handleWheel.bind(this));

        // Touch events for mobile
        this.chart.canvas.addEventListener('touchstart', this.handleTouchStart.bind(this));
        this.chart.canvas.addEventListener('touchmove', this.handleTouchMove.bind(this));
        this.chart.canvas.addEventListener('touchend', this.handleTouchEnd.bind(this));
    }

    handleMouseDown(event) {
        const rect = this.chart.canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;

        // Check if clicking on a data point for details
        const dataPoint = this.findNearestDataPoint(x, y);
        if (dataPoint) {
            this.showDataPointDetails(dataPoint);
        }

        // Start drag for panning
        this.interactionState.isDragging = true;
        this.interactionState.dragStartX = x;
        this.interactionState.dragStartY = y;
    }

    handleWheel(event) {
        event.preventDefault();

        // Zoom functionality
        const zoomDelta = event.deltaY > 0 ? 0.9 : 1.1;
        const newZoom = Math.max(0.5, Math.min(5.0, appState.chartState.zoom * zoomDelta));

        appState.chartState.zoom = newZoom;
        this.chart.draw();
    }

    findNearestDataPoint(mouseX, mouseY) {
        const threshold = 10;  # pixels
        const { mapX, mapY } = this.chart;

        for (let i = 0; i < appState.chartData.length; i++) {
            const point = appState.chartData[i];
            const pointX = mapX(i);
            const pointY = mapY(point.y);

            const distance = Math.sqrt(Math.pow(mouseX - pointX, 2) + Math.pow(mouseY - pointY, 2));

            if (distance < threshold) {
                return { ...point, index: i, screenX: pointX, screenY: pointY };
            }
        }

        return null;
    }
}
```

### Tooltip and Information Display

```javascript
// Tooltip system for data points
class TooltipManager {
    constructor(chart) {
        this.chart = chart;
        this.tooltip = null;
        this.createTooltip();
    }

    createTooltip() {
        this.tooltip = document.createElement('div');
        this.tooltip.className = 'chart-tooltip';
        this.tooltip.style.cssText = `
            position: absolute;
            background: rgba(30, 30, 30, 0.95);
            border: 1px solid #404040;
            border-radius: 4px;
            padding: 8px 12px;
            color: #cccccc;
            font: 11px Consolas, monospace;
            pointer-events: none;
            z-index: 1000;
            display: none;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        `;
        document.body.appendChild(this.tooltip);
    }

    showTooltip(dataPoint, mouseX, mouseY) {
        this.tooltip.innerHTML = `
            <div><strong>Time:</strong> ${dataPoint.x.toFixed(2)}s</div>
            <div><strong>Value:</strong> ${dataPoint.y.toFixed(2)}</div>
            <div><strong>Index:</strong> ${dataPoint.index}</div>
            ${dataPoint.peak ? '<div><strong>Peak:</strong> Yes</div>' : ''}
        `;

        this.tooltip.style.display = 'block';
        this.tooltip.style.left = (mouseX + 15) + 'px';
        this.tooltip.style.top = (mouseY - 30) + 'px';
    }

    hideTooltip() {
        this.tooltip.style.display = 'none';
    }
}
```

This Canvas-based real-time curve rendering system provides a high-performance, visually appealing interface for displaying NHEM signal data. The implementation emphasizes smooth rendering, VS Code theme consistency, and comprehensive error handling while maintaining 20 FPS update rates for real-time monitoring.