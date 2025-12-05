# Frontend Architecture Documentation

## Overview

The NHEM frontend is a single-page web application built with vanilla JavaScript, HTML5, and CSS3. It provides a professional real-time monitoring interface with VS Code-style dark theme and high-performance Canvas-based charting.

## Technology Stack

### Core Technologies
- **HTML5**: Modern semantic markup and Canvas API
- **CSS3**: Advanced styling with animations and transitions
- **JavaScript (ES6+)**: Modern JavaScript with async/await, classes, and modules
- **Canvas API**: High-performance 2D graphics for real-time plotting
- **Fetch API**: Modern HTTP requests for backend communication

### Architecture Pattern
- **Single Page Application (SPA)**: All functionality in one HTML file
- **Event-Driven Architecture**: Real-time updates through polling
- **Component-Based Design**: Modular JavaScript classes
- **Configuration-Driven**: External JSON configuration

## Project Structure

```
D:\ProjectPackage\NHEM\front\
├── index.html                 # Single-page application (135KB)
├── config.json               # Frontend configuration
├── README.md                 # Frontend documentation
└── assets/                   # Static assets (if present)
    ├── icons/                # UI icons and images
    └── styles/               # Additional CSS (if split)
```

## File Architecture

### index.html - The Complete Application

The entire frontend application is contained within `index.html`, which includes:

```html
<!DOCTYPE html>
<html>
<head>
    <!-- Meta tags, title, favicon -->
    <!-- Embedded CSS styling -->
    <!-- VS Code-style dark theme -->
</head>
<body>
    <!-- HTML structure -->
    <div id="app-container">
        <header>Controls and status</header>
        <main>Charts and visualizations</main>
        <aside>Configuration panels</aside>
    </div>

    <!-- Embedded JavaScript -->
    <script>
        // All application logic
        // Classes, utilities, and main application code
    </script>
</body>
</html>
```

### Configuration System

#### config.json
```json
{
  "server": {
    "host": "localhost",
    "api_port": 8421,
    "enable_cors": true
  },
  "display": {
    "fps": 45,
    "buffer_size": 100,
    "y_axis_min": 0,
    "y_axis_max": 200,
    "chart_width": 800,
    "chart_height": 400
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
  },
  "ui": {
    "theme": "dark",
    "language": "zh-CN",
    "auto_start": false
  }
}
```

## Core Architecture Components

### 1. Application Class Hierarchy

```javascript
// Main Application Controller
class NHEMFrontend {
    constructor() {
        this.config = new ConfigManager();
        this.api = new APIClient(this.config);
        this.charts = new ChartManager();
        this.controls = new ControlPanel();
        this.status = new StatusManager();
    }

    async initialize() {
        await this.loadConfiguration();
        this.setupEventListeners();
        this.startRealtimeUpdates();
    }
}

// Configuration Management
class ConfigManager {
    constructor() {
        this.config = {};
        this.observers = [];
    }

    load() { /* Load from config.json */ }
    save() { /* Save to localStorage */ }
    get(key) { /* Get configuration value */ }
    set(key, value) { /* Update and notify */ }
}

// API Communication
class APIClient {
    constructor(config) {
        this.baseURL = config.get('server.api_url');
        this.timeout = 5000;
    }

    async request(endpoint, options = {}) {
        // HTTP request implementation
        // Error handling and retry logic
    }

    async getRealtimeData(count = 100) {
        return this.request(`/data/realtime?count=${count}`);
    }

    async sendCommand(command, password) {
        return this.request('/control', {
            method: 'POST',
            body: new FormData(/* ... */)
        });
    }
}
```

### 2. Chart Rendering System

```javascript
// Main Chart Manager
class ChartManager {
    constructor() {
        this.canvases = new Map();
        this.contexts = new Map();
        this.dataBuffers = new Map();
    }

    createChart(containerId, options) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');

        this.canvases.set(containerId, canvas);
        this.contexts.set(containerId, ctx);
        this.dataBuffers.set(containerId, []);

        return new RealtimeChart(canvas, ctx, options);
    }
}

// Real-time Chart Implementation
class RealtimeChart {
    constructor(canvas, ctx, options) {
        this.canvas = canvas;
        this.ctx = ctx;
        this.options = options;
        this.data = [];
        this.maxPoints = options.maxPoints || 100;
    }

    addDataPoint(timestamp, value) {
        this.data.push({ timestamp, value });

        // Maintain buffer size
        if (this.data.length > this.maxPoints) {
            this.data.shift();
        }

        this.render();
    }

    render() {
        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw grid
        this.drawGrid();

        // Draw axes
        this.drawAxes();

        // Draw data
        this.drawData();

        // Draw peaks
        this.drawPeaks();

        // Draw ROI indicator
        this.drawROI();
    }
}
```

### 3. UI Component System

```javascript
// Control Panel Manager
class ControlPanel {
    constructor() {
        this.elements = new Map();
        this.callbacks = new Map();
        this.setupControls();
    }

    setupControls() {
        // System controls
        this.createButton('start-btn', '开始分析', this.startDetection);
        this.createButton('stop-btn', '停止分析', this.stopDetection);
        this.createButton('pause-btn', '暂停', this.pauseDetection);

        // ROI controls
        this.createROIControls();

        // Parameter controls
        this.createParameterControls();

        // Setup event listeners
        this.bindEvents();
    }

    createButton(id, text, callback) {
        const button = document.createElement('button');
        button.id = id;
        button.textContent = text;
        button.addEventListener('click', callback);

        this.elements.set(id, button);
        return button;
    }

    updateStatus(status, message) {
        const statusElement = document.getElementById('status');
        statusElement.textContent = message;
        statusElement.className = `status ${status}`;
    }
}

// Status Manager
class StatusManager {
    constructor() {
        this.statusElement = document.getElementById('system-status');
        this.metricsElement = document.getElementById('system-metrics');
        this.connectionElement = document.getElementById('connection-status');
    }

    updateSystemStatus(data) {
        this.statusElement.textContent = data.status;
        this.updateMetrics(data);
        this.updateConnectionStatus(data.connected);
    }

    updateMetrics(data) {
        const metrics = `
            FPS: ${data.fps} |
            Frames: ${data.frame_count} |
            Buffer: ${data.buffer_size} |
            Baseline: ${data.baseline.toFixed(2)}
        `;
        this.metricsElement.textContent = metrics;
    }
}
```

## Data Flow Architecture

### 1. Real-time Data Updates

```javascript
// Real-time Update Manager
class RealtimeUpdateManager {
    constructor(apiClient, chartManager) {
        this.api = apiClient;
        this.charts = chartManager;
        this.updateInterval = 50; // 20 FPS
        this.isRunning = false;
        this.lastUpdate = 0;
    }

    async start() {
        this.isRunning = true;
        this.updateLoop();
    }

    async stop() {
        this.isRunning = false;
    }

    async updateLoop() {
        while (this.isRunning) {
            const startTime = performance.now();

            try {
                const data = await this.api.getRealtimeData();
                this.processData(data);

                // Update charts
                this.charts.updateCharts(data);

                // Update UI
                this.updateUI(data);

            } catch (error) {
                console.error('Update failed:', error);
                this.handleError(error);
            }

            // Maintain 20 FPS
            const elapsed = performance.now() - startTime;
            const sleepTime = Math.max(0, this.updateInterval - elapsed);

            await this.sleep(sleepTime);
        }
    }

    processData(data) {
        // Validate data
        if (!data.series || !Array.isArray(data.series)) {
            throw new Error('Invalid data format');
        }

        // Process time series data
        data.series.forEach(point => {
            this.charts.addDataPoint('main-chart', point.t, point.value);
        });

        // Process ROI data if available
        if (data.roi_data && data.roi_data.gray_value) {
            this.charts.addDataPoint('roi-chart', data.timestamp, data.roi_data.gray_value);
            this.updateROIDisplay(data.roi_data);
        }

        // Process peak data
        if (data.peak_signal !== null) {
            this.handlePeakDetection(data.peak_signal, data.enhanced_peak);
        }
    }
}
```

### 2. Event System

```javascript
// Event Bus for Component Communication
class EventBus {
    constructor() {
        this.listeners = new Map();
    }

    on(event, callback) {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, []);
        }
        this.listeners.get(event).push(callback);
    }

    emit(event, data) {
        if (this.listeners.has(event)) {
            this.listeners.get(event).forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    console.error(`Event handler error for ${event}:`, error);
                }
            });
        }
    }

    off(event, callback) {
        if (this.listeners.has(event)) {
            const callbacks = this.listeners.get(event);
            const index = callbacks.indexOf(callback);
            if (index > -1) {
                callbacks.splice(index, 1);
            }
        }
    }
}

// Usage in main application
const eventBus = new EventBus();

// Listen for system events
eventBus.on('system:started', () => {
    controlPanel.updateStatus('success', '系统已启动');
});

eventBus.on('system:stopped', () => {
    controlPanel.updateStatus('info', '系统已停止');
});

eventBus.on('peak:detected', (peakData) => {
    charts.highlightPeak(peakData);
    ui.showPeakNotification(peakData);
});
```

## Canvas Rendering Architecture

### 1. High-Performance Chart Rendering

```javascript
// Optimized Canvas Renderer
class CanvasRenderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.backBuffer = document.createElement('canvas');
        this.backCtx = this.backBuffer.getContext('2d');

        // Set up device pixel ratio for sharp rendering
        this.setupHighDPIDisplay();
    }

    setupHighDPIDisplay() {
        const dpr = window.devicePixelRatio || 1;
        const rect = this.canvas.getBoundingClientRect();

        this.canvas.width = rect.width * dpr;
        this.canvas.height = rect.height * dpr;

        this.ctx.scale(dpr, dpr);

        this.backBuffer.width = rect.width * dpr;
        this.backBuffer.height = rect.height * dpr;

        this.backCtx.scale(dpr, dpr);
    }

    render(data) {
        // Render to back buffer first
        this.renderToBackBuffer(data);

        // Copy to main canvas (faster than direct rendering)
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.drawImage(this.backBuffer, 0, 0);
    }

    renderToBackBuffer(data) {
        const ctx = this.backCtx;
        const width = this.canvas.width;
        const height = this.canvas.height;

        // Clear with dark background
        ctx.fillStyle = '#1e1e1e';
        ctx.fillRect(0, 0, width, height);

        // Draw grid
        this.drawGrid(ctx, width, height);

        // Draw axes
        this.drawAxes(ctx, width, height);

        // Draw data line
        this.drawDataLine(ctx, data, width, height);

        // Draw peaks
        this.drawPeaks(ctx, data.peaks, width, height);
    }
}
```

### 2. Data Visualization

```javascript
// Data Line Rendering
drawDataLine(ctx, data, width, height) {
    if (!data || data.length < 2) return;

    ctx.strokeStyle = '#4fc3f7';
    ctx.lineWidth = 2;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    ctx.beginPath();

    const xScale = width / (data.length - 1);
    const yMin = Math.min(...data.map(d => d.value));
    const yMax = Math.max(...data.map(d => d.value));
    const yRange = yMax - yMin || 1;

    data.forEach((point, index) => {
        const x = index * xScale;
        const y = height - ((point.value - yMin) / yRange) * height;

        if (index === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });

    ctx.stroke();

    // Add glow effect
    ctx.shadowColor = '#4fc3f7';
    ctx.shadowBlur = 10;
    ctx.stroke();
    ctx.shadowBlur = 0;
}

// Peak Visualization
drawPeaks(ctx, peaks, width, height) {
    peaks.forEach(peak => {
        const x = (peak.index / this.data.length) * width;
        const y = height - ((peak.value - this.yMin) / this.yRange) * height;

        // Peak marker
        ctx.fillStyle = peak.color === 'green' ? '#4caf50' : '#f44336';
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, Math.PI * 2);
        ctx.fill();

        // Peak label
        ctx.fillStyle = '#ffffff';
        ctx.font = '12px monospace';
        ctx.fillText(`Peak ${peak.value.toFixed(1)}`, x + 10, y - 10);
    });
}
```

## UI/UX Architecture

### 1. VS Code-style Theme

```css
/* Main theme implementation */
:root {
    /* VS Code Dark Theme Colors */
    --background: #1e1e1e;
    --foreground: #d4d4d4;
    --panel-background: #252526;
    --border: #3e3e42;
    --accent: #007acc;
    --success: #4caf50;
    --warning: #ff9800;
    --error: #f44336;

    /* Chart colors */
    --chart-line: #4fc3f7;
    --chart-grid: #3e3e42;
    --chart-axis: #808080;
    --chart-peak-green: #4caf50;
    --chart-peak-red: #f44336;
}

body {
    background-color: var(--background);
    color: var(--foreground);
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
    margin: 0;
    padding: 20px;
    overflow: hidden;
}

/* Component styling */
.control-panel {
    background-color: var(--panel-background);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 16px;
    margin-bottom: 16px;
}

button {
    background-color: var(--accent);
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 4px;
    cursor: pointer;
    font-family: inherit;
    font-size: 14px;
    transition: background-color 0.2s;
}

button:hover {
    background-color: #005a9e;
}

button:disabled {
    background-color: var(--border);
    cursor: not-allowed;
}

/* Status indicators */
.status {
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: bold;
    text-transform: uppercase;
}

.status.running {
    background-color: var(--success);
    color: white;
}

.status.stopped {
    background-color: var(--warning);
    color: white;
}

.status.error {
    background-color: var(--error);
    color: white;
}
```

### 2. Responsive Design

```css
/* Responsive layout */
.app-container {
    display: grid;
    grid-template-columns: 300px 1fr;
    grid-template-rows: auto 1fr auto;
    gap: 16px;
    height: 100vh;
    max-width: 1600px;
    margin: 0 auto;
}

.header {
    grid-column: 1 / -1;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px;
    background-color: var(--panel-background);
    border: 1px solid var(--border);
    border-radius: 4px;
}

.sidebar {
    background-color: var(--panel-background);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 16px;
    overflow-y: auto;
}

.main-content {
    display: flex;
    flex-direction: column;
    gap: 16px;
}

.footer {
    grid-column: 1 / -1;
    background-color: var(--panel-background);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 12px;
    text-align: center;
    font-size: 12px;
}

/* Mobile responsive */
@media (max-width: 768px) {
    .app-container {
        grid-template-columns: 1fr;
        grid-template-rows: auto auto 1fr auto;
    }

    .sidebar {
        order: 3;
    }

    .main-content {
        order: 2;
    }

    .header {
        order: 1;
    }

    .footer {
        order: 4;
    }
}
```

## Performance Optimizations

### 1. Rendering Optimizations

```javascript
// Performance monitoring
class PerformanceMonitor {
    constructor() {
        this.frameCount = 0;
        this.lastTime = performance.now();
        this.fps = 0;
        this.frameTimeBuffer = [];
    }

    tick() {
        const now = performance.now();
        const delta = now - this.lastTime;

        this.frameTimeBuffer.push(delta);
        if (this.frameTimeBuffer.length > 60) {
            this.frameTimeBuffer.shift();
        }

        this.frameCount++;

        // Update FPS every second
        if (now - this.lastTime >= 1000) {
            this.fps = this.frameCount;
            this.frameCount = 0;
            this.lastTime = now;

            this.reportPerformance();
        }
    }

    reportPerformance() {
        const avgFrameTime = this.frameTimeBuffer.reduce((a, b) => a + b, 0) / this.frameTimeBuffer.length;
        const maxFrameTime = Math.max(...this.frameTimeBuffer);

        console.log(`Performance: ${this.fps} FPS, Avg: ${avgFrameTime.toFixed(2)}ms, Max: ${maxFrameTime.toFixed(2)}ms`);
    }
}

// Animation frame optimization
class AnimationScheduler {
    constructor() {
        this.callbacks = new Set();
        this.isRunning = false;
    }

    add(callback) {
        this.callbacks.add(callback);

        if (!this.isRunning) {
            this.start();
        }
    }

    remove(callback) {
        this.callbacks.delete(callback);

        if (this.callbacks.size === 0) {
            this.stop();
        }
    }

    start() {
        this.isRunning = true;
        this.scheduleNextFrame();
    }

    stop() {
        this.isRunning = false;
        if (this.frameId) {
            cancelAnimationFrame(this.frameId);
        }
    }

    scheduleNextFrame() {
        if (!this.isRunning) return;

        this.frameId = requestAnimationFrame(() => {
            const now = performance.now();

            // Execute all callbacks
            this.callbacks.forEach(callback => {
                try {
                    callback(now);
                } catch (error) {
                    console.error('Animation callback error:', error);
                }
            });

            this.scheduleNextFrame();
        });
    }
}
```

### 2. Memory Management

```javascript
// Data buffer management
class DataBuffer {
    constructor(maxSize = 1000) {
        this.maxSize = maxSize;
        this.data = [];
        this.observers = [];
    }

    add(item) {
        this.data.push(item);

        // Maintain buffer size
        if (this.data.length > this.maxSize) {
            this.data.shift();
        }

        // Notify observers
        this.notifyObservers();
    }

    clear() {
        this.data = [];
        this.notifyObservers();
    }

    addObserver(callback) {
        this.observers.push(callback);
    }

    removeObserver(callback) {
        const index = this.observers.indexOf(callback);
        if (index > -1) {
            this.observers.splice(index, 1);
        }
    }

    notifyObservers() {
        this.observers.forEach(callback => {
            try {
                callback(this.data);
            } catch (error) {
                console.error('Observer error:', error);
            }
        });
    }
}

// Canvas cleanup
class CanvasManager {
    constructor() {
        this.canvases = new Map();
        this.contexts = new Map();
    }

    createCanvas(id, width, height) {
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;

        const ctx = canvas.getContext('2d');

        this.canvases.set(id, canvas);
        this.contexts.set(id, ctx);

        return canvas;
    }

    destroyCanvas(id) {
        const canvas = this.canvases.get(id);
        if (canvas) {
            // Clear canvas content
            const ctx = this.contexts.get(id);
            if (ctx) {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
            }

            // Remove from DOM
            if (canvas.parentNode) {
                canvas.parentNode.removeChild(canvas);
            }

            // Clear references
            this.canvases.delete(id);
            this.contexts.delete(id);
        }
    }

    cleanup() {
        // Destroy all canvases
        for (const id of this.canvases.keys()) {
            this.destroyCanvas(id);
        }
    }
}
```

## Error Handling and Debugging

### 1. Error Boundary Pattern

```javascript
// Error boundary for UI components
class ErrorBoundary {
    constructor(component) {
        this.component = component;
        this.originalMethods = new Map();
        this.setupErrorHandling();
    }

    setupErrorHandling() {
        // Wrap component methods with error handling
        Object.getOwnPropertyNames(this.component.constructor.prototype)
            .filter(name => typeof this.component[name] === 'function')
            .forEach(name => {
                if (name !== 'constructor') {
                    this.wrapMethod(name);
                }
            });
    }

    wrapMethod(methodName) {
        const originalMethod = this.component[methodName];
        this.originalMethods.set(methodName, originalMethod);

        this.component[methodName] = (...args) => {
            try {
                return originalMethod.apply(this.component, args);
            } catch (error) {
                this.handleError(error, methodName);
                return null;
            }
        };
    }

    handleError(error, methodName) {
        console.error(`Error in ${methodName}:`, error);

        // Show user-friendly error message
        this.showErrorMessage(`操作失败: ${error.message}`);

        // Log detailed error for debugging
        this.logError(error, methodName);
    }

    showErrorMessage(message) {
        const errorElement = document.getElementById('error-message');
        if (errorElement) {
            errorElement.textContent = message;
            errorElement.style.display = 'block';

            // Auto-hide after 5 seconds
            setTimeout(() => {
                errorElement.style.display = 'none';
            }, 5000);
        }
    }

    logError(error, methodName) {
        const errorData = {
            timestamp: new Date().toISOString(),
            method: methodName,
            message: error.message,
            stack: error.stack,
            userAgent: navigator.userAgent,
            url: window.location.href
        };

        // Send to error logging service (if available)
        if (window.errorLogger) {
            window.errorLogger.log(errorData);
        }
    }
}
```

### 2. Debug Mode

```javascript
// Debug utilities
class DebugUtils {
    constructor() {
        this.enabled = this.getParameterByName('debug') === 'true';
        this.logLevels = ['error', 'warn', 'info', 'debug'];
        this.currentLevel = this.getParameterByName('log') || 'info';
    }

    getParameterByName(name) {
        const url = new URL(window.location.href);
        return url.searchParams.get(name);
    }

    log(level, message, data) {
        if (!this.enabled) return;

        const levelIndex = this.logLevels.indexOf(level);
        const currentIndex = this.logLevels.indexOf(this.currentLevel);

        if (levelIndex <= currentIndex) {
            const timestamp = new Date().toISOString();
            const logMessage = `[${timestamp}] [${level.toUpperCase()}] ${message}`;

            switch (level) {
                case 'error':
                    console.error(logMessage, data);
                    break;
                case 'warn':
                    console.warn(logMessage, data);
                    break;
                case 'info':
                    console.info(logMessage, data);
                    break;
                case 'debug':
                    console.log(logMessage, data);
                    break;
            }
        }
    }

    debug(message, data) {
        this.log('debug', message, data);
    }

    info(message, data) {
        this.log('info', message, data);
    }

    warn(message, data) {
        this.log('warn', message, data);
    }

    error(message, data) {
        this.log('error', message, data);
    }

    // Performance monitoring
    time(label) {
        if (this.enabled) {
            console.time(label);
        }
    }

    timeEnd(label) {
        if (this.enabled) {
            console.timeEnd(label);
        }
    }

    // Memory usage
    getMemoryUsage() {
        if (performance.memory) {
            return {
                used: performance.memory.usedJSHeapSize,
                total: performance.memory.totalJSHeapSize,
                limit: performance.memory.jsHeapSizeLimit
            };
        }
        return null;
    }

    logMemoryUsage() {
        const memory = this.getMemoryUsage();
        if (memory) {
            this.debug('Memory Usage', {
                used: `${(memory.used / 1024 / 1024).toFixed(2)} MB`,
                total: `${(memory.total / 1024 / 1024).toFixed(2)} MB`,
                limit: `${(memory.limit / 1024 / 1024).toFixed(2)} MB`
            });
        }
    }
}

// Global debug instance
const debug = new DebugUtils();
```

## Deployment and Build Process

### 1. Development Setup

```bash
# Simple file serving
python -m http.server 3000

# Or with Node.js
npx http-server -p 3000 -c-1

# Or with PHP (if available)
php -S localhost:3000
```

### 2. Production Optimization

```javascript
// Build optimization utilities
class BuildOptimizer {
    static optimizeHTML(html) {
        // Minify HTML
        return html
            .replace(/>\s+</g, '><')
            .replace(/\s+/g, ' ')
            .trim();
    }

    static optimizeCSS(css) {
        // Minify CSS
        return css
            .replace(/\/\*[\s\S]*?\*\//g, '') // Remove comments
            .replace(/\s+/g, ' ') // Collapse whitespace
            .replace(/;\s*}/g, '}') // Remove unnecessary semicolons
            .trim();
    }

    static optimizeJS(js) {
        // Basic JavaScript minification
        return js
            .replace(/\/\*[\s\S]*?\*\//g, '') // Remove block comments
            .replace(/\/\/.*$/gm, '') // Remove line comments
            .replace(/\s+/g, ' ') // Collapse whitespace
            .trim();
    }

    static async buildProduction() {
        const response = await fetch('index.html');
        let html = await response.text();

        // Extract and optimize CSS
        const cssMatch = html.match(/<style>([\s\S]*?)<\/style>/);
        if (cssMatch) {
            const optimizedCSS = this.optimizeCSS(cssMatch[1]);
            html = html.replace(cssMatch[0], `<style>${optimizedCSS}</style>`);
        }

        // Extract and optimize JavaScript
        const jsMatch = html.match(/<script>([\s\S]*?)<\/script>/);
        if (jsMatch) {
            const optimizedJS = this.optimizeJS(jsMatch[1]);
            html = html.replace(jsMatch[0], `<script>${optimizedJS}</script>`);
        }

        // Optimize HTML
        html = this.optimizeHTML(html);

        return html;
    }
}
```

## Testing Architecture

### 1. Unit Testing Framework

```javascript
// Simple test framework
class TestFramework {
    constructor() {
        this.tests = [];
        this.results = [];
    }

    test(name, testFunction) {
        this.tests.push({ name, testFunction });
    }

    async run() {
        console.log('Running tests...');

        for (const test of this.tests) {
            try {
                await test.testFunction();
                this.results.push({ name: test.name, passed: true });
                console.log(`✓ ${test.name}`);
            } catch (error) {
                this.results.push({ name: test.name, passed: false, error: error.message });
                console.log(`✗ ${test.name}: ${error.message}`);
            }
        }

        this.reportResults();
    }

    reportResults() {
        const passed = this.results.filter(r => r.passed).length;
        const total = this.results.length;

        console.log(`\nTests: ${passed}/${total} passed`);

        if (passed === total) {
            console.log('All tests passed!');
        } else {
            console.log('Some tests failed:');
            this.results.filter(r => !r.passed).forEach(result => {
                console.log(`  - ${result.name}: ${result.error}`);
            });
        }
    }

    assert(condition, message) {
        if (!condition) {
            throw new Error(message || 'Assertion failed');
        }
    }

    assertEqual(actual, expected, message) {
        if (actual !== expected) {
            throw new Error(message || `Expected ${expected}, got ${actual}`);
        }
    }
}

// Example tests
const tests = new TestFramework();

tests.test('APIClient initialization', () => {
    const config = { 'server.api_url': 'http://localhost:8421' };
    const client = new APIClient(config);

    tests.assertEqual(client.baseURL, 'http://localhost:8421/api');
    tests.assert(client.timeout > 0, 'Timeout should be positive');
});

tests.test('ChartManager buffer management', () => {
    const chartManager = new ChartManager();
    const chart = chartManager.createChart('test-chart', { maxPoints: 5 });

    // Add more points than buffer size
    for (let i = 0; i < 10; i++) {
        chart.addDataPoint(i, i * 10);
    }

    tests.assertEqual(chart.data.length, 5, 'Buffer should maintain max size');
    tests.assertEqual(chart.data[0].value, 50, 'Should keep recent data');
});

// Run tests if in debug mode
if (debug.enabled) {
    tests.run();
}
```

This frontend architecture provides a robust, high-performance web application with professional UI/UX, efficient real-time data handling, and comprehensive error handling. The modular design makes it maintainable and extensible while the optimization techniques ensure smooth 20 FPS updates even with large datasets.