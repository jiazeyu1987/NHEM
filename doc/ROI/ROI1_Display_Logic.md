# ROI1 Display Logic Documentation

## Overview

ROI1 (Region of Interest 1) is the primary large-area capture component of the NHEM dual ROI system. It serves as the main capture region for user visualization and as the source for ROI2 extraction. This document details the complete implementation of ROI1 display logic across the NHEM three-tier architecture.

## Architecture Overview

```
Screen Capture → ROI1 Processing → Multi-Client Display
     ↓              ↓                   ↓
  (1100x500)   (Base64 Encoding)   (Web/Desktop)
     ↓              ↓                   ↓
  PIL Image    → ROI1 Data Model → Frontend/Client
```

## 1. Backend Implementation

### 1.1 ROI Capture Service (`backend/app/core/roi_capture.py`)

#### Core Capture Method

**Location**: `capture_roi_internal()` (lines 134-195)

```python
def capture_roi_internal(self, roi_config: RoiConfig) -> Optional[RoiData]:
    """
    Captures ROI1 from screen and processes for display

    Process:
    1. Screen capture using PIL.ImageGrab
    2. Region extraction based on ROI configuration
    3. Image resizing for consistent display (200x150)
    4. Gray value calculation
    5. Base64 encoding for web transmission
    6. Error handling with fallback mechanisms
    """
```

#### Key Processing Steps

1. **Screen Capture**:
   ```python
   # Capture screen region using PIL
   screenshot = ImageGrab.grab(bbox=(roi_config.x1, roi_config.y1, roi_config.x2, roi_config.y2))
   ```

2. **Image Processing**:
   ```python
   # Convert to RGB if necessary
   if screenshot.mode != 'RGB':
       screenshot = screenshot.convert('RGB')

   # Calculate display dimensions (maintain aspect ratio)
   display_size = (200, 150)  # Standard ROI display size
   screenshot_resized = screenshot.resize(display_size, Image.Resampling.LANCZOS)
   ```

3. **Gray Value Calculation**:
   ```python
   # Convert to grayscale for analysis
   screenshot_gray = screenshot_resized.convert('L')
   gray_array = np.array(screenshot_gray)
   gray_value = float(np.mean(gray_array))
   ```

4. **Base64 Encoding**:
   ```python
   # Convert to base64 for web transmission
   buffer = io.BytesIO()
   screenshot_resized.save(buffer, format='PNG', optimize=True)
   screenshot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
   ```

#### Data Model (`backend/app/models.py`)

```python
class RoiData(BaseModel):
    """ROI1 data structure for display and analysis"""
    width: int           # Display width (200)
    height: int          # Display height (150)
    pixels: str          # Base64 encoded PNG image
    gray_value: float    # Average gray value (0-255)
    timestamp: datetime  # Capture timestamp

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
```

### 1.2 Configuration Management

#### Default Configuration (`backend/app/fem_config.json`)

```json
{
  "roi_capture": {
    "frame_rate": 4,
    "update_interval": 0.5,
    "default_config": {
      "x1": 480,
      "y1": 80,
      "x2": 1580,
      "y2": 580,
      "display_width": 200,
      "display_height": 150
    }
  }
}
```

#### Runtime Configuration Updates

- **API Endpoint**: `/config/update`
- **Hot Reloading**: Configuration changes applied immediately
- **Validation**: Coordinate bounds checking before capture

### 1.3 Data Storage and Caching

#### Circular Buffer Storage

**Location**: `backend/app/core/data_store.py` (lines 150-180)

```python
class DataStore:
    def __init__(self):
        self.roi_frames = deque(maxlen=500)  # Store last 500 ROI frames
        self.roi_lock = threading.Lock()     # Thread-safe access

    def add_roi_frame(self, frame_count: int, roi_data: RoiData):
        """Add ROI frame with frame count synchronization"""
        with self.roi_lock:
            self.roi_frames.append({
                'frame_count': frame_count,
                'roi_data': roi_data,
                'timestamp': datetime.now()
            })
```

#### Caching Strategy

- **Time-based Caching**: ROI capture result cached for 250ms
- **Configuration Detection**: Cache invalidated on ROI configuration changes
- **Performance Optimization**: Prevents excessive screen capture operations

### 1.4 API Layer Implementation

#### ROI Data Endpoint

**Location**: `backend/app/api/routes.py` (lines 890-920)

```python
@router.get("/data/realtime", response_model=RealtimeDataResponse)
async def realtime_data(count: int = Query(100, ge=1, le=1000)):
    """
    Get real-time data including ROI1 information

    Response Structure:
    {
        "roi1_data": {
            "width": 200,
            "height": 150,
            "pixels": "base64_encoded_image",
            "gray_value": 128.5,
            "timestamp": "2025-12-08T..."
        },
        "signal_data": [...],
        "peak_data": [...]
    }
}
```

## 2. Frontend Display Implementation

### 2.1 Canvas-based Rendering (`front/index.html`)

#### ROI Renderer Class

**Location**: Lines 2029-2075

```javascript
class RoiRenderer {
    constructor(canvasElement) {
        this.canvas = canvasElement;
        this.ctx = canvasElement.getContext('2d');
        this.offscreenCanvas = document.createElement('canvas');
        this.offscreenCtx = this.offscreenCanvas.getContext('2d');

        // Double buffering setup
        this.offscreenCanvas.width = 200;
        this.offscreenCanvas.height = 150;
    }

    render(roiData) {
        if (roiData.format === 'base64') {
            this.preloadAndRender(roiData.pixels);
        } else {
            this.drawPlaceholder(roiData.message || "ROI截图失败");
        }
    }
}
```

#### Image Preloading and Rendering

```javascript
preloadAndRender(base64Pixels) {
    const img = new Image();
    img.onload = () => {
        // Clear and render to offscreen canvas first
        this.offscreenCtx.clearRect(0, 0, 200, 150);
        this.offscreenCtx.drawImage(img, 0, 0, 200, 150);

        // Copy to main canvas (double buffering)
        this.ctx.clearRect(0, 0, 200, 150);
        this.ctx.drawImage(this.offscreenCanvas, 0, 0);
    };

    img.onerror = () => {
        this.drawPlaceholder("图像加载失败");
    };

    img.src = 'data:image/png;base64,' + base64Pixels;
}
```

#### State Management

```javascript
// ROI display states
const ROI_DISPLAY_STATES = {
    NOT_CONFIGURED: 'not_configured',
    CAPTURE_FAILED: 'capture_failed',
    SUCCESS: 'success',
    LOADING: 'loading'
};

// State tracking in appState
appState.roiDisplayState = ROI_DISPLAY_STATES.NOT_CONFIGURED;
```

### 2.2 API Integration

#### Data Fetching

**Location**: ApiService class (lines 1800-1850)

```javascript
class ApiService {
    async fetchRealtimeData(count = 100) {
        try {
            const response = await fetch(`${this.baseURL}/data/realtime?count=${count}`);
            const data = await response.json();

            if (data.roi1_data) {
                appState.roi1Data = data.roi1_data;
                this.updateRoiDisplay(data.roi1_data);
            }
        } catch (error) {
            console.error('ROI1 data fetch failed:', error);
            this.handleRoiError(error);
        }
    }
}
```

#### Update Mechanism

- **Polling Interval**: Configurable (default 50ms for 20 FPS)
- **Automatic Updates**: Continuous polling when system is running
- **Error Recovery**: Automatic retry with exponential backoff

### 2.3 Error Handling and Placeholder States

#### Placeholder Rendering

```javascript
drawPlaceholder(message) {
    this.ctx.fillStyle = '#1e1e1e';
    this.ctx.fillRect(0, 0, 200, 150);

    this.ctx.fillStyle = '#888888';
    this.ctx.font = '12px Consolas, monospace';
    this.ctx.textAlign = 'center';
    this.ctx.textBaseline = 'middle';
    this.ctx.fillText(message, 100, 75);
}
```

#### Error State Management

```javascript
updateRoiDisplay(roiData) {
    const roiCanvas = document.getElementById('roi-canvas');
    const roiRenderer = window.roiRenderer;

    if (!roiData || !roiData.pixels) {
        appState.roiDisplayState = ROI_DISPLAY_STATES.CAPTURE_FAILED;
        roiRenderer.drawPlaceholder("ROI截图失败");
        return;
    }

    appState.roiDisplayState = ROI_DISPLAY_STATES.SUCCESS;
    roiRenderer.render(roiData);
}
```

## 3. Python Client Implementation

### 3.1 Dual ROI Integration (`python_client/http_realtime_client.py`)

#### ROI1 Display Setup

**Location**: Lines 300-350

```python
def setup_roi_display(self):
    """Setup ROI1 display components"""
    # Create ROI display frame
    self.roi_frame = ttk.LabelFrame(self.main_frame, text="ROI 实时监控")

    # ROI1 display (left side for large region)
    self.roi_label_left = ttk.Label(self.roi_frame, text="ROI1 (大区域)")
    self.roi_canvas_left = tk.Canvas(
        self.roi_frame,
        width=250,
        height=188,  # Scaled from 200x150
        bg='black'
    )

    # Layout
    self.roi_label_left.grid(row=0, column=0, padx=5, pady=5)
    self.roi_canvas_left.grid(row=1, column=0, padx=5, pady=5)
```

### 3.2 ROI Data Processing

#### Callback-based Updates

**Location**: Lines 777-851

```python
def _handle_roi_update_callback(self, roi1_data, roi2_data=None):
    """Handle ROI data updates from polling loop"""
    try:
        # Process ROI1 data
        if roi1_data and roi1_data.get('pixels'):
            self._update_roi1_display(roi1_data)
        else:
            self._display_roi1_error("ROI1数据获取失败")

    except Exception as e:
        logger.error(f"ROI update callback failed: {e}")
        self._display_roi1_error(f"ROI处理错误: {str(e)}")
```

#### ROI1 Display Update

**Location**: Lines 944-990

```python
def _update_roi1_display(self, roi1_data):
    """Update ROI1 display with new data"""
    try:
        # Base64 decode
        image_data = base64.b64decode(roi1_data['pixels'])
        pil_image = Image.open(io.BytesIO(image_data))

        # Resize for display
        display_image = pil_image.resize((250, 188), Image.Resampling.LANCZOS)

        # Convert to PhotoImage for Tkinter
        photo = ImageTk.PhotoImage(display_image)

        # Update canvas
        self.roi_canvas_left.delete("all")
        self.roi_canvas_left.create_image(125, 94, image=photo)

        # Store reference to prevent garbage collection
        self.roi1_photo_ref = photo

        # Update info label
        gray_value = roi1_data.get('gray_value', 0)
        timestamp = roi1_data.get('timestamp', '')
        info_text = f"灰度值: {gray_value:.1f} | 时间: {timestamp}"

        self.roi_info_label.config(text=info_text)

    except Exception as e:
        logger.error(f"ROI1 display update failed: {e}")
        self._display_roi1_error(f"ROI1显示错误: {str(e)}")
```

### 3.3 Error Handling

```python
def _display_roi1_error(self, error_message):
    """Display ROI1 error state"""
    self.roi_canvas_left.delete("all")
    self.roi_canvas_left.create_text(
        125, 94,
        text=error_message,
        fill='red',
        font=('Arial', 10)
    )
```

## 4. Performance Characteristics

### 4.1 Processing Performance

- **Capture Rate**: 4 FPS (configurable)
- **Image Processing**: <50ms per frame
- **Memory Usage**: ~2MB for ROI buffer
- **Network Transfer**: ~30KB per ROI frame (base64 encoded)

### 4.2 Display Performance

- **Frontend Rendering**: 20 FPS canvas updates
- **Python Client**: Real-time Tkinter updates
- **Double Buffering**: Prevents flicker in frontend
- **Image Caching**: Reduces decode overhead

## 5. Configuration Parameters

### 5.1 Capture Configuration

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `frame_rate` | 4 FPS | 1-30 | ROI capture frame rate |
| `display_width` | 200 | 100-400 | ROI display width |
| `display_height` | 150 | 75-300 | ROI display height |
| `jpeg_quality` | 85 | 50-100 | Image compression quality |

### 5.2 Display Configuration

| Parameter | Frontend | Python Client | Description |
|-----------|----------|---------------|-------------|
| Canvas Size | 200x150 | 250x188 | Display dimensions |
| Update Rate | 20 FPS | Real-time | Refresh frequency |
| Color Format | RGB | RGB | Image color format |

## 6. Error Handling Strategy

### 6.1 Backend Error Handling

```python
# Capture failure handling
if screenshot is None:
    logger.error("Screen capture failed")
    return RoiData(
        width=0, height=0, pixels="",
        gray_value=0, timestamp=datetime.now()
    )

# Configuration validation
if roi_config.x2 <= roi_config.x1 or roi_config.y2 <= roi_config.y1:
    raise ValueError("Invalid ROI configuration")
```

### 6.2 Frontend Error Handling

- **Network Errors**: Automatic retry with exponential backoff
- **Image Decode Errors**: Placeholder display
- **Canvas Errors**: Fallback to text display

### 6.3 Python Client Error Handling

- **Base64 Decode Errors**: Error message display
- **Image Load Errors**: Fallback to placeholder text
- **Network Errors**: Connection status indicator

## 7. Synchronization and Timing

### 7.1 Frame Synchronization

- **Frame Count Linking**: ROI frames linked to main signal frame counts
- **Timestamp Correlation**: Precise timing for data analysis
- **Buffer Management**: Circular buffer prevents memory overflow

### 7.2 Display Timing

- **Frontend**: 20 FPS updates (50ms intervals)
- **Python Client**: Event-driven updates
- **Backend**: 4 FPS capture (250ms intervals)

## 8. Debugging and Monitoring

### 8.1 Logging Strategy

```python
# Backend logging
logger.info(f"ROI1 captured: {roi_config.x1},{roi_config.y1} → {roi_config.x2},{roi_config.y2}")
logger.debug(f"ROI1 processing time: {processing_time:.2f}ms")

# Frontend logging
console.log(`ROI1 display updated: ${roiData.width}x${roiData.height}`);

# Python client logging
logger.info(f"ROI1 display updated: gray_value={gray_value:.2f}")
```

### 8.2 Performance Monitoring

- **Capture Timing**: Track screen capture duration
- **Processing Metrics**: Monitor image resize and encoding performance
- **Display Performance**: Canvas rendering time measurement
- **Memory Usage**: ROI buffer memory consumption tracking

## 9. Integration Points

### 9.1 ROI2 Extraction Source

ROI1 serves as the source for ROI2 extraction:
- **Center Calculation**: ROI2 extracted from ROI1 center
- **Coordinate Transformation**: ROI1 coordinates → ROI2 coordinates
- **Quality Dependency**: ROI2 quality depends on ROI1 capture quality

### 9.2 Peak Detection Integration

- **Gray Value Input**: ROI1 gray value used for peak detection
- **Visualization**: ROI1 provides visual context for peak events
- **Timing Correlation**: ROI1 frames synchronized with peak detection data

This comprehensive documentation covers the complete ROI1 display logic implementation across all components of the NHEM system, providing detailed technical guidance for development and maintenance.