# API Endpoints Documentation

## API Overview

The NHEM backend provides a comprehensive RESTful API with the following main categories:
- **System Management** - Health checks and system control
- **Real-time Data** - Live data access and historical windows
- **ROI Management** - Region of Interest configuration and capture
- **Peak Detection** - Peak detection configuration and analysis
- **Configuration Management** - Runtime configuration management

## Base Configuration
- **Base URL**: `http://localhost:8421`
- **API Version**: 1.0.0
- **Content-Type**: `application/json`
- **Authentication**: Password-based for control commands (default: `31415`)

## 1. System Management Endpoints

### 1.1 Health Check
```http
GET /health
```

**Purpose**: System health verification
**Authentication**: None required
**Response Model**: `HealthResponse`

**Response Example**:
```json
{
  "status": "ok",
  "system": "NHEM API Server",
  "version": "1.0.0"
}
```

**Implementation**: `routes.py:health()`

---

### 1.2 System Status
```http
GET /status
```

**Purpose**: Current system state and statistics
**Authentication**: None required
**Response Model**: `StatusResponse`

**Response Example**:
```json
{
  "status": "running",
  "frame_count": 1250,
  "current_value": 127.45,
  "peak_signal": 1,
  "buffer_size": 100,
  "baseline": 120.0,
  "timestamp": "2025-12-05T12:30:45.123Z"
}
```

**Implementation**: `routes.py:status()`

---

### 1.3 System Control
```http
POST /control
Content-Type: application/x-www-form-urlencoded
```

**Purpose**: Execute system control commands
**Authentication**: Password required
**Parameters**:
- `command` (string): Control command
- `password` (string): Authentication password

**Available Commands**:
- `PEAK_SIGNAL` - Get current peak signal status
- `STATUS` - Get detailed system status
- `start_detection` - Start detection processing
- `stop_detection` - Stop detection processing
- `pause_detection` - Pause detection processing
- `resume_detection` - Resume detection processing

**Response Model**: Varies by command
- `PEAK_SIGNAL` → `PeakSignalResponse`
- `STATUS` → `ControlStatusResponse`
- Control commands → `ControlCommandResponse`

**Implementation**: `routes.py:control()`

## 2. Real-time Data Endpoints

### 2.1 Real-time Data
```http
GET /data/realtime?count=100
```

**Purpose**: Get current real-time data with ROI capture
**Authentication**: None required
**Parameters**:
- `count` (query): Number of data points (1-1000, default: 100)

**Response Model**: `RealtimeDataResponse`

**Response Example**:
```json
{
  "type": "realtime_data",
  "timestamp": "2025-12-05T12:30:45.123Z",
  "frame_count": 1250,
  "series": [
    {"t": 0.0, "value": 127.45},
    {"t": 0.022, "value": 128.12}
  ],
  "roi_data": {
    "width": 200,
    "height": 150,
    "pixels": "iVBORw0KGgoAAAANSUhEUgAA...",
    "gray_value": 127.45,
    "format": "base64"
  },
  "peak_signal": 1,
  "enhanced_peak": {
    "signal": 1,
    "color": "green",
    "confidence": 0.95,
    "threshold": 105.0,
    "in_peak_region": true,
    "frame_count": 1250
  },
  "baseline": 120.0
}
```

**Implementation**: `routes.py:realtime_data()`

---

### 2.2 Window Capture
```http
GET /data/window-capture?count=100
```

**Purpose**: Get historical data window
**Authentication**: None required
**Parameters**:
- `count` (query): Window size (50-200 frames)

**Response Model**: `WindowCaptureResponse`

**Response Example**:
```json
{
  "type": "window_capture",
  "timestamp": "2025-12-05T12:30:45.123Z",
  "window_size": 100,
  "frame_range": [1150, 1250],
  "series": [
    {"t": 0.0, "value": 125.67},
    {"t": 0.022, "value": 126.89}
  ],
  "capture_metadata": {
    "start_frame": 1150,
    "end_frame": 1250,
    "actual_frame_count": 100,
    "baseline": 120.0,
    "capture_duration": 2.2,
    "current_frame_count": 1250
  }
}
```

**Implementation**: `routes.py:window_capture()`

---

### 2.3 ROI Window Capture
```http
GET /data/roi-window-capture?count=100
```

**Purpose**: Get ROI-specific historical data
**Authentication**: None required
**Parameters**:
- `count` (query): ROI window size (50-500 frames)

**Response Model**: `RoiWindowCaptureResponse`

**Response Example**:
```json
{
  "type": "roi_window_capture",
  "timestamp": "2025-12-05T12:30:45.123Z",
  "window_size": 100,
  "roi_frame_range": [200, 300],
  "main_frame_range": [1150, 1250],
  "series": [
    {"t": 0.0, "gray_value": 127.45, "roi_index": 0},
    {"t": 0.016, "gray_value": 128.12, "roi_index": 1}
  ],
  "roi_config": {
    "x1": 1480, "y1": 480, "x2": 1580, "y2": 580,
    "width": 100, "height": 100,
    "center_x": 1530, "center_y": 530
  },
  "capture_metadata": {
    "roi_start_frame": 200,
    "roi_end_frame": 300,
    "actual_roi_frame_count": 100,
    "capture_duration": 1.6
  }
}
```

**Implementation**: `routes.py:roi_window_capture()`

---

### 2.4 ROI Window Capture with Peaks
```http
GET /data/roi-window-capture-with-peaks?count=100&threshold=105.0&margin_frames=6&difference_threshold=2.1&force_refresh=false
```

**Purpose**: Get ROI data with peak detection analysis
**Authentication**: None required
**Parameters**:
- `count` (query): Window size (50-500 frames)
- `threshold` (query): Peak detection threshold (optional)
- `margin_frames` (query): Boundary extension frames (optional)
- `difference_threshold` (query): Frame difference threshold (optional)
- `force_refresh` (query): Force cache refresh (default: false)

**Response Model**: `RoiWindowCaptureWithPeaksResponse`

**Response Example**:
```json
{
  "type": "roi_window_capture_with_peaks",
  "timestamp": "2025-12-05T12:30:45.123Z",
  "window_size": 100,
  "roi_frame_range": [200, 300],
  "main_frame_range": [1150, 1250],
  "series": [
    {"t": 0.0, "gray_value": 127.45, "roi_index": 0}
  ],
  "roi_config": {
    "x1": 1480, "y1": 480, "x2": 1580, "y2": 580
  },
  "capture_metadata": {
    "roi_start_frame": 200,
    "roi_end_frame": 300,
    "actual_roi_frame_count": 100
  },
  "peak_detection_results": {
    "green_peaks": [25, 78],
    "red_peaks": [45],
    "total_peaks": 3,
    "green_peak_count": 2,
    "red_peak_count": 1
  },
  "peak_detection_params": {
    "threshold": 105.0,
    "margin_frames": 6,
    "difference_threshold": 2.1,
    "data_points": 100
  }
}
```

**Implementation**: `routes.py:roi_window_capture_with_peaks()`

---

### 2.5 Waveform with Peaks
```http
GET /data/waveform-with-peaks?count=100&threshold=105.0&margin_frames=6&difference_threshold=2.1
```

**Purpose**: Generate waveform visualization with peak annotations
**Authentication**: None required
**Parameters**:
- `count` (query): Data points (10-500)
- `threshold` (query): Peak detection threshold (optional)
- `margin_frames` (query): Boundary extension frames (optional)
- `difference_threshold` (query): Frame difference threshold (optional)

**Response**: Custom JSON with image data

**Response Example**:
```json
{
  "success": true,
  "timestamp": "2025-12-05T12:30:45.123Z",
  "image_data": "iVBORw0KGgoAAAANSUhEUgAA...",
  "metadata": {
    "data_points": 100,
    "green_peaks": 2,
    "red_peaks": 1,
    "total_peaks": 3,
    "detection_params": {
      "threshold": 105.0,
      "margin_frames": 6,
      "difference_threshold": 2.1
    },
    "data_range": {
      "min": 95.2,
      "max": 145.8,
      "avg": 120.5
    }
  }
}
```

**Implementation**: `routes.py:waveform_with_peaks()`

## 3. ROI Management Endpoints

### 3.1 Set ROI Configuration
```http
POST /roi/config
Content-Type: application/x-www-form-urlencoded
```

**Purpose**: Configure ROI region
**Authentication**: Password required
**Parameters**:
- `x1`, `y1` (int): Top-left coordinates
- `x2`, `y2` (int): Bottom-right coordinates
- `password` (string): Authentication password

**Response Model**: `RoiConfigResponse`

**Response Example**:
```json
{
  "type": "roi_config",
  "timestamp": "2025-12-05T12:30:45.123Z",
  "config": {
    "x1": 1480,
    "y1": 480,
    "x2": 1580,
    "y2": 580
  },
  "success": true
}
```

**Implementation**: `routes.py:set_roi_config()`

---

### 3.2 Get ROI Configuration
```http
GET /roi/config
```

**Purpose**: Get current ROI configuration
**Authentication**: None required
**Response Model**: `RoiConfigResponse`

**Implementation**: `routes.py:get_roi_config()`

---

### 3.3 Manual ROI Capture (Deprecated)
```http
POST /roi/capture
Content-Type: application/x-www-form-urlencoded
```

**Purpose**: Manual ROI screenshot capture
**Authentication**: Password required
**Parameters**:
- `password` (string): Authentication password

**Response Model**: `RoiCaptureResponse`

**Note**: Use `/data/realtime` for automatic ROI capture

**Implementation**: `routes.py:capture_roi()`

---

### 3.4 ROI Frame Rate Management
```http
GET /roi/frame-rate
POST /roi/frame-rate
Content-Type: application/x-www-form-urlencoded
```

**Purpose**: Get/Set ROI capture frame rate
**Authentication**: Password required for POST
**POST Parameters**:
- `frame_rate` (int): FPS (1-60)
- `password` (string): Authentication password

**Response Model**: `RoiFrameRateResponse`

**Implementation**: `routes.py:get_roi_frame_rate()`, `routes.py:set_roi_frame_rate()`

## 4. Peak Detection Endpoints

### 4.1 Peak Detection Configuration
```http
GET /peak-detection/config
POST /peak-detection/config
Content-Type: application/x-www-form-urlencoded
```

**Purpose**: Get/Set peak detection parameters
**Authentication**: Password required for POST
**GET Response Model**: `PeakDetectionConfigResponse`

**POST Parameters**:
- `threshold` (float, optional): Detection threshold (50.0-200.0)
- `margin_frames` (int, optional): Boundary extension (1-20)
- `difference_threshold` (float, optional): Frame difference (0.1-10.0)
- `min_region_length` (int, optional): Minimum region length (1-20)

**Response Model**: `PeakDetectionConfigResponse`

**Response Example**:
```json
{
  "type": "peak_detection_config",
  "timestamp": "2025-12-05T12:30:45.123Z",
  "threshold": 105.0,
  "margin_frames": 6,
  "difference_threshold": 2.1,
  "min_region_length": 4,
  "success": true,
  "message": "Peak detection configuration updated: threshold=105.0"
}
```

**Implementation**: `routes.py:get_peak_detection_config()`, `routes.py:set_peak_detection_config()`

## 5. Data Processing Configuration

### 5.1 Data FPS Management
```http
GET /data/fps
POST /data/fps
Content-Type: application/x-www-form-urlencoded
```

**Purpose**: Get/Set data generation frame rate
**Authentication**: Password required for POST
**POST Parameters**:
- `fps` (int): Data generation FPS (10-120)
- `password` (string): Authentication password

**Response Model**: `DataFpsResponse`

**Implementation**: `routes.py:get_data_fps()`, `routes.py:set_data_fps()`

## 6. Configuration Management Endpoints

### 6.1 Get Configuration
```http
GET /config?section=peak_detection&password=31415
```

**Purpose**: Retrieve configuration information
**Authentication**: Password required
**Parameters**:
- `section` (query, optional): Configuration section name
- `password` (query): Authentication password

**Response**: Configuration JSON

**Implementation**: `routes.py:get_config()`

---

### 6.2 Update Configuration
```http
POST /config?section=peak_detection&password=31415
Content-Type: application/x-www-form-urlencoded
```

**Purpose**: Update configuration parameters
**Authentication**: Password required
**Parameters**:
- `section` (query, optional): Configuration section
- `key` (query, optional): Configuration key
- `value` (query, optional): Configuration value (JSON string)
- `config_data` (query, optional): Complete configuration (JSON string)
- `password` (query): Authentication password

**Response**: Update status

**Implementation**: `routes.py:update_config()`

---

### 6.3 Reload Configuration
```http
POST /config/reload?password=31415
```

**Purpose**: Reload configuration from file
**Authentication**: Password required
**Parameters**:
- `password` (query): Authentication password

**Response**: Reload status

**Implementation**: `routes.py:reload_config()`

---

### 6.4 Export Configuration
```http
GET /config/export?password=31415
```

**Purpose**: Export configuration as JSON
**Authentication**: Password required
**Parameters**:
- `password` (query): Authentication password

**Response**: Configuration JSON

**Implementation**: `routes.py:export_config()`

---

### 6.5 Import Configuration
```http
POST /config/import
Content-Type: application/x-www-form-urlencoded
```

**Purpose**: Import configuration from JSON
**Authentication**: Password required
**Parameters**:
- `config_json` (form): JSON configuration string
- `password` (form): Authentication password

**Response**: Import status

**Implementation**: `routes.py:import_config()`

## 7. Analysis Endpoint

### 7.1 Video Analysis
```http
POST /analyze
Content-Type: multipart/form-data
```

**Purpose**: Video analysis interface (simulated)
**Authentication**: None required
**Parameters**:
- `realtime` (form, optional): Real-time mode flag
- `duration` (form, optional): Analysis duration
- `file` (file, optional): Video file
- `roi_x`, `roi_y`, `roi_w`, `roi_h` (form, optional): ROI parameters
- `sample_fps` (form, optional): Sampling FPS

**Response Model**: `AnalyzeResponse`

**Implementation**: `routes.py:analyze()`

## Error Handling

### Standard Error Response Format
```json
{
  "type": "error",
  "timestamp": "2025-12-05T12:30:45.123Z",
  "error_code": "UNAUTHORIZED",
  "error_message": "Invalid password",
  "details": {
    "parameter": "password",
    "value": "wrong_password",
    "constraint": "Must match configured password"
  }
}
```

### Common Error Codes
- `UNAUTHORIZED` (401): Authentication failed
- `INVALID_PARAMETER` (400): Invalid request parameters
- `INVALID_ROI_COORDINATES` (400): Invalid ROI configuration
- `ROI_NOT_CONFIGURED` (400): ROI required but not configured
- `NO_DATA_AVAILABLE` (404): No data available for request
- `INTERNAL_ERROR` (500): Server internal error

## WebSocket Endpoint

### Real-time Data Streaming
```
ws://localhost:30415
```

**Purpose**: Real-time data streaming
**Authentication**: Password in initial message
**Message Format**: JSON
**Frequency**: 60 FPS

**Client Message**:
```json
{
  "type": "auth",
  "password": "31415"
}
```

**Server Response**:
```json
{
  "type": "data",
  "timestamp": "2025-12-05T12:30:45.123Z",
  "frame_count": 1250,
  "current_value": 127.45,
  "peak_signal": 1
}
```

**Implementation**: `core/socket_server.py`

## Rate Limiting and Performance

### Rate Limits
- **Control Commands**: 10 per minute per client
- **Configuration Updates**: 5 per minute per client
- **Data Requests**: No limit (optimized for real-time access)
- **WebSocket**: 60 FPS broadcasting limit

### Performance Optimizations
- **Circular Buffers**: Efficient memory usage for time-series data
- **Connection Pooling**: WebSocket client management
- **Asynchronous Processing**: Non-blocking I/O for better performance
- **Response Caching**: ROI capture results cached for short periods

This comprehensive API provides full control over the NHEM system with real-time data access, configuration management, and peak detection capabilities.