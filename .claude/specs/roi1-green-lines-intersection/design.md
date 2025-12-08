# Design Document

## Steering Document Alignment

This design aligns with established NHEM technical patterns and standards:

### Technical Standards Reference
- **Architecture Patterns**: Follows NHEM layered architecture (API → Core → Data) as defined in technical documentation
- **Configuration Management**: Integrates with existing fem_config.json + NHEM_* environment variables + ConfigManager pattern
- **API Design**: Extends existing `/api/roi/` namespace maintaining RESTful patterns and password authentication (default: 31415)
- **Data Models**: Uses Pydantic models matching existing patterns in `app/models.py`
- **Threading Model**: Compatible with existing 45 FPS DataProcessor and 60 FPS DataBroadcaster architecture
- **Memory Management**: Follows circular buffer patterns and thread-safe data structures

### Code Standards Alignment
- **Python Style**: PEP 8 compliance with type hints throughout
- **Error Handling**: Structured error responses matching existing NHEM API patterns
- **Logging**: Uses existing NHEM logging configuration with structured format
- **Testing**: Compatible with existing manual testing patterns via `/docs` and `/health` endpoints

### Security Standards
- **Authentication**: Inherits existing NHEM password-based authentication system
- **Input Validation**: Uses Pydantic models for API validation matching existing patterns
- **Data Privacy**: No persistent image storage, following existing NHEM data handling practices

## Overview

This design document provides the technical architecture and implementation approach for the ROI1 Green Lines Intersection Detection feature. The design leverages existing NHEM system components while adding new computer vision capabilities for anatomical landmark detection.

## Code Reuse Analysis

### Existing NHEM Components to Leverage

#### Core Infrastructure Components
- **ConfigManager** (`app/core/config_manager.py`): Runtime configuration management and persistence
  - **Reuse Pattern**: Extend existing configuration schema to include `line_detection` section
  - **Integration Points**: `get_config_manager()`, `update_config()`, JSON persistence mechanisms
  - **Benefits**: Automatic environment variable support (NHEM_ prefix), hot-reload capabilities

- **DataStore** (`app/core/data_store.py`): Thread-safe circular buffer implementation
  - **Reuse Pattern**: Create separate circular buffer for line intersection results
  - **Integration Points**: Existing lock mechanisms, circular buffer data structures
  - **Benefits**: Thread safety, automatic data aging, performance optimization

- **ROICapture Service** (`app/core/roi_capture.py`): Real-time screenshot capture service
  - **Reuse Pattern**: Extend existing ROI capture to provide ROI1 images for line detection
  - **Integration Points**: Existing screen capture mechanisms, base64 encoding, 4 FPS timing
  - **Benefits**: Proven screen capture integration, existing error handling

#### API Infrastructure
- **Route Handlers** (`app/api/routes.py`): Existing FastAPI endpoint patterns
  - **Reuse Pattern**: Add new endpoints following existing `/api/roi/` namespace patterns
  - **Integration Points**: Existing authentication decorators, error response handlers, Pydantic validation
  - **Benefits**: Consistent API behavior, automatic documentation via `/docs`

- **Authentication System**: Password-based control command authentication
  - **Reuse Pattern**: Inherit existing password validation (default: 31415)
  - **Integration Points**: Existing password checking middleware, control command patterns
  - **Benefits**: Security consistency, no new authentication mechanisms needed

#### Data Models
- **Pydantic Models** (`app/models.py`): Existing data validation patterns
  - **Reuse Pattern**: Create new models extending existing patterns (RoiData, BaseSuccessResponse)
  - **Integration Points**: Existing validation decorators, JSON serialization, response formatting
  - **Benefits**: Type safety, automatic validation, consistent API responses

#### Configuration System
- **AppConfig** (`app/config.py`): Pydantic-based settings management
  - **Reuse Pattern**: Extend AppConfig class with LineDetectionConfig nested settings
  - **Integration Points**: Existing environment variable loading, JSON configuration integration
  - **Benefits**: Automatic validation, environment variable override support

### Python Client Components to Extend

#### GUI Framework (`http_realtime_client.py`)
- **RealtimePlotter**: Existing matplotlib real-time plotting
  - **Reuse Pattern**: Add line detection overlay capabilities to existing plotting framework
  - **Integration Points**: Existing figure management, animation loop, canvas updates
  - **Benefits**: Proven real-time performance, existing UI integration

- **API Client**: Existing HTTP request handling with authentication
  - **Reuse Pattern**: Extend existing API client to handle new line intersection endpoints
  - **Integration Points**: Existing error handling, retry logic, connection management
  - **Benefits**: Network resilience, existing authentication flow

- **Configuration Management**: Local configuration loading and management
  - **Reuse Pattern**: Extend existing local_config_loader.py patterns
  - **Integration Points**: JSON configuration loading, default value handling
  - **Benefits**: Consistent configuration behavior across client and server

### Components to Build New

#### Computer Vision Processing
- **LineIntersectionDetector**: Core OpenCV processing engine
  - **Reason**: New functionality not present in existing codebase
  - **Dependencies**: OpenCV, numpy for image processing
  - **Integration**: Called by ROICapture service, results stored in DataStore

#### Python Client UI Components
- **LineDetectionWidget**: Matplotlib/Tkinter visualization widget
  - **Reason**: New UI requirements for line detection visualization
  - **Dependencies**: matplotlib, tkinter, existing GUI framework
  - **Integration**: Embeds in existing Python client main window

#### API Extensions
- **Line Intersection API Endpoints**: New REST endpoints for line detection control
  - **Reason**: New API functionality required for Python client integration
  - **Dependencies**: FastAPI, existing authentication system
  - **Integration**: Extends existing `/api/roi/` namespace

## Architecture

### System Context

The ROI1 Green Lines Intersection Detection feature integrates into the existing NHEM architecture as follows:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Python Client │◄──►│  FastAPI Backend │◄──►│  OpenCV Engine  │
│ run_realtime_   │    │  /api/roi/line-  │    │  HSV/Canny/     │
│ client.py       │    │  intersection*   │    │  Hough Processing│
│                 │    │                  │    │                 │
│ • matplotlib UI │    │ • REST APIs      │    │ • Color Segment │
│ • Tkinter       │    │ • Auth/NHESPW    │    │ • Edge Detection│
│ • Status Display│    │ • Data Store     │    │ • Line Math     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         ▲                       ▲                       ▲
         │                       │                       │
         │              ┌────────┴────────┐              │
         │              │                 │              │
         └──────────────┤  Existing NHEM  │──────────────┘
                        │  Infrastructure │
                        │                 │
                        │ • DataProcessor │
                        │ • DataStore     │
                        │ • ROICapture    │
                        │ • ConfigManager │
                        └─────────────────┘
```

### Component Design

#### 1. Backend Integration

**New API Endpoints** (extend existing `/api/roi/` namespace):
```
POST   /api/roi/line-intersection/enable     Enable line detection
POST   /api/roi/line-intersection/disable    Disable line detection
POST   /api/roi/line-intersection            Manual detection request
GET    /data/realtime/enhanced?include_line_intersection=true  Enhanced data
```

**Configuration Extension** (extends `AppConfig`):
```python
class LineDetectionConfig(BaseSettings):
    enabled: bool = False
    hsv_green_lower: Tuple[int, int, int] = (40, 50, 50)
    hsv_green_upper: Tuple[int, int, int] = (80, 255, 255)
    canny_low_threshold: int = 25
    canny_high_threshold: int = 80
    hough_threshold: int = 50
    hough_min_line_length: int = 15
    hough_max_line_gap: int = 8
    min_confidence: float = 0.4
    roi_processing_mode: str = "roi1_only"
```

#### 2. OpenCV Processing Engine

**Core Class Design**:
```python
class LineIntersectionDetector:
    """Handles green line detection and intersection calculation"""

    def __init__(self, config: LineDetectionConfig):
        self.config = config
        self._last_result = None
        self._cache_timeout = 0.1  # 100ms cache

    def detect_intersection(self, roi1_image: np.ndarray) -> Optional[LineIntersectionResult]:
        """Main detection pipeline with caching"""

    def _extract_green_mask(self, image: np.ndarray) -> np.ndarray:
        """HSV color space segmentation"""

    def _detect_lines(self, mask: np.ndarray) -> List[np.ndarray]:
        """Canny edge detection + Hough transformation"""

    def _calculate_intersection(self, lines: List[np.ndarray]) -> Optional[Tuple[float, float]]:
        """Virtual intersection point calculation"""

    def _calculate_confidence(self, lines: List[np.ndarray], intersection: Tuple[float, float], edge_quality: float, temporal_stability: float) -> float:
        """Confidence scoring algorithm matching requirements specification"""
        if len(lines) < 2 or not intersection:
            return 0.0

        # Calculate line lengths
        line1_length = np.sqrt((lines[0][2] - lines[0][0])**2 + (lines[0][3] - lines[0][1])**2)
        line2_length = np.sqrt((lines[1][2] - lines[1][0])**2 + (lines[1][3] - lines[1][1])**2)

        # Apply exact formula from requirements
        confidence = (line1_length + line2_length) / 200 * edge_quality * 0.9 + temporal_stability * 0.1

        return min(max(confidence, 0.0), 1.0)  # Clamp to [0.0, 1.0]
```

#### 3. Python Client Integration

**UI Component Design**:
```python
class LineDetectionWidget:
    """Matplotlib/Tkinter widget for line detection visualization"""

    def __init__(self, parent_frame: tk.Widget):
        self.parent_frame = parent_frame
        self.setup_matplotlib_canvas()
        self.setup_control_buttons()
        self.setup_status_display()

    def setup_matplotlib_canvas(self):
        """Create matplotlib figure for ROI1 visualization"""

    def setup_control_buttons(self):
        """Enable/disable toggle and manual refresh buttons with Chinese text"""
        # Enable/Disable toggle button
        self.toggle_btn = tk.Button(
            self.parent_frame,
            text="启用检测",  # Initial state text
            command=self.toggle_detection,
            width=12,
            height=2
        )
        self.toggle_btn.pack(side=tk.LEFT, padx=5)

        # Manual refresh button
        self.refresh_btn = tk.Button(
            self.parent_frame,
            text="手动检测",
            command=self.manual_detection,
            width=12,
            height=2
        )
        self.refresh_btn.pack(side=tk.LEFT, padx=5)

    def setup_status_display(self):
        """Status display with Chinese text and color coding"""
        self.status_label = tk.Label(
            self.parent_frame,
            text="线条相交点: 未启用",
            font=('Arial', 12),
            fg='gray'  # Initial gray color
        )
        self.status_label.pack(side=tk.BOTTOM, pady=5)

    def update_display(self, roi1_data: RoiData, intersection_result: Optional[LineIntersectionResult]):
        """Update matplotlib visualization and status display with Chinese text"""
        if not self.line_detection_enabled:
            # Disabled state - gray text
            self.status_label.config(text="线条相交点: 未启用", fg='gray')
            self.toggle_btn.config(text="启用检测", state=tk.NORMAL)
        elif intersection_result and intersection_result.has_intersection:
            # Success state - green text with coordinates
            x, y = intersection_result.intersection
            confidence = intersection_result.confidence * 100
            status_text = f"线条相交点: 已识别 ({int(x)}, {int(y)}) 置信度: {confidence:.1f}%"
            self.status_label.config(text=status_text, fg='green')
            self.toggle_btn.config(text="禁用检测", state=tk.NORMAL)
        elif self.line_detection_enabled:
            # Enabled but no detection - yellow/orange text
            self.status_label.config(text="线条相交点: 已启用 - 未识别", fg='orange')
            self.toggle_btn.config(text="禁用检测", state=tk.NORMAL)
        else:
            # Error state - red text
            error_msg = intersection_result.error_message if intersection_result else "未知错误"
            self.status_label.config(text=f"线条相交点: 检测错误: {error_msg}", fg='red')
            self.toggle_btn.config(text="启用检测", state=tk.NORMAL)

    def toggle_detection(self):
        """Handle toggle button with loading states"""
        # Show loading state during API call
        self.toggle_btn.config(text="处理中...", state=tk.DISABLED)
        self.refresh_btn.config(state=tk.DISABLED)

        try:
            if self.line_detection_enabled:
                # Disable detection
                response = self.api_call('POST', '/api/roi/line-intersection/disable')
                if response.get('success'):
                    self.line_detection_enabled = False
            else:
                # Enable detection
                response = self.api_call('POST', '/api/roi/line-intersection/enable')
                if response.get('success'):
                    self.line_detection_enabled = True
        finally:
            # Restore button states
            self.refresh_btn.config(state=tk.NORMAL)
            self.update_display(None, None)  # Update button text and status

    def manual_detection(self):
        """Handle manual refresh with loading state"""
        self.refresh_btn.config(text="检测中...", state=tk.DISABLED)
        self.toggle_btn.config(state=tk.DISABLED)

        try:
            # Trigger immediate detection
            response = self.api_call('POST', '/api/roi/line-intersection')
            # Process response and update display
        finally:
            # Restore button state
            self.refresh_btn.config(text="手动检测", state=tk.NORMAL)
            self.toggle_btn.config(state=tk.NORMAL)

    def draw_intersection(self, ax: plt.Axes, intersection: LineIntersectionResult):
        """Draw detected lines, intersection point, and confidence information"""
```

**Integration with Existing Client**:
```python
class RealtimeClient:  # Extend existing class
    def __init__(self):
        # Existing initialization...
        self.line_detection_widget = LineDetectionWidget(self.main_frame)
        self.line_detection_enabled = False

    def fetch_enhanced_data(self):
        """Extended to include line_intersection parameter"""
        params = {
            'count': 100,
            'include_line_intersection': str(self.line_detection_enabled).lower()
        }
        # ... existing fetch logic

    def toggle_line_detection(self):
        """Handle enable/disable with API calls"""
        if self.line_detection_enabled:
            response = self.api_call('POST', '/api/roi/line-intersection/disable')
        else:
            response = self.api_call('POST', '/api/roi/line-intersection/enable')
        # ... handle response and update UI
```

## Data Flow

### Processing Pipeline

```
ROI Capture (4 FPS) → ROI1/ROI2 Separation → ROI1 OpenCV Processing → Intersection Calculation → Result Cache → API Response → Python Client Display

Detailed Flow:
1. ROI Capture Service captures dual ROI data (ROI1 + ROI2)
2. ROI Processing Separation:
   - ROI1: Routes to LineIntersectionDetector for computer vision processing
   - ROI2: Routes to existing grayscale analysis pipeline (UNAFFECTED)
3. ROI1 LineIntersectionDetector processes image:
   - HSV conversion → Green mask extraction
   - Morphological operations → Noise reduction
   - Canny edge detection → Edge identification
   - Hough line transform → Line segment detection
   - Line filtering → Remove horizontal/vertical lines
   - Intersection calculation → Virtual intersection point
   - Confidence scoring → Quality assessment using formula: `(line1_length + line2_length) / 200 * edge_quality * 0.9 + temporal_stability * 0.1`
4. ROI2 Grayscale Analysis (existing pipeline unchanged):
   - Gray value calculation → Peak detection → Data storage
5. Result cached for 100ms to prevent redundant processing
6. Enhanced API response includes LineIntersectionResult for ROI1 only
7. Python client updates matplotlib visualization and status display

### ROI1-Only Processing Isolation

**Strict Processing Separation:**
- **Data Flow**: ROI1 and ROI2 data streams remain completely separate after capture
- **Threading**: Line detection runs in separate thread from ROI2 grayscale analysis
- **Memory**: No shared memory buffers between ROI1 and ROI2 processing
- **Configuration**: Independent configuration parameters for each ROI processing mode
- **API Responses**: Line intersection results only included for ROI1 data
- **Error Isolation**: Line detection errors cannot affect ROI2 grayscale processing

**Processing Guarantees:**
- ROI2 grayscale analysis maintains existing performance and reliability
- Line detection processing time (max 300ms) cannot delay ROI2 updates
- Memory usage for line detection is separate from ROI2 buffer allocations
- Configuration changes to line detection do not affect ROI2 parameters

### Data Models

**LineIntersectionResult** (extends existing `models.py`):
```python
class LineIntersectionResult(BaseModel):
    """Line detection and intersection result"""
    has_intersection: bool
    intersection: Optional[Tuple[float, float]]  # (x, y) coordinates
    confidence: float  # 0.0 to 1.0
    detected_lines: List[Tuple[Tuple[int, int, int, int], float]]  # ((x1,y1,x2,y2), confidence)
    processing_time_ms: float
    error_message: Optional[str] = None

    class Config:
        json_encoders = {
            # Handle numpy arrays for JSON serialization
        }
```

**Enhanced Realtime Response** (extends `DualRealtimeDataResponse`):
```python
class EnhancedRealtimeDataResponse(DualRealtimeDataResponse):
    """Extended realtime data with line intersection results"""
    line_intersection: Optional[LineIntersectionResult] = None
```

## Implementation Details

### OpenCV Algorithm Implementation

#### 1. Color Segmentation
```python
def _extract_green_mask(self, image: np.ndarray) -> np.ndarray:
    """Extract green pixels using HSV color space"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array(self.config.hsv_green_lower)
    upper_green = np.array(self.config.hsv_green_upper)
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Morphological operations to remove noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask
```

#### 2. Line Detection
```python
def _detect_lines(self, mask: np.ndarray) -> List[np.ndarray]:
    """Detect line segments using Canny + Hough"""
    edges = cv2.Canny(mask, self.config.canny_low_threshold, self.config.canny_high_threshold)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=self.config.hough_threshold,
        minLineLength=self.config.hough_min_line_length,
        maxLineGap=self.config.hough_max_line_gap
    )

    if lines is None:
        return []

    # Filter out horizontal and vertical lines
    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
        if 10 < abs(angle) < 80:  # Exclude near-horizontal/vertical lines
            filtered_lines.append(line[0])

    return filtered_lines
```

#### 3. Intersection Calculation
```python
def _calculate_intersection(self, lines: List[np.ndarray]) -> Optional[Tuple[float, float]]:
    """Calculate virtual intersection of two best lines"""
    if len(lines) < 2:
        return None

    # Select two best lines (longest and most non-parallel)
    best_intersection = None
    best_confidence = 0.0

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            intersection = self._line_intersection(lines[i], lines[j])
            if intersection:
                confidence = self._calculate_line_pair_confidence(lines[i], lines[j])
                if confidence > best_confidence:
                    best_intersection = intersection
                    best_confidence = confidence

    return best_intersection

def _line_intersection(self, line1: np.ndarray, line2: np.ndarray) -> Optional[Tuple[float, float]]:
    """Calculate intersection point of two lines (extended)"""
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if abs(denom) < 0.01:  # Lines are parallel
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)

    return (x, y)
```

### Python Client Visualization

#### Matplotlib Rendering
```python
def draw_intersection(self, ax: plt.Axes, intersection: LineIntersectionResult):
    """Draw intersection visualization on matplotlib axes"""
    x, y = intersection.intersection
    confidence = intersection.confidence

    # Color based on confidence
    color = 'red' if confidence > 0.7 else 'orange'

    # Draw detected lines
    for line_data, _ in intersection.detected_lines:
        x1, y1, x2, y2 = line_data
        ax.plot([x1, x2], [y1, y2], color='lime', linewidth=2, alpha=0.7)

    # Draw intersection point - outer circle
    outer_circle = patches.Circle((x, y), 6, fill=False, edgecolor=color, linewidth=2)
    ax.add_patch(outer_circle)

    # Inner filled circle
    inner_circle = patches.Circle((x, y), 3, fill=True, facecolor=color, edgecolor=color)
    ax.add_patch(inner_circle)

    # Crosshair lines
    ax.plot([x-8, x+8], [y, y], color=color, linewidth=2)
    ax.plot([x, x], [y-8, y+8], color=color, linewidth=2)

    # Coordinate and confidence text
    text = f"({int(x)}, {int(y)})\nc:{confidence:.2f}"
    ax.text(x+10, y-10, text, color='white', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
```

## Configuration Management

### Backend Configuration Extension

**fem_config.json Addition**:
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
    "min_confidence": 0.4,
    "roi_processing_mode": "roi1_only",
    "cache_timeout_ms": 100,
    "max_processing_time_ms": 300
  }
}
```

### Runtime Configuration API

**New Configuration Endpoints**:
```
GET    /config/line-detection          Get current line detection config
PUT    /config/line-detection          Update line detection config
POST   /config/line-detection/reset    Reset to defaults
```

## Performance Considerations

### Optimization Strategies

1. **Caching**: 100ms result cache to prevent redundant processing
2. **Memory Management**: Reuse numpy arrays and OpenCV objects
3. **Early Termination**: Skip processing if confidence < minimum threshold
4. **Async Processing**: Non-blocking line detection in ROI capture service
5. **Resource Limits**: Maximum processing time of 300ms enforced

### Resource Monitoring

```python
class LineDetectionMetrics:
    """Monitor performance and resource usage"""

    def __init__(self):
        self.processing_times = deque(maxlen=100)
        self.success_rate = RollingCounter(window=1000)
        self.memory_usage = 0

    def record_processing(self, duration_ms: float, success: bool):
        """Record processing metrics"""
        self.processing_times.append(duration_ms)
        self.success_rate.add(1 if success else 0)

    def get_performance_stats(self) -> dict:
        """Get current performance statistics"""
        return {
            "avg_processing_time_ms": np.mean(self.processing_times),
            "success_rate_percent": self.success_rate.rate() * 100,
            "memory_usage_mb": self.memory_usage / (1024 * 1024)
        }
```

## Error Handling

### Error Categories and Responses

1. **Input Validation Errors**:
   - Invalid ROI coordinates → 400 Bad Request
   - Malformed image data → 422 Unprocessable Entity

2. **Processing Errors**:
   - OpenCV operation failures → 500 Internal Server Error with details
   - Memory allocation failures → 503 Service Unavailable

3. **Configuration Errors**:
   - Invalid parameter ranges → 400 Bad Request
   - Missing required configuration → 400 Bad Request

### Fallback Strategies

```python
def safe_detect_intersection(self, image: np.ndarray) -> LineIntersectionResult:
    """Safe detection with comprehensive error handling"""
    start_time = time.time()

    try:
        # Validate input
        if image is None or image.size == 0:
            return LineIntersectionResult(
                has_intersection=False,
                error_message="Invalid image data"
            )

        # Processing pipeline with timeout
        result = self._detect_intersection_with_timeout(image, timeout_ms=280)

        # Validate result
        if result and result.confidence >= self.config.min_confidence:
            return result
        else:
            return LineIntersectionResult(
                has_intersection=False,
                error_message="Insufficient confidence or no intersection detected"
            )

    except cv2.error as e:
        logger.error(f"OpenCV processing error: {e}")
        return LineIntersectionResult(
            has_intersection=False,
            error_message=f"OpenCV processing error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error in line detection: {e}")
        return LineIntersectionResult(
            has_intersection=False,
            error_message=f"Processing error: {str(e)}"
        )
    finally:
        processing_time = (time.time() - start_time) * 1000
        self.metrics.record_processing(processing_time, result.has_intersection if result else False)
```

## Security Considerations

### Input Validation

```python
def validate_roi_coordinates(self, roi_config: RoiConfig) -> bool:
    """Validate ROI configuration for security"""
    return (
        roi_config.x1 >= 0 and roi_config.y1 >= 0 and
        roi_config.x2 > roi_config.x1 and roi_config.y2 > roi_config.y1 and
        roi_config.width() <= 1920 and roi_config.height() <= 1080 and
        roi_config.width() * roi_config.height() <= 1920 * 1080  # Max image size
    )
```

### Access Control

- All line detection API endpoints inherit existing NHEM password authentication
- Configuration changes require admin-level permissions
- Rate limiting applied to manual detection requests
- No persistent storage of image data beyond operational requirements

## Testing Strategy

### Unit Tests

```python
class TestLineIntersectionDetector:
    def test_green_mask_extraction(self):
        """Test HSV color segmentation with medical imaging patterns"""
        # Test with synthetic green line patterns
        # Validate HSV threshold accuracy (H:40-80, S:50-255, V:50-255)
        # Verify morphological operation effectiveness

    def test_line_detection(self):
        """Test Canny + Hough line detection with medical line patterns"""
        # Test Canny thresholds (25 low, 80 high) on medical images
        # Validate Hough parameters (min_length=15, max_gap=8)
        # Test horizontal/vertical line filtering

    def test_intersection_calculation(self):
        """Test virtual intersection calculation using parametric equations"""
        # Test mathematical accuracy with known line pairs
        # Validate parallel line detection (denominator < 0.01)
        # Test intersection coordinate bounds validation

    def test_confidence_scoring(self):
        """Test exact confidence formula: (line1_length + line2_length) / 200 * edge_quality * 0.9 + temporal_stability * 0.1"""
        # Validate confidence range [0.0, 1.0]
        # Test edge quality calculation
        # Test temporal stability weighting

    def test_error_handling(self):
        """Test error scenarios and fallbacks with medical context"""
        # Test insufficient green pixels error handling
        # Test OpenCV processing error recovery
        # Test timeout handling (300ms limit)
```

### Integration Tests

#### API Integration Tests
- **Authentication Integration**: Test with NHEM password system (default: 31415)
- **Endpoint Integration**: Test `/api/roi/line-intersection/*` endpoints with existing patterns
- **Response Format**: Validate LineIntersectionResult JSON serialization
- **Error Response Format**: Test structured error responses matching NHEM patterns

#### Python Client Integration Tests
- **Matplotlib Integration**: Test overlay rendering with existing RealtimePlotter
- **Tkinter Integration**: Test widget embedding in existing GUI framework
- **API Client Integration**: Test new endpoints with existing HTTP client
- **Configuration Integration**: Test local config loading with existing patterns

#### System Integration Tests
- **ROI Processing Separation**: Validate ROI1/ROI2 processing isolation
- **Performance Impact**: Test <300ms processing guarantee under load
- **Memory Usage**: Validate <50MB additional memory usage
- **Threading Safety**: Test concurrent ROI1/ROI2 processing

### Medical Validation Tests

#### Clinical Accuracy Tests
- **Anatomical Landmark Detection**: Test with medical ultrasound images containing known anatomical structures
- **Geometric Precision**: Validate ±5 pixel accuracy requirement on medical images
- **Confidence Validation**: Test confidence scoring on medical image quality variations
- **Success Rate**: Validate >90% detection rate on clear medical green line patterns

#### Medical Image Quality Tests
- **Lighting Variations**: Test under various medical imaging lighting conditions
- **Image Quality**: Test with different medical image resolutions and noise levels
- **Green Color Variations**: Test with medical monitor green color calibration variations
- **Motion Blur**: Test with patient movement scenarios

#### Clinical Workflow Integration Tests
- **Real-time Performance**: Validate 4 FPS processing during clinical simulations
- **User Interaction**: Test enable/disable controls during clinical workflow scenarios
- **Error Recovery**: Test system behavior during critical medical monitoring situations
- **Multi-display Integration**: Test with medical workstation multi-monitor setups

#### Cross-Platform Medical Environment Tests
- **Windows Medical Workstations**: Test on Windows 10/11 medical workstation environments
- **Medical Software Compatibility**: Test alongside common medical imaging applications
- **Hardware Variations**: Test with different medical graphics cards and display configurations
- **Network Conditions**: Test under various hospital network conditions

### Performance Benchmarking Tests

#### Processing Time Tests
- **Single Frame Processing**: Validate <300ms processing time under various loads
- **Memory Usage**: Monitor memory consumption during extended operation (24+ hours)
- **CPU Usage**: Validate impact on existing 45 FPS DataProcessor performance
- **Cache Efficiency**: Validate 100ms cache effectiveness

#### Load Testing
- **Concurrent Processing**: Test ROI1 line detection alongside ROI2 grayscale analysis
- **Extended Operation**: Test system stability over 24+ hour continuous operation
- **Resource Leaks**: Validate no memory or resource leaks during extended testing
- **Error Recovery**: Test system resilience after repeated error conditions

### User Acceptance Testing (UAT)

#### Clinical User Testing
- **Medical Technician Validation**: Test with actual medical imaging technicians
- **Usability Assessment**: Validate Chinese interface text and color coding effectiveness
- **Workflow Integration**: Test integration into existing clinical workflows
- **Training Requirements**: Assess training needs for clinical staff

#### Configuration Management Testing
- **GUI Configuration**: Test configuration changes through Python client interface
- **API Configuration**: Test configuration management via API endpoints
- **Environment Variable Override**: Test NHEM_* environment variable configuration
- **Configuration Persistence**: Test configuration save/restore functionality

## Deployment Considerations

### Dependencies

```python
# Backend requirements.txt additions
opencv-python>=4.5.0
numpy>=1.19.0

# Python client requirements.txt additions
matplotlib>=3.3.0
Pillow>=8.0.0
```

### Configuration Migration

- Automatic configuration file migration on startup
- Backward compatibility with existing configurations
- Default parameter values for new installations

### Rollback Strategy

- Feature flag to disable line detection if issues occur
- Graceful degradation to existing ROI functionality
- Configuration backup and restore procedures