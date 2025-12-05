# Data Flow Documentation

## System Data Flow Overview

The NHEM system processes data through a sophisticated pipeline that handles both simulated and real ROI (Region of Interest) data, performs advanced peak detection, and distributes results in real-time via multiple channels.

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   Data Sources   │───►│   DataProcessor  │───►│  Peak Detection  │
│                  │    │   (45 FPS)       │    │                  │
│ • Simulated      │    │                  │    │ • Enhanced       │
│ • ROI Capture    │    │ • Timing Control │    │ • Color Classif. │
│ • Fallback Logic │    │ • ROI Integration│    │ • Quality Score  │
└──────────────────┘    └──────────────────┘    └──────────────────┘
                                   │                       │
                                   ▼                       ▼
                           ┌──────────────────┐    ┌──────────────────┐
                           │   DataStore      │    │   DataStore      │
                           │                  │    │                  │
                           │ • Frame Buffer   │    │ • Peak Buffer    │
                           │ • ROI Buffer     │    │ • Status Track   │
                           │ • Thread Safe    │    │ • Statistics     │
                           └──────────────────┘    └──────────────────┘
                                   │                       │
                                   └───────────┬───────────┘
                                               │
                                               ▼
                                    ┌──────────────────┐
                                    │ DataBroadcaster  │
                                    │     (60 FPS)     │
                                    │                  │
                                    │ • WebSocket      │
                                    │ • Client Filter  │
                                    │ • Message Queue  │
                                    └──────────────────┘
                                               │
                                               ▼
                                    ┌──────────────────┐
                                    │   API Layer      │
                                    │                  │
                                    │ • HTTP Endpoints │
                                    │ • RESTful API    │
                                    │ • Real-time Data │
                                    └──────────────────┘
```

## 1. Data Processing Pipeline

### 1.1 Data Source Selection Logic

```python
# In processor.py _run() method
if roi_configured:
    # Use real ROI data
    roi_data = roi_capture_service.capture_roi(roi_config)
    if roi_data and roi_data.gray_value > 0:
        signal_value = roi_data.gray_value
        data_source = "ROI"
    else:
        # ROI capture failed, fallback to simulated data
        signal_value = base_value + 10.0 * math.sin(2 * math.pi * 0.5 * t)
        data_source = "Fallback"
else:
    # Use simulated data
    signal_value = base_value + 10.0 * math.sin(2 * math.pi * 0.5 * t)
    data_source = "Simulated"
```

**Data Source Priority**:
1. **Real ROI Data** (highest priority) - Captured from screen
2. **Fallback Data** - Simulated when ROI capture fails
3. **Simulated Data** - Used when ROI not configured

### 1.2 Processing Loop Flow

```
DataProcessor._run() (Main Thread - 45 FPS)
├── 1. Check ROI Configuration Status
│   ├── data_store.is_roi_configured()
│   └── data_store.get_roi_config()
├── 2. Data Source Selection
│   ├── ROI Capture (if configured)
│   │   ├── roi_capture_service.capture_roi()
│   │   ├── PIL screenshot processing
│   │   ├── Grayscale conversion
│   │   └── Average gray value calculation
│   └── Simulated Data (if ROI unavailable)
│       ├── Sine wave generation
│       └── Base value + noise
├── 3. Peak Detection Processing
│   ├── enhanced_peak_detector.process_frame()
│   ├── Dynamic threshold calculation
│   ├── Multiple slope analysis
│   ├── Peak region identification
│   └── Color classification (green/red)
├── 4. Data Storage
│   ├── data_store.add_frame() - Main data
│   ├── data_store.add_roi_frame() - ROI data
│   └── data_store.add_enhanced_peak() - Peak results
├── 5. Statistics Update
│   ├── Frame count increment
│   ├── Baseline calculation
│   └── Status tracking
└── 6. Precise Timing Control
    ├── 1.0 / FPS interval calculation
    ├── Sleep time adjustment
    └── Loop time monitoring
```

### 1.3 Data Flow Sequence Diagram

```
Time →
Frame 1     Frame 2     Frame 3     Frame 4     Frame 5
  │           │           │           │           │
  ▼           ▼           ▼           ▼           ▼
ROI Check   ROI Check   ROI Check   ROI Check   ROI Check
  │           │           │           │           │
  ▼           ▼           ▼           ▼           ▼
Capture     Capture     Capture     Capture     Capture
  │           │           │           │           │
  ▼           ▼           ▼           ▼           ▼
Process     Process     Process     Process     Process
  │           │           │           │           │
  ▼           ▼           ▼           ▼           ▼
Detect      Detect      Detect      Detect      Detect
  │           │           │           │           │
  ▼           ▼           ▼           ▼           ▼
Store       Store       Store       Store       Store
  │           │           │           │           │
  ▼           ▼           ▼           ▼           ▼
Broadcast   Broadcast   Broadcast   Broadcast   Broadcast
```

## 2. Peak Detection Data Flow

### 2.1 Enhanced Peak Detection Algorithm

```python
# enhanced_peak_detector.py process_frame()
def process_frame(self, signal_value: float, frame_count: int) -> dict:
    # 1. Update frame history
    self.frame_buffer.append(signal_value)
    self.frame_numbers.append(frame_count)

    # 2. Calculate dynamic baseline
    baseline = self._calculate_dynamic_baseline()

    # 3. Multi-method slope analysis
    slopes = self._calculate_multiple_slopes()

    # 4. Rising/falling edge detection
    rising_edges, falling_edges = self._detect_edges(slopes)

    # 5. Peak region identification
    peak_regions = self._identify_peak_regions(rising_edges, falling_edges)

    # 6. Peak validation and classification
    validated_peaks = self._validate_and_classify_peaks(peak_regions)

    # 7. Quality scoring
    confidence_scores = self._calculate_confidence_scores(validated_peaks)

    return {
        'peak_signal': peak_signal,
        'peak_color': peak_color,
        'peak_confidence': confidence,
        'threshold': threshold,
        'in_peak_region': in_peak_region,
        'frame_count': frame_count
    }
```

### 2.2 Peak Detection Flow Chart

```
Signal Input → Frame Buffer Update
     │
     ▼
Dynamic Baseline Calculation
     │
     ▼
Multi-Method Slope Analysis
├── Central Difference (3-point)
├── Central Difference (5-point)
├── Forward Difference
└── Backward Difference
     │
     ▼
Edge Detection
├── Rising Edges (positive slopes)
└── Falling Edges (negative slopes)
     │
     ▼
Peak Region Identification
├── Find rising→falling pairs
├── Apply margin_frames extension
└── Filter by min_region_length
     │
     ▼
Peak Validation
├── Threshold validation
├── Signal-to-noise ratio
└── Morphological validation
     │
     ▼
Color Classification
├── Green Peaks (stable - low frame difference)
└── Red Peaks (unstable - high frame difference)
     │
     ▼
Quality Scoring
├── Peak sharpness
├── Baseline stability
├── Signal consistency
└── Overall confidence
```

## 3. ROI Capture Data Flow

### 3.1 ROI Screenshot Capture Pipeline

```python
# roi_capture.py capture_roi()
def capture_roi(self, roi_config: RoiConfig) -> Optional[RoiData]:
    # 1. Timing Control
    current_time = time.time()
    if current_time - self.last_capture_time < self.frame_interval:
        return None  # Rate limiting

    # 2. Screenshot Capture
    try:
        screenshot = ImageGrab.grab()  # Full screen
    except Exception as e:
        logger.error(f"Screenshot failed: {e}")
        return None

    # 3. ROI Extraction
    roi_image = screenshot.crop((
        roi_config.x1, roi_config.y1,
        roi_config.x2, roi_config.y2
    ))

    # 4. Grayscale Conversion
    gray_image = roi_image.convert('L')

    # 5. Average Gray Value Calculation
    gray_array = np.array(gray_image)
    gray_value = float(np.mean(gray_array))

    # 6. Base64 Encoding
    buffer = io.BytesIO()
    roi_image.save(buffer, format='PNG')
    roi_base64 = base64.b64encode(buffer.getvalue()).decode()

    # 7. Create RoiData object
    return RoiData(
        width=roi_config.width,
        height=roi_config.height,
        pixels=roi_base64,
        gray_value=gray_value,
        format="base64"
    )
```

### 3.2 ROI Data Flow Architecture

```
Screen Capture
     │
     ▼
Full Screenshot (PIL.Image)
     │
     ▼
ROI Crop Operation
     │
     ▼
ROI Region Image (200x150 typical)
     │
     ▼
┌─────────────────┬─────────────────┐
│                 │                 │
│ Grayscale       │ Base64 Encoding │
│ Conversion      │                 │
│                 │                 │
▼                 ▼
Gray Array        PNG Bytes         │
│                 │                 │
│                 ▼                 │
│            Base64 String         │
│                 │                 │
│                 ▼                 │
│            API Transmission       │
│                                    │
▼                                    ▼
Average Gray Value                  Image Data
(127.45)                            (iVBORw0K...)
│                                    │
│                                    ▼
│                            ROI Data Object
│                                    │
└────────────────────┬─────────────────┘
                     ▼
              Peak Detection Input
```

## 4. Data Storage Architecture

### 4.1 Circular Buffer Data Flow

```python
# data_store.py storage structure
class DataStore:
    def __init__(self):
        # Main data storage - Frame buffer (100 frames default)
        self.frames = deque(maxlen=settings.buffer_size)

        # ROI-specific storage (500 frames default)
        self.roi_frames = deque(maxlen=settings.roi_buffer_size)

        # Peak detection results
        self.peak_data = deque(maxlen=settings.peak_buffer_size)

        # Enhanced peak detection results
        self.enhanced_peaks = deque(maxlen=settings.peak_buffer_size)
```

### 4.2 Data Storage Flow

```
Data Input
     │
     ▼
Thread-Safe Lock Acquisition
     │
     ▼
Data Validation
├── Timestamp validation
├── Range validation
└── Type validation
     │
     ▼
Buffer Operations
├── Append new data
├── Automatic old data removal
└── Buffer statistics update
     │
     ▼
Lock Release
     │
     ▼
Memory Management
├── Circular buffer overflow handling
├── Memory usage monitoring
└── Garbage collection
```

### 4.3 Data Query Flow

```
API Request
     │
     ▼
Data Query Validation
├── Parameter validation
├── Access rights check
└── Rate limiting
     │
     ▼
Lock Acquisition (Read Lock)
     │
     ▼
Data Retrieval
├── Slice operations
├── Data transformation
└── Statistics calculation
     │
     ▼
Lock Release
     │
     ▼
Response Formatting
├── JSON serialization
├── Model validation
└── API response
```

## 5. Real-time Broadcasting Data Flow

### 5.1 Broadcasting Architecture

```python
# data_broadcaster.py main loop
async def _broadcast_loop(self):
    while not self._stop_event.is_set():
        # 1. Get current data snapshot
        current_data = self._get_current_snapshot()

        # 2. Check for significant changes
        if self._has_significant_change(current_data):
            # 3. Format message
            message = self._format_broadcast_message(current_data)

            # 4. Filter clients based on subscriptions
            target_clients = self._filter_target_clients(message)

            # 5. Broadcast to clients
            await self._broadcast_to_clients(target_clients, message)

        # 6. Timing control (60 FPS)
        await asyncio.sleep(1.0 / 60)
```

### 5.2 WebSocket Data Distribution Flow

```
DataStore (Source)
     │
     ▼
DataBroadcaster (60 FPS)
     │
     ├── Get Current Snapshot
     ├── Detect Significant Changes
     ├── Format JSON Message
     └── Filter Target Clients
                │
                ▼
        SocketServer (Distributor)
                │
                ├── Client Authentication
                ├── Connection Management
                ├── Message Queue Processing
                └── Error Handling
                │
                ▼
        WebSocket Clients (Consumers)
                │
                ├── Real-time Display
                ├── Chart Updates
                └── Status Monitoring
```

### 5.3 Message Flow Types

```
1. Data Messages (Primary)
   ├─ Frame count
   ├─ Current value
   ├─ Peak signal
   ├─ Baseline
   └─ Timestamp

2. Status Messages (Periodic)
   ├─ System status
   ├─ Connection count
   ├─ Performance metrics
   └─ Health indicators

3. Configuration Messages (Event-driven)
   ├─ Configuration updates
   ├─ ROI changes
   ├─ Parameter adjustments
   └─ System state changes

4. Error Messages (As-needed)
   ├─ Connection errors
   ├─ Data processing errors
   ├─ System warnings
   └─ Performance alerts
```

## 6. API Data Flow

### 6.1 HTTP Request Processing Flow

```
Client Request
     │
     ▼
FastAPI Router
     │
     ├── Request validation
     ├── Parameter parsing
     └── Authentication check
     │
     ▼
Endpoint Handler
     │
     ├── Business logic execution
     ├── Data access (DataStore)
     ├── Service coordination
     └── Response preparation
     │
     ▼
Response Formatting
     │
     ├── Model validation
     ├── JSON serialization
     ├── Error handling
     └── HTTP status codes
     │
     ▼
HTTP Response
```

### 6.2 Real-time Data Request Flow

```
GET /data/realtime?count=100
     │
     ▼
1. Request Validation
     ├─ Parameter range check
     ├─ Access validation
     └─ Rate limiting
     │
     ▼
2. System Status Check
     ├─ Is system running?
     └─ Is data available?
     │
     ▼
3. Data Retrieval
     ├─ Get latest frames from DataStore
     ├─ ROI configuration check
     └─ Real-time ROI capture (if configured)
     │
     ▼
4. Data Processing
     ├─ Time series generation
     ├─ ROI data formatting
     └─ Peak signal integration
     │
     ▼
5. Response Assembly
     ├─ RealtimeDataResponse model
     ├─ Data validation
     └─ JSON serialization
     │
     ▼
6. HTTP Response
```

## 7. Configuration Data Flow

### 7.1 Configuration Update Flow

```
Configuration Update Request
     │
     ▼
1. Authentication
     ├─ Password validation
     └─ Authorization check
     │
     ▼
2. Parameter Validation
     ├─ Type validation
     ├─ Range validation
     └─ Business rule validation
     │
     ▼
3. ConfigManager Update
     ├─ JSON file update
     ├─ Atomic write operation
     └─ Schema validation
     │
     ▼
4. Runtime Update
     ├─ AppConfig reload
     ├─ Service notification
     └─ Parameter application
     │
     ▼
5. Service Integration
     ├─ DataProcessor configuration
     ├─ PeakDetector parameters
     ├─ ROI settings
     └─ Broadcasting settings
     │
     ▼
6. Response
     ├─ Success confirmation
     ├─ Updated configuration
     └─ Error details (if any)
```

### 7.2 Configuration Hierarchy Flow

```
1. Default Values (Code)
     │
     ▼
2. JSON Configuration File (fem_config.json)
     │
     ▼
3. Environment Variables (NHEM_* prefix)
     │
     ▼
4. Runtime Updates (API calls)
     │
     ▼
5. Active Configuration (AppConfig instance)
```

## 8. Error Handling Data Flow

### 8.1 Error Propagation Flow

```
Error Source
     │
     ▼
Error Detection
     ├─ Exception catching
     ├─ Value validation
     └─ System monitoring
     │
     ▼
Error Classification
     ├─ Business logic errors
     ├─ System errors
     ├─ Network errors
     └─ Data errors
     │
     ▼
Error Handling
     ├─ Error logging
     ├─ System state update
     ├─ Client notification
     └─ Recovery attempts
     │
     ▼
Error Response
     ├─ Standardized error format
     ├─ Error codes
     ├─ Human-readable messages
     └─ Debug information
```

## 9. Performance Optimization Data Flow

### 9.1 Memory Management Flow

```
Data Input
     │
     ▼
Memory Pool Allocation
     │
     ▼
Buffer Management
├── Circular buffers (bounded memory)
├── Automatic cleanup (old data removal)
└── Memory monitoring
     │
     ▼
Garbage Collection
├── PIL image cleanup
├── NumPy array management
└── Object lifecycle management
     │
     ▼
Performance Monitoring
├── Memory usage tracking
├── Processing latency measurement
└── Throughput analysis
```

### 9.2 Processing Optimization Flow

```
Timing Control
     │
     ▼
Precise Frame Timing (45 FPS)
     │
     ├─ Interval calculation (1/45 = 22.22ms)
     ├─ Processing time measurement
     └─ Sleep time adjustment
     │
     ▼
Load Balancing
├─ Thread utilization monitoring
├─ CPU usage optimization
└─ I/O operation batching
     │
     ▼
Quality Assurance
├─ Data integrity checks
├─ Processing validation
└─ Result verification
```

This comprehensive data flow documentation illustrates how data moves through the NHEM system from source to consumption, with detailed explanations of each processing stage, optimization techniques, and error handling mechanisms. The system is designed for high-performance real-time processing with robust data management and distribution capabilities.