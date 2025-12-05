# Class Hierarchy and Relationships

## Overview

The NHEM backend follows a well-structured class hierarchy with clear inheritance patterns, composition relationships, and separation of concerns. This documentation outlines all major classes, their relationships, and their roles in the system.

## 1. Configuration Hierarchy

### 1.1 Configuration Classes

```python
# Base class (external)
BaseSettings (pydantic_settings.BaseSettings)
    │
    ▼
AppConfig (app/config.py)
    │   ├── Multi-source configuration loading
    │   ├── Environment variable override support
    │   ├── Runtime configuration reloading
    │   └── Type validation and conversion
    │
    └── Dependencies:
        ├── ConfigManager (for JSON loading)
        ├── Environment variables (NHEM_* prefix)
        └── fem_config.json (primary configuration)
```

### 1.2 Configuration Management

```python
# Standalone class (no inheritance)
ConfigManager (app/core/config_manager.py)
    │   ├── JSON file operations
    │   ├── Atomic configuration updates
    │   ├── Schema validation
    │   └── Thread-safe access
    │
    └── Used by:
        ├── AppConfig (for JSON loading)
        ├── API endpoints (for configuration management)
        └── Runtime services (for configuration reloading)
```

### 1.3 Peak Detection Configuration

```python
# Dataclass for configuration
@dataclass
PeakDetectionConfig (app/core/enhanced_peak_detector.py)
    │   ├── threshold: float
    │   ├── margin_frames: int
    │   ├── difference_threshold: float
    │   └── min_region_length: int
    │
    └── Used by:
        ├── EnhancedPeakDetector (configuration)
        └── API endpoints (parameter updates)
```

## 2. Data Model Hierarchy

### 2.1 Base Model Classes

```python
# Base class (external)
BaseModel (pydantic.BaseModel)
    │
    ├── SystemStatus (Enum) - System state enumeration
    ├── HealthResponse - Health check response
    ├── StatusResponse - System status response
    ├── RealtimeDataResponse - Real-time data API response
    ├── ErrorResponse - Standardized error format
    ├── ControlCommandResponse - Control command response
    ├── PeakSignalResponse - Peak signal status response
    ├── ControlStatusResponse - Control status response
    ├── AnalyzeResponse - Video analysis response
    ├── BaseSuccessResponse - Generic success response
    └── [20+ other response models]
```

### 2.2 ROI-Related Models

```python
BaseModel
    │
    ├── RoiConfig - ROI configuration with validation
    │   ├── x1, y1, x2, y2: int (coordinates)
    │   ├── Validation methods
    │   └── Computed properties (center, width, height)
    │
    ├── RoiData - Captured ROI image data
    │   ├── width, height: int
    │   ├── pixels: str (base64 encoded)
    │   ├── gray_value: float
    │   └── format: str
    │
    ├── RoiConfigResponse - ROI configuration response
    ├── RoiCaptureResponse - ROI capture response
    ├── RoiFrameRateResponse - ROI frame rate response
    └── RoiTimeSeriesPoint - ROI time series data point
```

### 2.3 Peak Detection Models

```python
BaseModel
    │
    ├── PeakRegionData - Peak detection results
    │   ├── start_frame, end_frame, peak_frame: int
    │   ├── max_value: float
    │   ├── color: str ('green'/'red')
    │   ├── confidence: float
    │   └── difference: float
    │
    ├── EnhancedPeakSignal - Advanced peak detection data
    │   ├── signal: Optional[int] (1 for peak, None for no peak)
    │   ├── color: Optional[str] ('green'/'red')
    │   ├── confidence: float
    │   ├── threshold: float
    │   ├── in_peak_region: bool
    │   └── frame_count: int
    │
    └── PeakDetectionConfigResponse - Peak detection configuration
```

### 2.4 Time Series Models

```python
BaseModel
    │
    ├── TimeSeriesPoint - Main time series data
    │   ├── t: float (timestamp relative to start)
    │   └── value: float (signal value)
    │
    ├── RoiTimeSeriesPoint - ROI time series data
    │   ├── t: float
    │   ├── gray_value: float
    │   └── roi_index: int
    │
    └── AnalyzeSeriesPoint - Analysis time series data
        ├── t: float
        ├── value: float
        ├── ref: float (reference)
        ├── std: float (standard deviation)
        ├── high: float
        └── orange: float
```

### 2.5 Window Capture Models

```python
BaseModel
    │
    ├── WindowCaptureResponse - Historical data window
    │   ├── window_size: int
    │   ├── frame_range: Tuple[int, int]
    │   ├── series: List[TimeSeriesPoint]
    │   └── capture_metadata: Dict[str, Any]
    │
    ├── RoiWindowCaptureResponse - ROI historical window
    │   ├── window_size: int
    │   ├── roi_frame_range: Tuple[int, int]
    │   ├── main_frame_range: Tuple[int, int]
    │   ├── series: List[RoiTimeSeriesPoint]
    │   ├── roi_config: Dict[str, Any]
    │   └── capture_metadata: Dict[str, Any]
    │
    └── RoiWindowCaptureWithPeaksResponse - ROI window with peak analysis
        ├── [All fields from RoiWindowCaptureResponse]
        ├── peak_detection_results: Dict[str, Any]
        └── peak_detection_params: Dict[str, Any]
```

## 3. Core Processing Classes

### 3.1 Data Processor Hierarchy

```python
# Main processing class (no inheritance)
DataProcessor (app/core/processor.py)
    │   ├── Main processing loop (45 FPS)
    │   ├── ROI data integration
    │   ├── Peak detection coordination
    │   └── Data storage coordination
    │
    ├── Attributes:
    │   ├── _enhanced_detector: EnhancedPeakDetector
    │   ├── _thread: Thread (background processing)
    │   ├── _stop_event: Event (thread control)
    │   └── _frame_count: int (processing counter)
    │
    ├── Dependencies:
    │   ├── EnhancedPeakDetector (peak detection)
    │   ├── DataStore (data storage)
    │   ├── RoiCaptureService (ROI capture)
    │   └── AppConfig (configuration)
    │
    └── Lifecycle:
        ├── start() - Start processing thread
        ├── stop() - Stop processing thread
        ├── _run() - Main processing loop
        └── reload_peak_detection_config() - Configuration update
```

### 3.2 Enhanced Peak Detector

```python
# Peak detection class (no inheritance)
EnhancedPeakDetector (app/core/enhanced_peak_detector.py)
    │   ├── Advanced peak detection algorithms
    │   ├── Multiple slope calculation methods
    │   ├── Dynamic threshold calculation
    │   └── Color classification
    │
    ├── Attributes:
    │   ├── _config: PeakDetectionConfig
    │   ├── frame_buffer: deque (signal history)
    │   ├── frame_numbers: deque (frame tracking)
    │   └── baseline_history: deque (baseline tracking)
    │
    ├── Methods:
    │   ├── process_frame() - Main processing method
    │   ├── update_config() - Configuration update
    │   ├── _calculate_dynamic_baseline() - Baseline calculation
    │   ├── _calculate_multiple_slopes() - Slope analysis
    │   ├── _detect_edges() - Edge detection
    │   ├── _identify_peak_regions() - Peak region identification
    │   ├── _validate_and_classify_peaks() - Peak validation
    │   └── _calculate_confidence_scores() - Quality scoring
    │
    └── Uses:
        ├── numpy - Numerical computations
        ├── math - Mathematical functions
        └── PeakDetectionConfig - Configuration parameters
```

### 3.3 Data Store Class

```python
# Thread-safe data storage (no inheritance)
DataStore (app/core/data_store.py)
    │   ├── Thread-safe circular buffers
    │   ├── Multi-type data storage
    │   └── Statistics tracking
    │
    ├── Attributes:
    │   ├── frames: deque (main data buffer)
    │   ├── roi_frames: deque (ROI data buffer)
    │   ├── peak_data: deque (peak detection results)
    │   ├── enhanced_peaks: deque (enhanced peak results)
    │   ├── _status: SystemStatus (current system state)
    │   ├── _baseline: float (calculated baseline)
    │   └── _locks: Dict[str, Lock] (thread safety)
    │
    ├── Data Types:
    │   ├── Frame - Main data frame
    │   │   ├── index: int
    │   │   ├── timestamp: datetime
    │   │   └── value: float
    │   │
    │   └── RoiFrame - ROI-specific frame
    │       ├── index: int
    │       ├── timestamp: datetime
    │       ├── gray_value: float
    │       ├── roi_config: RoiConfig
    │       └── frame_count: int
    │
    ├── Methods:
    │   ├── add_frame() - Add main data frame
    │   ├── add_roi_frame() - Add ROI frame
    │   ├── add_enhanced_peak() - Add peak detection result
    │   ├── get_series() - Retrieve data series
    │   ├── get_roi_series() - Retrieve ROI series
    │   └── [Various getter and status methods]
    │
    └── Dependencies:
        ├── AppConfig (configuration)
        ├── threading (locks)
        ├── collections (deque)
        └── datetime (timestamps)
```

### 3.4 ROI Capture Service

```python
# ROI screenshot capture (no inheritance)
RoiCaptureService (app/core/roi_capture.py)
    │   ├── Screen screenshot capture
    │   ├── ROI region extraction
    │   ├── Image processing and encoding
    │   └── Frame rate control
    │
    ├── Attributes:
    │   ├── frame_rate: int (capture frequency)
    │   ├── last_capture_time: float (rate limiting)
    │   ├── frame_interval: float (timing control)
    │   └── capture_cache: Dict (optional caching)
    │
    ├── Methods:
    │   ├── capture_roi() - Main capture method
    │   ├── set_roi_frame_rate() - Frame rate configuration
    │   ├── get_roi_frame_rate() - Frame rate query
    │   └── clear_cache() - Cache management
    │
    ├── Dependencies:
    │   ├── PIL (Pillow) - Image processing
    │   ├── numpy - Numerical operations
    │   ├── io - BytesIO for image encoding
    │   ├── base64 - Image encoding
    │   └── DataStore - History storage
    │
    └── Processing Pipeline:
        1. Rate limiting check
        2. Full screen screenshot
        3. ROI region cropping
        4. Grayscale conversion
        5. Average value calculation
        6. Base64 encoding
        7. RoiData object creation
```

## 4. Communication Classes

### 4.1 WebSocket Server

```python
# WebSocket server implementation (no inheritance)
SocketServer (app/core/socket_server.py)
    │   ├── Asyncio-based WebSocket server
    │   ├── Client authentication and management
    │   ├── Message broadcasting
    │   └── Connection lifecycle management
    │
    ├── Attributes:
    │   ├── host: str (server address)
    │   ├── port: int (server port)
    │   ├── clients: Set[SocketClient] (active connections)
    │   ├── password: str (authentication)
    │   └── _server_task: Optional[Task] (asyncio task)
    │
    ├── Nested Classes:
    │   └── SocketClient - Client connection wrapper
    │       ├── websocket: WebSocketServerProtocol
    │       ├── authenticated: bool
    │       ├── client_id: str
    │       └── last_ping: float
    │
    ├── Methods:
    │   ├── start() - Start WebSocket server
    │   ├── stop() - Stop WebSocket server
    │   ├── handle_client() - Handle new client connection
    │   ├── broadcast_message() - Broadcast to all clients
    │   ├── remove_client() - Remove disconnected client
    │   └── _auth_client() - Authenticate client
    │
    ├── Dependencies:
    │   ├── asyncio - Async programming
    │   ├── websockets - WebSocket protocol
    │   ├── json - Message serialization
    │   ├── threading - Thread management
    │   └── AppConfig - Configuration
    │
    └── Message Flow:
        1. Client authentication
        2. Connection registration
        3. Message broadcasting
        4. Client management
        5. Connection cleanup
```

### 4.2 Data Broadcaster

```python
# Real-time data broadcasting (no inheritance)
DataBroadcaster (app/core/data_broadcaster.py)
    │   ├── High-frequency data broadcasting (60 FPS)
    │   ├── Client subscription filtering
    │   ├── Message queuing
    │   └── Change detection
    │
    ├── Attributes:
    │   ├── socket_server: SocketServer (client management)
    │   ├── data_store: DataStore (data source)
    │   ├── _stop_event: Event (broadcast control)
    │   ├── _last_broadcast_data: Dict (change detection)
    │   └── _broadcast_task: Optional[Task] (asyncio task)
    │
    ├── Methods:
    │   ├── start() - Start broadcasting
    │   ├── stop() - Stop broadcasting
    │   ├── _broadcast_loop() - Main broadcasting loop
    │   ├── _get_current_snapshot() - Get current data
    │   ├── _has_significant_change() - Change detection
    │   ├── _format_broadcast_message() - Message formatting
    │   ├── _filter_target_clients() - Client filtering
    │   └── _broadcast_to_clients() - Message distribution
    │
    ├── Dependencies:
    │   ├── SocketServer (client communication)
    │   ├── DataStore (data access)
    │   ├── asyncio (async broadcasting)
    │   ├── json (message serialization)
    │   └── time (timing control)
    │
    └── Broadcasting Flow:
        1. Get current data snapshot
        2. Check for significant changes
        3. Format JSON message
        4. Filter target clients
        5. Broadcast to clients
        6. Rate limiting (60 FPS)
```

## 5. Utility Classes

### 5.1 ROI Image Generator

```python
# Image generation utilities (module functions)
roi_image_generator (app/utils/roi_image_generator.py)
    │   ├── No class-based structure
    │   ├── Functional approach
    │   └── Utility functions only
    │
    ├── Functions:
    │   ├── create_roi_data_with_image() - Create ROI with image data
    │   └── generate_waveform_image_with_peaks() - Generate waveform visualization
    │
    ├── Dependencies:
    │   ├── PIL - Image creation and manipulation
    │   ├── numpy - Numerical computations
    │   ├── io - BytesIO for image encoding
    │   ├── base64 - Image encoding
    │   ├── DataStore - Data access
    │   └── typing - Type hints
    │
    └── Output:
        ├── Base64 encoded images
        ├── PNG format for API transmission
        └── Waveform visualizations with peak annotations
```

## 6. Global Instances and Singletons

### 6.1 System Singletons

```python
# Global instances (app-level singletons)
processor = DataProcessor()                    # Main processing engine
data_store = DataStore()                      # Thread-safe data storage
roi_capture_service = RoiCaptureService()     # ROI screenshot service
socket_server = SocketServer()                # WebSocket server
data_broadcaster = DataBroadcaster()          # Real-time broadcaster
settings = AppConfig()                        # Configuration instance

# Factory function
def get_config_manager() -> ConfigManager:
    """Get singleton ConfigManager instance"""
    return ConfigManager()

# Application factory
def create_app() -> FastAPI:
    """Create FastAPI application instance"""
    # FastAPI app creation with middleware and routing
```

### 6.2 Instance Relationships

```
AppConfig (settings)
    ├── Used by: All modules for configuration
    ├── Reloads from: ConfigManager
    └── Sources: JSON file, environment variables

DataStore (data_store)
    ├── Used by: processor, broadcaster, API endpoints
    ├── Provides: Data access and storage
    └── Thread-safety: Lock-based protection

DataProcessor (processor)
    ├── Uses: data_store, enhanced_detector, roi_capture
    ├── Controlled by: API control endpoints
    └── Runs in: Background thread (45 FPS)

RoiCaptureService (roi_capture_service)
    ├── Uses: PIL, DataStore (history)
    ├── Configuration: frame_rate, ROI regions
    └── Rate limiting: Controlled capture frequency

SocketServer (socket_server)
    ├── Used by: data_broadcaster, clients
    ├── Authentication: Password-based
    └── Protocol: WebSocket

DataBroadcaster (data_broadcaster)
    ├── Uses: socket_server, data_store
    ├── Frequency: 60 FPS broadcasting
    └── Filtering: Client subscription-based
```

## 7. Inheritance and Composition Patterns

### 7.1 Inheritance Patterns

```python
# 1. Pydantic Model Inheritance (External)
BaseModel (pydantic)
    ├── All response models
    ├── Configuration models
    └── Data validation models

# 2. Enum Inheritance
enum.Enum
    └── SystemStatus (running, stopped, error)

# 3. Settings Inheritance
BaseSettings (pydantic_settings)
    └── AppConfig (multi-source configuration)
```

### 7.2 Composition Patterns

```python
# 1. Service Composition
DataProcessor
    ├── EnhancedPeakDetector
    ├── DataStore (composition)
    └── RoiCaptureService (composition)

# 2. Infrastructure Composition
DataBroadcaster
    ├── SocketServer (composition)
    └── DataStore (composition)

# 3. Configuration Composition
AppConfig
    ├── ConfigManager (utility composition)
    ├── Environment variables (external composition)
    └── JSON file (external composition)
```

### 7.3 Dependency Injection Patterns

```python
# 1. Configuration Injection
All modules receive configuration through global 'settings' instance

# 2. Service Injection
DataProcessor receives services through module-level imports
- No constructor injection (simplified architecture)
- Service location pattern through global instances

# 3. Data Store Injection
All data access goes through global 'data_store' instance
- Centralized data management
- Thread-safe access
- Consistent interface
```

This class hierarchy demonstrates a well-architected system with clear separation of concerns, appropriate use of inheritance, and effective composition patterns. The design balances simplicity with functionality, using global instances for services while maintaining clean interfaces between components.