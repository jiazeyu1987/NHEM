# Directory Structure Analysis

## Project Root Structure

```
D:\ProjectPackage\NHEM\backend\
├── run.py                          # Main application entry point
├── __init__.py                     # Root package initialization
├── requirements.txt                # Python dependencies
├── app/                            # Main application package
│   ├── __init__.py
│   ├── config.py                   # Application configuration
│   ├── logging_config.py           # Logging system setup
│   ├── models.py                   # Pydantic data models
│   ├── peak_detection.py           # Peak detection algorithms
│   ├── fem_config.json             # Configuration file
│   ├── api/                        # FastAPI routes and endpoints
│   │   ├── __init__.py
│   │   └── routes.py               # All API endpoints defined here
│   ├── core/                       # Core business logic
│   │   ├── __init__.py
│   │   ├── config_manager.py       # Configuration file management
│   │   ├── data_broadcaster.py     # Real-time data broadcasting
│   │   ├── data_store.py           # In-memory data storage
│   │   ├── enhanced_peak_detector.py # Advanced peak detection
│   │   ├── processor.py            # Main data processing engine
│   │   ├── roi_capture.py          # ROI screenshot capture
│   │   └── socket_server.py        # WebSocket server
│   └── utils/                      # Utility functions
│       ├── __init__.py
│       └── roi_image_generator.py  # Image generation utilities
├── logs/                           # Log files directory
│   └── nhem_*.log                  # Timestamped log files
└── doc/                            # Documentation
    └── code_structure/             # Code structure documentation
        ├── README.md
        └── *.md                     # Various documentation files
```

## File Purposes and Responsibilities

### Root Level Files

#### `run.py` - Application Entry Point
**Purpose**: Main application startup and FastAPI server initialization
**Responsibilities**:
- Initialize logging system
- Load configuration settings
- Start FastAPI server with proper configuration
- Handle application lifecycle

**Key Functions**:
- `main()` - Application initialization
- `run_fastapi()` - FastAPI server startup with logging configuration

#### `requirements.txt` - Dependencies
**Purpose**: Python package dependencies specification
**Key Dependencies**:
- `fastapi==0.115.0` - Web framework
- `uvicorn==0.30.6` - ASGI server
- `numpy>=1.21.0` - Numerical computing
- `pydantic>=2.0.0` - Data validation
- `Pillow>=8.0.0` - Image processing

### Application Package (`app/`)

#### `config.py` - Configuration Management
**Purpose**: Pydantic-based configuration with multi-source loading
**Class**: `AppConfig(BaseSettings)`
**Features**:
- JSON file loading (`fem_config.json`)
- Environment variable overrides (`NHEM_*` prefix)
- Runtime configuration reloading
- Type validation and conversion

#### `logging_config.py` - Logging System
**Purpose**: Centralized logging configuration
**Features**:
- File-based logging with timestamps
- Console output for INFO and above
- Structured log formatting
- Log rotation and filtering

#### `models.py` - Data Models
**Purpose**: Pydantic models for API requests/responses and internal data structures
**Key Models**:
- `SystemStatus` - System state enumeration
- `HealthResponse`, `StatusResponse` - System status models
- `RealtimeDataResponse` - Real-time data API response
- `RoiConfig` - ROI configuration with validation
- `PeakRegionData` - Peak detection results
- `EnhancedPeakSignal` - Advanced peak detection data

#### `peak_detection.py` - Peak Detection Algorithms
**Purpose**: Core peak detection algorithms and utilities
**Key Functions**:
- `detect_peaks()` - Main peak detection with color classification
- `detect_white_peaks_by_threshold()` - Threshold-based detection
- `detect_white_curve_peaks()` - Morphological peak detection

### API Layer (`app/api/`)

#### `routes.py` - API Endpoints
**Purpose**: Complete FastAPI application with all HTTP endpoints
**Key Endpoint Groups**:
1. **System Management**: `/health`, `/status`, `/control`
2. **Real-time Data**: `/data/realtime`, `/data/window-capture`
3. **ROI Management**: `/roi/config`, `/roi/frame-rate`
4. **Peak Detection**: `/peak-detection/config`, `/data/waveform-with-peaks`
5. **Configuration**: `/config/*` (GET, POST, reload, export, import)

**Features**:
- Comprehensive error handling
- Request/response validation
- Password protection for control commands
- CORS middleware configuration

### Core Business Logic (`app/core/`)

#### `processor.py` - Main Processing Engine
**Class**: `DataProcessor`
**Purpose**: Core data processing loop running at configurable FPS
**Responsibilities**:
- Main processing loop (`_run()`)
- ROI data capture integration
- Peak detection coordination
- Data storage coordination
- Precise timing control

#### `data_store.py` - Data Storage
**Class**: `DataStore`
**Purpose**: Thread-safe in-memory data storage with circular buffers
**Features**:
- Frame data storage (configurable buffer size)
- ROI-specific data storage
- Peak detection results storage
- Thread-safe operations with locks
- Status and statistics tracking

#### `enhanced_peak_detector.py` - Advanced Peak Detection
**Class**: `EnhancedPeakDetector`
**Purpose**: Sophisticated peak detection with multiple algorithms
**Features**:
- Dynamic threshold calculation
- Multiple slope calculation methods
- Peak region validation
- Color classification (green/red peaks)
- Quality scoring and confidence

#### `roi_capture.py` - ROI Screenshot Capture
**Class**: `RoiCaptureService`
**Purpose**: Real-time screenshot capture from configured screen regions
**Features**:
- PIL-based image processing
- Configurable capture regions
- Base64 encoding for API transmission
- Frame rate control
- Error handling and fallback

#### `socket_server.py` - WebSocket Server
**Class**: `SocketServer`
**Purpose**: WebSocket server for real-time data streaming
**Features**:
- Asyncio-based event loop
- Client authentication and management
- Message broadcasting
- Connection lifecycle management

#### `data_broadcaster.py` - Real-time Broadcasting
**Class**: `DataBroadcaster`
**Purpose**: High-frequency real-time data distribution
**Features**:
- 60 FPS broadcasting loop
- Client subscription filtering
- Message sequencing
- WebSocket integration

#### `config_manager.py` - Configuration File Management
**Class**: `ConfigManager`
**Purpose**: JSON configuration file operations with validation
**Features**:
- Atomic file operations
- Schema validation
- Thread-safe access
- Configuration reloading

### Utility Layer (`app/utils/`)

#### `roi_image_generator.py` - Image Generation Utilities
**Purpose**: Image generation and processing utilities
**Features**:
- Base64 image encoding
- Waveform visualization
- Peak annotation
- ROI visualization

### Configuration File (`app/fem_config.json`)

**Purpose**: Runtime configuration storage
**Sections**:
- `server` - Network and CORS settings
- `data_processing` - FPS, buffer sizes
- `roi_capture` - ROI capture settings
- `peak_detection` - Peak detection parameters
- `security` - Authentication settings
- `logging` - Log level configuration

### Log Directory (`logs/`)

**Purpose**: Application log file storage
**Pattern**: `nhem_YYYYMMDD_HHMMSS.log`
**Features**:
- Timestamped log files
- Automatic log rotation
- Structured log formatting

## Architectural Patterns

### 1. **Layered Architecture**
```
API Layer (routes.py)
├── Core Business Logic (core/*)
│   ├── Data Processing (processor.py)
│   ├── Data Storage (data_store.py)
│   ├── Peak Detection (enhanced_peak_detector.py)
│   └── ROI Capture (roi_capture.py)
└── Utility Layer (utils/*)
```

### 2. **Separation of Concerns**
- **API Layer**: HTTP request/response handling
- **Core Layer**: Business logic and data processing
- **Utility Layer**: Helper functions and image processing
- **Configuration Layer**: Settings and configuration management

### 3. **Dependency Injection**
- Configuration injected via `settings` singleton
- Services injected through import patterns
- Loose coupling between components

### 4. **Thread Safety Design**
- Lock-protected shared data structures
- Separate threads for main processing and WebSocket server
- Thread-safe configuration management

### 5. **Configuration Management Pattern**
- Multi-source configuration (JSON → Environment → Defaults)
- Runtime configuration updates
- Atomic file operations
- Schema validation

This structure provides a clean, maintainable, and scalable foundation for the real-time signal processing system with clear separation of responsibilities and well-defined interfaces between components.