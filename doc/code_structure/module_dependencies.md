# Module Dependencies Analysis

## Dependency Graph Overview

```
run.py
└── app.api.routes
    ├── app.config (settings)
    ├── app.logging_config (init_logging)
    ├── app.models (Pydantic models)
    ├── app.core.data_store (data_store)
    ├── app.core.processor (processor)
    ├── app.core.roi_capture (roi_capture_service)
    ├── app.core.config_manager (get_config_manager)
    ├── app.utils (create_roi_data_with_image, generate_waveform_image_with_peaks)
    └── app.peak_detection (detect_peaks)

Core Module Dependencies:
├── data_store.py (minimal external deps)
├── config_manager.py (independent utility)
├── enhanced_peak_detector.py (algorithm-focused)
├── roi_capture.py → PIL, data_store
├── processor.py → data_store, enhanced_peak_detector, roi_capture
├── socket_server.py → asyncio, websockets
├── data_broadcaster.py → socket_server
└── roi_image_generator.py → PIL, numpy
```

## Detailed Dependency Analysis

### 1. **Entry Point Dependencies**

#### `run.py`
```python
Direct Dependencies:
- app.api.routes.app (FastAPI application)
- app.config.settings (configuration)
- app.core.processor.processor (data processing engine)
- app.logging_config.init_logging (logging setup)

Indirect Dependencies:
- All modules imported by routes.py
- Configuration system via settings
- Processing engine for data generation
```

### 2. **API Layer Dependencies**

#### `app/api/routes.py`
```python
Direct Imports:
- app.config.settings
- app.logging_config.init_logging
- app.models (25+ Pydantic models)
- app.core.data_store.data_store
- app.core.processor.processor
- app.core.roi_capture.roi_capture_service
- app.utils (image generation functions)
- app.peak_detection.detect_peaks

External Dependencies:
- fastapi (web framework)
- pydantic (data validation)
- PIL (image processing via utils)
- numpy (numerical operations via peak_detection)

Conditional Dependencies:
- app.core.config_manager (loaded dynamically in endpoint handlers)
```

### 3. **Core Module Dependencies**

#### `app/core/processor.py`
```python
Direct Dependencies:
- app.config.settings
- app.models (SystemStatus, EnhancedPeakSignal models)
- app.core.data_store.data_store
- app.core.enhanced_peak_detector (EnhancedPeakDetector, PeakDetectionConfig)
- app.core.roi_capture.roi_capture_service

External Dependencies:
- threading (background processing)
- time (precise timing control)
- math (signal generation)
- datetime (timestamping)
- logging (structured logging)

Runtime Dependencies:
- Dynamically reloads configuration via AppConfig.reload_from_json()
```

#### `app/core/data_store.py`
```python
Direct Dependencies:
- app.models (Frame, RoiFrame, SystemStatus models)
- app.config.settings (buffer sizes, timeouts)

External Dependencies:
- threading (locks for thread safety)
- datetime (timestamping)
- collections (deque for circular buffers)
- logging (status logging)

Minimal Dependencies:
- Designed to be lightweight and dependency-free for performance
- No external service dependencies
```

#### `app/core/enhanced_peak_detector.py`
```python
Direct Dependencies:
- app.config.settings (configuration values)
- dataclasses (PeakRegion dataclass)

External Dependencies:
- typing (type hints)
- logging (algorithm debugging)

Algorithm Dependencies:
- numpy (numerical calculations, imported locally)
- math (mathematical functions)

Self-Contained:
- Minimal external dependencies for performance
- Pure algorithm implementation
```

#### `app/core/roi_capture.py`
```python
Direct Dependencies:
- app.config.settings (ROI configuration)
- app.core.data_store.data_store (ROI history storage)
- app.models (RoiData, RoiConfig models)

External Dependencies:
- PIL (Pillow) - Image processing and screenshot capture
- io (BytesIO for image encoding)
- base64 (image encoding for API)
- threading (synchronization)
- logging (capture status)
- time (frame rate control)
- datetime (timestamping)

System Dependencies:
- Platform-specific screenshot APIs
- Window system integration
```

#### `app/core/socket_server.py`
```python
Direct Dependencies:
- app.config.settings (server configuration)

External Dependencies:
- asyncio (async programming)
- websockets (WebSocket protocol)
- json (message serialization)
- threading (thread management)
- logging (connection logging)
- datetime (timestamping)
- typing (type hints)

Network Dependencies:
- TCP socket programming
- WebSocket protocol implementation
- Client connection management
```

#### `app/core/data_broadcaster.py`
```python
Direct Dependencies:
- app.core.socket_server.socket_server
- app.core.data_store.data_store

External Dependencies:
- asyncio (async broadcasting)
- json (message serialization)
- threading (broadcast thread)
- time (timing control)
- logging (broadcast status)
- typing (type hints)

Runtime Dependencies:
- Depends on socket_server for client connections
- Depends on data_store for real-time data access
```

#### `app/core/config_manager.py`
```python
Direct Dependencies:
- app.config.settings (configuration defaults)

External Dependencies:
- json (configuration file parsing)
- pathlib (file path operations)
- threading (thread safety)
- logging (configuration operations)
- typing (type hints)

File System Dependencies:
- File I/O operations
- Atomic file writing
- Configuration validation
```

### 4. **Utility Module Dependencies**

#### `app/utils/roi_image_generator.py`
```python
Direct Dependencies:
- app.core.data_store.data_store (data access for visualization)
- app.models (data models for visualization)

External Dependencies:
- PIL (Pillow) - Image creation and manipulation
- numpy (numerical operations for waveforms)
- io (BytesIO for image encoding)
- base64 (image encoding)
- typing (type hints)
- logging (image generation status)

Image Processing Dependencies:
- PIL for all image operations
- NumPy for numerical computations
- Memory-efficient image encoding
```

### 5. **Configuration Dependencies**

#### `app/config.py`
```python
Direct Dependencies:
- pydantic_settings (BaseSettings)
- pydantic (AnyHttpUrl, validation)
- pathlib (Path for config file)
- typing (type hints)
- logging (configuration status)

Dynamic Dependencies:
- app.core.config_manager (loaded dynamically in __init__)
- JSON configuration file parsing
- Environment variable handling

Validation Dependencies:
- Pydantic validation framework
- Type conversion and validation
```

#### `app/logging_config.py`
```python
Direct Dependencies:
- logging (standard library)
- pathlib (Path for log files)
- datetime (timestamping)
- os (directory operations)

Minimal Dependencies:
- Standard library only
- No external dependencies
```

### 6. **Data Models Dependencies**

#### `app/models.py`
```python
Direct Dependencies:
- pydantic (BaseModel, Field, validation)
- datetime (timestamping)
- enum (enumerations)
- typing (type hints)

Pure Definitions:
- No runtime dependencies
- Validation only through Pydantic
- Type definitions for API contracts
```

### 7. **Peak Detection Dependencies**

#### `app/peak_detection.py`
```python
Direct Dependencies:
- typing (type hints)
- logging (algorithm status)

External Dependencies:
- numpy (numerical operations)

Algorithm Dependencies:
- scipy (signal processing, optional)
- Pure Python implementation for reliability

Performance Dependencies:
- NumPy for vectorized operations
- Optimized algorithm implementations
```

## Dependency Strength Analysis

### **Strong Dependencies** (Required for functionality)
- `routes.py` → `config.py` (Configuration required)
- `routes.py` → `models.py` (API contracts required)
- `processor.py` → `data_store.py` (Data storage required)
- `processor.py` → `enhanced_peak_detector.py` (Peak detection required)
- `data_broadcaster.py` → `socket_server.py` (Broadcasting required)

### **Medium Dependencies** (Can be substituted)
- `roi_capture.py` → `PIL` (Could use other image libraries)
- `enhanced_peak_detector.py` → `numpy` (Could use pure Python)
- `socket_server.py` → `websockets` (Could use other WebSocket libraries)

### **Weak Dependencies** (Optional/Utility)
- `routes.py` → `utils.roi_image_generator` (Image generation for visualization)
- `processor.py` → `roi_capture.py` (Fallback to simulated data)
- `config_manager.py` → `json` (Could use other config formats)

## Circular Dependency Analysis

### **No Circular Dependencies Detected**
The codebase follows a clean dependency hierarchy:
```
Level 0: models.py, logging_config.py (no internal dependencies)
Level 1: config.py, peak_detection.py (minimal dependencies)
Level 2: core modules (depend on Level 0-1)
Level 3: utils, routes.py (depend on all previous levels)
```

### **Dependency Direction Rules**
1. **Core → Configuration**: Core modules can read configuration
2. **API → Core**: API layer uses core functionality
3. **Core → Utils**: Core modules can use utilities
4. **No Reverse Dependencies**: Utils don't depend on core, core doesn't depend on API

## Performance Implications

### **Heavy Dependencies**
- `PIL` (ROI capture): Image processing overhead
- `numpy` (Peak detection): Memory usage for large arrays
- `asyncio/websockets` (Real-time): Memory per connection

### **Lightweight Dependencies**
- `data_store.py`: Minimal overhead, optimized for performance
- `enhanced_peak_detector.py`: Algorithm-focused, minimal I/O
- `config_manager.py`: File I/O only on configuration changes

### **Optimization Opportunities**
1. Lazy loading of heavy dependencies
2. Connection pooling for WebSocket clients
3. Memory pooling for NumPy arrays
4. Asynchronous file I/O for configuration

## Security Considerations

### **External Dependencies Security**
- `PIL`: Image processing vulnerability surface
- `websockets`: Network attack surface
- File I/O operations: Path traversal protection

### **Dependency Updates**
- Regular security updates for PIL/Pillow
- WebSocket library security patches
- NumPy vulnerability monitoring

This dependency analysis shows a well-architected system with clear separation of concerns, minimal circular dependencies, and appropriate use of external libraries for specialized functionality.