# Module Dependency Graph

## Visual Dependency Map

```
┌─────────────────────────────────────────────────────────────────┐
│                        ENTRY POINT                              │
│                           run.py                                │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API LAYER                                  │
│                 app/api/routes.py                               │
└─────────────┬─────────────────────┬─────────────────────────────┘
              │                     │
              ▼                     ▼
┌─────────────────┐    ┌─────────────────────────────────────────────────┐
│   CONFIGURATION │    │                  MODELS                        │
│   app/config.py │    │               app/models.py                    │
│                 │    │                                                 │
│ • Settings      │    │ • Pydantic Models                              │
│ • JSON Loading  │    │ • API Contracts                                │
│ • Environment   │    │ • Validation                                   │
└─────────────────┘    └─────────────────────────────────────────────────┘
              │                     │
              ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CORE LAYER                                  │
│                                                                     │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────┐     │
│  │    data_    │  │   enhanced_  │  │      roi_capture         │     │
│  │    store.py │  │  peak_       │  │        .py              │     │
│  │             │  │  detector.py │  │                         │     │
│  │ • Storage   │  │              │  │ • Screenshot            │     │
│  │ • Buffer    │  │ • Detection  │  │ • Image Processing      │     │
│  │ • ThreadSafe│  │ • Algorithms │  │ • PIL Integration       │     │
│  └─────────────┘  └──────────────┘  └─────────────────────────┘     │
│         │                │                      │                   │
│         └────────────────┼──────────────────────┘                   │
│                          │                                          │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                        processor.py                            │ │
│  │                                                                 │ │
│  │ • Main Processing Loop                                         │ │
│  │ • ROI Integration                                              │ │
│  │ • Peak Detection Coordination                                   │ │
│  │ • Data Storage Coordination                                     │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└─────────────────┬───────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                  REAL-TIME LAYER                               │
│                                                                     │
│  ┌──────────────┐        ┌─────────────────────────────────────┐  │
│  │ socket_      │◄──────►│      data_broadcaster.py             │  │
│  │ server.py    │        │                                     │  │
│  │              │        │ • 60 FPS Broadcasting                │  │
│  │ • WebSocket  │        │ • Client Filtering                   │  │
│  │ • Asyncio    │        │ • Message Queuing                   │  │
│  │ • Client Mgmt│        │ • Real-time Distribution             │  │
│  └──────────────┘        └─────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    UTILITY LAYER                                 │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │             utils/roi_image_generator.py                    │  │
│  │                                                             │  │
│  │ • Base64 Encoding                                           │  │
│  │ • Waveform Visualization                                    │  │
│  │ • Peak Annotation                                           │  │
│  │ • PIL Integration                                           │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │           core/config_manager.py                            │  │
│  │                                                             │  │
│  │ • JSON File Operations                                      │  │
│  │ • Atomic Updates                                            │  │
│  │ • Schema Validation                                         │  │
│  │ • Thread Safety                                             │  │
│  └─────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Dependency Flow Diagram

```
┌──────────────────┐
│   External API   │
│    Requests      │
└─────────┬────────┘
          │
          ▼
┌──────────────────┐    imports    ┌──────────────────┐
│  FastAPI Routes  │──────────────►│   Pydantic       │
│   (routes.py)    │◄──────────────│    Models        │
└─────────┬────────┘               │   (models.py)    │
          │                       └──────────────────┘
          ▼
┌──────────────────┐    reads     ┌──────────────────┐
│   Configuration  │◄──────────────│  JSON Config     │
│    (config.py)   │──────────────►│                  │
└─────────┬────────┘    writes    │                  │
          │                       └──────────────────┘
          ▼
┌──────────────────┐    manages   ┌──────────────────┐
│   Data Store     │◄──────────────│   Processing     │
│  (data_store.py) │──────────────►│   Engine         │
└─────────┬────────┘    consumes   │ (processor.py)   │
          │                       └─────────┬────────┘
          ▼                                 │
┌──────────────────┐    provides            │
│  Peak Detection  │◄────────────────────────┘
│   Algorithms     │
│(peak_detection)  │
└─────────┬────────┘
          │
          ▼
┌──────────────────┐    captures   ┌──────────────────┐
│   ROI Capture    │──────────────►│   Screen         │
│  (roi_capture)   │◄──────────────│   Screenshot     │
└─────────┬────────┘               │                  │
          │                       └──────────────────┘
          ▼
┌──────────────────┐    broadcasts ┌──────────────────┐
│  WebSocket       │◄──────────────│   Real-time      │
│  Server          │──────────────►│   Data           │
│(socket_server)   │    serves     │                  │
└──────────────────┘               └──────────────────┘
```

## Module Interaction Matrix

| Module | routes | config | models | data_store | processor | enhanced_peak | roi_capture | socket_server | utils |
|--------|--------|--------|--------|------------|-----------|---------------|-------------|---------------|-------|
| **routes** | Self | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ |
| **config** | ✗ | Self | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✗ |
| **models** | ✗ | ✗ | Self | ✗ | ✓ | ✓ | ✓ | ✗ | ✓ |
| **data_store** | ✗ | ✓ | ✓ | Self | ✓ | ✗ | ✓ | ✗ | ✓ |
| **processor** | ✗ | ✓ | ✓ | ✓ | Self | ✓ | ✓ | ✗ | ✗ |
| **enhanced_peak** | ✗ | ✓ | ✗ | ✗ | ✗ | Self | ✗ | ✗ | ✗ |
| **roi_capture** | ✗ | ✓ | ✓ | ✓ | ✗ | ✗ | Self | ✗ | ✓ |
| **socket_server** | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | Self | ✗ |
| **utils** | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | Self |

**Legend:**
- ✓ = Direct import dependency
- ✗ = No direct dependency

## Import Dependency Tree

```
run.py
└── app.api.routes
    ├── app.config
    │   └── app.core.config_manager (dynamic import)
    ├── app.logging_config
    ├── app.models
    ├── app.core.data_store
    ├── app.core.processor
    │   ├── app.config
    │   ├── app.models
    │   ├── app.core.data_store
    │   ├── app.core.enhanced_peak_detector
    │   │   ├── app.config
    │   │   └── app.models (via PeakDetectionConfig)
    │   └── app.core.roi_capture
    │       ├── app.config
    │       ├── app.core.data_store
    │       └── app.models
    ├── app.core.roi_capture
    ├── app.utils.roi_image_generator
    │   ├── app.core.data_store
    │   └── app.models
    └── app.peak_detection
```

## Layer Dependencies

### **Layer 1: Foundation (No internal dependencies)**
```
app/models.py              # Pure Pydantic models
app/logging_config.py      # Standard logging setup
```

### **Layer 2: Configuration & Utilities**
```
app/config.py              # Depends on Layer 1 + pydantic_settings
app/core/config_manager.py # Independent utility
app/peak_detection.py     # Algorithm-focused, minimal deps
app/utils/roi_image_generator.py # Depends on models + PIL + numpy
```

### **Layer 3: Core Business Logic**
```
app/core/data_store.py     # Depends on Layer 1 + config
app/core/enhanced_peak_detector.py # Depends on Layer 2
app/core/roi_capture.py    # Depends on Layer 1, 2 + PIL
app/core/socket_server.py  # Depends on config + websockets
app/core/processor.py      # Depends on all previous core modules
app/core/data_broadcaster.py # Depends on socket_server + data_store
```

### **Layer 4: API Layer**
```
app/api/routes.py          # Depends on all previous layers
```

### **Layer 5: Application**
```
run.py                     # Depends on API layer
```

## External Dependencies Map

```
Python Standard Library:
├── threading              # concurrency in processor, socket_server
├── time                   # timing in processor, data_broadcaster
├── datetime               # timestamps everywhere
├── logging                # structured logging
├── json                   # config management, WebSocket messages
├── pathlib                # file operations
├── collections            # deque in data_store
├── typing                 # type hints everywhere
├── io                     # BytesIO for image encoding
├── base64                 # image encoding for API
├── enum                   # SystemStatus enum
└── os                     # directory operations

Third-Party Libraries:
├── fastapi                # web framework
├── uvicorn                # ASGI server
├── pydantic               # data validation
├── pydantic_settings      # configuration management
├── Pillow (PIL)           # image processing
├── numpy                  # numerical operations
└── websockets             # WebSocket server
```

## Runtime Dependency Graph

```
Startup Phase:
run.py → logging_config → config → routes → FastAPI app

Configuration Loading:
config → config_manager → JSON file → Environment variables

Processing Loop:
processor → data_store (storage)
         → enhanced_peak_detector (analysis)
         → roi_capture (data source)
         → config (settings)

Real-time Broadcasting:
data_broadcaster → socket_server → WebSocket clients
                 → data_store (real-time data)

API Requests:
routes → models (validation)
      → data_store (data access)
      → processor (control)
      → utils (visualization)
      → config_manager (configuration updates)
```

## Performance Dependency Analysis

### **Critical Path Dependencies**
1. **Main Processing Loop**: processor → enhanced_peak_detector → data_store
2. **Real-time Broadcasting**: data_broadcaster → socket_server → network I/O
3. **API Response**: routes → data_store → models → JSON serialization

### **Heavy Dependencies**
- **PIL/Pillow**: Image processing in roi_capture and utils
- **NumPy**: Numerical computations in peak_detection and enhanced_peak_detector
- **WebSockets**: Memory per connection in socket_server

### **Lightweight Dependencies**
- **data_store**: Optimized circular buffers, minimal overhead
- **config_manager**: File I/O only on configuration changes
- **models**: Pure Pydantic validation, no runtime overhead

This dependency analysis reveals a well-structured system with clear separation of concerns, minimal circular dependencies, and appropriate use of external libraries for specialized functionality.