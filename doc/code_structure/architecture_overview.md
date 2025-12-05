# NHEM Backend Architecture Overview

## System Purpose

NHEM (New HEM Monitor) is a sophisticated real-time signal processing system designed for HEM (高回声事件) detection. The system processes data from configurable ROI (Region of Interest) areas, performs advanced peak detection algorithms, and provides real-time data streaming and analysis capabilities.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        EXTERNAL INTERFACES                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Web UI        │  │  Mobile Client  │  │  Third-party    │ │
│  │   (HTTP API)    │  │  (WebSocket)    │  │  Integration    │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────┬─────────────────────┬───────────────────────┘
                  │                     │
                  ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API LAYER                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  FastAPI Server │  │  RESTful API    │  │  WebSocket      │ │
│  │  (Port 8421)    │  │  Endpoints      │  │  Server         │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BUSINESS LOGIC                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  DataProcessor  │  │ Peak Detection  │  │  ROI Capture    │ │
│  │  (45 FPS)       │  │  Engine         │  │  Service        │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   DataStore     │  │ DataBroadcaster │  │ ConfigManager   │ │
│  │  (Storage)      │  │ (60 FPS)        │  │ (JSON Config)   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                     DATA SOURCES                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  Screen Capture │  │  Simulated Data │  │  Configuration  │ │
│  │  (ROI Regions)  │  │  (Sine Waves)   │  │  Files          │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Core Architectural Principles

### 1. **Layered Architecture**
The system follows a strict layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────┐
│              Presentation Layer           │  ← HTTP API, WebSocket
├─────────────────────────────────────────┤
│               Business Layer              │  ← Processing, Detection
├─────────────────────────────────────────┤
│                Data Layer                │  ← Storage, Configuration
└─────────────────────────────────────────┘
```

**Benefits:**
- Clear separation of responsibilities
- Easy testing and maintenance
- Independent component development
- Flexible technology choices per layer

### 2. **Event-Driven Architecture**
The system uses event-driven patterns for real-time processing:

```
Data Sources → Processing Events → Detection Events → Storage Events → Broadcasting Events
```

**Key Events:**
- Frame capture events (45 Hz)
- Peak detection events
- Configuration change events
- Client connection events
- Data update events (60 Hz broadcasting)

### 3. **Multi-threaded Processing**
Concurrent processing for optimal performance:

```
Main Thread:          FastAPI Server (HTTP Requests)
Processing Thread:    DataProcessor (45 FPS data processing)
WebSocket Thread:     SocketServer (real-time connections)
Broadcasting Thread:  DataBroadcaster (60 FPS data distribution)
```

## System Components

### 1. **Data Processing Pipeline**

```
Data Input → Validation → Processing → Analysis → Storage → Distribution
     │           │           │          │         │           │
     ▼           ▼           ▼          ▼         ▼           ▼
ROI/Simulated   Parameter   Peak      Quality   Thread-     WebSocket
Data           Validation   Detection  Scoring   Safe Storage  Clients
```

**Key Characteristics:**
- **High Frequency**: 45 FPS processing loop
- **Dual Data Sources**: Real ROI + Simulated fallback
- **Advanced Detection**: Multi-algorithm peak detection
- **Thread Safety**: Lock-protected data structures
- **Real-time Distribution**: 60 FPS broadcasting

### 2. **Peak Detection Engine**

```
Signal Input → Buffer Management → Baseline Calculation → Slope Analysis → Peak Detection → Classification → Quality Scoring
```

**Detection Algorithms:**
- **Threshold-based**: Simple amplitude detection
- **Morphological**: Shape-based peak identification
- **Multi-method Slope**: Multiple slope calculation approaches
- **Adaptive Thresholding**: Dynamic baseline adjustment

**Peak Classification:**
- **Green Peaks**: Stable, high-confidence peaks
- **Red Peaks**: Unstable, lower-confidence peaks

### 3. **ROI Capture System**

```
Screen Capture → ROI Extraction → Grayscale Conversion → Value Calculation → Encoding → API Distribution
```

**Features:**
- **Configurable Regions**: Flexible ROI configuration
- **High-Performance**: PIL-based image processing
- **Rate Limited**: Configurable capture frequency
- **Fallback Handling**: Graceful degradation
- **API Integration**: Base64 encoding for web transmission

### 4. **Real-time Communication**

```
Data Updates → Message Formatting → Client Filtering → WebSocket Broadcasting → Client Reception
```

**Communication Features:**
- **Dual Protocol**: HTTP REST + WebSocket
- **High Frequency**: 60 FPS data broadcasting
- **Client Filtering**: Subscription-based distribution
- **Authentication**: Password-protected access
- **Message Sequencing**: Ordered delivery

## Data Flow Architecture

### 1. **Processing Flow**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Source   │───►│  DataProcessor  │───►│ Peak Detection  │
│                 │    │    (45 FPS)     │    │   Engine        │
│ • ROI Capture   │    │                 │    │                 │
│ • Simulation    │    │ • Timing        │    │ • Multi-Method  │
│ • Fallback      │    │ • Integration   │    │ • Classification│
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   DataStore     │    │   DataStore     │
                       │                 │    │                 │
                       │ • Frame Buffer  │    │ • Peak Buffer   │
                       │ • ROI Buffer    │    │ • Statistics    │
                       │ • Thread Safe   │    │ • Status Track  │
                       └─────────────────┘    └─────────────────┘
```

### 2. **Distribution Flow**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   DataStore     │───►│ DataBroadcaster │───►│ SocketServer    │
│                 │    │    (60 FPS)     │    │                 │
│ • Real-time     │    │                 │    │ • WebSocket     │
│ • Historical    │    │ • Change Detect │    │ • Client Mgmt   │
│ • Statistics    │    │ • Message Queue │    │ • Auth          │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │   API Layer     │
                                              │                 │
                                              │ • HTTP Endpoints│
                                              │ • RESTful API   │
                                              │ • Real-time Data│
                                              └─────────────────┘
```

## Configuration Architecture

### 1. **Multi-Source Configuration**

```
Priority Order: Environment Variables > JSON File > Code Defaults

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Environment    │───►│   AppConfig     │◄───│  Code Defaults  │
│  Variables      │    │                 │    │                 │
│ (NHEM_* prefix) │    │ • Validation    │    │ • Base Values   │
│ • Runtime       │    │ • Type Conv.    │    │ • Fallbacks     │
│ • Overrides     │    │ • Reloadable    │    │ • Constants     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                ▲
                                │
                       ┌─────────────────┐
                       │ JSON Config     │
                       │ (fem_config)    │
                       │                 │
                       │ • Persistent    │
                       │ • Structured    │
                       │ • Validated     │
                       └─────────────────┘
```

### 2. **Configuration Categories**

```json
{
  "server": {
    "host": "0.0.0.0",
    "api_port": 8421,
    "socket_port": 30415,
    "enable_cors": true
  },
  "data_processing": {
    "fps": 45,
    "buffer_size": 100,
    "max_frame_count": 10000
  },
  "roi_capture": {
    "frame_rate": 2,
    "default_config": {"x1": 1480, "y1": 480, "x2": 1580, "y2": 580}
  },
  "peak_detection": {
    "threshold": 105.0,
    "margin_frames": 6,
    "difference_threshold": 2.1
  },
  "security": {
    "password": "31415"
  }
}
```

## Performance Architecture

### 1. **High-Frequency Processing**

```
Processing Loops:
├── Main Processing: 45 FPS (22.22ms intervals)
├── Broadcasting: 60 FPS (16.67ms intervals)
├── ROI Capture: Configurable (2-60 FPS)
└── API Requests: Event-driven (no fixed rate)

Timing Precision:
├── Sub-millisecond accuracy
├── Adaptive sleep timing
├── Performance monitoring
└── Load balancing
```

### 2. **Memory Management**

```
Memory Architecture:
├── Circular Buffers (bounded memory)
│   ├── Frame Buffer: 100 frames
│   ├── ROI Buffer: 500 frames
│   └── Peak Buffer: Configurable
├── Automatic Cleanup (old data removal)
├── Thread-Safe Operations
└── Memory Usage Monitoring

Performance Optimizations:
├── PIL Image Cleanup
├── NumPy Array Management
├── Connection Pooling
└── Message Queue Optimization
```

### 3. **Concurrency Design**

```
Thread Responsibilities:
├── Main Thread: FastAPI server, HTTP requests
├── Processing Thread: Data generation, peak detection
├── WebSocket Thread: Real-time connections
├── Broadcasting Thread: Data distribution
└── Timer Threads: Various periodic tasks

Synchronization:
├── Locks (for shared data structures)
├── Events (for thread control)
├── Queues (for message passing)
└── Atomic Operations (for counters)
```

## Security Architecture

### 1. **Authentication & Authorization**

```
Security Layers:
├── Password Authentication (control commands)
├── WebSocket Authentication (connection establishment)
├── CORS Configuration (web access control)
└── Input Validation (parameter checking)

Protected Operations:
├── System Control (start/stop/pause)
├── Configuration Updates
├── ROI Configuration Changes
└── Administrative Functions
```

### 2. **Data Protection**

```
Data Security:
├── Input Validation (Pydantic models)
├── Type Checking (runtime validation)
├── Range Validation (parameter bounds)
├── SQL Injection Prevention (no database)
└── XSS Protection (output encoding)

Error Handling:
├── Standardized Error Responses
├── Secure Error Messages
├── Logging (security events)
└── Exception Handling (graceful degradation)
```

## Scalability Architecture

### 1. **Horizontal Scalability**

```
Scaling Strategies:
├── Stateless API Design
├── Connection Pooling
├── Load Balancing Ready
├── Configuration Externalization
└── Resource Isolation

Bottleneck Analysis:
├── CPU (intensive processing)
├── Memory (buffer management)
├── I/O (file operations)
└── Network (WebSocket connections)
```

### 2. **Vertical Scalability**

```
Performance Tuning:
├── Algorithm Optimization (O(1) operations)
├── Memory Efficiency (circular buffers)
├── I/O Optimization (batching)
└── CPU Utilization (multi-threading)

Resource Management:
├── Connection Limits
├── Memory Limits
├── Processing Limits
└── Rate Limiting
```

## Technology Stack

### 1. **Core Technologies**

```
Web Framework: FastAPI 0.115.0
ASGI Server: Uvicorn 0.30.6
Language: Python 3.8+
Data Validation: Pydantic >=2.0.0
Image Processing: Pillow >=8.0.0
Numerical Computing: NumPy >=1.21.0
Real-time: WebSockets + Asyncio
Configuration: Pydantic Settings
```

### 2. **Architecture Patterns**

```
Design Patterns:
├── Singleton Pattern (global instances)
├── Observer Pattern (event broadcasting)
├── Factory Pattern (FastAPI app creation)
├── Strategy Pattern (peak detection algorithms)
└── Decorator Pattern (middleware)

Architectural Styles:
├── REST API (HTTP endpoints)
├── Event-Driven (real-time processing)
├── Microservice-ready (component isolation)
└── Layered Architecture (separation of concerns)
```

This architecture provides a robust, scalable, and maintainable foundation for real-time signal processing with advanced peak detection capabilities. The design emphasizes performance, reliability, and flexibility while maintaining clean separation of concerns and comprehensive error handling.