# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NHEM (New HEM Monitor) is a real-time signal processing system for HEM (高回声事件 - High Echo Event) detection and monitoring. The system consists of three main components:

1. **Backend** - FastAPI-based real-time data processing server (Python)
2. **Frontend** - HTML5 Canvas-based real-time visualization web UI
3. **Python Client** - Desktop monitoring client with matplotlib plotting

## Development Commands

### Backend Development
```bash
# Navigate to backend directory
cd backend

# Install dependencies
pip install -r requirements.txt

# Run the backend server
python run.py

# Test and validation scripts
python test_syntax_check.py          # Syntax validation
python test_compile.py               # Compilation test
python test_enhanced_endpoint.py     # Enhanced API endpoint test
python test_memory_management.py     # Memory management validation
python test_line_intersection.py     # Line detection functionality test

# The backend runs on http://localhost:8421
# API endpoints available at http://localhost:8421/docs
# Health check at http://localhost:8421/health
# System status at http://localhost:8421/status
# Line detection API at /line_detection/*
```

### Frontend Development
```bash
# Navigate to frontend directory
cd front

# Serve with Python (recommended)
python -m http.server 3000

# Or serve with Node.js
npx http-server -p 3000

# Access frontend at http://localhost:3000
```

### Python Client Development
```bash
# Navigate to client directory
cd python_client

# Install dependencies
pip install matplotlib requests pillow numpy opencv-python  # Add OpenCV for line detection

# Run full GUI client (recommended)
python http_realtime_client.py

# Run simple client
python simple_http_client.py

# CLI usage
python client.py --help
python client.py status
python client.py start-detection --password 31415

# Line detection testing and validation
python test_line_detection.py           # Line detection API test
python line_detection_api_example.py    # Line detection API usage example
python validate_task32.py               # Task 32 validation script
```

## Architecture Overview

### Backend Architecture (`backend/`)
- **Entry Point**: `run.py` - FastAPI application startup with uvicorn server
- **Core Processing**: `app/core/processor.py` - 45 FPS data processing loop with configurable parameters
- **Data Storage**: `app/core/data_store.py` - Thread-safe circular buffers (100-10000 frames)
- **Peak Detection**: `app/core/enhanced_peak_detector.py` - Advanced detection with threshold, margin_frames, difference_threshold
- **API Layer**: `app/api/routes.py` - Complete RESTful API (30+ endpoints)
- **WebSocket**: `app/core/socket_server.py` - Real-time data streaming via port 30415
- **Configuration**: `app/fem_config.json` - Main configuration with NHEM_* environment variable support
- **ROI Capture**: `app/core/roi_capture.py` - Dual ROI system with screen capture at separate frame rate
- **Data Broadcasting**: `app/core/data_broadcaster.py` - 60 FPS WebSocket data streaming
- **Config Manager**: `app/core/config_manager.py` - Runtime configuration updates and persistence
- **Line Detection**: `app/core/line_intersection_detector.py` - Advanced green line detection and intersection analysis (NEW)

### Frontend Architecture (`front/`)
- **Single File Application**: `index.html` - Complete SPA (135KB)
- **Technology**: HTML5 + CSS3 + Vanilla JavaScript (ES6+)
- **Real-time Charting**: HTML5 Canvas (20 FPS updates)
- **UI Theme**: VS Code-style dark theme
- **Configuration**: `config.json` - Frontend settings

### Python Client Architecture (`python_client/`)
- **Main GUI Client**: `http_realtime_client.py` - Complete monitoring application with ROI capture and real-time plotting
- **Plotting Component**: `realtime_plotter.py` - Matplotlib real-time plotting at 20 FPS
- **API Client**: `client.py` - Command-line interface for system control
- **Simple Client**: `simple_http_client.py` - Lightweight monitoring client
- **Line Detection API Client**: `line_detection_api_client.py` - Advanced line detection API integration (NEW)
- **Line Detection Widget**: `line_detection_widget.py` - GUI component for line detection control (NEW)
- **Configuration**: `http_client_config.json` - Client-specific ROI and peak detection settings
- **Line Detection State**: `line_detection_state.json` - Persistent line detection configuration (NEW)
- **Local Config**: `local_config_loader.py` - Configuration management utilities

## Key Performance Characteristics

### Backend Performance
- **Main Processing**: 45 FPS (22.22ms intervals)
- **WebSocket Broadcasting**: 60 FPS (16.67ms intervals)
- **Memory Management**: Circular buffers (100-10000 frames)
- **API Response Time**: <50ms for most endpoints

### Frontend Performance
- **Update Rate**: 20 FPS (50ms intervals)
- **Canvas Rendering**: Optimized double buffering
- **Memory Usage**: ~2MB for data buffers

### Python Client Performance
- **Update Rate**: 20 FPS (configurable)
- **Memory Usage**: ~50MB for GUI + data
- **Multi-threading**: Separate threads for GUI and network

## Configuration Files

### Backend Configuration (`backend/app/fem_config.json`)
Main backend configuration with:
- Server settings (host, ports)
- Data processing parameters (FPS, buffer sizes)
- ROI capture configuration
- Peak detection parameters
- Security settings

### Frontend Configuration (`front/config.json`)
Frontend-specific settings:
- Server connection details
- Display parameters (FPS, Y-axis range)
- ROI capture settings
- Peak detection thresholds

### Python Client Configuration (`python_client/http_client_config.json`)
Client-specific configuration:
- Server connection settings
- ROI parameters
- Peak detection settings

## Data Flow Architecture

```
Backend Processing Pipeline:
Data Sources → DataProcessor(45FPS) → Peak Detection → DataStore → DataBroadcaster(60FPS) → WebSocket Clients
                                   ↓
                            Dual ROI Capture Service(4FPS) → ROI1 + ROI2 Extraction → Base64 Encoding
                                   ↓
                            Line Intersection Detection → Green Line Analysis → Intersection Points
                                   ↓
                            ConfigManager ← Runtime API Updates → Persistence

Frontend Data Flow:
Backend API → HTTP Polling(20FPS) → Canvas Rendering → UI Updates → User Interaction → Configuration Updates

Python Client Data Flow:
Backend API → Async Requests (20FPS) → Matplotlib Rendering → GUI Updates → ROI Controls → Line Detection API Integration
```

## Manual Control Requirements

**Important**: The NHEM system requires manual control to start data processing:

1. **Server Startup**: Backend starts automatically but data processing remains stopped
2. **Detection Control**: Frontend/client must send "start_detection" command with password
3. **ROI Configuration**: Must be configured before detection can begin
4. **Manual Dependencies**: No automatic startup - all processes require explicit user initiation
5. **Password Authentication**: All control commands require password (default: 31415)

## Important Development Notes

### Backend Development
- Uses FastAPI with uvicorn, Pydantic for data validation and structured responses
- Thread-safe circular buffers with locks for concurrent data processing
- Peak detection classifies peaks as green (stable) or red (unstable) based on signal characteristics
- **Dual ROI System**: ROI1 (large region) and ROI2 (50x50 center region) for coordinated analysis (NEW)
- **Line Detection**: Advanced green line detection using OpenCV with intersection point analysis (NEW)
- ROI capture integrates with screen capture via `app/core/roi_capture.py` with fallback to simulated data
- Configuration hierarchy: JSON file (`app/fem_config.json`) → Environment variables (`NHEM_*` prefix) → Code defaults
- Runtime configuration updates via API endpoints with persistence in JSON format
- Medical-grade error handling with severity classification and recovery mechanisms (NEW)

### Frontend Development
- Single-page application architecture
- Uses modern JavaScript (ES6+) patterns
- Canvas-based rendering for performance
- Mock mode available for offline development
- Responsive VS Code-style UI design

### Python Client Development
- Supports multiple client variants (Full GUI, Simple GUI, CLI)
- Matplotlib integration for real-time plotting
- Tkinter for GUI components
- **Line Detection Integration**: API client with automatic retry, timeout control, and status management (NEW)
- **Widget System**: Modular GUI components for line detection control and visualization (NEW)
- Configuration hierarchy: JSON → environment variables → defaults
- Persistent state management for line detection configuration

## Testing and Debugging

### Backend Testing
- API endpoints documented at http://localhost:8421/docs (30+ endpoints)
- Health check available at `/health` - returns server status and configuration
- System status at `/status` - detailed system information and processing state
- Real-time data at `/data/realtime?count=N` - fetch latest N data points
- **Line Detection API**: `/line_detection/*` endpoints for green line detection and intersection analysis (NEW)
- **Enhanced Real-time Endpoint**: `/enhanced_realtime` for combined data and visualization (NEW)
- Control commands require password authentication via `/control/start_detection`, `/control/stop_detection`
- Configuration management via `/config/get`, `/config/update` endpoints

### Frontend Debugging
- Browser developer tools for network requests
- Mock mode switch for offline development
- Console logging for debugging
- Real-time status display

### Python Client Debugging
- Detailed logging output with structured format
- API connection testing with `client.py status`
- Configuration validation on startup from `http_client_config.json`
- Real-time plotting diagnostics in matplotlib window
- Network error handling with retry logic and connection status display
- ROI capture debugging with base64 image preview
- **Line Detection Debugging**: API client integration testing with `line_detection_api_client.py` (NEW)
- Test scripts available: `test_line_detection.py`, `test_line_intersection.py`, `test_memory_management.py` (NEW)

## Security Considerations

- Password-based authentication for control commands (default: 31415)
- CORS configuration for web access
- Input validation via Pydantic models
- No sensitive data in client-side configurations

## Project Structure

```
NHEM/
├── backend/                    # FastAPI backend server
│   ├── app/
│   │   ├── api/               # REST API endpoints (30+ endpoints)
│   │   ├── core/              # Core processing components
│   │   │   ├── processor.py   # 45 FPS data processing loop
│   │   │   ├── data_store.py  # Thread-safe circular buffers
│   │   │   ├── enhanced_peak_detector.py  # Advanced detection
│   │   │   └── socket_server.py  # 60 FPS WebSocket streaming
│   │   ├── config.py          # Pydantic-based configuration
│   │   └── models.py          # Data models
│   ├── requirements.txt       # Python dependencies
│   ├── fem_config.json        # Main configuration file
│   └── run.py                # Application entry point
├── front/                     # HTML5 frontend
│   ├── index.html            # Single-page application (135KB)
│   └── config.json           # Frontend configuration
├── python_client/             # Python monitoring clients
│   ├── run_realtime_client.py # Full GUI client
│   ├── simple_http_client.py # Simple GUI client
│   ├── client.py             # CLI interface
│   ├── realtime_plotter.py   # Matplotlib plotting component
│   └── http_client_config.json
├── doc/                       # Comprehensive documentation
│   ├── code_structure/       # Architecture documentation (11 files)
│   └── single_task/          # Task-specific documentation
├── .claude/                   # Claude Code specifications
│   ├── commands/             # Custom slash commands
│   ├── agents/               # Specialized agent configurations
│   └── templates/            # Document templates
└── CLAUDE.md                 # This file
```

## Key Configuration Parameters

### Backend Processing (`backend/app/fem_config.json`)
- **Data Processing FPS**: 45 (configurable via `NHEM_FPS`)
- **ROI Capture Rate**: 4 FPS (separate from main processing)
- **WebSocket Streaming**: 60 FPS to real-time clients
- **Buffer Sizes**: 100-10000 frames for memory management
- **Peak Detection**: threshold (104.0), margin_frames (5), difference_threshold (1.1), min_region_length (5)
- **Dual ROI**: ROI1 (1100x500) + ROI2 (50x50 center extraction) (NEW)
- **Line Detection**: OpenCV-based green line detection with configurable HSV thresholds (NEW)
- **Performance Monitoring**: Memory tracking, processing time metrics, alert thresholds (NEW)
- **Default ROI**: x1=1480, y1=480, x2=1580, y2=580

### Client Configuration (`python_client/http_client_config.json`)
- **Polling Interval**: 50ms (20 FPS) for real-time updates
- **ROI Settings**: Default region coordinates and capture parameters
- **Peak Detection**: Client-side thresholds for local analysis
- **Server Connection**: localhost:8421 with timeout settings
- **Line Detection API**: Integration settings for line detection client (NEW)
- **State Management**: Persistent configuration file path and recovery settings (NEW)

### Frontend Configuration (`front/config.json`)
- **Update Rate**: 20 FPS canvas rendering
- **Display Range**: Y-axis 0-200 for signal visualization
- **Server Connection**: API endpoint and WebSocket configuration

## Additional Development Tools

### Custom Claude Code Commands
The project includes specialized slash commands in `.claude/commands/`:
- `/spec-*` commands for specification management and requirements tracking
- `/bug-*` commands for bug tracking and systematic fixing workflows
- `/dual-roi-system` command for dual ROI functionality development

### Documentation Templates
Professional templates available in `.claude/templates/`:
- Requirements, design, and task templates
- Bug report and analysis templates
- Product and technical specification templates

### Performance Monitoring
- Built-in health check at `/health` - server status and configuration validation
- System status monitoring at `/status` - processing state and performance metrics
- Real-time data access at `/data/realtime?count=N` - fetch latest data points
- WebSocket streaming on port 30415 for real-time client updates
- API rate limiting and connection pooling for scalability

## Development Best Practices

### Backend Development
- Use FastAPI with Pydantic for data validation
- Implement thread-safe data structures for concurrent processing
- Follow layered architecture: API → Core → Data
- Add comprehensive logging with structured format
- Use circular buffers for memory management

### Frontend Development
- Maintain ES6+ JavaScript compatibility
- Optimize for 20 FPS canvas rendering performance
- Use VS Code-style UI theme for consistency
- Implement proper error boundaries and retry logic
- Test cross-browser compatibility

### Python Client Development
- Follow PEP 8 style guidelines with type hints
- Support multiple client variants (GUI/CLI)
- Use matplotlib for real-time plotting with hardware acceleration
- Implement proper exception handling and retry logic
- Support Python 3.8+ compatibility

## Advanced Features

### Dual ROI System
The system supports coordinated dual ROI processing:
- **ROI1**: Large capture region (1100x500 default) for user display
- **ROI2**: Small center region (50x50) extracted from ROI1 for precise analysis
- **Coordinate Transformation**: Automatic mapping between screen and image coordinates
- **Adaptive Sizing**: ROI2 scales proportionally with ROI1 dimensions
- **Independent Processing**: Separate frame rates for capture vs analysis

### Line Detection and Intersection Analysis
Advanced computer vision capabilities using OpenCV:
- **Green Line Detection**: HSV color space filtering for green line identification
- **Intersection Point Calculation**: Geometric analysis of line crossing points
- **Performance Monitoring**: Memory usage tracking and processing time metrics
- **Medical-Grade Error Handling**: Severity classification with automatic recovery
- **Real-time Processing**: Configurable detection parameters and timeout controls

### Enhanced API Endpoints
New API capabilities for advanced functionality:
- **Line Detection API**: `/line_detection/*` endpoints for detection control and results
- **Enhanced Real-time**: `/enhanced_realtime` for combined data and visualization
- **Performance Monitoring**: Built-in metrics collection and alerting
- **State Management**: Persistent configuration and recovery mechanisms

### Line Detection Client Integration
Python client features for line detection:
- **API Client Class**: Automatic retry, session management, and error handling
- **GUI Widget**: Tkinter-based control panel for line detection operations
- **State Persistence**: Configuration saved in `line_detection_state.json`
- **Status Monitoring**: Real-time detection status and performance metrics
- **Chinese Language Support**: Localized interface elements and documentation