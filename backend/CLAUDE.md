# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Start the Server
```bash
python run.py
```
The server starts on http://localhost:8421 by default.

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Testing
No specific test framework is configured. Tests should be run manually by starting the server and making API requests.

## High-Level Architecture

### Project Overview
NHEM (New HEM Monitor) is a FastAPI-based real-time HEM (高回声事件) detection system that processes ROI (Region of Interest) data at configurable frame rates and performs intelligent peak detection.

### Core Components

**Data Flow Architecture:**
- **DataProcessor** (`app/core/processor.py`) - Main processing loop that generates or captures ROI data at configured FPS
- **DataStore** (`app/core/data_store.py`) - Thread-safe circular buffer for storing time-series data
- **EnhancedPeakDetector** (`app/core/enhanced_peak_detector.py`) - Advanced peak detection with configurable parameters
- **ROICaptureService** (`app/core/roi_capture.py`) - Real-time ROI screenshot capture from screen

**Configuration System:**
- **Multi-layer Configuration** - JSON file (`app/fem_config.json`) + Environment Variables (`NHEM_` prefix) + Code defaults
- **ConfigManager** (`app/core/config_manager.py`) - Handles runtime configuration updates and persistence
- **AppConfig** (`app/config.py`) - Pydantic-based settings with validation

**API Layer** (`app/api/routes.py`):
- RESTful endpoints for real-time data, control commands, configuration management
- WebSocket support for real-time data streaming (socket port 30415)
- Comprehensive error handling with structured responses

### Key Design Patterns

**Threading Model:**
- Main thread: FastAPI server
- Background thread: DataProcessor running at configured FPS
- Thread-safe data structures using locks and circular buffers

**Data Processing Pipeline:**
1. DataProcessor generates signals or captures ROI data
2. EnhancedPeakDetector analyzes for peaks using configurable thresholds
3. Results stored in DataStore circular buffer
4. API endpoints provide real-time access to processed data

**Configuration Management:**
- Runtime configuration changes via API endpoints
- Persistent storage in JSON format
- Environment variable override support

### Important Implementation Details

**Peak Detection Algorithm:**
- Three-parameter detection: threshold, margin_frames, difference_threshold
- Classifies peaks as green (stable) or red (unstable) based on signal characteristics
- Configurable minimum region length to filter noise

**ROI Processing:**
- Configurable rectangular regions with validation
- Real-time screenshot capture at separate frame rate from main processing
- Fallback to simulated data if ROI capture fails

**Memory Management:**
- Circular buffers prevent memory leaks
- Configurable buffer sizes for different data types
- Automatic cleanup of old data

### Environment Variables (NHEM_ prefix)
- `NHEM_HOST` - Server address (default: 0.0.0.0)
- `NHEM_API_PORT` - HTTP API port (default: 8421)
- `NHEM_SOCKET_PORT` - WebSocket port (default: 30415)
- `NHEM_FPS` - Data processing frame rate (default: 45)
- `NHEM_LOG_LEVEL` - Logging level (default: INFO)
- `NHEM_PASSWORD` - Admin password for control commands (default: 31415)

### Configuration File Structure
The `app/fem_config.json` contains all runtime configuration:
- Server settings (ports, CORS)
- Data processing parameters (FPS, buffer sizes)
- ROI capture settings (frame rate, default coordinates)
- Peak detection thresholds and parameters
- Security settings

### Manual Control Flow
The system requires manual startup:
1. Server starts automatically via `python run.py`
2. Data processing remains stopped until frontend sends "start_detection" command
3. ROI must be configured before detection can start
4. All control commands require password authentication

### Migration Notes
This project was migrated from "NewFEM" to "NHEM":
- Environment variable prefix changed from `NEWFEM_` to `NHEM_`
- Log file prefix changed from "newfem_" to "nhem_"
- Default FPS changed from 60 to 45 to match frontend
- All core functionality and API compatibility preserved