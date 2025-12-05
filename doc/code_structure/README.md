# NHEM Backend Code Structure Documentation

This directory contains comprehensive documentation of the NHEM backend codebase structure, architecture, and design patterns.

## Documentation Files

### 1. **Architecture Overview**
- `architecture_overview.md` - High-level system architecture and design principles
- `directory_structure.md` - Detailed breakdown of project directory organization
- `module_dependencies.md` - Module dependency graphs and import relationships

### 2. **Core Components Analysis**
- `core_modules.md` - In-depth analysis of core processing modules
- `api_endpoints.md` - Complete API endpoint documentation with request/response models
- `data_models.md` - Data model definitions and relationships
- `configuration_system.md` - Configuration management architecture

### 3. **Data Processing**
- `data_flow.md` - Data flow through the processing pipeline
- `peak_detection.md` - Peak detection algorithms and implementation details
- `roi_processing.md` - ROI capture and processing pipeline

### 4. **Real-time Systems**
- `websocket_server.md` - WebSocket server implementation and client management
- `broadcasting_system.md` - Real-time data broadcasting architecture
- `threading_model.md` - Concurrency and threading design

### 5. **Technical Details**
- `class_hierarchy.md` - Class inheritance and relationship diagrams
- **`dependency_graph.md`** - Visual dependency graphs (this file)
- `api_mapping.md` - API endpoint to function mapping
- `configuration_schema.md` - Complete configuration file schema

## Quick Reference

### Key Architectural Patterns
- **Layered Architecture**: Clear separation between API, Core, and Utility layers
- **Multi-threaded Processing**: Main processing loop + WebSocket server + Data broadcasting
- **Configuration Management**: JSON + Environment variables + Runtime updates
- **Real-time Data Flow**: In-memory circular buffers + WebSocket broadcasting

### Core Processing Pipeline
```
DataProcessor (45 FPS) → EnhancedPeakDetector → DataStore → DataBroadcaster → WebSocket Clients
```

### Configuration Hierarchy
```
fem_config.json → Environment Variables (NHEM_*) → Code Defaults → AppConfig
```

### API Structure
- **System Management**: `/health`, `/status`, `/control`
- **Real-time Data**: `/data/realtime`, `/data/window-capture`, `/data/roi-window-capture`
- **ROI Management**: `/roi/config`, `/roi/frame-rate`, `/roi/capture`
- **Peak Detection**: `/peak-detection/config`, `/data/waveform-with-peaks`
- **Configuration**: `/config/*` (GET, POST, reload, export, import)

## Development Notes

This documentation is automatically generated and should be kept in sync with code changes. When making structural changes:

1. Update the relevant documentation files
2. Regenerate dependency graphs if modules are added/removed
3. Update API endpoint documentation for new endpoints
4. Review configuration schema for any changes

## Tools Used

- **Static Analysis**: Manual code review and import analysis
- **Dependency Mapping**: Custom Python script for import graph generation
- **API Documentation**: Extracted from FastAPI route definitions
- **Configuration Schema**: Generated from Pydantic models and JSON config