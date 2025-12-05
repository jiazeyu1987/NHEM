# NHEM Project Code Structure Documentation

This directory contains comprehensive documentation of the complete NHEM project codebase, including backend, frontend, and Python client architectures.

## Documentation Files

### 1. **Architecture Overview**
- `architecture_overview.md` - High-level system architecture and design principles
- `directory_structure.md` - Detailed breakdown of project directory organization
- `module_dependencies.md` - Module dependency graphs and import relationships

### 2. **Backend Analysis**
- `class_hierarchy.md` - Backend class inheritance and relationship diagrams
- **`dependency_graph.md`** - Visual dependency graphs
- `api_endpoints.md` - Complete backend API endpoint documentation
- `data_flow.md` - Backend data flow through the processing pipeline
- `configuration_system.md` - Backend configuration management architecture

### 3. **Frontend Architecture**
- `frontend_architecture.md` - Complete frontend architecture documentation
  - Single-page application structure
  - Canvas-based real-time charting
  - VS Code-style UI/UX design
  - JavaScript component hierarchy
  - Performance optimization techniques
  - Error handling and debugging

### 4. **Python Client Architecture**
- `python_client_architecture.md` - Comprehensive Python client documentation
  - Multiple client variants (Full GUI, Simple GUI, CLI)
  - Matplotlib real-time plotting
  - Tkinter GUI framework
  - Configuration management
  - API client with retry logic
  - Performance monitoring
  - Testing framework
  - Security considerations

### 5. **Client-Server Interaction**
- `client_server_interaction.md` - Client-server communication patterns
  - API integration methods
  - Real-time data synchronization
  - WebSocket vs HTTP polling
  - Authentication and security
  - Error handling and recovery
  - Performance optimization

## Quick Reference

### Backend Core Architecture
- **Entry Point**: `run.py` - FastAPI application startup
- **Main Processing**: `app/core/processor.py` - 45 FPS data processing loop
- **Data Storage**: `app/core/data_store.py` - Thread-safe circular buffers
- **Peak Detection**: `app/core/enhanced_peak_detector.py` - Advanced detection algorithms
- **API Layer**: `app/api/routes.py` - Complete RESTful API with 30+ endpoints
- **WebSocket**: `app/core/socket_server.py` - Real-time data streaming

### Frontend Architecture
- **Technology Stack**: HTML5 + CSS3 + Vanilla JavaScript (ES6+)
- **Single File**: `front/index.html` (135KB) - Complete SPA application
- **Real-time Charting**: HTML5 Canvas API (20 FPS updates)
- **UI Theme**: VS Code-style dark theme
- **API Integration**: HTTP polling at 50ms intervals (20 FPS)
- **Configuration**: `front/config.json` - Frontend-specific settings

### Python Client Architecture
- **Full GUI**: `run_realtime_client.py` - Complete monitoring application
- **Simple GUI**: `simple_http_client.py` - Lightweight monitoring
- **CLI Tool**: `client.py` - Scriptable API access
- **Real-time Plotting**: Matplotlib with Tkinter integration (20 FPS)
- **Dependencies**: requests, matplotlib, pillow, numpy
- **Configuration**: `http_client_config.json` - Client settings

## System Data Flow

```
Backend Processing Pipeline:
Data Sources → DataProcessor(45FPS) → Peak Detection → DataStore → DataBroadcaster(60FPS) → WebSocket Clients
                                   ↓
                            ROI Capture Service(5FPS) → Image Processing → Base64 Encoding

Frontend Data Flow:
Backend API → HTTP Polling(20FPS) → Canvas Rendering → UI Updates → User Interaction

Python Client Data Flow:
Backend API → Async Requests → Matplotlib Rendering → GUI Updates → User Controls
```

## Performance Characteristics

### Backend Performance
- **Main Processing**: 45 FPS (22.22ms intervals)
- **WebSocket Broadcasting**: 60 FPS (16.67ms intervals)
- **Memory Management**: Circular buffers (100-10000 frames)
- **Thread Safety**: Lock-protected data structures
- **API Response Time**: <50ms for most endpoints

### Frontend Performance
- **Update Rate**: 20 FPS (50ms intervals)
- **Canvas Rendering**: Optimized double buffering
- **Memory Usage**: ~2MB for data buffers
- **Network Latency**: HTTP polling overhead
- **Browser Compatibility**: Modern browsers with Canvas support

### Python Client Performance
- **Update Rate**: 20 FPS (configurable)
- **Matplotlib Rendering**: Hardware-accelerated when available
- **Memory Usage**: ~50MB for GUI + data
- **Network Efficiency**: Connection pooling and retry logic
- **Multi-threading**: Separate threads for GUI and network

## Key Design Patterns

### Backend Patterns
- **Layered Architecture**: API → Core → Data layers
- **Event-Driven Processing**: Real-time data pipeline
- **Singleton Pattern**: Global service instances
- **Observer Pattern**: Data broadcasting to clients
- **Factory Pattern**: FastAPI application creation

### Frontend Patterns
- **Single Page Application**: All-in-one HTML file
- **Event-Driven UI**: User interaction handling
- **Observer Pattern**: Real-time data updates
- **Module Pattern**: JavaScript component organization
- **Configuration Pattern**: External JSON configuration

### Python Client Patterns
- **Factory Pattern**: Multiple client variants
- **Observer Pattern**: GUI event handling
- **Strategy Pattern**: Different plotting strategies
- **Command Pattern**: CLI command structure
- **Configuration Pattern**: Hierarchical configuration

## Configuration Hierarchy

### Backend Configuration
```
Priority: Environment Variables > JSON File > Code Defaults

fem_config.json → NHEM_* Environment Variables → AppConfig defaults
```

### Frontend Configuration
```
config.json → Browser Settings → Hardcoded defaults
```

### Python Client Configuration
```
http_client_config.json → ~/.nhem/client_config.json → NHEM_* Environment Variables → Defaults
```

## Security Architecture

### Authentication
- **Backend**: Password-based control commands (default: 31415)
- **Frontend**: Public endpoints only (no authentication required)
- **Python Client**: Password storage in configuration files

### Data Protection
- **Input Validation**: Pydantic models for all API endpoints
- **CORS Configuration**: Configurable cross-origin access
- **Error Handling**: Sanitized error messages
- **Rate Limiting**: Built-in rate limiting for control commands

## Development Guidelines

### Backend Development
- Follow FastAPI best practices
- Use Pydantic for all data models
- Implement proper error handling
- Add comprehensive logging
- Write unit tests for core functionality

### Frontend Development
- Maintain ES6+ compatibility
- Use modern JavaScript patterns
- Implement proper error boundaries
- Optimize for 20 FPS performance
- Test across multiple browsers

### Python Client Development
- Follow PEP 8 style guidelines
- Use type hints for better maintainability
- Implement proper exception handling
- Add comprehensive test coverage
- Support Python 3.8+ compatibility

## Testing Strategy

### Backend Testing
- **Unit Tests**: Individual component testing
- **Integration Tests**: API endpoint testing
- **Performance Tests**: Load and stress testing
- **Security Tests**: Authentication and input validation

### Frontend Testing
- **Browser Testing**: Cross-browser compatibility
- **Performance Testing**: 20 FPS rendering validation
- **Error Handling**: Network failure scenarios
- **Usability Testing**: User interaction testing

### Python Client Testing
- **Unit Tests**: Component functionality testing
- **Integration Tests**: API client testing
- **GUI Testing**: Tkinter interface testing
- **Performance Tests**: Real-time plotting performance

## Deployment Considerations

### Backend Deployment
- **Production**: Use Gunicorn/Uvicorn with proper process management
- **Containerization**: Docker support available
- **Environment Variables**: Production configuration via environment
- **Monitoring**: Built-in health checks and status endpoints

### Frontend Deployment
- **Static Files**: Can be served by any web server
- **CDN**: Suitable for content delivery networks
- **Browser Cache**: Proper cache headers for static assets
- **HTTPS**: Recommended for production deployments

### Python Client Deployment
- **Package Distribution**: PyPI package installation
- **Dependencies**: Clear dependency specification
- **Cross-platform**: Windows, macOS, Linux support
- **Virtual Environment**: Recommended for isolation

This comprehensive documentation provides complete architectural understanding of the NHEM system across all its components, enabling effective development, maintenance, and extension of the platform.