# Implementation Plan

## Task Overview
This implementation plan breaks down the dual-ROI system into atomic, manageable tasks that can be executed independently. The approach prioritizes extending existing components while maintaining full backward compatibility. Tasks are organized by component and follow a logical dependency chain from data models through backend implementation to frontend enhancements.

## Steering Document Compliance
All tasks follow existing NHEM patterns and conventions:
- **FastAPI/Pydantic Patterns**: Extend existing models and validation
- **Threading Model**: Maintain thread-safe operations using existing lock patterns
- **Configuration Management**: Leverage multi-layer configuration hierarchy
- **Memory Management**: Use circular buffers with proper cleanup
- **Error Handling**: Follow existing error response and logging patterns

## Atomic Task Requirements
**Each task must meet these criteria for optimal agent execution:**
- **File Scope**: Touches 1-3 related files maximum
- **Time Boxing**: Completable in 15-30 minutes
- **Single Purpose**: One testable outcome per task
- **Specific Files**: Must specify exact files to create/modify
- **Agent-Friendly**: Clear input/output with minimal context switching

## Tasks

### 1. Core Data Models Extension

- [x] 1.1 Create dual-ROI data models in backend/app/models.py
  - File: backend/app/models.py (extend existing)
  - Add DualRoiConfig class with mode, large_roi, small_roi_size fields
  - Add DualRoiFrame class with large_roi, small_roi, timestamp fields
  - Add validation methods for ROI coordinate bounds and minimum sizes
  - Purpose: Establish type-safe data structures for dual-ROI system
  - _Leverage: existing RoiConfig, RoiData, RoiFrame classes_
  - _Requirements: 1.1, 1.2_

- [x] 1.2 Create API response models in backend/app/models.py
  - File: backend/app/models.py (continue from task 1.1)
  - Add RoiConfigResponse class for backward-compatible API responses
  - Add DualRoiFrameResponse class for dual-ROI data endpoints
  - Add helper methods for mode detection and response formatting
  - Purpose: Provide structured API responses supporting both ROI modes
  - _Leverage: existing response model patterns, Pydantic validation_
  - _Requirements: 4.1, 4.2_

### 2. Configuration Management Enhancement

- [x] 2.1 Extend configuration manager in backend/app/core/config_manager.py
  - File: backend/app/core/config_manager.py (modify existing)
  - Add get_dual_roi_config() method with mode detection logic
  - Add set_dual_roi_config() method with validation and migration
  - Add validate_dual_roi_coordinates() method with screen bounds checking
  - Purpose: Provide centralized dual-ROI configuration management
  - _Leverage: existing ConfigManager class, coordinate validation patterns_
  - _Requirements: 1.2, 1.3_

- [x] 2.2 Add migration utilities in backend/app/core/config_manager.py
  - File: backend/app/core/config_manager.py (continue from task 2.1)
  - Add migrate_single_to_dual() method for seamless upgrades
  - Add backup_config() and restore_config() methods for safe migration
  - Add validate_migration() method to ensure data integrity
  - Purpose: Enable safe migration from single to dual-ROI configurations
  - _Leverage: existing configuration persistence, JSON handling utilities_
  - _Requirements: 1.6, Migration requirements_

### 3. Enhanced ROI Capture Service

- [x] 3.1 Extend ROI capture service in backend/app/core/roi_capture.py
  - File: backend/app/core/roi_capture.py (modify existing)
  - Add capture_dual_roi() method returning DualRoiFrame
  - Add extract_small_roi() method with center extraction logic
  - Add set_large_roi_frame_rate() method for independent frame rate control
  - Purpose: Capture both ROI regions with independent frame rates
  - _Leverage: existing RoiCaptureService class, PIL image processing_
  - _Requirements: 2.2, 3.1_

- [x] 3.2 Add dual-ROI validation and error handling in backend/app/core/roi_capture.py
  - File: backend/app/core/roi_capture.py (continue from task 3.1)
  - Add validate_roi_bounds() method with automatic adjustment
  - Add handle_capture_failure() method with graceful fallback
  - Add log_capture_metrics() method for performance monitoring
  - Purpose: Ensure robust dual-ROI capture with proper error handling
  - _Leverage: existing error handling patterns, logging utilities_
  - _Requirements: 2.3, Reliability requirements_

### 4. Data Store Extension

- [x] 4.1 Enhance data store for dual-ROI in backend/app/core/data_store.py
  - File: backend/app/core/data_store.py (modify existing)
  - Add _large_roi_buffer and _large_roi_frame_count attributes
  - Add store_dual_roi_data() method with thread-safe operations
  - Add get_large_roi_history() method for historical data access
  - Purpose: Manage separate buffers for large and small ROI data
  - _Leverage: existing DataStore class, circular buffer patterns, thread locks_
  - _Requirements: 3.5, 3.6_

- [x] 4.2 Add dual-ROI buffer management in backend/app/core/data_store.py
  - File: backend/app/core/data_store.py (continue from task 4.1)
  - Add cleanup_large_roi_buffer() method for memory management
  - Add get_dual_roi_status() method for buffer status reporting
  - Add validate_buffer_sizes() method for memory usage monitoring
  - Purpose: Ensure efficient memory usage and buffer management
  - _Leverage: existing buffer management, cleanup patterns_
  - _Requirements: Performance requirements_

### 5. API Layer Enhancement

- [x] 5.1 Extend ROI configuration endpoints in backend/app/api/routes.py
  - File: backend/app/api/routes.py (modify existing)
  - Modify GET /roi/config to support mode query parameter
  - Modify POST /roi/config to auto-detect single vs dual configuration
  - Add backward-compatible response formatting
  - Purpose: Provide unified API endpoints for both ROI modes
  - _Leverage: existing FastAPI endpoints, authentication middleware_
  - _Requirements: 4.1, 4.2_

- [x] 5.2 Add dual-ROI data endpoints in backend/app/api/routes.py
  - File: backend/app/api/routes.py (continue from task 5.1)
  - Add GET /roi/dual-frame endpoint returning DualRoiFrameResponse
  - Add GET /roi/large-history endpoint for large ROI historical data
  - Add proper error handling and response validation
  - Purpose: Provide access to dual-ROI specific data and history
  - _Leverage: existing endpoint patterns, Pydantic response models_
  - _Requirements: 4.3, 4.4_

### 6. Data Processing Integration

- [x] 6.1 Update data processor for dual-ROI in backend/app/core/processor.py
  - File: backend/app/core/processor.py (modify existing)
  - Modify main processing loop to extract small ROI from large ROI
  - Add dual_roi_mode detection and processing branch
  - Integrate with enhanced ROI capture service
  - Purpose: Process dual-ROI data while maintaining analytical consistency
  - _Leverage: existing DataProcessor class, processing pipeline_
  - _Requirements: 3.1, 3.2_

- [ ] 6.2 Update WebSocket broadcasting for dual-ROI in backend/app/core/socket_server.py
  - File: backend/app/core/socket_server.py (modify existing)
  - Add large ROI data to WebSocket message structure
  - Maintain backward compatibility with existing message format
  - Add dual-ROI mode indicators in message metadata
  - Purpose: Broadcast both ROI data types via WebSocket
  - _Leverage: existing WebSocket broadcasting, message formatting_
  - _Requirements: 4.3, Compatibility requirements_

### 7. Configuration Schema Update

- [ ] 7.1 Update fem_config.json schema in backend/app/fem_config.json
  - File: backend/app/fem_config.json (modify existing)
  - Add dual_roi configuration section with mode, large_roi, small_roi fields
  - Preserve existing roi_capture section for backward compatibility
  - Add validation schema for new configuration fields
  - Purpose: Support dual-ROI configuration with migration path
  - _Leverage: existing configuration structure, JSON schema validation_
  - _Requirements: 1.4, 1.5_

- [ ] 7.2 Update environment variable handling in backend/app/config.py
  - File: backend/app/config.py (modify existing)
  - Add NHEM_DUAL_ROI_MODE environment variable support
  - Add NHEM_LARGE_ROI_* environment variables for configuration
  - Ensure backward compatibility with existing environment variables
  - Purpose: Support dual-ROI configuration via environment variables
  - _Leverage: existing PydanticSettings, environment variable patterns_
  - _Requirements: 1.4, Compatibility requirements_

### 8. Frontend Backend Integration

- [ ] 8.1 Add dual-ROI configuration parsing functions in front/index.html
  - File: front/index.html (modify existing, add to configuration section)
  - Add parseDualRoiConfig(config) function returning DualRoiConfig object
  - Add validateDualRoiConfig(config) function with coordinate validation
  - Add fallback to single-ROI mode when dual config is invalid
  - Purpose: Enable frontend to parse and validate dual-ROI configurations
  - _Leverage: existing parseConfig() and validateConfig() functions_
  - _Requirements: 5.1, 5.2_

- [ ] 8.2 Add mode detection logic in front/index.html
  - File: front/index.html (continue from task 8.1, add to UI initialization section)
  - Add detectRoiMode() function returning 'single' or 'dual'
  - Add initializeModeUI() function to show/hide dual-ROI components
  - Add updateModeIndicator() function to display current mode
  - Purpose: Provide automatic UI adaptation based on ROI mode
  - _Leverage: existing UI initialization patterns, DOM manipulation_
  - _Requirements: 5.1, 5.4_

- [ ] 8.3 Add fetchDualRoiConfig() function in front/index.html
  - File: front/index.html (continue from task 8.2, add to API client section)
  - Create fetchDualRoiConfig() function with GET /roi/config?mode=dual
  - Add proper error handling and response validation
  - Add fallback to legacy endpoint if dual endpoint fails
  - Purpose: Fetch dual-ROI configuration from backend API
  - _Leverage: existing fetchConfig() function pattern_
  - _Requirements: 4.1, 5.3_

- [ ] 8.4 Add fetchDualRoiFrame() function in front/index.html
  - File: front/index.html (continue from task 8.3, add to API client section)
  - Create fetchDualRoiFrame() function with GET /roi/dual-frame
  - Add Base64 decoding for both large and small ROI images
  - Add retry logic with exponential backoff
  - Purpose: Fetch real-time dual-ROI data for display
  - _Leverage: existing fetchData() function pattern_
  - _Requirements: 4.3, 5.3_

### 9. Frontend UI Enhancement

- [ ] 9.1 Add large ROI canvas element in front/index.html
  - File: front/index.html (continue from task 8.4, add to display section)
  - Add HTML canvas element with id="largeRoiCanvas" (300x200 default)
  - Add CSS styling consistent with existing ROI display
  - Add container div for dual-ROI layout management
  - Purpose: Create visual display area for large ROI image
  - _Leverage: existing ROI canvas CSS styling and layout patterns_
  - _Requirements: 5.1, 5.5_

- [ ] 9.2 Add small ROI extraction overlay in front/index.html
  - File: front/index.html (continue from task 9.1, modify large ROI canvas)
  - Add overlay canvas showing 50x50 center extraction area
  - Add semi-transparent rectangle indicator
  - Add coordinate text display showing extraction region
  - Purpose: Visualize small ROI extraction area within large ROI
  - _Leverage: existing overlay patterns, canvas drawing utilities_
  - _Requirements: 5.1, 5.5_

- [ ] 9.3 Add mode switching controls in front/index.html
  - File: front/index.html (continue from task 9.2, add to control panel)
  - Add radio buttons for 'Single ROI' and 'Dual ROI' modes
  - Add mode indicator showing current active mode
  - Add enable/disable logic for dual-ROI specific controls
  - Purpose: Allow users to switch between ROI modes
  - _Leverage: existing form control patterns, event handlers_
  - _Requirements: 5.1, 5.4_

- [ ] 9.4 Add large ROI coordinate inputs in front/index.html
  - File: front/index.html (continue from task 9.3, add to configuration panel)
  - Add input fields for x1, y1, x2, y2 with labels
  - Add validation messages and error indicators
  - Add coordinate range checking and auto-correction
  - Purpose: Provide user interface for large ROI configuration
  - _Leverage: existing ROI coordinate input patterns_
  - _Requirements: 5.1, 5.2_

### 10. Real-time Data Display

- [ ] 10.1 Modify updateDisplay() function for dual-ROI in front/index.html
  - File: front/index.html (continue from task 9.4, modify existing updateDisplay function)
  - Add dual-ROI mode detection and branching logic
  - Add large ROI data parsing from dual-ROI API response
  - Maintain backward compatibility with single-ROI data format
  - Purpose: Update main display function to handle dual-ROI data
  - _Leverage: existing updateDisplay() function, data parsing patterns_
  - _Requirements: 5.1, 5.5_

- [ ] 10.2 Add large ROI image rendering in front/index.html
  - File: front/index.html (continue from task 10.1, add rendering section)
  - Add renderLargeRoi(imageData) function for canvas drawing
  - Add Base64 to image conversion and scaling
  - Add overlay drawing for small ROI extraction area
  - Purpose: Render large ROI image with extraction indicator
  - _Leverage: existing ROI rendering patterns, canvas drawing utilities_
  - _Requirements: 5.1, 5.5_

- [ ] 10.3 Add dual-ROI status indicators in front/index.html
  - File: front/index.html (continue from task 10.2, add status section)
  - Add updateDualRoiStatus() function showing mode and connection state
  - Add validation error display for ROI configuration issues
  - Add connection status indicators for both ROI data streams
  - Purpose: Provide clear feedback on dual-ROI system status
  - _Leverage: existing status display components, CSS styling_
  - _Requirements: 5.4, 3.3_

### 11. Testing Implementation

- [ ] 11.1 Create backend unit tests for dual-ROI models
  - File: tests/test_dual_roi_models.py (create new)
  - Test DualRoiConfig.validation with valid/invalid coordinates
  - Test coordinate boundary adjustment for screen edges
  - Test mode detection logic ('single' vs 'dual')
  - Test small ROI extraction from various large ROI sizes
  - Purpose: Ensure dual-ROI data models work correctly
  - _Leverage: existing test patterns, pytest framework_
  - _Requirements: 1.1, 1.2, 3.1_

- [ ] 11.2 Create integration tests for dual-ROI API endpoints
  - File: tests/test_dual_roi_api.py (create new)
  - Test GET /roi/config backward compatibility (legacy mode)
  - Test GET /roi/config?mode=dual with dual configuration
  - Test POST /roi/config auto-detection of single vs dual
  - Test GET /roi/dual-frame with various configuration states
  - Purpose: Verify API layer functionality and compatibility
  - _Leverage: existing API test patterns, FastAPI test client_
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

### 12. Documentation and Final Integration

- [ ] 12.1 Update API documentation in backend/app/api/routes.py
  - File: backend/app/api/routes.py (continue from task 5.2)
  - Add OpenAPI documentation for new dual-ROI endpoints
  - Update existing endpoint documentation for dual-ROI support
  - Add examples and response format specifications
  - Purpose: Document new API capabilities for developers
  - _Leverage: existing FastAPI documentation patterns_
  - _Requirements: Documentation requirements_

- [ ] 12.2 Update frontend configuration file in front/config.json
  - File: front/config.json (modify existing)
  - Add dual-ROI default configuration settings
  - Preserve existing settings for backward compatibility
  - Add UI layout preferences for dual-ROI mode
  - Purpose: Provide default configuration for dual-ROI frontend
  - _Leverage: existing configuration structure_
  - _Requirements: 5.1, 5.2_

## Implementation Dependencies

Tasks should be executed in numerical order to respect dependencies:
- Tasks 1.x must be completed before 2.x (data models foundation)
- Tasks 2.x must be completed before 3.x (configuration management)
- Tasks 3.x must be completed before 4.x (capture service integration)
- Tasks 4.x-6.x can be executed in parallel (backend components)
- Tasks 7.x must be completed after 1.x-6.x (configuration schema)
- Tasks 8.x-10.x must be completed after backend components (frontend integration)
- Tasks 11.x-12.x should be completed last (testing and documentation)