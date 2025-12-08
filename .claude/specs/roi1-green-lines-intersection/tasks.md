# Tasks Document

## Task Overview

Implementation follows atomic task methodology with each task requiring 15-30 minutes and touching 1-3 related files. Tasks are structured for independent agent execution with clear requirements traceability and leverage of existing NHEM components.

## Steering Document Compliance

Tasks adhere to established NHEM technical patterns:
- **Architecture**: Follow NHEM layered architecture (API → Core → Data)
- **Configuration**: Leverage fem_config.json + NHEM_* environment variables + ConfigManager
- **API Design**: Extend existing `/api/roi/` namespace with password authentication
- **Data Models**: Use Pydantic models matching existing patterns in `app/models.py`
- **Python Style**: PEP 8 compliance with type hints throughout

## Atomic Task Requirements

Each task is:
- **15-30 minute completion time** for single agent
- **1-3 related files maximum** per task
- **Clear success criteria** with measurable outcomes
- **Independent execution** without active dependencies
- **Traceable to specific requirements** using reference patterns

## Task Format Guidelines

- `- [ ] Task number. Brief description (≤100 chars)`
- `_Requirements: X.Y_` links to specific requirements
- `_Leverage: path/to/file_` indicates existing components to reuse
- Clear input/output specifications for agent execution

## Atomic Tasks

### Backend Data Models

- [x] 1. Create LineDetectionConfig Pydantic model in backend/app/models.py
  _Requirements: 1.1, 2.5, 7.2_
  _Leverage: backend/app/models.py (existing AppConfig patterns)_

- [x] 2. Create LineIntersectionResult Pydantic model in backend/app/models.py
  _Requirements: 2.1-2.5, 4.4, 5.5_
  _Leverage: backend/app/models.py (existing response patterns)_

- [x] 3. Add EnhancedRealtimeDataResponse extending DualRealtimeDataResponse
  _Requirements: 7.1_
  _Leverage: backend/app/models.py (existing DualRealtimeDataResponse)_

- [x] 4. Extend AppConfig with LineDetectionConfig in backend/app/config.py
  _Requirements: 7.2_
  _Leverage: backend/app/config.py (existing nested config patterns)_

- [x] 5. Add line_detection section to fem_config.json with default parameters
  _Requirements: 1.1-1.5, 2.1-2.5_
  _Leverage: backend/app/fem_config.json (existing config structure)_

### Core Computer Vision Engine

- [x] 6. Create LineIntersectionDetector class structure in backend/app/core/line_intersection_detector.py
  _Requirements: 1.1-1.5_
  _Leverage: backend/app/core/ (existing core component patterns)_

- [x] 7. Implement HSV color segmentation with morphological operations
  _Requirements: 1.1, 1.2_
  _Leverage: backend/app/core/line_intersection_detector.py_

- [x] 8. Implement Canny edge detection and Hough line transformation
  _Requirements: 1.3, 1.4_
  _Leverage: backend/app/core/line_intersection_detector.py_

- [x] 9. Add virtual intersection calculation with parametric equations
  _Requirements: 2.1, 2.2_
  _Leverage: backend/app/core/line_intersection_detector.py_

- [x] 10. Implement confidence scoring formula and parallel line detection
  _Requirements: 2.3-2.5_
  _Leverage: backend/app/core/line_intersection_detector.py_

- [x] 11. Add error handling and performance optimization (300ms timeout)
  _Requirements: 2.2, 2.4, NF-Performance_
  _Leverage: backend/app/core/line_intersection_detector.py_

### API Endpoints

- [ ] 12. Add line detection control endpoints (enable/disable) to backend/app/api/routes.py
  _Requirements: 6.1, 6.2, 7.2, 7.3_
  _Leverage: backend/app/api/routes.py (existing authentication patterns)_

- [x] 13. Add manual detection endpoint to backend/app/api/routes.py
  _Requirements: 6.4_
  _Leverage: backend/app/api/routes.py (existing POST endpoint patterns)_

- [x] 14. Enhance /data/realtime/enhanced endpoint with include_line_intersection parameter
  _Requirements: 7.1, 3.3_
  _Leverage: backend/app/api/routes.py (existing query parameter patterns)_

- [x] 15. Add configuration management endpoints for line detection settings
  _Requirements: 7.2_
  _Leverage: backend/app/api/routes.py (existing config endpoint patterns)_

### ROI Service Integration

- [ ] 16. Extend ROICapture service for dual ROI processing in backend/app/core/roi_capture.py
  _Requirements: 3.1-3.5_
  _Leverage: backend/app/core/roi_capture.py (existing capture service)_

- [ ] 17. Add line detection processing to ROI capture with ROI1-only isolation
  _Requirements: 3.1, 3.4, 3.5_
  _Leverage: backend/app/core/roi_capture.py_

- [x] 18. Create line detection circular buffer in backend/app/core/data_store.py
  _Requirements: 3.4_
  _Leverage: backend/app/core/data_store.py (existing circular buffer patterns)_

- [x] 19. Add result caching with 100ms timeout to prevent redundant processing
  _Requirements: NF-Performance, NF-Reliability_
  _Leverage: backend/app/core/data_store.py_

### Python Client UI Components

- [x] 20. Create LineDetectionWidget matplotlib canvas structure in python_client/line_detection_widget.py
  _Requirements: 5.1, 5.2_
  _Leverage: python_client/realtime_plotter.py (existing matplotlib patterns)_

- [x] 21. Implement line overlay rendering and intersection point visualization
  _Requirements: 5.2, 5.3, 5.5_
  _Leverage: python_client/line_detection_widget.py_

- [ ] 22. Add Chinese control buttons with loading states (启用检测/禁用检测/处理中)
  _Requirements: 6.1, 6.2, 6.5_
  _Leverage: python_client/http_realtime_client.py (existing Tkinter patterns)_

- [x] 23. Implement Chinese status display with color coding (gray/yellow/green/red)
  _Requirements: 4.1-4.5_
  _Leverage: python_client/line_detection_widget.py_

- [ ] 24. Add API client integration for line detection endpoints
  _Requirements: 6.1, 6.3, 6.4, 7.1_
  _Leverage: python_client/http_realtime_client.py (existing HTTP client)_

### Python Client Integration

- [x] 25. Integrate LineDetectionWidget into main Python client window layout
  _Requirements: 6.1-6.5_
  _Leverage: python_client/http_realtime_client.py (existing main window)_

- [x] 26. Extend real-time data fetching with include_line_intersection parameter
  _Requirements: 7.1_
  _Leverage: python_client/http_realtime_client.py (existing data fetching)_

- [x] 27. Add line detection state management to application lifecycle
  _Requirements: 6.1, 6.2_
  _Leverage: python_client/http_realtime_client.py_

- [x] 28. Extend local configuration for line detection settings
  _Requirements: 7.2_
  _Leverage: python_client/http_client_config.json_

### Performance and Error Handling

- [ ] 29. Add performance monitoring for line detection processing times
  _Requirements: NF-Performance, NF-Reliability_
  _Leverage: backend/app/core/line_intersection_detector.py_

- [x] 30. Implement memory management for OpenCV objects and numpy arrays
  _Requirements: NF-Performance, NF-Reliability_
  _Leverage: backend/app/core/line_intersection_detector.py_

- [x] 31. Add comprehensive error handling for medical-grade reliability
  _Requirements: NF-Reliability, 4.5_
  _Leverage: backend/app/core/line_intersection_detector.py_

- [x] 32. Add client-side error handling and user feedback mechanisms
  _Requirements: 4.5, NF-Usability_
  _Leverage: python_client/line_detection_widget.py_

### Testing and Validation

- [ ] 33. Create unit tests for line detection algorithm accuracy (±5 pixels)
  _Requirements: NF-Performance, 1.5_
  _Leverage: tests/ (existing test structure)_

- [ ] 34. Add medical validation tests with anatomical landmark detection
  _Requirements: 1.1-1.5, NF-Reliability_
  _Leverage: tests/test_medical_validation.py_

- [ ] 35. Create performance benchmarks for <300ms processing requirement
  _Requirements: NF-Performance_
  _Leverage: tests/ (existing performance test patterns)_

- [ ] 36. Add clinical workflow integration testing scenarios
  _Requirements: 6.1-6.5, NF-Usability_
  _Leverage: tests/test_clinical_workflow.py_

## Task Dependencies

### Sequential Dependencies
- **Tasks 1-5**: Backend foundation (models → processing → APIs → service integration)
- **Tasks 6-11**: Computer vision engine (class structure → segmentation → detection → calculation → optimization)
- **Tasks 12-15**: API layer (control → data → configuration)
- **Tasks 16-19**: Service integration (ROI capture → data store → caching)
- **Tasks 20-24**: UI components (canvas → rendering → controls → status → API client)
- **Tasks 25-28**: Client integration (layout → data fetching → state → config)
- **Tasks 29-32**: Performance and error handling (monitoring → memory → server errors → client errors)
- **Tasks 33-36**: Testing and validation (unit → medical → performance → clinical)

### Parallel Execution Groups
- **Group A**: Tasks 1-5 (backend foundation) - can run in parallel after initial models
- **Group B**: Tasks 6-11 (computer vision) - sequential within engine
- **Group C**: Tasks 12-15 (APIs) - can run in parallel with Group D
- **Group D**: Tasks 16-19 (service integration) - can run in parallel with Group C
- **Group E**: Tasks 20-24 (UI components) - can run in parallel after APIs
- **Group F**: Tasks 25-28 (client integration) - sequential after UI components
- **Group G**: Tasks 29-32 (optimization) - can run in parallel with testing
- **Group H**: Tasks 33-36 (testing) - can run in parallel after core implementation

## Task Completion Criteria

Each task is complete when:
- **Functionality**: Specified features work correctly
- **Integration**: Integrates seamlessly with existing NHEM components
- **Testing**: Passes relevant unit and integration tests
- **Documentation**: Code is properly documented and follows NHEM patterns
- **Performance**: Meets specified performance requirements
- **Error Handling**: Handles error scenarios gracefully

## Risk Mitigation Strategies

### Technical Risks
- **OpenCV Performance**: Tasks 11, 29-30 address optimization and monitoring
- **Memory Management**: Tasks 30-31 implement proper cleanup and monitoring
- **API Conflicts**: Tasks 12-15 leverage existing NHEM patterns for compatibility
- **Threading Issues**: Tasks 16-19 ensure proper isolation from existing ROI2 processing

### Medical Context Risks
- **Accuracy Requirements**: Tasks 6-11 implement precise algorithms with validation
- **Clinical Workflow**: Tasks 16-19 maintain ROI1/ROI2 isolation to avoid disruption
- **Medical Environment**: Tasks 33-36 validate for medical workstation compatibility

### Timeline Risks
- **Dependencies**: Clear sequential and parallel execution paths identified
- **Atomic Scope**: Each task designed for 15-30 minute completion
- **Parallel Opportunities**: Multiple execution groups enable concurrent development