# Requirements Document

## Introduction

The ROI1 Green Lines Intersection Detection feature extends the NHEM (New HEM Monitor) system to automatically detect two non-intersecting green lines within the ROI1 region and calculate their virtual intersection point (extended line intersection). This feature enhances the system's analytical capabilities by providing precise geometric measurements for medical monitoring and quality control applications.

The feature processes only the ROI1 region for line detection, while ROI2 remains dedicated to grayscale analysis, maintaining the existing dual ROI functionality. All user interface elements are implemented in the Python client using matplotlib and Tkinter, providing real-time visualization and control.

## Technical Definitions

### Coordinate System
- **Pixel Coordinates**: All coordinates use pixel-based system with origin (0,0) at top-left of ROI1
- **ROI Bounds**: Intersection coordinates are relative to ROI1 top-left corner
- **Display Scaling**: Matplotlib coordinates map 1:1 with pixel coordinates for accurate visualization

### Confidence Scoring Algorithm
- **Line Length Weight**: Longer lines receive higher confidence scores (length/100)
- **Detection Quality**: Stronger edges and cleaner green pixels increase confidence
- **Intersection Stability**: Consistent intersection points across frames improve confidence
- **Formula**: `confidence = (line1_length + line2_length) / 200 * edge_quality * 0.9 + temporal_stability * 0.1`
- **Range**: 0.0 to 1.0, where >0.7 indicates high confidence, 0.4-0.7 medium confidence, <0.4 low confidence

### Non-Parallel Line Criteria
- **Angle Threshold**: Lines with angle difference < 5 degrees are considered parallel
- **Mathematical Test**: Denominator of intersection calculation must be > 0.01 to avoid numerical instability
- **Direction Filtering**: Horizontal and vertical lines (angles < 10° or > 80° from horizontal) are excluded as they provide poor intersection geometry

### OpenCV Processing Pipeline
- **Version Compatibility**: Requires OpenCV 4.5+ for optimal HSV handling and morphological operations
- **Memory Management**: All intermediate images use proper memory release and avoid unnecessary copying
- **Processing Order**: HSV conversion → Green mask → Morphological cleanup → Canny edges → Hough lines → Intersection calculation

## Alignment with Product Vision

This feature supports NHEM's goal of providing comprehensive real-time signal processing and monitoring by:
- Adding advanced computer vision capabilities for geometric analysis
- Enhancing measurement precision through automated line detection
- Maintaining the existing dual ROI architecture for comprehensive monitoring
- Providing professional-grade visualization tools for medical and industrial applications
- Ensuring seamless integration with the existing NHEM ecosystem

## Requirements

### Requirement 1: ROI1 Green Line Detection

**User Story:** As a medical imaging technician analyzing ultrasound HEM events, I want the system to automatically detect anatomical landmark lines represented in green within ROI1 and calculate their virtual intersection point, so that I can obtain precise geometric measurements for medical diagnosis and treatment planning.

#### Acceptance Criteria

1. WHEN the system processes ROI1 data THEN the system SHALL extract green colored pixels using HSV color space segmentation with thresholds H(40-80), S(50-255), V(50-255)
2. WHEN green pixels are identified THEN the system SHALL apply morphological operations to remove noise and isolate line structures
3. WHEN line structures are isolated THEN the system SHALL detect line segments using Canny edge detection with thresholds 25 (low) and 80 (high)
4. WHEN edges are detected THEN the system SHALL apply Hough line transformation with parameters optimized for line detection (distance resolution 1px, angle resolution 1°, minimum length 15px, maximum gap 8px)
5. IF at least two non-parallel lines are detected THEN the system SHALL calculate their virtual intersection point using parametric line equations

### Requirement 2: Virtual Intersection Point Calculation

**User Story:** As a medical researcher studying anatomical alignment patterns, I want the system to calculate virtual intersection points from anatomical landmark lines that may not physically intersect within the visible ROI area, so that I can measure geometric relationships and track structural changes for medical research and clinical assessment.

#### Acceptance Criteria

1. WHEN two line segments are detected THEN the system SHALL calculate their intersection using the formula: intersection = (x1 + t*(x2-x1), y1 + t*(y2-y1)) where t is the parameter value
2. WHEN lines are parallel (denominator approaches zero) THEN the system SHALL return null and log a parallel lines detected message
3. WHEN intersection is calculated THEN the system SHALL validate that the intersection point falls within reasonable coordinate bounds
4. WHEN multiple intersection points are calculated THEN the system SHALL select the best intersection based on line confidence scores
5. WHEN intersection is calculated THEN the system SHALL compute confidence score based on line lengths and detection quality

### Requirement 3: ROI1-Only Processing

**User Story:** As a clinical system administrator managing multi-modal medical imaging workflows, I want line detection to run only on ROI1 to maintain ROI2's dedicated grayscale signal analysis function, so that anatomical geometry analysis and hemodynamic signal monitoring can work optimally without computational interference or resource conflicts.

#### Acceptance Criteria

1. WHEN line detection is enabled THEN the system SHALL process only ROI1 image data and ignore ROI2 for line detection
2. WHEN ROI2 data is processed THEN the system SHALL perform only grayscale value calculation and peak detection
3. WHEN dual ROI data is returned THEN the system SHALL include line intersection results only for ROI1 data
4. WHEN processing ROI1 THEN the system SHALL maintain the existing 4 FPS capture rate and caching mechanisms
5. WHEN ROI2 processing is requested THEN the system SHALL not execute any line detection algorithms

### Requirement 4: Real-time Status Display

**User Story:** As a clinical monitoring technician performing real-time medical imaging assessment, I want real-time status information about anatomical landmark line detection displayed in the Python client interface, so that I can quickly understand the current detection state and make informed clinical decisions based on geometric analysis results.

#### Acceptance Criteria

1. WHEN line detection is not enabled THEN the status display SHALL show "线条相交点: 未启用" in gray color
2. WHEN line detection is enabled but no intersection is detected THEN the status display SHALL show "线条相交点: 已启用 - 未识别" in yellow/orange color
3. WHEN intersection is successfully detected THEN the status display SHALL show "线条相交点: 已识别 (x, y) 置信度: XX%" in green color with exact integer coordinates
4. WHEN detection encounters an error THEN the status display SHALL show "线条相交点: 检测错误: [error_message]" in red color
5. WHEN status changes occur THEN the display SHALL update immediately in the Tkinter interface

### Requirement 5: Matplotlib Visualization

**User Story:** As a medical researcher conducting anatomical geometry studies, I want to see the detected anatomical landmark lines and their calculated intersection point visualized on the ROI1 medical image, so that I can visually verify the detection accuracy and validate geometric measurements for clinical research applications.

#### Acceptance Criteria

1. WHEN ROI1 image data is available THEN the system SHALL display it using matplotlib with appropriate scaling
2. WHEN green lines are detected THEN the system SHALL overlay the detected lines on the matplotlib canvas
3. WHEN intersection point is calculated THEN the system SHALL mark it with a red circle (outer, 6px radius) and orange circle (inner, 3px radius)
4. WHEN intersection point is marked THEN the system SHALL display crosshair lines extending 8 pixels from the center point
5. WHEN intersection confidence is calculated THEN the system SHALL display coordinates and confidence text in a semi-transparent black box: "(x, y)\nc:confidence"

### Requirement 6: User Control Interface

**User Story:** As a clinical system operator managing real-time medical imaging workflows, I want to control the anatomical landmark detection feature through intuitive buttons in the Python client interface, so that I can enable, disable, and refresh detection as needed during clinical examinations and medical procedures.

#### Acceptance Criteria

1. WHEN the line detection toggle button is clicked THEN the system SHALL send appropriate API calls to enable or disable detection
2. WHEN detection is disabled THEN the toggle button SHALL display "启用检测" text
3. WHEN detection is enabled THEN the toggle button SHALL display "禁用检测" text and show active styling
4. WHEN the manual refresh button is clicked THEN the system SHALL trigger an immediate line detection on current ROI1 data
5. WHEN operations are in progress THEN buttons SHALL show loading state ("处理中..." or "检测中...") and be disabled to prevent conflicts

### Requirement 7: API Integration

**User Story:** As a clinical system architect integrating medical imaging modules, I want the anatomical landmark detection feature to integrate seamlessly with existing NHEM medical monitoring APIs, so that it works within the established clinical system architecture and maintains compatibility with other medical analysis components.

#### Acceptance Criteria

1. WHEN fetching real-time data THEN the system SHALL call `/data/realtime/enhanced` with `include_line_intersection=true` parameter when detection is enabled
2. WHEN enabling detection THEN the system SHALL send POST request to `/api/roi/line-intersection/enable` with authentication
3. WHEN disabling detection THEN the system SHALL send POST request to `/api/roi/line-intersection/disable` with authentication
4. WHEN making manual detection requests THEN the system SHALL call `/api/roi/line-intersection` with ROI coordinates and authentication
5. WHEN API responses are received THEN the system SHALL handle success/error states appropriately and update UI status

## Non-Functional Requirements

### Performance
- **Processing Time**: Single frame processing time SHALL be less than 300ms to maintain real-time performance
- **Detection Success Rate**: In clear green line conditions, detection success rate SHALL be greater than 90%
- **Coordinate Precision**: Intersection point calculation accuracy SHALL be within ±5 pixels of true value using pixel coordinate system with origin at top-left
- **Memory Usage**: Additional memory usage for line detection SHALL not exceed 50MB beyond existing allocations
- **Update Frequency**: Status display updates SHALL occur within 100ms of data changes
- **Algorithm Efficiency**: OpenCV operations SHALL be optimized with proper memory management and vectorized processing
- **Resource Management**: OpenCV version compatibility SHALL be maintained (4.5+ recommended)

### Reliability
- **Error Recovery**: System SHALL recover gracefully from detection failures without crashing the main application
- **Connection Resilience**: Network interruptions SHALL not affect local processing capabilities
- **Data Consistency**: ROI1 and ROI2 processing SHALL remain synchronized despite different update rates
- **Cache Efficiency**: Results SHALL be cached for 100ms to prevent unnecessary recalculations
- **Long-term Stability**: System SHALL operate continuously for extended periods (24+ hours) without degradation
- **Error Handling**: Specific error scenarios SHALL be handled with appropriate fallback procedures:
  - Insufficient green pixels: Return "insufficient green pixels detected" status
  - No lines detected: Return "no lines detected" status with no intersection calculation
  - Parallel lines only: Return "parallel lines detected, no intersection possible" status
  - OpenCV processing errors: Log specific error and return "processing error" status
  - Invalid ROI coordinates: Return "invalid ROI configuration" status

### Usability
- **Status Clarity**: Detection status SHALL be clearly indicated through color coding and descriptive text
- **Control Intuition**: Enable/disable controls SHALL be intuitive and responsive to user actions
- **Visual Feedback**: Matplotlib visualization SHALL provide immediate visual confirmation of detection results
- **Error Messaging**: Error information SHALL be user-friendly and actionable with specific error descriptions
- **Performance Visibility**: Processing time and confidence scores SHALL be displayed to users
- **Medical Context**: All user interface elements SHALL be designed for medical monitoring professionals with clear terminology

### Security
- **API Authentication**: All line detection API calls SHALL include proper password authentication using the NHEM security system
- **Input Validation**: All ROI coordinates and image data SHALL be validated before processing
- **Data Privacy**: No sensitive image data SHALL be stored permanently beyond operational requirements
- **Access Control**: Line detection features SHALL respect existing user permission and access control systems

---

**Version**: 1.0
**Created**: 2025-12-07
**Specification**: ROI1 Green Lines Intersection Detection
**Target Implementation**: Python client with OpenCV integration