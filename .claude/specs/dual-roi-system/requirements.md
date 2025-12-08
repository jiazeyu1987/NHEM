# Requirements Document

## Introduction

This document outlines the requirements for implementing a dual-ROI (Region of Interest) system in the NHEM (New HEM Monitor) project. The dual-ROI system will introduce a large configurable ROI region alongside a fixed 50x50 small ROI region that is automatically extracted from the center of the large ROI. This enhancement will provide users with more flexible data analysis capabilities while maintaining backward compatibility with the existing single-ROI functionality.

The dual-ROI system addresses the need for broader context awareness (through the large ROI) while maintaining precise analysis capabilities (through the small ROI). The large ROI will be used for visual context and display, while the small ROI will continue to serve as the primary source for data analysis and peak detection, ensuring consistency with existing analytical algorithms.

## Alignment with Product Vision

This feature supports the NHEM project's goal of providing a comprehensive real-time signal processing system for HEM detection and monitoring. The dual-ROI system enhances the system's analytical capabilities by:

1. **Improved Context Awareness**: Users can visualize a larger region while maintaining precise analysis
2. **Enhanced Flexibility**: Configurable large ROI allows adaptation to different monitoring scenarios
3. **Backward Compatibility**: Existing workflows and configurations remain functional
4. **Performance Optimization**: Efficient resource utilization through shared processing pipelines

## Requirements

### Requirement 1: Dual-ROI Configuration Management

**User Story:** As a system administrator, I want to configure both large and small ROI regions, so that I can optimize the monitoring area for different scenarios.

#### Acceptance Criteria

1. WHEN the system starts THEN it SHALL support both dual-ROI mode and legacy single-ROI mode
2. WHEN in dual-ROI mode THEN the system SHALL store configuration for both large ROI (configurable) and small ROI (fixed 50x50)
3. WHEN setting large ROI configuration THEN the system SHALL validate coordinates are within screen bounds
4. WHEN large ROI is configured THEN the system SHALL automatically calculate small ROI coordinates as the center 50x50 region
5. IF no large ROI is configured THEN the system SHALL operate in legacy single-ROI mode
6. WHEN configuration is saved THEN it SHALL persist across system restarts

### Requirement 2: Large ROI Display and Visualization

**User Story:** As a monitoring operator, I want to see the large ROI region in real-time, so that I can maintain visual context of the monitoring area.

#### Acceptance Criteria

1. WHEN in dual-ROI mode THEN the frontend SHALL display the large ROI image (Read from fem_config.json)
2. WHEN large ROI is updated THEN the display SHALL refresh at the independently configured frame rate (default 5 FPS, separate from 45 FPS main processing)
3. IF large ROI capture fails THEN the system SHALL gracefully fallback to the last successful capture
4. WHEN displaying large ROI THEN the interface SHALL overlay visual indicators showing the small ROI extraction area
5. WHEN screen resolution changes THEN the system SHALL automatically adjust ROI coordinates to remain within bounds

### Requirement 3: Small ROI Data Processing and Analysis

**User Story:** As a data analyst, I want the system to use the small ROI for all data analysis and peak detection, so that analytical consistency is maintained.

#### Acceptance Criteria

1. WHEN processing data THEN the system SHALL extract the 50x50 small ROI from the center of the large ROI
2. WHEN performing peak detection THEN the system SHALL use data from the small ROI only
3. WHEN generating time-series data THEN the system SHALL calculate gray values from the small ROI
4. IF large ROI is smaller than 50x50 THEN the system SHALL use the entire large ROI for analysis
5. WHEN storing historical data THEN the system SHALL save small ROI data to existing buffers
6. WHEN small ROI extraction fails THEN the system SHALL log the error and continue with last successful data

### Requirement 4: API Extension and Backward Compatibility

**User Story:** As a developer, I want extended API endpoints that support dual-ROI while maintaining compatibility with existing clients, so that integration remains seamless.

#### Acceptance Criteria

1. WHEN existing clients call ROI endpoints THEN the system SHALL continue to work with single-ROI configuration
2. WHEN new clients request dual-ROI data THEN the system SHALL return both large and small ROI information
3. WHEN configuring ROI via API THEN the system SHALL accept both legacy (single) and new dual-ROI formats
4. IF API request specifies dual-ROI mode THEN the system SHALL validate large ROI coordinates before saving
5. WHEN querying ROI status THEN the system SHALL indicate current mode (single vs dual) and relevant configurations
6. WHEN deprecated endpoints are called THEN the system SHALL provide appropriate responses without breaking functionality

### Requirement 5: Frontend UI Enhancement

**User Story:** As a monitoring operator, I want an enhanced interface that shows both ROI regions and relevant controls, so that I can effectively manage the dual-ROI system.

#### Acceptance Criteria

1. WHEN in dual-ROI mode THEN the interface SHALL display both large ROI visualization and small ROI data charts
2. WHEN configuring ROI THEN users SHALL be able to set large ROI coordinates through input fields
3. WHEN switching modes THEN the interface SHALL adapt seamlessly between single and dual-ROI layouts
4. WHEN displaying ROI data THEN the system SHALL show clear visual indicators for mode and configuration status
5. IF large ROI is not configured THEN the interface SHALL display appropriate setup instructions
6. WHEN adjusting ROI settings THEN real-time preview SHALL be available before applying changes

## Non-Functional Requirements

### Performance
- Dual-ROI processing SHALL NOT increase overall system memory usage by more than 20%
- Large ROI capture frame rate SHALL be independently configurable from main processing FPS
- Small ROI extraction SHALL complete within 5ms per frame to maintain real-time performance
- API response times SHALL remain under 50ms for dual-ROI endpoints

### Security
- ROI coordinate validation SHALL prevent out-of-bounds screen access
- Configuration changes SHALL require appropriate authentication (existing password system)
- Input validation SHALL follow existing Pydantic model patterns
- Error messages SHALL not expose sensitive system information

### Reliability
- Small ROI extraction SHALL have 99.9% success rate when large ROI is properly configured
- System SHALL gracefully fallback to legacy mode if dual-ROI processing fails
- Configuration persistence SHALL survive system restarts and crashes
- Memory management SHALL prevent leaks in ROI image buffers

### Usability
- Mode switching SHALL be intuitive and clearly indicated in the UI
- Configuration interface SHALL provide real-time validation feedback
- Visual indicators SHALL clearly distinguish between large and small ROI displays
- System SHALL provide helpful error messages and setup guidance

### Compatibility
- Existing single-ROI configurations SHALL continue to work without modification
- API versioning SHALL maintain backward compatibility for legacy clients
- Python client SHALL support both ROI modes without breaking changes
- Configuration schema changes SHALL be additive and non-destructive
- Historical data SHALL remain accessible after mode switching
- WebSocket message formats SHALL maintain compatibility with existing clients

### Migration and Transition
- System SHALL provide seamless upgrade path from single to dual-ROI configurations
- Configuration validation SHALL prevent data corruption during mode transitions
- Existing ROI settings SHALL be automatically converted to dual-ROI format when appropriate
- Users SHALL be able to revert from dual-ROI to single-ROI mode without data loss