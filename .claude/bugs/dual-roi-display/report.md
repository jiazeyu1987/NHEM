# Bug Report

## Bug Summary
The system currently has dual ROI infrastructure in place but the second ROI display is not functional. The user needs a second ROI display that:
- Uses the same capture frequency as the primary ROI
- Reads configuration from the configuration file
- Displays to the right of the existing ROI
- Only provides display functionality (no additional features)

## Bug Details

### Expected Behavior
1. Two ROI displays should be visible side by side in the frontend
2. Both ROI displays should capture at the same frequency
3. Both ROI configurations should be read from the backend configuration file
4. The second ROI should only display captured images and gray values (no additional controls)
5. Both ROI regions should update in real-time during detection

### Actual Behavior
1. Only one ROI display is currently visible and functional
2. The frontend has placeholder code for a second ROI (`roi-canvas2`, `roiRenderer2`) but it's hidden (`display: none`)
3. Backend configuration only supports one ROI configuration
4. Backend ROI capture service only handles single ROI processing

### Steps to Reproduce
1. Start the NHEM system backend (`python run.py`)
2. Open frontend in browser (`http://localhost:3000`)
3. Configure ROI settings and start detection
4. Observe that only one ROI display is visible in the "ROI 监控" panel
5. Check browser developer tools to see hidden second ROI elements

### Environment
- **Version**: NHEM (New HEM Monitor) latest
- **Platform**: Windows (based on file paths)
- **Configuration**: Default fem_config.json with single ROI settings

## Impact Assessment

### Severity
- [ ] Critical - System unusable
- [x] High - Major functionality broken
- [ ] Medium - Feature impaired but workaround exists
- [ ] Low - Minor issue or cosmetic

### Affected Users
Users who need to monitor two separate regions of interest simultaneously for comprehensive HEM detection analysis.

### Affected Features
- ROI monitoring display functionality
- Real-time dual-region observation capabilities
- Configuration file utilization for multiple ROI settings

## Additional Context

### Error Messages
```
No explicit error messages - functionality is simply missing/incomplete
```

### Screenshots/Media
Based on code analysis, the frontend contains:
- `roi-canvas` (visible) - Primary ROI display
- `roi-canvas2` (hidden with `display: none`) - Second ROI display placeholder
- `roiRenderer2` instance exists but not used
- `appState.roiData2` and `appState.roi2Configured` variables exist but not populated

### Related Issues
- Infrastructure exists in frontend but not connected to backend
- Backend ROI configuration only supports single ROI settings
- Backend ROI capture service needs extension for dual ROI processing

## Initial Analysis

### Suspected Root Cause
The system appears to have partially implemented dual ROI support in the frontend but lacks:
1. Backend configuration changes to support second ROI settings
2. Backend ROI capture service modifications to handle two ROI regions
3. Frontend activation of the hidden second ROI display elements
4. Data flow from backend to populate second ROI data

### Affected Components
- **Backend**: `backend/app/fem_config.json` - needs second ROI configuration
- **Backend**: `backend/app/core/roi_capture.py` - needs dual ROI processing
- **Backend**: `backend/app/models.py` - may need second ROI data models
- **Backend**: `backend/app/api/routes.py` - needs second ROI endpoints
- **Frontend**: `front/index.html` - needs activation of existing second ROI display
- **Configuration**: ROI coordinate storage and retrieval for two regions

---

### Bug Analysis Template
# Bug Analysis

## Root Cause Analysis

### Investigation Summary
The system has a foundation for dual ROI support but it's incomplete. The frontend contains placeholder elements and code for a second ROI display, but the backend lacks support for configuring and processing multiple ROI regions simultaneously.

### Root Cause
1. **Incomplete Implementation**: Dual ROI support was started but not finished
2. **Configuration Limitation**: Backend configuration schema only supports one ROI definition
3. **Missing Backend Logic**: ROI capture service only processes single ROI region
4. **Frontend Integration Gap**: Frontend has infrastructure but no backend data to display

### Contributing Factors
- Frontend development progressed ahead of backend implementation
- Configuration file structure not designed for multiple ROI regions
- ROI capture service designed for single-region processing

## Technical Details

### Affected Code Locations

- **File**: `backend/app/fem_config.json`
  - **Function/Method**: Configuration schema
  - **Lines**: 17-26 (roi_capture section)
  - **Issue**: Only supports single ROI configuration

- **File**: `backend/app/core/roi_capture.py`
  - **Function/Method**: `RoiCaptureService.capture_roi()`
  - **Lines**: 61-127
  - **Issue**: Processes only one ROI at a time

- **File**: `front/index.html`
  - **Function/Method**: Frontend display logic
  - **Lines**: 577-587 (roi2-container)
  - **Issue**: Second ROI container hidden with `display: none`

### Data Flow Analysis
Current single ROI flow:
Screen → RoiCaptureService → DataStore → API → Frontend → Display

Missing dual ROI flow:
Screen → RoiCaptureService → (process 2 ROIs) → DataStore → API → Frontend → (display 2 ROIs)

### Dependencies
- PIL/Pillow for screen capture and image processing
- FastAPI for API endpoints
- Frontend JavaScript for real-time display

## Impact Analysis

### Direct Impact
- Users cannot monitor two regions simultaneously
- Limited analysis capability for complex HEM detection scenarios
- Existing frontend infrastructure goes unused

### Indirect Impact
- Reduced system flexibility for different monitoring needs
- Incomplete feature implementation may confuse users
- Missing opportunity for enhanced comparative analysis

### Risk Assessment
Low risk to existing functionality - current single ROI works fine. Risk is in implementing changes without breaking existing behavior.

## Solution Approach

### Fix Strategy
1. **Backend Configuration Extension**: Add second ROI configuration to fem_config.json
2. **Backend Processing Enhancement**: Modify ROI capture service to handle two regions
3. **Backend API Updates**: Add endpoints for second ROI data and configuration
4. **Frontend Activation**: Enable existing second ROI display elements

### Alternative Solutions
1. **Configuration Only**: Allow users to switch between ROI configurations (not simultaneous)
2. **Sequential Capture**: Alternate between two ROI regions at same frequency
3. **Multiple Service Instances**: Run separate ROI capture services (more complex)

### Risks and Trade-offs
- **Performance**: Processing two ROIs may increase CPU usage
- **Complexity**: Adds configuration complexity for users
- **Compatibility**: Must maintain backward compatibility with single ROI setups

## Implementation Plan

### Changes Required

1. **Change 1**: Extend backend configuration schema
   - File: `backend/app/fem_config.json`
   - Modification: Add second ROI configuration section with coordinates and settings

2. **Change 2**: Enhance ROI capture service
   - File: `backend/app/core/roi_capture.py`
   - Modification: Modify capture service to process two ROI regions simultaneously

3. **Change 3**: Update backend API routes
   - File: `backend/app/api/routes.py`
   - Modification: Add endpoints for second ROI configuration and data

4. **Change 4**: Activate frontend second ROI display
   - File: `front/index.html`
   - Modification: Remove `display: none` from roi2-container and add data population logic

5. **Change 5**: Update data models if needed
   - File: `backend/app/models.py`
   - Modification: Add support for second ROI data structures

### Testing Strategy
- Test both ROI displays update correctly at same frequency
- Verify configuration changes persist correctly
- Ensure backward compatibility with existing single ROI setups
- Validate performance impact is acceptable

### Rollback Plan
- Keep second ROI configuration optional
- Add feature flag to enable/disable dual ROI mode
- Maintain existing single ROI code path as default

---