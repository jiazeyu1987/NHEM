# Bug Analysis

## Root Cause Analysis

### Investigation Summary
I conducted a thorough investigation of the NHEM system's ROI implementation and found that dual ROI support is partially implemented but incomplete. The frontend has infrastructure for displaying two ROIs, but the backend only supports single ROI processing and configuration.

### Root Cause
The bug is caused by **incomplete implementation of dual ROI functionality** across the entire stack:

1. **Configuration Schema Limitation**: The backend configuration (`fem_config.json`) only supports one ROI definition in `roi_capture.default_config`

2. **Backend Processing Gap**: The `RoiCaptureService` in `roi_capture.py` processes only one ROI at a time and returns a single `RoiData` object

3. **Data Model Limitation**: The `RealtimeDataResponse` model only has one `roi_data` field, missing `roi_data2` that the frontend expects

4. **API Response Gap**: Backend endpoints don't provide second ROI data, yet frontend code at lines 2290-2296 expects `data.roi_data2` and `data.roi2_configured`

5. **Frontend Integration**: Frontend has complete dual ROI infrastructure but cannot function without backend data

### Contributing Factors
- Frontend development progressed ahead of backend implementation
- Single ROI design didn't anticipate multi-ROI requirements
- Configuration file structure not designed for multiple ROI regions
- ROI capture service designed for single-region processing
- Data storage and API layers don't support multiple ROI streams

## Technical Details

### Affected Code Locations

- **File**: `backend/app/fem_config.json`
  - **Function/Method**: Configuration schema (lines 17-26)
  - **Issue**: `roi_capture` section only supports `default_config` with single ROI coordinates

- **File**: `backend/app/core/roi_capture.py`
  - **Function/Method**: `RoiCaptureService.capture_roi()` (lines 61-127)
  - **Issue**: Takes single `RoiConfig` parameter, returns single `RoiData`
  - **Impact**: Cannot process two ROI regions simultaneously

- **File**: `backend/app/models.py`
  - **Function/Method**: `RealtimeDataResponse` class (lines 47-56)
  - **Issue**: Only has `roi_data: RoiData` field, missing second ROI support
  - **Impact**: API response structure doesn't support dual ROI data

- **File**: `backend/app/api/routes.py`
  - **Function/Method**: `/data/realtime` endpoint (lines 180-286)
  - **Issue**: Creates and returns only single `roi_data` object
  - **Impact**: Frontend never receives second ROI data it expects

- **File**: `front/index.html`
  - **Function/Method**: Frontend display logic (lines 578-585, 2203-2209)
  - **Issue**: `roi2-container` has `display: none`, backend data never populates `appState.roiData2`
  - **Impact**: Second ROI display remains hidden permanently

### Data Flow Analysis

**Current Single ROI Flow:**
```
Screen → RoiCaptureService.capture_roi(roi_config) → RoiData → RealtimeDataResponse.roi_data → Frontend → Display
```

**Missing Dual ROI Flow (Expected by Frontend):**
```
Screen → RoiCaptureService.capture_dual_rois(roi_config1, roi_config2) → (RoiData1, RoiData2) →
RealtimeDataResponse(roi_data, roi_data2) → Frontend → Display both ROIs
```

**Current Frontend Logic Gap:**
```javascript
// Frontend expects this data structure but never receives it
if (data.roi_data2) {
    appState.roiData2 = data.roi_data2;
    appState.roi2Configured = data.roi2_configured || true;
} else {
    appState.roiData2 = null;
    appState.roi2Configured = false;
}
```

### Dependencies
- **PIL/Pillow**: Screen capture and image processing (supports multiple regions)
- **FastAPI**: API endpoints (need extension for dual ROI)
- **Pydantic Models**: Data validation (need extension for dual ROI)
- **Thread Safety**: ROI capture service runs in background thread

## Impact Analysis

### Direct Impact
- Users cannot monitor two separate regions simultaneously for comprehensive HEM analysis
- Existing frontend infrastructure (`roiRenderer2`, `roi-canvas2`, UI elements) goes unused
- System capability is limited despite having UI infrastructure ready

### Indirect Impact
- Reduced system flexibility for different monitoring scenarios
- Incomplete feature implementation may confuse users who see the UI elements
- Missing opportunity for enhanced comparative analysis between two regions
- Development effort already invested in frontend is wasted

### Risk Assessment
**Low risk to existing functionality** - current single ROI works fine.
**Medium implementation risk** - changes needed across multiple layers (config, service, API, models) must be coordinated.

## Solution Approach

### Fix Strategy

**Phase 1: Configuration Extension**
- Extend `fem_config.json` to support second ROI configuration
- Maintain backward compatibility with existing single ROI configs
- Add `roi2_config` section alongside existing `default_config`

**Phase 2: Backend Service Enhancement**
- Modify `RoiCaptureService` to capture two ROI regions from single screen capture
- Add `capture_dual_rois()` method that processes both regions simultaneously
- Maintain performance by reusing single screen capture for both ROIs

**Phase 3: Data Model Updates**
- Extend `RealtimeDataResponse` to include optional `roi_data2` field
- Add configuration flags for dual ROI mode
- Ensure backward compatibility for single ROI responses

**Phase 4: API Layer Updates**
- Modify `/data/realtime` endpoint to return dual ROI data when configured
- Add endpoints for second ROI configuration management
- Update ROI configuration endpoints to handle dual ROI setup

**Phase 5: Frontend Activation**
- Remove `display: none` from `roi2-container` when second ROI is configured
- Ensure frontend gracefully handles missing second ROI data
- Add configuration UI for second ROI setup

### Alternative Solutions

1. **Configuration Switching**: Allow users to switch between ROI presets (not simultaneous)
   - *Pros*: Simpler implementation
   - *Cons*: Doesn't meet requirement for simultaneous dual monitoring

2. **Sequential Capture**: Alternate between two ROI regions at same frequency
   - *Pros*: Uses existing infrastructure
   - *Cons*: Not truly simultaneous, reduces effective frame rate

3. **Multiple Service Instances**: Run separate ROI capture services for each region
   - *Pros*: True parallel processing
   - *Cons*: Double resource usage, increased complexity

**Recommended**: Primary fix strategy - simultaneous capture from single screen capture

### Risks and Trade-offs

**Performance Impact**:
- Processing two ROI regions increases CPU usage (estimated +30-50%)
- Memory usage increases with additional ROI data storage
- Network bandwidth increases with larger API responses

**Complexity Trade-offs**:
- Configuration becomes more complex for users
- More code paths to test and maintain
- Increased potential for configuration errors

**Compatibility Considerations**:
- Must maintain backward compatibility with existing single ROI setups
- Feature flag approach needed for gradual rollout
- Default behavior must remain unchanged

## Implementation Plan

### Changes Required

1. **Configuration Schema Extension**
   - **File**: `backend/app/fem_config.json`
   - **Modification**: Add `roi2_config` section in `roi_capture`
   - **Backward Compatibility**: Make second ROI optional, default disabled

2. **ROI Capture Service Enhancement**
   - **File**: `backend/app/core/roi_capture.py`
   - **Modification**: Add `capture_dual_rois()` method
   - **Optimization**: Single screen capture, multiple region processing
   - **Code Reuse**: Leverage existing `_capture_roi_internal()` for both regions

3. **Data Model Updates**
   - **File**: `backend/app/models.py`
   - **Modification**: Add optional `roi_data2: Optional[RoiData]` to `RealtimeDataResponse`
   - **Add**: `roi2_configured: bool = False` field for frontend state

4. **API Endpoint Updates**
   - **File**: `backend/app/api/routes.py`
   - **Modification**: Update `/data/realtime` to return dual ROI data when configured
   - **Add**: Second ROI configuration endpoints (`/roi/config2`)
   - **Logic**: Conditional dual ROI processing based on configuration

5. **Frontend Activation**
   - **File**: `front/index.html`
   - **Modification**: Remove hardcoded `display: none` from `roi2-container`
   - **Enhancement**: Add configuration UI for second ROI setup
   - **Graceful Handling**: Ensure UI works when second ROI not configured

### Testing Strategy

**Unit Testing**:
- Test `capture_dual_rois()` method with various ROI configurations
- Validate configuration loading and saving for dual ROI
- Test API response structures for both single and dual ROI modes

**Integration Testing**:
- End-to-end dual ROI data flow from capture to display
- Configuration persistence across server restarts
- Performance testing with both ROIs active

**Regression Testing**:
- Ensure existing single ROI functionality remains unchanged
- Verify backward compatibility with existing configurations
- Test all existing API endpoints still work correctly

**Edge Case Testing**:
- Invalid second ROI coordinates
- Overlapping ROI regions
- Performance with high ROI capture rates
- Memory usage over extended periods

### Rollback Plan

**Configuration Safety**:
- Second ROI configuration is optional and defaults to disabled
- Single ROI configuration remains unchanged and functional
- Feature flag approach allows immediate rollback

**Code Safety**:
- New methods added without modifying existing single ROI code paths
- Backward compatible API responses
- Frontend gracefully handles missing second ROI data

**Immediate Rollback Options**:
1. Disable dual ROI mode via configuration
2. Remove second ROI configuration from JSON file
3. Feature flag in code to disable dual ROI processing
4. Revert to previous single ROI only behavior

## Code Reuse Opportunities

### Existing Components That Can Help

1. **Image Processing Infrastructure**: `RoiCaptureService._capture_roi_internal()` can be reused for both ROI regions
2. **Configuration Management**: Existing `config_manager` infrastructure can handle second ROI settings
3. **Data Storage**: `RoiFrame` data structure can support multiple ROI indices
4. **Frontend Infrastructure**: All UI components (`roiRenderer2`, canvas elements) already exist
5. **API Patterns**: Existing ROI endpoints provide templates for second ROI endpoints

### Integration Points

- **DataStore Integration**: Extend existing `_roi_frames` storage to support multiple ROI indices
- **ConfigManager Integration**: Use existing configuration system for second ROI settings
- **WebSocket Integration**: Leverage existing real-time data broadcasting for dual ROI updates
- **Peak Detection Integration**: Apply existing detection algorithms to both ROI data streams

---

## Next Steps

This analysis provides a comprehensive understanding of the dual ROI implementation gap and a detailed roadmap for addressing it. The solution leverages significant existing infrastructure while maintaining full backward compatibility.

The fix requires coordinated changes across all system layers but can be implemented incrementally with built-in rollback capabilities.

**Key Success Factors**:
1. Maintain backward compatibility with existing single ROI setups
2. Implement dual ROI as an optional enhancement
3. Leverage existing infrastructure wherever possible
4. Comprehensive testing of both single and dual ROI modes