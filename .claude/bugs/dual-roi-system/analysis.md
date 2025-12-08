# Bug Analysis

## Root Cause Analysis

### Investigation Summary
I have thoroughly investigated the NHEM codebase and confirmed that the current system is architected for single ROI capture only. The data pipeline from ROI capture through processing to display assumes a single stream of ROI data. To implement dual ROI functionality, we need to extend this architecture while maintaining backward compatibility.

### Root Cause
System architecture designed around single ROI data flow:

**Configuration Level**:
- `fem_config.json` has single `roi_capture.default_config` section with x1, y1, x2, y2 coordinates
- No support for second ROI configuration

**Backend Processing Level**:
- `RoiCaptureService` in `backend/app/core/roi_capture.py` processes single ROI per capture call
- `DataStore` in `backend/app/core/data_store.py` maintains single `_roi_frames` buffer for ROI history
- API endpoints return single `roi_data` object in responses

**Data Model Level**:
- `RoiConfig` model represents single ROI region
- `RoiData` model contains single ROI image data
- `RealtimeDataResponse` includes single `roi_data` field

**Frontend Display Level**:
- Single `#roi-canvas` element (200x150) for ROI display
- Frontend processes single ROI data stream from API responses

### Contributing Factors
1. **Single ROI State Management**: All data structures assume one active ROI configuration
2. **Monolithic Capture Logic**: ROI capture service handles one region at a time
3. **API Response Format**: All ROI-related endpoints return single ROI data
4. **Frontend Canvas Layout**: Only one canvas element for ROI visualization
5. **Tight Coupling**: ROI configuration, capture, storage, and display are tightly integrated for single ROI flow

## Technical Details

### Affected Code Locations

#### Backend ROI Capture System
- **File**: `backend/app/core/roi_capture.py`
  - **Function/Method**: `RoiCaptureService.capture_roi()` (lines 61-127)
  - **Issue**: Processes single `RoiConfig` parameter, needs to handle dual ROI capture
  - **Current Logic**: Captures one ROI region, calculates gray value, stores to single buffer

- **File**: `backend/app/core/data_store.py`
  - **Function/Method**: `_roi_frames: Deque[RoiFrame]` (line 47)
  - **Issue**: Single buffer for ROI history, needs separate buffers for each ROI
  - **Current Logic**: Stores ROI frames in single deque with `_roi_frame_count` counter

#### API Layer
- **File**: `backend/app/api/routes.py`
  - **Function/Method**: `/data/realtime` endpoint (lines 201-283)
  - **Issue**: Returns single `roi_data` field in `RealtimeDataResponse`
  - **Current Logic**: Captures one ROI, returns single `RoiData` object

- **File**: `backend/app/api/routes.py`
  - **Function/Method**: ROI endpoints (`/roi/config`, `/roi/capture`) (lines 536-664)
  - **Issue**: All endpoints handle single ROI configuration and data
  - **Current Logic**: Single ROI config management in JSON and data store

#### Configuration Management
- **File**: `backend/app/fem_config.json`
  - **Section**: `roi_capture.default_config` (lines 17-26)
  - **Issue**: Only one ROI configuration section
  - **Current Structure**: `{"x1": 1480, "y1": 480, "x2": 1580, "y2": 580}`

#### Data Models
- **File**: `backend/app/models.py`
  - **Function/Method**: `RoiConfig` class (lines 128-158)
  - **Issue**: Single ROI configuration model
  - **Current Logic**: One set of coordinates for single ROI region

- **File**: `backend/app/models.py`
  - **Function/Method**: `RealtimeDataResponse` class (lines 47-56)
  - **Issue**: Contains single `roi_data: RoiData` field
  - **Current Logic**: API responses include only one ROI data object

#### Frontend Display
- **File**: `front/index.html`
  - **Function/Method**: `#roi-canvas` element (line 245)
  - **Issue**: Single canvas element for ROI display
  - **Current Logic**: One 200x150 canvas shows single ROI image

- **File**: `front/index.html`
  - **Function/Method**: ROI data processing JavaScript
  - **Issue**: Frontend expects single ROI data in API responses
  - **Current Logic**: Processes single `roi_data` object and renders to one canvas

### Data Flow Analysis

**Current Single ROI Flow:**
```
JSON Config (roi_capture.default_config)
    ↓
RoiConfig object
    ↓
RoiCaptureService.capture_roi(roi_config)
    ↓
Single RoiData (gray_value + base64_image)
    ↓
DataStore._roi_frames (single buffer)
    ↓
API Response (single roi_data field)
    ↓
Frontend Canvas (#roi-canvas)
    ↓
Single ROI Display
```

**Required Dual ROI Flow:**
```
JSON Config (roi_capture.config + roi_capture.config2)
    ↓
RoiConfig object 1 + RoiConfig object 2
    ↓
RoiCaptureService.capture_dual_roi()
    ↓
RoiData1 + RoiData2 (different capture logic)
    ↓
DataStore._roi_frames1 + DataStore._roi_frames2
    ↓
API Response (roi_data + roi_data2 fields)
    ↓
Frontend Canvas1 + Canvas2 (side-by-side)
    ↓
Dual ROI Display
```

**Special Requirements from Bug Report:**
- **ROI1 (New)**: Reads from config table, displays on right
- **ROI2 (Original)**: Fixed 50x50 from ROI1 center, displays left, all analysis uses ROI2

### Dependencies
- FastAPI for API endpoints
- HTML5 Canvas for rendering
- Pydantic for data validation
- PIL/Pillow for image processing
- Thread-safe data structures (deque)
- JSON configuration management

## Impact Analysis

### Direct Impact
- **ROI capture service**: Needs method to handle dual ROI capture with different logic
- **Data storage**: Requires separate buffers for each ROI stream
- **API layer**: Must return dual ROI data while maintaining backward compatibility
- **Frontend**: Needs second canvas element and side-by-side rendering
- **Configuration**: Schema extension to support second ROI configuration

### Indirect Impact
- **Python client**: Requires updates to display dual ROI data
- **Performance**: Double capture processing load (2x image processing)
- **Memory usage**: Additional storage for second ROI history buffer
- **Documentation**: Need to update all ROI-related documentation

### Risk Assessment
- **High Risk**: Changes to core ROI capture logic could affect existing functionality
- **Medium Risk**: API response format changes must maintain backward compatibility
- **Low Risk**: Frontend display additions (non-breaking)
- **Low Risk**: Configuration schema extensions (backward compatible)

## Solution Approach

### Fix Strategy
Based on the bug requirements, implement specialized dual ROI system:

1. **Configuration Schema Extension**:
   - Add `roi2_config` section for new ROI (right side)
   - Keep existing `default_config` for backward compatibility
   - New ROI reads from config, original ROI auto-calculated

2. **Dual ROI Capture Logic**:
   - **ROI1 (New)**: Full capture from configuration coordinates, displays right
   - **ROI2 (Original)**: Fixed 50x50 from ROI1 center, displays left
   - Both capture at same frequency using shared frame rate setting

3. **Data Storage Extension**:
   - Separate buffers: `_roi_frames1` and `_roi_frames2`
   - Independent frame counters for each ROI stream
   - Maintain existing analysis logic with ROI2 data

4. **API Response Enhancement**:
   - Add `roi_data2` field to `RealtimeDataResponse`
   - Maintain `roi_data` field for backward compatibility
   - Existing endpoints return both ROI datasets

5. **Frontend Dual Display**:
   - Add second canvas element (`#roi-canvas2`)
   - Side-by-side layout: ROI2 (left) + ROI1 (right)
   - Independent gray value displays for each ROI

### Alternative Solutions Considered

**Option A**: Separate ROI services (rejected - over-engineering)
- Pros: Complete isolation
- Cons: Double code complexity, synchronization challenges

**Option B**: Extend existing service (chosen)
- Pros: Reuse existing logic, simpler implementation
- Cons: More complex single service

**Option C**: Configurable number of ROIs (rejected - over-complex)
- Pros: Flexible for future expansion
- Cons: YAGNI principle, unnecessary complexity

### Implementation Requirements Based on Bug Report

**ROI1 (New ROI - Right Side)**:
- Reads coordinates from configuration table (`roi2_config`)
- Full-size capture with resizing to 200x150
- Displays on the right side of frontend
- No analysis performed on this ROI

**ROI2 (Original ROI - Left Side)**:
- Fixed 50x50 capture from ROI1 center coordinates
- `center_x - 25, center_y - 25, center_x + 25, center_y + 25`
- Displays on the left side of frontend
- All existing peak detection and analysis uses this ROI data
- Does not read from configuration table

**Both ROIs**:
- Same capture frequency (shared frame rate setting)
- Independent gray value calculation
- Independent history tracking
- Side-by-side frontend display

## Implementation Plan

### Changes Required

1. **Configuration Schema Extension**
   - File: `backend/app/fem_config.json`
   - Add `roi2_config` section with x1, y1, x2, y2 coordinates
   - Maintain `default_config` for backward compatibility

2. **Data Models Enhancement**
   - File: `backend/app/models.py`
   - Extend `RealtimeDataResponse` with `roi_data2: Optional[RoiData]`
   - Add new response models for dual ROI configuration

3. **ROI Capture Service Update**
   - File: `backend/app/core/roi_capture.py`
   - Add `capture_dual_roi()` method
   - ROI1: Capture from `roi2_config`, ROI2: Capture 50x50 from ROI1 center
   - Return tuple of (RoiData, RoiData)

4. **Data Store Enhancement**
   - File: `backend/app/core/data_store.py`
   - Add `_roi_frames2` buffer and `_roi_frame_count2` counter
   - Separate `add_roi_frame1()` and `add_roi_frame2()` methods
   - Maintain existing API methods for ROI2 (backward compatibility)

5. **API Layer Extensions**
   - File: `backend/app/api/routes.py`
   - Update `/data/realtime` to capture and return both ROIs
   - Extend ROI configuration endpoints to handle dual ROI
   - Maintain backward compatibility with single ROI responses

6. **Frontend Dual Display Implementation**
   - File: `front/index.html`
   - Add `#roi-canvas2` element and styling
   - Side-by-side layout implementation
   - Update JavaScript to handle dual ROI data
   - Independent gray value displays

### Testing Strategy

**Unit Testing**:
- Test dual ROI capture logic with different configurations
- Verify 50x50 center extraction accuracy
- Test configuration loading and validation

**Integration Testing**:
- Verify API responses include both ROI datasets
- Test backward compatibility with single ROI clients
- End-to-end dual ROI data flow testing

**Frontend Testing**:
- Visual verification of side-by-side display
- Test canvas rendering for both ROI streams
- Verify responsive layout behavior

**Performance Testing**:
- Measure impact of dual capture on frame rate
- Memory usage comparison with single ROI
- Validate 45 FPS processing target is maintained

### Rollback Plan
- **Configuration**: Single ROI config remains valid
- **Feature Flag**: Add `enable_dual_roi` setting to disable feature
- **API Compatibility**: Single ROI responses still supported
- **Frontend Fallback**: Graceful handling of missing ROI2 data