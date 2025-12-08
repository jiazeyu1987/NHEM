# Bug Report

## Bug Summary
ROI panels are not properly separated into different vertical panels - both ROI1 and ROI2 are appearing in the same "实时信息" (Real-time Information) panel instead of being in separate stacked panels.

## Bug Details

### Expected Behavior
- **Vertical Stack Layout**: ROI1 should be in the top panel and ROI2 should be in the bottom panel
- **Panel Separation**: Two distinct panels with clear visual separation
- **Independent Control**: Each ROI panel should be independently managed and displayed
- **Vertical Organization**: One panel positioned above the other in the interface

### Actual Behavior
- **Same Panel Placement**: Both ROI1 and ROI2 are displayed in the same "实时信息" panel
- **Vertical Stacking**: ROI2 appears directly below ROI1 within the same panel container
- **Panel Confusion**: No clear visual separation between different ROI panels
- **Layout Issues**: The interface doesn't show the intended vertical panel separation

### Steps to Reproduce
1. Start the Python client: `python http_realtime_client.py`
2. Wait for the application to initialize and connect
3. Observe the left-side "实时信息" panel
4. Look for ROI display areas
5. Note that both ROI1 and ROI2 are in the same panel with ROI2 below ROI1
6. Expected: Two separate panels vertically stacked, Actual: Single panel with both ROIs

### Environment
- **Version**: NHEM Python Client (http_realtime_client.py)
- **Platform**: Windows (based on file paths)
- **Configuration**: Default configuration with dual ROI support
- **GUI Framework**: tkinter with ttk components

## Impact Assessment

### Severity
- [x] Medium - Feature impaired but workaround exists
- [ ] Critical - System unusable
- [ ] High - Major functionality broken
- [ ] Low - Minor issue or cosmetic

### Affected Users
- Users who need clear visual separation between different ROI monitoring areas
- Users who want to distinguish between analysis ROI and configuration ROI
- Users who need to monitor multiple ROI regions independently

### Affected Features
- Python Client ROI display functionality
- Dual ROI monitoring capability
- Panel layout and organization
- Visual separation of different ROI types

## Additional Context

### Error Messages
```
No explicit error messages - this is a layout/design issue
```

### Screenshots/Media
Based on current implementation:
- Current: Single "实时信息" panel containing both ROI1 and ROI2 vertically stacked
- Expected: Two separate panels - "ROI1 监控" (top) and "ROI2 监控" (bottom)
- Current labels: "ROI1 Screenshot (分析)" and "ROI2 Screenshot (配置)" within same panel

### Related Issues
- Recent dual ROI implementation completed
- Frontend vs Python client display differences
- Panel layout organization requirements

## Initial Analysis

### Suspected Root Cause
The implementation placed both ROI display areas within the same `info_frame` (实时信息 panel) instead of creating separate panels for each ROI. The code shows:

```python
# Both ROIs placed in same info_frame
roi1_frame = ttk.LabelFrame(self.info_frame, text="ROI1 Screenshot (分析)")
roi2_frame = ttk.LabelFrame(self.info_frame, text="ROI2 Screenshot (配置)")
```

### Affected Components
- **File**: `python_client/http_realtime_client.py`
  - **Function/Method**: `_build_widgets()`
  - **Lines**: 382-426 (ROI frame creation)
  - **Issue**: Both ROI frames created within same parent panel

- **File**: `python_client/http_realtime_client.py`
  - **Function/Method**: Main layout structure
  - **Lines**: 270-278 (info_frame creation)
  - **Issue**: Single panel structure doesn't support vertical panel separation

## Initial Investigation Plan
1. Analyze current panel structure and layout hierarchy
2. Identify where ROI1 and ROI2 frames are currently placed
3. Determine correct approach for creating separate vertical panels
4. Implement panel separation while maintaining existing functionality
5. Test both normal and compact modes for proper display

## Expected Fix Approach
Create separate LabelFrame panels for each ROI with proper vertical stacking:
1. Top panel: "ROI1 监控" containing ROI1 screenshot and controls
2. Bottom panel: "ROI2 监控" containing ROI2 screenshot and controls
3. Maintain all existing functionality and data processing
4. Preserve existing UI components and layout structure

---

### Bug Analysis Template
# Bug Analysis

## Root Cause Analysis

### Investigation Summary
The root cause is architectural - both ROI display areas are created within the same parent container (`info_frame`) instead of being organized into separate panels. The current implementation creates:

```
info_frame (实时信息)
├── status_info
├── config_frame
├── roi1_frame (ROI1 Screenshot)
├── roi2_frame (ROI2 Screenshot)
└── other components
```

But the requirement is for vertical panel separation:
```
Top Panel: roi1_panel (ROI1 监控)
├── ROI1 screenshot display
├── ROI1 information and controls
└── ...

Bottom Panel: roi2_panel (ROI2 监控)
├── ROI2 screenshot display
├── ROI2 information and controls
└── ...

Other Panel: info_frame (实时信息)
├── status_info
├── config_frame
└── other real-time information
```

### Root Cause
**Panel Architecture Issue**: The implementation conflates ROI display with general real-time information, failing to create dedicated panels for each ROI type.

**Layout Design Flaw**: The current design doesn't support the required vertical stack layout where each ROI has its own distinct panel.

### Contributing Factors
1. **Inheritance from Single ROI Design**: The code evolved from single ROI display and simply added ROI2 to the existing panel
2. **Panel Hierarchy**: Lack of clear separation between different functional areas (monitoring vs configuration vs real-time data)
3. **Visual Organization**: No clear visual demarcation between different ROI types and purposes

## Technical Details

### Affected Code Locations

- **File**: `python_client/http_realtime_client.py`
  - **Function/Method**: `_build_widgets()`
  - **Lines**: 273-426 (Panel layout structure)
  - **Issue**: Both ROI frames created within same info_frame parent

- **File**: `python_client/http_realtime_client.py`
  - **Function/Method**: Main frame layout
  - **Lines**: 269-278 (Frame creation)
  - **Issue**: Binary layout doesn't support vertical panel stack

### Data Flow Analysis
**Current Flow:**
```
Main Frame (main_frame)
├── Left Frame (info_frame) - Contains both ROIs
│   ├── ROI1 Frame → ROI1 Display
│   └── ROI2 Frame → ROI2 Display
└── Right Frame (right_frame) - Charts

Expected Flow:**
```
Main Frame (main_frame)
├── Left Frame - Contains general info and controls
├── Top Panel (roi1_panel) - ROI1 monitoring
│   ├── ROI1 Display
│   └── ROI1 Controls
├── Bottom Panel (roi2_panel) - ROI2 monitoring
│   ├── ROI2 Display
│   └── ROI2 Controls
└── Right Frame (right_frame) - Charts
```

### Dependencies
- **tkinter**: ttk.LabelFrame components for panel creation
- **Image Processing**: PIL for ROI image display (200x400)
- **Data Updates**: ROI update logic for both displays

## Impact Analysis

### Direct Impact
- **Visual Confusion**: Users cannot easily distinguish between different ROI purposes
- **Interface Clutter**: Single panel contains too much information
- **Layout Flexibility**: Limited ability to show/hide different ROI panels independently

### Indirect Impact
- **User Experience**: Difficult to understand which ROI serves which purpose
- **Monitoring Efficiency**: Reduced ability to focus on specific ROI types
- **Scalability**: Adding more ROI types would exacerbate the clutter

### Risk Assessment
- **Medium Risk**: Users can still use the functionality but with reduced clarity
- **No Data Loss**: All ROI data processing and updates work correctly
- **UI/UX Impact**: Significantly affects user understanding and workflow efficiency

## Solution Approach

### Fix Strategy
**Vertical Panel Separation**: Create distinct panels for each ROI type with proper vertical stacking.

1. **Restructure Layout**: Move ROI displays out of general info_frame
2. **Create Dedicated Panels**:
   - `roi1_panel`: Top panel for ROI1 monitoring
   - `roi2_panel`: Bottom panel for ROI2 monitoring
3. **Maintain Compatibility**: Preserve all existing functionality and data processing
4. **Enhanced Organization**: Clear visual and functional separation

### Alternative Solutions
1. **Tab-based Interface**: Use tabs to switch between ROI displays
   - *Pros*: Clean separation, easy navigation
   - *Cons*: Cannot view both ROIs simultaneously

2. **Expandable Panels**: Collapsible panels with headers
   - *Pros*: Customizable visibility
   - *Cons*: More complex UI state management

3. **Side-by-Side Layout**: Horizontal separation instead of vertical
   - *Pros*: Both ROIs always visible
   - *Cons*: Requires more horizontal space

**Recommended**: Vertical panel separation for clear visual hierarchy and simultaneous viewing

## Implementation Plan

### Changes Required

1. **Layout Restructuring**
   - **File**: `python_client/http_realtime_client.py`
   - **Modification**: Restructure main frame layout to support three vertical sections
   - **New Components**: Create roi1_panel and roi2_frame as separate containers

2. **Panel Migration**
   - **File**: `python_client/http_realtime_client.py`
   - **Modification**: Move ROI1 and ROI2 frames to their dedicated panels
   - **Preserve**: All existing ROI functionality and UI components

3. **Layout Management**
   - **File**: `python_client/http_realtime_client.py`
   - **Modification**: Implement proper vertical stacking with pack() layout management
   - **Responsive Design**: Ensure both normal and compact modes work correctly

### Testing Strategy
- **Layout Verification**: Confirm proper vertical stacking in both modes
- **Functionality Testing**: Ensure ROI updates work correctly in new panels
- **UI/UX Testing**: Verify clear visual separation and intuitive organization
- **Regression Testing**: Confirm no existing functionality is broken

### Rollback Plan
- **Configuration Backup**: Save current layout structure before changes
- **Revert Strategy**: If issues occur, return ROI frames to info_frame
- **Feature Flag**: Optional parameter to switch between layouts
- **Testing**: Verify rollback restores original functionality

---