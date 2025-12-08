# Bug Report

## Bug Summary
ROI screenshot area in http_realtime_client.py needs to be split into left and right sections with a "|" separator, displaying the same ROI content in both sections

## Bug Details

### Expected Behavior
The ROI screenshot display area in the Python client (http_realtime_client.py) should be divided into two sections:
- **Left section**: Shows the current ROI capture from backend
- **Center**: A vertical "|" text separator between sections
- **Right section**: Shows the same ROI content as the left section
- Both sections should maintain the current 200×150 pixel display size
- Both sections should update simultaneously with the same ROI data

### Actual Behavior
Currently, the ROI screenshot display in the Python client shows only a single ROI image in one `ttk.Label` widget (lines 386-388). There is no split display or separator, and users cannot see the ROI content duplicated side by side.

The current implementation uses:
- Single `ttk.Label` widget at line 386: `self.roi_label = ttk.Label(roi_frame, text="Waiting for ROI data...", relief="sunken", background="white")`
- Single ROI image display in the `update_roi_screenshot()` method (lines 719-776)
- No layout support for dual ROI display or visual separator

### Steps to Reproduce
1. Start the Python HTTP client: `python http_realtime_client.py`
2. Ensure backend server is running and connected
3. Start ROI detection functionality
4. Observe the ROI Screenshot panel in the left information panel
5. Note that only one ROI image is displayed, without any split layout or "|" separator
6. Compare with the expected dual-section layout requirement

### Environment
- **Version**: NHEM Python Client (http_realtime_client.py)
- **Platform**: Desktop application using tkinter GUI
- **Dependencies**: Python 3.x, PIL/Pillow, tkinter, requests
- **Configuration**: Standard HTTP client with ROI screenshot functionality

## Impact Assessment

### Severity
- [x] Medium - Feature impaired but workaround exists (users can see ROI but not in split format)

### Affected Users
- Users requiring side-by-side ROI comparison functionality in the desktop client
- Users who need to monitor ROI changes with visual duplication in Python GUI
- Applications requiring split ROI display for analysis purposes in desktop environment

### Affected Features
- ROI screenshot display functionality in Python client
- Visual monitoring interface in desktop application
- User experience for ROI analysis in tkinter GUI

## Additional Context

### Error Messages
```
No specific error messages - this is a feature enhancement request rather than a functional bug
```

### Screenshots/Media
Current implementation shows:
- Single ROI label widget in tkinter frame
- 200×150 pixel ROI image display
- No split layout or separator functionality

Expected implementation should show:
- Two ROI display sections side by side
- Vertical "|" separator between sections
- Identical ROI content in both sections

### Related Issues
- Previous similar fix implemented in web frontend (front/index.html)
- ROI capture and processing functionality already exists in backend
- Single ROI display working correctly in current implementation

## Initial Analysis

### Suspected Root Cause
The current ROI display implementation in the Python client was designed for single image display using a single `ttk.Label` widget. The tkinter layout structure and image handling logic need to be modified to support dual ROI display with a separator.

### Affected Components
- **File**: `python_client/http_realtime_client.py`
  - **Function**: `__init__()` (lines 381-388) - ROI frame and label creation
  - **Function**: `update_roi_screenshot()` (lines 719-776) - ROI image update logic
  - **Widget**: `self.roi_label` - Single label widget for ROI display
  - **Layout**: ROI frame structure needs dual widget support

---

**File Location**: `.claude/bugs/http-client-roi-split/report.md`
**Status**: Ready for review and approval to proceed to analysis phase