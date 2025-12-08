# Bug Report

## Bug Summary
ROI screenshot area needs to be split into left and right sections with a "|" separator, displaying the same ROI content in both sections

## Bug Details

### Expected Behavior
The ROI screenshot display area should be divided into two equal sections:
- **Left section**: Shows the current ROI capture
- **Center**: A vertical "|" text separator
- **Right section**: Shows the same ROI content as the left section
- Both sections should maintain the current 200×150 pixel dimensions (total 400×150 + separator)
- The display should show identical ROI content in both sections simultaneously

### Actual Behavior
Currently, the ROI display shows only a single ROI screenshot in one canvas element (200×150 pixels). There is no split display or separator, and users cannot see the ROI content duplicated side by side.

### Steps to Reproduce
1. Start the NHEM backend server (`python run.py`)
2. Launch the frontend (`python -m http.server 3000` or open `front/index.html`)
3. Access the application in a browser at `http://localhost:3000`
4. Configure and start ROI detection
5. Observe the ROI screenshot display area - only one ROI image is shown
6. Note the absence of split display and "|" separator

### Environment
- **Version**: NHEM (New HEM Monitor) with dual ROI system implementation
- **Platform**: Web-based frontend (HTML5, CSS3, JavaScript)
- **Browser**: Any modern browser supporting HTML5 Canvas
- **Configuration**: Current ROI display with single canvas element at `roi-canvas`

## Impact Assessment

### Severity
- [x] Medium - Feature impaired but workaround exists (users can see ROI but not in split format)

### Affected Users
- Users requiring side-by-side ROI comparison functionality
- Users who need to monitor ROI changes with visual duplication
- Applications requiring split ROI display for analysis purposes

### Affected Features
- ROI screenshot display functionality
- Visual monitoring interface
- User experience for ROI analysis

## Additional Context

### Error Messages
```
No specific error messages - this is a feature enhancement request rather than a bug in functionality
```

### Screenshots/Media
Current implementation shows:
- Single ROI canvas element (200×150 pixels)
- VS Code dark theme styling
- No split layout or separator

Expected implementation should show:
- Two ROI canvases side by side (each 200×150 pixels)
- Vertical "|" separator between them
- Identical ROI content in both sections

### Related Issues
- Dual ROI system implementation already exists in backend (`DualRoiFrame`, `DualRoiConfig` in `models.py`)
- ROI capture functionality implemented in `temp_roi_capture.py`
- Frontend RoiRenderer class ready for extension in `front/index.html` (lines 2029-2134)

## Initial Analysis

### Suspected Root Cause
The current ROI display implementation was designed for single ROI display and does not include the split layout functionality. The RoiRenderer class and associated HTML/CSS need to be modified to support dual display with separator.

### Affected Components
- **Frontend**: `front/index.html` - ROI display HTML structure and styling
- **Frontend**: RoiRenderer class (lines 2029-2134) - JavaScript rendering logic
- **CSS**: ROI canvas styling and layout
- **Display Logic**: `updateRoiDisplay()` function - ROI update mechanism

---

**File Location**: `.claude/bugs/roi-screenshot-split/report.md`
**Status**: Ready for review and approval to proceed to analysis phase