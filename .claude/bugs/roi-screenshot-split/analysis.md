# Bug Analysis

## Root Cause Analysis

### Investigation Summary
I've conducted a thorough investigation of the ROI display implementation and identified the specific code locations that need modification. The current system uses a single canvas element with a dedicated RoiRenderer class that handles image rendering with double buffering for smooth performance.

### Root Cause
The ROI screenshot area is currently implemented as a single canvas element (`#roi-canvas`) with fixed dimensions (200×150 pixels). The system lacks the HTML structure, CSS styling, and JavaScript logic required to display a split view with two identical ROI sections separated by a "|" character.

### Contributing Factors
1. **HTML Structure**: Only one canvas element exists in the ROI monitoring panel
2. **CSS Styling**: No styling for split layout or separator element
3. **JavaScript Logic**: RoiRenderer class designed for single canvas rendering
4. **Layout Constraints**: Fixed container width not designed for dual display

## Technical Details

### Affected Code Locations

**HTML Structure (line 565)**:
- **File**: `front/index.html`
- **Element**: `<canvas id="roi-canvas" width="200" height="150"></canvas>`
- **Issue**: Single canvas element needs to be replaced with dual canvas structure

**CSS Styling (lines 245-252)**:
```css
#roi-canvas {
    background-color: #000;
    border: 1px solid var(--border-color);
    width: 200px;
    height: 150px;
    margin: 0 auto;
    image-rendering: pixelated;
}
```
- **Issue**: Single canvas styling needs to support dual canvas layout with separator

**JavaScript RoiRenderer Class (lines 2029-2134)**:
- **Constructor**: `RoiRenderer(canvasId)` - designed for single canvas
- **Rendering Logic**: `render(roiData)` - renders to single canvas only
- **Issue**: Class needs modification to handle dual canvas rendering

**ROI Update Function (around line 2350)**:
- **Function**: `updateRoiDisplay()` - updates single ROI display
- **Issue**: Needs modification to update both canvas elements

### Data Flow Analysis
```
Current Flow:
Backend ROI Data → Single RoiRenderer → Single Canvas Display

Required Flow:
Backend ROI Data → DualRoiRenderer → Left Canvas + Right Canvas + Separator
```

### Dependencies
- **HTML Canvas API**: For dual canvas rendering
- **CSS Flexbox**: For layout management
- **JavaScript DOM Manipulation**: For dynamic updates
- **Existing RoiRenderer**: Base functionality to extend

## Impact Analysis

### Direct Impact
- Users cannot see ROI content in split format as requested
- Missing visual comparison capability
- Limited monitoring display options

### Indirect Impact
- No functional impact on ROI capture or processing
- Backend dual ROI system remains underutilized
- User experience not optimized for analysis workflows

### Risk Assessment
- **Low Risk**: Changes are confined to frontend display layer
- **No Backend Impact**: Existing ROI processing remains unchanged
- **Backward Compatibility**: Can maintain fallback to single display

## Solution Approach

### Fix Strategy
Implement a dual ROI display system by:
1. **HTML Structure**: Replace single canvas with dual canvas container
2. **CSS Layout**: Create flexbox layout with separator styling
3. **JavaScript Logic**: Extend RoiRenderer to handle dual canvas rendering
4. **Backward Compatibility**: Maintain existing functionality

### Alternative Solutions
1. **Single Canvas Split**: Draw both ROIs on single canvas with divider
   - Pros: Single element, simpler DOM
   - Cons: Complex rendering calculations, less flexible

2. **CSS Grid Layout**: Use CSS grid instead of flexbox
   - Pros: Precise control over layout
   - Cons: More complex, browser compatibility concerns

3. **Dynamic Toggle**: Add option to switch between single/dual modes
   - Pros: Maximum flexibility
   - Cons: Increased complexity, additional UI controls

### Risks and Trade-offs
- **Performance**: Minimal impact, dual rendering is efficient
- **Maintenance**: Slightly increased code complexity
- **User Experience**: Significant improvement for monitoring workflows

## Implementation Plan

### Changes Required

1. **HTML Structure Modification**
   - **File**: `front/index.html` (line 565)
   - **Modification**: Replace single canvas with dual canvas container

2. **CSS Styling Updates**
   - **File**: `front/index.html` (lines 245-252)
   - **Modification**: Add dual canvas layout and separator styling

3. **JavaScript RoiRenderer Enhancement**
   - **File**: `front/index.html` (lines 2029-2134)
   - **Modification**: Extend class to support dual canvas rendering

4. **ROI Update Function Updates**
   - **File**: `front/index.html` (around line 2350)
   - **Modification**: Update to handle dual canvas elements

### Testing Strategy
1. **Unit Testing**: Verify RoiRenderer dual functionality
2. **Integration Testing**: Test ROI data flow to dual displays
3. **Visual Testing**: Verify separator and layout rendering
4. **Performance Testing**: Ensure no performance degradation

### Rollback Plan
- Keep original single canvas code as commented fallback
- Maintain backward compatibility flag
- CSS can easily revert to single canvas styling

---

**File Location**: `.claude/bugs/roi-screenshot-split/analysis.md`
**Status**: Analysis complete, ready for implementation phase