# Python Client Matplotlib Real-time Curve Rendering Documentation

## Overview

This document covers the Matplotlib-based real-time curve rendering implementation in the NHEM Python client, focusing on the complete rendering pipeline from API data reception to visual display.

## Architecture Overview

### File Location
- **Main File**: `D:\ProjectPackage\NHEM\python_client\realtime_plotter.py`
- **Supporting Files**:
  - `http_realtime_client.py` (Main GUI application)
  - `simple_http_client.py` (Simplified GUI)
  - `run_realtime_client.py` (Entry point)

### Core Components

```
API Client → Data Manager → Matplotlib Plotter → Tkinter Canvas
    ↓              ↓                ↓                ↓
HTTP Request   Async Processing   Animation Loop  GUI Updates
```

## Matplotlib Plotting Architecture

### Main Plotter Class

```python
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Any

class RealtimePlotter:
    """
    High-performance real-time matplotlib plotter with Tkinter integration
    Features: Multi-series support, peak visualization, zoom/pan, performance monitoring
    """

    def __init__(self, parent_frame, config: Dict[str, Any]):
        self.parent_frame = parent_frame
        self.config = config

        # Data buffers for efficient memory management
        self.max_points = config.get('display', {}).get('buffer_size', 1000)
        self.time_data = deque(maxlen=self.max_points)
        self.signal_data = deque(maxlen=self.max_points)
        self.roi_data = deque(maxlen=self.max_points)
        self.baseline_data = deque(maxlen=self.max_points)

        # Peak data storage
        self.peak_times = deque(maxlen=100)
        self.peak_values = deque(maxlen=100)
        self.peak_colors = deque(maxlen=100)
        self.enhanced_peaks = deque(maxlen=100)

        # Performance monitoring
        self.update_times = deque(maxlen=100)
        self.frame_count = 0
        self.start_time = None

        # Setup matplotlib figure and axes
        self.setup_figure()
        self.setup_plots()
        self.setup_animation()
        self.setup_canvas()

    def setup_figure(self):
        """Setup matplotlib figure with dark theme"""
        # Create figure with specific size
        figsize = self.config.get('display', {}).get('chart_size', (10, 6))
        self.fig = plt.figure(figsize=figsize, facecolor='#1e1e1e')

        # Set dark theme colors
        plt.rcParams['figure.facecolor'] = '#1e1e1e'
        plt.rcParams['axes.facecolor'] = '#252526'
        plt.rcParams['axes.edgecolor'] = '#3e3e42'
        plt.rcParams['axes.labelcolor'] = '#cccccc'
        plt.rcParams['text.color'] = '#cccccc'
        plt.rcParams['grid.color'] = '#3e3e42'
        plt.rcParams['xtick.color'] = '#cccccc'
        plt.rcParams['ytick.color'] = '#cccccc'

        # Configure for better performance
        plt.rcParams['path.simplify'] = True
        plt.rcParams['path.snap'] = True
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.2

    def setup_plots(self):
        """Setup main signal plot with multiple subplots"""
        # Create subplot layout
        gs = self.fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)

        # Main signal plot
        self.ax_main = self.fig.add_subplot(gs[0])
        self.ax_main.set_title('Real-time Signal Monitoring', color='#cccccc', pad=20)
        self.ax_main.set_ylabel('Signal Value', color='#cccccc')
        self.ax_main.grid(True, alpha=0.2)

        # ROI display plot
        self.ax_roi = self.fig.add_subplot(gs[1])
        self.ax_roi.set_title('ROI Gray Value', color='#cccccc')
        self.ax_roi.set_xlabel('Time (seconds)', color='#cccccc')
        self.ax_roi.set_ylabel('Gray Value', color='#cccccc')
        self.ax_roi.grid(True, alpha=0.2)

        # Configure Y-axis ranges
        self.ax_main.set_ylim(0, 200)  # Fixed range for signal
        self.ax_roi.set_ylim(0, 255)     # Full grayscale range

        # Initialize line objects for efficient updating
        self.lines = {}
        self.setup_line_objects()

    def setup_line_objects(self):
        """Initialize all line objects for efficient updates"""
        # Main signal line (VS Code green theme)
        self.lines["signal"], = self.ax_main.plot(
            [], [],
            color='#4ec9b0', linewidth=2.0,
            label='Signal', alpha=0.8
        )

        # Baseline line (VS Code yellow theme)
        self.lines["baseline"], = self.ax_main.plot(
            [], [],
            color='#dcdcaa', linewidth=1.0,
            linestyle='--', label='Baseline', alpha=0.7
        )

        # Enhanced peak lines
        self.lines["enhanced_peaks_green"], = self.ax_main.plot(
            [], [], 'o', color='#4ec9b0', markersize=8,
            label='Green Peaks', alpha=0.9, markeredgewidth=2, markeredgecolor='#2d6b2d'
        )

        self.lines["enhanced_peaks_red"], = self.ax_main.plot(
            [], [], 'o', color='#f48771', markersize=8,
            label='Red Peaks', alpha=0.9, markeredgewidth=2, markeredgecolor='#8b3a3a'
        )

        # Traditional peak indicators
        self.lines["peaks"], = self.ax_main.plot(
            [], [], '^', color='#ff6b6b', markersize=6,
            label='Traditional Peaks', alpha=0.8
        )

        # ROI signal line
        self.lines["roi"], = self.ax_roi.plot(
            [], [], color='#569cd6', linewidth=1.5,
            label='ROI Signal', alpha=0.8
        )

        # Configure legends
        self.ax_main.legend(loc='upper right', facecolor='#252526', edgecolor='#3e3e42')
        self.ax_roi.legend(loc='upper right', facecolor='#252526', edgecolor='#3e3e42')
```

## Real-time Animation System

### Animation Loop Implementation

```python
def setup_animation(self):
    """Setup matplotlib animation for real-time updates"""
    # Create animation object with optimized settings
    self.animation = animation.FuncAnimation(
        self.fig,
        self.update_plot,
        interval=self.config.get('display', {}).get('update_interval', 50),  # 20 FPS
        blit=False,  # blit=True can cause issues with complex plots
        cache_frame_data=False,  # Disable frame caching for memory efficiency
        repeat=True,
        init_func=self.init_animation
    )

def init_animation(self):
    """Initialize animation (called once)"""
    # Clear all line data
    for line in self.lines.values():
        line.set_data([], [])

    return list(self.lines.values())

def update_plot(self, frame=None):
    """
    Main animation update function called at 20 FPS
    Handles all data updates and rendering
    """
    start_time = time.time()

    try:
        # Update time-based data
        self.update_time_series()

        # Update plot data
        self.update_line_data()

        # Adjust plot limits dynamically
        self.update_plot_limits()

        # Update annotations and markers
        self.update_annotations()

        # Performance monitoring
        self.monitor_performance(start_time)

    except Exception as e:
        print(f"Animation update error: {e}")
        # Don't raise exception to prevent animation stopping

    return list(self.lines.values())

def update_time_series(self):
    """Update time-based data series with rolling window"""
    if len(self.time_data) == 0:
        return

    # Convert deque to numpy array for efficient operations
    time_array = np.array(self.time_data)
    signal_array = np.array(self.signal_data)

    # Calculate rolling baseline (using last N points)
    baseline_window = min(50, len(signal_array))
    if baseline_window > 0:
        baseline = np.mean(signal_array[-baseline_window:])
        baseline_array = np.full_like(time_array, baseline)
        self.baseline_data.clear()
        self.baseline_data.extend(baseline_array)
    else:
        baseline_array = np.zeros_like(time_array)

def update_line_data(self):
    """Update all line objects with current data"""
    # Convert to numpy arrays for matplotlib
    time_array = np.array(self.time_data) if self.time_data else np.array([])
    signal_array = np.array(self.signal_data) if self.signal_data else np.array([])
    roi_array = np.array(self.roi_data) if self.roi_data else np.array([])
    baseline_array = np.array(self.baseline_data) if self.baseline_data else np.array([])

    # Update main signal line
    if len(time_array) > 0:
        self.lines["signal"].set_data(time_array, signal_array)

    # Update baseline line
    if len(baseline_array) > 0:
        self.lines["baseline"].set_data(time_array, baseline_array)

    # Update enhanced peak markers
    if len(self.peak_times) > 0:
        self.lines["enhanced_peaks_green"].set_data(
            [t for t, c in zip(self.peak_times, self.peak_colors) if c == 'green'],
            [v for v, c in zip(self.peak_values, self.peak_colors) if c == 'green']
        )

        self.lines["enhanced_peaks_red"].set_data(
            [t for t, c in zip(self.peak_times, self.peak_colors) if c == 'red'],
            [v for v, c in zip(self.peak_values, self.peak_colors) if c == 'red']
        )

    # Update ROI signal line
    if len(time_array) > 0 and len(roi_array) > 0:
        # Align ROI data with main timeline
        roi_time_array = time_array[-len(roi_array):]
        self.lines["roi"].set_data(roi_time_array, roi_array)

def update_plot_limits(self):
    """Update plot limits for optimal viewing"""
    if len(self.time_data) == 0:
        return

    # Get current data range
    time_min = min(self.time_data)
    time_max = max(self.time_data)

    # Add padding for better visualization
    time_padding = (time_max - time_min) * 0.05

    # Update X-axis limits (dynamic based on data)
    if len(self.time_data) >= self.max_points:
        # Full buffer case: show sliding window
        x_min = time_min
        x_max = time_max + time_padding
    else:
        # Partial buffer case: show from start with future space
        x_min = 0
        x_max = time_max + 1.0 + time_padding

    self.ax_main.set_xlim(x_min, x_max)
    self.ax_roi.set_xlim(x_min, x_max)

    # Y-axis limits remain fixed for consistency
    self.ax_main.set_ylim(0, 200)
    self.ax_roi.set_ylim(0, 255)
```

## Data Integration System

### Real-time Data Manager

```python
class RealtimeDataManager:
    """Manages real-time data flow from API to plotter"""

    def __init__(self, api_client, plotter, config):
        self.api_client = api_client
        self.plotter = plotter
        self.config = config

        # Update control
        self.is_running = False
        self.update_interval = config.get('display', {}).get('update_interval', 0.05)  # 50ms
        self.last_update = 0

        # Performance tracking
        self.update_times = deque(maxlen=100)
        self.error_count = 0
        self.last_data_time = None

        # Event callbacks
        self.callbacks = {
            'data_received': [],
            'error': [],
            'connection_lost': [],
            'connection_restored': []
        }

    async def start_updates(self):
        """Start real-time data updates"""
        if self.is_running:
            return

        self.is_running = True
        print(f"Starting real-time updates at {1/self.update_interval:.1f} FPS")

        try:
            await self.update_loop()
        except Exception as e:
            print(f"Update loop error: {e}")
            raise
        finally:
            self.is_running = False
            print("Real-time updates stopped")

    async def update_loop(self):
        """Main update loop with error handling and backoff"""
        consecutive_errors = 0
        max_consecutive_errors = 5

        while self.is_running:
            try:
                start_time = asyncio.get_event_loop().time()

                # Fetch data from server
                data = await self.api_client.get_realtime_data(
                    count=self.config.get('display', {}).get('buffer_size', 100)
                )

                # Validate and process data
                if self.validate_data(data):
                    await self.process_data(data)
                    self.trigger_callback('data_received', data)

                    # Reset error count on success
                    consecutive_errors = 0
                    self.error_count = 0

                    # Trigger connection restored callback if needed
                    if self.error_count > 0:
                        self.trigger_callback('connection_restored')

                # Calculate sleep time to maintain update interval
                elapsed = asyncio.get_event_loop().time() - start_time
                sleep_time = max(0, self.update_interval - elapsed)

                await asyncio.sleep(sleep_time)

            except Exception as e:
                consecutive_errors += 1
                self.error_count += 1

                print(f"Update error (attempt {consecutive_errors}/{max_consecutive_errors}): {e}")
                self.trigger_callback('error', e)

                # Exponential backoff on consecutive errors
                if consecutive_errors >= max_consecutive_errors:
                    self.trigger_callback('connection_lost')
                    await asyncio.sleep(5)  # Wait before retrying
                    consecutive_errors = 0
                else:
                    await asyncio.sleep(min(2 ** consecutive_errors, 10))

    async def process_data(self, data: Dict[str, Any]):
        """Process received data and update plotter"""
        try:
            # Extract time series data
            series = data.get('series', [])
            if not series:
                print("No time series data received")
                return

            # Process each data point
            current_time = asyncio.get_event_loop().time()
            base_time = current_time  # Use current time as reference

            for i, point in enumerate(series):
                # Calculate relative time
                relative_time = point.get('t', i * 0.022)  # 22ms per frame

                # Add data point to plotter
                self.plotter.add_data_point(
                    timestamp=base_time + relative_time,
                    value=point.get('value', 0)
                )

            # Handle ROI data if available
            roi_data = data.get('roi_data')
            if roi_data and roi_data.get('gray_value', 0) > 0:
                self.plotter.add_roi_data_point(
                    timestamp=base_time,
                    gray_value=roi_data['gray_value']
                )

            # Handle enhanced peak information
            enhanced_peak = data.get('enhanced_peak')
            if enhanced_peak and enhanced_peak.get('signal') == 1:
                self.plotter.add_enhanced_peak(
                    timestamp=base_time,
                    value=data.get('current_value', 0),
                    color=enhanced_peak.get('color', 'unknown'),
                    confidence=enhanced_peak.get('confidence', 0.0)
                )

            # Update plotter's internal state
            self.plotter.update_frame_count(data.get('frame_count', 0))
            self.plotter.update_baseline(data.get('baseline', 0))

            self.last_data_time = current_time

        except Exception as e:
            print(f"Data processing error: {e}")
            raise
```

### Plotter Data Management

```python
def add_data_point(self, timestamp, value):
    """Add new signal data point with efficient buffer management"""
    # Convert timestamp to relative time if this is the first point
    if not self.time_data:
        self.start_time = timestamp

    # Calculate relative time in seconds
    relative_time = (timestamp - self.start_time).total_seconds()

    # Add to circular buffers
    self.time_data.append(relative_time)
    self.signal_data.append(value)

def add_roi_data_point(self, timestamp, gray_value):
    """Add ROI gray value data point"""
    if not self.start_time:
        self.start_time = timestamp

    relative_time = (timestamp - self.start_time).total_seconds()
    self.roi_data.append(gray_value)

def add_enhanced_peak(self, timestamp, value, color, confidence):
    """Add enhanced peak detection result"""
    if not self.start_time:
        self.start_time = timestamp

    relative_time = (timestamp - self.start_time).total_seconds()

    self.enhanced_peaks.append({
        'time': relative_time,
        'value': value,
        'color': color,
        'confidence': confidence,
        'timestamp': timestamp
    })

    # Also add to peak marker arrays
    self.peak_times.append(relative_time)
    self.peak_values.append(value)
    self.peak_colors.append(color)
```

## Tkinter Integration

### Canvas Embedding

```python
def setup_canvas(self):
    """Setup Tkinter canvas integration for matplotlib"""
    # Create canvas for matplotlib figure
    self.canvas = FigureCanvasTkAgg(self.fig, master=self.parent_frame)

    # Configure canvas
    self.canvas.draw()
    self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Add navigation toolbar
    toolbar_frame = ttk.Frame(self.parent_frame)
    toolbar_frame.pack(fill=tk.X, side=tk.BOTTOM)

    self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
    self.toolbar.update()

    # Pack toolbar
    self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    # Configure matplotlib toolbar theme to match dark theme
    self.style_toolbar()

def style_toolbar(self):
    """Apply dark theme to matplotlib toolbar"""
    try:
        # Access toolbar buttons and apply dark theme
        for widget in self.toolbar.winfo_children():
            if isinstance(widget, ttk.Button):
                widget.configure(style='Dark.TButton')
            elif isinstance(widget, ttk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, ttk.Button):
                        child.configure(style='Dark.TButton')
    except Exception as e:
        print(f"Toolbar styling error: {e}")

class DarkThemeStyle:
    """Custom ttk style for dark theme integration"""

    def __init__(self):
        self.style = ttk.Style()
        self.setup_dark_theme()

    def setup_dark_theme(self):
        # Configure dark theme colors
        dark_colors = {
            'bg': '#1e1e1e',
            'fg': '#cccccc',
            'selectbg': '#094771',
            'selectfg': '#ffffff',
            'button': '#252526',
            'buttonhover': '#2a2d2e'
        }

        # Configure styles
        self.style.configure('Dark.TFrame', background=dark_colors['bg'])
        self.style.configure('Dark.TButton',
                           background=dark_colors['button'],
                           foreground=dark_colors['fg'],
                           borderwidth=1,
                           focuscolor=dark_colors['selectbg'])
        self.style.map('Dark.TButton',
                      background=[('active', dark_colors['buttonhover'])])
```

## Performance Optimization

### Memory Management

```python
class PerformanceMonitor:
    """Monitor and optimize matplotlib performance"""

    def __init__(self, plotter):
        self.plotter = plotter
        self.metrics = {
            'frame_times': deque(maxlen=100),
            'memory_usage': deque(maxlen=100),
            'fps': deque(maxlen=100),
            'data_sizes': deque(maxlen=100)
        }

    def monitor_frame_time(self, frame_time):
        """Monitor frame rendering time"""
        self.metrics['frame_times'].append(frame_time)

        # Check for performance issues
        if frame_time > 0.1:  # 100ms threshold
            print(f"Warning: Slow frame render time: {frame_time*1000:.1f}ms")

    def get_performance_report(self):
        """Generate comprehensive performance report"""
        frame_times = list(self.metrics['frame_times'])
        fps_values = list(self.metrics['fps'])

        report = {
            'avg_frame_time': np.mean(frame_times) if frame_times else 0,
            'max_frame_time': np.max(frame_times) if frame_times else 0,
            'avg_fps': np.mean(fps_values) if fps_values else 0,
            'min_fps': np.min(fps_values) if fps_values else 0,
            'memory_usage': self.get_memory_usage()
        }

        return report

    def optimize_plotter_settings(self):
        """Apply performance optimizations to plotter"""
        # Optimize matplotlib settings for real-time rendering
        plt.rcParams['animation.html'] = 'html5'
        plt.rcParams['animation.writer'] = 'ffmpeg'
        plt.rcParams['animation.bitrate'] = 2000

        # Optimize path rendering
        plt.rcParams['path.simplify'] = True
        plt.rcParams['path.snap'] = True

        # Disable unnecessary features for performance
        plt.rcParams['figure.max_open_warning'] = 0
        plt.rcParams['axes.autolimit_mode'] = 'round_numbers'

class MemoryOptimizedPlotter(RealtimePlotter):
    """Memory-optimized version of the plotter"""

    def __init__(self, parent_frame, config):
        super().__init__(parent_frame, config)

        # Reduce data resolution for memory efficiency
        self.data_reduction_factor = config.get('display', {}).get('data_reduction', 1)

        # Memory usage tracking
        self.memory_tracker = MemoryTracker()

    def add_data_point(self, timestamp, value):
        """Add data point with memory optimization"""
        # Only keep every nth point if reduction is enabled
        if self.data_reduction_factor > 1:
            if len(self.time_data) % self.data_reduction_factor != 0:
                return  # Skip this point

        super().add_data_point(timestamp, value)
        self.memory_tracker.track_data_size(len(self.time_data))

class MemoryTracker:
    """Track memory usage and optimize when necessary"""

    def __init__(self, max_memory_mb=100):
        self.max_memory_mb = max_memory_mb
        self.current_memory_mb = 0

    def track_data_size(self, data_size):
        """Track current memory usage"""
        # Estimate memory usage (each data point ~24 bytes)
        estimated_bytes = data_size * 24
        self.current_memory_mb = estimated_bytes / (1024 * 1024)

        # Trigger optimization if needed
        if self.current_memory_mb > self.max_memory_mb:
            print(f"Memory usage high: {self.current_memory_mb:.1f}MB, triggering optimization")
            return True
        return False
```

## Advanced Features

### Peak Detection Visualization

```python
def update_annotations(self):
    """Update peak annotations and indicators"""
    # Clear existing annotations
    for text in self.ax_main.texts:
        text.remove()

    # Add enhanced peak annotations
    if self.enhanced_peaks:
        recent_peak = self.enhanced_peaks[-1]  # Show most recent peak
        if recent_peak['confidence'] > 0.7:  # High confidence peaks only
            self.ax_main.annotate(
                f"{recent_peak['color'].upper()} PEAK",
                xy=(recent_peak['time'], recent_peak['value']),
                xytext=(recent_peak['time'] + 1, recent_peak['value'] + 10),
                fontsize=8,
                color='#4ec9b0' if recent_peak['color'] == 'green' else '#f48771',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='#252526', edgecolor='#3e3e42'),
                arrowprops=dict(arrowstyle='->', color='#4ec9b0' if recent_peak['color'] == 'green' else '#f48771')
            )

def add_peak_statistics(self):
    """Add peak statistics display to plot"""
    if not self.peak_times:
        return

    # Calculate peak statistics
    total_peaks = len(self.peak_times)
    green_peaks = sum(1 for color in self.peak_colors if color == 'green')
    red_peaks = total_peaks - green_peaks

    if total_peaks > 0:
        # Calculate peak rate (peaks per minute)
        time_span = (self.peak_times[-1] - self.peak_times[0]) if len(self.peak_times) > 1 else 1
        peak_rate = (total_peaks / time_span) * 60 if time_span > 0 else 0

        # Add statistics text box
        stats_text = f"Total Peaks: {total_peaks}\nGreen: {green_peaks}\nRed: {red_peaks}\nRate: {peak_rate:.1f}/min"

        self.ax_main.text(
            0.02, 0.98, stats_text,
            transform=self.ax_main.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#252526', alpha=0.8, edgecolor='#3e3e42'),
            color='#cccccc'
        )
```

### Interactive Features

```python
def setup_interactivity(self):
    """Setup mouse interaction for zooming and panning"""
    from matplotlib.widgets import Button, Slider

    # Add zoom controls
    ax_zoom_in = plt.axes([0.8, 0.01, 0.08, 0.04])
    ax_zoom_out = plt.axes([0.89, 0.01, 0.08, 0.04])
    ax_reset = plt.axes([0.98, 0.01, 0.08, 0.04])

    self.btn_zoom_in = Button(ax_zoom_in, 'Zoom In', color='#4ec9b0')
    self.btn_zoom_out = Button(ax_zoom_out, 'Zoom Out', color='#dcdcaa')
    self.btn_reset = Button(ax_reset, 'Reset', color='#f48771')

    # Connect button events
    self.btn_zoom_in.on_clicked(self.zoom_in)
    self.btn_zoom_out.on_clicked(self.zoom_out)
    self.btn_reset.on_clicked(self.reset_view)

    # Add threshold slider
    ax_threshold = plt.axes([0.15, 0.02, 0.3, 0.03])
    self.threshold_slider = Slider(
        ax_threshold, 'Threshold', 50, 200,
        valinit=105, color='#4ec9b0'
    )
    self.threshold_slider.on_changed(self.update_threshold)

def zoom_in(self, event):
    """Zoom in the plot"""
    xlim = self.ax_main.get_xlim()
    ylim = self.ax_main.get_ylim()

    x_center = (xlim[0] + xlim[1]) / 2
    y_center = (ylim[0] + ylim[1]) / 2

    x_range = (xlim[1] - xlim[0]) * 0.8
    y_range = (ylim[1] - ylim[0]) * 0.8

    self.ax_main.set_xlim(x_center - x_range/2, x_center + x_range/2)
    self.ax_main.set_ylim(y_center - y_range/2, y_center + y_range/2)
    self.ax_roi.set_xlim(x_center - x_range/2, x_center + x_range/2)

    self.fig.canvas.draw_idle()

def update_threshold(self, val):
    """Update peak detection threshold"""
    new_threshold = self.threshold_slider.val
    print(f"Peak threshold updated to: {new_threshold}")

    # This would typically trigger an API call to update backend
    # For demonstration, just update the visual threshold line
    if hasattr(self, 'threshold_line'):
        self.threshold_line.set_ydata([new_threshold, new_threshold])
        self.fig.canvas.draw_idle()
```

### Export Capabilities

```python
def export_current_view(self, filename=None):
    """Export current plot view to image file"""
    if filename is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"nhem_plot_{timestamp}.png"

    try:
        self.fig.savefig(
            filename,
            dpi=150,
            facecolor='#1e1e1e',
            edgecolor='none',
            bbox_inches='tight',
            pad_inches=0.1
        )
        print(f"Plot exported to: {filename}")
        return True
    except Exception as e:
        print(f"Export failed: {e}")
        return False

def export_data(self, filename=None):
    """Export current data to CSV file"""
    if filename is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"nhem_data_{timestamp}.csv"

    try:
        import csv

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            writer.writerow(['Time(s)', 'Signal', 'ROI_Value', 'Baseline', 'Peak'])

            # Write data
            min_length = min(len(self.time_data), len(self.signal_data))
            for i in range(min_length):
                roi_val = self.roi_data[i] if i < len(self.roi_data) else ''
                baseline_val = self.baseline_data[i] if i < len(self.baseline_data) else ''
                peak_val = 1 if i in self.peak_times else ''

                writer.writerow([
                    f"{self.time_data[i]:.3f}",
                    f"{self.signal_data[i]:.2f}",
                    f"{roi_val}",
                    f"{baseline_val:.2f}",
                    peak_val
                ])

        print(f"Data exported to: {filename}")
        return True
    except Exception as e:
        print(f"Data export failed: {e}")
        return False
```

This Matplotlib-based real-time curve rendering system provides a professional, high-performance visualization solution with comprehensive features including multi-series plotting, peak detection visualization, interactive controls, and export capabilities. The system is optimized for memory efficiency and smooth 20 FPS updates while maintaining the VS Code dark theme consistency.