# Python Client Architecture Documentation

## Overview

The NHEM Python client ecosystem provides multiple client implementations for different use cases, ranging from full-featured GUI applications to lightweight command-line tools. All clients share a common architecture for API communication and configuration management.

## Technology Stack

### Core Dependencies
- **GUI Framework**: Tkinter (Python standard library)
- **Data Visualization**: Matplotlib 3.3.0+
- **HTTP Communication**: Requests 2.25.0+
- **Image Processing**: Pillow 8.0.0+
- **Data Processing**: NumPy 1.20.0+
- **Configuration**: JSON-based configuration management

### Architecture Pattern
- **Modular Design**: Separate components for API, GUI, and data processing
- **Observer Pattern**: Event-driven UI updates
- **Factory Pattern**: Multiple client variants from shared components
- **Configuration-Driven**: External configuration files with runtime updates

## Project Structure

```
D:\ProjectPackage\NHEM\python_client\
├── run_realtime_client.py          # Main entry point (recommended)
├── http_realtime_client.py         # Full-featured GUI application
├── simple_http_client.py           # Simplified GUI version
├── realtime_plotter.py             # Real-time plotting component
├── client.py                       # Command-line API client
├── local_config_loader.py          # Configuration management
├── http_client_config.json         # Client configuration
└── README.md                       # Comprehensive documentation
```

## Client Variants

### 1. Full GUI Client (`http_realtime_client.py`)

**Purpose**: Professional monitoring with complete feature set
**Target Users**: Researchers, engineers, system administrators

#### Architecture
```python
class NHEMRealtimeClient:
    """Full-featured GUI client with advanced monitoring capabilities"""

    def __init__(self):
        self.config = LocalConfigLoader()
        self.api_client = APIClient(self.config)
        self.gui = MainWindow(self.api_client, self.config)
        self.plotter = RealtimePlotter(self.gui.canvas_frame)
        self.updater = DataUpdater(self.api_client, self.plotter)

    def run(self):
        """Start the GUI application"""
        self.setup_gui()
        self.start_realtime_updates()
        self.gui.mainloop()
```

#### Key Features
- **Real-time Plotting**: 20 FPS matplotlib charts with multiple data series
- **Interactive Controls**: Start/stop/pause/resume system control
- **ROI Configuration**: Visual ROI setup with live preview
- **Parameter Tuning**: Real-time adjustment of detection parameters
- **System Monitoring**: CPU, memory, and network status display
- **Data Export**: CSV and PNG export capabilities
- **Configuration Management**: Runtime configuration with persistence

### 2. Simplified GUI Client (`simple_http_client.py`)

**Purpose**: Lightweight monitoring with essential features
**Target Users**: Quick deployments, basic monitoring needs

#### Architecture
```python
class SimpleHTTPClient:
    """Simplified GUI client with essential monitoring features"""

    def __init__(self):
        self.config = self.load_config()
        self.api = APIClient(self.config)
        self.root = tk.Tk()
        self.setup_gui()

    def setup_gui(self):
        """Create simplified GUI layout"""
        # Basic chart display
        # Essential control buttons
        # Status display
        # Minimal configuration options
```

#### Key Features
- **Basic Real-time Charts**: Single data series visualization
- **Essential Controls**: Start/stop functionality
- **Status Display**: System health and basic metrics
- **Lightweight**: Minimal resource usage
- **Easy Deployment**: Single file deployment

### 3. Command-Line Client (`client.py`)

**Purpose**: Scriptable API access for automation
**Target Users**: Developers, system integrators, automated testing

#### Architecture
```python
class CommandLineClient:
    """Command-line interface for API access"""

    def __init__(self):
        self.parser = self.setup_argument_parser()
        self.config = self.load_config()
        self.api_client = APIClient(self.config)

    def run(self, args=None):
        """Execute command-line operations"""
        args = self.parser.parse_args(args)

        if args.command == 'status':
            self.show_status()
        elif args.command == 'start':
            self.start_detection()
        elif args.command == 'config':
            self.manage_config(args)
        # ... other commands
```

#### Key Features
- **Complete API Coverage**: Access to all backend endpoints
- **Scriptable Operations**: Suitable for automation and batch processing
- **Configuration Management**: JSON-based configuration manipulation
- **Data Export**: Export real-time data to files
- **Testing Support**: API endpoint testing and validation

## Core Architecture Components

### 1. Configuration Management

#### LocalConfigLoader
```python
class LocalConfigLoader:
    """Configuration management with automatic discovery and validation"""

    def __init__(self, config_file='http_client_config.json'):
        self.config_file = config_file
        self.config = self._load_config()
        self.observers = []

    def _load_config(self):
        """Load configuration from multiple sources"""
        # Priority: Environment > Config File > Defaults

        config = self._get_default_config()

        # Load from file
        file_config = self._load_from_file()
        if file_config:
            config.update(file_config)

        # Override with environment variables
        env_config = self._load_from_environment()
        if env_config:
            config.update(env_config)

        # Validate configuration
        self._validate_config(config)

        return config

    def _load_from_file(self):
        """Load configuration from JSON file"""
        config_paths = [
            self.config_file,
            os.path.expanduser('~/.nhem/client_config.json'),
            '/etc/nhem/client_config.json'
        ]

        for path in config_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    warnings.warn(f"Failed to load config from {path}: {e}")

        return None

    def _load_from_environment(self):
        """Load configuration from environment variables"""
        env_config = {}

        # Map environment variables to config keys
        env_mappings = {
            'NHEM_BASE_URL': ['server', 'base_url'],
            'NHEM_PASSWORD': ['server', 'password'],
            'NHEM_UPDATE_INTERVAL': ['display', 'update_interval'],
            'NHEM_BUFFER_SIZE': ['display', 'buffer_size'],
            'NHEM_LOG_LEVEL': ['logging', 'level']
        }

        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                self._set_nested_config(env_config, config_path, value)

        return env_config
```

#### Configuration Structure
```json
{
  "server": {
    "base_url": "http://localhost:8421",
    "password": "31415",
    "timeout": 30,
    "retry_attempts": 3
  },
  "display": {
    "update_interval": 0.05,
    "buffer_size": 100,
    "max_framerate": 20,
    "chart_width": 10,
    "chart_height": 6,
    "theme": "dark"
  },
  "roi": {
    "default_x1": 1480,
    "default_y1": 480,
    "default_x2": 1580,
    "default_y2": 580
  },
  "peak_detection": {
    "show_peaks": true,
    "peak_threshold": 105.0,
    "color_peaks": true
  },
  "export": {
    "default_format": "csv",
    "include_metadata": true,
    "compression": false
  },
  "logging": {
    "level": "INFO",
    "file": "nhem_client.log",
    "max_size": "10MB",
    "backup_count": 5
  }
}
```

### 2. API Client Architecture

#### APIClient Class
```python
class APIClient:
    """HTTP API client with retry logic and error handling"""

    def __init__(self, config):
        self.base_url = config['server']['base_url']
        self.password = config['server']['password']
        self.timeout = config['server']['timeout']
        self.retry_attempts = config['server']['retry_attempts']
        self.session = requests.Session()

        # Configure session
        self.session.headers.update({
            'User-Agent': f'NHEM-Python-Client/{VERSION}',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })

    async def get_realtime_data(self, count=100):
        """Get real-time data with retry logic"""
        endpoint = f"{self.base_url}/data/realtime"
        params = {'count': count}

        for attempt in range(self.retry_attempts):
            try:
                response = self.session.get(
                    endpoint,
                    params=params,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()

            except requests.exceptions.RequestException as e:
                if attempt == self.retry_attempts - 1:
                    raise APIError(f"Failed to get realtime data: {e}")

                await asyncio.sleep(2 ** attempt)  # Exponential backoff

    async def send_control_command(self, command):
        """Send control command to server"""
        endpoint = f"{self.base_url}/control"

        data = {
            'command': command,
            'password': self.password
        }

        try:
            response = self.session.post(
                endpoint,
                data=data,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            raise APIError(f"Failed to send control command: {e}")

    def get_system_status(self):
        """Get system status information"""
        endpoint = f"{self.base_url}/status"

        try:
            response = self.session.get(endpoint, timeout=self.timeout)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            raise APIError(f"Failed to get system status: {e}")
```

### 3. Real-time Plotting System

#### RealtimePlotter Class
```python
class RealtimePlotter:
    """High-performance real-time plotting with matplotlib"""

    def __init__(self, parent_frame, config):
        self.config = config
        self.parent_frame = parent_frame

        # Data buffers
        self.time_buffer = deque(maxlen=config['display']['buffer_size'])
        self.value_buffer = deque(maxlen=config['display']['buffer_size'])
        self.roi_buffer = deque(maxlen=config['display']['buffer_size'])
        self.peak_buffer = deque(maxlen=config['display']['buffer_size'])

        # Setup matplotlib
        self.setup_matplotlib()
        self.setup_plots()

        # Animation
        self.animation = None
        self.is_running = False

    def setup_matplotlib(self):
        """Configure matplotlib for optimal performance"""
        # Use TkAgg backend for better performance
        matplotlib.use('TkAgg')

        # Configure figure and axes
        self.figure = plt.figure(
            figsize=self.config['display']['chart_size'],
            facecolor='#1e1e1e'  # Dark theme
        )

        # Main signal plot
        self.ax_main = self.figure.add_subplot(211)
        self.ax_main.set_facecolor('#252526')
        self.ax_main.set_title('Real-time Signal', color='#d4d4d4')
        self.ax_main.set_ylabel('Value', color='#d4d4d4')
        self.ax_main.grid(True, alpha=0.2, color='#3e3e42')

        # ROI plot
        self.ax_roi = self.figure.add_subplot(212)
        self.ax_roi.set_facecolor('#252526')
        self.ax_roi.set_title('ROI Gray Value', color='#d4d4d4')
        self.ax_roi.set_xlabel('Time (s)', color='#d4d4d4')
        self.ax_roi.set_ylabel('Gray Value', color='#d4d4d4')
        self.ax_roi.grid(True, alpha=0.2, color='#3e3e42')

        # Style configuration
        self.figure.tight_layout()

    def setup_plots(self):
        """Initialize plot lines and artists"""
        # Main signal line
        self.line_main, = self.ax_main.plot([], [], 'c-', linewidth=1.5, label='Signal')
        self.line_roi, = self.ax_roi.plot([], [], 'g-', linewidth=1.5, label='ROI')

        # Peak markers
        self.peaks_main = self.ax_main.scatter([], [], c='red', s=50, zorder=5, label='Peaks')
        self.peaks_roi = self.ax_roi.scatter([], [], c='orange', s=30, zorder=5)

        # ROI indicator
        self.roi_indicator = None

        # Legends
        self.ax_main.legend(loc='upper right')
        self.ax_roi.legend(loc='upper right')

    def start_animation(self):
        """Start real-time animation"""
        if self.is_running:
            return

        self.is_running = True

        # Create animation
        self.animation = FuncAnimation(
            self.figure,
            self.update_plot,
            interval=int(self.config['display']['update_interval'] * 1000),
            blit=True,
            cache_frame_data=False
        )

        # Embed in Tkinter
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.parent_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, self.parent_frame)
        toolbar.update()

    def update_plot(self, frame):
        """Animation update function"""
        if not self.time_buffer:
            return self.line_main, self.line_roi, self.peaks_main, self.peaks_roi

        # Convert to numpy arrays for better performance
        time_array = np.array(self.time_buffer)
        value_array = np.array(self.value_buffer)
        roi_array = np.array(self.roi_buffer)

        # Update main signal plot
        self.line_main.set_data(time_array, value_array)

        # Auto-scale axes
        if len(time_array) > 1:
            time_min, time_max = time_array.min(), time_array.max()
            value_min, value_max = value_array.min(), value_array.max()

            # Add padding
            time_padding = (time_max - time_min) * 0.05
            value_padding = (value_max - value_min) * 0.1

            self.ax_main.set_xlim(time_min - time_padding, time_max + time_padding)
            self.ax_main.set_ylim(value_min - value_padding, value_max + value_padding)

        # Update ROI plot
        if np.any(roi_array > 0):  # Only update if ROI data available
            self.line_roi.set_data(time_array, roi_array)

            roi_min, roi_max = roi_array.min(), roi_array.max()
            roi_padding = (roi_max - roi_min) * 0.1
            self.ax_roi.set_ylim(roi_min - roi_padding, roi_max + roi_padding)

        # Update peak markers
        peak_times, peak_values = self.get_peak_coordinates()
        if peak_times:
            self.peaks_main.set_offsets(np.c_[peak_times, peak_values])

        return self.line_main, self.line_roi, self.peaks_main, self.peaks_roi

    def add_data_point(self, timestamp, value, roi_value=None, peak_info=None):
        """Add new data point to buffers"""
        # Convert timestamp to relative time
        if not self.time_buffer:
            self.start_time = timestamp

        relative_time = (timestamp - self.start_time).total_seconds()

        # Add to buffers
        self.time_buffer.append(relative_time)
        self.value_buffer.append(value)

        if roi_value is not None:
            self.roi_buffer.append(roi_value)

        # Process peak information
        if peak_info and peak_info.get('signal') == 1:
            self.peak_buffer.append({
                'time': relative_time,
                'value': value,
                'color': peak_info.get('color', 'red')
            })

    def get_peak_coordinates(self):
        """Get peak coordinates for visualization"""
        if not self.peak_buffer:
            return [], []

        peaks = list(self.peak_buffer)
        times = [p['time'] for p in peaks]
        values = [p['value'] for p in peaks]

        return times, values
```

### 4. GUI Architecture

#### MainWindow Class
```python
class MainWindow:
    """Main GUI window with layout and event handling"""

    def __init__(self, api_client, config):
        self.api_client = api_client
        self.config = config
        self.root = tk.Tk()

        # State management
        self.is_connected = False
        self.is_monitoring = False
        self.current_data = None

        # GUI components
        self.setup_window()
        self.create_menu()
        self.create_toolbar()
        self.create_main_area()
        self.create_status_bar()

        # Event bindings
        self.setup_event_handlers()

    def setup_window(self):
        """Configure main window properties"""
        self.root.title("NHEM Real-time Client")
        self.root.geometry("1200x800")
        self.root.configure(bg='#1e1e1e')

        # Window icon and styling
        self.setup_styles()

        # Center window on screen
        self.center_window()

    def create_menu(self):
        """Create application menu bar"""
        menubar = tk.Menu(self.root, bg='#2d2d30', fg='#cccccc')

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0, bg='#2d2d30', fg='#cccccc')
        file_menu.add_command(label="Export Data", command=self.export_data)
        file_menu.add_command(label="Export Chart", command=self.export_chart)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit_application)
        menubar.add_cascade(label="File", menu=file_menu)

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0, bg='#2d2d30', fg='#cccccc')
        view_menu.add_checkbutton(label="Show Grid", command=self.toggle_grid)
        view_menu.add_checkbutton(label="Show Peaks", command=self.toggle_peaks)
        view_menu.add_separator()
        view_menu.add_command(label="Settings", command=self.show_settings)
        menubar.add_cascade(label="View", menu=view_menu)

        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0, bg='#2d2d30', fg='#cccccc')
        tools_menu.add_command(label="ROI Configuration", command=self.show_roi_config)
        tools_menu.add_command(label="Peak Detection Settings", command=self.show_peak_settings)
        tools_menu.add_separator()
        tools_menu.add_command(label="System Diagnostics", command=self.show_diagnostics)
        menubar.add_cascade(label="Tools", menu=tools_menu)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0, bg='#2d2d30', fg='#cccccc')
        help_menu.add_command(label="Documentation", command=self.show_documentation)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menubar)

    def create_toolbar(self):
        """Create application toolbar"""
        toolbar = ttk.Frame(self.root, style='Toolbar.TFrame')
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)

        # Control buttons
        self.btn_connect = ttk.Button(
            toolbar, text="Connect", command=self.toggle_connection,
            style='Success.TButton'
        )
        self.btn_connect.pack(side=tk.LEFT, padx=2)

        self.btn_start = ttk.Button(
            toolbar, text="Start Detection", command=self.start_detection,
            style='Primary.TButton', state='disabled'
        )
        self.btn_start.pack(side=tk.LEFT, padx=2)

        self.btn_stop = ttk.Button(
            toolbar, text="Stop Detection", command=self.stop_detection,
            style='Danger.TButton', state='disabled'
        )
        self.btn_stop.pack(side=tk.LEFT, padx=2)

        self.btn_pause = ttk.Button(
            toolbar, text="Pause", command=self.pause_detection,
            style='Warning.TButton', state='disabled'
        )
        self.btn_pause.pack(side=tk.LEFT, padx=2)

        # Separator
        ttk.Separator(toolbar, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=5)

        # Export buttons
        ttk.Button(toolbar, text="Export Data", command=self.export_data).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Export Chart", command=self.export_chart).pack(side=tk.LEFT, padx=2)

        # Status indicator
        self.status_label = ttk.Label(toolbar, text="Disconnected", style='Status.TLabel')
        self.status_label.pack(side=tk.RIGHT, padx=10)

    def create_main_area(self):
        """Create main application area"""
        # Create paned window for resizable layout
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel - Charts
        left_frame = ttk.Frame(main_paned, style='Panel.TFrame')
        main_paned.add(left_frame, weight=3)

        # Chart area
        self.chart_frame = ttk.Frame(left_frame, style='Chart.TFrame')
        self.chart_frame.pack(fill=tk.BOTH, expand=True)

        # Right panel - Information and controls
        right_frame = ttk.Frame(main_paned, style='Panel.TFrame')
        main_paned.add(right_frame, weight=1)

        # Create right panel content
        self.create_info_panel(right_frame)
        self.create_control_panel(right_frame)

    def create_info_panel(self, parent):
        """Create information display panel"""
        info_frame = ttk.LabelFrame(parent, text="System Information", style='Info.TLabelframe')
        info_frame.pack(fill=tk.X, padx=5, pady=5)

        # Status display
        status_frame = ttk.Frame(info_frame)
        status_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(status_frame, text="Status:").pack(side=tk.LEFT)
        self.status_display = ttk.Label(status_frame, text="Disconnected", style='Status.TLabel')
        self.status_display.pack(side=tk.LEFT, padx=(10, 0))

        # Frame count
        frame_frame = ttk.Frame(info_frame)
        frame_frame.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(frame_frame, text="Frame Count:").pack(side=tk.LEFT)
        self.frame_count_display = ttk.Label(frame_frame, text="0")
        self.frame_count_display.pack(side=tk.LEFT, padx=(10, 0))

        # Current value
        value_frame = ttk.Frame(info_frame)
        value_frame.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(value_frame, text="Current Value:").pack(side=tk.LEFT)
        self.current_value_display = ttk.Label(value_frame, text="0.00")
        self.current_value_display.pack(side=tk.LEFT, padx=(10, 0))

        # Peak signal
        peak_frame = ttk.Frame(info_frame)
        peak_frame.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(peak_frame, text="Peak Signal:").pack(side=tk.LEFT)
        self.peak_signal_display = ttk.Label(peak_frame, text="None")
        self.peak_signal_display.pack(side=tk.LEFT, padx=(10, 0))

        # Baseline
        baseline_frame = ttk.Frame(info_frame)
        baseline_frame.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(baseline_frame, text="Baseline:").pack(side=tk.LEFT)
        self.baseline_display = ttk.Label(baseline_frame, text="0.00")
        self.baseline_display.pack(side=tk.LEFT, padx=(10, 0))

    def create_control_panel(self, parent):
        """Create control panel"""
        control_frame = ttk.LabelFrame(parent, text="Controls", style='Control.TLabelframe')
        control_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Update interval
        interval_frame = ttk.Frame(control_frame)
        interval_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(interval_frame, text="Update Interval (ms):").pack(side=tk.LEFT)
        self.interval_var = tk.StringVar(value=str(int(self.config['display']['update_interval'] * 1000)))
        interval_spinbox = ttk.Spinbox(interval_frame, from_=10, to=1000, textvariable=self.interval_var, width=10)
        interval_spinbox.pack(side=tk.RIGHT)

        # Buffer size
        buffer_frame = ttk.Frame(control_frame)
        buffer_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(buffer_frame, text="Buffer Size:").pack(side=tk.LEFT)
        self.buffer_var = tk.StringVar(value=str(self.config['display']['buffer_size']))
        buffer_spinbox = ttk.Spinbox(buffer_frame, from_=10, to=1000, textvariable=self.buffer_var, width=10)
        buffer_spinbox.pack(side=tk.RIGHT)

        # Peak detection settings
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)

        peak_frame = ttk.LabelFrame(control_frame, text="Peak Detection", style='Peak.TLabelframe')
        peak_frame.pack(fill=tk.X, padx=5, pady=5)

        # Threshold
        threshold_frame = ttk.Frame(peak_frame)
        threshold_frame.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(threshold_frame, text="Threshold:").pack(side=tk.LEFT)
        self.threshold_var = tk.StringVar(value=str(self.config['peak_detection']['peak_threshold']))
        threshold_scale = ttk.Scale(threshold_frame, from_=50, to=200, variable=self.threshold_var, orient=tk.HORIZONTAL)
        threshold_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))

        # Auto-detected peaks
        self.auto_peak_var = tk.BooleanVar(value=self.config['peak_detection']['show_peaks'])
        ttk.Checkbutton(peak_frame, text="Show Auto-detected Peaks", variable=self.auto_peak_var).pack(pady=5)
```

## Data Processing Architecture

### 1. Real-time Data Updates

#### DataUpdater Class
```python
class DataUpdater:
    """Manages real-time data updates from the server"""

    def __init__(self, api_client, plotter, config):
        self.api_client = api_client
        self.plotter = plotter
        self.config = config

        # Update control
        self.is_running = False
        self.update_interval = config['display']['update_interval']
        self.last_update = 0

        # Performance monitoring
        self.update_times = deque(maxlen=100)
        self.error_count = 0

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
        self.info("Starting real-time updates")

        try:
            await self.update_loop()
        except Exception as e:
            self.error(f"Update loop error: {e}")
            self.trigger_callback('error', e)
        finally:
            self.is_running = False
            self.info("Real-time updates stopped")

    async def update_loop(self):
        """Main update loop"""
        consecutive_errors = 0
        max_consecutive_errors = 5

        while self.is_running:
            try:
                start_time = time.time()

                # Fetch data from server
                data = await self.api_client.get_realtime_data(
                    count=self.config['display']['buffer_size']
                )

                # Process data
                await self.process_data(data)

                # Update performance metrics
                update_time = time.time() - start_time
                self.update_times.append(update_time)

                # Reset error count on success
                consecutive_errors = 0

                # Trigger connection restored callback if needed
                if self.error_count > 0:
                    self.trigger_callback('connection_restored')

                self.error_count = 0

                # Calculate sleep time to maintain update interval
                elapsed = time.time() - start_time
                sleep_time = max(0, self.update_interval - elapsed)

                await asyncio.sleep(sleep_time)

            except Exception as e:
                consecutive_errors += 1
                self.error_count += 1

                self.error(f"Update error (attempt {consecutive_errors}/{max_consecutive_errors}): {e}")
                self.trigger_callback('error', e)

                # Exponential backoff on consecutive errors
                if consecutive_errors >= max_consecutive_errors:
                    self.trigger_callback('connection_lost')
                    await asyncio.sleep(5)  # Wait before retrying
                    consecutive_errors = 0
                else:
                    await asyncio.sleep(min(2 ** consecutive_errors, 10))  # Exponential backoff

    async def process_data(self, data):
        """Process received data and update visualizations"""
        try:
            # Validate data structure
            if not self.validate_data(data):
                raise ValueError("Invalid data structure")

            # Extract time series data
            series = data.get('series', [])
            if not series:
                self.warning("No time series data received")
                return

            # Process each data point
            for point in series:
                timestamp = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
                value = point['value']

                # Get ROI data if available
                roi_value = None
                roi_data = data.get('roi_data')
                if roi_data and roi_data.get('gray_value'):
                    roi_value = roi_data['gray_value']

                # Get peak information
                peak_info = data.get('enhanced_peak')

                # Update plotter
                self.plotter.add_data_point(timestamp, value, roi_value, peak_info)

            # Trigger data received callback
            self.trigger_callback('data_received', data)

        except Exception as e:
            self.error(f"Data processing error: {e}")
            raise

    def validate_data(self, data):
        """Validate received data structure"""
        required_fields = ['type', 'timestamp', 'frame_count', 'series']

        for field in required_fields:
            if field not in data:
                self.error(f"Missing required field: {field}")
                return False

        # Validate series data
        series = data.get('series', [])
        if not isinstance(series, list):
            self.error("Series must be a list")
            return False

        for point in series:
            if not isinstance(point, dict) or 't' not in point or 'value' not in point:
                self.error("Invalid series point format")
                return False

        return True

    def add_callback(self, event, callback):
        """Add event callback"""
        if event in self.callbacks:
            self.callbacks[event].append(callback)

    def trigger_callback(self, event, data=None):
        """Trigger event callback"""
        if event in self.callbacks:
            for callback in self.callbacks[event]:
                try:
                    callback(data)
                except Exception as e:
                    self.error(f"Callback error for {event}: {e}")
```

### 2. Performance Monitoring

#### PerformanceMonitor Class
```python
class PerformanceMonitor:
    """Monitor system and application performance"""

    def __init__(self):
        self.metrics = {
            'update_times': deque(maxlen=1000),
            'memory_usage': deque(maxlen=1000),
            'cpu_usage': deque(maxlen=1000),
            'network_latency': deque(maxlen=1000),
            'error_count': 0,
            'total_updates': 0
        }

        self.start_time = time.time()
        self.last_report = time.time()

    def record_update_time(self, update_time):
        """Record data update time"""
        self.metrics['update_times'].append(update_time)
        self.metrics['total_updates'] += 1

    def record_error(self):
        """Record error occurrence"""
        self.metrics['error_count'] += 1

    def get_performance_report(self):
        """Generate performance report"""
        current_time = time.time()
        uptime = current_time - self.start_time

        # Calculate statistics
        update_times = list(self.metrics['update_times'])

        report = {
            'uptime': uptime,
            'total_updates': self.metrics['total_updates'],
            'error_count': self.metrics['error_count'],
            'error_rate': self.metrics['error_count'] / max(self.metrics['total_updates'], 1),
        }

        if update_times:
            report['update_performance'] = {
                'avg_time': np.mean(update_times),
                'min_time': np.min(update_times),
                'max_time': np.max(update_times),
                'std_time': np.std(update_times),
                'median_time': np.median(update_times),
                'p95_time': np.percentile(update_times, 95),
                'fps': 1.0 / np.mean(update_times)
            }

        # System metrics
        if psutil:
            process = psutil.Process()

            # Memory usage
            memory_info = process.memory_info()
            report['memory'] = {
                'rss': memory_info.rss,
                'vms': memory_info.vms,
                'percent': process.memory_percent()
            }

            # CPU usage
            report['cpu'] = {
                'percent': process.cpu_percent()
            }

        return report

    def log_performance_report(self):
        """Log performance report"""
        report = self.get_performance_report()

        logger.info("Performance Report:")
        logger.info(f"  Uptime: {report['uptime']:.1f}s")
        logger.info(f"  Total Updates: {report['total_updates']}")
        logger.info(f"  Error Rate: {report['error_rate']:.2%}")

        if 'update_performance' in report:
            perf = report['update_performance']
            logger.info(f"  Update FPS: {perf['fps']:.1f}")
            logger.info(f"  Avg Update Time: {perf['avg_time']*1000:.1f}ms")
            logger.info(f"  P95 Update Time: {perf['p95_time']*1000:.1f}ms")

        if 'memory' in report:
            mem = report['memory']
            logger.info(f"  Memory Usage: {mem['rss']/1024/1024:.1f}MB ({mem['percent']:.1f}%)")
```

## Testing Framework

### 1. Unit Tests

#### Test Structure
```python
import unittest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

class TestAPIClient(unittest.TestCase):
    """Test API client functionality"""

    def setUp(self):
        self.config = {
            'server': {
                'base_url': 'http://localhost:8421',
                'password': 'test_password',
                'timeout': 5,
                'retry_attempts': 2
            }
        }
        self.client = APIClient(self.config)

    @patch('requests.Session.get')
    def test_get_realtime_data_success(self, mock_get):
        """Test successful realtime data retrieval"""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            'type': 'realtime_data',
            'series': [{'t': 0.0, 'value': 100.0}],
            'frame_count': 1
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Test
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(self.client.get_realtime_data())
        loop.close()

        # Assertions
        self.assertEqual(result['type'], 'realtime_data')
        self.assertEqual(len(result['series']), 1)
        self.assertEqual(result['series'][0]['value'], 100.0)

    @patch('requests.Session.get')
    def test_get_realtime_data_retry(self, mock_get):
        """Test retry logic on failure"""
        # Mock failure then success
        mock_response_fail = Mock()
        mock_response_fail.raise_for_status.side_effect = Exception("Network error")

        mock_response_success = Mock()
        mock_response_success.json.return_value = {'type': 'test'}
        mock_response_success.raise_for_status.return_value = None

        mock_get.side_effect = [mock_response_fail, mock_response_success]

        # Test
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(self.client.get_realtime_data())
        loop.close()

        # Assertions
        self.assertEqual(mock_get.call_count, 2)  # Should retry once
        self.assertEqual(result['type'], 'test')

class TestRealtimePlotter(unittest.TestCase):
    """Test real-time plotting functionality"""

    def setUp(self):
        self.config = {
            'display': {
                'buffer_size': 10,
                'update_interval': 0.05,
                'chart_size': (10, 6)
            }
        }

        # Create mock Tkinter root
        self.root = Mock()
        self.parent_frame = Mock()

    def test_buffer_management(self):
        """Test data buffer management"""
        # Create plotter
        plotter = RealtimePlotter(self.parent_frame, self.config)

        # Add more data than buffer size
        for i in range(15):
            timestamp = datetime.now() + timedelta(seconds=i)
            plotter.add_data_point(timestamp, float(i))

        # Assertions
        self.assertEqual(len(plotter.time_buffer), 10)  # Should maintain buffer size
        self.assertEqual(len(plotter.value_buffer), 10)

        # Should keep most recent data
        self.assertEqual(plotter.value_buffer[-1], 14.0)
```

### 2. Integration Tests

#### TestClientIntegration
```python
class TestClientIntegration(unittest.TestCase):
    """Integration tests for complete client functionality"""

    @classmethod
    def setUpClass(cls):
        """Setup test environment"""
        # Start test server
        cls.server_thread = threading.Thread(target=cls.start_test_server)
        cls.server_thread.daemon = True
        cls.server_thread.start()

        # Wait for server to start
        time.sleep(1)

    def test_full_client_workflow(self):
        """Test complete client workflow"""
        # Create client
        config = {
            'server': {
                'base_url': 'http://localhost:8421',
                'password': 'test_password'
            },
            'display': {
                'update_interval': 0.1,
                'buffer_size': 50
            }
        }

        client = NHEMRealtimeClient()
        client.config = config
        client.api_client = APIClient(config)

        # Test connection
        status = client.api_client.get_system_status()
        self.assertIn('status', status)

        # Test data retrieval
        data = asyncio.run(client.api_client.get_realtime_data(10))
        self.assertIn('series', data)
        self.assertLessEqual(len(data['series']), 10)

    def test_error_handling(self):
        """Test error handling and recovery"""
        # Test with invalid server
        config = {
            'server': {
                'base_url': 'http://invalid-server:8421',
                'password': 'test_password',
                'timeout': 1,
                'retry_attempts': 2
            }
        }

        client = APIClient(config)

        # Should raise APIError after retries
        with self.assertRaises(APIError):
            asyncio.run(client.get_realtime_data())
```

## Deployment and Distribution

### 1. Package Structure

#### setup.py
```python
from setuptools import setup, find_packages

setup(
    name="nhem-python-client",
    version="1.0.0",
    description="NHEM Python Client for Real-time Monitoring",
    author="NHEM Development Team",
    author_email="dev@nhem.org",
    url="https://github.com/nhem-org/python-client",

    packages=find_packages(),
    py_modules=[
        'run_realtime_client',
        'http_realtime_client',
        'simple_http_client',
        'realtime_plotter',
        'client',
        'local_config_loader'
    ],

    install_requires=[
        'requests>=2.25.0',
        'matplotlib>=3.3.0',
        'pillow>=8.0.0',
        'numpy>=1.20.0',
        'psutil>=5.8.0'  # Optional for performance monitoring
    ],

    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'pytest-asyncio>=0.18.0',
            'pytest-cov>=2.12.0',
            'black>=21.0.0',
            'flake8>=3.9.0'
        ],
        'docs': [
            'sphinx>=4.0.0',
            'sphinx-rtd-theme>=0.5.0'
        ]
    },

    entry_points={
        'console_scripts': [
            'nhem-gui=run_realtime_client:main',
            'nhem-simple=simple_http_client:main',
            'nhem-cli=client:main'
        ]
    },

    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],

    python_requires=">=3.8"
)
```

### 2. Installation Options

#### Development Installation
```bash
# Clone repository
git clone https://github.com/nhem-org/python-client.git
cd python-client

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Install development dependencies
pip install -e .[dev]
```

#### Production Installation
```bash
# Install from PyPI (when published)
pip install nhem-python-client

# Or install from repository
pip install git+https://github.com/nhem-org/python-client.git
```

### 3. Docker Support

#### Dockerfile
```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tk-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install application
RUN pip install -e .

# Create non-root user
RUN useradd -m -u 1000 nhem
USER nhem

# Expose application (if running web version)
EXPOSE 3000

# Default command
CMD ["python", "run_realtime_client.py"]
```

## Security Considerations

### 1. Authentication and Authorization

```python
class SecurityManager:
    """Handle security-related operations"""

    def __init__(self, config):
        self.config = config
        self.password = config.get('server', {}).get('password')

    def validate_password(self, provided_password):
        """Validate provided password"""
        if not provided_password:
            return False

        # Use constant-time comparison to prevent timing attacks
        return secrets.compare_digest(provided_password.encode(), self.password.encode())

    def encrypt_sensitive_data(self, data):
        """Encrypt sensitive configuration data"""
        # Simple encryption for configuration files
        # In production, use proper key management
        key = self.get_encryption_key()

        f = Fernet(key)
        encrypted_data = f.encrypt(json.dumps(data).encode())

        return encrypted_data.decode()

    def get_encryption_key(self):
        """Get or create encryption key"""
        key_file = os.path.expanduser('~/.nhem/encryption.key')

        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            # Generate new key
            key = Fernet.generate_key()

            # Ensure directory exists
            os.makedirs(os.path.dirname(key_file), exist_ok=True)

            # Save key with restricted permissions
            with open(key_file, 'wb') as f:
                f.write(key)

            os.chmod(key_file, 0o600)  # Read/write for owner only

            return key
```

### 2. Input Validation

```python
class InputValidator:
    """Validate user inputs and API responses"""

    @staticmethod
    def validate_server_url(url):
        """Validate server URL"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    @staticmethod
    def validate_numeric_input(value, min_val=None, max_val=None):
        """Validate numeric input"""
        try:
            num_value = float(value)

            if min_val is not None and num_value < min_val:
                return False

            if max_val is not None and num_value > max_val:
                return False

            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    def sanitize_config_data(config):
        """Sanitize configuration data"""
        # Remove potentially dangerous entries
        dangerous_keys = ['__import__', 'eval', 'exec', 'open']

        for key in dangerous_keys:
            if key in config:
                del config[key]

        # Validate nested structures
        if isinstance(config, dict):
            for key, value in config.items():
                if isinstance(value, dict):
                    config[key] = InputValidator.sanitize_config_data(value)

        return config
```

This Python client architecture provides a comprehensive, modular, and secure foundation for real-time monitoring applications. The design emphasizes performance, usability, and maintainability while providing multiple deployment options for different use cases.