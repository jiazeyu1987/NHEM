# Client-Server Interaction Documentation

## Overview

The NHEM system supports multiple client types interacting with a central backend server through well-defined protocols and APIs. This documentation covers all interaction patterns, communication protocols, and integration methods between clients and the server.

## Communication Protocols

### 1. HTTP REST API

#### Protocol Characteristics
- **Protocol**: HTTP/1.1
- **Data Format**: JSON
- **Content-Type**: `application/json` (most endpoints), `application/x-www-form-urlencoded` (control endpoints)
- **Authentication**: Password-based for protected operations
- **Rate Limiting**: Built-in for control commands

#### Base URL Structure
```
Production: http://localhost:8421
Development: http://localhost:8421
Custom: Configurable via environment variables (NHEM_BASE_URL)
```

#### Request/Response Pattern
```javascript
// Standard API Request Pattern
GET /endpoint?parameter=value

// Response Structure
{
  "type": "response_type",
  "timestamp": "2025-12-05T12:30:45.123Z",
  "success": true,
  "data": { /* response data */ }
}

// Error Response Structure
{
  "type": "error",
  "timestamp": "2025-12-05T12:30:45.123Z",
  "error_code": "ERROR_CODE",
  "error_message": "Human readable error message",
  "details": {
    "parameter": "field_name",
    "value": "provided_value",
    "constraint": "validation_rule"
  }
}
```

### 2. WebSocket Protocol

#### Protocol Characteristics
- **Protocol**: WebSocket (RFC 6455)
- **Port**: 30415 (configurable via NHEM_SOCKET_PORT)
- **Message Format**: JSON
- **Connection**: Persistent, bidirectional
- **Authentication**: Password required in initial message
- **Broadcasting**: Server-to-client message distribution

#### WebSocket Message Flow
```javascript
// Client Authentication Message
{
  "type": "auth",
  "password": "31415"
}

// Server Response
{
  "type": "auth_response",
  "success": true,
  "message": "Authentication successful"
}

// Data Broadcast Message (60 FPS)
{
  "type": "data",
  "timestamp": "2025-12-05T12:30:45.123Z",
  "frame_count": 1250,
  "current_value": 127.45,
  "peak_signal": 1,
  "baseline": 120.0,
  "status": "running"
}
```

## Client Types and Integration Methods

### 1. Web Frontend (Vanilla JavaScript)

#### Architecture Pattern
```javascript
// Frontend API Client Implementation
class NHEMFrontendAPI {
    constructor(config) {
        this.baseURL = config.server.base_url;
        this.password = config.server.password;
        this.timeout = config.server.timeout || 5000;
    }

    async getRealtimeData(count = 100) {
        const url = `${this.baseURL}/data/realtime?count=${count}`;

        try {
            const response = await fetch(url, {
                method: 'GET',
                headers: {
                    'Accept': 'application/json'
                },
                signal: AbortSignal.timeout(this.timeout)
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Failed to fetch realtime data:', error);
            throw error;
        }
    }

    async sendControlCommand(command) {
        const formData = new FormData();
        formData.append('command', command);
        formData.append('password', this.password);

        try {
            const response = await fetch(`${this.baseURL}/control`, {
                method: 'POST',
                body: formData,
                signal: AbortSignal.timeout(this.timeout)
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Failed to send control command:', error);
            throw error;
        }
    }
}
```

#### Real-time Data Synchronization
```javascript
// Real-time Update Manager
class RealtimeUpdateManager {
    constructor(apiClient, updateCallback) {
        this.api = apiClient;
        this.updateCallback = updateCallback;
        this.updateInterval = 50; // 20 FPS (50ms)
        this.isRunning = false;
        this.lastUpdate = 0;
    }

    async start() {
        if (this.isRunning) return;

        this.isRunning = true;
        console.log('Starting real-time updates at 20 FPS');

        while (this.isRunning) {
            const startTime = performance.now();

            try {
                const data = await this.api.getRealtimeData();
                this.processData(data);
                this.updateCallback(data);

            } catch (error) {
                console.error('Update failed:', error);
                this.handleUpdateError(error);
            }

            // Maintain 20 FPS timing
            const elapsed = performance.now() - startTime;
            const sleepTime = Math.max(0, this.updateInterval - elapsed);

            await this.sleep(sleepTime);
        }
    }

    processData(data) {
        // Validate data structure
        if (!data.series || !Array.isArray(data.series)) {
            throw new Error('Invalid data structure received');
        }

        // Update charts with new data
        this.updateCharts(data);

        // Handle ROI data if available
        if (data.roi_data && data.roi_data.gray_value) {
            this.updateROIDisplay(data.roi_data);
        }

        // Handle peak detection results
        if (data.peak_signal !== null) {
            this.handlePeakDetection(data.peak_signal);
        }
    }

    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}
```

#### Error Handling and Recovery
```javascript
// Frontend Error Handler
class FrontendErrorHandler {
    constructor(updateManager) {
        this.updateManager = updateManager;
        this.consecutiveErrors = 0;
        this.maxConsecutiveErrors = 5;
        this.retryDelay = 1000; // Start with 1 second
        this.maxRetryDelay = 30000; // Max 30 seconds
    }

    handleUpdateError(error) {
        this.consecutiveErrors++;

        console.error(`Update error ${this.consecutiveErrors}/${this.maxConsecutiveErrors}:`, error);

        if (this.consecutiveErrors >= this.maxConsecutiveErrors) {
            this.handleConnectionLost();
        } else {
            // Exponential backoff
            const delay = Math.min(this.retryDelay * Math.pow(2, this.consecutiveErrors - 1), this.maxRetryDelay);
            this.scheduleRetry(delay);
        }
    }

    handleConnectionLost() {
        console.error('Connection lost, stopping updates');
        this.updateManager.stop();
        this.showConnectionError();
        this.scheduleReconnection();
    }

    scheduleReconnection() {
        console.log('Attempting to reconnect in 10 seconds...');
        setTimeout(() => {
            this.consecutiveErrors = 0;
            this.retryDelay = 1000; // Reset retry delay
            this.updateManager.start().catch(error => {
                console.error('Reconnection failed:', error);
                this.scheduleReconnection();
            });
        }, 10000);
    }

    showConnectionError() {
        const errorElement = document.getElementById('connection-error');
        if (errorElement) {
            errorElement.style.display = 'block';
            errorElement.textContent = '连接丢失，尝试重新连接中...';
        }
    }
}
```

### 2. Python GUI Client (Matplotlib + Tkinter)

#### API Client Implementation
```python
import asyncio
import aiohttp
import json
from typing import Dict, Any, Optional

class PythonAPIClient:
    """Async HTTP API client for Python applications"""

    def __init__(self, config: Dict[str, Any]):
        self.base_url = config['server']['base_url']
        self.password = config['server']['password']
        self.timeout = aiohttp.ClientTimeout(total=config['server']['timeout'])
        self.retry_attempts = config['server']['retry_attempts']

        # Session configuration
        self.headers = {
            'User-Agent': f'NHEM-Python-Client/{VERSION}',
            'Accept': 'application/json'
        }

    async def get_realtime_data(self, count: int = 100) -> Dict[str, Any]:
        """Get real-time data with retry logic"""
        url = f"{self.base_url}/data/realtime"
        params = {'count': count}

        for attempt in range(self.retry_attempts):
            try:
                async with aiohttp.ClientSession(timeout=self.timeout, headers=self.headers) as session:
                    async with session.get(url, params=params) as response:
                        response.raise_for_status()
                        return await response.json()

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt == self.retry_attempts - 1:
                    raise APIError(f"Failed to get realtime data after {self.retry_attempts} attempts: {e}")

                # Exponential backoff
                await asyncio.sleep(2 ** attempt)

    async def send_control_command(self, command: str) -> Dict[str, Any]:
        """Send control command to server"""
        url = f"{self.base_url}/control"

        data = aiohttp.FormData()
        data.add_field('command', command)
        data.add_field('password', self.password)

        try:
            async with aiohttp.ClientSession(timeout=self.timeout, headers=self.headers) as session:
                async with session.post(url, data=data) as response:
                    response.raise_for_status()
                    return await response.json()

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            raise APIError(f"Failed to send control command '{command}': {e}")

    async def get_system_status(self) -> Dict[str, Any]:
        """Get system status information"""
        url = f"{self.base_url}/status"

        try:
            async with aiohttp.ClientSession(timeout=self.timeout, headers=self.headers) as session:
                async with session.get(url) as response:
                    response.raise_for_status()
                    return await response.json()

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            raise APIError(f"Failed to get system status: {e}")
```

#### Real-time Data Updates
```python
import asyncio
import threading
from datetime import datetime
from collections import deque

class RealtimeDataManager:
    """Manage real-time data updates for Python GUI"""

    def __init__(self, api_client, plotter, config):
        self.api_client = api_client
        self.plotter = plotter
        self.config = config

        # Data buffers
        self.time_buffer = deque(maxlen=config['display']['buffer_size'])
        self.value_buffer = deque(maxlen=config['display']['buffer_size'])
        self.roi_buffer = deque(maxlen=config['display']['buffer_size'])
        self.peak_buffer = deque(maxlen=config['display']['buffer_size'])

        # Update control
        self.is_running = False
        self.update_interval = config['display']['update_interval']
        self.start_time = None

        # Performance monitoring
        self.update_times = deque(maxlen=100)
        self.error_count = 0

    async def start_updates(self):
        """Start real-time data updates"""
        if self.is_running:
            return

        self.is_running = True
        self.start_time = datetime.now()
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
                    count=self.config['display']['buffer_size']
                )

                # Process and display data
                await self.process_data(data)

                # Performance monitoring
                update_time = asyncio.get_event_loop().time() - start_time
                self.update_times.append(update_time)

                # Reset error count on success
                consecutive_errors = 0
                self.error_count = 0

                # Calculate sleep time
                elapsed = asyncio.get_event_loop().time() - start_time
                sleep_time = max(0, self.update_interval - elapsed)

                await asyncio.sleep(sleep_time)

            except Exception as e:
                consecutive_errors += 1
                self.error_count += 1

                print(f"Update error (attempt {consecutive_errors}/{max_consecutive_errors}): {e}")

                if consecutive_errors >= max_consecutive_errors:
                    print("Maximum consecutive errors reached, stopping updates")
                    await self.handle_connection_lost()
                    consecutive_errors = 0
                else:
                    # Exponential backoff
                    backoff_time = min(2 ** consecutive_errors, 10)
                    await asyncio.sleep(backoff_time)

    async def process_data(self, data: Dict[str, Any]):
        """Process received data and update visualizations"""
        # Validate data structure
        if not self.validate_data(data):
            raise ValueError("Invalid data structure")

        # Extract time series data
        series = data.get('series', [])
        if not series:
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

            # Update plotter with new data
            self.plotter.add_data_point(timestamp, value, roi_value, peak_info)

        # Update GUI status
        self.update_status(data)

    def validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate received data structure"""
        required_fields = ['type', 'timestamp', 'frame_count', 'series']

        for field in required_fields:
            if field not in data:
                print(f"Missing required field: {field}")
                return False

        # Validate series data
        series = data.get('series', [])
        if not isinstance(series, list):
            print("Series must be a list")
            return False

        for point in series:
            if not isinstance(point, dict) or 't' not in point or 'value' not in point:
                print("Invalid series point format")
                return False

        return True

    async def handle_connection_lost(self):
        """Handle connection loss and recovery"""
        print("Connection lost, attempting recovery...")

        # Stop updates temporarily
        self.is_running = False

        # Wait before attempting reconnection
        await asyncio.sleep(10)

        # Restart updates
        self.is_running = True
        print("Attempting to restart updates...")
```

#### WebSocket Integration (Optional)
```python
import websockets
import json
import asyncio

class WebSocketClient:
    """WebSocket client for real-time data streaming"""

    def __init__(self, config, message_callback):
        self.config = config
        self.message_callback = message_callback
        self.websocket = None
        self.is_connected = False

    async def connect(self):
        """Connect to WebSocket server"""
        uri = f"ws://{self.config['server']['host']}:{self.config['server']['socket_port']}"

        try:
            self.websocket = await websockets.connect(uri)
            self.is_connected = True

            # Authenticate
            auth_message = {
                "type": "auth",
                "password": self.config['server']['password']
            }
            await self.websocket.send(json.dumps(auth_message))

            # Start message listener
            asyncio.create_task(self.listen_for_messages())

            print(f"Connected to WebSocket: {uri}")

        except Exception as e:
            print(f"WebSocket connection failed: {e}")
            self.is_connected = False
            raise

    async def listen_for_messages(self):
        """Listen for incoming WebSocket messages"""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self.message_callback(data)
                except json.JSONDecodeError as e:
                    print(f"Failed to parse WebSocket message: {e}")

        except websockets.exceptions.ConnectionClosed:
            print("WebSocket connection closed")
            self.is_connected = False
        except Exception as e:
            print(f"WebSocket error: {e}")
            self.is_connected = False

    async def disconnect(self):
        """Disconnect from WebSocket server"""
        if self.websocket:
            await self.websocket.close()
            self.is_connected = False
            print("Disconnected from WebSocket")
```

### 3. Command-Line Client

#### CLI Integration
```python
import argparse
import json
import sys
from typing import List, Dict, Any

class CommandLineInterface:
    """Command-line interface for NHEM API interaction"""

    def __init__(self):
        self.parser = self.setup_argument_parser()
        self.config = self.load_config()
        self.api_client = APIClient(self.config)

    def setup_argument_parser(self) -> argparse.ArgumentParser:
        """Setup command-line argument parser"""
        parser = argparse.ArgumentParser(
            description="NHEM Command-line Client",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  %(prog)s status                    # Get system status
  %(prog)s start                     # Start detection
  %(prog)s stop                      # Stop detection
  %(prog)s data --count 50           # Get 50 data points
  %(prog)s config --get              # Get configuration
  %(prog)s config --set threshold 120  # Set threshold to 120
            """
        )

        # Global options
        parser.add_argument('--server', help='Server URL',
                           default=self.get_default_server_url())
        parser.add_argument('--password', help='Server password',
                           default=self.get_default_password())
        parser.add_argument('--timeout', type=int, help='Request timeout (seconds)',
                           default=30)
        parser.add_argument('--format', choices=['json', 'table', 'csv'],
                           help='Output format', default='json')
        parser.add_argument('--verbose', '-v', action='store_true',
                           help='Verbose output')

        # Subcommands
        subparsers = parser.add_subparsers(dest='command', help='Available commands')

        # Status command
        status_parser = subparsers.add_parser('status', help='Get system status')

        # Control commands
        start_parser = subparsers.add_parser('start', help='Start detection')
        stop_parser = subparsers.add_parser('stop', help='Stop detection')
        pause_parser = subparsers.add_parser('pause', help='Pause detection')
        resume_parser = subparsers.add_parser('resume', help='Resume detection')

        # Data commands
        data_parser = subparsers.add_parser('data', help='Get real-time data')
        data_parser.add_argument('--count', type=int, default=100,
                               help='Number of data points to retrieve')
        data_parser.add_argument('--output', '-o', help='Output file')

        # Configuration commands
        config_parser = subparsers.add_parser('config', help='Configuration management')
        config_group = config_parser.add_mutually_exclusive_group(required=True)
        config_group.add_argument('--get', action='store_true', help='Get configuration')
        config_group.add_argument('--set', nargs=2, metavar=('KEY', 'VALUE'),
                                  help='Set configuration value')
        config_group.add_argument('--section', help='Get specific configuration section')

        # ROI commands
        roi_parser = subparsers.add_parser('roi', help='ROI configuration')
        roi_parser.add_argument('--get', action='store_true', help='Get ROI configuration')
        roi_parser.add_argument('--set', nargs=4, metavar=('X1', 'Y1', 'X2', 'Y2'),
                               type=int, help='Set ROI coordinates')

        return parser

    async def run(self, args: List[str] = None):
        """Run command-line interface"""
        args = self.parser.parse_args(args)

        if not args.command:
            self.parser.print_help()
            return 1

        try:
            # Update config with command-line arguments
            if args.server:
                self.config['server']['base_url'] = args.server
            if args.password:
                self.config['server']['password'] = args.password
            if args.timeout:
                self.config['server']['timeout'] = args.timeout

            # Create API client with updated config
            self.api_client = APIClient(self.config)

            # Execute command
            result = await self.execute_command(args)

            # Format and output result
            self.output_result(result, args.format)

            return 0

        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            return 1
        except Exception as e:
            if args.verbose:
                import traceback
                traceback.print_exc()
            else:
                print(f"Error: {e}")
            return 1

    async def execute_command(self, args) -> Dict[str, Any]:
        """Execute specific command"""
        if args.command == 'status':
            return await self.api_client.get_system_status()

        elif args.command in ['start', 'stop', 'pause', 'resume']:
            return await self.api_client.send_control_command(f"{args.command}_detection")

        elif args.command == 'data':
            data = await self.api_client.get_realtime_data(args.count)

            # Save to file if requested
            if args.output:
                await self.save_data_to_file(data, args.output)

            return data

        elif args.command == 'config':
            if args.get:
                return await self.api_client.get_configuration()
            elif args.set:
                key, value = args.set
                return await self.api_client.set_configuration(key, value)
            elif args.section:
                return await self.api_client.get_configuration_section(args.section)

        elif args.command == 'roi':
            if args.get:
                return await self.api_client.get_roi_config()
            elif args.set:
                x1, y1, x2, y2 = args.set
                return await self.api_client.set_roi_config(x1, y1, x2, y2)

        else:
            raise ValueError(f"Unknown command: {args.command}")

    def output_result(self, result: Dict[str, Any], format_type: str):
        """Format and output result"""
        if format_type == 'json':
            print(json.dumps(result, indent=2, default=str))
        elif format_type == 'table':
            self.print_table(result)
        elif format_type == 'csv':
            self.print_csv(result)
        else:
            print(result)

    def print_table(self, data: Dict[str, Any]):
        """Print data in table format"""
        if isinstance(data, dict):
            for key, value in data.items():
                print(f"{key:20}: {value}")
        else:
            print(data)

    def print_csv(self, data: Dict[str, Any]):
        """Print data in CSV format"""
        if 'series' in data:
            print("timestamp,value")
            for point in data['series']:
                print(f"{point.get('t', '')},{point.get('value', '')}")
        else:
            print("key,value")
            for key, value in data.items():
                print(f"{key},{value}")
```

## Data Synchronization Patterns

### 1. Request-Response Pattern

#### HTTP Polling Implementation
```javascript
// Frontend HTTP Polling
class HTTPPollingManager {
    constructor(apiClient, updateCallback, options = {}) {
        this.api = apiClient;
        this.callback = updateCallback;
        this.interval = options.interval || 50; // 20 FPS
        this.maxRetries = options.maxRetries || 5;
        this.retryDelay = options.retryDelay || 1000;

        this.isRunning = false;
        this.retryCount = 0;
        this.lastSuccessfulUpdate = null;
    }

    async start() {
        if (this.isRunning) return;

        this.isRunning = true;
        console.log(`Starting HTTP polling at ${1000/this.interval} FPS`);

        while (this.isRunning) {
            const startTime = performance.now();

            try {
                const data = await this.api.getRealtimeData();

                // Reset retry count on success
                this.retryCount = 0;
                this.lastSuccessfulUpdate = Date.now();

                // Process data
                this.callback(data);

            } catch (error) {
                this.retryCount++;
                console.error(`Polling error ${this.retryCount}/${this.maxRetries}:`, error);

                if (this.retryCount >= this.maxRetries) {
                    console.error('Max retries reached, stopping polling');
                    this.isRunning = false;
                    this.handleConnectionLost();
                    return;
                }

                // Exponential backoff
                await this.sleep(this.retryDelay * Math.pow(2, this.retryCount - 1));
                continue;
            }

            // Calculate sleep time to maintain polling interval
            const elapsed = performance.now() - startTime;
            const sleepTime = Math.max(0, this.interval - elapsed);

            await this.sleep(sleepTime);
        }
    }

    stop() {
        this.isRunning = false;
        console.log('HTTP polling stopped');
    }

    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    handleConnectionLost() {
        // Notify user of connection loss
        this.showConnectionError();

        // Schedule reconnection attempt
        setTimeout(() => {
            console.log('Attempting to restart polling...');
            this.retryCount = 0;
            this.start();
        }, 10000);
    }
}
```

### 2. Event-Driven Pattern

#### WebSocket Event Handling
```python
# Python WebSocket Event Handler
class WebSocketEventHandler:
    def __init__(self, plotter, gui_manager):
        self.plotter = plotter
        self.gui = gui_manager
        self.event_handlers = {
            'data': self.handle_data_event,
            'status': self.handle_status_event,
            'error': self.handle_error_event,
            'auth_response': self.handle_auth_response
        }

    async def handle_message(self, message_data):
        """Handle incoming WebSocket message"""
        try:
            message_type = message_data.get('type')

            if message_type in self.event_handlers:
                await self.event_handlers[message_type](message_data)
            else:
                print(f"Unknown message type: {message_type}")

        except Exception as e:
            print(f"Error handling message: {e}")

    async def handle_data_event(self, data):
        """Handle real-time data event"""
        timestamp = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
        value = data.get('current_value', 0)
        peak_signal = data.get('peak_signal')

        # Update plotter
        self.plotter.add_data_point(timestamp, value, peak_info={'signal': peak_signal})

        # Update GUI status
        self.gui.update_status({
            'frame_count': data.get('frame_count', 0),
            'current_value': value,
            'peak_signal': peak_signal,
            'status': data.get('status', 'unknown')
        })

    async def handle_status_event(self, data):
        """Handle system status event"""
        print(f"System status: {data}")
        self.gui.update_system_status(data)

    async def handle_error_event(self, data):
        """Handle error event"""
        error_message = data.get('error_message', 'Unknown error')
        print(f"Server error: {error_message}")
        self.gui.show_error(error_message)

    async def handle_auth_response(self, data):
        """Handle authentication response"""
        if data.get('success'):
            print("WebSocket authentication successful")
            self.gui.set_connection_status('connected')
        else:
            print("WebSocket authentication failed")
            self.gui.set_connection_status('authentication_failed')
```

### 3. Hybrid Pattern

#### Frontend Hybrid Implementation
```javascript
// Hybrid client with HTTP polling and optional WebSocket
class HybridClientManager {
    constructor(config, updateCallback) {
        this.config = config;
        this.callback = updateCallback;

        // Primary: HTTP polling
        this.httpPoller = new HTTPPollingManager(
            new NHEMFrontendAPI(config),
            this.handleDataUpdate.bind(this),
            { interval: 50 } // 20 FPS
        );

        // Optional: WebSocket for better performance
        this.websocketClient = null;
        this.useWebSocket = config.websocket?.enabled || false;

        // Fallback management
        this.usingWebSocket = false;
        this.lastUpdateTime = 0;
    }

    async start() {
        console.log('Starting hybrid client manager...');

        if (this.useWebSocket) {
            // Try WebSocket first
            try {
                this.websocketClient = new WebSocketClient(this.config);
                await this.websocketClient.connect();
                this.usingWebSocket = true;
                console.log('Using WebSocket for real-time updates');

            } catch (error) {
                console.warn('WebSocket connection failed, falling back to HTTP polling:', error);
                this.useWebSocket = false;
            }
        }

        if (!this.usingWebSocket) {
            // Use HTTP polling
            await this.httpPoller.start();
            console.log('Using HTTP polling for real-time updates');
        }
    }

    handleDataUpdate(data) {
        // Common data processing for both HTTP and WebSocket
        this.lastUpdateTime = Date.now();

        // Process data
        this.processData(data);

        // Forward to callback
        this.callback(data);
    }

    processData(data) {
        // Validate and process data
        if (!data || !data.series) {
            console.warn('Invalid data received:', data);
            return;
        }

        // Check for stale data
        const dataAge = Date.now() - new Date(data.timestamp).getTime();
        if (dataAge > 5000) { // 5 seconds threshold
            console.warn(`Stale data received: ${dataAge}ms old`);
        }
    }

    async fallbackToHTTP() {
        """Fallback from WebSocket to HTTP polling"""
        if (this.usingWebSocket) {
            console.log('Falling back to HTTP polling...');

            // Disconnect WebSocket
            if (this.websocketClient) {
                await this.websocketClient.disconnect();
                this.websocketClient = null;
            }

            this.usingWebSocket = false;

            // Start HTTP polling
            await this.httpPoller.start();

            console.log('Fallback to HTTP polling completed');
        }
    }
}
```

## Error Handling and Recovery

### 1. Network Error Handling

#### JavaScript Error Handler
```javascript
class NetworkErrorHandler {
    constructor(updateManager) {
        this.updateManager = updateManager;
        this.errorCounts = new Map();
        this.maxConsecutiveErrors = 5;
        this.recoveryStrategies = [
            this.retryWithBackoff.bind(this),
            this.switchToFallback.bind(this),
            this.notifyUser.bind(this)
        ];
    }

    handleError(error, context = 'unknown') {
        const count = this.errorCounts.get(context) || 0;
        this.errorCounts.set(context, count + 1);

        console.error(`Network error in ${context} (attempt ${count + 1}):`, error);

        // Choose recovery strategy based on error count
        const strategyIndex = Math.min(count, this.recoveryStrategies.length - 1);
        this.recoveryStrategies[strategyIndex](error, context, count);
    }

    async retryWithBackoff(error, context, count) {
        // Exponential backoff
        const delay = Math.min(1000 * Math.pow(2, count), 30000);

        console.log(`Retrying ${context} in ${delay}ms...`);
        setTimeout(() => {
            this.updateManager.start().catch(e => this.handleError(e, context));
        }, delay);
    }

    switchToFallback(error, context, count) {
        console.log(`Max retries reached for ${context}, switching to fallback`);

        if (this.updateManager.fallbackToHTTP) {
            this.updateManager.fallbackToHTTP();
        } else {
            this.notifyUser(error, context, count);
        }
    }

    notifyUser(error, context, count) {
        // Show user-friendly error message
        const errorMessage = this.getErrorMessage(error);
        this.showUserError(errorMessage);

        // Schedule reconnection attempt
        setTimeout(() => {
            this.errorCounts.delete(context);
            this.updateManager.start().catch(e => this.handleError(e, context));
        }, 10000);
    }

    getErrorMessage(error) {
        if (error.name === 'TypeError' && error.message.includes('Failed to fetch')) {
            return '网络连接失败，请检查网络连接';
        } else if (error.message.includes('timeout')) {
            return '请求超时，服务器响应缓慢';
        } else if (error.status === 401) {
            return '认证失败，请检查密码配置';
        } else if (error.status >= 500) {
            return '服务器内部错误，请稍后重试';
        } else {
            return `未知错误: ${error.message}`;
        }
    }

    showUserError(message) {
        const errorElement = document.getElementById('network-error');
        if (errorElement) {
            errorElement.textContent = message;
            errorElement.style.display = 'block';

            // Auto-hide after 5 seconds
            setTimeout(() => {
                errorElement.style.display = 'none';
            }, 5000);
        }
    }
}
```

### 2. Data Validation and Sanitization

#### Python Data Validator
```python
from typing import Dict, Any, List
import json
from datetime import datetime

class DataValidator:
    """Validate and sanitize incoming data from server"""

    @staticmethod
    def validate_realtime_data(data: Dict[str, Any]) -> bool:
        """Validate realtime data structure"""
        required_fields = ['type', 'timestamp', 'frame_count', 'series']

        # Check required fields
        for field in required_fields:
            if field not in data:
                return False

        # Validate data type
        if data['type'] != 'realtime_data':
            return False

        # Validate timestamp
        try:
            datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            return False

        # Validate frame count
        if not isinstance(data['frame_count'], int) or data['frame_count'] < 0:
            return False

        # Validate series
        if not isinstance(data['series'], list):
            return False

        for point in data['series']:
            if not isinstance(point, dict):
                return False

            if 't' not in point or 'value' not in point:
                return False

            if not isinstance(point['t'], (int, float)):
                return False

            if not isinstance(point['value'], (int, float)):
                return False

        return True

    @staticmethod
    def validate_roi_data(roi_data: Dict[str, Any]) -> bool:
        """Validate ROI data structure"""
        required_fields = ['width', 'height', 'pixels', 'gray_value', 'format']

        for field in required_fields:
            if field not in roi_data:
                return False

        # Validate dimensions
        if not isinstance(roi_data['width'], int) or roi_data['width'] <= 0:
            return False

        if not isinstance(roi_data['height'], int) or roi_data['height'] <= 0:
            return False

        # Validate gray value
        if not isinstance(roi_data['gray_value'], (int, float)):
            return False

        if not (0 <= roi_data['gray_value'] <= 255):
            return False

        return True

    @staticmethod
    def sanitize_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize incoming data to prevent issues"""
        sanitized = {}

        # Copy safe fields
        safe_fields = ['type', 'timestamp', 'frame_count', 'series', 'peak_signal',
                      'enhanced_peak', 'baseline', 'roi_data']

        for field in safe_fields:
            if field in data:
                try:
                    # Deep copy for nested structures
                    if isinstance(data[field], (dict, list)):
                        sanitized[field] = json.loads(json.dumps(data[field]))
                    else:
                        sanitized[field] = data[field]
                except (TypeError, ValueError):
                    # Skip problematic fields
                    continue

        return sanitized

    @staticmethod
    def validate_config_data(config: Dict[str, Any]) -> bool:
        """Validate configuration data"""
        # Check for potentially dangerous content
        dangerous_keys = ['__import__', 'eval', 'exec', 'open', 'file']

        for key in config:
            if key in dangerous_keys:
                return False

        # Recursively validate nested structures
        for value in config.values():
            if isinstance(value, dict):
                if not DataValidator.validate_config_data(value):
                    return False

        return True
```

## Performance Optimization

### 1. Frontend Optimization

#### Canvas Performance Optimization
```javascript
class OptimizedCanvasRenderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');

        // Performance optimization
        this.backBuffer = document.createElement('canvas');
        this.backCtx = this.backBuffer.getContext('2d');

        // Set up high DPI display
        this.setupHighDPIDisplay();

        // Performance monitoring
        this.frameCount = 0;
        this.lastFpsUpdate = performance.now();
        this.currentFps = 0;
    }

    setupHighDPIDisplay() {
        const dpr = window.devicePixelRatio || 1;
        const rect = this.canvas.getBoundingClientRect();

        // Set canvas resolution
        this.canvas.width = rect.width * dpr;
        this.canvas.height = rect.height * dpr;

        // Set back buffer resolution
        this.backBuffer.width = rect.width * dpr;
        this.backBuffer.height = rect.height * dpr;

        // Scale contexts
        this.ctx.scale(dpr, dpr);
        this.backCtx.scale(dpr, dpr);
    }

    render(data) {
        const startTime = performance.now();

        // Render to back buffer
        this.renderToBackBuffer(data);

        // Copy to main canvas (faster than direct rendering)
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.drawImage(this.backBuffer, 0, 0);

        // Update FPS counter
        this.updateFps(performance.now() - startTime);
    }

    renderToBackBuffer(data) {
        const ctx = this.backCtx;
        const width = this.canvas.width;
        const height = this.canvas.height;

        // Clear with dark background
        ctx.fillStyle = '#1e1e1e';
        ctx.fillRect(0, 0, width, height);

        // Batch draw operations
        ctx.save();

        // Draw grid
        this.drawGrid(ctx, width, height);

        // Draw axes
        this.drawAxes(ctx, width, height);

        // Draw data line (single operation)
        this.drawDataLine(ctx, data, width, height);

        // Draw peaks (batch operation)
        if (data.peaks && data.peaks.length > 0) {
            this.drawPeaks(ctx, data.peaks, width, height);
        }

        ctx.restore();
    }

    updateFps(frameTime) {
        this.frameCount++;
        const now = performance.now();

        if (now - this.lastFpsUpdate >= 1000) {
            this.currentFps = this.frameCount;
            this.frameCount = 0;
            this.lastFpsUpdate = now;

            // Log performance issues
            if (this.currentFps < 15) {
                console.warn(`Low FPS detected: ${this.currentFps}`);
            }
        }
    }
}
```

### 2. Python Client Optimization

#### Matplotlib Performance Optimization
```python
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import numpy as np

class OptimizedPlotter:
    """Optimized matplotlib plotter for real-time data"""

    def __init__(self, figure, max_points=1000):
        self.figure = figure
        self.max_points = max_points

        # Use deque for efficient append/pop operations
        self.x_data = deque(maxlen=max_points)
        self.y_data = deque(maxlen=max_points)

        # Setup plot with optimized settings
        self.setup_plot()

        # Animation control
        self.animation = None
        self.is_running = False

    def setup_plot(self):
        """Setup plot with performance optimizations"""
        # Create axis
        self.ax = self.figure.add_subplot(111)

        # Set dark background (faster rendering)
        self.ax.set_facecolor('#1e1e1e')
        self.figure.patch.set_facecolor('#2d2d30')

        # Create line object (reused for updates)
        self.line, = self.ax.plot([], [], 'c-', linewidth=1.5)

        # Configure axes
        self.ax.grid(True, alpha=0.2, color='#3e3e42')
        self.ax.set_xlabel('Time (s)', color='#cccccc')
        self.ax.set_ylabel('Value', color='#cccccc')

        # Style optimization
        self.ax.tick_params(colors='#cccccc')
        for spine in self.ax.spines.values():
            spine.set_color('#3e3e42')

    def add_data_point(self, x, y):
        """Add data point efficiently"""
        self.x_data.append(x)
        self.y_data.append(y)

    def update_plot(self, frame):
        """Optimized plot update function"""
        if len(self.x_data) == 0:
            return self.line,

        # Convert to numpy arrays for faster operations
        x_array = np.array(self.x_data)
        y_array = np.array(self.y_data)

        # Update line data (single operation)
        self.line.set_data(x_array, y_array)

        # Auto-scale axes efficiently
        if len(x_array) > 1:
            x_min, x_max = x_array.min(), x_array.max()
            y_min, y_max = y_array.min(), y_array.max()

            # Add padding
            x_padding = (x_max - x_min) * 0.05
            y_padding = (y_max - y_min) * 0.1

            self.ax.set_xlim(x_min - x_padding, x_max + x_padding)
            self.ax.set_ylim(y_min - y_padding, y_max + y_padding)

        return self.line,

    def start_animation(self, interval=50):
        """Start animation with optimized settings"""
        if self.is_running:
            return

        self.is_running = True

        # Use blitting for better performance
        self.animation = animation.FuncAnimation(
            self.figure,
            self.update_plot,
            interval=interval,
            blit=True,
            cache_frame_data=False  # Important for performance
        )

        # Optimize matplotlib settings
        plt.rcParams['animation.html'] = 'html5'
        plt.rcParams['path.simplify'] = True
        plt.rcParams['path.snap'] = True

    def stop_animation(self):
        """Stop animation"""
        if self.animation:
            self.animation.event_source.stop()
        self.is_running = False
```

## Security Considerations

### 1. Authentication Security

#### Secure Password Handling
```python
import secrets
import hashlib
from cryptography.fernet import Fernet
import keyring

class SecureAuthManager:
    """Handle authentication securely"""

    def __init__(self, config):
        self.config = config
        self.service_name = "nhem_client"

    def get_password(self) -> str:
        """Get password from secure storage"""
        # Try system keyring first
        try:
            password = keyring.get_password(self.service_name, "server_password")
            if password:
                return password
        except Exception:
            pass

        # Fallback to config file (less secure)
        return self.config.get('server', {}).get('password', '')

    def store_password(self, password: str):
        """Store password in secure storage"""
        try:
            keyring.set_password(self.service_name, "server_password", password)
            print("Password stored securely in system keyring")
        except Exception:
            print("Warning: Could not store password in secure storage")

    def validate_password(self, provided_password: str, expected_password: str) -> bool:
        """Constant-time password comparison to prevent timing attacks"""
        return secrets.compare_digest(
            provided_password.encode(),
            expected_password.encode()
        )

    def encrypt_sensitive_data(self, data: dict) -> str:
        """Encrypt sensitive configuration data"""
        key = self._get_encryption_key()
        f = Fernet(key)

        encrypted_data = f.encrypt(json.dumps(data).encode())
        return encrypted_data.decode()

    def decrypt_sensitive_data(self, encrypted_data: str) -> dict:
        """Decrypt sensitive configuration data"""
        key = self._get_encryption_key()
        f = Fernet(key)

        decrypted_data = f.decrypt(encrypted_data.encode())
        return json.loads(decrypted_data.decode())

    def _get_encryption_key(self) -> bytes:
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

            # Save with restricted permissions
            with open(key_file, 'wb') as f:
                f.write(key)

            os.chmod(key_file, 0o600)  # Read/write for owner only

            return key
```

### 2. Input Validation and Sanitization

#### Secure Input Handler
```python
import re
import html
from typing import Any, Dict, List

class SecureInputHandler:
    """Handle user input securely"""

    # Patterns for dangerous content
    DANGEROUS_PATTERNS = [
        r'<script[^>]*>.*?</script>',  # Script tags
        r'javascript:',                # JavaScript URLs
        r'on\w+\s*=',                # Event handlers
        r'expression\s*\(',          # CSS expressions
    ]

    # Allowed characters for different input types
    ALLOWED_PATTERNS = {
        'numeric': r'^[0-9\.]+$',
        'integer': r'^-?\d+$',
        'float': r'^-?\d+\.?\d*$',
        'coordinate': r'^-?\d+$',
        'alphanumeric': r'^[a-zA-Z0-9_\-\.]+$',
        'url': r'^https?://[^\s/$.?#].[^\s]*$',
    }

    @classmethod
    def sanitize_input(cls, input_str: str, input_type: str = 'alphanumeric') -> str:
        """Sanitize user input"""
        if not input_str:
            return ""

        # HTML encode to prevent XSS
        sanitized = html.escape(input_str)

        # Remove dangerous patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE | re.DOTALL)

        # Apply type-specific validation
        if input_type in cls.ALLOWED_PATTERNS:
            pattern = cls.ALLOWED_PATTERNS[input_type]
            if not re.match(pattern, sanitized):
                raise ValueError(f"Invalid {input_type} input: {input_str}")

        return sanitized

    @classmethod
    def validate_coordinates(cls, x1: int, y1: int, x2: int, y2: int) -> bool:
        """Validate ROI coordinates"""
        # Check if coordinates are integers
        if not all(isinstance(coord, int) for coord in [x1, y1, x2, y2]):
            return False

        # Check if coordinates are positive
        if any(coord < 0 for coord in [x1, y1, x2, y2]):
            return False

        # Check if x1 < x2 and y1 < y2
        if x1 >= x2 or y1 >= y2:
            return False

        # Check reasonable bounds (screen size)
        max_coord = 10000
        if any(coord > max_coord for coord in [x1, y1, x2, y2]):
            return False

        return True

    @classmethod
    def validate_api_response(cls, response_data: Any) -> bool:
        """Validate API response for security"""
        if not isinstance(response_data, dict):
            return False

        # Check for dangerous content
        response_str = json.dumps(response_data)
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, response_str, re.IGNORECASE | re.DOTALL):
                return False

        # Validate nested structures
        for value in response_data.values():
            if isinstance(value, dict):
                if not cls.validate_api_response(value):
                    return False
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        if not cls.validate_api_response(item):
                            return False

        return True
```

This comprehensive client-server interaction documentation provides complete understanding of how different clients communicate with the NHEM backend, including error handling, performance optimization, and security considerations. The patterns and examples shown here can be used as reference for developing new client integrations or improving existing ones.