# Dual ROI Integration Guide

## Overview

This document provides comprehensive examples and integration patterns for using the NHEM Dual ROI system in various applications and scenarios.

## Prerequisites

- NHEM Backend server running on `http://localhost:8421`
- Properly configured ROI region
- Basic understanding of HEM detection concepts

## Quick Start Examples

### 1. Basic Dual ROI Setup with Python

```python
import requests
import time
import json

class DualROIClient:
    def __init__(self, base_url="http://localhost:8421"):
        self.base_url = base_url
        self.session = requests.Session()

    def set_roi_config(self, x1, y1, x2, y2, password="31415"):
        """Configure ROI1 region (ROI2 automatically extracted from center)"""
        url = f"{self.base_url}/api/roi/config"
        data = {
            "x1": x1, "y1": y1,
            "x2": x2, "y2": y2,
            "password": password
        }
        response = self.session.post(url, data=data)
        return response.json()

    def get_dual_roi_frame(self):
        """Get both ROI1 and ROI2 data"""
        url = f"{self.base_url}/api/roi/dual-frame"
        response = self.session.get(url)
        return response.json()

    def start_detection(self, password="31415"):
        """Start the detection process"""
        url = f"{self.base_url}/control"
        data = {"command": "start_detection", "password": password}
        response = self.session.post(url, data=data)
        return response.json()

# Usage Example
client = DualROIClient()

# Configure ROI region (screen coordinates)
result = client.set_roi_config(480, 80, 1580, 580)
print("ROI Config:", result)

# Start detection
result = client.start_detection()
print("Detection Started:", result)

# Monitor dual ROI data
try:
    while True:
        dual_frame = client.get_dual_roi_frame()

        roi1_gray = dual_frame["roi1_data"]["gray_value"]
        roi2_gray = dual_frame["roi2_data"]["gray_value"]

        print(f"ROI1: {roi1_gray:.1f}, ROI2: {roi2_gray:.1f}")

        # Save ROI images if needed
        if "roi1_data" in dual_frame:
            roi1_data = dual_frame["roi1_data"]["pixels"]
            # Process base64 image data...

        if "roi2_data" in dual_frame:
            roi2_data = dual_frame["roi2_data"]["pixels"]
            # Process base64 image data...

        time.sleep(0.05)  # 20 FPS

except KeyboardInterrupt:
    print("Monitoring stopped")
```

### 2. Real-time Dual ROI Monitoring with Matplotlib

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import requests
import base64
from io import BytesIO
from PIL import Image
import numpy as np

class DualROIMonitor:
    def __init__(self, base_url="http://localhost:8421"):
        self.base_url = base_url
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        self.fig.suptitle('NHEM Dual ROI Real-time Monitor', fontsize=16)

        # Setup subplots
        self.setup_plots()

        # Data storage for plotting
        self.timestamps = []
        self.roi1_values = []
        self.roi2_values = []
        self.max_points = 200

    def setup_plots(self):
        # ROI1 Image Display
        self.ax1.set_title('ROI1 - Large Region')
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')
        self.ax1.axis('off')

        # ROI2 Image Display
        self.ax2.set_title('ROI2 - Center Region (50x50)')
        self.ax2.set_xlabel('X')
        self.ax2.set_ylabel('Y')
        self.ax2.axis('off')

        # Gray Value Timeline
        self.ax3.set_title('Gray Values Over Time')
        self.ax3.set_xlabel('Time (seconds)')
        self.ax3.set_ylabel('Gray Value')
        self.ax3.grid(True, alpha=0.3)

        # Statistics Display
        self.ax4.set_title('Real-time Statistics')
        self.ax4.axis('off')

    def get_dual_roi_data(self):
        """Fetch dual ROI data from server"""
        try:
            response = requests.get(f"{self.base_url}/api/roi/dual-frame")
            return response.json()
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def base64_to_image(self, base64_str):
        """Convert base64 string to PIL Image"""
        if base64_str.startswith('data:image/png;base64,'):
            base64_str = base64_str.replace('data:image/png;base64,', '')

        image_data = base64.b64decode(base64_str)
        return Image.open(BytesIO(image_data))

    def update_plots(self, frame):
        """Update all plots with new data"""
        dual_data = self.get_dual_roi_data()

        if not dual_data:
            return

        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax4.clear()

        # Re-setup labels
        self.ax1.set_title('ROI1 - Large Region')
        self.ax1.axis('off')
        self.ax2.set_title('ROI2 - Center Region (50x50)')
        self.ax2.axis('off')
        self.ax4.set_title('Real-time Statistics')
        self.ax4.axis('off')

        # Display ROI1 image
        if 'roi1_data' in dual_data:
            roi1_img = self.base64_to_image(dual_data['roi1_data']['pixels'])
            self.ax1.imshow(roi1_img, cmap='gray')
            roi1_gray = dual_data['roi1_data']['gray_value']

        # Display ROI2 image
        if 'roi2_data' in dual_data:
            roi2_img = self.base64_to_image(dual_data['roi2_data']['pixels'])
            self.ax2.imshow(roi2_img, cmap='gray')
            roi2_gray = dual_data['roi2_data']['gray_value']

        # Update timeline data
        current_time = len(self.timestamps) * 0.05  # 20 FPS
        self.timestamps.append(current_time)
        self.roi1_values.append(roi1_gray)
        self.roi2_values.append(roi2_gray)

        # Limit data points
        if len(self.timestamps) > self.max_points:
            self.timestamps.pop(0)
            self.roi1_values.pop(0)
            self.roi2_values.pop(0)

        # Update timeline plot
        self.ax3.clear()
        self.ax3.plot(self.timestamps, self.roi1_values, 'b-', label='ROI1', linewidth=2)
        self.ax3.plot(self.timestamps, self.roi2_values, 'r-', label='ROI2', linewidth=2)
        self.ax3.set_title('Gray Values Over Time')
        self.ax3.set_xlabel('Time (seconds)')
        self.ax3.set_ylabel('Gray Value')
        self.ax3.legend()
        self.ax3.grid(True, alpha=0.3)

        # Update statistics
        if self.roi1_values and self.roi2_values:
            roi1_avg = np.mean(self.roi1_values[-50:])  # Last 50 points
            roi2_avg = np.mean(self.roi2_values[-50:])
            roi1_std = np.std(self.roi1_values[-50:])
            roi2_std = np.std(self.roi2_values[-50:])

            stats_text = f"""
Current Values:
ROI1: {roi1_gray:.2f}
ROI2: {roi2_gray:.2f}

Recent Statistics (50 samples):
ROI1 - Avg: {roi1_avg:.2f}, Std: {roi1_std:.2f}
ROI2 - Avg: {roi2_avg:.2f}, Std: {roi2_std:.2f}

Difference: {roi1_gray - roi2_gray:.2f}
Ratio: {roi2_gray/roi1_gray:.3f if roi1_gray > 0 else 0:.3f}
            """

            self.ax4.text(0.1, 0.5, stats_text, transform=self.ax4.transAxes,
                          fontsize=12, verticalalignment='center',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    def start_monitoring(self):
        """Start real-time monitoring"""
        anim = FuncAnimation(self.fig, self.update_plots, interval=50, blit=False)
        plt.tight_layout()
        plt.show()
        return anim

# Usage
monitor = DualROIMonitor()
animation = monitor.start_monitoring()
```

### 3. Web Frontend Integration

```javascript
class DualROIWebClient {
    constructor(baseUrl = 'http://localhost:8421') {
        this.baseUrl = baseUrl;
        this.updateInterval = 50; // 20 FPS
        this.isMonitoring = false;
    }

    async configureROI(x1, y1, x2, y2, password = '31415') {
        const formData = new FormData();
        formData.append('x1', x1);
        formData.append('y1', y1);
        formData.append('x2', x2);
        formData.append('y2', y2);
        formData.append('password', password);

        const response = await fetch(`${this.baseUrl}/api/roi/config`, {
            method: 'POST',
            body: formData
        });

        return await response.json();
    }

    async getDualROI() {
        const response = await fetch(`${this.baseUrl}/api/roi/dual-frame`);
        return await response.json();
    }

    async startMonitoring(callback) {
        this.isMonitoring = true;

        const monitor = async () => {
            if (!this.isMonitoring) return;

            try {
                const dualData = await this.getDualROI();
                callback(dualData);
            } catch (error) {
                console.error('Error fetching dual ROI data:', error);
            }

            setTimeout(monitor, this.updateInterval);
        };

        monitor();
    }

    stopMonitoring() {
        this.isMonitoring = false;
    }
}

// HTML Integration Example
const client = new DualROIWebClient();

// Setup canvas elements
const roi1Canvas = document.getElementById('roi1-canvas');
const roi2Canvas = document.getElementById('roi2-canvas');
const roi1Ctx = roi1Canvas.getContext('2d');
const roi2Ctx = roi2Canvas.getContext('2d');

// Update callback
client.startMonitoring((dualData) => {
    // Update ROI1 display
    if (dualData.roi1_data) {
        const img1 = new Image();
        img1.onload = () => {
            roi1Ctx.clearRect(0, 0, roi1Canvas.width, roi1Canvas.height);
            roi1Ctx.drawImage(img1, 0, 0, roi1Canvas.width, roi1Canvas.height);
        };
        img1.src = dualData.roi1_data.pixels;

        document.getElementById('roi1-gray').textContent =
            `ROI1: ${dualData.roi1_data.gray_value.toFixed(1)}`;
    }

    // Update ROI2 display
    if (dualData.roi2_data) {
        const img2 = new Image();
        img2.onload = () => {
            roi2Ctx.clearRect(0, 0, roi2Canvas.width, roi2Canvas.height);
            roi2Ctx.drawImage(img2, 0, 0, roi2Canvas.width, roi2Canvas.height);
        };
        img2.src = dualData.roi2_data.pixels;

        document.getElementById('roi2-gray').textContent =
            `ROI2: ${dualData.roi2_data.gray_value.toFixed(1)}`;

        // Update ROI2 status indicator
        const statusElement = document.getElementById('roi2-status');
        if (dualData.roi2_data.gray_value === 0) {
            statusElement.className = 'status-warning';
            statusElement.textContent = '⚠️ ROI2: 0.0';
        } else {
            statusElement.className = 'status-normal';
            statusElement.textContent = '✅ ROI2: Normal';
        }
    }
});

// HTML structure example:
/*
<div class="dual-roi-monitor">
    <div class="roi-display">
        <canvas id="roi1-canvas" width="200" height="150"></canvas>
        <div id="roi1-gray">ROI1: --</div>
    </div>
    <div class="roi-display">
        <canvas id="roi2-canvas" width="100" height="100"></canvas>
        <div id="roi2-status" class="status-normal">ROI2: --</div>
    </div>
</div>
*/
```

### 4. Advanced Analysis with Peak Detection

```python
import requests
import numpy as np
from datetime import datetime
import json

class AdvancedDualROIAnalyzer:
    def __init__(self, base_url="http://localhost:8421"):
        self.base_url = base_url
        self.peak_history = []
        self.roi_data_buffer = []

    def get_enhanced_data(self, count=100):
        """Get enhanced data with peak detection information"""
        url = f"{self.base_url}/data/realtime/enhanced?count={count}"
        response = requests.get(url)
        return response.json()

    def analyze_peak_patterns(self, data_window=50):
        """Analyze peak detection patterns using dual ROI data"""
        enhanced_data = self.get_enhanced_data(data_window)

        if 'data' not in enhanced_data:
            return None

        analysis = {
            'total_frames': len(enhanced_data['data']),
            'peak_frames': 0,
            'green_peaks': 0,
            'red_peaks': 0,
            'roi1_avg': 0,
            'roi2_avg': 0,
            'roi2_roi1_correlation': 0,
            'peak_confidence_avg': 0,
            'peak_regions': []
        }

        roi1_values = []
        roi2_values = []
        peak_confidences = []

        for frame in enhanced_data['data']:
            roi1_values.append(frame.get('roi1_gray', 0))
            roi2_values.append(frame.get('roi2_gray', 0))

            if frame.get('peak_signal') == 1:
                analysis['peak_frames'] += 1
                peak_confidences.append(frame.get('peak_confidence', 0))

                if frame.get('peak_color') == 'green':
                    analysis['green_peaks'] += 1
                elif frame.get('peak_color') == 'red':
                    analysis['red_peaks'] += 1

                # Record peak region
                analysis['peak_regions'].append({
                    'timestamp': frame.get('timestamp'),
                    'roi1_gray': frame.get('roi1_gray', 0),
                    'roi2_gray': frame.get('roi2_gray', 0),
                    'peak_color': frame.get('peak_color'),
                    'confidence': frame.get('peak_confidence', 0)
                })

        # Calculate statistics
        if roi1_values:
            analysis['roi1_avg'] = np.mean(roi1_values)
            analysis['roi2_avg'] = np.mean(roi2_values)
            analysis['roi2_roi1_correlation'] = np.corrcoef(roi1_values, roi2_values)[0, 1]

        if peak_confidences:
            analysis['peak_confidence_avg'] = np.mean(peak_confidences)

        return analysis

    def detect_anomalies(self, threshold=2.0):
        """Detect anomalies in ROI2 data"""
        enhanced_data = self.get_enhanced_data(100)

        if 'data' not in enhanced_data:
            return []

        roi2_values = [frame.get('roi2_gray', 0) for frame in enhanced_data['data']]

        if not roi2_values:
            return []

        mean_roi2 = np.mean(roi2_values)
        std_roi2 = np.std(roi2_values)

        anomalies = []
        for i, frame in enumerate(enhanced_data['data']):
            roi2_val = frame.get('roi2_gray', 0)

            # Z-score anomaly detection
            if std_roi2 > 0:
                z_score = abs(roi2_val - mean_roi2) / std_roi2
                if z_score > threshold:
                    anomalies.append({
                        'frame_index': i,
                        'timestamp': frame.get('timestamp'),
                        'roi2_value': roi2_val,
                        'z_score': z_score,
                        'roi1_value': frame.get('roi1_gray', 0),
                        'peak_signal': frame.get('peak_signal', 0)
                    })

        return anomalies

    def generate_report(self):
        """Generate comprehensive analysis report"""
        analysis = self.analyze_peak_patterns()
        anomalies = self.detect_anomalies()

        report = {
            'timestamp': datetime.now().isoformat(),
            'analysis': analysis,
            'anomalies': {
                'count': len(anomalies),
                'items': anomalies[:10]  # Top 10 anomalies
            },
            'recommendations': []
        }

        # Generate recommendations
        if analysis:
            if analysis['roi2_roi1_correlation'] < 0.5:
                report['recommendations'].append(
                    "Low correlation between ROI1 and ROI2. Consider adjusting ROI region."
                )

            if analysis['peak_confidence_avg'] < 0.7:
                report['recommendations'].append(
                    "Low peak confidence detected. Consider adjusting peak detection parameters."
                )

            if anomalies:
                report['recommendations'].append(
                    f"Found {len(anomalies)} anomalies. Investigate ROI2 data quality."
                )

        return report

# Usage Example
analyzer = AdvancedDualROIAnalyzer()

# Generate comprehensive report
report = analyzer.generate_report()
print("Analysis Report:")
print(json.dumps(report, indent=2))
```

### 5. Configuration Management

```python
class DualROIConfigManager:
    def __init__(self, base_url="http://localhost:8421", config_file="dual_roi_config.json"):
        self.base_url = base_url
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self):
        """Load configuration from file"""
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self.get_default_config()

    def save_config(self):
        """Save configuration to file"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)

    def get_default_config(self):
        """Get default configuration"""
        return {
            "roi": {
                "x1": 480,
                "y1": 80,
                "x2": 1580,
                "y2": 580,
                "frame_rate": 4
            },
            "peak_detection": {
                "threshold": 104.0,
                "margin_frames": 5,
                "difference_threshold": 1.1,
                "min_region_length": 5
            },
            "monitoring": {
                "update_rate": 50,  # milliseconds
                "save_images": False,
                "log_level": "INFO"
            }
        }

    def apply_config(self, password="31415"):
        """Apply configuration to server"""
        # Set ROI configuration
        roi_config = self.config["roi"]
        roi_url = f"{self.base_url}/api/roi/config"
        roi_data = {
            "x1": roi_config["x1"],
            "y1": roi_config["y1"],
            "x2": roi_config["x2"],
            "y2": roi_config["y2"],
            "password": password
        }

        roi_response = requests.post(roi_url, data=roi_data)

        # Set peak detection configuration
        peak_config = self.config["peak_detection"]
        peak_url = f"{self.base_url}/peak-detection/config"
        peak_data = {
            **peak_config,
            "password": password
        }

        peak_response = requests.post(peak_url, data=peak_data)

        return {
            "roi_config": roi_response.json(),
            "peak_config": peak_response.json()
        }

    def update_config(self, section, key, value):
        """Update configuration value"""
        if section in self.config:
            self.config[section][key] = value
            self.save_config()
        else:
            raise ValueError(f"Unknown configuration section: {section}")

# Usage Example
config_manager = DualROIConfigManager()

# Update ROI configuration
config_manager.update_config("roi", "x1", 500)
config_manager.update_config("roi", "frame_rate", 5)

# Apply configuration to server
result = config_manager.apply_config()
print("Configuration applied:", result)
```

## Error Handling Best Practices

### 1. Robust Error Handling

```python
import logging
from requests.exceptions import RequestException, Timeout, ConnectionError

class RobustDualROIClient:
    def __init__(self, base_url="http://localhost:8421", max_retries=3):
        self.base_url = base_url
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.timeout = 5.0  # 5 second timeout

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def get_dual_roi_with_retry(self):
        """Get dual ROI data with retry logic"""
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(f"{self.base_url}/api/roi/dual-frame")
                response.raise_for_status()
                return response.json()

            except Timeout:
                self.logger.warning(f"Timeout on attempt {attempt + 1}")
                if attempt == self.max_retries - 1:
                    raise

            except ConnectionError:
                self.logger.warning(f"Connection error on attempt {attempt + 1}")
                if attempt == self.max_retries - 1:
                    raise

            except RequestException as e:
                self.logger.error(f"Request failed: {e}")
                raise

            except json.JSONDecodeError as e:
                self.logger.error(f"Invalid JSON response: {e}")
                raise

            time.sleep(1 * (attempt + 1))  # Exponential backoff

        return None

    def validate_dual_roi_response(self, data):
        """Validate dual ROI response data"""
        if not data:
            raise ValueError("Empty response data")

        required_keys = ['type', 'timestamp', 'roi1_data', 'roi2_data']
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")

        # Validate ROI data
        for roi_key in ['roi1_data', 'roi2_data']:
            roi_data = data[roi_key]
            required_roi_keys = ['width', 'height', 'pixels', 'gray_value']
            for key in required_roi_keys:
                if key not in roi_data:
                    raise ValueError(f"Missing ROI key {key} in {roi_key}")

            # Validate gray value range
            gray_value = roi_data['gray_value']
            if not (0 <= gray_value <= 255):
                raise ValueError(f"Invalid gray value {gray_value} in {roi_key}")

        return True
```

### 2. Graceful Degradation

```python
class GracefulDualROI:
    def __init__(self, base_url="http://localhost:8421"):
        self.base_url = base_url
        self.fallback_mode = False

    def get_dual_roi_or_fallback(self):
        """Get dual ROI data with fallback to single ROI or simulated data"""
        try:
            # Try dual ROI first
            response = requests.get(f"{self.base_url}/api/roi/dual-frame", timeout=2)
            response.raise_for_status()
            data = response.json()

            # Validate response
            if self.validate_response(data):
                self.fallback_mode = False
                return data
            else:
                raise ValueError("Invalid response data")

        except Exception as e:
            self.logger.warning(f"Dual ROI failed ({e}), trying single ROI")

            try:
                # Fallback to single ROI
                response = requests.get(f"{self.base_url}/api/roi/frame", timeout=2)
                response.raise_for_status()
                data = response.json()

                # Convert single ROI to dual ROI format
                dual_data = self.convert_to_dual_format(data)
                self.fallback_mode = "single_roi"
                return dual_data

            except Exception as e2:
                self.logger.error(f"Single ROI failed ({e2}), using simulated data")

                # Final fallback to simulated data
                return self.generate_simulated_data()

    def convert_to_dual_format(self, single_roi_data):
        """Convert single ROI data to dual ROI format"""
        import random
        import datetime

        roi_data = single_roi_data.get('roi_data', {})

        return {
            "type": "dual_roi_frame",
            "timestamp": datetime.datetime.now().isoformat(),
            "roi1_data": roi_data,
            "roi2_data": {
                "width": 50,
                "height": 50,
                "pixels": roi_data.get('pixels', ''),
                "gray_value": roi_data.get('gray_value', 0) + random.uniform(-5, 5)
            }
        }

    def generate_simulated_data(self):
        """Generate simulated dual ROI data"""
        import random
        import datetime

        base_value = 120.0
        noise = random.uniform(-10, 10)

        return {
            "type": "dual_roi_frame",
            "timestamp": datetime.datetime.now().isoformat(),
            "roi1_data": {
                "width": 200,
                "height": 150,
                "pixels": "",
                "gray_value": base_value + noise
            },
            "roi2_data": {
                "width": 50,
                "height": 50,
                "pixels": "",
                "gray_value": base_value + noise + random.uniform(-5, 5)
            }
        }
```

## Performance Optimization

### 1. Connection Pooling

```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class OptimizedDualROIClient:
    def __init__(self, base_url="http://localhost:8421"):
        self.base_url = base_url

        # Setup session with connection pooling
        self.session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )

        # Configure adapter with connection pooling
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=20,
            pool_block=False
        )

        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
```

### 2. Asynchronous Processing

```python
import aiohttp
import asyncio
import json

class AsyncDualROIClient:
    def __init__(self, base_url="http://localhost:8421"):
        self.base_url = base_url

    async def get_dual_roi_async(self):
        """Asynchronously get dual ROI data"""
        timeout = aiohttp.ClientTimeout(total=2.0)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.get(f"{self.base_url}/api/roi/dual-frame") as response:
                    response.raise_for_status()
                    return await response.json()

            except asyncio.TimeoutError:
                print("Request timeout")
                return None
            except Exception as e:
                print(f"Request failed: {e}")
                return None

    async def monitor_dual_roi_async(self, callback, interval=0.05):
        """Asynchronously monitor dual ROI data"""
        while True:
            data = await self.get_dual_roi_async()
            if data:
                callback(data)
            await asyncio.sleep(interval)

# Usage
async def main():
    client = AsyncDualROIClient()

    def data_callback(data):
        print(f"ROI1: {data['roi1_data']['gray_value']:.1f}")
        print(f"ROI2: {data['roi2_data']['gray_value']:.1f}")

    await client.monitor_dual_roi_async(data_callback)

# Run the async monitor
# asyncio.run(main())
```

This comprehensive guide provides everything needed to integrate the NHEM Dual ROI system into various applications with proper error handling, performance optimization, and real-world usage patterns.