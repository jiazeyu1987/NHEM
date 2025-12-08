#!/usr/bin/env python3
"""
Test script for Line Detection Configuration Manager
"""

import json
import os
import tempfile
from line_detection_config_manager import LineDetectionConfigManager, load_line_detection_config, save_line_detection_config


def test_config_validation():
    """Test configuration validation"""
    print("Testing configuration validation...")

    # Test valid configuration
    valid_config = {
        "config_metadata": {
            "version": "2.0.0",
            "format_version": "1.0.0",
            "schema_version": "1.0.0",
            "created_at": "2025-12-08T12:30:00Z",
            "updated_at": "2025-12-08T12:30:00Z",
            "created_by": "test",
            "description": "Test configuration",
            "migration_history": []
        },
        "line_detection": {
            "enabled": True,
            "auto_start": False,
            "update_interval": 100,
            "detection": {
                "hsv_thresholds": {
                    "green_lower_h": 40,
                    "green_upper_h": 80,
                    "green_lower_s": 50,
                    "green_upper_s": 255,
                    "green_lower_v": 50,
                    "green_upper_v": 255
                },
                "edge_detection": {
                    "canny_low_threshold": 25,
                    "canny_high_threshold": 80
                },
                "confidence_settings": {
                    "min_confidence": 0.4,
                    "medium_confidence_threshold": 0.5,
                    "high_confidence_threshold": 0.7
                }
            }
        }
    }

    manager = LineDetectionConfigManager()
    is_valid, message = manager.validate_config(valid_config)
    print(f"Valid config validation: {is_valid}, message: {message}")

    # Test invalid configuration
    invalid_config = {
        "config_metadata": {
            "version": "2.0.0"
            # Missing required fields
        },
        "line_detection": {
            "detection": {
                "hsv_thresholds": {
                    "green_lower_h": 80,  # Invalid: lower > upper
                    "green_upper_h": 40
                }
            }
        }
    }

    is_valid, message = manager.validate_config(invalid_config)
    print(f"Invalid config validation: {is_valid}, message: {message}")

    print("Configuration validation tests completed.\n")


def test_config_operations():
    """Test configuration save/load operations"""
    print("Testing configuration operations...")

    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        test_config_path = f.name

    try:
        # Create test configuration
        test_config = {
            "config_metadata": {
                "version": "2.0.0",
                "format_version": "1.0.0",
                "schema_version": "1.0.0",
                "created_at": "2025-12-08T12:30:00Z",
                "updated_at": "2025-12-08T12:30:00Z",
                "created_by": "test",
                "description": "Test configuration",
                "migration_history": []
            },
            "roi": {"x1": "10", "y1": "10", "x2": "200", "y2": "150"},
            "peak_detection": {"threshold": "105.0"},
            "line_detection": {
                "enabled": True,
                "auto_start": True,
                "ui": {
                    "enable_widget": True,
                    "show_control_panel": True,
                    "display_colors": {
                        "primary_line": "#00FF00",
                        "background": "#1E1E1E"
                    }
                },
                "detection": {
                    "hsv_thresholds": {
                        "green_lower_h": 40,
                        "green_upper_h": 80
                    }
                }
            }
        }

        # Save configuration
        with open(test_config_path, 'w') as f:
            json.dump(test_config, f, indent=2)

        # Test loading with configuration manager
        manager = LineDetectionConfigManager(test_config_path)
        success, message, config_data = manager.load_config()

        print(f"Load configuration: {success}")
        print(f"Message: {message}")

        if success:
            # Test getting line detection config
            line_config = manager.get_line_detection_config()
            print(f"Line detection enabled: {line_config.get('enabled')}")
            print(f"Auto start: {line_config.get('auto_start')}")

            # Test updating configuration
            updates = {
                "ui": {
                    "show_debug_panel": True
                },
                "detection": {
                    "performance": {
                        "cache_timeout_ms": 200
                    }
                }
            }

            success, message = manager.update_line_detection_config(updates)
            print(f"Update configuration: {success}")
            print(f"Message: {message}")

            # Test backup creation
            success, message = manager.create_backup()
            print(f"Create backup: {success}")
            print(f"Message: {message}")

            # Test export
            export_path = test_config_path.replace('.json', '_export.json')
            success, message = manager.export_config(export_path)
            print(f"Export configuration: {success}")
            print(f"Message: {message}")

            # Clean up export file
            if os.path.exists(export_path):
                os.remove(export_path)

    finally:
        # Clean up temporary file
        if os.path.exists(test_config_path):
            os.remove(test_config_path)

    print("Configuration operations tests completed.\n")


def test_convenience_functions():
    """Test convenience functions"""
    print("Testing convenience functions...")

    # Test loading with convenience function
    success, message, line_config = load_line_detection_config()

    if success:
        print(f"Line detection config loaded: {bool(line_config)}")
        print(f"Enabled: {line_config.get('enabled', 'not found')}")
        print(f"Auto start: {line_config.get('auto_start', 'not found')}")
    else:
        print(f"Failed to load: {message}")

    print("Convenience functions tests completed.\n")


def test_config_schema_compliance():
    """Test configuration schema compliance"""
    print("Testing configuration schema compliance...")

    # Test with actual configuration file
    config_path = os.path.join(os.path.dirname(__file__), "http_client_config.json")

    if os.path.exists(config_path):
        manager = LineDetectionConfigManager(config_path)
        success, message, config_data = manager.load_config()

        if success:
            print("✅ Configuration file loaded successfully")

            # Check for required sections
            line_config = manager.get_line_detection_config()
            required_sections = ["ui", "detection", "performance", "api", "synchronization", "backup", "import_export"]

            for section in required_sections:
                if section in line_config:
                    print(f"✅ Section '{section}' present")
                else:
                    print(f"❌ Section '{section}' missing")

            # Check for specific important settings
            ui_config = line_config.get("ui", {})
            if "display_colors" in ui_config:
                print("✅ UI color settings present")
            else:
                print("❌ UI color settings missing")

            detection_config = line_config.get("detection", {})
            if "hsv_thresholds" in detection_config:
                print("✅ HSV threshold settings present")
            else:
                print("❌ HSV threshold settings missing")

        else:
            print(f"❌ Failed to load configuration: {message}")
    else:
        print(f"❌ Configuration file not found: {config_path}")

    print("Schema compliance tests completed.\n")


def main():
    """Main test function"""
    print("=" * 60)
    print("Line Detection Configuration Manager Tests")
    print("=" * 60)

    try:
        test_config_validation()
        test_config_operations()
        test_convenience_functions()
        test_config_schema_compliance()

        print("=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()