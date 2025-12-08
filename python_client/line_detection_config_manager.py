#!/usr/bin/env python3
"""
Line Detection Configuration Manager
Comprehensive configuration management for line detection settings
"""

import json
import os
import logging
import shutil
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import csv

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    # Create a dummy yaml module for compatibility
    class DummyYAML:
        @staticmethod
        def dump(data, stream=None, **kwargs):
            if stream:
                stream.write("# YAML not available - using JSON format\n")
                json.dump(data, stream, indent=2)
            else:
                return "# YAML not available"

        @staticmethod
        def safe_load(stream):
            # Remove potential YAML comments and parse as JSON
            content = stream.read()
            lines = [line for line in content.split('\n') if not line.strip().startswith('#')]
            cleaned_content = '\n'.join(lines)
            try:
                return json.loads(cleaned_content)
            except:
                return {}

    yaml = DummyYAML()

try:
    import gzip
    GZIP_AVAILABLE = True
except ImportError:
    GZIP_AVAILABLE = False


class LineDetectionConfigManager:
    """Line Detection Configuration Manager"""

    def __init__(self, config_file_path: Optional[str] = None):
        """
        Initialize configuration manager

        Args:
            config_file_path: Path to configuration file
        """
        self.config_file_path = config_file_path or self._get_default_config_path()
        self.config_data = None
        self.logger = logging.getLogger(__name__)
        self._schema = self._load_schema()

    def _get_default_config_path(self) -> str:
        """Get default configuration file path"""
        return os.path.join(os.path.dirname(__file__), "http_client_config.json")

    def _load_schema(self) -> Dict[str, Any]:
        """Load configuration validation schema"""
        return {
            "config_metadata": {
                "required": True,
                "fields": {
                    "version": {"type": str, "required": True},
                    "format_version": {"type": str, "required": True},
                    "schema_version": {"type": str, "required": True},
                    "created_at": {"type": str, "required": True},
                    "updated_at": {"type": str, "required": True},
                    "created_by": {"type": str, "required": True},
                    "description": {"type": str, "required": True},
                    "migration_history": {"type": list, "required": True}
                }
            },
            "line_detection": {
                "required": True,
                "fields": {
                    "enabled": {"type": bool, "required": True},
                    "auto_start": {"type": bool, "required": True},
                    "update_interval": {"type": int, "min": 50, "max": 5000},
                    "ui": {
                        "type": dict,
                        "fields": {
                            "enable_widget": {"type": bool},
                            "show_control_panel": {"type": bool},
                            "show_statistics_panel": {"type": bool},
                            "auto_expand_results": {"type": bool},
                            "refresh_on_data_update": {"type": bool}
                        }
                    },
                    "detection": {
                        "type": dict,
                        "fields": {
                            "hsv_thresholds": {
                                "type": dict,
                                "fields": {
                                    "green_lower_h": {"type": int, "min": 0, "max": 179},
                                    "green_lower_s": {"type": int, "min": 0, "max": 255},
                                    "green_lower_v": {"type": int, "min": 0, "max": 255},
                                    "green_upper_h": {"type": int, "min": 0, "max": 179},
                                    "green_upper_s": {"type": int, "min": 0, "max": 255},
                                    "green_upper_v": {"type": int, "min": 0, "max": 255},
                                    "auto_adjust": {"type": bool},
                                    "adaptive_threshold": {"type": bool}
                                }
                            },
                            "edge_detection": {
                                "type": dict,
                                "fields": {
                                    "canny_low_threshold": {"type": int, "min": 0, "max": 255},
                                    "canny_high_threshold": {"type": int, "min": 0, "max": 255},
                                    "canny_kernel_size": {"type": int, "min": 1, "max": 7},
                                    "blur_kernel_size": {"type": int, "min": 1, "max": 15}
                                }
                            },
                            "hough_parameters": {
                                "type": dict,
                                "fields": {
                                    "hough_threshold": {"type": int, "min": 1, "max": 200},
                                    "hough_min_line_length": {"type": int, "min": 1, "max": 100},
                                    "hough_max_line_gap": {"type": int, "min": 1, "max": 50},
                                    "max_lines": {"type": int, "min": 1, "max": 500}
                                }
                            },
                            "confidence_settings": {
                                "type": dict,
                                "fields": {
                                    "min_confidence": {"type": float, "min": 0.0, "max": 1.0},
                                    "high_confidence_threshold": {"type": float, "min": 0.0, "max": 1.0},
                                    "medium_confidence_threshold": {"type": float, "min": 0.0, "max": 1.0}
                                }
                            }
                        }
                    }
                }
            }
        }

    def load_config(self) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Load configuration file

        Returns:
            Tuple[bool, str, Optional[Dict]]: (success, message, config_data)
        """
        try:
            if not os.path.exists(self.config_file_path):
                error_msg = f"Configuration file not found: {self.config_file_path}"
                self.logger.error(error_msg)
                return False, error_msg, None

            with open(self.config_file_path, 'r', encoding='utf-8') as f:
                self.config_data = json.load(f)

            # Validate configuration
            validation_result = self.validate_config(self.config_data)
            if not validation_result[0]:
                return validation_result

            success_msg = f"Configuration loaded successfully: {self.config_file_path}"
            self.logger.info(success_msg)
            return True, success_msg, self.config_data

        except json.JSONDecodeError as e:
            error_msg = f"JSON parsing error: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg, None

        except Exception as e:
            error_msg = f"Error loading configuration: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg, None

    def save_config(self, config_data: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        """
        Save configuration to file

        Args:
            config_data: Configuration data to save (uses loaded config if None)

        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            data_to_save = config_data or self.config_data
            if not data_to_save:
                return False, "No configuration data to save"

            # Update metadata
            if "config_metadata" in data_to_save:
                data_to_save["config_metadata"]["updated_at"] = datetime.now(timezone.utc).isoformat()

            # Validate before saving
            validation_result = self.validate_config(data_to_save)
            if not validation_result[0]:
                return validation_result

            # Create backup if file exists
            if os.path.exists(self.config_file_path):
                backup_result = self.create_backup()
                if not backup_result[0]:
                    self.logger.warning(f"Failed to create backup: {backup_result[1]}")

            # Save configuration
            with open(self.config_file_path, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=2, ensure_ascii=False)

            if config_data:
                self.config_data = config_to_save

            success_msg = f"Configuration saved successfully: {self.config_file_path}"
            self.logger.info(success_msg)
            return True, success_msg

        except Exception as e:
            error_msg = f"Error saving configuration: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg

    def validate_config(self, config_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate configuration data against schema

        Args:
            config_data: Configuration data to validate

        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        try:
            if not isinstance(config_data, dict):
                return False, "Configuration must be a dictionary"

            # Validate metadata
            if "config_metadata" not in config_data:
                return False, "Missing config_metadata section"

            metadata = config_data["config_metadata"]
            required_metadata = ["version", "format_version", "schema_version", "created_at", "updated_at"]
            for field in required_metadata:
                if field not in metadata:
                    return False, f"Missing required metadata field: {field}"

            # Validate line detection section
            if "line_detection" not in config_data:
                return False, "Missing line_detection section"

            line_detection = config_data["line_detection"]

            # Validate detection parameters ranges
            if "detection" in line_detection:
                detection = line_detection["detection"]

                if "hsv_thresholds" in detection:
                    hsv = detection["hsv_thresholds"]
                    # Check HSV ranges
                    if "green_lower_h" in hsv and "green_upper_h" in hsv:
                        if hsv["green_lower_h"] >= hsv["green_upper_h"]:
                            return False, "HSV lower hue must be less than upper hue"

                    if "green_lower_s" in hsv and "green_upper_s" in hsv:
                        if hsv["green_lower_s"] >= hsv["green_upper_s"]:
                            return False, "HSV lower saturation must be less than upper saturation"

                    if "green_lower_v" in hsv and "green_upper_v" in hsv:
                        if hsv["green_lower_v"] >= hsv["green_upper_v"]:
                            return False, "HSV lower value must be less than upper value"

                if "edge_detection" in detection:
                    edge = detection["edge_detection"]
                    if "canny_low_threshold" in edge and "canny_high_threshold" in edge:
                        if edge["canny_low_threshold"] >= edge["canny_high_threshold"]:
                            return False, "Canny low threshold must be less than high threshold"

                if "confidence_settings" in detection:
                    confidence = detection["confidence_settings"]
                    thresholds = ["min_confidence", "medium_confidence_threshold", "high_confidence_threshold"]
                    values = [confidence.get(t, 0) for t in thresholds]

                    # Ensure thresholds are in ascending order
                    if values != sorted(values):
                        return False, "Confidence thresholds must be in ascending order"

            return True, "Configuration is valid"

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def get_line_detection_config(self) -> Dict[str, Any]:
        """
        Get line detection configuration section

        Returns:
            Dict: Line detection configuration
        """
        if not self.config_data:
            return {}

        return self.config_data.get("line_detection", {})

    def update_line_detection_config(self, updates: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Update line detection configuration

        Args:
            updates: Configuration updates

        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            if not self.config_data:
                return False, "No configuration loaded"

            # Deep merge updates
            current_config = self.config_data.get("line_detection", {})
            merged_config = self._deep_merge(current_config, updates)

            # Validate updated configuration
            test_config = self.config_data.copy()
            test_config["line_detection"] = merged_config

            validation_result = self.validate_config(test_config)
            if not validation_result[0]:
                return validation_result

            # Apply updates
            self.config_data["line_detection"] = merged_config

            return True, "Line detection configuration updated successfully"

        except Exception as e:
            error_msg = f"Error updating configuration: {str(e)}"
            return False, error_msg

    def _deep_merge(self, base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        result = base.copy()

        for key, value in updates.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def create_backup(self) -> Tuple[bool, str]:
        """
        Create backup of current configuration

        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            if not os.path.exists(self.config_file_path):
                return False, "Configuration file does not exist"

            # Get line detection config for backup settings
            line_config = self.get_line_detection_config()
            backup_config = line_config.get("backup", {})

            # Create backup directory
            backup_dir = backup_config.get("backup_directory", "./backups/line_detection/")
            Path(backup_dir).mkdir(parents=True, exist_ok=True)

            # Generate backup filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix = backup_config.get("backup_file_prefix", "line_detection_config")
            filename = f"{prefix}_{timestamp}.json"

            if backup_config.get("include_timestamp_in_filename", True):
                filename = f"{prefix}_{timestamp}.json"
            else:
                filename = f"{prefix}.json"

            backup_path = os.path.join(backup_dir, filename)

            # Copy file
            shutil.copy2(self.config_file_path, backup_path)

            # Compress if enabled
            if backup_config.get("compress_backups", False) and GZIP_AVAILABLE:
                compressed_path = backup_path + ".gz"
                with open(backup_path, 'rb') as f_in:
                    with gzip.open(compressed_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(backup_path)
                backup_path = compressed_path

            # Clean old backups
            max_backups = backup_config.get("max_backup_files", 10)
            self._cleanup_old_backups(backup_dir, prefix, max_backups)

            success_msg = f"Backup created successfully: {backup_path}"
            self.logger.info(success_msg)
            return True, success_msg

        except Exception as e:
            error_msg = f"Error creating backup: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg

    def _cleanup_old_backups(self, backup_dir: str, prefix: str, max_backups: int):
        """Clean up old backup files"""
        try:
            backup_files = []
            for filename in os.listdir(backup_dir):
                if filename.startswith(prefix):
                    filepath = os.path.join(backup_dir, filename)
                    backup_files.append((filepath, os.path.getmtime(filepath)))

            # Sort by modification time (oldest first)
            backup_files.sort(key=lambda x: x[1])

            # Remove excess backups
            while len(backup_files) > max_backups:
                oldest_file = backup_files.pop(0)[0]
                os.remove(oldest_file)
                self.logger.info(f"Removed old backup: {oldest_file}")

        except Exception as e:
            self.logger.warning(f"Error cleaning up old backups: {str(e)}")

    def export_config(self, export_path: str, format_type: str = "json") -> Tuple[bool, str]:
        """
        Export configuration to file

        Args:
            export_path: Export file path
            format_type: Export format (json, yaml, csv)

        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            if not self.config_data:
                return False, "No configuration loaded"

            line_config = self.get_line_detection_config()
            export_config = line_config.get("import_export", {})

            # Filter sensitive data if enabled
            config_to_export = self.config_data.copy()
            if export_config.get("export_filter_sensitive_data", True):
                config_to_export = self._filter_sensitive_data(config_to_export)

            # Add metadata
            if export_config.get("include_metadata", True):
                metadata = {
                    "export_timestamp": datetime.now(timezone.utc).isoformat(),
                    "export_format": format_type,
                    "exported_by": "line_detection_config_manager",
                    "config_version": self.config_data.get("config_metadata", {}).get("version", "unknown")
                }
                config_to_export["_export_metadata"] = metadata

            # Export based on format
            if format_type.lower() == "json":
                with open(export_path, 'w', encoding='utf-8') as f:
                    if export_config.get("export_pretty_format", True):
                        json.dump(config_to_export, f, indent=2, ensure_ascii=False)
                    else:
                        json.dump(config_to_export, f, ensure_ascii=False)

            elif format_type.lower() == "yaml":
                if not YAML_AVAILABLE:
                    return False, "YAML support not available. Please install PyYAML package."
                with open(export_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config_to_export, f, default_flow_style=False, allow_unicode=True)

            elif format_type.lower() == "csv":
                self._export_to_csv(config_to_export, export_path)

            else:
                return False, f"Unsupported export format: {format_type}"

            success_msg = f"Configuration exported successfully: {export_path}"
            self.logger.info(success_msg)
            return True, success_msg

        except Exception as e:
            error_msg = f"Error exporting configuration: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg

    def import_config(self, import_path: str, merge: bool = False) -> Tuple[bool, str]:
        """
        Import configuration from file

        Args:
            import_path: Import file path
            merge: Whether to merge with existing config or replace

        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            if not os.path.exists(import_path):
                return False, f"Import file not found: {import_path}"

            # Determine file format
            file_ext = os.path.splitext(import_path)[1].lower()

            # Load configuration
            if file_ext == '.json':
                with open(import_path, 'r', encoding='utf-8') as f:
                    imported_config = json.load(f)
            elif file_ext in ['.yaml', '.yml']:
                if not YAML_AVAILABLE:
                    return False, "YAML support not available. Please install PyYAML package."
                with open(import_path, 'r', encoding='utf-8') as f:
                    imported_config = yaml.safe_load(f)
            elif file_ext == '.csv':
                imported_config = self._import_from_csv(import_path)
            else:
                return False, f"Unsupported import format: {file_ext}"

            # Remove export metadata if present
            if "_export_metadata" in imported_config:
                del imported_config["_export_metadata"]

            # Validate imported configuration
            validation_result = self.validate_config(imported_config)
            line_config = self.get_line_detection_config()
            import_config = line_config.get("import_export", {})

            if import_config.get("validation_on_import", True) and not validation_result[0]:
                if import_config.get("import_strict_validation", False):
                    return False, f"Import validation failed: {validation_result[1]}"
                else:
                    self.logger.warning(f"Import validation warning: {validation_result[1]}")

            # Create backup before importing
            backup_result = self.create_backup()
            if not backup_result[0]:
                self.logger.warning(f"Failed to create backup before import: {backup_result[1]}")

            # Merge or replace configuration
            if merge:
                updated_config = self._deep_merge(self.config_data or {}, imported_config)
            else:
                updated_config = imported_config

            # Update metadata
            if "config_metadata" in updated_config:
                updated_config["config_metadata"]["updated_at"] = datetime.now(timezone.utc).isoformat()

                # Add migration entry
                migration_history = updated_config["config_metadata"].get("migration_history", [])
                migration_history.append({
                    "version": updated_config["config_metadata"].get("version", "unknown"),
                    "description": f"Configuration imported from {import_path}",
                    "migration_date": datetime.now(timezone.utc).isoformat()
                })
                updated_config["config_metadata"]["migration_history"] = migration_history

            # Save updated configuration
            save_result = self.save_config(updated_config)
            if not save_result[0]:
                return save_result

            success_msg = f"Configuration imported successfully: {import_path}"
            self.logger.info(success_msg)
            return True, success_msg

        except Exception as e:
            error_msg = f"Error importing configuration: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg

    def _filter_sensitive_data(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter sensitive data from configuration"""
        filtered_config = config_data.copy()

        # List of sensitive keys to filter
        sensitive_keys = ["password", "auth_token", "api_key", "secret_key"]

        def filter_recursive(obj):
            if isinstance(obj, dict):
                return {
                    k: "FILTERED" if any(sensitive in k.lower() for sensitive in sensitive_keys)
                    else filter_recursive(v)
                    for k, v in obj.items()
                }
            elif isinstance(obj, list):
                return [filter_recursive(item) for item in obj]
            else:
                return obj

        return filter_recursive(filtered_config)

    def _export_to_csv(self, config_data: Dict[str, Any], export_path: str):
        """Export configuration to CSV format"""
        with open(export_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Section", "Key", "Value", "Type"])

            def write_section(section_name: str, section_data: Any, parent_path: str = ""):
                if isinstance(section_data, dict):
                    for key, value in section_data.items():
                        current_path = f"{parent_path}.{key}" if parent_path else key
                        if isinstance(value, dict):
                            write_section(section_name, value, current_path)
                        else:
                            writer.writerow([
                                section_name,
                                current_path,
                                str(value) if value is not None else "",
                                type(value).__name__
                            ])

            for section_name, section_data in config_data.items():
                if section_name != "config_metadata":  # Skip metadata in CSV export
                    write_section(section_name, section_data)

    def _import_from_csv(self, import_path: str) -> Dict[str, Any]:
        """Import configuration from CSV format"""
        config = {}

        with open(import_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                section = row["Section"]
                key = row["Key"]
                value = row["Value"]
                data_type = row["Type"]

                if section not in config:
                    config[section] = {}

                # Convert value based on type
                if data_type == "bool":
                    value = value.lower() in ["true", "1", "yes"]
                elif data_type == "int":
                    value = int(value) if value else 0
                elif data_type == "float":
                    value = float(value) if value else 0.0

                # Set nested key
                keys = key.split(".")
                current = config[section]
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = value

        return config

    def sync_with_backend(self, backend_config: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Synchronize configuration with backend settings

        Args:
            backend_config: Backend configuration to sync with

        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            if not self.config_data:
                return False, "No configuration loaded"

            line_config = self.get_line_detection_config()
            sync_config = line_config.get("synchronization", {})

            if not sync_config.get("auto_sync_with_backend", True):
                return True, "Backend synchronization disabled"

            # Extract backend line detection settings
            backend_line_config = backend_config.get("line_detection", {})

            if not backend_line_config:
                return True, "No line detection configuration in backend"

            # Apply tolerance for synchronization
            tolerance = sync_config.get("sync_threshold_tolerance", 0.1)

            # Update detection parameters
            detection_updates = {}

            if "hsv_green_lower" in backend_line_config:
                hsv_lower = backend_line_config["hsv_green_lower"]
                current_hsv_lower = line_config.get("detection", {}).get("hsv_thresholds", {})

                # Apply tolerance check
                for i, key in enumerate(["green_lower_h", "green_lower_s", "green_lower_v"]):
                    if key in current_hsv_lower:
                        diff = abs(current_hsv_lower[key] - hsv_lower[i])
                        if diff > tolerance * 255:  # Scale tolerance for 0-255 range
                            if "hsv_thresholds" not in detection_updates:
                                detection_updates["hsv_thresholds"] = {}
                            detection_updates["hsv_thresholds"][key] = hsv_lower[i]

            if "hsv_green_upper" in backend_line_config:
                hsv_upper = backend_line_config["hsv_green_upper"]
                current_hsv_upper = line_config.get("detection", {}).get("hsv_thresholds", {})

                for i, key in enumerate(["green_upper_h", "green_upper_s", "green_upper_v"]):
                    if key in current_hsv_upper:
                        diff = abs(current_hsv_upper[key] - hsv_upper[i])
                        if diff > tolerance * 255:
                            if "hsv_thresholds" not in detection_updates:
                                detection_updates["hsv_thresholds"] = {}
                            detection_updates["hsv_thresholds"][key] = hsv_upper[i]

            # Update other parameters
            backend_mappings = {
                "canny_low_threshold": ["edge_detection", "canny_low_threshold"],
                "canny_high_threshold": ["edge_detection", "canny_high_threshold"],
                "hough_threshold": ["hough_parameters", "hough_threshold"],
                "hough_min_line_length": ["hough_parameters", "hough_min_line_length"],
                "hough_max_line_gap": ["hough_parameters", "hough_max_line_gap"],
                "min_confidence": ["confidence_settings", "min_confidence"],
                "roi_processing_mode": ["roi_processing", "roi_processing_mode"]
            }

            for backend_key, client_path in backend_mappings.items():
                if backend_key in backend_line_config:
                    backend_value = backend_line_config[backend_key]
                    current_value = line_config

                    for path_key in client_path:
                        current_value = current_value.get(path_key, {})

                    diff = abs(float(current_value) - float(backend_value))
                    if diff > tolerance * max(1.0, abs(backend_value)):
                        if len(client_path) == 1:
                            detection_updates[client_path[0]] = backend_value
                        else:
                            if client_path[0] not in detection_updates:
                                detection_updates[client_path[0]] = {}
                            detection_updates[client_path[0]][client_path[1]] = backend_value

            # Apply updates if any
            if detection_updates:
                updates = {"detection": detection_updates}
                result = self.update_line_detection_config(updates)
                if not result[0]:
                    return result
                return True, f"Synchronized {len(detection_updates)} parameters with backend"
            else:
                return True, "Configuration is already synchronized with backend"

        except Exception as e:
            error_msg = f"Error syncing with backend: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg


# Convenience functions
def load_line_detection_config(config_path: Optional[str] = None) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Convenience function to load line detection configuration

    Args:
        config_path: Configuration file path

    Returns:
        Tuple[bool, str, Optional[Dict]]: (success, message, config_data)
    """
    manager = LineDetectionConfigManager(config_path)
    success, message, config_data = manager.load_config()
    if success:
        return True, message, manager.get_line_detection_config()
    return success, message, None


def save_line_detection_config(config_updates: Dict[str, Any], config_path: Optional[str] = None) -> Tuple[bool, str]:
    """
    Convenience function to save line detection configuration updates

    Args:
        config_updates: Configuration updates
        config_path: Configuration file path

    Returns:
        Tuple[bool, str]: (success, message)
    """
    manager = LineDetectionConfigManager(config_path)
    manager.load_config()  # Load existing config
    success, message = manager.update_line_detection_config(config_updates)
    if not success:
        return success, message

    return manager.save_config()


if __name__ == "__main__":
    # Test the configuration manager
    logging.basicConfig(level=logging.INFO)

    manager = LineDetectionConfigManager()

    # Test loading configuration
    print("Testing configuration loading...")
    success, message, config_data = manager.load_config()
    print(f"Load result: {success}")
    print(f"Message: {message}")

    if success:
        # Test validation
        print("\nTesting configuration validation...")
        validation_result = manager.validate_config(config_data)
        print(f"Validation result: {validation_result}")

        # Test backup creation
        print("\nTesting backup creation...")
        backup_result = manager.create_backup()
        print(f"Backup result: {backup_result}")

        # Test export
        print("\nTesting configuration export...")
        export_result = manager.export_config("./test_export.json")
        print(f"Export result: {export_result}")

        print("\nLine detection configuration loaded successfully!")
    else:
        print("Failed to load configuration!")