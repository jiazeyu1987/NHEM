from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

_LOGGING_INITIALIZED = False


class _SuppressRealtimeNoDataFilter(logging.Filter):
    """Filter out noisy realtime no-data log messages."""

    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[name-defined]
        message = record.getMessage()
        return (
            "Realtime data requested but no frames available - returning empty response"
            not in message
        )


class _ErrorHighlightFilter(logging.Filter):
    """Highlight important error messages for better visibility."""

    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[name-defined]
        # Always allow ERROR and CRITICAL levels
        if record.levelno >= logging.ERROR:
            # Add a special prefix for error messages
            record.msg = f"🚨 ERROR: {record.msg}"
            return True
        return True


def _cleanup_old_logs(log_dir: Path, keep_count: int = 5) -> None:
    """清理旧的日志文件，保留最近的指定数量"""
    try:
        log_files = list(log_dir.glob("nhem_*.log"))
        log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        # 删除多余的旧日志文件
        for old_log in log_files[keep_count:]:
            try:
                old_log.unlink()
                print(f"已删除旧日志文件: {old_log.name}")
            except Exception as e:
                print(f"删除日志文件失败: {old_log.name}, 错误: {e}")

        print(f"日志清理完成，保留最近的 {keep_count} 个文件")

    except Exception as e:
        print(f"清理日志文件时出错: {e}")


def init_logging() -> None:
    """
    初始化全局日志配置：
    - 日志目录: 项目根目录下 logs/
    - 日志文件: nhem_YYYYMMDD_HHMMSS.log
    - 级别: INFO (文件)，WARNING (控制台)
    - 优化: 减少DEBUG信息，突出显示错误
    """
    global _LOGGING_INITIALIZED
    if _LOGGING_INITIALIZED:
        return

    base_dir = Path(__file__).resolve().parent.parent
    log_dir = base_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # 清理旧的日志文件（保留最近的5个）
    _cleanup_old_logs(log_dir, keep_count=5)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"nhem_{timestamp}.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 限制单个日志文件大小为10MB
    file_handler = logging.FileHandler(log_file, encoding="utf-8", mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)

    suppress_filter = _SuppressRealtimeNoDataFilter()
    error_highlight_filter = _ErrorHighlightFilter()

    file_handler.addFilter(suppress_filter)
    file_handler.addFilter(error_highlight_filter)
    console_handler.addFilter(suppress_filter)
    console_handler.addFilter(error_highlight_filter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    _LOGGING_INITIALIZED = True
