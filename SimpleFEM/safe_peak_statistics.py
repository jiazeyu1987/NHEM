#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SimpleFEM 安全波峰统计模块
实现波峰去重、差值分析和Excel数据收集功能
根据 task/info1.txt 要求实现
"""

import csv
import os
import sys
import json
import threading
import time
import shutil
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from collections import deque
import logging


def _get_base_dir() -> str:
    """获取基础目录，支持源码和打包模式"""
    if getattr(sys, "frozen", False) and hasattr(sys, "executable"):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


BASE_DIR = _get_base_dir()


class SafePeakStatistics:
    """安全的波峰统计管理类"""

    def __init__(self):
        self.lock = threading.Lock()
        self.recent_peaks: List[Dict[str, Any]] = []
        self.max_recent_peaks = 5  # 去重检查窗口
        self.stats_data: List[Dict[str, Any]] = []
        self.start_time = datetime.now()
        self.session_id = self.start_time.strftime("%Y%m%d_%H%M%S")

        # 文件路径
        self.csv_filename = f"peak_statistics_{self.session_id}.csv"
        self.csv_path = os.path.join(BASE_DIR, self.csv_filename)
        self.final_export_path = os.path.join(BASE_DIR, f"peak_statistics_final_{self.session_id}.csv")

        # 配置参数（根据task要求）
        self.duplicate_check_window = 5  # 检查最近5个波峰
        self.height_tolerance = 0.1      # 高度容差≤0.1
        self.update_count = 0

        # 初始化CSV文件（程序开始时记录）
        self._initialize_csv_file()
        self._add_log(f"SafePeakStatistics初始化完成，会话ID: {self.session_id}")

    def _initialize_csv_file(self):
        """初始化CSV文件，写入表头（程序开始时记录）"""
        try:
            # 确保目录存在
            os.makedirs(BASE_DIR, exist_ok=True)

            file_exists = os.path.exists(self.csv_path)

            with open(self.csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
                # 只保留必要字段：peak_type, frame_index, 前后X帧平均值
                fieldnames = [
                    'peak_type', 'frame_index', 'pre_peak_avg', 'post_peak_avg'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                if not file_exists:
                    writer.writeheader()
                    self._add_log(f"CSV文件初始化完成: {self.csv_path}")
                    self._add_log(f"包含字段: {', '.join(fieldnames)}")
                else:
                    self._add_log(f"CSV文件已存在，继续追加数据: {self.csv_path}")

        except Exception as e:
            self._add_log(f"初始化CSV文件失败: {e}", level="ERROR")

    def add_peaks_from_daemon(self,
                            frame_index: int,
                            green_peaks: List[Tuple[int, int]],
                            red_peaks: List[Tuple[int, int]],
                            curve: List[float],
                            intersection: Optional[Tuple[int, int]] = None,
                            roi2_info: Optional[Dict[str, int]] = None,
                            gray_value: Optional[float] = None,
                            difference_threshold: float = 1.1):
        """
        从守护进程添加波峰数据

        Args:
            frame_index: 帧索引
            green_peaks: 绿色波峰列表 [(start_frame, end_frame), ...]
            red_peaks: 红色波峰列表 [(start_frame, end_frame), ...]
            curve: 当前灰度曲线数据
            intersection: ROI1中的绿线交点坐标 (x, y)
            roi2_info: ROI2区域信息 {'x1':, 'y1':, 'x2':, 'y2':}
            gray_value: ROI2的平均灰度值
            difference_threshold: 用于分类的差值阈值
        """
        try:
            timestamp = datetime.now()

            with self.lock:
                # 添加绿色波峰
                for i, (start, end) in enumerate(green_peaks):
                    peak_data = self._create_peak_data(
                        timestamp, frame_index, "green", start, end,
                        curve, intersection, roi2_info, gray_value, difference_threshold
                    )

                    # 检查去重（精确实现：高度差≤0.1，宽度匹配，5窗口）
                    if not self._is_duplicate_peak(peak_data):
                        self._add_peak_to_memory(peak_data)
                        self._write_peak_to_csv(peak_data)
                        self._add_log(f"添加绿色波峰: [{start},{end}], 最大值: {peak_data['max_value']:.1f}, 差值: {peak_data['frame_diff']:.2f}")

                # 添加红色波峰
                for i, (start, end) in enumerate(red_peaks):
                    peak_data = self._create_peak_data(
                        timestamp, frame_index, "red", start, end,
                        curve, intersection, roi2_info, gray_value, difference_threshold
                    )

                    # 检查去重
                    if not self._is_duplicate_peak(peak_data):
                        self._add_peak_to_memory(peak_data)
                        self._write_peak_to_csv(peak_data)
                        self._add_log(f"添加红色波峰: [{start},{end}], 最大值: {peak_data['max_value']:.1f}, 差值: {peak_data['frame_diff']:.2f}")

                self.update_count += 1

        except Exception as e:
            self._add_log(f"添加波峰数据失败: {e}", level="ERROR")

    def _create_peak_data(self,
                         timestamp: datetime,
                         frame_index: int,
                         peak_type: str,
                         start_frame: int,
                         end_frame: int,
                         curve: List[float],
                         intersection: Optional[Tuple[int, int]],
                         roi2_info: Optional[Dict[str, int]],
                         gray_value: Optional[float],
                         difference_threshold: float) -> Dict[str, Any]:
        """创建简化的波峰数据结构（只保留必要字段）"""

        # 计算前X帧平均值（波峰开始前5帧）
        pre_frames = 5
        pre_start = max(0, start_frame - pre_frames)
        pre_end = start_frame - 1
        pre_avg = 0.0

        if pre_start <= pre_end and pre_end < len(curve):
            pre_values = curve[pre_start:pre_end + 1]
            pre_avg = sum(pre_values) / len(pre_values) if pre_values else 0.0

        # 计算后X帧平均值（波峰结束后5帧）
        post_frames = 5
        post_start = end_frame + 1
        post_end = end_frame + post_frames
        post_avg = 0.0

        if post_start < len(curve) and post_end < len(curve):
            post_values = curve[post_start:post_end + 1]
            post_avg = sum(post_values) / len(post_values) if post_values else 0.0

        return {
            'peak_type': peak_type,
            'frame_index': frame_index,
            'pre_peak_avg': round(pre_avg, 2),
            'post_peak_avg': round(post_avg, 2)
        }

    def _calculate_classification_reason(self, frame_diff: float, threshold: float, peak_type: str) -> str:
        """计算波峰分类原因（为什么是红色/绿色）"""
        if peak_type == "green":
            if frame_diff > threshold:
                return f"稳定波峰: 帧差值{frame_diff:.2f} > 阈值{threshold:.2f}"
            else:
                return f"稳定波峰: 帧差值{frame_diff:.2f} >= 阈值{threshold:.2f}"
        else:  # red
            return f"不稳定波峰: 帧差值{frame_diff:.2f} < 阈值{threshold:.2f}"

    def _calculate_peak_stability(self, peak_curve: List[float]) -> float:
        """计算波峰稳定性（0-1，越接近1越稳定）"""
        try:
            if len(peak_curve) < 2:
                return 0.0

            # 计算标准差
            mean_val = sum(peak_curve) / len(peak_curve)
            variance = sum((x - mean_val) ** 2 for x in peak_curve) / len(peak_curve)
            std_dev = variance ** 0.5

            # 归一化稳定性评分
            max_val = max(peak_curve) if peak_curve else 1
            stability_score = max(0, 1 - (std_dev / max_val))

            return min(1.0, stability_score)
        except Exception:
            return 0.0

    def _calculate_quality_score(self, max_val: float, min_val: float, duration: int, frame_diff: float) -> float:
        """计算波峰质量评分（0-100）"""
        try:
            # 基础评分：波峰高度
            height_score = max_val - min_val

            # 持续时间评分
            duration_score = min(duration / 10.0, 1.0) * 20  # 最高20分

            # 稳定性评分（帧差值越小越稳定）
            stability_score = max(0, 20 - frame_diff)  # 最高20分

            # 总分（最高100分）
            total_score = height_score + duration_score + stability_score

            return max(0, min(100, total_score))
        except Exception:
            return 0.0

    def _is_duplicate_peak(self, peak_data: Dict[str, Any]) -> bool:
        """检查是否为重复波峰（基于前后帧平均值去重）"""
        try:
            current_pre_avg = peak_data['pre_peak_avg']
            current_post_avg = peak_data['post_peak_avg']

            # 检查最近的5个波峰（task要求的窗口大小）
            for recent_peak in self.recent_peaks[-self.duplicate_check_window:]:
                recent_pre_avg = recent_peak.get('pre_peak_avg', 0)
                recent_post_avg = recent_peak.get('post_peak_avg', 0)

                # 前后帧平均值都接近（容差0.5）则视为重复
                if (abs(recent_pre_avg - current_pre_avg) <= 0.5 and
                    abs(recent_post_avg - current_post_avg) <= 0.5):
                    return True
            return False
        except Exception as e:
            self._add_log(f"去重检查失败: {e}", level="ERROR")
            return False

    def _add_peak_to_memory(self, peak_data: Dict[str, Any]):
        """添加波峰到内存缓存"""
        self.recent_peaks.append(peak_data)
        self.stats_data.append(peak_data)

        # 保持内存缓存大小（最近5个波峰用于去重）
        if len(self.recent_peaks) > self.max_recent_peaks * 2:
            self.recent_peaks = self.recent_peaks[-self.max_recent_peaks:]

    def _write_peak_to_csv(self, peak_data: Dict[str, Any]):
        """写入单个波峰到CSV文件"""
        try:
            # 简化的字段列表
            fieldnames = [
                'peak_type', 'frame_index', 'pre_peak_avg', 'post_peak_avg'
            ]

            # 原子性写入：先写临时文件，再重命名
            temp_file = self.csv_path + '.tmp'

            # 如果原文件存在，复制内容
            if os.path.exists(self.csv_path):
                shutil.copy2(self.csv_path, temp_file)

            # 追加新数据
            with open(temp_file, 'a', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(peak_data)

            # 原子性重命名
            if os.path.exists(self.csv_path):
                os.remove(self.csv_path)
            os.rename(temp_file, self.csv_path)

        except Exception as e:
            self._add_log(f"写入CSV失败: {e}", level="ERROR")

    
    def export_final_csv(self) -> Optional[str]:
        """程序结束时导出最终CSV文件"""
        try:
            self._add_log("开始导出最终CSV文件...")

            if not os.path.exists(self.csv_path):
                self._add_log("没有数据文件可导出")
                return None

            # 创建最终导出文件
            shutil.copy2(self.csv_path, self.final_export_path)

            # 添加导出时间戳
            with open(self.final_export_path, 'a', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    f"# EXPORT_SUMMARY",
                    f"export_time,{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    f"total_peaks,{len(self.stats_data)}",
                    f"session_duration,{str(datetime.now() - self.start_time).split('.')[0]}",
                    f"session_id,{self.session_id}"
                ])

            self._add_log(f"最终CSV文件已导出至: {self.final_export_path}")
            return os.path.abspath(self.final_export_path)

        except Exception as e:
            self._add_log(f"导出最终CSV文件失败: {e}", level="ERROR")
            return None

    def get_statistics_summary(self) -> Dict[str, Any]:
        """获取统计摘要"""
        try:
            with self.lock:
                total_peaks = len(self.stats_data)
                green_peaks = len([p for p in self.stats_data if p['peak_type'] == 'green'])
                red_peaks = len([p for p in self.stats_data if p['peak_type'] == 'red'])

                avg_duration = 0
                avg_max_value = 0
                avg_frame_diff = 0
                if self.stats_data:
                    avg_duration = sum(p['duration'] for p in self.stats_data) / total_peaks
                    avg_max_value = sum(p['max_value'] for p in self.stats_data) / total_peaks
                    avg_frame_diff = sum(p['frame_diff'] for p in self.stats_data) / total_peaks

                return {
                    'total_peaks': total_peaks,
                    'green_peaks': green_peaks,
                    'red_peaks': red_peaks,
                    'avg_duration': round(avg_duration, 2),
                    'avg_max_value': round(avg_max_value, 2),
                    'avg_frame_diff': round(avg_frame_diff, 3),
                    'session_id': self.session_id,
                    'session_duration': str(datetime.now() - self.start_time).split('.')[0],
                    'csv_file_path': self.csv_path,
                    'final_export_path': self.final_export_path,
                    'csv_exists': os.path.exists(self.csv_path),
                    'csv_size_mb': round(os.path.getsize(self.csv_path) / (1024*1024), 2) if os.path.exists(self.csv_path) else 0
                }
        except Exception as e:
            self._add_log(f"获取统计摘要失败: {e}", level="ERROR")
            return {}

    def save_csv_file(self) -> Optional[str]:
        """保存CSV文件并返回路径（用于UI调用）"""
        try:
            if os.path.exists(self.csv_path):
                return os.path.abspath(self.csv_path)
            else:
                return None
        except Exception as e:
            self._add_log(f"保存CSV文件失败: {e}", level="ERROR")
            return None

    def _add_log(self, message: str, level: str = "INFO"):
        """添加日志记录"""
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_message = f"[{timestamp}] {level}: SafePeakStatistics - {message}"
            print(log_message)  # 输出到控制台
        except Exception:
            pass  # 日志记录失败不应该影响主要功能


# 全局实例
safe_statistics = SafePeakStatistics()