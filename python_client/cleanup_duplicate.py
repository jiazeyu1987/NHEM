#!/usr/bin/env python3
"""
Clean up duplicate methods in line_detection_widget.py
"""

import os

def cleanup_file():
    """Remove duplicate methods and fix the file"""

    input_file = 'line_detection_widget.py'
    temp_file = 'line_detection_widget_fixed.py'

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find LineDetectionWidget class end (around line 3309-3310)
    widget_end = None
    for i in range(3300, 3320):
        if '# ============ Task 32:' in lines[i]:
            widget_end = i
            break

    if widget_end is None:
        print("Could not find LineDetectionWidget class end")
        return

    # Find ClientErrorHandler class start (around line 3384)
    clienthandler_start = None
    for i in range(3370, 3390):
        if 'class ClientErrorHandler:' in lines[i]:
            clienthandler_start = i
            break

    if clienthandler_start is None:
        print("Could not find ClientErrorHandler class start")
        return

    # Create the fixed file: keep first widget_end lines + skip duplicate + keep from clienthandler_start to end
    with open(temp_file, 'w', encoding='utf-8') as f:
        # Write LineDetectionWidget part
        f.writelines(lines[:widget_end + 1])

        # Write ClientErrorHandler and rest
        f.writelines(lines[clienthandler_start:])

    # Replace the original file
    os.replace(temp_file, input_file)
    print(f"Fixed file: removed duplicate methods from line {widget_end} to {clienthandler_start}")

if __name__ == "__main__":
    cleanup_file()