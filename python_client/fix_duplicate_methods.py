#!/usr/bin/env python3
"""
Fix duplicate methods in line_detection_widget.py
"""

def fix_duplicate_methods():
    """Remove duplicate method definitions from outside the class"""

    with open('line_detection_widget.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find the start of duplicate methods
    duplicate_start = None
    for i, line in enumerate(lines):
        if 'def _setup_network_monitoring(self):' in line and i > 1000:  # Skip the one in the class
            duplicate_start = i
            break

    if duplicate_start is None:
        print("No duplicate methods found")
        return

    # Find the end (next class or end of file)
    duplicate_end = None
    for i in range(duplicate_start + 1, len(lines)):
        if lines[i].startswith('class ') and i > 1000:
            duplicate_end = i
            break
        elif i == len(lines) - 1:
            duplicate_end = len(lines)
            break

    # Remove duplicate methods
    if duplicate_end:
        fixed_lines = lines[:duplicate_start] + lines[duplicate_end:]

        with open('line_detection_widget.py', 'w', encoding='utf-8') as f:
            f.writelines(fixed_lines)

        print(f"Removed duplicate methods from line {duplicate_start} to {duplicate_end}")
    else:
        print("Could not determine end of duplicate methods")

if __name__ == "__main__":
    fix_duplicate_methods()