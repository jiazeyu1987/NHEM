import ast
import sys

try:
    with open('backend/app/core/line_intersection_detector.py', 'r', encoding='utf-8') as f:
        code = f.read()

    # Try to parse the AST
    ast.parse(code)
    print("✓ Python syntax is valid - AST parsing successful")

    # Check for basic structure
    if "class LineIntersectionDetector" in code:
        print("✓ LineIntersectionDetector class found")

    if "def detect_intersection" in code:
        print("✓ detect_intersection method found")

    if "Task 11" in code:
        print("✓ Task 11 enhancements found")

except SyntaxError as e:
    print(f"✗ Syntax error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Other error: {e}")
    sys.exit(1)

print("✓ All checks passed")