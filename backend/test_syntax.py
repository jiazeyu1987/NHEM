#!/usr/bin/env python3
"""
Simple script to test the syntax of the routes.py file
"""
import ast
import sys

def test_syntax():
    try:
        with open('app/api/routes.py', 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse the AST
        ast.parse(content)
        print("✅ Syntax is valid")
        return True

    except SyntaxError as e:
        print(f"❌ Syntax error: {e}")
        print(f"   Line {e.lineno}: {e.text}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_syntax()
    sys.exit(0 if success else 1)