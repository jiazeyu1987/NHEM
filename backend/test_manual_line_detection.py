#!/usr/bin/env python3
"""
Test script for the new manual line detection endpoint
"""

import json
import base64
import requests
import time

def test_manual_line_detection_endpoint():
    """Test the POST /api/roi/line-intersection endpoint"""

    # Server configuration
    BASE_URL = "http://localhost:8421"
    ENDPOINT = "/api/roi/line-intersection"
    PASSWORD = "31415"

    print(f"Testing manual line detection endpoint at {BASE_URL}{ENDPOINT}")

    # Test case 1: Invalid password
    print("\n=== Test 1: Invalid password ===")
    test_request = {
        "password": "wrong_password",
        "roi_coordinates": {
            "x1": 100,
            "y1": 100,
            "x2": 200,
            "y2": 200
        },
        "force_refresh": False,
        "include_debug_info": True
    }

    try:
        response = requests.post(f"{BASE_URL}{ENDPOINT}", json=test_request, timeout=10)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Success: {result.get('success')}")
            print(f"Message: {result.get('message')}")
        else:
            print(f"Error Response: {response.text}")
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Make sure the backend is running on localhost:8421")
        return False
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

    # Test case 2: Missing input data
    print("\n=== Test 2: Missing input data ===")
    test_request = {
        "password": PASSWORD,
        "force_refresh": False,
        "include_debug_info": True
    }

    try:
        response = requests.post(f"{BASE_URL}{ENDPOINT}", json=test_request, timeout=10)
        print(f"Status Code: {response.status_code}")
        result = response.json()
        print(f"Success: {result.get('success')}")
        print(f"Message: {result.get('message')}")
    except Exception as e:
        print(f"❌ Request failed: {e}")

    # Test case 3: Valid ROI coordinates (this will test the endpoint structure)
    print("\n=== Test 3: Valid ROI coordinates ===")
    test_request = {
        "password": PASSWORD,
        "roi_coordinates": {
            "x1": 1480,
            "y1": 480,
            "x2": 1580,
            "y2": 580
        },
        "force_refresh": False,
        "include_debug_info": True
    }

    try:
        response = requests.post(f"{BASE_URL}{ENDPOINT}", json=test_request, timeout=30)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: {result.get('success')}")
            print(f"Message: {result.get('message')}")
            if result.get('result'):
                detection_result = result['result']
                print(f"Detection Result:")
                print(f"  - Has intersection: {detection_result.get('has_intersection')}")
                print(f"  - Intersection point: {detection_result.get('intersection')}")
                print(f"  - Confidence: {detection_result.get('confidence'):.3f}")
                print(f"  - Processing time: {detection_result.get('processing_time_ms'):.2f}ms")

            if result.get('processing_info'):
                proc_info = result['processing_info']
                print(f"Processing Info:")
                print(f"  - Input mode: {proc_info.get('input_mode')}")
                print(f"  - Total time: {proc_info.get('total_time_ms', 0):.2f}ms")
                print(f"  - Detection time: {proc_info.get('detection_time_ms', 0):.2f}ms")
        else:
            print(f"❌ Error Response: {response.text}")
    except Exception as e:
        print(f"❌ Request failed: {e}")

    print("\n=== Test completed ===")
    return True

def test_endpoint_exists():
    """Test if the endpoint exists by checking health and endpoint availability"""

    BASE_URL = "http://localhost:8421"

    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running and accessible")
            return True
        else:
            print(f"❌ Server returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Server is not running on localhost:8421")
        return False
    except Exception as e:
        print(f"❌ Failed to connect to server: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing NHEM Manual Line Detection Endpoint")
    print("=" * 50)

    if test_endpoint_exists():
        test_manual_line_detection_endpoint()
    else:
        print("\n❌ Please start the NHEM backend server first:")
        print("   cd D:\\ProjectPackage\\NHEM\\backend")
        print("   python run.py")