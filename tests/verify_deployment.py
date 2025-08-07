#!/usr/bin/env python3
"""
Verify that the application can run in a Vercel-like environment
Run this locally to test before deployment
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all required imports work"""
    try:
        import flask
        import torch
        import numpy
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_app_creation():
    """Test that the Flask app can be created"""
    try:
        from app import app
        print("✅ Flask app created successfully")
        print(f"   Template folder: {app.template_folder}")
        print(f"   Max content length: {app.config.get('MAX_CONTENT_LENGTH', 'Not set')}")
        return True
    except Exception as e:
        print(f"❌ App creation failed: {e}")
        return False

def test_api_wrapper():
    """Test that the Vercel API wrapper works"""
    try:
        from api.app import handler
        print("✅ Vercel API wrapper loaded successfully")
        return True
    except Exception as e:
        print(f"❌ API wrapper failed: {e}")
        return False

def test_routes():
    """Test that key routes are available"""
    try:
        from app import app
        with app.test_client() as client:
            # Test health endpoint
            response = client.get('/health')
            if response.status_code == 200:
                print("✅ Health endpoint working")
            else:
                print(f"❌ Health endpoint failed: {response.status_code}")
                return False
                
            # Test main index (should return HTML)
            response = client.get('/')
            if response.status_code == 200:
                print("✅ Main index working")
            else:
                print(f"❌ Main index failed: {response.status_code}")
                return False
                
        return True
    except Exception as e:
        print(f"❌ Route testing failed: {e}")
        return False

def test_template():
    """Test that templates can be found"""
    try:
        template_path = os.path.join(os.path.dirname(__file__), 'templates', 'index.html')
        if os.path.exists(template_path):
            print("✅ Template file found")
            return True
        else:
            print(f"❌ Template file not found at: {template_path}")
            return False
    except Exception as e:
        print(f"❌ Template test failed: {e}")
        return False

def main():
    """Run all verification tests"""
    print("🚀 Verifying Vercel deployment readiness...\n")
    
    tests = [
        ("Import Test", test_imports),
        ("App Creation", test_app_creation),
        ("API Wrapper", test_api_wrapper),
        ("Routes Test", test_routes),
        ("Template Test", test_template),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
        
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ready for Vercel deployment.")
        return True
    else:
        print(f"❌ {total - passed} test(s) failed. Fix issues before deploying.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)