#!/usr/bin/env python3
"""
Test the serverless function locally without Vercel CLI
This simulates how Vercel would run the function
"""

import os
import sys
import tempfile
from werkzeug.test import Client

# Handle different Werkzeug versions
try:
    from werkzeug.wrappers import BaseResponse
except ImportError:
    from werkzeug.wrappers import Response as BaseResponse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_serverless_function():
    """Test the serverless function wrapper"""
    print("🧪 Testing Serverless Function Locally")
    print("=" * 50)
    
    # Test 1: Import the handler
    try:
        from api.app import handler
        print("✅ Serverless handler imported successfully")
        app = handler
    except Exception as e:
        print(f"❌ Failed to import handler: {e}")
        return False
    
    # Test 2: Create test client
    try:
        client = Client(app, BaseResponse)
        print("✅ Test client created successfully")
    except Exception as e:
        print(f"❌ Failed to create test client: {e}")
        return False
    
    # Test 3: Health endpoint
    try:
        response = client.get('/health')
        if response.status_code == 200:
            print("✅ Health endpoint working")
            data = response.get_json()
            print(f"   Response: {data}")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")
        return False
    
    # Test 4: Main page
    try:
        response = client.get('/')
        if response.status_code == 200:
            print("✅ Main page loads successfully")
            content_length = len(response.data)
            print(f"   Response size: {content_length} bytes")
            # Check if it contains expected HTML elements
            content = response.data.decode('utf-8')
            if 'ML Model Tensor Debugger' in content:
                print("   Contains expected title ✅")
            else:
                print("   Missing expected title ❌")
        else:
            print(f"❌ Main page failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Main page error: {e}")
        return False
    
    # Test 5: Static content check
    try:
        response = client.get('/')
        content = response.data.decode('utf-8')
        
        checks = [
            ('CSS styles', '<style>' in content or 'stylesheet' in content),
            ('JavaScript', '<script>' in content),
            ('Upload form', 'upload' in content.lower()),
            ('File input', 'input' in content and 'file' in content),
        ]
        
        for check_name, result in checks:
            status = "✅" if result else "❌"
            print(f"   {status} {check_name}")
            
    except Exception as e:
        print(f"❌ Content check error: {e}")
    
    # Test 6: POST endpoint (simulate file upload structure)
    try:
        # Test upload endpoint with no files (should get error but not crash)
        response = client.post('/upload', data={})
        # Should return an error but not crash
        if response.status_code in [400, 500]:  # Expected error codes
            print("✅ Upload endpoint responds to invalid requests properly")
        else:
            print(f"⚠️  Upload endpoint unexpected response: {response.status_code}")
    except Exception as e:
        print(f"❌ Upload endpoint error: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 Local serverless function test completed!")
    print("\nTo run with Vercel dev server:")
    print("1. Run: vercel login")
    print("2. Run: vercel dev")
    print("3. Visit: http://localhost:3000")
    
    return True

def test_environment():
    """Test the environment setup"""
    print("\n🔧 Environment Check")
    print("-" * 30)
    
    # Check Python path
    print(f"Python path: {sys.executable}")
    
    # Check current directory
    print(f"Current directory: {os.getcwd()}")
    
    # Check key files exist
    files_to_check = [
        'vercel.json',
        'api/app.py', 
        'app.py',
        'templates/index.html',
        'requirements.txt'
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} (missing)")
    
    # Check imports
    try:
        import flask
        import torch
        print("✅ Required packages available")
    except ImportError as e:
        print(f"❌ Import error: {e}")

if __name__ == "__main__":
    test_environment()
    success = test_serverless_function()
    
    if success:
        print("\n🚀 Ready for Vercel deployment!")
    else:
        print("\n❌ Fix issues before deploying to Vercel")
    
    sys.exit(0 if success else 1)