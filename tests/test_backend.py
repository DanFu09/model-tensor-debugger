#!/usr/bin/env python3
"""Test the backend dual .pth processing"""
import requests
import os

# Test uploading the two .pth files to the dual .pth endpoint
def test_dual_pth_upload():
    url = 'http://127.0.0.1:5000/dual_pth_upload'
    
    # Prepare files for upload
    files = {
        'model1': ('test_tensor_1.pth', open('test_tensor_1.pth', 'rb'), 'application/octet-stream'),
        'model2': ('test_tensor_2.pth', open('test_tensor_2.pth', 'rb'), 'application/octet-stream')
    }
    
    try:
        print("Sending request to dual .pth upload endpoint...")
        response = requests.post(url, files=files)
        
        print(f"Response status: {response.status_code}")
        print(f"Response content: {response.text[:500]}...")  # First 500 chars
        
        if response.status_code == 200:
            data = response.json()
            print(f"Success! Found {len(data['matches'])} matches")
            for i, match in enumerate(data['matches']):
                print(f"  Match {i}: {match['stage_display']}")
                print(f"    Shapes: {match['model1_data']['shape']} vs {match['model2_data']['shape']}")
        else:
            print(f"Error: {response.status_code}")
            
    except Exception as e:
        print(f"Request failed: {e}")
    finally:
        # Close files
        files['model1'][1].close()
        files['model2'][1].close()

if __name__ == '__main__':
    test_dual_pth_upload()