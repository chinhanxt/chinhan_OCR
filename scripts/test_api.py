import requests
import sys

import os
API_URL = os.getenv("OCR_API_URL", "http://localhost:3000/v1/ocr")

def test_ocr(image_path, mode="gundam"):
    print(f"Sending request to {API_URL} for image: {image_path}...")
    
    import os
    with open(image_path, "rb") as f:
        files = {"file": (os.path.basename(image_path), f, "image/png")}
        params = {"mode": mode}
        response = requests.post(API_URL, files=files, params=params)
        
    if response.status_code == 200:
        data = response.json()
        print("\n--- OCR Result ---")
        print(data.get("parsed_text", ""))
    else:
        print(f"Error {response.status_code}: {response.text}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_api.py <path_to_image.jpg> [gundam|base]")
    else:
        img = sys.argv[1]
        m = sys.argv[2] if len(sys.argv) > 2 else "gundam"
        test_ocr(img, m)
