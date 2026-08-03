import requests

def test_stream_api():
    url = "http://localhost:3000/v1/ocr/stream?mode=gundam"
    files = {"file": ("test.png", open("/home/ai_server_1/Unlimited-OCR-main/assets/baidu.png", "rb"), "image/png")}
    response = requests.post(url, files=files, stream=True)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")
    lines = []
    for line in response.iter_lines():
        if line:
            lines.append(line.decode("utf-8"))
    assert any("event: page_data" in l or "page_index" in l for l in lines)
    print("SSE Stream test passed!")

if __name__ == "__main__":
    test_stream_api()
