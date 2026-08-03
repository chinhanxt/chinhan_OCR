# Real-Time SSE Streaming & Stitch Blueprint UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Server-Sent Events (SSE) streaming API (`POST /v1/ocr/stream`) and Stitch Blueprint drawing UI animation to optimize multi-page PDF processing from 237s latency down to 3-4s first-page streaming response time.

**Architecture:** Add an async generator `POST /v1/ocr/stream` endpoint in FastAPI that yields SSE events (`event: page_data`) per PDF page as PyTorch GPU finishes inference. In the frontend, connect using `EventSource` / `fetch` reader and animate each incoming page container with a neon cyan Stitch Blueprint outline draw effect.

**Tech Stack:** FastAPI `StreamingResponse`, PyMuPDF (`fitz`), PyTorch GPU, Vanilla CSS `@keyframes stitchBorderDraw`, HTML5 marked.js.

## Global Constraints

- **Model Invariance**: Retain `prompt='<image>document parsing.'`, `crop_mode`, `image_size`, `no_repeat_ngram_size=35`, `ngram_window` verbatim.
- **Port Mapping**: Docker container runs on port `8000`, mapped to host port `3000`.

---

### Task 1: Add Real-Time SSE Streaming API (`POST /v1/ocr/stream`) in `app.py`

**Files:**
- Modify: `docker_unlimited_ocr/app.py`

**Interfaces:**
- Consumes: `UploadFile`, `run_ocr_on_single_image()`, `clean_ocr_to_markdown()`.
- Produces: `POST /v1/ocr/stream` returning `StreamingResponse(content, media_type="text/event-stream")`.

- [ ] **Step 1: Write test for streaming SSE endpoint in `test_stream_endpoint.py`**

```python
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
```

- [ ] **Step 2: Add `POST /v1/ocr/stream` endpoint to `app.py`**

```python
from fastapi.responses import StreamingResponse
import json

@app.post("/v1/ocr/stream")
async def process_ocr_stream(
    file: UploadFile = File(...),
    mode: str = Query("gundam"),
    max_length: int = Query(32768)
):
    active_model, active_tokenizer = get_model()
    content = await file.read()
    filename = file.filename or "uploaded_file"
    is_pdf = content.startswith(b"%PDF-") or filename.lower().endswith(".pdf")

    ext = ".pdf" if is_pdf else ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    async def event_generator():
        start_time = asyncio.get_event_loop().time()
        try:
            if is_pdf:
                doc = fitz.open(tmp_path)
                total_pages = len(doc)
                for i in range(total_pages):
                    page = doc[i]
                    pix = page.get_pixmap(dpi=150)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_p{i+1}.png") as page_tmp:
                        pix.save(page_tmp.name)
                        page_path = page_tmp.name
                    try:
                        raw_text = await asyncio.to_thread(run_ocr_on_single_image, page_path, mode, max_length, active_model, active_tokenizer)
                        clean_md = clean_ocr_to_markdown(raw_text)
                        elapsed = round(asyncio.get_event_loop().time() - start_time, 2)
                        data = {
                            "page_index": i + 1,
                            "total_pages": total_pages,
                            "raw_text": raw_text,
                            "clean_markdown": clean_md,
                            "elapsed_seconds": elapsed,
                            "is_complete": False
                        }
                        yield f"event: page_data\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
                    finally:
                        if os.path.exists(page_path):
                            os.remove(page_path)
            else:
                raw_text = await asyncio.to_thread(run_ocr_on_single_image, tmp_path, mode, max_length, active_model, active_tokenizer)
                clean_md = clean_ocr_to_markdown(raw_text)
                elapsed = round(asyncio.get_event_loop().time() - start_time, 2)
                data = {
                    "page_index": 1,
                    "total_pages": 1,
                    "raw_text": raw_text,
                    "clean_markdown": clean_md,
                    "elapsed_seconds": elapsed,
                    "is_complete": True
                }
                yield f"event: page_data\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

            total_elapsed = round(asyncio.get_event_loop().time() - start_time, 2)
            yield f"event: complete\ndata: {json.dumps({'status': 'finished', 'total_seconds': total_elapsed})}\n\n"
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

- [ ] **Step 3: Test SSE streaming endpoint via python test**

Run: `python3 test_stream_endpoint.py`
Expected: PASS with "SSE Stream test passed!"

---

### Task 2: Implement Stitch Blueprint UI Animation & SSE Frontend Streaming in `app.py`

**Files:**
- Modify: `docker_unlimited_ocr/app.py` (`web_ui_index`)

**Interfaces:**
- Consumes: `POST /v1/ocr/stream` SSE stream via `fetch` ReadableStream.
- Produces: Dynamic Stitch Blueprint page containers with `@keyframes stitchBorderDraw` and scanning laser effect.

- [ ] **Step 1: Add Stitch CSS Animations to `web_ui_index`**

```css
@keyframes stitchBorderDraw {
    0% {
        box-shadow: 0 0 0 0 rgba(37, 99, 235, 0.4), inset 0 0 0 2px #3b82f6;
        border-color: #2563eb;
    }
    50% {
        box-shadow: 0 0 20px 4px rgba(6, 182, 212, 0.4), inset 0 0 12px 2px #06b6d4;
        border-color: #06b6d4;
    }
    100% {
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.05);
        border-color: #e2e8f0;
    }
}

.stitch-card {
    position: relative;
    border: 2px solid #3b82f6;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    background: #ffffff;
    animation: stitchBorderDraw 2s ease-out forwards;
    overflow: hidden;
}

.stitch-laser-scan {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 3px;
    background: linear-gradient(90deg, transparent, #06b6d4, #3b82f6, transparent);
    box-shadow: 0 0 10px #06b6d4;
    animation: laserScan 2.5s ease-in-out infinite;
    pointer-events: none;
}

@keyframes laserScan {
    0% { top: 0%; opacity: 1; }
    50% { top: 98%; opacity: 1; }
    100% { top: 0%; opacity: 0.2; }
}
```

- [ ] **Step 2: Update `executeOCR()` JS function in `web_ui_index` to handle `/v1/ocr/stream` SSE events**

```javascript
async function executeOCR() {
    if (!currentFile) {
        alert("Vui lòng chọn tệp Ảnh hoặc tệp PDF trước khi bấm trích xuất!");
        return;
    }

    const btn = document.getElementById('submitBtn');
    const btnText = document.getElementById('btnText');
    const btnSpinner = document.getElementById('btnSpinner');
    const statusMsg = document.getElementById('statusMsg');

    btn.disabled = true;
    btnSpinner.style.display = 'inline-block';
    btnText.innerText = '⏳ STITCH AI ĐANG TRÍCH XUẤT...';
    statusMsg.innerText = '⏳ Đang truyền dữ liệu và vẽ trang tài liệu...';

    startTimer();

    const formData = new FormData();
    formData.append('file', currentFile);

    const previewArea = document.getElementById('previewArea');
    const cleanTextArea = document.getElementById('cleanText');
    const rawTextArea = document.getElementById('rawText');

    previewArea.innerHTML = '';
    cleanTextArea.value = '';
    rawTextArea.value = '';

    try {
        const response = await fetch(`/v1/ocr/stream?mode=${currentMode}`, {
            method: 'POST',
            body: formData
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });

            const eventBlocks = buffer.split('\n\n');
            buffer = eventBlocks.pop();

            for (const block of eventBlocks) {
                if (!block.trim()) continue;
                const lines = block.split('\n');
                let eventName = '';
                let dataStr = '';

                for (const line of lines) {
                    if (line.startsWith('event: ')) eventName = line.slice(7).trim();
                    if (line.startsWith('data: ')) dataStr = line.slice(6).trim();
                }

                if (eventName === 'page_data' && dataStr) {
                    const pageObj = JSON.parse(dataStr);
                    appendPageStitchUI(pageObj);
                } else if (eventName === 'complete') {
                    const totalSec = stopTimer();
                    statusMsg.innerText = `✨ Hoàn tất toàn bộ tài liệu trong ${totalSec} giây!`;
                }
            }
        }
    } catch (err) {
        stopTimer();
        alert("Lỗi kết nối Server Stream: " + err.message);
    } finally {
        btn.disabled = false;
        btnSpinner.style.display = 'none';
        btnText.innerText = '🚀 BẮT ĐẦU TRÍCH XUẤT OCR';
    }
}

function appendPageStitchUI(pageObj) {
    const previewArea = document.getElementById('previewArea');
    const cleanTextArea = document.getElementById('cleanText');
    const rawTextArea = document.getElementById('rawText');

    // Tab 1: Stitch Blueprint Card
    const card = document.createElement('div');
    card.className = 'stitch-card';
    card.id = `page-card-${pageObj.page_index}`;
    card.innerHTML = `
        <div class="stitch-laser-scan"></div>
        <div class="page-break-divider">📄 TRANG ${pageObj.page_index} / ${pageObj.total_pages} (${pageObj.elapsed_seconds}s)</div>
        <div class="stitch-content">${marked.parse(pageObj.clean_markdown)}</div>
    `;
    previewArea.appendChild(card);
    previewArea.scrollTop = previewArea.scrollHeight;

    // Tab 2 & 3: Append text
    cleanTextArea.value += `\n*— Trang ${pageObj.page_index} —*\n` + pageObj.clean_markdown + '\n';
    cleanTextArea.scrollTop = cleanTextArea.scrollHeight;

    rawTextArea.value += `--- Page ${pageObj.page_index} ---\n` + pageObj.raw_text + '\n';
    rawTextArea.scrollTop = rawTextArea.scrollHeight;

    // Remove laser scan after animation completes
    setTimeout(() => {
        const laser = card.querySelector('.stitch-laser-scan');
        if (laser) laser.remove();
    }, 4000);
}
```

- [ ] **Step 3: Restart docker container & test in browser**

Run: `docker compose restart`
Expected: Container restarts, streaming endpoint live on `http://192.168.92.139:3000/`.
