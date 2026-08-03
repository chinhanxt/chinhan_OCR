# Design Specification: Real-Time SSE Streaming & Stitch Blueprint UI for Unlimited-OCR

**Date:** 2026-08-01  
**Target System:** Unlimited-OCR Docker FastAPI GPU Server & Light Theme Web UI  
**Goal:** Supercharge multi-page PDF processing speed from 237s latency down to **3-4s first-page streaming response** using Server-Sent Events (SSE), GPU batch parallelization, and a futuristic Stitch Blueprint drawing animation.

---

## 1. High-Level Architecture & Speed Optimization

### 1.1 Problem Statement
Currently, multi-page PDF files (e.g. 15MB PDFs with 10-15 pages) are processed sequentially, and the API holds the HTTP connection open until **all pages finish processing** (taking 237.86 seconds total). The user sees no output until the entire file is done.

### 1.2 Proposed Architecture Solution
1. **Server-Sent Events (SSE) Endpoint (`POST /v1/ocr/stream`)**:
   - Streams JSON events per page as soon as PyTorch GPU finishes processing each page/batch.
   - Reduces initial page latency from 237s to **~3.5 seconds**.

2. **Parallel CPU Page Conversion & GPU Batch Execution**:
   - `PyMuPDF (fitz)` renders PDF pages into images using multi-threaded execution (`ThreadPoolExecutor`).
   - PyTorch GPU inference runs batched / queued execution on NVIDIA RTX 5060 Ti CUDA streams.

3. **3-Stage Real-Time Pipeline**:
   - **Stage 1 (Extraction)**: Model outputs raw text with layout bounding tags `<|det|>`.
   - **Stage 2 (Transformation)**: `clean_ocr_to_markdown()` strips `<|det|>` tags and formats headers/tables.
   - **Stage 3 (Streaming Event)**: FastAPI yields SSE data chunk to client.

---

## 2. API Contract & Data Schemas

### Endpoint: `POST /v1/ocr/stream`

**Query Parameters:**
- `mode`: `"gundam"` (640px cropped mode) or `"base"` (1024px full-page mode).
- `max_length`: Output token limit (default: `32768`).

**Request Body:** `multipart/form-data` with `file` field (Image or PDF).

**Response Content-Type:** `text/event-stream`

#### SSE Events:

1. **`page_data` Event**:
```json
event: page_data
data: {
  "page_index": 1,
  "total_pages": 15,
  "raw_text": "<|det|>title [10, 20, 30, 40]<|/det|>HUTECH",
  "clean_markdown": "# HUTECH",
  "elapsed_seconds": 3.42,
  "is_complete": false
}
```

2. **`complete` Event**:
```json
event: complete
data: {
  "status": "finished",
  "total_pages": 15,
  "total_seconds": 42.10
}
```

3. **`error` Event**:
```json
event: error
data: {
  "error": "Detailed error message"
}
```

---

## 3. Frontend Web UI & Stitch Blueprint Drawing Effect

### 3.1 Stitch-Style Blueprint Border Animation (`@keyframes stitchBorderDraw`)
- Each newly received page container renders with a neon cyan glowing border animation (`#2563eb` to `#06b6d4`) that sweeps around the perimeter of the page card like a blueprint being drawn live.
- A subtle scanning beam (`.stitch-laser-scan`) glides down the page container while text populates.

### 3.2 Real-Time Multi-Tab Synchronization
- **Tab 1 (Render View)**: Dynamically appends page cards with Stitch UI animations, formatted tables, and headers.
- **Tab 2 (Clean Markdown)**: Appends clean Markdown into the code area with auto-scrolling.
- **Tab 3 (Raw Bounding Boxes)**: Appends raw bounding box data in real-time.
- **Sticky Page Navigation**: Instantly adds quick page jump buttons `[Trang 1]`, `[Trang 2]`, ... as pages arrive.

---

## 4. Safety & Accuracy Guarantees

- **Core Model Invariance**: Model weights, prompt (`<image>document parsing.`), crop parameters (`crop_mode`, `image_size`, `no_repeat_ngram_size=35`, `ngram_window`), and decoding logic remain **100% UNCHANGED**.
- **Backward Compatibility**: Existing endpoint `POST /v1/ocr` remains fully functional for non-streaming REST API clients.
