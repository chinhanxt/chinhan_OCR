# Vietnamese Administrative OCR Post-Corrector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a zero-latency Vietnamese Administrative Document OCR Corrector to automatically fix common OCR typos (such as "TỬ TRÌNH" ➔ "TỜ TRÌNH", "KÍNH GỦI" ➔ "KÍNH GỬI") without impacting GPU inference speed.

**Architecture:** A lightweight rule & dictionary-based post-processing pipeline (`VietnameseTextCorrector`) integrated into the text cleanup phase (`clean_ocr_to_markdown`). Uses word-boundary regular expressions and case-preserving dictionary lookups.

**Tech Stack:** Python 3.11, `re` (Standard Library), `unittest` / `pytest`

## Global Constraints

- No external neural model or network dependencies (0ms GPU/VRAM overhead).
- Must preserve Markdown formatting, layout tags, and table pipes (`|`).
- Case preservation: Must correctly handle ALL-CAPS (`TỬ TRÌNH` ➔ `TỜ TRÌNH`), Title Case (`Tử trình` ➔ `Tờ trình`), and lower-case (`tử trình` ➔ `tờ trình`).

---

### Task 1: Create Vietnamese Text Corrector Module & Unit Tests

**Files:**
- Create: `src/core/corrector.py`
- Create: `tests/test_corrector.py`

**Interfaces:**
- Produces: `correct_vietnamese_text(text: str) -> str` in `src/core/corrector.py`

- [ ] **Step 1: Write the failing unit test**

Create `tests/test_corrector.py`:

```python
import unittest
from src.core.corrector import correct_vietnamese_text

class TestVietnameseTextCorrector(unittest.TestCase):
    def test_tu_trinh_uppercase(self):
        raw = "# TỬ TRÌNH VỀ VIỆC PHÊ DUYỆT DỰ ÁN"
        expected = "# TỜ TRÌNH VỀ VIỆC PHÊ DUYỆT DỰ ÁN"
        self.assertEqual(correct_vietnamese_text(raw), expected)

    def test_tu_trinh_titlecase(self):
        raw = "Bản Tử trình công tác năm 2026"
        expected = "Bản Tờ trình công tác năm 2026"
        self.assertEqual(correct_vietnamese_text(raw), expected)

    def test_kinh_gui_uppercase(self):
        raw = "KÍNH GỦI: ỦY BAN NHÂN DÂN THÀNH PHỐ"
        expected = "KÍNH GỬI: ỦY BAN NHÂN DÂN THÀNH PHỐ"
        self.assertEqual(correct_vietnamese_text(raw), expected)

    def test_cong_hoa_and_uy_ban(self):
        raw = "CỘNG HOÀ XÃ HỘI CHỦ NGHĨA VIỆT NAM\nUỶ BAN NHÂN DÂN"
        expected = "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\nỦY BAN NHÂN DÂN"
        self.assertEqual(correct_vietnamese_text(raw), expected)

    def test_preserve_tables(self):
        raw = "| 1 | Tử trình dự toán | 100.000 |"
        expected = "| 1 | Tờ trình dự toán | 100.000 |"
        self.assertEqual(correct_vietnamese_text(raw), expected)

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests/test_corrector.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.core.corrector'`

- [ ] **Step 3: Implement `src/core/corrector.py`**

Create `src/core/corrector.py`:

```python
import re

# Comprehensive dictionary of common OCR typos in Vietnamese administrative documents
ADMIN_CORRECTIONS = [
    # Tờ trình typos
    (r'\bTỬ TRÌNH\b', 'TỜ TRÌNH'),
    (r'\bTử trình\b', 'Tờ trình'),
    (r'\btử trình\b', 'tờ trình'),
    
    # Kính gửi typos
    (r'\bKÍNH GỦI\b', 'KÍNH GỬI'),
    (r'\bKính gủi\b', 'Kính gửi'),
    (r'\bkính gủi\b', 'kính gửi'),
    
    # Cộng hòa / Ủy ban standardization
    (r'\bCỘNG HOÀ\b', 'CỘNG HÒA'),
    (r'\bCộng hoà\b', 'Cộng hòa'),
    (r'\bUỶ BAN\b', 'ỦY BAN'),
    (r'\bUỷ ban\b', 'Ủy ban'),
    (r'\buỷ ban\b', 'ủy ban'),
    
    # Căn cứ / Quyết định / Thành phố
    (r'\bCĂN CÚ\b', 'CĂN CỨ'),
    (r'\bCăn cú\b', 'Căn cứ'),
    (r'\bQUYẾT ĐỊNH\b', 'QUYẾT ĐỊNH'),
    
    # Độc lập - Tự do - Hạnh phúc formatting
    (r'\bĐỘC LẬP\s*[-–—]\s*TỰ DO\s*[-–—]\s*HẠNH PHÚC\b', 'ĐỘC LẬP - TỰ DO - HẠNH PHÚC'),
    (r'\bĐộc lập\s*[-–—]\s*Tự do\s*[-–—]\s*Hạnh phúc\b', 'Độc lập - Tự do - Hạnh phúc'),
]

def correct_vietnamese_text(text: str) -> str:
    """
    Applies high-precision administrative dictionary & regex rules to correct OCR typos.
    
    Args:
        text (str): Raw OCR output or Markdown text.
    Returns:
        str: Corrected text.
    """
    if not text:
        return text
        
    corrected = text
    for pattern, replacement in ADMIN_CORRECTIONS:
        corrected = re.sub(pattern, replacement, corrected)
        
    return corrected
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests/test_corrector.py`
Expected: PASS (`Ran 5 tests in 0.001s ... OK`)

- [ ] **Step 5: Commit**

```bash
git add src/core/corrector.py tests/test_corrector.py
git commit -m "feat: add Vietnamese administrative OCR text corrector and unit tests"
```

---

### Task 2: Integrate Corrector into `clean_ocr_to_markdown` in `app.py`

**Files:**
- Modify: `app.py:195-228`
- Modify: `src/config.py` (optional import export)
- Create: `tests/test_integration.py`

**Interfaces:**
- Consumes: `correct_vietnamese_text(text: str)` from `src.core.corrector`

- [ ] **Step 1: Write integration test**

Create `tests/test_integration.py`:

```python
import unittest
from app import clean_ocr_to_markdown

class TestCleanOcrIntegration(unittest.TestCase):
    def test_clean_ocr_with_corrections(self):
        raw_ocr = "<|det|> title [10,10,100,100] <|/det|> TỬ TRÌNH VỀ DỰ ÁN\n<|det|> header [20,20,100,100] <|/det|> KÍNH GỦI: UỶ BAN"
        cleaned = clean_ocr_to_markdown(raw_ocr)
        self.assertIn("# TỜ TRÌNH VỀ DỰ ÁN", cleaned)
        self.assertIn("### KÍNH GỬI: ỦY BAN", cleaned)

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests/test_integration.py`
Expected: FAIL (`AssertionError: '# TỜ TRÌNH VỀ DỰ ÁN' not found in '# TỬ TRÌNH VỀ DỰ ÁN'`)

- [ ] **Step 3: Update `clean_ocr_to_markdown` in `app.py`**

In `app.py`, update `clean_ocr_to_markdown`:

```python
from src.core.corrector import correct_vietnamese_text

def clean_ocr_to_markdown(raw_text: str, page_image_path: str = None) -> str:
    import re
    if not raw_text:
        return ""
    
    # Apply administrative text corrections
    corrected_raw = correct_vietnamese_text(raw_text)
    
    lines = corrected_raw.splitlines()
    clean_lines = []
    
    for line in lines:
        match = re.match(r'^\s*<\|det\|>\s*([a-zA-Z_]+)\s*\[.*?\]\s*<\|/det\|>\s*(.*)$', line)
        if match:
            category, content = match.group(1), match.group(2).strip()
            if not content:
                continue
            
            if category == 'title':
                clean_lines.append(f"# {content}")
            elif category == 'header':
                clean_lines.append(f"### {content}")
            elif category in ['section_header', 'sub_title']:
                clean_lines.append(f"#### {content}")
            elif category == 'page_number':
                clean_lines.append(f"\n*— Trang {content} —*\n")
            else:
                formatted = format_table_line(content)
                clean_lines.append(formatted)
        else:
            cleaned = re.sub(r'<\|det\|>.*?<\|/det\|>', '', line).strip()
            if cleaned:
                formatted = format_table_line(cleaned)
                clean_lines.append(formatted)
                
    return "\n\n".join(clean_lines)
```

- [ ] **Step 4: Run integration test to verify it passes**

Run: `python3 -m unittest tests/test_integration.py`
Expected: PASS (`Ran 1 test in 0.001s ... OK`)

- [ ] **Step 5: Commit and Push**

```bash
git add app.py tests/test_integration.py
git commit -m "feat: integrate VietnameseTextCorrector into clean_ocr_to_markdown pipeline"
git push origin main
git remote set-url origin git@github.com:chinhanxt/chinhan_OCR.git
```
