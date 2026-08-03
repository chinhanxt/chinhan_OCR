import re

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
    
    # Thù lao / Khoản chi phí / Lưu
    (r'\bTHỦ LÀO\b', 'THÙ LAO'),
    (r'\bThủ lao\b', 'Thù lao'),
    (r'\bthủ lao\b', 'thù lao'),
    (r'\bKHOAÒN\b', 'KHOẢN'),
    (r'\bKhoaòn\b', 'Khoản'),
    (r'\bkhoaòn\b', 'khoản'),
    (r'\bLuru:', 'Lưu:'),
    (r'\bluru:', 'lưu:'),
    
    # Độc lập - Tự do - Hạnh phúc formatting
    (r'\bĐỘC LẬP\s*[-–—]\s*TỰ DO\s*[-–—]\s*HẠNH PHÚC\b', 'ĐỘC LẬP - TỰ DO - HẠNH PHÚC'),
    (r'\bĐộc lập\s*[-–—]\s*Tự do\s*[-–—]\s*Hạnh phúc\b', 'Độc lập - Tự do - Hạnh phúc'),
]

def fix_vietnamese_dates(text: str) -> str:
    """
    Cleans OCR punctuation noise around Vietnamese dates and corrects invalid day numbers (>31).
    E.g. 'ngày. 45.tháng .1. năm 2025' -> 'ngày 25 tháng 1 năm 2025'
    """
    if not text:
        return text

    # Clean up irregular dots and spaces around "ngày", "tháng", "năm"
    text = re.sub(r'\bngày[\.\s]+', 'ngày ', text, flags=re.IGNORECASE)
    text = re.sub(r'[\.\s]+tháng[\.\s]+', ' tháng ', text, flags=re.IGNORECASE)
    text = re.sub(r'[\.\s]+năm[\.\s]+', ' năm ', text, flags=re.IGNORECASE)
    text = re.sub(r'[\.\s]+năm\b', ' năm', text, flags=re.IGNORECASE)

    # Correct impossible day numbers (>31) caused by OCR digit confusion (e.g. 45 -> 25)
    def fix_day(m):
        prefix = m.group(1)
        day_str = m.group(2)
        suffix = m.group(3)
        day = int(day_str)
        if day > 31:
            if 40 <= day <= 49:
                day_str = str(day - 20)  # e.g., 45 -> 25
            elif 32 <= day <= 39:
                day_str = str(day - 10)  # e.g., 35 -> 25
        return f"{prefix}{day_str}{suffix}"

    text = re.sub(r'(ngày\s+)(\d{2})(\s+tháng)', fix_day, text, flags=re.IGNORECASE)
    return text

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
        
    corrected = fix_vietnamese_dates(corrected)
    return corrected
