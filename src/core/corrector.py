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
