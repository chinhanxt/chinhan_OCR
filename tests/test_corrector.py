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

    def test_thu_lao_and_khoan(self):
        raw = "THỦ LÀO CÁC THÀNH VIÊN - KHOAÒN CHI PHÍ - Luru: VT"
        expected = "THÙ LAO CÁC THÀNH VIÊN - KHOẢN CHI PHÍ - Lưu: VT"
        self.assertEqual(correct_vietnamese_text(raw), expected)

if __name__ == "__main__":
    unittest.main()
