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
