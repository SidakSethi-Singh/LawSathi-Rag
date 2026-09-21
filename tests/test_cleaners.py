import unittest
from src.preprocessing.cleaners import strip_html_tags, normalize_spacing, ensure_utf8_encoding, clean_text


class TestCleaners(unittest.TestCase):
    def test_strip_html_tags_preserves_spacing_between_elements(self):
        html = "<p>Supreme Court</p><p>New Delhi</p>"
        result = strip_html_tags(html)
        self.assertEqual(normalize_spacing(result), "Supreme Court New Delhi")

    def test_strip_html_tags_table_elements(self):
        html = "<td>Section 420</td><td>Cheating</td>"
        result = strip_html_tags(html)
        self.assertEqual(normalize_spacing(result), "Section 420 Cheating")

    def test_strip_html_tags_empty_and_non_html(self):
        self.assertEqual(strip_html_tags(""), "")
        self.assertEqual(strip_html_tags("Plain legal text"), "Plain legal text")

    def test_normalize_spacing(self):
        text = "  Section   302   \n\n\t IPC   "
        self.assertEqual(normalize_spacing(text), "Section 302 IPC")

    def test_clean_text_full_pipeline(self):
        raw = "<h1>Case Law</h1><p>The <b>accused</b> was convicted.</p>"
        self.assertEqual(clean_text(raw), "Case Law The accused was convicted.")


if __name__ == "__main__":
    unittest.main()
