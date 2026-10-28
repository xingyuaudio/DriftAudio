"""Tests for exact caption CSV conversion."""
import csv
import tempfile
import unittest
from pathlib import Path

from tools.make_text_csv import convert


class CaptionCsvTests(unittest.TestCase):
    def test_exact_rows_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest = Path(folder) / "captions.tsv"
            output = Path(folder) / "captions.csv"
            with manifest.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["id", "caption"], delimiter="\t")
                writer.writeheader()
                writer.writerows([
                    {"id": "a", "caption": "A bell rings"},
                    {"id": "b", "caption": "A dog barks"},
                ])
            self.assertEqual(convert(manifest, output), 2)
            with output.open(newline="", encoding="utf-8") as stream:
                self.assertEqual(list(csv.DictReader(stream)), [
                    {"name": "a", "caption": "A bell rings"},
                    {"name": "b", "caption": "A dog barks"},
                ])
            with self.assertRaises(FileExistsError):
                convert(manifest, output)


if __name__ == "__main__":
    unittest.main()
