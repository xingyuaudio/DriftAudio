import tempfile
import unittest
from pathlib import Path
from driftaudio.evaluate import validate_inputs


class EvaluationProtocolTests(unittest.TestCase):
    def test_caption_and_id_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root/'manifest.tsv'
            text = root/'text.csv'
            manifest.write_text('id\tcaption\na\tdog\n')
            text.write_text('name,caption\na,dog\n')
            gt, pred = root/'gt', root/'pred'
            gt.mkdir(); pred.mkdir()
            (gt/'a.wav').touch(); (pred/'a.flac').touch()
            self.assertEqual(validate_inputs(manifest, text, gt, pred), {'a'})
            text.write_text('name,caption\na,cat\n')
            with self.assertRaises(ValueError): validate_inputs(manifest, text, gt, pred)
            text.write_text('name,caption\na,dog\n')
            (pred/'b.flac').touch()
            with self.assertRaises(ValueError): validate_inputs(manifest, text, gt, pred)
