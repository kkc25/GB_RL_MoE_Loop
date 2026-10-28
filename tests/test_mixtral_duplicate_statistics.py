"""Same-role duplicate observability; cross-role rejection remains separate."""
import json
from pathlib import Path
import tempfile
import unittest

from mixtral_lab.protocol import prepare_data
from tests.test_mixtral_protocol import config


class DuplicateStatisticsTests(unittest.TestCase):
    def test_same_role_duplicates_are_visible_without_resampling(self):
        class TinyTokenizer:
            pad_token_id = 0
            def encode(self, text, **kwargs):
                if text.startswith('common'):
                    return [1, 2, 3]
                return [1, 2, 4 if 'selection' in text else 5]
        with tempfile.TemporaryDirectory() as directory:
            c = config(Path(directory))
            c['sequences']['train'] = 2
            Path(c['data_files']['train']).write_text('\n'.join(json.dumps(row) for row in [
                {'id': 'train-a', 'text': 'common prefix different raw tail a'},
                {'id': 'train-b', 'text': 'common prefix different raw tail b'},
            ])+'\n')
            for role in ('selection', 'calibration'):
                Path(c['data_files'][role]).write_text(json.dumps({'id': role, 'text': role+' content'})+'\n')
            data, identity = prepare_data(c, TinyTokenizer())
            stats = identity['train']['effective_input_statistics']
            self.assertEqual(stats['documents'], 2)
            self.assertEqual(stats['unique_effective_sequences'], 1)
            self.assertEqual(stats['redundant_documents'], 1)
            self.assertEqual(stats['redundant_document_fraction'], .5)
            self.assertEqual(stats['documents_in_duplicate_groups'], 2)
            self.assertEqual(stats['duplicate_groups'][0]['document_ids'], ['train-a', 'train-b'])
            self.assertEqual(stats['duplicate_groups'][0]['effective_token_count'], 3)
            self.assertEqual(len(data['train']), 2)
            self.assertEqual(identity['train']['supervised_tokens'], 4)
            self.assertFalse(stats['sampling_changed'])
            self.assertEqual(identity['selection']['effective_input_statistics']['duplicate_groups'], [])
            Path(c['data_files']['selection']).write_text(json.dumps({'id': 'new-id', 'text': 'common other raw tail'})+'\n')
            with self.assertRaisesRegex(ValueError, 'across development roles'):
                prepare_data(c, TinyTokenizer())


if __name__ == '__main__':
    unittest.main()
