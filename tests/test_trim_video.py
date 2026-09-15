"""Run: python -m unittest test_trim_video (integration uses DJI_TRIM_TEST_FILE)."""
import os
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from trim_video import seconds, trim


class TimeTests(unittest.TestCase):
    def test_optional_and_formats(self):
        self.assertIsNone(seconds(None))
        for value in (10.5, '10.5', '00:10,5', '00:00:10.5'):
            self.assertEqual(seconds(value), Fraction(21, 2))
        self.assertEqual(seconds('1:02:03'), 3723)

    def test_invalid(self):
        for value in ('', -1, 'nan', 'inf', '00:60', '1:2:3:4'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                seconds(value)


@unittest.skipUnless(os.environ.get('DJI_TRIM_TEST_FILE'), 'Set DJI_TRIM_TEST_FILE for real-video tests')
class IntegrationTests(unittest.TestCase):
    def test_optional_boundaries_and_repeated_trim(self):
        source = os.environ['DJI_TRIM_TEST_FILE']
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            root = Path(folder)
            # Every trim automatically compares all track packets, metadata,
            # frame associations, quaternion values and relative timestamps.
            base = trim(source, 10, 11, root / 'base.mp4')
            self.assertEqual((base['start_s'], base['end_s']), (9.6, 11.4))
            begin = trim(root / 'base.mp4', None, .5, root / 'begin.mp4')
            self.assertEqual((begin['start_s'], begin['end_s']), (0, .6))
            end = trim(root / 'base.mp4', .7, None, root / 'end.mp4')
            self.assertEqual((end['start_s'], end['end_s']), (.6, 1.8))
            full = trim(root / 'base.mp4', None, None, root / 'full.mp4')
            self.assertEqual(full['quaternions'], base['quaternions'])
            self.assertEqual(full['frames'], base['frames'])
            with self.assertRaises(FileExistsError):
                trim(root / 'base.mp4', None, None, root / 'full.mp4')
            for start, stop in ((1, 0), (0, 100), (-1, None), (2, None)):
                with self.subTest(start=start, stop=stop), self.assertRaises(ValueError):
                    trim(root / 'base.mp4', start, stop, root / 'invalid.mp4')
            self.assertFalse((root / 'invalid.mp4').exists())
            self.assertFalse(list(root.glob('*.partial')))


if __name__ == '__main__':
    unittest.main()
