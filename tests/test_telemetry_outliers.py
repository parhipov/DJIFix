import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research' / 'session'))
sys.path.insert(0, str(ROOT / 'src'))

from analyze_telemetry_outliers import FIELDS, analyze_session


class TelemetryOutlierTests(unittest.TestCase):
    def test_detects_stabilized_only_impulse(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            session = Path(folder)
            n, dt = 200, .02
            time_s = (np.arange(n) + .5) * dt
            reference = np.column_stack([
                3 * np.sin(time_s), 2 * np.cos(time_s * .7), np.sin(time_s * 1.3)
            ])
            values = {
                'telemetry_gyro.csv': reference,
                'telemetry_image_source.csv': reference + .02 * np.sin(time_s[:, None] * 3),
                'telemetry_image_stabilized.csv': reference * .15,
            }
            values['telemetry_image_stabilized.csv'][100, 1] += 25
            for filename, motion in values.items():
                with (session / filename).open('w', newline='', encoding='utf-8') as handle:
                    writer = csv.writer(handle)
                    writer.writerow(FIELDS)
                    for i, row in enumerate(motion):
                        writer.writerow([time_s[i], i, i + 1, *row, 1, 500])
            report = analyze_session(session)
            self.assertEqual(report['event_count'], 1)
            self.assertAlmostEqual(report['events'][0]['peak_time_s'], time_s[100])
            self.assertEqual(report['events'][0]['peak_axis'], 'Y')
            self.assertGreater(report['events'][0]['strength_0_100'], 50)


if __name__ == '__main__':
    unittest.main()
