import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research'))

import numpy as np

from selective import correction, limit_rotation, rotation_vector_quat, spectral_filter, window_gate
from denoise import KEEP


class LocalCorrectionTests(unittest.TestCase):
    def test_gate_support_and_overlap(self):
        t = np.linspace(0, 10, 1001)
        w = window_gate(t, [(2, 4), (3, 5)], 0.5)
        self.assertTrue(np.all(w[(t <= 2) | (t >= 5)] == 0))
        self.assertTrue(np.all(w[(t >= 2.5) & (t <= 4.5)] == 1))
        self.assertLessEqual(w.max(), 1)

    def test_stationary_attitude(self):
        q = np.tile([1., 0., 0., 0.], (200, 1))
        theta, speed = correction(q, 50, [(np.array([0, 0, 1]), KEEP)], 1)
        np.testing.assert_array_equal(theta, 0)
        np.testing.assert_array_equal(speed, 0)

    def test_noise_phase_and_fps(self):
        for fps in (50, 60, 100):
            t = np.arange(10 * fps) / fps
            v = np.zeros((len(t), 3))
            v[:, 2] = .1 * np.sin(2 * np.pi * 10 * t)
            q = rotation_vector_quat(v)
            theta, _ = correction(q, fps, [(np.array([0, 0, 1]), KEEP)], 1)
            # A phase/off-by-one error reinforces rather than cancels this.
            mid = slice(2 * fps, -2 * fps)
            self.assertLess(np.std((v + theta)[mid, 2]), .3 * np.std(v[mid, 2]))
            np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1, atol=1e-14)

    def test_large_telemetry_rate_does_not_disable_correction(self):
        t = np.arange(500) / 50
        v = np.zeros((len(t), 3))
        v[:, 2] = 5 * np.sin(2 * np.pi * 10 * t)
        theta, speed = correction(rotation_vector_quat(v), 50,
                                  [(np.array([0, 0, 1]), KEEP)], 1)
        self.assertGreater(speed.max(), 100)
        mid = slice(100, -100)
        self.assertLess(np.std((v + theta)[mid, 2]), .3 * np.std(v[mid, 2]))

    def test_filter_does_not_wrap_end_into_start(self):
        x = np.zeros(1000)
        y = x.copy()
        y[-1] = 1
        self.assertLess(abs(spectral_filter(y, KEEP, 50)[0]), 1e-4)

    def test_amplitude_bound_preserves_direction_and_zero(self):
        v = np.array([[0., 0., 0.], [3., 4., 0.], [.0001, 0., 0.]])
        bounded = limit_rotation(v, 1.)
        self.assertTrue(np.all(np.linalg.norm(bounded, axis=1) <= 1))
        np.testing.assert_array_equal(bounded[0], 0)
        np.testing.assert_allclose(bounded[1, 0] / bounded[1, 1], 3 / 4)
        np.testing.assert_allclose(bounded[2], v[2], atol=1e-11)


if __name__ == '__main__':
    unittest.main()
