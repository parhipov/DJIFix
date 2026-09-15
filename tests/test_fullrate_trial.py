import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research'))

import numpy as np

from fullrate_trial import apply_correction, sample_shifted, smooth_quaternions
from selective import rotation_vector_quat
from rollfix import logv, qmul, qconj


class FullRateTests(unittest.TestCase):
    def test_time_shift_sign_and_amount(self):
        grid = np.arange(2001) / 1000
        v = np.zeros((len(grid), 3))
        v[:, 1] = grid * 100
        q = rotation_vector_quat(v)
        times = np.array([.5, 1., 1.5])
        baseline = sample_shifted(grid, q, times, 0)
        for extra_ms in (-300., -100., -6., 6., 100., 300.):
            shifted = sample_shifted(grid, q, times, extra_ms)
            relative = logv(qmul(qconj(baseline), shifted))
            np.testing.assert_allclose(relative[:, 1], extra_ms * .1, atol=1e-9)
            np.testing.assert_allclose(relative[:, [0, 2]], 0, atol=1e-9)
        with self.assertRaises(ValueError):
            sample_shifted(grid, q, np.array([0.]), -6)

    def test_strength_scales_relative_rotation_and_preserves_window(self):
        q = rotation_vector_quat(np.array([[10., 0., 0.]] * 3))
        target = qmul(q, rotation_vector_quat(np.array([[0., 4., 0.]] * 3)))
        gate = np.array([0., .5, 1.])
        for gain in (0., 1., 1.25, 1.5):
            result, applied = apply_correction(q, target, gate, gain)
            relative = logv(qmul(qconj(q), result))
            expected = np.zeros((3, 3))
            expected[:, 1] = 4 * gate * gain
            np.testing.assert_allclose(relative, expected, atol=1e-10)
            np.testing.assert_array_equal(result[0], q[0])
            np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1, atol=1e-14)

    def test_quaternion_sign_does_not_change_result(self):
        t = np.arange(4000) / 1000
        v = np.zeros((len(t), 3))
        v[:, 1] = np.sin(t)
        q = rotation_vector_quat(v)
        flipped = q.copy()
        flipped[::7] *= -1
        a = smooth_quaternions(q, 1000, 8)
        b = smooth_quaternions(flipped, 1000, 8)
        np.testing.assert_allclose(abs(np.sum(a * b, axis=1)), 1., atol=1e-12)

    def test_removes_between_frame_oscillation(self):
        t = np.arange(6000) / 1000
        v = np.zeros((len(t), 3))
        v[:, 1] = .5 * np.sin(2 * np.pi * 50 * t)
        q = rotation_vector_quat(v)
        # At 50 fps all frame samples miss this motion completely.
        np.testing.assert_allclose(v[::20], 0, atol=1e-10)
        out = smooth_quaternions(q, 1000, 8)
        self.assertLess(logv(out)[1000:-1000].std(), 0.005)

    def test_preserves_slow_constant_rotation(self):
        t = np.arange(8000) / 1000
        v = np.zeros((len(t), 3))
        v[:, 1] = 5 * t
        q = rotation_vector_quat(v)
        out = smooth_quaternions(q, 1000, 2)
        error = logv(qmul(qconj(q), out))[2000:-2000]
        self.assertLess(np.linalg.norm(error, axis=1).max(), .001)


if __name__ == '__main__':
    unittest.main()
