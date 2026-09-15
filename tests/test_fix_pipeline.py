import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from fix_pipeline import (event_correction, fit_roll_axis, image_rotations,   # noqa: E402
                          scalar_gain, shifted_cumsum, spectral_keep,
                          window_disagreement)
from rollfix import logv, qmul, qconj   # noqa: E402
from rotmath import rotation_vector_quat   # noqa: E402
from timing import frame_shift_us   # noqa: E402
from denoise import BAND_HZ   # noqa: E402


class _Clip:
    frame_readout_time_ms = 13.58


def _frames(exposures):
    return [{'camera': {'exposure_time_ms': e}} if e is not None else {'camera': {}} for e in exposures]


class TimingTests(unittest.TestCase):
    def test_exposure_shift_sign_and_gap_fill(self):
        d = frame_shift_us(_Clip(), _frames([19.6, None, 7.0]), 'exposure')
        # readout/2 - exposure/2: 6.79 - 9.8 = -3.01 ms; 6.79 - 3.5 = +3.29 ms
        np.testing.assert_allclose(d[[0, 2]] / 1000, [-3.01, 3.29], atol=1e-6)
        self.assertTrue(-3.01e3 < d[1] < 3.29e3)   # the missing one is interpolated
        np.testing.assert_array_equal(frame_shift_us(_Clip(), _frames([5.0]), 'none'), [0.0])
        # variation-only mode: zero at the median exposure, +2.5 ms where 5 ms shorter
        np.testing.assert_allclose(frame_shift_us(_Clip(), _frames([10.0, 10.0, 5.0]), 'exposure_var') / 1000,
                                   [0.0, 0.0, 2.5], atol=1e-9)
        np.testing.assert_allclose(frame_shift_us(_Clip(), _frames([5.0, 5.0]), 'constant', 2.5), [2500, 2500])

    def test_shifted_cumsum_is_sum_before_frame(self):
        c = np.array([[1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]])
        np.testing.assert_array_equal(shifted_cumsum(c)[:, 0], [0, 1, 3])


class EventTests(unittest.TestCase):
    def test_event_is_zero_net_and_gated(self):
        fps, n = 50.0, 400
        uh = np.array([0.0, 0.0, 1.0])
        delta = np.zeros((n, 3))
        delta[200:215, 0] = 0.5             # a 0.3 s pitch disagreement of 0.5 deg/frame
        delta[200:215, 2] = 0.3             # roll component must be ignored here
        reliable = np.ones(n, bool)
        rate = np.full(n, 10.0)
        c, events = event_correction(delta, reliable, rate, fps, uh, thr_deg=0.12)
        self.assertEqual(len(events), 1)
        from fix_pipeline import weight_events
        events[0]['shift_check'] = {'verdict': 'opposed', 'corr': -0.5, 'gain': -1}
        np.testing.assert_array_equal(weight_events([dict(events[0])], n), 0.0)   # opposed -> dropped
        # a scaled copy of a real pan is not an event: image 0.6 x telemetry
        inc = np.zeros((n, 3))
        inc[200:215, 0] = 1.25                        # telemetry pans 1.25 deg/frame
        scaled = np.zeros((n, 3))
        scaled[200:215, 0] = -0.5                     # image shows 0.75 -> disagreement -0.5 = 40 %
        c4, ev4 = event_correction(scaled, reliable, rate, fps, uh, thr_deg=0.12, inc=inc)
        self.assertEqual(ev4, [])
        missing = np.zeros((n, 3))
        missing[200:215, 0] = -1.0                    # image shows 0.25 of a 1.25 pan -> event
        c5, ev5 = event_correction(missing, reliable, rate, fps, uh, thr_deg=0.12, inc=inc)
        self.assertEqual(len(ev5), 1)
        self.assertAlmostEqual(c[:, 0].sum(), 0.0, places=9)      # shape kept, net rotation removed
        np.testing.assert_array_equal(c[:, 2], 0.0)               # roll untouched
        # the bridge model straightens the telemetry's own pitch/yaw rate through the window
        inc_b = np.zeros((n, 3))
        inc_b[205:210, 1] = 2.0                                    # a 0.1 s yaw swing in the telemetry
        cb, evb = event_correction(delta, reliable, rate, fps, uh, thr_deg=0.12, inc=inc_b, model='bridge')
        self.assertEqual(len(evb), 1)
        self.assertLess(np.abs((inc_b + cb)[200:215, 1]).max(), 0.5)   # the swing is mostly gone
        self.assertTrue(np.all(c[:190] == 0) and np.all(c[225:] == 0))
        # unreliable windows or fast motion switch the repair off
        c2, ev2 = event_correction(delta, np.zeros(n, bool), rate, fps, uh)
        c3, ev3 = event_correction(delta, reliable, np.full(n, 300.0), fps, uh)
        self.assertEqual((len(ev2), len(ev3)), (0, 0))
        np.testing.assert_array_equal(c2, 0.0)

    def test_short_blip_is_not_an_event(self):
        n, uh = 100, np.array([0.0, 0.0, 1.0])
        delta = np.zeros((n, 3))
        delta[50, 1] = 5.0
        c, events = event_correction(delta, np.ones(n, bool), np.full(n, 5.0), 50.0, uh)
        self.assertEqual(events, [])


class WindowTests(unittest.TestCase):
    def test_disagreement_recovers_injected_error(self):
        fps, n, gap = 50.0, 300, 5
        rng = np.random.default_rng(1)
        # a smooth true motion + a telemetry error burst on the (mapped) y axis
        t = np.arange(n + 1) / fps
        true_inc = np.column_stack([0.2 * np.sin(2 * np.pi * 0.5 * t), 0.1 * np.cos(2 * np.pi * 0.3 * t), 0.05 * t * 0])
        err = np.zeros_like(true_inc)
        err[150:170, 1] = 0.4
        tel_inc = true_inc + err

        def integrate(inc):
            q = np.tile([1.0, 0, 0, 0], (len(inc) + 1, 1))
            for i in range(len(inc)):
                q[i + 1] = qmul(q[i], rotation_vector_quat(inc[i:i + 1]))[0]
                q[i + 1] /= np.linalg.norm(q[i + 1])
            return q
        qt = integrate(tel_inc)
        qtrue = integrate(true_inc)
        s = np.sign(np.sum(qtrue[:-gap] * qtrue[gap:], axis=1))
        T_true = logv(qmul(qconj(qtrue[:-gap]), qtrue[gap:] * s[:, None]))
        # image in OpenCV axes: x negated relative to the telemetry frame
        img = T_true * np.array([-1, 1, 1]) + rng.normal(0, 0.01, T_true.shape)
        d = {'homog_deg': img, 'kabsch_deg': img + rng.normal(0, 0.01, img.shape),
             'npts_win': np.full(len(img), 500)}
        rate = np.linalg.norm(tel_inc, axis=1) * fps
        w = window_disagreement(qt, d, fps, gap, rate)
        self.assertEqual((w['axis_perm'], w['axis_signs']), ([0, 1, 2], [-1, 1, 1]))
        D = w['D']
        # windows fully inside the burst see -0.4 deg/frame * 5 on y (the 2 s
        # running median takes a little of it: the burst is 0.4 s of the window)
        inside = np.arange(150, 165)
        np.testing.assert_allclose(D[inside, 1], -2.0, atol=0.35)
        self.assertTrue(np.all(np.abs(D[:130]) < 0.2))
        self.assertTrue(w['reliable'][inside].all())


class ImageRotationTests(unittest.TestCase):
    def test_kabsch1_is_mapped_and_padded(self):
        d = {'kabsch1_deg': np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), 'roll_deg': np.array([0.1, 0.2]),
             'roll_n_deg': np.array([0.3, 0.4])}
        inc_img, roll, src = image_rotations(d, 3)
        self.assertEqual(src, 'kabsch1')
        np.testing.assert_array_equal(inc_img[0], [-1.0, 2.0, 3.0])   # x negated
        self.assertTrue(np.isnan(inc_img[2]).all())
        np.testing.assert_array_equal(roll[:2], [0.3, 0.4])           # undistorted curl preferred
        inc2, roll2, src2 = image_rotations({'roll_deg': np.array([0.1])}, 2)
        self.assertIsNone(inc2)
        self.assertEqual(src2, 'roll_deg')

    def test_scalar_gain(self):
        rng = np.random.default_rng(3)
        x = rng.normal(0, 1, 1000)
        g, c, n = scalar_gain(x, 0.8 * x, np.full(1000, 50.0))
        self.assertAlmostEqual(g, 0.8)
        self.assertAlmostEqual(c, 1.0)
        g2, c2, n2 = scalar_gain(x, x, np.full(1000, 5.0))       # too slow -> no fit
        self.assertEqual((g2, n2), (1.0, 0))


class CalibrationTests(unittest.TestCase):
    def test_roll_axis_fit_and_fallback(self):
        rng = np.random.default_rng(0)
        inc = rng.normal(0, 1, (2000, 3))
        u = np.array([0.1, -0.3, 0.95])
        img = inc @ u * 1.2
        rate = np.full(2000, 60.0)
        uh, g, corr, n = fit_roll_axis(inc, img, rate)
        np.testing.assert_allclose(uh, u / np.linalg.norm(u), atol=1e-9)
        self.assertAlmostEqual(g, 1.2 * np.linalg.norm(u), places=9)
        uh2, g2, corr2, n2 = fit_roll_axis(inc, rng.normal(0, 1, 2000), rate)   # no relation -> fallback
        self.assertLess(corr2, 0.9)

    def test_spectral_keep_is_one_below_floor_and_smaller_where_telemetry_is_noisier(self):
        rng = np.random.default_rng(2)
        fps, n = 50.0, 3000
        t = np.arange(n) / fps
        real = np.sin(2 * np.pi * 2 * t)
        tel = real + rng.normal(0, 0.5, n)          # broadband excess
        img = real + rng.normal(0, 0.05, n)
        keep, info = spectral_keep(tel, img, fps, np.ones(n, bool), floor_hz=4.0)
        self.assertTrue(np.all(keep[BAND_HZ < 4.0] == 1.0))
        self.assertTrue(np.all(keep[BAND_HZ >= 8.0] < 0.5))
        self.assertIsNone(spectral_keep(tel, img, fps, np.zeros(n, bool)))


if __name__ == '__main__':
    unittest.main()


class LensTests(unittest.TestCase):
    def test_gyroflow_profile_is_rescaled_to_the_video(self):
        import json, tempfile, os
        from lenscal import load_gyroflow_lens
        prof = {'name': 'test', 'calib_dimension': {'w': 1920, 'h': 1440},
                'fisheye_params': {'camera_matrix': [[800.0, 0, 960.0], [0, 800.0, 720.0], [0, 0, 1]],
                                   'distortion_coeffs': [0.1, -0.05, 0.01, 0.0]}}
        fd, path = tempfile.mkstemp(suffix='.json')
        with os.fdopen(fd, 'w', encoding='utf-8') as fh:
            json.dump(prof, fh)
        try:
            lens = load_gyroflow_lens(path, 3840, 2880)
        finally:
            os.remove(path)
        self.assertAlmostEqual(lens['f'], 1600.0)
        self.assertAlmostEqual(lens['cx'], 1920.0)
        self.assertAlmostEqual(lens['cy'], 1440.0)
        self.assertEqual(lens['D'], [0.1, -0.05, 0.01, 0.0])


class SpikeTests(unittest.TestCase):
    def test_spike_is_found_only_when_the_image_disagrees(self):
        from fix_pipeline import detect_spikes
        n, uh = 200, np.array([0.0, 0.0, 1.0])
        inc = np.zeros((n, 3))
        inc[:, 1] = 0.3                      # steady 15 deg/s yaw
        inc[100, 1] = 1.8                    # one-frame telemetry jump
        inc[150, 1] = 1.8                    # the same jump, but real this time
        img = inc.copy()
        img[100, 1] = 0.3                    # the image did not see the first jump
        fine = img - inc
        spike, c, contaminated = detect_spikes(inc, fine, fine, np.full(n, 15.0), uh, np.ones(n, bool))
        self.assertEqual(list(np.flatnonzero(spike)), [100])
        self.assertAlmostEqual(c[100, 1], -1.5, places=6)   # back to the local median
        self.assertTrue(contaminated[95] and contaminated[105] and not contaminated[100])
