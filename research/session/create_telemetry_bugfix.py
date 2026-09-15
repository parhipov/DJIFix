"""Create a Gyroflow telemetry sidecar from one analyzed session folder.

Edit SESSION_DIR below and run:
    python create_telemetry_bugfix.py

Or override the folder from the command line:
    python create_telemetry_bugfix.py artifacts/sessions/YYYYMMDD_HHMMSS
"""
from __future__ import annotations

# Session folder. Reports, CSV and the Gyroflow sidecar are written back here.
SESSION_DIR = r'K:\Work\Python\DJI fix\artifacts\sessions\o4pro'

# Recommended nonlinear presets:
#   'adaptive_weak'     -- 100 ms context, only the detected range
#   'adaptive_balanced' -- 250 ms context, range + 100 ms on each side
#   'adaptive_strong'   -- 2 s context, range + 1 s on each side
PRESET = 'adaptive_balanced'

# 'context_curve': fit each local rotation axis from clean samples immediately
# before and after the bug. Other modes are retained for A/B comparison.
CORRECTION_MODEL = 'context_curve'  # context_curve | endpoint_bridge | lowpass

# Visual A/B winner for this clip with Gyroflow rolling-shutter correction ON.
# It defines the existing per-frame alignment and the manual constant variant.
DEFAULT_SHUTTER_VARIANT = 'end'  # start | center | end

# Experimental context-curve policy. Keep both measurements in the report,
# but do not let the size of the detected defect, imperfect optical agreement
# or fit complexity prevent the replacement from reaching its fitted target.
CONTEXT_CURVE_USE_RATE_GUARD = False
CONTEXT_CURVE_USE_MOTION_GUARDS = False
CONTEXT_CURVE_USE_AGREEMENT_GUARD = False
CONTEXT_CURVE_USE_QUALITY_GUARD = False
IMAGE_SOURCE_TARGET_KINDS = ('gyro_fault', 'micro_jitter')

PRESETS = {
    # Preserves more original motion and limits the intervention.
    'weak': {
        'cutoff_hz': 12.0,
        'fade_s': 0.03,
        'min_gain': 0.20,
        'max_gain': 0.65,
        'max_correction_deg': 0.75,
    },
    # The original/default settings.
    'balanced': {
        'cutoff_hz': 8.0,
        'fade_s': 0.04,
        'min_gain': 0.35,
        'max_gain': 1.0,
        'max_correction_deg': 1.5,
    },
    # Removes more high-frequency motion and allows a larger correction.
    'strong': {
        'cutoff_hz': 5.0,
        'fade_s': 0.06,
        'min_gain': 0.60,
        'max_gain': 1.25,
        'max_correction_deg': 2.5,
    },
    'strong2': {
        'cutoff_hz': 2.0,
        'fade_s': 0.09,
        'min_gain': 0.80,
        'max_gain': 1.99,
        'max_correction_deg': 5.5,
    },
    'adaptive_weak': {
        'cutoff_hz': 6.5,
        'fade_s': 0.07,
        'min_gain': 0.45,
        'max_gain': 0.90,
        'max_correction_deg': 1.2,
        'strength_gamma': 1.25,
        'rate_soft_limit_deg_s': 4.0,
        'rate_compression_power': 4.0,
        'agreement_free_deg_s': 5.0,
        'agreement_soft_scale_deg_s': 8.0,
        'motion_interaction_start_deg_s': 3.5,
        'gyro_motion_soft_scale_deg_s': 65.0,
        'high_motion_veto_start_deg_s': 100.0,
        'high_motion_veto_rate_start_deg_s': 3.0,
        'high_motion_veto_strength': 2.5,
        'gyroflow_margin_s': 0.05,
        'minimum_adaptive_scale': 0.08,
        'micro_jitter_gain_boost': 1.5,
        'bridge_blend': 1.0,
        'bridge_range_scale': 1.0,
        'curve_context_s': 0.10,
        'curve_repair_margin_s': 0.0,
        'curve_points_per_side': 5,
        'curve_degree': 2,
        'curve_fit_soft_scale_deg': 0.15,
    },
    'adaptive_balanced': {
        'cutoff_hz': 5.0,
        'fade_s': 0.06,
        'min_gain': 0.60,
        'max_gain': 1.25,
        'max_correction_deg': 2.0,
        # Nonlinear response. The rate knee is deliberately close to the point
        # where the first A/B run started producing bad results.
        'strength_gamma': 1.15,
        'rate_soft_limit_deg_s': 5.0,
        'rate_compression_power': 4.0,
        'agreement_free_deg_s': 6.0,
        'agreement_soft_scale_deg_s': 10.0,
        'motion_interaction_start_deg_s': 4.0,
        'gyro_motion_soft_scale_deg_s': 80.0,
        'high_motion_veto_start_deg_s': 110.0,
        'high_motion_veto_rate_start_deg_s': 3.5,
        'high_motion_veto_strength': 2.0,
        # A wider cosine edge reduces sharp spectral content which Gyroflow can
        # turn into pre-ringing around a local telemetry edit.
        'gyroflow_margin_s': 0.04,
        'minimum_adaptive_scale': 0.12,
        'micro_jitter_gain_boost': 2.25,
        'bridge_blend': 1.0,
        'bridge_range_scale': 2.0,
        'curve_context_s': 0.25,
        'curve_repair_margin_s': 0.10,
        'curve_points_per_side': 10,
        'curve_degree': 3,
        'curve_fit_soft_scale_deg': 0.25,
    },
    'adaptive_strong': {
        'cutoff_hz': 4.0,
        'fade_s': 0.075,
        'min_gain': 0.70,
        'max_gain': 1.45,
        'max_correction_deg': 3.0,
        'strength_gamma': 1.05,
        'rate_soft_limit_deg_s': 7.0,
        'rate_compression_power': 4.0,
        'agreement_free_deg_s': 8.0,
        'agreement_soft_scale_deg_s': 14.0,
        'motion_interaction_start_deg_s': 5.5,
        'gyro_motion_soft_scale_deg_s': 110.0,
        'high_motion_veto_start_deg_s': 140.0,
        'high_motion_veto_rate_start_deg_s': 5.0,
        'high_motion_veto_strength': 1.5,
        'gyroflow_margin_s': 0.05,
        'minimum_adaptive_scale': 0.18,
        'micro_jitter_gain_boost': 3.0,
        'bridge_blend': 1.0,
        'bridge_range_scale': 3.0,
        'curve_context_s': 2.0,
        'curve_repair_margin_s': 1.0,
        'curve_points_per_side': 32,
        'curve_degree': 3,
        'curve_fit_soft_scale_deg': 0.75,
    },
}

if PRESET not in PRESETS:
    raise ValueError(f'Unknown PRESET {PRESET!r}; choose from {tuple(PRESETS)}')

_SETTINGS = PRESETS[PRESET]
CUTOFF_HZ = _SETTINGS['cutoff_hz']
FADE_S = _SETTINGS['fade_s']
MIN_GAIN = _SETTINGS['min_gain']
MAX_GAIN = _SETTINGS['max_gain']
MAX_CORRECTION_DEG = _SETTINGS['max_correction_deg']
STRENGTH_GAMMA = _SETTINGS.get('strength_gamma', 1.0)
RATE_SOFT_LIMIT_DEG_S = _SETTINGS.get('rate_soft_limit_deg_s', float('inf'))
RATE_COMPRESSION_POWER = _SETTINGS.get('rate_compression_power', 4.0)
AGREEMENT_FREE_DEG_S = _SETTINGS.get('agreement_free_deg_s', float('inf'))
AGREEMENT_SOFT_SCALE_DEG_S = _SETTINGS.get('agreement_soft_scale_deg_s', 1.0)
MOTION_INTERACTION_START_DEG_S = _SETTINGS.get(
    'motion_interaction_start_deg_s', float('inf'))
GYRO_MOTION_SOFT_SCALE_DEG_S = _SETTINGS.get(
    'gyro_motion_soft_scale_deg_s', 1.0)
HIGH_MOTION_VETO_START_DEG_S = _SETTINGS.get(
    'high_motion_veto_start_deg_s', float('inf'))
HIGH_MOTION_VETO_RATE_START_DEG_S = _SETTINGS.get(
    'high_motion_veto_rate_start_deg_s', float('inf'))
HIGH_MOTION_VETO_STRENGTH = _SETTINGS.get('high_motion_veto_strength', 0.0)
GYROFLOW_MARGIN_S = _SETTINGS.get('gyroflow_margin_s', 0.0)
MINIMUM_ADAPTIVE_SCALE = _SETTINGS.get('minimum_adaptive_scale', 1.0)
MICRO_JITTER_GAIN_BOOST = _SETTINGS.get('micro_jitter_gain_boost', 1.0)
BRIDGE_BLEND = _SETTINGS.get('bridge_blend', 1.0)
BRIDGE_RANGE_SCALE = _SETTINGS.get('bridge_range_scale', 1.0)
CURVE_CONTEXT_S = _SETTINGS.get('curve_context_s', 0.10)
CURVE_REPAIR_MARGIN_S = _SETTINGS.get('curve_repair_margin_s', 0.0)
CURVE_POINTS_PER_SIDE = _SETTINGS.get('curve_points_per_side', 8)
CURVE_DEGREE = _SETTINGS.get('curve_degree', 3)
CURVE_FIT_SOFT_SCALE_DEG = _SETTINGS.get('curve_fit_soft_scale_deg', 0.25)
CURVE_FIT_FALLBACK_MULTIPLIER = 4.0
CONTEXT_MIN_CONFIDENCE = 0.20

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import median_filter
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'research'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from align import corrections as alignment_corrections, slerp_series
from dji_o4 import read_telemetry
from fullrate_trial import smooth_quaternions
from rollfix import logv, qconj, qmul
from selective import limit_rotation, rotation_vector_quat, window_gate
import quat
import sidecar


FIELDS = ('time_s', 'frame_a', 'frame_b', 'x_deg_s', 'y_deg_s', 'z_deg_s',
          'confidence', 'support')


def load_csv(path: Path) -> dict[str, np.ndarray]:
    data = np.atleast_1d(np.genfromtxt(path, delimiter=',', names=True, encoding='utf-8'))
    if data.dtype.names != FIELDS:
        raise ValueError(f'{path}: unexpected telemetry columns')
    return {name: np.asarray(data[name], dtype=float) for name in FIELDS}


def write_csv(path: Path, source: dict[str, np.ndarray], values: np.ndarray) -> None:
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        for i, row in enumerate(values):
            writer.writerow([
                f'{source["time_s"][i]:.9f}', int(source['frame_a'][i]),
                int(source['frame_b'][i]), *(f'{value:.9g}' for value in row),
                f'{source["confidence"][i]:.6f}', int(source['support'][i]),
            ])


def nonlinear_gain(strength: float) -> float:
    """Convert detector strength to gain without linear extrapolation."""
    x = float(np.clip(strength, 0, 100)) / 100
    shaped = x ** STRENGTH_GAMMA
    return float(MIN_GAIN + (MAX_GAIN - MIN_GAIN) * shaped)


def correction_rate_rms(correction_deg: np.ndarray, sample_time_s: np.ndarray,
                        frame_time_s: np.ndarray) -> float:
    """Predict what the video-frame-rate consumer sees, not IMU-rate edge noise."""
    if len(correction_deg) < 3 or len(frame_time_s) < 3:
        return 0.0
    at_frames = np.column_stack([
        np.interp(frame_time_s, sample_time_s, correction_deg[:, axis],
                  left=0.0, right=0.0)
        for axis in range(3)
    ])
    rate = np.gradient(at_frames, frame_time_s, axis=0)
    return float(np.sqrt(np.mean(np.sum(rate * rate, axis=1))))


def adaptive_scale(predicted_rate: float, context_agreement_error: float,
                   context_motion_rms: float) -> tuple[float, float, float, float, float]:
    """Limit intervention using only motion measured outside detected bugs."""
    if np.isfinite(RATE_SOFT_LIMIT_DEG_S):
        ratio = predicted_rate / RATE_SOFT_LIMIT_DEG_S
        rate_scale = (1 + ratio ** RATE_COMPRESSION_POWER) ** (
            -1 / RATE_COMPRESSION_POWER)
    else:
        rate_scale = 1.0
    excess = max(0.0, context_agreement_error - AGREEMENT_FREE_DEG_S)
    if np.isfinite(AGREEMENT_FREE_DEG_S):
        agreement_scale = 1 / np.sqrt(
            1 + (excess / AGREEMENT_SOFT_SCALE_DEG_S) ** 2)
    else:
        agreement_scale = 1.0
    if np.isfinite(MOTION_INTERACTION_START_DEG_S):
        correction_excess = max(
            0.0, predicted_rate - MOTION_INTERACTION_START_DEG_S)
        interaction = ((context_motion_rms / GYRO_MOTION_SOFT_SCALE_DEG_S) *
                       (correction_excess / MOTION_INTERACTION_START_DEG_S))
        motion_scale = 1 / np.sqrt(1 + interaction ** 2)
    else:
        motion_scale = 1.0
    gyro_excess = max(0.0, context_motion_rms - HIGH_MOTION_VETO_START_DEG_S)
    rate_excess = max(0.0, predicted_rate - HIGH_MOTION_VETO_RATE_START_DEG_S)
    if np.isfinite(HIGH_MOTION_VETO_START_DEG_S):
        joint_risk = ((gyro_excess / GYRO_MOTION_SOFT_SCALE_DEG_S) *
                      (rate_excess / HIGH_MOTION_VETO_RATE_START_DEG_S))
        veto_scale = np.exp(-HIGH_MOTION_VETO_STRENGTH * joint_risk)
    else:
        veto_scale = 1.0
    combined = float(np.clip(
        rate_scale * agreement_scale * motion_scale * veto_scale,
                             MINIMUM_ADAPTIVE_SCALE, 1.0))
    return (float(rate_scale), float(agreement_scale), float(motion_scale),
            float(veto_scale), combined)


def _evenly_spaced(indices: np.ndarray, count: int) -> np.ndarray:
    if len(indices) <= count:
        return indices
    positions = np.linspace(0, len(indices) - 1, count).round().astype(int)
    return indices[np.unique(positions)]


def context_curve_target(ts: np.ndarray, qs: np.ndarray, start: float, end: float,
                         query_time: np.ndarray,
                         excluded_intervals: list[tuple[float, float]]) -> tuple[np.ndarray, dict]:
    """Fit a robust local attitude curve using only clean samples around a gap."""
    clean = np.ones(len(ts), dtype=bool)
    for excluded_start, excluded_end in excluded_intervals:
        clean &= ~((ts >= excluded_start) & (ts <= excluded_end))
    left = np.flatnonzero(clean & (ts >= start - CURVE_CONTEXT_S) & (ts < start))
    right = np.flatnonzero(clean & (ts > end) & (ts <= end + CURVE_CONTEXT_S))
    left = _evenly_spaced(left, CURVE_POINTS_PER_SIDE)
    right = _evenly_spaced(right, CURVE_POINTS_PER_SIDE)
    anchors = np.r_[left, right]
    if len(left) < 2 or len(right) < 2:
        anchor_time = np.array([start, end])
        anchor_q = slerp_series(ts, qs, anchor_time)
        return slerp_series(anchor_time, anchor_q, query_time), {
            'target_method': 'endpoint_bridge',
            'curve_fallback': 'endpoint_bridge',
            'curve_anchor_count': int(len(anchors)),
            'curve_fit_rms_deg': None,
        }

    anchor_time = ts[anchors]
    reference_q = slerp_series(ts, qs, np.array([start]))[0]
    reference = np.broadcast_to(reference_q, (len(anchors), 4))
    relative = logv(qmul(qconj(reference), qs[anchors]))
    predicted = np.empty((len(query_time), 3))
    fitted_anchors = np.empty_like(relative)

    def robust_side_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        degree = min(int(CURVE_DEGREE), len(x) - 1)
        weights = np.ones(len(x))
        coefficients = None
        for _ in range(3):
            coefficients = np.polyfit(x, y, degree, w=np.sqrt(weights))
            residual = y - np.polyval(coefficients, x)
            scale = 1.4826 * np.median(np.abs(residual - np.median(residual)))
            if scale < 1e-8:
                break
            weights = np.minimum(1.0, 1.5 * scale /
                                 np.maximum(np.abs(residual), 1e-12))
        return coefficients

    duration = end - start
    u = np.clip((query_time - start) / duration, 0, 1)
    h00 = 2 * u ** 3 - 3 * u ** 2 + 1
    h10 = u ** 3 - 2 * u ** 2 + u
    h01 = -2 * u ** 3 + 3 * u ** 2
    h11 = u ** 3 - u ** 2
    left_count = len(left)
    for axis in range(3):
        left_x, right_x = ts[left] - start, ts[right] - end
        left_y = relative[:left_count, axis]
        right_y = relative[left_count:, axis]
        left_coeff = robust_side_fit(left_x, left_y)
        right_coeff = robust_side_fit(right_x, right_y)
        y0, y1 = np.polyval(left_coeff, 0.0), np.polyval(right_coeff, 0.0)
        m0 = np.polyval(np.polyder(left_coeff), 0.0)
        m1 = np.polyval(np.polyder(right_coeff), 0.0)
        predicted[:, axis] = (h00 * y0 + h10 * duration * m0 +
                              h01 * y1 + h11 * duration * m1)
        fitted_anchors[:left_count, axis] = np.polyval(left_coeff, left_x)
        fitted_anchors[left_count:, axis] = np.polyval(right_coeff, right_x)
    residual_norm = np.linalg.norm(relative - fitted_anchors, axis=1)
    fit_rms = float(np.sqrt(np.mean(residual_norm ** 2)))
    if fit_rms > CURVE_FIT_SOFT_SCALE_DEG * CURVE_FIT_FALLBACK_MULTIPLIER:
        anchor_time = np.array([start, end])
        anchor_q = slerp_series(ts, qs, anchor_time)
        return slerp_series(anchor_time, anchor_q, query_time), {
            'target_method': 'endpoint_bridge',
            'curve_fallback': 'endpoint_bridge_unreliable_cubic_fit',
            'curve_anchor_count': int(len(anchors)),
            'curve_left_anchor_count': int(len(left)),
            'curve_right_anchor_count': int(len(right)),
            'curve_left_degree_used': min(int(CURVE_DEGREE), len(left) - 1),
            'curve_right_degree_used': min(int(CURVE_DEGREE), len(right) - 1),
            'curve_fit_rms_deg': fit_rms,
        }
    target = qmul(np.broadcast_to(reference_q, (len(query_time), 4)),
                  rotation_vector_quat(predicted))
    return target, {
        'target_method': 'context_curve',
        'curve_fallback': None,
        'curve_anchor_count': int(len(anchors)),
        'curve_left_anchor_count': int(len(left)),
        'curve_right_anchor_count': int(len(right)),
        'curve_left_degree_used': min(int(CURVE_DEGREE), len(left) - 1),
        'curve_right_degree_used': min(int(CURVE_DEGREE), len(right) - 1),
        'curve_fit_rms_deg': fit_rms,
    }


def image_source_target(ts: np.ndarray, qs: np.ndarray, start: float, end: float,
                        query_time: np.ndarray,
                        gyro_telemetry: dict[str, np.ndarray],
                        source_telemetry: dict[str, np.ndarray],
                        mapping: dict) -> tuple[np.ndarray, dict]:
    """Replace gyro detail with image detail and pin both boundaries.

    Full image X/Y velocity contains translation and parallax, so integrating it
    as an absolute gyro drifts badly. Keep the gyro's low-frequency trajectory,
    replace only its local detail with measured image detail, then constrain the
    resulting orientation to the original attitude at both window boundaries.
    """
    source_time = source_telemetry['time_s']
    source_common = np.column_stack([
        source_telemetry[f'{axis}_deg_s'] for axis in 'xyz'
    ])
    gyro_common = np.column_stack([
        gyro_telemetry[f'{axis}_deg_s'] for axis in 'xyz'
    ])
    valid = (
        np.all(np.isfinite(source_common), axis=1) &
        np.all(np.isfinite(gyro_common), axis=1) &
        np.isfinite(source_time) &
        (source_telemetry['confidence'] >= CONTEXT_MIN_CONFIDENCE)
    )
    valid &= (source_time >= start - 0.1) & (source_time <= end + 0.1)
    if np.count_nonzero(valid) < 2:
        anchor_time = np.array([start, end])
        anchor_q = slerp_series(ts, qs, anchor_time)
        return slerp_series(anchor_time, anchor_q, query_time), {
            'target_method': 'image_source_endpoint_fallback',
            'image_source_sample_count': int(np.count_nonzero(valid)),
            'image_source_endpoint_drift_deg': None,
        }

    frame_dt = float(np.median(np.diff(source_time)))
    detail_window = max(3, round(0.12 / frame_dt))
    if detail_window % 2 == 0:
        detail_window += 1

    def detail(values: np.ndarray) -> np.ndarray:
        filled = values.copy()
        indices = np.arange(len(filled))
        for axis in range(3):
            good = np.isfinite(filled[:, axis])
            filled[:, axis] = np.interp(
                indices, indices[good], filled[good, axis])
        baseline = np.column_stack([
            median_filter(filled[:, axis], size=detail_window, mode='nearest')
            for axis in range(3)
        ])
        return filled - baseline

    target_common = gyro_common + detail(source_common) - detail(gyro_common)

    # compare_stabilization_telemetry stores all CSVs in the common image-axis
    # system: common[j] = raw[permutation[j]] * signs[j]. Undo that mapping.
    permutation = np.asarray(mapping['axis_permutation'], dtype=int)
    signs = np.asarray(mapping['axis_signs'], dtype=float)
    target_raw = np.empty_like(target_common)
    target_raw[:, permutation] = target_common / signs

    # The CSV stores time_s = (frame_a + 0.5) / fps, but the rates in it were
    # computed from attitudes sampled readout/2 later (see compare_stabilization_
    # telemetry.gyro_series). Integrating them on the stored grid would place the
    # replacement 6.8 ms early and invert the phase of near-Nyquist detail.
    rate_time = source_time + float(mapping.get('frame_readout_center_offset_ms', 0.0)) / 1000.0
    inside = ts[(ts > start) & (ts < end)]
    grid = np.r_[start, inside, end]
    rate = np.column_stack([
        np.interp(grid, rate_time[valid], target_raw[valid, axis])
        for axis in range(3)
    ])
    integrated = np.empty((len(grid), 4))
    integrated[0] = slerp_series(ts, qs, np.array([start]))[0]
    dt = np.diff(grid)
    midpoint_rate = (rate[:-1] + rate[1:]) * 0.5
    # The stored gyro proxy uses R_next^-1 * R_previous, hence the minus sign.
    steps = rotation_vector_quat(-midpoint_rate * dt[:, None])
    for i, step in enumerate(steps):
        integrated[i + 1] = qmul(integrated[i], step)
        integrated[i + 1] /= np.linalg.norm(integrated[i + 1])

    wanted_end = slerp_series(ts, qs, np.array([end]))
    endpoint_error = logv(qmul(qconj(integrated[-1:]), wanted_end))[0]
    u = (grid - start) / (end - start)
    endpoint_blend = u * u * (3 - 2 * u)
    constrained = qmul(
        integrated,
        rotation_vector_quat(endpoint_blend[:, None] * endpoint_error))
    constrained /= np.linalg.norm(constrained, axis=1, keepdims=True)
    return slerp_series(grid, constrained, query_time), {
        'target_method': 'image_source_detail_constrained_integration',
        'image_source_sample_count': int(np.count_nonzero(valid)),
        'image_source_endpoint_drift_deg': float(np.linalg.norm(endpoint_error)),
    }


def corrected_frame_velocity(sidecar_path: Path, telemetry: dict[str, np.ndarray],
                             mapping: dict, read_instant_ms: float = 0.0) -> np.ndarray:
    clip, samples, _ = read_telemetry(str(sidecar_path))
    t = np.array([sample.t_us / 1e6 for sample in samples])
    q = np.array([sample.q_cam for sample in samples])
    order = np.argsort(t, kind='stable')
    t, q = t[order], q[order]
    unique = np.r_[True, np.diff(t) > 0]
    t, q = t[unique], q[unique]
    fps = float(clip.fps)
    frame_a = telemetry['frame_a']
    offset = read_instant_ms / 1000
    query = np.column_stack([frame_a / fps, (frame_a + 1) / fps]) + offset
    sampled = slerp_series(t, q, query.ravel()).reshape(-1, 2, 4)
    r0 = Rotation.from_quat(sampled[:, 0][:, [1, 2, 3, 0]])
    r1 = Rotation.from_quat(sampled[:, 1][:, [1, 2, 3, 0]])
    raw = np.degrees((r1.inv() * r0).as_rotvec()) * fps
    permutation = list(mapping['axis_permutation'])
    signs = np.asarray(mapping['axis_signs'], dtype=float)
    return raw[:, permutation] * signs


def create_bugfix(folder: Path) -> dict:
    folder = Path(folder).resolve()
    outliers = json.loads((folder / 'telemetry_outliers.json').read_text(encoding='utf-8'))
    session = json.loads((folder / 'session.json').read_text(encoding='utf-8'))
    telemetry = load_csv(folder / 'telemetry_gyro.csv')
    source_telemetry = load_csv(folder / 'telemetry_image_source.csv')
    if (len(source_telemetry['time_s']) != len(telemetry['time_s']) or
            not np.allclose(source_telemetry['time_s'], telemetry['time_s'],
                            rtol=0.0, atol=1e-7)):
        raise ValueError('telemetry_gyro.csv and telemetry_image_source.csv '
                         'must use the same time grid')
    frame_time = telemetry['time_s']
    gyro_frame_values = np.column_stack([
        telemetry[f'{axis}_deg_s'] for axis in 'xyz'
    ])
    source_frame_values = np.column_stack([
        source_telemetry[f'{axis}_deg_s'] for axis in 'xyz'
    ])
    context_values_valid = (
        np.all(np.isfinite(gyro_frame_values), axis=1) &
        np.all(np.isfinite(source_frame_values), axis=1) &
        (source_telemetry['confidence'] >= CONTEXT_MIN_CONFIDENCE)
    )
    source = Path(session['source_video']['path']).resolve()
    if not source.is_file():
        raise FileNotFoundError(f'Original DJI video not found: {source}')

    clip, samples, frames = read_telemetry(str(source))
    fps, fs = float(clip.fps), float(clip.imu_sampling_rate)
    if CORRECTION_MODEL not in ('context_curve', 'endpoint_bridge', 'lowpass'):
        raise ValueError('Unknown CORRECTION_MODEL')
    if not (0 <= MIN_GAIN <= MAX_GAIN):
        raise ValueError('Invalid correction parameters at top of script')
    if CORRECTION_MODEL == 'lowpass' and not 0 < CUTOFF_HZ < fs / 2:
        raise ValueError('Invalid low-pass cutoff at top of script')
    t = np.array([sample.t_us / 1e6 for sample in samples])
    q = np.array([sample.q_cam for sample in samples])
    order = np.argsort(t, kind='stable')
    ts, qs = t[order], q[order]
    unique = np.r_[True, np.diff(ts) > 0]
    ts, qs = ts[unique], qs[unique]

    events = outliers.get('events', [])
    curve_exclusions = [
        (float(event['start_s']), float(event['end_s'])) for event in events
    ]
    event_specs = []
    support_gate = np.zeros(len(t))
    for event in events:
        start, end = float(event['start_s']), float(event['end_s'])
        if not (0 <= start < end <= len(frames) / fps):
            raise ValueError(f'Invalid bug interval: {start}..{end}')
        strength = float(event.get('strength_0_100', 100))
        if CORRECTION_MODEL == 'endpoint_bridge':
            center = (start + end) / 2
            scaled_half_duration = (end - start) * BRIDGE_RANGE_SCALE / 2
            window_start = max(0.0, center - scaled_half_duration)
            window_end = min(len(frames) / fps, center + scaled_half_duration)
            fade_base = FADE_S + GYROFLOW_MARGIN_S
        elif CORRECTION_MODEL == 'context_curve':
            window_start = max(0.0, start - CURVE_REPAIR_MARGIN_S)
            window_end = min(len(frames) / fps, end + CURVE_REPAIR_MARGIN_S)
            fade_base = FADE_S
        else:
            window_start = max(0.0, start - GYROFLOW_MARGIN_S)
            window_end = min(len(frames) / fps, end + GYROFLOW_MARGIN_S)
            fade_base = FADE_S + GYROFLOW_MARGIN_S
        fade = min(fade_base, (window_end - window_start) / 2)
        event_gate = window_gate(t, [(window_start, window_end)], fade)
        support_gate = np.maximum(support_gate, event_gate)
        event_specs.append((event, start, end, window_start, window_end,
                            strength, nonlinear_gain(strength), event_gate))

    # Never let an interval already identified as faulty influence how much it
    # should be corrected. This also excludes neighbouring detected bugs.
    detected_bug_frames = np.zeros(len(frame_time), dtype=bool)
    for _, start, end, *_ in event_specs:
        detected_bug_frames |= ((frame_time >= start) & (frame_time <= end))

    active = np.flatnonzero(support_gate > 0)
    corrected_series = q.copy()
    applied = np.empty((0, 3))
    applied_events = []
    if len(active):
        lowpass_difference = None
        if CORRECTION_MODEL == 'lowpass':
            pad = max(2.5, 5 / CUTOFF_HZ)
            filter_start = max(ts[0], t[active[0]] - pad)
            filter_end = min(ts[-1], t[active[-1]] + pad)
            grid = np.arange(filter_start, filter_end + .5 / fs, 1 / fs)
            dense = slerp_series(ts, qs, grid)
            filtered = smooth_quaternions(dense, fs, CUTOFF_HZ)
            target = slerp_series(grid, filtered, t[active])
            lowpass_difference = logv(qmul(qconj(q[active]), target))

        # Corrections from overlapping events are selected by the stronger
        # local gate instead of being added and accidentally doubled.
        applied_full = np.zeros((len(t), 3))
        priority = np.zeros(len(t))
        for (event, start, end, window_start, window_end, strength, base_gain,
             event_gate) in event_specs:
            event_indices = np.flatnonzero(event_gate > 0)
            trigger_kind = event.get('trigger_kind', 'large_impulse')
            if CORRECTION_MODEL in ('endpoint_bridge', 'context_curve'):
                # One means exactly on the endpoint-to-endpoint path. Values
                # above one would overshoot the path, so presets vary blend only.
                kind_gain_multiplier = 1.0
                candidate_gain = BRIDGE_BLEND
            else:
                kind_gain_multiplier = (MICRO_JITTER_GAIN_BOOST
                                        if trigger_kind == 'micro_jitter' else 1.0)
                candidate_gain = base_gain * kind_gain_multiplier
            curve_info = {}
            if CORRECTION_MODEL == 'endpoint_bridge':
                anchor_time = np.array([window_start, window_end])
                anchor_q = slerp_series(ts, qs, anchor_time)
                bridge = slerp_series(anchor_time, anchor_q, t[event_indices])
                event_difference = logv(qmul(qconj(q[event_indices]), bridge))
            elif CORRECTION_MODEL == 'context_curve':
                if trigger_kind in IMAGE_SOURCE_TARGET_KINDS:
                    curve, curve_info = image_source_target(
                        ts, qs, window_start, window_end, t[event_indices],
                        telemetry, source_telemetry, session['gyro_mapping'])
                else:
                    curve, curve_info = context_curve_target(
                        ts, qs, window_start, window_end, t[event_indices],
                        curve_exclusions)
                event_difference = logv(qmul(qconj(q[event_indices]), curve))
            else:
                event_difference = lowpass_difference[
                    np.searchsorted(active, event_indices)]
            proposed = event_difference.copy()
            proposed = proposed * event_gate[event_indices, None] * candidate_gain
            frame_pad = max(2 / fps, GYROFLOW_MARGIN_S)
            frame_mask = ((frame_time >= window_start - frame_pad) &
                          (frame_time <= window_end + frame_pad))
            predicted_rate = correction_rate_rms(
                proposed, t[event_indices], frame_time[frame_mask])
            context_frames = (
                (((frame_time >= window_start - CURVE_CONTEXT_S) &
                  (frame_time < window_start)) |
                 ((frame_time > window_end) &
                  (frame_time <= window_end + CURVE_CONTEXT_S))) &
                ~detected_bug_frames & context_values_valid
            )
            context_gyro = gyro_frame_values[context_frames]
            context_source = source_frame_values[context_frames]
            context_sample_count = int(np.count_nonzero(context_frames))
            if context_sample_count:
                # Consensus estimates real camera motion; their difference
                # estimates reference disagreement. Neither sees the bad span.
                context_motion = (context_gyro + context_source) * 0.5
                context_motion_rms = float(np.sqrt(
                    np.mean(np.sum(context_motion ** 2, axis=1))))
                context_agreement_error = float(np.sqrt(np.mean(np.sum(
                    (context_gyro - context_source) ** 2, axis=1))))
            else:
                # Missing clean evidence must not suppress the repair.
                context_motion_rms = 0.0
                context_agreement_error = 0.0
            limiter_predicted_rate = (
                predicted_rate
                if CORRECTION_MODEL != 'context_curve' or
                CONTEXT_CURVE_USE_RATE_GUARD else 0.0
            )
            limiter_context_motion = (
                context_motion_rms
                if CORRECTION_MODEL != 'context_curve' or
                CONTEXT_CURVE_USE_MOTION_GUARDS else 0.0
            )
            limiter_agreement_error = (
                context_agreement_error
                if CORRECTION_MODEL != 'context_curve' or
                CONTEXT_CURVE_USE_AGREEMENT_GUARD else 0.0
            )
            (rate_scale, agreement_scale, motion_scale, veto_scale,
             combined_scale) = adaptive_scale(
                limiter_predicted_rate, limiter_agreement_error,
                limiter_context_motion)
            fit_rms = curve_info.get('curve_fit_rms_deg')
            if (fit_rms is not None and
                    (CORRECTION_MODEL != 'context_curve' or
                     CONTEXT_CURVE_USE_QUALITY_GUARD)):
                fit_ratio = fit_rms / CURVE_FIT_SOFT_SCALE_DEG
                curve_quality_scale = (1 + fit_ratio ** 4) ** (-0.25)
            else:
                curve_quality_scale = 1.0
            effective_gain = candidate_gain * combined_scale * curve_quality_scale
            event_applied = event_difference * event_gate[event_indices, None]
            event_applied *= effective_gain
            event_priority = event_gate[event_indices] * effective_gain
            replace = event_priority > priority[event_indices]
            selected = event_indices[replace]
            applied_full[selected] = event_applied[replace]
            priority[selected] = event_priority[replace]
            applied_events.append({
                'id': event.get('id'), 'start_s': start, 'end_s': end,
                'trigger_kind': trigger_kind,
                'applied_window_start_s': window_start,
                'applied_window_end_s': window_end,
                'strength_0_100': strength,
                'base_nonlinear_gain': base_gain,
                'kind_gain_multiplier': kind_gain_multiplier,
                'predicted_correction_rate_rms_deg_s': predicted_rate,
                'rate_limiter_input_deg_s': limiter_predicted_rate,
                'rate_limiter_scale': rate_scale,
                'context_sample_count': context_sample_count,
                'context_gyro_source_agreement_rms_deg_s': context_agreement_error,
                'agreement_limiter_input_deg_s': limiter_agreement_error,
                'agreement_scale': agreement_scale,
                'context_motion_rms_deg_s': context_motion_rms,
                'motion_limiter_input_deg_s': limiter_context_motion,
                'motion_interaction_scale': motion_scale,
                'high_motion_veto_scale': veto_scale,
                'adaptive_scale': combined_scale,
                'curve_quality_scale': float(curve_quality_scale),
                'correction_gain': effective_gain,
                **curve_info,
            })
        # A bridge has a geometrically defined target, but reaching it during a
        # fast pan means erasing real acceleration: on the Pro clip seven
        # large_impulse events received 1.6-10.7 deg of edit, a 40-250 px lurch
        # that the 0.12 s impulse metric cannot see. Bound every model.
        applied = limit_rotation(applied_full[active], MAX_CORRECTION_DEG)
        corrected = qmul(q[active], rotation_vector_quat(applied))
        corrected_series[active] = corrected

    # Keep the repaired quaternion curve identical and vary only its time
    # placement. This isolates content repair from external-sidecar timing.
    corrected_sorted = corrected_series[order][unique]
    frame_of = np.array([sample.frame for sample in samples], dtype=int)
    readout_ms = float(clip.frame_readout_time_ms or 0.0)
    if DEFAULT_SHUTTER_VARIANT not in ('start', 'center', 'end'):
        raise ValueError('DEFAULT_SHUTTER_VARIANT must be start, center or end')
    shutter_instants = {'start': 0.0, 'center': readout_ms / 2, 'end': readout_ms}
    base_read_instant_ms = shutter_instants[DEFAULT_SHUTTER_VARIANT]
    _, _, _, _, _, _, base_delta_us = alignment_corrections(
        str(source), read_instant_ms=base_read_instant_ms)

    mean_delta_us = float(np.median(base_delta_us))
    timing_variants = {
        'none': {
            'path': folder / 'telemetry_bugfix_timing_none.mp4',
            'delta_us': np.zeros_like(base_delta_us),
            'strategy': 'No quaternion-content retiming.',
        },
        'mean': {
            'path': folder / 'telemetry_bugfix_timing_mean.mp4',
            'delta_us': np.full_like(base_delta_us, mean_delta_us),
            'strategy': 'One median DJI-reference shift for the whole clip.',
        },
        'per_frame': {
            'path': folder / 'telemetry_bugfix.mp4',
            'delta_us': base_delta_us,
            'strategy': 'Current DJI-reference shift, independently per frame.',
        },
        'manual_end': {
            'path': folder / 'telemetry_bugfix_timing_manual_end.mp4',
            'delta_us': np.full_like(
                base_delta_us, -base_read_instant_ms * 1000.0),
            'strategy': ('Constant manual shift corresponding to the visual '
                         'end-of-readout winner; no per-frame dbgi component.'),
        },
    }
    variant_reports = {}
    for name, variant in timing_variants.items():
        variant_path = variant['path']
        delta_us = variant['delta_us']
        aligned = slerp_series(
            ts, corrected_sorted, t + delta_us[frame_of] / 1e6)
        edits = {}
        for sample, aligned_q in zip(samples, aligned):
            if sample.inverted:
                aligned_q = -aligned_q
            edits[sample.frame, sample.index_in_frame] = quat.norm(
                quat.gyroflow_to_dji(tuple(map(float, aligned_q))))
        info = sidecar.build(str(source), str(variant_path), edits)
        variant_reports[name] = {
            'path': str(variant_path),
            'strategy': variant['strategy'],
            'base_shutter_variant': DEFAULT_SHUTTER_VARIANT,
            'assumed_read_instant_ms': float(base_read_instant_ms),
            'alignment_shift_mean_ms': float(np.mean(delta_us) / 1000),
            'alignment_shift_std_ms': float(np.std(delta_us) / 1000),
            'alignment_frame_step_rms_ms': float(
                np.sqrt(np.mean(np.diff(delta_us) ** 2)) / 1000
                if len(delta_us) > 1 else 0.0),
            'alignment_frame_step_peak_ms': float(
                np.max(np.abs(np.diff(delta_us))) / 1000
                if len(delta_us) > 1 else 0.0),
            'edited_records': int(info['edited_records']),
        }

    output_mp4 = timing_variants['per_frame']['path']
    sidecar_info = variant_reports['per_frame']
    # Sampled the same way telemetry_gyro.csv was, so the two files compare.
    fixed_velocity = corrected_frame_velocity(
        output_mp4, telemetry, session['gyro_mapping'],
        read_instant_ms=float(session['gyro_mapping'].get('frame_readout_center_offset_ms', 0.0)))
    output_csv = folder / 'telemetry_gyro_fixed.csv'
    write_csv(output_csv, telemetry, fixed_velocity)

    magnitude = np.linalg.norm(applied, axis=1) if len(applied) else np.array([0.0])
    report = {
        'format_version': 2,
        'session': str(folder),
        'source_video': str(source),
        'input_files': ['telemetry_outliers.json', 'telemetry_gyro.csv',
                        'telemetry_image_source.csv', 'session.json'],
        'output_sidecar': str(output_mp4),
        'default_timing_variant': 'per_frame',
        'default_shutter_variant': DEFAULT_SHUTTER_VARIANT,
        'timing_variants': variant_reports,
        'output_csv': str(output_csv),
        'events': applied_events,
        'event_count': len(applied_events),
        'preset': PRESET,
        'correction_model': CORRECTION_MODEL,
        'edited_records': int(sidecar_info['edited_records']),
        'frame_readout_time_ms': readout_ms,
        'cutoff_hz': CUTOFF_HZ,
        'fade_s': FADE_S,
        'gain_range': [MIN_GAIN, MAX_GAIN],
        'max_correction_limit_deg': MAX_CORRECTION_DEG,
        'adaptive_parameters': {
            'strength_gamma': STRENGTH_GAMMA,
            'rate_soft_limit_deg_s': RATE_SOFT_LIMIT_DEG_S,
            'rate_compression_power': RATE_COMPRESSION_POWER,
            'agreement_free_deg_s': AGREEMENT_FREE_DEG_S,
            'agreement_soft_scale_deg_s': AGREEMENT_SOFT_SCALE_DEG_S,
            'motion_interaction_start_deg_s': MOTION_INTERACTION_START_DEG_S,
            'gyro_motion_soft_scale_deg_s': GYRO_MOTION_SOFT_SCALE_DEG_S,
            'high_motion_veto_start_deg_s': HIGH_MOTION_VETO_START_DEG_S,
            'high_motion_veto_rate_start_deg_s': HIGH_MOTION_VETO_RATE_START_DEG_S,
            'high_motion_veto_strength': HIGH_MOTION_VETO_STRENGTH,
            'gyroflow_margin_s': GYROFLOW_MARGIN_S,
            'minimum_adaptive_scale': MINIMUM_ADAPTIVE_SCALE,
            'micro_jitter_gain_boost': MICRO_JITTER_GAIN_BOOST,
            'bridge_blend': BRIDGE_BLEND,
            'bridge_range_scale': BRIDGE_RANGE_SCALE,
            'curve_context_s': CURVE_CONTEXT_S,
            'curve_repair_margin_s': CURVE_REPAIR_MARGIN_S,
            'curve_points_per_side': CURVE_POINTS_PER_SIDE,
            'curve_degree': CURVE_DEGREE,
            'curve_fit_soft_scale_deg': CURVE_FIT_SOFT_SCALE_DEG,
            'curve_fit_fallback_multiplier': CURVE_FIT_FALLBACK_MULTIPLIER,
            'context_min_confidence': CONTEXT_MIN_CONFIDENCE,
            'context_curve_use_rate_guard': CONTEXT_CURVE_USE_RATE_GUARD,
            'context_curve_use_motion_guards': CONTEXT_CURVE_USE_MOTION_GUARDS,
            'context_curve_use_agreement_guard': CONTEXT_CURVE_USE_AGREEMENT_GUARD,
            'context_curve_use_quality_guard': CONTEXT_CURVE_USE_QUALITY_GUARD,
            'image_source_target_kinds': list(IMAGE_SOURCE_TARGET_KINDS),
        },
        'peak_applied_correction_deg': float(magnitude.max()),
        'rms_applied_correction_deg': float(np.sqrt(np.mean(magnitude ** 2))),
        'note': ('Gyro-fault and micro-jitter events use constrained integration '
                 'of source-image motion detail; large impulses fit a context '
                 'attitude curve. Motion and agreement guards '
                 'use only clean context outside every detected bug. Rate, motion, '
                 'agreement and fit-quality suppression are disabled for '
                 'context_curve. The '
                 'default timing variant is '
                 'the visual winner with Gyroflow rolling-shutter correction on.'),
    }
    (folder / 'telemetry_bugfix.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path, nargs='?', default=Path(SESSION_DIR),
                        help='session folder (default: SESSION_DIR at top of script)')
    args = parser.parse_args()
    folder = args.folder if args.folder.is_absolute() else ROOT / args.folder
    report = create_bugfix(folder)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
