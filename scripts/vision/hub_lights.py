"""Which HUB is active in each 2026 shift, read from the physical HUB lights (no broadcast overlay, no scores).

The light boxes come from the calibration's height projection: the lit band along the top of each HUB's front face.
Readings are calibrated per match across teleop, and the first active alliance must be clear and agree with the
published shift timing at every phase; otherwise the schedule stays unknown. The light boxes are derived from geometry
instead of hand-set.
"""
import cv2
import numpy as np

TELEOP_START_S = 23.0        # match time; auto 20 s + 3 s grace, matching fuel_count's periods
FACE_HALF_WIDTH_M = 0.6      # half the HUB's front face width
LIGHT_BAND_Z_M = (1.75, 1.85)  # the lit rim band (rim at 1.829 m); clearest two-state split on 2026cc qm1 and qm3


def for_hub(calibration, hub_name):
    """The calibration with `height_projection` set to `hub_name`'s own one when it has per-HUB projections
    (hub_local_calibration.py: wide distorted broadcast lenses get one local camera per HUB)."""
    by_hub = (calibration or {}).get('height_projection_by_hub') or {}
    if hub_name in by_hub:
        return {**calibration, 'height_projection': {**(calibration.get('height_projection') or {}), **by_hub[hub_name]}}
    return calibration


def height_project(calibration, points, width=None, height=None):
    """(N, 2) picture pixels for (N, 3) field points (x, y, z m) through the calibration's height projection,
    scaled to a `width` x `height` frame; None when the calibration has no height projection."""
    hp = (calibration or {}).get('height_projection')
    if not hp or not hp.get('floor_to_image') or not hp.get('height_column'):
        return None
    h = np.asarray(hp['floor_to_image'], float)
    col = np.asarray(hp['height_column'], float)
    pts = np.asarray(points, float).reshape(-1, 3)
    homog = (h[:, :2] @ pts[:, :2].T) + h[:, 2:3] + np.outer(col, pts[:, 2])
    uv = (homog[:2] / homog[2]).T
    sw = float(calibration.get('source_width') or width or 1)
    sh = float(calibration.get('source_height') or height or 1)
    if width and height:
        uv = uv * np.array([width / sw, height / sh])
    return uv


def light_rois(calibration, hubs, front_y_sign=None):
    """{'red': [x0, y0, x1, y1], 'blue': [...]} normalized light boxes, or None without a height projection. The lit
    face is the one toward the camera: +y (Chezy Champs' broadcast) unless the calibration says
    `light_face_y_sign: -1` (a camera on the y = 0 side, as at 2026arc and 2026caclv)."""
    if front_y_sign is None:
        front_y_sign = float(calibration.get('light_face_y_sign', 1.0))
    sw = float(calibration.get('source_width') or 1920)
    sh = float(calibration.get('source_height') or 1080)
    rois = {}
    for name, (x, y) in hubs.items():
        side = 'red' if 'red' in name else 'blue'
        face_y = y + front_y_sign * FACE_HALF_WIDTH_M
        band = calibration.get('light_band_z') or LIGHT_BAND_Z_M   # tag calibrations sample the lit panels themselves
        corners = [(x + dx, face_y, z) for dx in (-FACE_HALF_WIDTH_M * 0.8, FACE_HALF_WIDTH_M * 0.8) for z in band]
        uv = height_project(for_hub(calibration, name), corners)
        if uv is None:
            return None
        u0, v0 = uv.min(0)
        u1, v1 = uv.max(0)
        rois[side] = [max(0.0, u0 / sw), max(0.0, v0 / sh), min(1.0, u1 / sw), min(1.0, v1 / sh)]
    return rois


def _light_score(frame, roi, side):
    """Color advantage of a lit band over its neutral channel (BGR)."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = roi
    patch = frame[round(y1 * height):round(y2 * height), round(x1 * width):round(x2 * width)]
    if patch.size == 0:
        raise ValueError('light box is empty')
    channel = 2 if side == 'red' else 0
    return float(np.median(patch[:, :, channel].astype(np.int16) - patch[:, :, 1].astype(np.int16)))


def _threshold(scores):
    low, high = np.percentile(scores, (20, 80))
    if high - low < 18:
        raise ValueError('HUB light does not show two clear states')
    return low + .35 * (high - low)


def phase_intervals(shift1_side, teleop_start_s=TELEOP_START_S):
    other = 'blue' if shift1_side == 'red' else 'red'
    s = teleop_start_s
    return [(s, s + 10, 'both'), (s + 10, s + 35, shift1_side), (s + 35, s + 60, other),
            (s + 60, s + 85, shift1_side), (s + 85, s + 110, other), (s + 110, s + 140, 'both')]


def shift1_from_scores(observations, teleop_start_s=TELEOP_START_S):
    """observations: [(offset_s after teleop start, {'red': score, 'blue': score})]. Returns (side, evidence) where
    side is 'red' | 'blue' (whose hub is active in shift 1) or 'unknown' with the reason in evidence."""
    try:
        thresholds = {side: _threshold([v[side] for _, v in observations]) for side in ('red', 'blue')}
    except ValueError as error:
        return 'unknown', {'reason': str(error)}

    def state(values):
        active = [side for side in ('red', 'blue') if values[side] >= thresholds[side]]
        return active[0] if len(active) == 1 else 'both' if len(active) == 2 else 'none'

    shift1 = [state(v) for off, v in observations if 15 <= off <= 30]
    first = max(('red', 'blue'), key=shift1.count)
    if shift1.count(first) < 3:
        return 'unknown', {'reason': 'first active alliance not clear in the HUB lights', 'shift1_states': shift1}
    for begin, end, expected in phase_intervals(first, 0.0):
        mid = (begin + end) / 2
        nearby = [state(v) for off, v in observations if abs(off - mid) < 5]
        if nearby.count(expected) < max(1, (3 * len(nearby) + 4) // 5):
            return 'unknown', {'reason': f'HUB lights disagree with the shift timing at teleop +{mid:.0f}s', 'states': nearby}
    return first, {'thresholds': {k: round(v, 1) for k, v in thresholds.items()}, 'samples': len(observations)}


def shift1_from_video(video_path, rois, match_start_s=0.0, teleop_start_s=TELEOP_START_S):
    """Read the HUB lights every 5 s across teleop. Returns (side, evidence) as shift1_from_scores."""
    if not rois or set(rois) != {'red', 'blue'}:
        return 'unknown', {'reason': 'no HUB light boxes (calibration has no height projection)'}
    cap = cv2.VideoCapture(str(video_path))
    observations = []
    try:
        for offset in range(5, 140, 5):
            cap.set(cv2.CAP_PROP_POS_MSEC, (match_start_s + teleop_start_s + offset) * 1000)
            ok, frame = cap.read()
            if not ok:
                return 'unknown', {'reason': f'cannot read the frame at teleop +{offset}s'}
            observations.append((offset, {side: _light_score(frame, rois[side], side) for side in ('red', 'blue')}))
    finally:
        cap.release()
    side, evidence = shift1_from_scores(observations, teleop_start_s)
    evidence['rois'] = rois
    return side, evidence


def expected_state(t_match_s, shift1_side, teleop_start_s=TELEOP_START_S):
    """The lit HUBs at match second `t_match_s` in teleop: 'both', 'red', 'blue', or None outside teleop."""
    for begin, end, state in phase_intervals(shift1_side, teleop_start_s):
        if begin <= t_match_s < end:
            return state
    return None


def light_series(video_path, rois, start_s=0.0, end_s=None, step_s=0.25):
    """[(video second, {'red': score, 'blue': score})] sampled every `step_s` (decoding sequentially, so it is fast)."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps if cap.get(cv2.CAP_PROP_FRAME_COUNT) else None
    end_s = end_s if end_s is not None else (total or start_s + 240.0)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)
    series, next_t, i = [], start_s, int(round(start_s * fps))
    try:
        while True:
            if not cap.grab():                 # decode every frame; convert only the sampled ones (read = grab + retrieve)
                break
            t = i / fps
            i += 1
            if t > end_s:
                break
            if t + 1e-6 >= next_t:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                series.append((t, {side: _light_score(frame, rois[side], side) for side in ('red', 'blue')}))
                next_t += step_s
    finally:
        cap.release()
    return series


def match_start_from_lights(series, search=(0.0, 90.0), step_s=0.25, min_agreement=0.8):
    """(match start video second, shift1 side, evidence) from the HUB lights alone, or (None, 'unknown', evidence).

    The lights follow a fixed teleop pattern (transition both lit, four 25 s shifts alternating, endgame both lit),
    so the start is the offset whose pattern agrees best with the observed lit states. No scoreboard or clock."""
    if len(series) < 40:
        return None, 'unknown', {'reason': 'too few light samples'}
    try:
        thresholds = {side: _threshold([v[side] for _, v in series]) for side in ('red', 'blue')}
    except ValueError as error:
        return None, 'unknown', {'reason': str(error)}
    times = np.array([t for t, _ in series])
    lit = {side: np.array([v[side] >= thresholds[side] for _, v in series]) for side in ('red', 'blue')}
    best = (-1.0, None, None)
    for t0 in np.arange(search[0], search[1] + 1e-9, step_s):
        for side in ('red', 'blue'):
            agree = n = 0
            for k, t in enumerate(times):
                state = expected_state(t - t0, side)
                if state is None:
                    continue
                want_red, want_blue = state in ('both', 'red'), state in ('both', 'blue')
                agree += int(lit['red'][k] == want_red) + int(lit['blue'][k] == want_blue)   # numpy bools: + is OR
                n += 2
            if n >= 400 and agree / n > best[0]:      # at least ~100 s of teleop inside the video
                best = (agree / n, float(t0), side)
    score, t0, side = best
    evidence = {'agreement': round(score, 3), 'thresholds': {k: round(v, 1) for k, v in thresholds.items()}}
    if t0 is None or score < min_agreement:
        return None, 'unknown', {**evidence, 'reason': 'HUB lights do not follow the shift pattern clearly'}
    return t0, side, evidence
