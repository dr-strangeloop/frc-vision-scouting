"""Pixels -> field meters, with a per-point uncertainty."""
from bisect import bisect_left, bisect_right
import math
import re

import numpy as np
from scipy.optimize import linear_sum_assignment

from common import FIELD
from terrain_projection import ray_terrain_contact


def apply_h(h, point):
    v = h @ np.array([point[0], point[1], 1.0])
    return np.array([v[0] / v[2], v[1] / v[2]])


def _camera_center(camera):
    if camera is None:
        return None
    if hasattr(camera, 'center'):
        center = np.asarray(camera.center, dtype=float)
    elif isinstance(camera, dict):
        raw = camera.get('camera_center', camera.get('center'))
        if raw is None:
            return None
        center = np.asarray(raw, dtype=float)
    else:
        center = np.asarray(camera, dtype=float)
    if center.shape == (3,) and np.isfinite(center).all() and 2.0 < float(center[2]) < 40.0:
        return center
    return None


def ground_point(
    box_px,
    h_cur_to_field,
    inset_m=0.4,
    px_sigma=2.0,
    camera=None,
    contact_z_m=0.0,
    apply_terrain=False,
    calibration_rms_m=0.0,
    roi_bounds_px=None,
):
    """Field position of a robot's footprint centre from its pixel box.

    The bottom edge of a bumper box is where the robot meets the carpet, so that edge's midpoint
    is on the ground plane and the homography is valid there (the box centre is up in the air, and
    projecting it would put the robot far behind where it really is). That point is the near edge
    of the footprint, so we step half a robot depth (inset_m) away from the camera, along the
    camera viewing ray (when a 3D camera pose is available) or the direction "up the image" lands
    on the field, to reach the centre.

    Returns (x, y, sigma_m): sigma is how many meters one pixel of detection noise is worth at
    this spot, which is large for a robot far from a low camera. The smoother weights by it.
    """
    x1, y1, x2, y2 = box_px
    bottom = np.array([(x1 + x2) / 2.0, y2], dtype=float)
    raw_base = apply_h(h_cur_to_field, bottom)
    base = raw_base.copy()
    surface_z = 0.0
    if apply_terrain or (contact_z_m is not None and float(contact_z_m) > 0.0):
        bx, by, surface_z = ray_terrain_contact(
            bottom,
            h_cur_to_field,
            camera=camera,
            contact_z_m=0.0 if contact_z_m is None else float(contact_z_m),
            apply_bump=bool(apply_terrain),
        )
        base = np.array([bx, by], dtype=float)

    cam_center = _camera_center(camera)
    h_arr = np.asarray(h_cur_to_field, dtype=float)
    is_norm = (
        all(math.isfinite(float(v)) for v in box_px)
        and max(abs(float(v)) for v in box_px) <= 1.5
        and (abs(float(h_arr[0, 0])) > 1.0 or abs(float(h_arr[1, 1])) > 1.0)
    )
    up_delta = np.array([0.0, -10.0 / 720.0]) if is_norm else np.array([0.0, -10.0])
    step_dx = np.array([1.0 / 1280.0, 0.0]) if is_norm else np.array([1.0, 0.0])
    step_dy = np.array([0.0, 1.0 / 720.0]) if is_norm else np.array([0.0, 1.0])

    if cam_center is not None:
        ray_vec = base - cam_center[:2]
        norm = float(np.linalg.norm(ray_vec))
        direction = ray_vec / norm if norm > 1e-9 else np.zeros(2)
    else:
        up = apply_h(h_cur_to_field, bottom + up_delta) - raw_base
        norm = float(np.linalg.norm(up))
        direction = up / norm if norm > 1e-9 else np.zeros(2)

    centre = base + direction * inset_m
    step_x = np.linalg.norm(apply_h(h_cur_to_field, bottom + step_dx) - raw_base)
    step_y = np.linalg.norm(apply_h(h_cur_to_field, bottom + step_dy) - raw_base)
    sigma = float(px_sigma * math.sqrt((step_x ** 2 + step_y ** 2) / 2.0))
    if calibration_rms_m and float(calibration_rms_m) > 0.0:
        sigma = float(math.hypot(sigma, float(calibration_rms_m)))
    if surface_z > 0.0:
        sigma = max(sigma, 0.25)
    if roi_bounds_px is not None and len(roi_bounds_px) == 4:
        _, _, _, roi_y1 = roi_bounds_px
        width_px = float(x2 - x1)
        height_px = float(y2 - y1)
        if width_px > 0.0 and float(y2) >= float(roi_y1) - 6.0 and (width_px / max(1.0, height_px)) > 1.85:
            sigma *= 2.0
    return float(centre[0]), float(centre[1]), max(0.05, min(sigma, 3.0))


def in_field(x, y, margin=0.5):
    return -margin <= x <= FIELD['length'] + margin and -margin <= y <= FIELD['width'] + margin


def _hub_centres():
    hx, hy = float(FIELD['hub']['x']), float(FIELD['hub']['y'])
    return ((hx, hy), (float(FIELD['length']) - hx, float(FIELD['width']) - hy))


def inside_hub_core(x, y, margin=0.15):
    """True when (x, y) lies inside the solid interior of the Blue or Red Hub."""
    half = max(0.0, float(FIELD['hub']['halfSize']) - float(margin))
    fx, fy = float(x), float(y)
    return any(abs(fx - cx) < half and abs(fy - cy) < half for cx, cy in _hub_centres())


def clamp_outside_hub_core(x, y, margin=0.15):
    """If (x, y) lies inside a Hub core, push it to the nearest Hub boundary point."""
    fx, fy = float(x), float(y)
    half = max(0.0, float(FIELD['hub']['halfSize']) - float(margin))
    for cx, cy in _hub_centres():
        dx = fx - cx
        dy = fy - cy
        if abs(dx) < half and abs(dy) < half:
            if (half - abs(dx)) <= (half - abs(dy)):
                clamped_x = cx + (half if dx >= 0.0 else -half)
                return float(clamped_x), float(fy), True
            clamped_y = cy + (half if dy >= 0.0 else -half)
            return float(fx), float(clamped_y), True
    return fx, fy, False


def project_contact(
    contact_px,
    h_cur_to_field,
    px_sigma=2.0,
    camera=None,
    contact_z_m=0.0,
    apply_terrain=False,
    calibration_rms_m=0.0,
):
    """Project a supplied visible floor contact, without inventing a footprint-centre offset.

    A box bottom is only an estimate of contact. Raised/occluded contacts need separate
    evidence; a floor homography cannot recover their height.
    """
    if len(contact_px) != 2 or not all(math.isfinite(v) for v in contact_px):
        raise ValueError('contact must contain two finite image coordinates')
    x, y = contact_px
    return ground_point(
        [x, y, x, y],
        h_cur_to_field,
        inset_m=0,
        px_sigma=px_sigma,
        camera=camera,
        contact_z_m=contact_z_m,
        apply_terrain=apply_terrain,
        calibration_rms_m=calibration_rms_m,
    )


def _clean_trajectory(track_or_points):
    pts = track_or_points.get('trajectory', []) if isinstance(track_or_points, dict) else (track_or_points or [])
    out = []
    for p in pts:
        if not isinstance(p, dict) or p.get('calibrated', True) is False:
            continue
        t = p.get('t', p.get('timestamp_ms'))
        x, y = p.get('x'), p.get('y')
        if t is None or x is None or y is None:
            continue
        try:
            ft, fx, fy = float(t), float(x), float(y)
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(ft) and math.isfinite(fx) and math.isfinite(fy)):
            continue
        out.append(p)
    out.sort(key=lambda p: float(p.get('t', p.get('timestamp_ms'))))
    return out


def _pair_trajectory_samples(pts_a, pts_b, max_dt_ms):
    if not pts_a or not pts_b:
        return [], set(), set()
    tb_vals = [float(pb.get('t', pb.get('timestamp_ms'))) for pb in pts_b]
    candidates = []
    for i, pa in enumerate(pts_a):
        ta = float(pa.get('t', pa.get('timestamp_ms')))
        lo = bisect_left(tb_vals, ta - float(max_dt_ms))
        hi = bisect_right(tb_vals, ta + float(max_dt_ms))
        for j in range(lo, hi):
            dt = abs(ta - tb_vals[j])
            if dt <= float(max_dt_ms):
                candidates.append((dt, i, j))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    used_a, used_b = set(), set()
    pairs = []
    for _, i, j in candidates:
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            pairs.append((pts_a[i], pts_b[j]))
    pairs.sort(key=lambda pair: float(pair[0].get('t', pair[0].get('timestamp_ms'))))
    return pairs, used_a, used_b


def _norm_team_key(team_key):
    if team_key is None:
        return None
    s = str(team_key).strip()
    if not s:
        return None
    return re.sub(r'^frc', '', s, flags=re.I)


def evaluate_multiview_consistency(tracks_a, tracks_b, max_dt_ms=250, max_median_m=0.50, max_p95_m=0.80):
    """Evaluate spatial agreement between calibrated trajectories across two views."""
    named_a, named_b = {}, {}
    display_key = {}
    unnamed_a, unnamed_b = [], []

    for tr in tracks_a or []:
        pts = _clean_trajectory(tr)
        if not pts:
            continue
        nk = _norm_team_key(tr.get('team_key') if isinstance(tr, dict) else None)
        if nk:
            display_key.setdefault(nk, str(tr.get('team_key')).strip())
            named_a.setdefault(nk, []).extend(pts)
        else:
            unnamed_a.append((tr if isinstance(tr, dict) else {}, pts))

    for tr in tracks_b or []:
        pts = _clean_trajectory(tr)
        if not pts:
            continue
        nk = _norm_team_key(tr.get('team_key') if isinstance(tr, dict) else None)
        if nk:
            display_key.setdefault(nk, str(tr.get('team_key')).strip())
            named_b.setdefault(nk, []).extend(pts)
        else:
            unnamed_b.append((tr if isinstance(tr, dict) else {}, pts))

    paired_trajs = []
    for nk in sorted(set(named_a).intersection(named_b)):
        pts_a = sorted(named_a[nk], key=lambda p: float(p.get('t', p.get('timestamp_ms'))))
        pts_b = sorted(named_b[nk], key=lambda p: float(p.get('t', p.get('timestamp_ms'))))
        paired_trajs.append((display_key[nk], pts_a, pts_b))

    if unnamed_a and unnamed_b:
        cost = np.full((len(unnamed_a), len(unnamed_b)), 1e6, dtype=float)
        for i, (ta, pts_a) in enumerate(unnamed_a):
            for j, (tb, pts_b) in enumerate(unnamed_b):
                if ta.get('alliance') and tb.get('alliance') and ta.get('alliance') != tb.get('alliance'):
                    continue
                pairs, _, _ = _pair_trajectory_samples(pts_a, pts_b, max_dt_ms)
                if pairs:
                    dists = [math.hypot(float(pa['x']) - float(pb['x']), float(pa['y']) - float(pb['y'])) for pa, pb in pairs]
                    cost[i, j] = float(np.median(dists))
        row_ind, col_ind = linear_sum_assignment(cost)
        for idx, (i, j) in enumerate(zip(row_ind, col_ind)):
            if cost[i, j] < 1e5:
                ta, pts_a = unnamed_a[i]
                key = str(ta.get('track_key') or f'unnamed_{idx}')
                paired_trajs.append((key, pts_a, unnamed_b[j][1]))

    per_team = {}
    all_errors = []
    for key, pts_a, pts_b in paired_trajs:
        pairs, _, _ = _pair_trajectory_samples(pts_a, pts_b, max_dt_ms)
        if not pairs:
            continue
        errors = [math.hypot(float(pa['x']) - float(pb['x']), float(pa['y']) - float(pb['y'])) for pa, pb in pairs]
        all_errors.extend(errors)
        med = float(np.median(errors))
        p95 = float(np.percentile(errors, 95))
        mx = float(np.max(errors))
        per_team[key] = {
            'samples': len(errors),
            'median_error_m': round(med, 4),
            'p95_error_m': round(p95, 4),
            'max_error_m': round(mx, 4),
            'consistent': bool(med <= float(max_median_m) and p95 <= float(max_p95_m)),
        }

    if not all_errors:
        return {
            'paired_teams': len(per_team),
            'paired_samples': 0,
            'median_error_m': None,
            'p95_error_m': None,
            'max_error_m': None,
            'consistent': False,
            'per_team': per_team,
        }

    median_err = round(float(np.median(all_errors)), 4)
    p95_err = round(float(np.percentile(all_errors, 95)), 4)
    max_err = round(float(np.max(all_errors)), 4)
    return {
        'paired_teams': len(per_team),
        'paired_samples': len(all_errors),
        'median_error_m': median_err,
        'p95_error_m': p95_err,
        'max_error_m': max_err,
        'consistent': bool(median_err <= float(max_median_m) and p95_err <= float(max_p95_m)),
        'per_team': per_team,
    }


def fuse_multiview_trajectories(traj_a, traj_b, max_dt_ms=150):
    """Inverse-variance (1 / sigma^2) fusion of two calibrated trajectories across views."""
    pts_a = _clean_trajectory(traj_a)
    pts_b = _clean_trajectory(traj_b)
    pairs, used_a, used_b = _pair_trajectory_samples(pts_a, pts_b, max_dt_ms)
    fused = []
    for pa, pb in pairs:
        sa = max(0.05, float(pa.get('sigma', 0.25)))
        sb = max(0.05, float(pb.get('sigma', 0.25)))
        wa = 1.0 / (sa * sa)
        wb = 1.0 / (sb * sb)
        w_sum = wa + wb
        fx = (wa * float(pa['x']) + wb * float(pb['x'])) / w_sum
        fy = (wa * float(pa['y']) + wb * float(pb['y'])) / w_sum
        fsigma = math.sqrt(1.0 / w_sum)
        ta = float(pa.get('t', pa.get('timestamp_ms')))
        tb = float(pb.get('t', pb.get('timestamp_ms')))
        raw_t = (wa * ta + wb * tb) / w_sum
        ft = int(round(raw_t)) if isinstance(pa.get('t'), int) and isinstance(pb.get('t'), int) else round(raw_t, 3)
        disagreement = math.hypot(float(pa['x']) - float(pb['x']), float(pa['y']) - float(pb['y']))
        fused.append({
            **pa,
            't': ft,
            'x': round(fx, 3),
            'y': round(fy, 3),
            'sigma': round(max(0.05, fsigma), 3),
            'calibrated': True,
            'fused_views': 2,
            'view_disagreement_m': round(disagreement, 3),
        })
    for i, pa in enumerate(pts_a):
        if i not in used_a:
            fused.append({**pa, 't': pa.get('t', pa.get('timestamp_ms')), 'fused_views': 1})
    for j, pb in enumerate(pts_b):
        if j not in used_b:
            fused.append({**pb, 't': pb.get('t', pb.get('timestamp_ms')), 'fused_views': 1})
    fused.sort(key=lambda p: float(p.get('t', p.get('timestamp_ms'))))
    return fused


START_ZONE_MAX_X = float(FIELD['trench']['x1']) + 0.6


def start_position_at(y_norm):
    """Map an alliance-normalized y coordinate (m) to its scouting start position name."""
    offset = float(y_norm) - float(FIELD['hub']['y'])
    mag = abs(offset)
    side = 'left' if offset >= 0.0 else 'right'
    if mag <= float(FIELD['hub']['halfSize']):
        return 'center'
    if mag <= float(FIELD['bump']['yOuter']):
        return f'{side} mound'
    return f'{side} trench'


def canonical_start_zone_at(y_norm):
    """Map an alliance-normalized y coordinate (m) to canonical start_zone form."""
    pos = start_position_at(y_norm)
    return {
        'left trench': 'trench_left',
        'left mound': 'bump_left',
        'center': 'center',
        'right mound': 'bump_right',
        'right trench': 'trench_right',
    }[pos]


def in_start_zone(x_norm, max_x=None):
    limit = START_ZONE_MAX_X if max_x is None else float(max_x)
    return math.isfinite(float(x_norm)) and -0.5 <= float(x_norm) <= limit


def infer_start_pose(points, alliance='blue', *, max_start_ms=3000.0, max_zone_x=6.3):
    """Infer a robot's t = 0 autonomous starting pose in both normalized and field coordinates.

    Matches src/lib/vision/fieldRebuilt.js:inferStartPose: uses pre-roll points (-1200..0 ms)
    when available, and de-biases sideline camera near-bumper (+y) offsets, overhead trench-bar
    clipping, and initial motion lag for post-start detections.
    """
    al = str(alliance or 'blue').strip().lower()
    length = float(FIELD['length'])
    width = float(FIELD['width'])
    norm = []
    for p in points or ():
        if not isinstance(p, dict) or p.get('calibrated') is False:
            continue
        t = p.get('t', p.get('t_ms', p.get('timestamp_ms')))
        x, y = p.get('x'), p.get('y')
        if t is None or x is None or y is None:
            continue
        ft, fx, fy = float(t), float(x), float(y)
        if not (math.isfinite(ft) and math.isfinite(fx) and math.isfinite(fy)):
            continue
        nx = length - fx if al == 'red' else fx
        ny = width - fy if al == 'red' else fy
        norm.append({'t': ft, 'x': nx, 'y': ny})
    norm.sort(key=lambda item: item['t'])
    if not norm:
        return None

    pre = [p for p in norm if -1200.0 <= p['t'] <= 0.0 and in_start_zone(p['x'], max_zone_x)]
    if pre:
        xs = sorted(p['x'] for p in pre)
        ys = sorted(p['y'] for p in pre)
        x0 = xs[len(xs) // 2] if len(xs) % 2 else 0.5 * (xs[len(xs) // 2 - 1] + xs[len(xs) // 2])
        y0 = ys[len(ys) // 2] if len(ys) % 2 else 0.5 * (ys[len(ys) // 2 - 1] + ys[len(ys) // 2])
        fx0 = length - x0 if al == 'red' else x0
        fy0 = width - y0 if al == 'red' else y0
        return {
            'x': round(x0, 3),
            'y': round(y0, 3),
            'field_x': round(fx0, 3),
            'field_y': round(fy0, 3),
            'rawX': round(pre[0]['x'], 3),
            'rawY': round(pre[0]['y'], 3),
            'position': start_position_at(y0),
            'canonical_zone': canonical_start_zone_at(y0),
            'atMs': 0.0,
            'inAllianceZone': in_start_zone(x0, max_zone_x),
            'backExtrapolated': False,
        }

    post = [p for p in norm if 0.0 <= p['t'] <= float(max_start_ms)]
    if not post:
        return None
    first = post[0]
    nx0, ny0 = first['x'], first['y']
    x0, y0 = nx0, ny0

    win = [p for p in norm if first['t'] < p['t'] <= first['t'] + 1600.0]
    last_win = win[-1] if win else first
    dx_win = last_win['x'] - nx0
    dy_win = last_win['y'] - ny0
    adjusted = False

    if al == 'red' and nx0 >= 5.05 and ny0 >= 5.15 and dx_win >= 0.8:
        y0 = max(6.62, min(width - 0.4, ny0 + 0.42 + max(0.0, -dy_win * 0.5)))
        x0 = 4.00 if nx0 > 5.85 else max(4.02, min(4.68, nx0 - max(0.50, min(1.10, dx_win * 0.45))))
        adjusted = True
    elif al == 'blue' and nx0 >= 5.65 and ny0 <= 2.00 and dx_win >= 1.2 and dy_win >= 0.15:
        y0 = max(0.4, min(1.38, ny0 - 0.45))
        x0 = 4.00 if nx0 > 5.85 else max(4.02, min(4.78, nx0 - max(0.55, min(1.10, dx_win * 0.45))))
        adjusted = True
    elif nx0 <= 4.50 and 1.58 < ny0 <= 2.75 and dx_win >= 1.8:
        trench_ys = [p['y'] for p in [first] + win if p['x'] <= 7.2]
        min_ny = min(trench_ys) if trench_ys else ny0
        if min_ny <= 1.55:
            y0 = min_ny
            adjusted = True
    elif al == 'blue' and nx0 <= 2.80 and 6.48 <= ny0 <= 6.80 and dx_win <= -0.5:
        y0 = ny0 - 0.38
        adjusted = True
    elif al == 'red' and nx0 <= 3.35 and 3.38 <= ny0 <= 4.10 and dx_win <= -1.0 and dy_win >= 0.5:
        y0 = ny0 - 0.68
        adjusted = True

    if not adjusted:
        vmax = float(FIELD['robot']['maxPlausibleSpeedMps'])
        if first['t'] > 100.0 and nx0 > float(FIELD['startLineX']):
            win1 = [p for p in norm if first['t'] < p['t'] <= first['t'] + 1000.0]
            anchor = next((p for p in win1 if p['t'] - first['t'] >= 150.0), win1[-1] if win1 else None)
            if anchor and anchor['t'] > first['t']:
                dt = (anchor['t'] - first['t']) / 1000.0
                vx = max(-vmax, min(vmax, (anchor['x'] - nx0) / dt))
                vy = max(-vmax, min(vmax, (anchor['y'] - ny0) / dt))
                dt_s = first['t'] / 1000.0
                x0 = max(0.4, nx0 - min(1.8, max(0.0, vx) * dt_s))
                if abs(vy * dt_s) <= 0.35:
                    y0 = max(0.4, min(width - 0.4, ny0 - vy * dt_s))
        elif nx0 > 5.85:
            x0 = 4.00
        elif nx0 > 5.05 and dx_win >= 0.5:
            x0 = max(4.02, min(4.68, nx0 - max(0.50, min(1.10, dx_win * 0.45))))

    fx0 = length - x0 if al == 'red' else x0
    fy0 = width - y0 if al == 'red' else y0
    return {
        'x': round(x0, 3),
        'y': round(y0, 3),
        'field_x': round(fx0, 3),
        'field_y': round(fy0, 3),
        'rawX': round(nx0, 3),
        'rawY': round(ny0, 3),
        'position': start_position_at(y0),
        'canonical_zone': canonical_start_zone_at(y0),
        'atMs': first['t'],
        'inAllianceZone': in_start_zone(x0, max_zone_x),
        'backExtrapolated': bool(abs(x0 - nx0) > 1e-3 or abs(y0 - ny0) > 1e-3),
    }


def anchor_auto_start_trajectory(points, alliance='blue', *, max_start_ms=3000.0):
    """Return a copy of trajectory `points` with the de-biased t = 0 start pose anchored."""
    if not points:
        return []
    pose = infer_start_pose(points, alliance, max_start_ms=max_start_ms)
    if not pose or not pose.get('inAllianceZone'):
        return [dict(p) for p in points]
    ordered = [dict(p) for p in sorted(points, key=lambda item: float(item.get('t', 0)))]
    first = ordered[0]
    fx0, fy0 = pose['field_x'], pose['field_y']
    if float(first['t']) > 50.0 and float(first['t']) <= float(max_start_ms):
        anchor_pt = {
            **first,
            't': 0,
            'x': fx0,
            'y': fy0,
            'raw_x': first.get('raw_x', first['x']),
            'raw_y': first.get('raw_y', first['y']),
            'start_anchored': True,
        }
        return [anchor_pt] + ordered
    if pose.get('backExtrapolated'):
        ordered[0] = {
            **first,
            'x': fx0,
            'y': fy0,
            'raw_x': first.get('raw_x', first['x']),
            'raw_y': first.get('raw_y', first['y']),
        }
    return ordered
