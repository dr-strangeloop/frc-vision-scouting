#!/usr/bin/env python3
"""Per-robot scouting features from vision-counted balls (hub entries, launches, robot tracks, or fuel_rows).

Computes from ball-by-ball vision evidence (never anchored to TBA or scoreboard):
  * shot locations on the field: per-shot (x, y) in WPILib field meters and alliance-normalized (norm_x, norm_y),
    clustered shooting spots (0.8 m radius), distance bands to own hub, and a 2D field heatmap grid.
  * shooting bursts and cycles: balls per burst/cycle, duration, active firing rate (balls/s), peak balls/s,
    and time between cycles (reload/reposition pauses and start-to-start cycle periods).
  * auto / teleop / endgame breakdown: per-period launches, hub entries, linked scored, bursts, cycles, and rates,
    plus hub-active tracking across the 2026 REBUILT shifts.
  * accuracy where both launch and entry are seen: 1-to-1 launch-to-entry linking (same-track continuity first,
    then distance-aware flight-time matching within [0.3 s, 3.0 s]), separating solo-shooter windows from
    shared multi-robot volleys so accuracy is always in [0, 1].
"""
import argparse
from bisect import bisect_left, bisect_right
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fuel_count as FC  # noqa: E402
from camera3d import camera_from_pixel_homography  # noqa: E402
from common import AUTO_END_MS, AUTO_GRACE_MS, FIELD  # noqa: E402

FIELD_LENGTH = float(FIELD['length'])
FIELD_WIDTH = float(FIELD['width'])
HALF_LENGTH = FIELD_LENGTH / 2.0
HALF_WIDTH = FIELD_WIDTH / 2.0
OWN_HUB_X = float(FIELD['hub']['x'])
OWN_HUB_Y = float(FIELD['hub']['y'])
BLUE_HUB = (OWN_HUB_X, OWN_HUB_Y)
RED_HUB = (FIELD_LENGTH - OWN_HUB_X, FIELD_WIDTH - OWN_HUB_Y)
HUBS = {'blue_hub': BLUE_HUB, 'red_hub': RED_HUB}
ALLIANCE_HUB = {'blue': BLUE_HUB, 'red': RED_HUB}

TELEOP_START_MS = AUTO_END_MS + AUTO_GRACE_MS          # 23_000 ms
TRANSITION_END_MS = TELEOP_START_MS + 10_000           # 33_000 ms
SHIFT_MS = 25_000                                      # four 25 s shifts: 33_000..133_000 ms
ENDGAME_START_MS = TRANSITION_END_MS + 4 * SHIFT_MS    # 133_000 ms
MATCH_MS = ENDGAME_START_MS + 30_000                   # 163_000 ms
AUTO_WINNER_MIN_MARGIN = 3

BURST_GAP_MS = 750
CYCLE_GAP_MS = 2500
SPOT_RADIUS_M = 0.8
CELL_SIZE_M = 0.8
ENTRY_WINDOW_S = FC.ENTRY_WINDOW_S                     # (0.3, 3.0) s
TRAJECTORY_INTERP_GAP_MS = 1500
TRAJECTORY_SNAP_GAP_MS = 800
SOLO_WINDOW_MS = 2500
MIN_RATE_DURATION_MS = 100
MAX_PLAUSIBLE_BPS = 30.0
MIN_SHOTS_FOR_CLAIM = 5
SNAPSHOT_MAX_BYTES = 60_000

DISTANCE_BANDS = (
    ('under 2 m', 0.0, 2.0),
    ('2 to 3 m', 2.0, 3.0),
    ('3 to 4 m', 3.0, 4.0),
    ('4 m and over', 4.0, float('inf')),
)


def _round(val, digits=2):
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return round(f, digits)


def _mean(xs):
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return sum(vals) / len(vals) if vals else None


def _median(xs):
    vals = sorted(float(x) for x in xs if x is not None and math.isfinite(float(x)))
    if not vals:
        return None
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 == 1 else 0.5 * (vals[mid - 1] + vals[mid])


def _sd(xs):
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if len(vals) < 2:
        return None
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def normalize_to_alliance(x, y, alliance):
    """Rotate a field-frame point (x, y) in meters into the alliance-normalized frame (own driver station at x = 0)."""
    if x is None or y is None:
        return None, None
    fx, fy = float(x), float(y)
    if not (math.isfinite(fx) and math.isfinite(fy)):
        return None, None
    if alliance == 'red':
        return FIELD_LENGTH - fx, FIELD_WIDTH - fy
    return fx, fy


def denormalize_from_alliance(nx, ny, alliance):
    """Inverse of normalize_to_alliance (180-degree rotation about field centre)."""
    return normalize_to_alliance(nx, ny, alliance)


def distance_to_own_hub(norm_x, norm_y):
    """Distance in meters from an alliance-normalized point (norm_x, norm_y) to the own hub."""
    if norm_x is None or norm_y is None:
        return None
    return math.hypot(float(norm_x) - OWN_HUB_X, float(norm_y) - OWN_HUB_Y)


def distance_to_alliance_hub(x, y, alliance):
    """Distance in meters from a field-frame point (x, y) to `alliance`'s hub."""
    if x is None or y is None:
        return None
    nx, ny = normalize_to_alliance(x, y, alliance)
    return distance_to_own_hub(nx, ny)


def zone_at(norm_x, norm_y):
    """Named field zone of an alliance-normalized point (matches src/lib/vision/fieldRebuilt.js zoneAt)."""
    if norm_x is None or norm_y is None:
        return None
    nx, ny = float(norm_x), float(norm_y)
    if not (math.isfinite(nx) and math.isfinite(ny)):
        return None
    if nx <= HALF_LENGTH:
        px, py, side = nx, ny, 'own'
    else:
        px, py, side = FIELD_LENGTH - nx, FIELD_WIDTH - ny, 'opp'

    def in_box(x0, x1, y0, y1):
        return x0 <= px <= x1 and y0 <= py <= y1

    tw = FIELD['tower']
    if in_box(tw['x0'], tw['x1'], tw['y0'], tw['y1']):
        return f'{side}_tower'
    op = FIELD['outpost']
    if in_box(op['x0'], op['x1'], op['y0'], op['y1']) or in_box(op['x0'], op['x1'], FIELD_WIDTH - op['y1'], FIELD_WIDTH - op['y0']):
        return f'{side}_outpost'
    hub_offset = abs(py - OWN_HUB_Y)
    in_hub_band = float(FIELD['startLineX']) <= px <= float(FIELD['trench']['x1'])
    if in_hub_band and hub_offset <= float(FIELD['hub']['halfSize']):
        return f'{side}_hub'
    if in_hub_band and hub_offset <= float(FIELD['bump']['yOuter']):
        return f'{side}_bump'
    if in_hub_band:
        return f'{side}_trench'
    if px < float(FIELD['startLineX']):
        return f'{side}_alliance_zone'
    return 'neutral'


def derive_auto_winner(red_auto, blue_auto, min_margin=AUTO_WINNER_MIN_MARGIN):
    """Return (auto_winner, uncertain) from vision auto hub-entry counts.

    When |red_auto - blue_auto| <= min_margin (or both 0), returns (None, True) so shift hub activity
    stays None (unknown) rather than guessing which hub was active.
    """
    if red_auto is None or blue_auto is None:
        return None, True
    r, b = int(red_auto), int(blue_auto)
    if r == 0 and b == 0:
        return None, True
    if abs(r - b) <= int(min_margin):
        return None, True
    return ('red' if r > b else 'blue'), False


def hub_active(alliance, t_ms, auto_winner):
    """Whether `alliance`'s hub is active at match time `t_ms` (matches fuelAnalytics.js hubActive)."""
    if t_ms is None:
        return None
    t = float(t_ms)
    if t < TRANSITION_END_MS or t >= ENDGAME_START_MS:
        return True
    if not auto_winner or alliance not in ('red', 'blue'):
        return None
    shift = int((t - TRANSITION_END_MS) // SHIFT_MS)
    winner_active = (shift % 2 == 1)
    return winner_active if alliance == auto_winner else (not winner_active)


def shot_phase(t_ms, scoring=False):
    """Two-way match phase ('auto' | 'teleop') matching fuel_count.py."""
    limit = AUTO_END_MS + (AUTO_GRACE_MS if scoring else 0)
    return 'auto' if float(t_ms) <= limit else 'teleop'


def shot_period(t_ms, scoring=False):
    """Three-way match period ('auto' | 'teleop' | 'endgame')."""
    t = float(t_ms)
    limit = AUTO_END_MS + (AUTO_GRACE_MS if scoring else 0)
    if t <= limit:
        return 'auto'
    if t >= ENDGAME_START_MS:
        return 'endgame'
    return 'teleop'


def _project_norm_uv(u_norm, v_norm, h_norm):
    """Project normalized image point (u_norm, v_norm) in [0, 1]^2 to field meters (x, y) via homography_norm."""
    if h_norm is None:
        return None, None
    h = np.asarray(h_norm, dtype=float)
    if h.shape != (3, 3):
        return None, None
    vec = h @ np.array([float(u_norm), float(v_norm), 1.0], dtype=float)
    if abs(vec[2]) < 1e-9:
        return None, None
    x, y = float(vec[0] / vec[2]), float(vec[1] / vec[2])
    if not (math.isfinite(x) and math.isfinite(y)):
        return None, None
    return float(np.clip(x, -1.0, FIELD_LENGTH + 1.0)), float(np.clip(y, -1.0, FIELD_WIDTH + 1.0))


def _point_xy(pt, h_norm=None):
    """Extract field (x, y) in meters from a trajectory point dict, projecting `box` if x/y are absent."""
    if not isinstance(pt, dict):
        return None, None
    x, y = pt.get('x'), pt.get('y')
    if x is not None and y is not None:
        try:
            fx, fy = float(x), float(y)
            if math.isfinite(fx) and math.isfinite(fy):
                return fx, fy
        except (TypeError, ValueError):
            pass
    box = pt.get('box')
    if box and len(box) == 4 and h_norm is not None:
        u_norm = 0.5 * (float(box[0]) + float(box[2]))
        v_norm = float(box[3])
        return _project_norm_uv(u_norm, v_norm, h_norm)
    return None, None


def _build_xy_index(candidates, h_norm=None):
    pts = []
    for r in candidates:
        for p in r.get('trajectory') or []:
            if not isinstance(p, dict) or p.get('t') is None:
                continue
            x, y = _point_xy(p, h_norm=h_norm)
            if x is not None and y is not None:
                pts.append((float(p['t']), x, y))
    if not pts:
        return None
    pts.sort(key=lambda item: item[0])
    return [p[0] for p in pts], pts


def _eval_xy_index(xy_index, t_ms, interp_gap_ms=TRAJECTORY_INTERP_GAP_MS, snap_gap_ms=TRAJECTORY_SNAP_GAP_MS):
    if xy_index is None:
        return None, None
    times, pts = xy_index
    t = float(t_ms)
    idx = bisect_left(times, t)
    if 0 < idx < len(pts):
        t0, x0, y0 = pts[idx - 1]
        t1, x1, y1 = pts[idx]
        dt = t1 - t0
        if dt <= interp_gap_ms:
            if dt <= 1e-6:
                return x0, y0
            alpha = (t - t0) / dt
            return x0 + alpha * (x1 - x0), y0 + alpha * (y1 - y0)
    best = None
    for cand_idx in (idx - 1, idx):
        if 0 <= cand_idx < len(pts):
            ct, cx, cy = pts[cand_idx]
            gap = abs(ct - t)
            if gap <= snap_gap_ms and (best is None or gap < best[0]):
                best = (gap, cx, cy)
    return (best[1], best[2]) if best else (None, None)


def _build_box_index(robot):
    pts = []
    for p in robot.get('trajectory') or []:
        box = p.get('box')
        if box is not None and len(box) == 4 and p.get('t') is not None:
            pts.append((float(p['t']), box))
    if not pts:
        return None
    pts.sort(key=lambda item: item[0])
    return [p[0] for p in pts], pts


def _eval_box_index(box_index, t_ms, max_gap_ms=FC.BOX_MAX_GAP_MS):
    if box_index is None:
        return None
    times, pts = box_index
    t = float(t_ms)
    idx = bisect_left(times, t)
    best = None
    for cand_idx in (idx - 1, idx):
        if 0 <= cand_idx < len(pts):
            ct, cbox = pts[cand_idx]
            gap = abs(ct - t)
            if gap <= max_gap_ms and (best is None or gap < best[0]):
                best = (gap, cbox)
    return best[1] if best else None


def robot_position_at(robot_or_tracks, t_ms, team_key=None, calibration=None,
                      interp_gap_ms=TRAJECTORY_INTERP_GAP_MS, snap_gap_ms=TRAJECTORY_SNAP_GAP_MS):
    """Interpolate a robot's field position (x, y) in meters at `t_ms` from its trajectory.

    `robot_or_tracks` can be a single robot track dict or a list of robot track dicts.
    """
    h_norm = (calibration or {}).get('homography_norm') if isinstance(calibration, dict) else None
    if isinstance(robot_or_tracks, dict):
        candidates = [robot_or_tracks]
    else:
        candidates = [
            r for r in (robot_or_tracks or [])
            if isinstance(r, dict) and (team_key is None or r.get('team_key') == team_key)
        ]
    xy_idx = _build_xy_index(candidates, h_norm=h_norm)
    return _eval_xy_index(xy_idx, t_ms, interp_gap_ms=interp_gap_ms, snap_gap_ms=snap_gap_ms)


def hub_entries_with_tracks(tracks, polygons, offset_s, min_obs=FC.MIN_TRACK_OBS, drop_px=FC.ENTRY_DROP_PX):
    """Like fuel_count.hub_entries, also recording `track_index` so same-track launches and entries link directly."""
    poly_bounds = []
    for name, poly in (polygons or {}).items():
        pts = np.asarray(poly, dtype=float).reshape(-1, 2)
        poly_bounds.append((
            name,
            poly,
            float(pts[:, 0].min()) - 1.0,
            float(pts[:, 1].min()) - 1.0,
            float(pts[:, 0].max()) + 1.0,
            float(pts[:, 1].max()) + 1.0,
        ))
    entries = []
    for track_idx, track in enumerate(tracks or []):
        obs = FC._obs(track)
        if len(obs) < min_obs or obs[-1][6] - obs[0][6] < drop_px:
            continue
        for name, poly, u0, v0, u1, v1 in poly_bounds:
            hit_bbox = False
            for o in obs:
                if u0 <= o[5] <= u1 and v0 <= o[6] <= v1:
                    hit_bbox = True
                    break
            if not hit_bbox:
                continue
            inside = [
                (u0 <= o[5] <= u1 and v0 <= o[6] <= v1)
                and cv2.pointPolygonTest(poly, (float(o[5]), float(o[6])), False) >= 0
                for o in obs
            ]
            if not any(inside) or all(inside):
                continue
            first = inside.index(True)
            entries.append({
                't_ms': int(round((obs[first][1] - offset_s) * 1000)),
                'hub': name,
                'alliance': FC.HUBS[name],
                'track_index': track_idx,
                'track_id': track_idx,
            })
            break
    entries.sort(key=lambda e: e['t_ms'])
    return entries


def _infer_robot_alliance(robot, calibration=None):
    """Return robot['alliance'] if set, else infer 'blue' | 'red' from early auto trajectory x (< HALF_LENGTH is blue)."""
    if not isinstance(robot, dict):
        return None
    if robot.get('alliance') in ('red', 'blue'):
        return robot['alliance']
    h_norm = (calibration or {}).get('homography_norm') if isinstance(calibration, dict) else None
    early_xs = []
    all_xs = []
    for p in robot.get('trajectory') or []:
        x, _ = _point_xy(p, h_norm=h_norm)
        if x is not None:
            all_xs.append(x)
            if float(p.get('t', 99999)) <= AUTO_END_MS:
                early_xs.append(x)
    sample = early_xs if early_xs else all_xs
    if not sample:
        return None
    med_x = _median(sample)
    return 'blue' if med_x < HALF_LENGTH else 'red'


def launches_with_positions(tracks, robot_tracks, offset_s=0.0, width=1280, height=720, calibration=None):
    """Detect every ball track climbing out of a robot box (same criteria as fuel_count.launches),
    enriched with the robot's field coordinates (x, y), alliance-normalized (norm_x, norm_y),
    distance to own hub, field zone, and `track_index`.
    """
    h_norm = (calibration or {}).get('homography_norm') if isinstance(calibration, dict) else None
    valid_robots = [r for r in (robot_tracks or []) if isinstance(r, dict)]
    robot_alliances = {id(r): _infer_robot_alliance(r, calibration=calibration) for r in valid_robots}
    robot_box_idx = {id(r): _build_box_index(r) for r in valid_robots}
    robot_xy_idx = {id(r): _build_xy_index([r], h_norm=h_norm) for r in valid_robots}
    team_xy_idx = {}
    for r in valid_robots:
        tk = r.get('team_key')
        if tk and tk not in team_xy_idx:
            team_xy_idx[tk] = _build_xy_index([x for x in valid_robots if x.get('team_key') == tk], h_norm=h_norm)

    out = []
    for track_idx, track in enumerate(tracks or []):
        obs = FC._obs(track)
        if len(obs) < FC.MIN_TRACK_OBS:
            continue
        head = obs[:FC.LAUNCH_HEAD_OBS]
        if head[0][3] - head[-1][3] < FC.LAUNCH_RISE_PX:
            continue
        t_ms = int(round((obs[0][1] - offset_s) * 1000))
        u, v = obs[0][2] / width, obs[0][3] / height
        top = min(o[3] for o in obs) / height
        best = None
        for robot in valid_robots:
            box = _eval_box_index(robot_box_idx[id(robot)], t_ms)
            if box is None:
                continue
            w, h = box[2] - box[0], box[3] - box[1]
            if not (box[0] - FC.BOX_EXPAND_X * w <= u <= box[2] + FC.BOX_EXPAND_X * w and box[1] - FC.BOX_ABOVE * h <= v <= box[3]):
                continue
            if box[1] - top < FC.LAUNCH_CLIMB * h:
                continue
            d = math.hypot(u - (box[0] + box[2]) / 2, v - (box[1] + box[3]) / 2)
            if best is None or d < best[0]:
                best = (d, robot, box)
        if best:
            _, robot, box = best
            team_key = robot.get('team_key')
            alliance = robot.get('alliance') or robot_alliances.get(id(robot))
            x, y = _eval_xy_index(robot_xy_idx.get(id(robot)), t_ms)
            if (x is None or y is None) and team_key:
                x, y = _eval_xy_index(team_xy_idx.get(team_key), t_ms)
            if (x is None or y is None) and box is not None and h_norm is not None:
                x, y = _project_norm_uv(0.5 * (box[0] + box[2]), box[3], h_norm)
            nx, ny = normalize_to_alliance(x, y, alliance)
            dist_m = distance_to_own_hub(nx, ny)
            out.append({
                't_ms': t_ms,
                'team_key': team_key,
                'alliance': alliance,
                'track_index': track_idx,
                'track_id': track_idx,
                'x': _round(x, 3),
                'y': _round(y, 3),
                'norm_x': _round(nx, 3),
                'norm_y': _round(ny, 3),
                'distance_m': _round(dist_m, 2),
                'zone': zone_at(nx, ny),
                'u': _round(float(obs[0][2]), 1),
                'v': _round(float(obs[0][3]), 1),
                'ru': _round(float(obs[0][5]), 1),
                'rv': _round(float(obs[0][6]), 1),
            })
    out.sort(key=lambda e: e['t_ms'])
    return out


def _enrich_shots_positions(shots, robot_tracks, calibration=None):
    """Ensure every shot dict has field (x, y), alliance-normalized (norm_x, norm_y), distance_m, and zone."""
    h_norm = (calibration or {}).get('homography_norm') if isinstance(calibration, dict) else None
    valid_robots = [r for r in (robot_tracks or []) if isinstance(r, dict)]
    team_alliance = {}
    team_xy_idx = {}
    for r in valid_robots:
        tk = r.get('team_key')
        if tk:
            if tk not in team_alliance:
                inferred = _infer_robot_alliance(r, calibration=calibration)
                if inferred:
                    team_alliance[tk] = inferred
            if tk not in team_xy_idx:
                team_xy_idx[tk] = _build_xy_index([x for x in valid_robots if x.get('team_key') == tk], h_norm=h_norm)
    enriched = []
    for raw in shots or []:
        s = dict(raw)
        t_ms = int(round(float(s.get('t_ms', s.get('tMs', s.get('started_ms', 0))))))
        s['t_ms'] = t_ms
        tk = s.get('team_key')
        alliance = s.get('alliance') or team_alliance.get(tk)
        if not alliance and s.get('target') in FC.HUBS:
            alliance = FC.HUBS[s['target']]
        s['alliance'] = alliance
        x, y = s.get('x'), s.get('y')
        if (x is None or y is None) and isinstance(s.get('launch'), (list, tuple)) and len(s['launch']) >= 2:
            x, y = s['launch'][0], s['launch'][1]
        if (x is None or y is None) and tk and tk in team_xy_idx:
            x, y = _eval_xy_index(team_xy_idx[tk], t_ms)
        nx, ny = normalize_to_alliance(x, y, alliance)
        dist_m = distance_to_own_hub(nx, ny)
        if dist_m is None:
            dist_m = s.get('distance_m') if s.get('distance_m') is not None else s.get('distance_to_hub_m')
            if dist_m is None:
                dist_m = s.get('distance')
        s['x'] = _round(x, 3)
        s['y'] = _round(y, 3)
        s['norm_x'] = _round(nx, 3)
        s['norm_y'] = _round(ny, 3)
        s['distance_m'] = _round(dist_m, 2)
        s['distance'] = s['distance_m']
        s['zone'] = s.get('zone') or zone_at(nx, ny)
        if s['x'] is not None and s['y'] is not None and not isinstance(s.get('launch'), list):
            s['launch'] = [s['x'], s['y'], 0.6]
        enriched.append(s)
    enriched.sort(key=lambda item: item['t_ms'])
    return enriched


def _expected_flight_ms(distance_m):
    """Expected launch-to-hub-entry flight time in ms as a function of distance to hub."""
    if distance_m is None or not math.isfinite(float(distance_m)):
        return 1100.0
    val = 550.0 + 135.0 * float(distance_m)
    return 650.0 if val < 650.0 else (1800.0 if val > 1800.0 else val)


def link_launches_to_entries(shots, entries, window_s=ENTRY_WINDOW_S):
    """Link launches (`shots`) and hub `entries` 1-to-1 where both launch and entry are seen.

    Pass 1 links any launch and entry that share the same continuous `track_index`.
    Pass 2 solves a distance-aware minimum-cost 1-to-1 assignment within `window_s = (0.3, 3.0)` s
    per alliance, tagging each link as:
      * 'same_track'    (confidence 0.98): one continuous ball track from robot box into hub opening
      * 'solo_window'   (confidence 0.90): only one robot of that alliance launched in the entry window
      * 'shared_window' (confidence 0.65): multiple robots of that alliance launched in the entry window

    Returns `(linked_shots, linked_entries)` as new lists of dicts without mutating inputs.
    """
    lo_ms = int(round(window_s[0] * 1000))
    hi_ms = int(round(window_s[1] * 1000))
    out_shots = [dict(s) for s in (shots or [])]
    out_entries = [dict(e) for e in (entries or [])]

    used_shots = set()
    used_entries = set()

    def compatible(entry, shot):
        # Counter assignments (including unknown) are canonical, not another chance to rename a ball.
        return 'attribution' not in entry or (entry.get('team_key') is not None
                                             and entry.get('team_key') == shot.get('team_key'))

    # Preserve pre-existing explicit links when present (e.g. from fuel_rows or ballistic shots)
    for si, s in enumerate(out_shots):
        pre_entry_ms = s.get('entry_t_ms') if s.get('entry_t_ms') is not None else s.get('score_ms')
        if s.get('result') == 'scored' or s.get('scored') is True or pre_entry_ms is not None:
            s['scored'] = True
            s['result'] = 'scored'
            s['entry_t_ms'] = int(round(float(pre_entry_ms))) if pre_entry_ms is not None else s['t_ms']
            s['score_ms'] = s['entry_t_ms']
            s['flight_ms'] = max(0, s['entry_t_ms'] - s['t_ms']) if pre_entry_ms is not None else None
            s['link_kind'] = s.get('link_kind') or 'prelinked'
            s['unambiguous'] = bool(s.get('unambiguous', True))
            s['confidence'] = float(s.get('confidence') or 0.85)
            used_shots.add(si)

    # Pass 1: exact same ball track
    entry_by_track = {}
    for ej, e in enumerate(out_entries):
        t_idx = e.get('track_index') if e.get('track_index') is not None else e.get('track_id')
        if t_idx is not None:
            entry_by_track.setdefault(t_idx, []).append(ej)

    for si, s in enumerate(out_shots):
        if si in used_shots:
            continue
        t_idx = s.get('track_index') if s.get('track_index') is not None else s.get('track_id')
        if t_idx is None or t_idx not in entry_by_track:
            continue
        for ej in entry_by_track[t_idx]:
            if ej in used_entries:
                continue
            e = out_entries[ej]
            if s.get('alliance') == e.get('alliance') and e['t_ms'] >= s['t_ms'] and compatible(e, s):
                used_shots.add(si)
                used_entries.add(ej)
                flight_ms = int(e['t_ms'] - s['t_ms'])
                s.update({
                    'scored': True,
                    'result': 'scored',
                    'entry_t_ms': int(e['t_ms']),
                    'score_ms': int(e['t_ms']),
                    'flight_ms': flight_ms,
                    'link_kind': 'same_track',
                    'unambiguous': True,
                    'confidence': 0.98,
                })
                e.update({
                    'team_key': s.get('team_key'),
                    'launch_t_ms': int(s['t_ms']),
                    'flight_ms': flight_ms,
                    'link_kind': 'same_track',
                    'confidence': 0.98,
                })
                break

    # Pre-index alliance shot times for O(log N) window team checks
    alliance_shot_records = {'red': ([], []), 'blue': ([], [])}
    for s in sorted(out_shots, key=lambda x: int(x['t_ms'])):
        c = s.get('alliance')
        if c in alliance_shot_records:
            alliance_shot_records[c][0].append(int(s['t_ms']))
            alliance_shot_records[c][1].append(s.get('team_key'))

    # Pass 2: 1-to-1 flight-time assignment within [lo_ms, hi_ms] per alliance
    for color in ('red', 'blue'):
        cand_s = [i for i, s in enumerate(out_shots) if i not in used_shots and s.get('alliance') == color]
        cand_e = [j for j, e in enumerate(out_entries) if j not in used_entries and e.get('alliance') == color]
        if not cand_s or not cand_e:
            continue
        inf = 1e9
        s_times = np.array([int(out_shots[si]['t_ms']) for si in cand_s], dtype=float)[:, None]
        exp_dts = np.array([_expected_flight_ms(out_shots[si].get('distance_m')) for si in cand_s], dtype=float)[:, None]
        e_times = np.array([int(out_entries[ej]['t_ms']) for ej in cand_e], dtype=float)[None, :]
        dt_mat = e_times - s_times
        valid_mask = (dt_mat >= lo_ms) & (dt_mat <= hi_ms)
        valid_mask &= np.array([[compatible(out_entries[ej], out_shots[si]) for ej in cand_e]
                                for si in cand_s], dtype=bool)
        cost = np.where(valid_mask, np.abs(dt_mat - exp_dts), inf)
        row_ind, col_ind = linear_sum_assignment(cost)
        all_times, all_teams = alliance_shot_records[color]
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] >= inf:
                continue
            si, ej = cand_s[r], cand_e[c]
            used_shots.add(si)
            used_entries.add(ej)
            s, e = out_shots[si], out_entries[ej]
            et = int(e['t_ms'])
            st = int(s['t_ms'])
            flight_ms = et - st
            i0 = bisect_left(all_times, et - hi_ms)
            i1 = bisect_right(all_times, et - lo_ms)
            window_teams = set(all_teams[i0:i1])
            solo = len(window_teams) <= 1
            kind = 'solo_window' if solo else 'shared_window'
            conf = 0.90 if solo else 0.65
            s.update({
                'scored': True,
                'result': 'scored',
                'entry_t_ms': et,
                'score_ms': et,
                'flight_ms': flight_ms,
                'link_kind': kind,
                'unambiguous': solo,
                'confidence': conf,
            })
            e.update({
                'team_key': s.get('team_key'),
                'launch_t_ms': st,
                'flight_ms': flight_ms,
                'link_kind': kind,
                'confidence': conf,
            })

    # Mark remaining unlinked shots and check whether each was launched in a solo window
    for color in ('red', 'blue'):
        all_times, all_teams = alliance_shot_records[color]
        for si, s in enumerate(out_shots):
            if s.get('alliance') != color or si in used_shots:
                continue
            st = int(s['t_ms'])
            my_tk = s.get('team_key')
            i0 = bisect_left(all_times, st - SOLO_WINDOW_MS)
            i1 = bisect_right(all_times, st + SOLO_WINDOW_MS)
            teammate_near = any(tk != my_tk for tk in all_teams[i0:i1])
            solo = not teammate_near
            raw_res = s.get('result')
            res = raw_res if raw_res in ('miss', 'missed', 'rim', 'short') else ('miss' if solo else 'unknown')
            s.update({
                'scored': False,
                'result': res,
                'entry_t_ms': None,
                'score_ms': None,
                'flight_ms': None,
                'link_kind': None,
                'unambiguous': solo,
                'confidence': float(s.get('confidence') or (0.85 if solo else 0.55)),
            })

    return out_shots, out_entries


def shooting_spots(shots, radius_m=SPOT_RADIUS_M):
    """Greedy deterministic clustering of a robot's shots in the alliance-normalized frame (matches shotProfile.js),
    followed by a centroid-merge pass so spots whose running means drifted within `radius_m` combine cleanly.
    """
    spots = []
    ordered = sorted((shots or []), key=lambda s: int(s.get('t_ms', 0)))
    for s in ordered:
        nx, ny = s.get('norm_x'), s.get('norm_y')
        if nx is None or ny is None:
            nx, ny = normalize_to_alliance(s.get('x'), s.get('y'), s.get('alliance'))
        if nx is None or ny is None:
            continue
        best = None
        for spot in spots:
            d = math.hypot(spot['norm_x'] - nx, spot['norm_y'] - ny)
            if d <= radius_m and (best is None or d < best[0]):
                best = (d, spot)
        if best is None:
            spots.append({'norm_x': float(nx), 'norm_y': float(ny), 'members': [s]})
        else:
            spot = best[1]
            spot['members'].append(s)
            n = len(spot['members'])
            spot['norm_x'] += (float(nx) - spot['norm_x']) / n
            spot['norm_y'] += (float(ny) - spot['norm_y']) / n

    # Merge spots whose final centroids drifted within radius_m of each other
    changed = True
    while changed and len(spots) > 1:
        changed = False
        spots.sort(key=lambda sp: -len(sp['members']))
        for i in range(len(spots)):
            for j in range(i + 1, len(spots)):
                if math.hypot(spots[i]['norm_x'] - spots[j]['norm_x'], spots[i]['norm_y'] - spots[j]['norm_y']) <= radius_m:
                    merged_members = spots[i]['members'] + spots[j]['members']
                    spots[i]['members'] = merged_members
                    n_m = len(merged_members)
                    spots[i]['norm_x'] = sum(
                        float(m['norm_x'] if m.get('norm_x') is not None else normalize_to_alliance(m.get('x'), m.get('y'), m.get('alliance'))[0])
                        for m in merged_members
                    ) / n_m
                    spots[i]['norm_y'] = sum(
                        float(m['norm_y'] if m.get('norm_y') is not None else normalize_to_alliance(m.get('x'), m.get('y'), m.get('alliance'))[1])
                        for m in merged_members
                    ) / n_m
                    spots.pop(j)
                    changed = True
                    break
            if changed:
                break

    total = sum(len(sp['members']) for sp in spots)
    out = []
    for sp in spots:
        members = sp['members']
        alliance = members[0].get('alliance')
        team_key = members[0].get('team_key')
        nx, ny = sp['norm_x'], sp['norm_y']
        fx, fy = denormalize_from_alliance(nx, ny, alliance)
        dist = distance_to_own_hub(nx, ny)
        n_shots = len(members)
        n_scored = sum(1 for m in members if m.get('scored') or m.get('result') == 'scored')
        out.append({
            'x': _round(fx, 2),
            'y': _round(fy, 2),
            'field_x': _round(fx, 2),
            'field_y': _round(fy, 2),
            'norm_x': _round(nx, 2),
            'norm_y': _round(ny, 2),
            'distance_m': _round(dist, 2),
            'distance': _round(dist, 1),
            'zone': zone_at(nx, ny),
            'shots': n_shots,
            'scored': n_scored,
            'accuracy': _round(n_scored / n_shots, 2) if n_shots else None,
            'share': _round(n_shots / total, 2) if total else 0.0,
            'team_key': team_key,
            'alliance': alliance,
        })
    out.sort(key=lambda item: (-item['shots'], item['distance_m'] or 0.0))
    return out


def accuracy_by_distance(shots):
    """Accuracy grouped into standard distance bands from the robot's own hub (matches shotProfile.js)."""
    out = []
    for label, lo, hi in DISTANCE_BANDS:
        in_band = [
            s for s in (shots or [])
            if s.get('distance_m') is not None and lo <= float(s['distance_m']) < hi
        ]
        if not in_band:
            continue
        n_shots = len(in_band)
        n_scored = sum(1 for s in in_band if s.get('scored') or s.get('result') == 'scored')
        out.append({
            'label': label,
            'from': lo,
            'to': hi if math.isfinite(hi) else None,
            'shots': n_shots,
            'scored': n_scored,
            'accuracy': _round(n_scored / n_shots, 2) if n_shots else None,
        })
    return out


def build_heatmap(shots, cell_size_m=CELL_SIZE_M):
    """2D field heatmap grid over [0, FIELD_LENGTH] x [0, FIELD_WIDTH] in meters.

    Returns sparse non-empty cells sorted by shot count descending, ready for SVG overlay on FieldBackdrop.
    """
    cell_size = max(0.2, float(cell_size_m))
    nx = max(1, int(math.ceil(FIELD_LENGTH / cell_size)))
    ny = max(1, int(math.ceil(FIELD_WIDTH / cell_size)))
    cell_w = FIELD_LENGTH / nx
    cell_h = FIELD_WIDTH / ny
    bins = {}
    for s in shots or []:
        x, y = s.get('x'), s.get('y')
        if x is None or y is None:
            continue
        gx = min(nx - 1, max(0, int(math.floor(float(x) / cell_w))))
        gy = min(ny - 1, max(0, int(math.floor(float(y) / cell_h))))
        bucket = bins.setdefault((gx, gy), {'shots': 0, 'scored': 0, 'alliances': []})
        bucket['shots'] += 1
        if s.get('scored') or s.get('result') == 'scored':
            bucket['scored'] += 1
        if s.get('alliance'):
            bucket['alliances'].append(s['alliance'])

    max_shots = max((b['shots'] for b in bins.values()), default=0)
    cells = []
    for (gx, gy), b in bins.items():
        x0 = gx * cell_w
        y0 = gy * cell_h
        cx = x0 + 0.5 * cell_w
        cy = y0 + 0.5 * cell_h
        dom_alliance = max(('red', 'blue'), key=lambda c: b['alliances'].count(c)) if b['alliances'] else None
        norm_x, norm_y = normalize_to_alliance(cx, cy, dom_alliance)
        n_shots = b['shots']
        n_scored = b['scored']
        cells.append({
            'gx': int(gx),
            'gy': int(gy),
            'x': _round(cx, 2),
            'y': _round(cy, 2),
            'norm_x': _round(norm_x, 2),
            'norm_y': _round(norm_y, 2),
            'x0': _round(x0, 2),
            'y0': _round(y0, 2),
            'width': _round(cell_w, 2),
            'height': _round(cell_h, 2),
            'shots': int(n_shots),
            'scored': int(n_scored),
            'accuracy': _round(n_scored / n_shots, 2) if n_shots else None,
            'intensity': _round(n_shots / max_shots, 3) if max_shots else 0.0,
        })
    cells.sort(key=lambda c: (-c['shots'], c['gx'], c['gy']))
    return {
        'cell_size_m': _round(cell_size, 2),
        'nx': int(nx),
        'ny': int(ny),
        'frame': 'field',
        'total_located_shots': sum(c['shots'] for c in cells),
        'cells': cells,
    }


def _burst_bps(times_ms):
    """(bps, peak_bps) for a sorted list of shot timestamps in ms."""
    n = len(times_ms)
    if n < 2:
        return None, None
    dur_ms = times_ms[-1] - times_ms[0]
    if dur_ms < MIN_RATE_DURATION_MS:
        return None, None
    bps = min(MAX_PLAUSIBLE_BPS, (n - 1) / (dur_ms / 1000.0))
    peak = bps
    if n >= 3:
        for k in (3, 4, 5):
            if n < k:
                continue
            for i in range(n - k + 1):
                span_ms = times_ms[i + k - 1] - times_ms[i]
                if span_ms >= max(MIN_RATE_DURATION_MS, (k - 1) * 40):
                    cand = min(MAX_PLAUSIBLE_BPS, (k - 1) / (span_ms / 1000.0))
                    if cand > peak:
                        peak = cand
    return _round(bps, 2), _round(peak, 2)


def _group_location_summary(members):
    """Mean field and alliance-normalized location for a group of shots."""
    xs = [s['x'] for s in members if s.get('x') is not None]
    ys = [s['y'] for s in members if s.get('y') is not None]
    alliance = members[0].get('alliance') if members else None
    if not xs or not ys:
        return {'x': None, 'y': None, 'norm_x': None, 'norm_y': None, 'distance_m': None, 'zone': None}
    mx, my = _mean(xs), _mean(ys)
    nx, ny = normalize_to_alliance(mx, my, alliance)
    dist = distance_to_own_hub(nx, ny)
    return {
        'x': _round(mx, 2),
        'y': _round(my, 2),
        'norm_x': _round(nx, 2),
        'norm_y': _round(ny, 2),
        'distance_m': _round(dist, 2),
        'zone': zone_at(nx, ny),
    }


def group_bursts(shots, gap_ms=BURST_GAP_MS):
    """Group a robot's shots into continuous bursts (consecutive shots separated by <= `gap_ms`)."""
    ordered = sorted((shots or []), key=lambda s: int(s.get('t_ms', 0)))
    groups = []
    for s in ordered:
        t = int(s['t_ms'])
        if groups and t - groups[-1][-1]['t_ms'] <= gap_ms:
            groups[-1].append(s)
        else:
            groups.append([s])

    bursts = []
    for idx, members in enumerate(groups):
        times = [int(m['t_ms']) for m in members]
        start_ms, end_ms = times[0], times[-1]
        dur_ms = end_ms - start_ms
        n_balls = len(members)
        n_scored = sum(1 for m in members if m.get('scored') or m.get('result') == 'scored')
        bps, peak_bps = _burst_bps(times)
        loc = _group_location_summary(members)
        bursts.append({
            'burst_index': idx,
            'team_key': members[0].get('team_key'),
            'alliance': members[0].get('alliance'),
            'start_ms': start_ms,
            'end_ms': end_ms,
            'duration_ms': dur_ms,
            'duration_s': _round(dur_ms / 1000.0, 2),
            'balls': n_balls,
            'count': n_balls,
            'scored': n_scored,
            'accuracy': _round(n_scored / n_balls, 2) if n_balls else None,
            'bps': bps,
            'rate_bps': bps,
            'peak_bps': peak_bps,
            'phase': shot_period(start_ms),
            **loc,
        })
    return bursts


def summarize_bursts(bursts):
    """Aggregate statistics across a robot's bursts."""
    if not bursts:
        return {
            'count': 0,
            'multi_ball_count': 0,
            'total_balls': 0,
            'mean_balls': None,
            'median_balls': None,
            'max_balls': 0,
            'mean_duration_s': None,
            'median_duration_s': None,
            'mean_bps': None,
            'median_bps': None,
            'peak_bps': None,
        }
    counts = [b['balls'] for b in bursts]
    multi = [b for b in bursts if b['balls'] >= 2]
    durs = [b['duration_s'] for b in multi] if multi else [b['duration_s'] for b in bursts]
    bps_vals = [b['bps'] for b in bursts if b.get('bps') is not None]
    peak_vals = [b['peak_bps'] for b in bursts if b.get('peak_bps') is not None]
    return {
        'count': len(bursts),
        'multi_ball_count': len(multi),
        'total_balls': sum(counts),
        'mean_balls': _round(_mean(counts), 2),
        'median_balls': _round(_median(counts), 1),
        'max_balls': int(max(counts)),
        'mean_duration_s': _round(_mean(durs), 2),
        'median_duration_s': _round(_median(durs), 2),
        'mean_bps': _round(_mean(bps_vals), 2),
        'median_bps': _round(_median(bps_vals), 2),
        'peak_bps': _round(max(peak_vals), 2) if peak_vals else None,
    }


def group_cycles(shots, cycle_gap_ms=CYCLE_GAP_MS, burst_gap_ms=BURST_GAP_MS):
    """Group a robot's shots into shooting cycles (visits separated by > `cycle_gap_ms`)."""
    ordered = sorted((shots or []), key=lambda s: int(s.get('t_ms', 0)))
    groups = []
    for s in ordered:
        t = int(s['t_ms'])
        if groups and t - groups[-1][-1]['t_ms'] <= cycle_gap_ms:
            groups[-1].append(s)
        else:
            groups.append([s])

    cycles = []
    for idx, members in enumerate(groups):
        times = [int(m['t_ms']) for m in members]
        start_ms, end_ms = times[0], times[-1]
        dur_ms = end_ms - start_ms
        n_balls = len(members)
        n_scored = sum(1 for m in members if m.get('scored') or m.get('result') == 'scored')
        bps, peak_bps = _burst_bps(times)
        sub_bursts = group_bursts(members, gap_ms=burst_gap_ms)
        prev = cycles[-1] if cycles else None
        gap_before_ms = (start_ms - prev['end_ms']) if prev is not None else None
        c2c_ms = (start_ms - prev['start_ms']) if prev is not None else None
        gap_s = _round(gap_before_ms / 1000.0, 2) if gap_before_ms is not None else None
        c2c_s = _round(c2c_ms / 1000.0, 2) if c2c_ms is not None else None
        loc = _group_location_summary(members)
        cycles.append({
            'cycle_index': idx,
            'team_key': members[0].get('team_key'),
            'alliance': members[0].get('alliance'),
            'start_ms': start_ms,
            'end_ms': end_ms,
            'duration_ms': dur_ms,
            'duration_s': _round(dur_ms / 1000.0, 2),
            'balls': n_balls,
            'count': n_balls,
            'scored': n_scored,
            'accuracy': _round(n_scored / n_balls, 2) if n_balls else None,
            'burst_count': len(sub_bursts),
            'bps': bps,
            'rate_bps': bps,
            'peak_bps': peak_bps,
            'gap_before_ms': gap_before_ms,
            'gap_before_s': gap_s,
            'gap_s': gap_s,
            'cycle_to_cycle_ms': c2c_ms,
            'cycle_to_cycle_s': c2c_s,
            'phase': shot_period(start_ms),
            **loc,
        })
    return cycles


def summarize_cycles(cycles):
    """Aggregate cycle statistics and time-between-cycles summary."""
    if not cycles:
        empty_tbc = {
            'count': 0,
            'gaps_s': [],
            'mean_s': None,
            'median_s': None,
            'min_s': None,
            'max_s': None,
            'sd_s': None,
            'cycle_to_cycle_s': [],
            'mean_cycle_to_cycle_s': None,
            'median_cycle_to_cycle_s': None,
        }
        empty_summary = {
            'count': 0,
            'total_balls': 0,
            'mean_balls': None,
            'median_balls': None,
            'max_balls': 0,
            'mean_duration_s': None,
            'median_duration_s': None,
            'mean_gap_s': None,
            'median_gap_s': None,
            'min_gap_s': None,
            'max_gap_s': None,
            'mean_cycle_s': None,
            'median_cycle_s': None,
            'shooting_share': None,
        }
        return empty_summary, empty_tbc

    counts = [c['balls'] for c in cycles]
    durs = [c['duration_s'] for c in cycles]
    gaps = [c['gap_s'] for c in cycles[1:] if c.get('gap_s') is not None]
    c2c = [c['cycle_to_cycle_s'] for c in cycles[1:] if c.get('cycle_to_cycle_s') is not None]
    span_ms = cycles[-1]['end_ms'] - cycles[0]['start_ms']
    shooting_ms = sum(c['duration_ms'] for c in cycles)

    tbc = {
        'count': len(gaps),
        'gaps_s': gaps,
        'mean_s': _round(_mean(gaps), 2),
        'median_s': _round(_median(gaps), 2),
        'min_s': _round(min(gaps), 2) if gaps else None,
        'max_s': _round(max(gaps), 2) if gaps else None,
        'sd_s': _round(_sd(gaps), 2),
        'cycle_to_cycle_s': c2c,
        'mean_cycle_to_cycle_s': _round(_mean(c2c), 2),
        'median_cycle_to_cycle_s': _round(_median(c2c), 2),
    }
    summary = {
        'count': len(cycles),
        'total_balls': sum(counts),
        'mean_balls': _round(_mean(counts), 2),
        'median_balls': _round(_median(counts), 1),
        'max_balls': int(max(counts)),
        'mean_duration_s': _round(_mean(durs), 2),
        'median_duration_s': _round(_median(durs), 2),
        'mean_gap_s': tbc['mean_s'],
        'median_gap_s': tbc['median_s'],
        'min_gap_s': tbc['min_s'],
        'max_gap_s': tbc['max_s'],
        'mean_cycle_s': tbc['mean_cycle_to_cycle_s'],
        'median_cycle_s': tbc['median_cycle_to_cycle_s'],
        'shooting_share': _round(shooting_ms / span_ms, 2) if span_ms > 0 else None,
    }
    return summary, tbc


def _phase_stats(shots_in_phase, credited_scored, burst_gap_ms=BURST_GAP_MS, cycle_gap_ms=CYCLE_GAP_MS):
    n_shots = len(shots_in_phase)
    linked = sum(1 for s in shots_in_phase if s.get('scored') or s.get('result') == 'scored')
    bursts = group_bursts(shots_in_phase, gap_ms=burst_gap_ms)
    cycles = group_cycles(shots_in_phase, cycle_gap_ms=cycle_gap_ms, burst_gap_ms=burst_gap_ms)
    bps_vals = [b['bps'] for b in bursts if b.get('bps') is not None]
    peak_vals = [b['peak_bps'] for b in bursts if b.get('peak_bps') is not None]
    return {
        'shots': int(n_shots),
        'scored': int(credited_scored),
        'balls_in_hub': int(credited_scored),
        'linked_scored': int(linked),
        'accuracy': _round(linked / n_shots, 2) if n_shots else None,
        'bursts': len(bursts),
        'cycles': len(cycles),
        'mean_bps': _round(_mean(bps_vals), 2),
        'peak_bps': _round(max(peak_vals), 2) if peak_vals else None,
    }


def _robot_findings(shots, spots, bands, burst_summary, cycle_summary):
    """Plain-language scouting findings for one robot (compatible with shotProfile.js)."""
    n_shots = len(shots)
    findings = []
    if n_shots >= MIN_SHOTS_FOR_CLAIM:
        if spots:
            top = spots[0]
            if top['share'] >= 0.6 and top.get('distance_m') is not None:
                findings.append(f"Shoots mostly from one spot ({int(round(top['share'] * 100))}% of shots), {top['distance_m']:.1f} m from the hub.")
            elif len(spots) >= 3:
                findings.append(f"Shoots from several places ({len(spots)} spots), so it is not tied to one position.")
        solid = [b for b in bands if b['shots'] >= 3 and b.get('accuracy') is not None]
        if len(solid) >= 2:
            best = max(solid, key=lambda b: b['accuracy'])
            worst = min(solid, key=lambda b: b['accuracy'])
            if best['accuracy'] - worst['accuracy'] >= 0.25:
                findings.append(f"Most accurate {best['label']} ({int(round(best['accuracy'] * 100))}%), weakest {worst['label']} ({int(round(worst['accuracy'] * 100))}%).")
        if (burst_summary.get('median_balls') or 0) >= 3:
            bps_txt = f" at {burst_summary['mean_bps']:.1f} balls/s" if burst_summary.get('mean_bps') else ''
            findings.append(f"Empties the hopper in bursts of about {burst_summary['median_balls']:g} balls{bps_txt}.")
        if (cycle_summary.get('median_gap_s') or 0) >= 6.0:
            findings.append(f"{cycle_summary['count']} shooting cycles (median {cycle_summary['median_balls']:g} balls/cycle, {cycle_summary['median_gap_s']:.1f} s between cycles).")
    elif n_shots > 0:
        findings.append(f"Only {n_shots} tracked shot{'' if n_shots == 1 else 's'}, too few to say where it shoots from.")
    return findings


def _parse_observations(observations):
    """Turn vision_observations rows (from fuel_rows.py or fuel.py) into `(entries, shots)`."""
    valid = [
        o for o in (observations or [])
        if isinstance(o, dict) and o.get('review_status') not in ('rejected', 'unobservable')
        and o.get('source') != 'scoreboard' and (o.get('value') or {}).get('source') != 'scoreboard'
    ]
    entries = []
    shots = []
    has_counter_scored = any(
        o.get('observation_type') == 'fuel_scored' and (o.get('source') == 'fuel_counter' or (o.get('value') or {}).get('source') == 'fuel_counter')
        for o in valid
    )
    for o in valid:
        otype = o.get('observation_type')
        val = o.get('value') if isinstance(o.get('value'), dict) else {}
        t_ms = int(round(float(o.get('started_ms') or val.get('t_ms') or 0)))
        team_key = o.get('team_key')
        alliance = o.get('alliance')
        if not alliance and val.get('hub') in FC.HUBS:
            alliance = FC.HUBS[val['hub']]
        count = max(1, int(round(float(val.get('count') or 1))))
        if otype == 'fuel_scored':
            if has_counter_scored and o.get('source') == 'fuel_ballistic':
                continue
            hub = val.get('hub') or (f'{alliance}_hub' if alliance in ('red', 'blue') else None)
            for _ in range(count):
                entries.append({
                    't_ms': t_ms,
                    'hub': hub,
                    'alliance': alliance,
                    'team_key': team_key,
                    'launch_t_ms': val.get('launch_ms') or val.get('launch_t_ms'),
                    'hub_active': o.get('hub_active') if 'hub_active' in o else val.get('hub_active'),
                    'confidence': o.get('confidence') if o.get('confidence') is not None else val.get('confidence'),
                })
        elif otype in ('fuel_attempt', 'fuel_shot'):
            launch_xyz = val.get('launch')
            x = val.get('x') if val.get('x') is not None else (launch_xyz[0] if isinstance(launch_xyz, (list, tuple)) and len(launch_xyz) >= 2 else None)
            y = val.get('y') if val.get('y') is not None else (launch_xyz[1] if isinstance(launch_xyz, (list, tuple)) and len(launch_xyz) >= 2 else None)
            score_ms = val.get('score_ms') if val.get('score_ms') is not None else (val.get('entry') or {}).get('t_ms')
            for _ in range(count):
                shots.append({
                    't_ms': t_ms,
                    'team_key': team_key,
                    'alliance': alliance,
                    'x': x,
                    'y': y,
                    'launch': launch_xyz,
                    'distance_m': val.get('distance_to_hub_m') if val.get('distance_to_hub_m') is not None else val.get('distance_m'),
                    'speed_mps': val.get('speed_mps'),
                    'elevation_deg': val.get('elevation_deg'),
                    'result': val.get('result'),
                    'score_ms': score_ms,
                    'entry_t_ms': score_ms,
                    'confidence': o.get('confidence') if o.get('confidence') is not None else val.get('confidence'),
                })
    return entries, shots


def _credit_entries_subset(subset_entries, shots, members_by_alliance):
    """Credit a subset of hub entries to robots per alliance, returning ({team_key: int}, {alliance: int})."""
    explicit = [e for e in subset_entries if e.get('team_key')]
    needs_credit = [e for e in subset_entries if not e.get('team_key') and 'attribution' not in e]
    resolved_unknown = {color: sum(1 for e in subset_entries if not e.get('team_key')
                                   and 'attribution' in e and e.get('alliance') == color)
                        for color in ('red', 'blue')}
    credit, unattr = FC.credit_entries(needs_credit, shots)
    for e in explicit:
        color = e.get('alliance')
        tk = e.get('team_key')
        if color in ('red', 'blue'):
            credit[(color, tk)] = credit.get((color, tk), 0.0) + 1.0
    per_robot = {}
    per_unattr = {'red': 0, 'blue': 0}
    for color in ('red', 'blue'):
        members = members_by_alliance.get(color, [])
        fractions = {tk: credit.get((color, tk), 0.0) for tk in members}
        whole = FC._largest_remainder(fractions, int(round(sum(fractions.values()))))
        for tk in members:
            per_robot[tk] = whole.get(tk, 0)
        unnamed = int(round(credit.get((color, None), 0.0)))
        per_unattr[color] = int(unattr.get(color, 0)) + unnamed + resolved_unknown[color]
    return per_robot, per_unattr


def _credit_by_period(entries, shots, members_by_alliance):
    """Credit hub entries per period ('auto', 'teleop_only', 'endgame') and by hub activity ('active', 'inactive')."""
    period_entries = {
        'auto': [e for e in entries if shot_period(e['t_ms'], scoring=True) == 'auto'],
        'teleop_only': [e for e in entries if shot_period(e['t_ms'], scoring=True) == 'teleop'],
        'endgame': [e for e in entries if shot_period(e['t_ms'], scoring=True) == 'endgame'],
        'active': [e for e in entries if e.get('hub_active') is True],
        'inactive': [e for e in entries if e.get('hub_active') is False],
        'unknown': [e for e in entries if e.get('hub_active') is None],
    }
    robot_period_scored = {}
    unattributed_period = {}
    for p_name, p_ents in period_entries.items():
        r_map, u_map = _credit_entries_subset(p_ents, shots, members_by_alliance)
        robot_period_scored[p_name] = r_map
        unattributed_period[p_name] = u_map
    return robot_period_scored, unattributed_period


def compute_features(entries=(), shots=(), robot_tracks=(), *, tracks=None, calibration=None, observations=None,
                     offset_s=0.0, width=1280, height=720, burst_gap_ms=BURST_GAP_MS, cycle_gap_ms=CYCLE_GAP_MS,
                     spot_radius_m=SPOT_RADIUS_M, cell_size_m=CELL_SIZE_M):
    """Compute per-robot and per-alliance scouting features from counted balls.

    Inputs may be:
      * `entries` + `shots` + `robot_tracks` (from `fuel_count.py` or `fuel_rows.py`), OR
      * `tracks` (raw ball tracks) + `calibration` + `robot_tracks`, OR
      * `observations` (`vision_observations` rows) + `robot_tracks`.
    """
    if observations is not None and not entries and not shots:
        entries, shots = _parse_observations(observations)

    if tracks is not None and (not entries or not shots):
        if calibration and not entries:
            h_norm = np.asarray(calibration['homography_norm'], dtype=float)
            h_ref_to_field = h_norm @ np.diag([1.0 / width, 1.0 / height, 1.0])
            camera = camera_from_pixel_homography(h_ref_to_field, int(width), int(height))
            if camera is not None:
                polys = FC.hub_polygons(camera, HUBS)
                entries = hub_entries_with_tracks(tracks, polys, offset_s)
        if not shots:
            shots = launches_with_positions(tracks, robot_tracks, offset_s=offset_s, width=width, height=height, calibration=calibration)

    entries = [dict(e) for e in (entries or []) if 0 <= float(e.get('t_ms', -1)) <= MATCH_MS]
    shots = [dict(s) for s in (shots or []) if 0 <= float(s.get('t_ms', s.get('tMs', s.get('started_ms', -1)))) <= MATCH_MS]
    shots = _enrich_shots_positions(shots, robot_tracks, calibration=calibration)

    # Derive vision-only autoWinner from auto hub entries
    red_auto_entries = sum(1 for e in entries if e.get('alliance') == 'red' and shot_phase(e['t_ms'], scoring=True) == 'auto')
    blue_auto_entries = sum(1 for e in entries if e.get('alliance') == 'blue' and shot_phase(e['t_ms'], scoring=True) == 'auto')
    auto_winner, auto_winner_uncertain = derive_auto_winner(red_auto_entries, blue_auto_entries)

    linked_shots, linked_entries = link_launches_to_entries(shots, entries)

    # Annotate entries with hub_active
    for e in linked_entries:
        if 'hub_active' not in e or (e.get('hub_active') is None and 'attribution' not in e):
            e['hub_active'] = hub_active(e.get('alliance'), e['t_ms'], auto_winner)

    # Annotate shots with phase, period, hub_active, and camelCase aliases for JS UI consumers
    for idx, s in enumerate(linked_shots):
        t_ms = int(s['t_ms'])
        entry_ms = s.get('entry_t_ms')
        eff_ms = int(entry_ms) if entry_ms is not None else t_ms
        s['id'] = s.get('id') or f"{s.get('team_key') or 'anon'}-{t_ms}-{idx}"
        s['tMs'] = t_ms
        s['scoreMs'] = s.get('score_ms')
        s['phase'] = shot_phase(t_ms, scoring=False)
        s['period'] = shot_period(t_ms, scoring=False)
        s['scored_phase'] = 'auto' if (s['phase'] == 'auto' and eff_ms <= TELEOP_START_MS) else 'teleop'
        s['scored_period'] = shot_period(eff_ms, scoring=True)
        s['hub_active'] = hub_active(s.get('alliance'), eff_ms, auto_winner)
        s['hubActive'] = s['hub_active']

    # Collect unique named robots in deterministic order
    team_order = []
    team_alliance = {}
    for r in robot_tracks or []:
        tk = r.get('team_key')
        if tk and tk not in team_alliance:
            team_order.append(tk)
            team_alliance[tk] = _infer_robot_alliance(r, calibration=calibration)
    for s in linked_shots:
        tk = s.get('team_key')
        if tk and tk not in team_alliance:
            team_order.append(tk)
            team_alliance[tk] = s.get('alliance')
        elif tk and not team_alliance.get(tk) and s.get('alliance'):
            team_alliance[tk] = s.get('alliance')
    for e in linked_entries:
        tk = e.get('team_key')
        if tk and tk not in team_alliance:
            team_order.append(tk)
            team_alliance[tk] = e.get('alliance')
        elif tk and not team_alliance.get(tk) and e.get('alliance'):
            team_alliance[tk] = e.get('alliance')

    members_by_alliance = {
        color: [tk for tk in team_order if team_alliance.get(tk) == color]
        for color in ('red', 'blue')
    }
    robot_period_scored, unattributed_period = _credit_by_period(linked_entries, linked_shots, members_by_alliance)

    match_duration_min = MATCH_MS / 60_000.0
    robots_out = []
    for tk in team_order:
        alliance = team_alliance.get(tk)
        mine = [s for s in linked_shots if s.get('team_key') == tk]
        n_shots = len(mine)
        auto_s = [s for s in mine if s['phase'] == 'auto']
        teleop_s = [s for s in mine if s['phase'] == 'teleop']
        teleop_only_s = [s for s in mine if s['period'] == 'teleop']
        endgame_s = [s for s in mine if s['period'] == 'endgame']

        a_scored = int(robot_period_scored['auto'].get(tk, 0))
        to_scored = int(robot_period_scored['teleop_only'].get(tk, 0))
        eg_scored = int(robot_period_scored['endgame'].get(tk, 0))
        t_scored = to_scored + eg_scored
        tot_scored = a_scored + t_scored

        unk_balls = int(robot_period_scored['unknown'].get(tk, 0))
        inact_balls = int(robot_period_scored['inactive'].get(tk, 0))
        conf_active = max(0, tot_scored - inact_balls - unk_balls)
        active_scored = None if unk_balls > 0 else conf_active

        linked_sc = sum(1 for s in mine if s.get('scored'))
        acc = _round(linked_sc / n_shots, 2) if n_shots else None

        same_track_shots = [s for s in mine if s.get('link_kind') == 'same_track']
        unamb_shots = [s for s in mine if s.get('unambiguous')]
        unamb_sc = sum(1 for s in unamb_shots if s.get('scored'))
        shared_shots = [s for s in mine if not s.get('unambiguous')]
        shared_sc = sum(1 for s in shared_shots if s.get('scored'))
        flights = [s['flight_ms'] for s in mine if s.get('flight_ms') is not None]

        acc_detail = {
            'overall': acc,
            'linked_scored': int(linked_sc),
            'shots': int(n_shots),
            'same_track_shots': len(same_track_shots),
            'same_track_scored': len(same_track_shots),
            'unambiguous_shots': len(unamb_shots),
            'unambiguous_scored': int(unamb_sc),
            'unambiguous_accuracy': _round(unamb_sc / len(unamb_shots), 2) if unamb_shots else None,
            'shared_window_shots': len(shared_shots),
            'shared_window_scored': int(shared_sc),
            'shared_window_accuracy': _round(shared_sc / len(shared_shots), 2) if shared_shots else None,
            'mean_flight_ms': int(round(_mean(flights))) if flights else None,
            'median_flight_ms': int(round(_median(flights))) if flights else None,
        }

        spots = shooting_spots(mine, radius_m=spot_radius_m)
        bands = accuracy_by_distance(mine)
        hmap = build_heatmap(mine, cell_size_m=cell_size_m)
        bursts = group_bursts(mine, gap_ms=burst_gap_ms)
        b_sum = summarize_bursts(bursts)
        cycles = group_cycles(mine, cycle_gap_ms=cycle_gap_ms, burst_gap_ms=burst_gap_ms)
        c_sum, tbc = summarize_cycles(cycles)

        phases = {
            'auto': _phase_stats(auto_s, a_scored, burst_gap_ms=burst_gap_ms, cycle_gap_ms=cycle_gap_ms),
            'teleop': _phase_stats(teleop_s, t_scored, burst_gap_ms=burst_gap_ms, cycle_gap_ms=cycle_gap_ms),
            'teleop_only': _phase_stats(teleop_only_s, to_scored, burst_gap_ms=burst_gap_ms, cycle_gap_ms=cycle_gap_ms),
            'endgame': _phase_stats(endgame_s, eg_scored, burst_gap_ms=burst_gap_ms, cycle_gap_ms=cycle_gap_ms),
        }

        active_known = [s for s in mine if s.get('hub_active') is not None]
        active_true = sum(1 for s in active_known if s.get('hub_active') is True)
        inactive_shots = max(inact_balls, sum(1 for s in mine if s.get('hub_active') is False))
        active_shooting_s = _round(sum(c['duration_ms'] for c in cycles) / 1000.0, 2)
        dists = [s['distance_m'] for s in mine if s.get('distance_m') is not None]

        rates = {
            'shots_per_min': _round(n_shots / match_duration_min, 2),
            'scored_per_min': _round(tot_scored / match_duration_min, 2),
            'linked_scored_per_min': _round(linked_sc / match_duration_min, 2),
            'burst_bps': b_sum['mean_bps'],
            'median_burst_bps': b_sum['median_bps'],
            'peak_bps': b_sum['peak_bps'],
            'sustained_bps': _round(n_shots / (MATCH_MS / 1000.0), 2),
            'active_shooting_s': active_shooting_s,
            'active_hub_fraction': _round(active_true / len(active_known), 2) if active_known else None,
        }

        findings = _robot_findings(mine, spots, bands, b_sum, c_sum)

        robots_out.append({
            'team_key': tk,
            'alliance': alliance,
            'shots': int(n_shots),
            'scored': int(tot_scored),
            'balls_in_hub': int(tot_scored),
            'ballsInHub': int(tot_scored),
            'active_scored': active_scored,
            'confirmed_scored': int(conf_active),
            'inactive_scored': int(inact_balls),
            'inactive_balls': int(inact_balls),
            'unknown_hub_balls': int(unk_balls),
            'linked_scored': int(linked_sc),
            'accuracy': acc,
            'auto_shots': len(auto_s),
            'teleop_shots': len(teleop_s),
            'teleop_only_shots': len(teleop_only_s),
            'endgame_shots': len(endgame_s),
            'auto_scored': int(a_scored),
            'auto_balls_in_hub': int(a_scored),
            'teleop_scored': int(t_scored),
            'teleop_balls_in_hub': int(t_scored),
            'teleop_only_scored': int(to_scored),
            'endgame_scored': int(eg_scored),
            'endgame_balls_in_hub': int(eg_scored),
            'first_shot_ms': int(mine[0]['t_ms']) if mine else None,
            'firstShotMs': int(mine[0]['t_ms']) if mine else None,
            'distance_m': _round(_mean(dists), 2),
            'distanceM': _round(_mean(dists), 1),
            'shots_while_inactive': int(inactive_shots),
            'shotsWhileInactive': int(inactive_shots),
            'volleys': b_sum['count'],
            'volley_size': b_sum['mean_balls'],
            'bps': b_sum['mean_bps'],
            'max_bps': b_sum['peak_bps'],
            'phases': phases,
            'accuracy_detail': acc_detail,
            'spots': spots,
            'shot_locations': spots,
            'distance_bands': bands,
            'bands': bands,
            'cycle': {
                'volleys': b_sum['count'],
                'medianVolleySize': b_sum['median_balls'],
                'medianPauseS': c_sum['median_gap_s'],
                'shootingShare': c_sum['shooting_share'],
            },
            'heatmap': hmap,
            'bursts': bursts,
            'burst_summary': b_sum,
            'cycles': cycles,
            'cycle_summary': c_sum,
            'time_between_cycles': tbc,
            'rates': rates,
            'findings': findings,
        })

    robots_out.sort(key=lambda r: (str(r.get('alliance') or ''), -(r.get('scored') or 0), -(r.get('shots') or 0)))
    robots_by_team = {r['team_key']: r for r in robots_out if r.get('team_key')}

    alliances_out = {}
    for color in ('red', 'blue'):
        a_entries = [e for e in linked_entries if e.get('alliance') == color]
        a_shots = [s for s in linked_shots if s.get('alliance') == color]
        a_auto_e = [e for e in a_entries if shot_period(e['t_ms'], scoring=True) == 'auto']
        a_to_e = [e for e in a_entries if shot_period(e['t_ms'], scoring=True) == 'teleop']
        a_eg_e = [e for e in a_entries if shot_period(e['t_ms'], scoring=True) == 'endgame']

        a_auto_s = [s for s in a_shots if s['phase'] == 'auto']
        a_tele_s = [s for s in a_shots if s['phase'] == 'teleop']
        a_to_s = [s for s in a_shots if s['period'] == 'teleop']
        a_eg_s = [s for s in a_shots if s['period'] == 'endgame']

        unattr_total = sum(unattributed_period[p][color] for p in ('auto', 'teleop_only', 'endgame'))
        linked_sc = sum(1 for s in a_shots if s.get('scored'))
        confirmed_active = sum(1 for e in a_entries if e.get('hub_active') is True)
        inactive_e = sum(1 for e in a_entries if e.get('hub_active') is False)
        unknown_e = sum(1 for e in a_entries if e.get('hub_active') is None)

        alliances_out[color] = {
            'scored': len(a_entries),
            'balls_in_hub': len(a_entries),
            'ballsInHub': len(a_entries),
            'active_scored': None if unknown_e > 0 else int(confirmed_active),
            'confirmed_scored': int(confirmed_active),
            'inactive_scored': int(inactive_e),
            'inactive_balls': int(inactive_e),
            'unknown_hub_balls': int(unknown_e),
            'auto_scored': len(a_auto_e),
            'auto_balls_in_hub': len(a_auto_e),
            'teleop_scored': len(a_to_e) + len(a_eg_e),
            'teleop_balls_in_hub': len(a_to_e) + len(a_eg_e),
            'teleop_only_scored': len(a_to_e),
            'endgame_scored': len(a_eg_e),
            'endgame_balls_in_hub': len(a_eg_e),
            'shots': len(a_shots),
            'linked_scored': int(linked_sc),
            'unattributed': int(unattr_total),
            'accuracy': _round(linked_sc / len(a_shots), 2) if a_shots else None,
            'timeline': FC._timeline(a_entries),
            'phases': {
                'auto': _phase_stats(a_auto_s, len(a_auto_e), burst_gap_ms=burst_gap_ms, cycle_gap_ms=cycle_gap_ms),
                'teleop': _phase_stats(a_tele_s, len(a_to_e) + len(a_eg_e), burst_gap_ms=burst_gap_ms, cycle_gap_ms=cycle_gap_ms),
                'teleop_only': _phase_stats(a_to_s, len(a_to_e), burst_gap_ms=burst_gap_ms, cycle_gap_ms=cycle_gap_ms),
                'endgame': _phase_stats(a_eg_s, len(a_eg_e), burst_gap_ms=burst_gap_ms, cycle_gap_ms=cycle_gap_ms),
            },
            'heatmap': build_heatmap(a_shots, cell_size_m=cell_size_m),
        }

    all_spots = [sp for r in robots_out for sp in r['spots']]
    all_bursts = sorted((b for r in robots_out for b in r['bursts']), key=lambda b: b['start_ms'])
    all_cycles = sorted((c for r in robots_out for c in r['cycles']), key=lambda c: c['start_ms'])

    return {
        'version': 1,
        'method': 'ball_features (fuel_features.py)',
        'auto_winner': auto_winner,
        'autoWinner': auto_winner,
        'auto_winner_uncertain': auto_winner_uncertain,
        'alliances': alliances_out,
        'robots': robots_out,
        'robots_by_team': robots_by_team,
        'shot_locations': all_spots,
        'spots': all_spots,
        'shots': linked_shots,
        'shot_events': linked_shots,
        'entries': linked_entries,
        'bursts': all_bursts,
        'cycles': all_cycles,
        'heatmap': build_heatmap(linked_shots, cell_size_m=cell_size_m),
    }


def _compact_robot_for_snapshot(r, max_spots=6, max_cells=8, max_cycles=12, max_bursts=10):
    return {
        'team_key': r['team_key'],
        'alliance': r['alliance'],
        'shots': r['shots'],
        'scored': r['scored'],
        'balls_in_hub': r['balls_in_hub'],
        'active_scored': r['active_scored'],
        'confirmed_scored': r['confirmed_scored'],
        'inactive_scored': r['inactive_scored'],
        'unknown_hub_balls': r['unknown_hub_balls'],
        'linked_scored': r['linked_scored'],
        'accuracy': r['accuracy'],
        'auto_shots': r['auto_shots'],
        'teleop_shots': r['teleop_shots'],
        'teleop_only_shots': r['teleop_only_shots'],
        'endgame_shots': r['endgame_shots'],
        'auto_scored': r['auto_scored'],
        'teleop_scored': r['teleop_scored'],
        'teleop_only_scored': r['teleop_only_scored'],
        'endgame_scored': r['endgame_scored'],
        'first_shot_ms': r['first_shot_ms'],
        'distance_m': r['distance_m'],
        'shots_while_inactive': r['shots_while_inactive'],
        'volleys': r['volleys'],
        'volley_size': r['volley_size'],
        'bps': r['bps'],
        'max_bps': r['max_bps'],
        'phases': r['phases'],
        'accuracy_detail': r['accuracy_detail'],
        'spots': r['spots'][:max_spots],
        'distance_bands': r['distance_bands'],
        'heatmap': {**r['heatmap'], 'cells': r['heatmap']['cells'][:max_cells]},
        'burst_summary': r['burst_summary'],
        'cycle_summary': r['cycle_summary'],
        'time_between_cycles': r['time_between_cycles'],
        'rates': r['rates'],
        'findings': r['findings'],
        'cycles': r['cycles'][:max_cycles],
        'bursts': r['bursts'][:max_bursts],
    }


def from_fuel_count_payload(payload, robot_tracks=(), *, entries=(), shots=(), tracks=None, calibration=None,
                            offset_s=0.0, width=1280, height=720, max_bytes=SNAPSHOT_MAX_BYTES, **kwargs):
    """Attach `features` to a `fuel_count.summarize(...)` payload and enrich `payload['robots']`,
    keeping the serialized JSON under `max_bytes` (60 KB default for vision_reference_snapshots).
    """
    if not isinstance(payload, dict):
        return payload
    features = compute_features(
        entries=entries, shots=shots, robot_tracks=robot_tracks,
        tracks=tracks, calibration=calibration, offset_s=offset_s, width=width, height=height, **kwargs
    )
    by_team = features.get('robots_by_team') or {}
    summary_robots = []
    for r in (payload.get('robots') or features['robots']):
        tk = r.get('team_key')
        feat = by_team.get(tk)
        if feat:
            summary_robots.append({
                'team_key': tk,
                'alliance': feat['alliance'],
                'shots': feat['shots'],
                'scored': r.get('scored', feat['scored']),
                'balls_in_hub': r.get('scored', feat['balls_in_hub']),
                'active_scored': r.get('active_scored', feat['active_scored']),
                'identity_evidence': r.get('identity_evidence'),
                'tracked_share': r.get('tracked_share'),
                'attribution_complete': r.get('attribution_complete'),
                'confirmed_scored': feat['confirmed_scored'],
                'inactive_scored': feat['inactive_scored'],
                'unknown_hub_balls': feat['unknown_hub_balls'],
                'linked_scored': feat['linked_scored'],
                'accuracy': feat['accuracy'],
                'auto_shots': feat['auto_shots'],
                'teleop_shots': feat['teleop_shots'],
                'endgame_shots': feat['endgame_shots'],
                'auto_scored': r.get('auto_scored', feat['auto_scored']),
                'teleop_scored': r.get('teleop_scored', feat['teleop_scored']),
                'endgame_scored': feat['endgame_scored'],
                'first_shot_ms': feat['first_shot_ms'],
                'distance_m': feat['distance_m'],
                'shots_while_inactive': feat['shots_while_inactive'],
                'volleys': feat['volleys'],
                'volley_size': feat['volley_size'],
                'bps': feat['bps'],
                'max_bps': feat['max_bps'],
                'timeline': r.get('timeline'),  # summarize's per-second count; Replay needs it to show the robot
            })
        else:
            summary_robots.append(dict(r))

    clean_alliances = {
        c: {k: v for k, v in a.items() if k != 'heatmap'}
        for c, a in (payload.get('alliances') or features['alliances']).items()
    }
    for limits in ((6, 8, 12, 10), (5, 6, 8, 6), (4, 4, 5, 4), (3, 3, 3, 2), (2, 2, 2, 1)):
        compact_robots = [_compact_robot_for_snapshot(r, *limits) for r in features['robots']]
        compact_features = {
            'version': 1,
            'method': features['method'],
            'auto_winner': features['auto_winner'],
            'auto_winner_uncertain': features['auto_winner_uncertain'],
            'alliances': {
                c: {k: v for k, v in a.items() if k not in ('timeline', 'heatmap')}
                for c, a in features['alliances'].items()
            },
            'robots': compact_robots,
            'shot_locations': [sp for r in compact_robots for sp in r['spots']],
            'heatmap': {**features['heatmap'], 'cells': features['heatmap']['cells'][:limits[1] * 2]},
        }
        candidate = {
            **payload,
            'alliances': clean_alliances,
            'auto_winner': payload.get('auto_winner', features['auto_winner']),
            'auto_winner_uncertain': payload.get('auto_winner') is None if 'auto_winner' in payload else features['auto_winner_uncertain'],
            'robots': summary_robots,
            'features': compact_features,
        }
        if not max_bytes or len(json.dumps(candidate, separators=(',', ':'))) <= max_bytes:
            return candidate
    # Core observed counts and their replay timelines take priority over
    # optional trajectory analytics. Never hand the runner an oversized snapshot
    # merely because every feature trimming level was still too large.
    candidate.pop('features', None)
    candidate['stats'] = {**candidate.get('stats', {}), 'feature_snapshot_omitted': 'size_limit'}
    if max_bytes and len(json.dumps(candidate, separators=(',', ':'))) > max_bytes:
        raise ValueError('core fuel snapshot exceeds size limit')
    return candidate


def _load_robot_tracks_for_pickle(pkl_path, data):
    """No stored robot tracks in this public copy: pass `robot_tracks` to compute_features_from_pickle."""
    return []


def compute_features_from_pickle(pkl_path, robot_tracks=None, offset_s=None, **kwargs):
    """Load a pickled match evaluation dict and compute scouting features."""
    import pickle
    path = Path(pkl_path).expanduser()
    data = pickle.loads(path.read_bytes())
    tracks = data.get('tracks') or []
    view = data.get('view') or {}
    calibration = kwargs.pop('calibration', None) or view.get('field_calibration')
    if offset_s is None:
        min_t = min((tr[0][1] for tr in tracks if tr), default=0.02)
        offset_s = round(max(0.0, float(min_t) - 0.02), 2)
    if robot_tracks is None:
        robot_tracks = _load_robot_tracks_for_pickle(path, data)
    shots = kwargs.pop('shots', ())
    if not robot_tracks and not shots and isinstance(data.get('shots'), list):
        # Fallback when offline with no robot tracks: use the pickle's fitted ballistic shots attributed by target hub
        raw_shots = []
        for idx, s in enumerate(data['shots']):
            if not isinstance(s, dict):
                continue
            alliance = s.get('alliance') or (FC.HUBS.get(s.get('target')) if s.get('target') else None)
            tk = s.get('team_key') or (f'{alliance}_shooter' if alliance else None)
            raw_shots.append({**s, 'team_key': tk, 'alliance': alliance})
        shots = raw_shots
    return compute_features(
        shots=shots,
        robot_tracks=robot_tracks,
        tracks=tracks,
        calibration=calibration,
        offset_s=offset_s,
        **kwargs,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--pickle', help='pickled match eval dict (tracks, view, ...)')
    ap.add_argument('--robot-tracks', help='JSON list of robot tracks (team_key, alliance, trajectory)')
    ap.add_argument('--calibration', help='field calibration JSON')
    ap.add_argument('--offset-s', type=float, default=None, help='video offset seconds of match start')
    ap.add_argument('--out', required=True, help='output JSON path')
    args = ap.parse_args(argv)

    calibration = json.loads(Path(args.calibration).read_text()) if args.calibration else None
    robot_tracks = json.loads(Path(args.robot_tracks).read_text()) if args.robot_tracks else None

    if args.pickle:
        result = compute_features_from_pickle(
            args.pickle,
            robot_tracks=robot_tracks,
            offset_s=args.offset_s,
            calibration=calibration,
        )
    else:
        result = compute_features(
            robot_tracks=robot_tracks or [],
            calibration=calibration,
            offset_s=args.offset_s or 0.0,
        )
    Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
