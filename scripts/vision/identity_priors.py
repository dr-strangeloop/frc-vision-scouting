"""Cross-run identity reuse (auto <-> scouting) and event auto-path fingerprinting.

Transfers verified bumper-read identities across runs of the same match by geometric
trajectory and bounding-box correspondence, and matches repeatable autonomous paths
across matches of an event in alliance-normalized field coordinates. Never consumes
TBA or human scouting data.
"""
import itertools
import json
import math
import os
import statistics
import threading
from pathlib import Path

import cache
from common import AUTO_END_MS, FIELD

VMAX = float(FIELD['robot']['maxPlausibleSpeedMps'])
TRUSTED_PRIOR_SOURCES = frozenset({'reads', 'closeup_crop', 'zoom_crop', None})


def _canon_team_key(team):
    if isinstance(team, bool) or team is None:
        return None
    s = str(team).strip()
    if not s:
        return None
    if s.startswith('frc') and s[3:].isdigit():
        return s
    return f'frc{s}' if s.isdigit() else s


def _match_candidate_team(team, candidate_teams):
    canon = _canon_team_key(team)
    if not canon:
        return None
    for cand in candidate_teams or ():
        if _canon_team_key(cand) == canon:
            return str(cand)
    return canon


def _valid_box(box):
    return (
        isinstance(box, (list, tuple))
        and len(box) == 4
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in box)
        and box[2] > box[0]
        and box[3] > box[1]
    )


def _box_iou(a, b):
    if not (_valid_box(a) and _valid_box(b)):
        return 0.0
    ix = max(0.0, min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0])))
    iy = max(0.0, min(float(a[3]), float(b[3])) - max(float(a[1]), float(b[1])))
    inter = ix * iy
    if inter <= 0.0:
        return 0.0
    area_a = (float(a[2]) - float(a[0])) * (float(a[3]) - float(a[1]))
    area_b = (float(b[2]) - float(b[0])) * (float(b[3]) - float(b[1]))
    return inter / max(1e-12, area_a + area_b - inter)


def _track_source(track):
    if not isinstance(track, dict):
        return None
    if 'identity_source' in track and track.get('identity_source') is not None:
        return track.get('identity_source')
    metrics = track.get('metrics')
    if isinstance(metrics, dict):
        return metrics.get('identity_source')
    return None


def _track_swap_risks(track):
    if not isinstance(track, dict):
        return []
    return track.get('swap_risks') or []


def _is_trusted_prior_track(track, min_prior_conf=0.60):
    if not isinstance(track, dict):
        return False
    team_key = _canon_team_key(track.get('team_key'))
    if not team_key:
        return False
    try:
        conf = float(track.get('identity_confidence') or 0.0)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(conf) or conf < min_prior_conf:
        return False
    if _track_swap_risks(track):
        return False
    if _track_source(track) not in TRUSTED_PRIOR_SOURCES:
        return False
    return True


def _extract_raw_points(item):
    if isinstance(item, dict):
        return item.get('points') if item.get('points') is not None else (item.get('trajectory') or [])
    return getattr(item, 'points', None) or getattr(item, 'trajectory', None) or []


def _points_to_ms(raw_points, offset_s=0.0, default_unit='auto'):
    """Normalize a list of point dicts to sorted [{'t_ms': int, 'x': float, 'y': float, 'box': list|None, 'calibrated': bool}]."""
    valid = []
    for p in raw_points or ():
        if not isinstance(p, dict):
            continue
        t_val = p.get('t_ms') if 't_ms' in p and p.get('t_ms') is not None else p.get('t')
        x_val, y_val = p.get('x'), p.get('y')
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) for v in (t_val, x_val, y_val)):
            continue
        valid.append((float(t_val), 't_ms' in p and p.get('t_ms') is not None, float(x_val), float(y_val), p.get('box'), p.get('calibrated', True)))
    if not valid:
        return []
    if all(has_tms for _, has_tms, _, _, _, _ in valid) or default_unit == 'ms':
        is_ms = True
    elif default_unit == 's':
        is_ms = False
    else:
        ts = sorted(t for t, _, _, _, _, _ in valid)
        dts = [ts[i] - ts[i - 1] for i in range(1, len(ts)) if ts[i] > ts[i - 1]]
        if dts:
            med_dt = statistics.median(dts)
            is_ms = (med_dt >= 10.0)
        else:
            is_ms = (offset_s == 0.0 and max(abs(t) for t in ts) > 1000.0)
    out = []
    for t_val, has_tms, x_val, y_val, box, cal in valid:
        t_ms = int(round(t_val)) if (has_tms or is_ms) else int(round((t_val - float(offset_s)) * 1000.0))
        out.append({
            't_ms': t_ms,
            'x': x_val,
            'y': y_val,
            'box': list(box) if _valid_box(box) else None,
            'calibrated': cal is not False,
        })
    out.sort(key=lambda pt: pt['t_ms'])
    dedup = []
    for pt in out:
        if not dedup or dedup[-1]['t_ms'] != pt['t_ms']:
            dedup.append(pt)
    return dedup


def _reconstruct_chain_points(chain, tracklets, offset_s=0.0):
    raw = []
    for idx in chain or ():
        if 0 <= idx < len(tracklets):
            raw.extend(_extract_raw_points(tracklets[idx]))
    return _points_to_ms(raw, offset_s=offset_s, default_unit='auto')


def _pair_points_exact(pts_a, pts_b, max_dt_ms=150, dt_shift=0):
    if not pts_a or not pts_b:
        return []
    pairs = []
    used_b = set()
    for pa in pts_a:
        best_j = None
        best_dt = max_dt_ms + 1
        ta = pa['t_ms']
        for j, pb in enumerate(pts_b):
            if j in used_b:
                continue
            dt = abs((pb['t_ms'] + dt_shift) - ta)
            if dt <= max_dt_ms and dt < best_dt:
                best_dt = dt
                best_j = j
        if best_j is not None:
            used_b.add(best_j)
            pb_item = pts_b[best_j] if dt_shift == 0 else {**pts_b[best_j], 't_ms': pts_b[best_j]['t_ms'] + dt_shift}
            pairs.append((pa, pb_item))
    return pairs


def _pair_distance_stats(pairs):
    if not pairs:
        return float('inf'), float('inf'), float('inf'), 0.0
    both_cal = all(pa.get('calibrated', True) and pb.get('calibrated', True) for pa, pb in pairs)
    ious = [_box_iou(pa['box'], pb['box']) for pa, pb in pairs if pa.get('box') and pb.get('box')]
    median_iou = statistics.median(ious) if ious else 0.0
    if not both_cal:
        return float('inf'), float('inf'), float('inf'), median_iou
    dists = [math.hypot(pa['x'] - pb['x'], pa['y'] - pb['y']) for pa, pb in pairs]
    median_dist_m = statistics.median(dists)
    dxs = [pa['x'] - pb['x'] for pa, pb in pairs]
    dys = [pa['y'] - pb['y'] for pa, pb in pairs]
    med_dx = statistics.median(dxs)
    med_dy = statistics.median(dys)
    if math.hypot(med_dx, med_dy) <= 0.55 or median_iou >= 0.65:
        c_dists = [math.hypot((pa['x'] - pb['x']) - med_dx, (pa['y'] - pb['y']) - med_dy) for pa, pb in pairs]
        centered_dist_m = statistics.median(c_dists)
    else:
        centered_dist_m = median_dist_m
    eff_dist_m = min(median_dist_m, centered_dist_m + 0.10)
    return median_dist_m, centered_dist_m, eff_dist_m, median_iou


def _pair_points_by_time(pts_a, pts_b, max_dt_ms=150, min_overlap_pts=4, allow_shift=True):
    if not pts_a or not pts_b:
        return []
    pairs_0 = _pair_points_exact(pts_a, pts_b, max_dt_ms=max_dt_ms, dt_shift=0)
    if not allow_shift:
        return pairs_0
    med_0, centered_0, eff_0, _ = _pair_distance_stats(pairs_0)
    if len(pairs_0) >= min_overlap_pts and med_0 <= 0.45 and centered_0 <= 0.18:
        return pairs_0

    candidates = [(0, pairs_0, med_0, eff_0)]
    for dt_shift in (-2000, -1500, -1000, -500, -250, 250, 500, 1000, 1500, 2000):
        shifted_pairs = _pair_points_exact(pts_a, pts_b, max_dt_ms=max_dt_ms, dt_shift=dt_shift)
        if shifted_pairs:
            med_d, _, eff_d, _ = _pair_distance_stats(shifted_pairs)
            candidates.append((dt_shift, shifted_pairs, med_d, eff_d))

    max_n = max((len(p) for _, p, _, _ in candidates), default=0)
    if max_n == 0:
        return []

    def _rank_candidate(item):
        dt_shift, pairs, med_d, eff_d = item
        n = len(pairs)
        enough = 1 if n >= min_overlap_pts else 0
        overlap_term = min(1.0, n / max(float(min_overlap_pts), max_n * 0.85))
        dist_val = eff_d if math.isfinite(eff_d) else (med_d if math.isfinite(med_d) else 0.0)
        shift_pen = 0.01 if dt_shift != 0 else 0.0
        return (enough, overlap_term - dist_val - shift_pen, n, -abs(dt_shift))

    best = max(candidates, key=_rank_candidate)
    return best[1]


def match_prior_run_identities(
    chains,
    tracklets,
    prior_tracks,
    alliance,
    offset_s=0.0,
    min_prior_conf=0.60,
    max_dt_ms=150,
    min_overlap_pts=4,
    min_box_iou=0.45,
    max_field_dist_m=0.50,
    min_dist_margin_m=0.60,
):
    """Transfer trusted identities from a prior run of the same match to current chains."""
    if not chains or not prior_tracks:
        return []
    eligible_priors = []
    team_counts = {}
    for tr in prior_tracks:
        if not isinstance(tr, dict) or tr.get('alliance') != alliance:
            continue
        if not _is_trusted_prior_track(tr, min_prior_conf=min_prior_conf):
            continue
        tk = _canon_team_key(tr.get('team_key'))
        team_counts[tk] = team_counts.get(tk, 0) + 1
        eligible_priors.append((tk, tr))
    # Drop any team claimed by more than one prior track on the same alliance
    unique_priors = []
    for tk, tr in eligible_priors:
        if team_counts.get(tk, 0) != 1:
            continue
        pts = _points_to_ms(_extract_raw_points(tr), offset_s=0.0, default_unit='auto')
        if len(pts) >= min_overlap_pts:
            unique_priors.append((tk, float(tr.get('identity_confidence') or 0.0), pts))
    if not unique_priors:
        return []

    chain_pts = [_reconstruct_chain_points(c, tracklets, offset_s=offset_s) for c in chains]

    def _solve_for_shift(shift_ms):
        candidates = {}
        for c_idx, c_pts in enumerate(chain_pts):
            if len(c_pts) < min_overlap_pts:
                continue
            for p_idx, (team_key, prior_conf, raw_p_pts) in enumerate(unique_priors):
                if shift_ms is None:
                    pairs = _pair_points_by_time(
                        c_pts, raw_p_pts, max_dt_ms=max_dt_ms, min_overlap_pts=min_overlap_pts, allow_shift=True
                    )
                else:
                    p_pts = raw_p_pts if shift_ms == 0 else [{**pt, 't_ms': pt['t_ms'] + shift_ms} for pt in raw_p_pts]
                    pairs = _pair_points_by_time(
                        c_pts, p_pts, max_dt_ms=max_dt_ms, min_overlap_pts=min_overlap_pts, allow_shift=False
                    )
                overlap_pts = len(pairs)
                if overlap_pts < min_overlap_pts:
                    continue
                both_cal = all(pa.get('calibrated', True) and pb.get('calibrated', True) for pa, pb in pairs)
                median_dist_m, centered_dist_m, eff_dist_m, median_iou = _pair_distance_stats(pairs)
                has_ious = any(pa.get('box') and pb.get('box') for pa, pb in pairs)

                shared_p_pts = [pb for _, pb in pairs]
                competitor_dist_m = float('inf')
                competitor_iou = 0.0
                if both_cal or has_ious:
                    for k_idx, k_pts in enumerate(chain_pts):
                        if k_idx == c_idx or not k_pts:
                            continue
                        k_pairs = _pair_points_by_time(k_pts, shared_p_pts, max_dt_ms=max_dt_ms, allow_shift=False)
                        if k_pairs:
                            if both_cal:
                                k_dists = [math.hypot(ka['x'] - pb['x'], ka['y'] - pb['y']) for ka, pb in k_pairs]
                                competitor_dist_m = min(competitor_dist_m, statistics.median(k_dists))
                            k_ious = [_box_iou(ka['box'], pb['box']) for ka, pb in k_pairs if ka.get('box') and pb.get('box')]
                            if k_ious:
                                competitor_iou = max(competitor_iou, statistics.median(k_ious))

                margin_ok = (
                    (competitor_dist_m - median_dist_m >= min_dist_margin_m)
                    or (
                        median_iou >= 0.65
                        and competitor_dist_m - eff_dist_m >= min_dist_margin_m
                        and (competitor_iou <= 0.25 or median_iou - competitor_iou >= 0.45)
                    )
                ) if math.isfinite(median_dist_m) else (competitor_iou <= 0.25 or median_iou - competitor_iou >= 0.35)
                if not margin_ok:
                    continue
                if not (eff_dist_m <= max_field_dist_m or median_iou >= min_box_iou):
                    continue

                dist_term = max(0.0, 1.0 - eff_dist_m / 1.0) if math.isfinite(eff_dist_m) else 0.0
                geom_term = (0.85 * max(dist_term, median_iou) + 0.15 * min(dist_term, median_iou)) if has_ious else dist_term
                match_score = prior_conf * geom_term * min(1.0, overlap_pts / 10.0)
                if match_score <= 0.0:
                    continue
                candidates[(c_idx, p_idx)] = {
                    'chain_index': c_idx,
                    'source': 'prior_run',
                    'team': team_key,
                    'confidence': round(min(0.98, match_score), 3),
                    'overlap_points': overlap_pts,
                    'median_dist_m': round(median_dist_m, 3) if math.isfinite(median_dist_m) else None,
                    'centered_dist_m': round(centered_dist_m, 3) if math.isfinite(centered_dist_m) else None,
                    'median_iou': round(median_iou, 3),
                    '_score': match_score,
                }

        if not candidates:
            return -1.0, []

        c_indices = sorted({c_idx for c_idx, _ in candidates})
        p_indices = sorted({p_idx for _, p_idx in candidates})
        best_total = -1.0
        best_chosen = []
        if len(c_indices) <= len(p_indices):
            for perm in itertools.permutations(p_indices + [None] * len(c_indices), len(c_indices)):
                chosen = [candidates[(c_i, p_i)] for c_i, p_i in zip(c_indices, perm) if p_i is not None and (c_i, p_i) in candidates]
                if len({item['team'] for item in chosen}) != len(chosen):
                    continue
                total = sum(item['_score'] for item in chosen)
                if total > best_total:
                    best_total, best_chosen = total, chosen
        else:
            for perm in itertools.permutations(c_indices + [None] * len(p_indices), len(p_indices)):
                chosen = [candidates[(c_i, p_i)] for c_i, p_i in zip(perm, p_indices) if c_i is not None and (c_i, p_i) in candidates]
                if len({item['chain_index'] for item in chosen}) != len(chosen):
                    continue
                total = sum(item['_score'] for item in chosen)
                if total > best_total:
                    best_total, best_chosen = total, chosen
        return best_total, best_chosen

    best_key = (-1, -1.0)
    best_chosen = []
    target_matches = min(len([c for c in chain_pts if len(c) >= min_overlap_pts]), len(unique_priors))
    for shift_ms in (0, -250, 250, -500, 500, -1000, 1000, -1500, 1500, -2000, 2000, None):
        score, chosen = _solve_for_shift(shift_ms)
        # Prefer shift=0 slightly unless a non-zero shift matches more chains or markedly better score
        adj_score = score - (0.02 if shift_ms != 0 else 0.0)
        key = (len(chosen), adj_score)
        if key > best_key:
            best_key = key
            best_chosen = chosen
        if shift_ms == 0 and len(chosen) == target_matches and score >= 0.80 * target_matches:
            break

    best_chosen.sort(key=lambda item: item['chain_index'])
    return [{k: v for k, v in item.items() if k != '_score'} for item in best_chosen]


def anchor_early_tracklets_from_prior(
    tracklets,
    prior_tracks,
    alliance,
    offset_s=0.0,
    auto_window_s=23.0,
    min_prior_conf=0.60,
):
    """Anchor early (0..auto_window_s) tracklets to verified prior-run identities before solve_alliance."""
    if not tracklets or not prior_tracks:
        return 0
    window_ms = int(round(float(auto_window_s) * 1000.0))
    early_indices = []
    for idx, tr in enumerate(tracklets):
        pts = _points_to_ms(_extract_raw_points(tr), offset_s=offset_s, default_unit='auto')
        if not pts:
            continue
        start_val = tr.get('start') if isinstance(tr, dict) else getattr(tr, 'start', None)
        if isinstance(start_val, (int, float)) and not isinstance(start_val, bool) and math.isfinite(float(start_val)):
            rel_start_s = float(start_val) - float(offset_s)
            if abs(rel_start_s - pts[0]['t_ms'] / 1000.0) > 1.0:
                is_early = rel_start_s <= float(auto_window_s)
            else:
                is_early = pts[0]['t_ms'] <= window_ms
        else:
            is_early = pts[0]['t_ms'] <= window_ms
        if is_early:
            early_indices.append(idx)

    if not early_indices:
        return 0

    matches = match_prior_run_identities(
        [[i] for i in early_indices],
        tracklets,
        prior_tracks,
        alliance,
        offset_s=offset_s,
        min_prior_conf=min_prior_conf,
    )
    anchored = 0
    for item in matches:
        c_idx = item.get('chain_index')
        if c_idx is None or not (0 <= c_idx < len(early_indices)):
            continue
        tr_idx = early_indices[c_idx]
        tr = tracklets[tr_idx]
        ev_item = {k: v for k, v in item.items() if k != 'chain_index'}
        if isinstance(tr, dict):
            ev_list = list(tr.get('evidence') or [])
            ev_list.append(ev_item)
            tr['evidence'] = ev_list
        else:
            ev_list = list(getattr(tr, 'evidence', None) or [])
            ev_list.append(ev_item)
            tr.evidence = ev_list
        anchored += 1
    return anchored


def _runs_cache_path(match_key, cache_dir=None):
    base = Path(cache_dir) if cache_dir is not None else cache.root()
    return base / 'identity' / 'runs' / f'{match_key}.json'


def _write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f'.{os.getpid()}.{threading.get_ident()}.tmp')
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.replace(path)


def _merge_trusted_tracks(existing, incoming, min_prior_conf=0.60):
    by_team = {}
    for source_list in (existing or (), incoming or ()):
        # Within a single run list, drop any team that appears twice on the same alliance
        counts = {}
        for tr in source_list:
            if _is_trusted_prior_track(tr, min_prior_conf=min_prior_conf):
                tk = _canon_team_key(tr.get('team_key'))
                counts[tk] = counts.get(tk, 0) + 1
        for tr in source_list:
            if not _is_trusted_prior_track(tr, min_prior_conf=min_prior_conf):
                continue
            tk = _canon_team_key(tr.get('team_key'))
            if counts.get(tk, 0) != 1:
                continue
            pts = _points_to_ms(_extract_raw_points(tr), offset_s=0.0, default_unit='auto')
            if len(pts) < 4:
                continue
            conf = float(tr.get('identity_confidence') or 0.0)
            entry = {
                'team_key': tk,
                'alliance': tr.get('alliance'),
                'identity_confidence': round(conf, 3),
                'identity_source': _track_source(tr) or 'reads',
                'swap_risks': [],
                'trajectory': [
                    {
                        't': pt['t_ms'],
                        'x': round(pt['x'], 3),
                        'y': round(pt['y'], 3),
                        **({'box': [round(v, 4) for v in pt['box']]} if pt.get('box') else {}),
                        'calibrated': pt.get('calibrated', True),
                    }
                    for pt in pts
                    if pt['t_ms'] <= 30000
                ] or [
                    {
                        't': pt['t_ms'],
                        'x': round(pt['x'], 3),
                        'y': round(pt['y'], 3),
                        **({'box': [round(v, 4) for v in pt['box']]} if pt.get('box') else {}),
                        'calibrated': pt.get('calibrated', True),
                    }
                    for pt in pts[:120]
                ],
            }
            prev = by_team.get(tk)
            if prev is None or conf >= float(prev.get('identity_confidence') or 0.0):
                by_team[tk] = entry
    return list(by_team.values())


def save_run_identity_cache(match_key, tracks, cache_dir=None):
    """Save trusted 'reads' tracks for match_key to the local identity cache."""
    if not match_key or (cache_dir is None and not cache.enabled()):
        return None
    path = _runs_cache_path(match_key, cache_dir=cache_dir)
    existing = []
    if path.exists():
        try:
            data = json.loads(path.read_text())
            existing = data.get('tracks', []) if isinstance(data, dict) else data
        except (OSError, ValueError):
            existing = []
    merged = _merge_trusted_tracks(existing, tracks)
    _write_json_atomic(path, {'match_key': match_key, 'tracks': merged})
    return path


def load_prior_run_tracks(match_key, cache_dir=None, allow_db=False, exclude_run_id=None):
    """Load trusted prior tracks for match_key from the local run cache (earlier runs of the same match)."""
    if not match_key:
        return []
    if cache_dir is not None or cache.enabled():
        path = _runs_cache_path(match_key, cache_dir=cache_dir)
        if path.exists():
            try:
                data = json.loads(path.read_text())
                tracks = data.get('tracks', []) if isinstance(data, dict) else data
                trusted = [t for t in tracks if _is_trusted_prior_track(t)]
                if trusted:
                    return trusted
            except (OSError, ValueError):
                pass
    return []


# ---------------------------------------------------------------------------
# Auto-path fingerprinting across matches of an event
# ---------------------------------------------------------------------------

def normalize_to_alliance(x, y, alliance):
    """Rotate field coordinates into the alliance-normalized frame (own wall at x=0, left at +y)."""
    fx, fy = float(x), float(y)
    if alliance == 'red':
        return float(FIELD['length']) - fx, float(FIELD['width']) - fy
    return fx, fy


_START_POSITION_ALIASES = {
    'hub': 'center',
    'center': 'center',
    'left_bump': 'left mound',
    'left mound': 'left mound',
    'right_bump': 'right mound',
    'right mound': 'right mound',
    'left_trench': 'left trench',
    'left trench': 'left trench',
    'right_trench': 'right trench',
    'right trench': 'right trench',
}


def _canon_start_position(pos):
    if not isinstance(pos, str):
        return None
    s = pos.strip().lower()
    return _START_POSITION_ALIASES.get(s, s)


def start_position_at(norm_y):
    """Partition normalized y in [0, FIELD['width']] into the 5 autonomous start lanes (matching fieldRebuilt.js)."""
    offset = float(norm_y) - float(FIELD['hub']['y'])
    mag = abs(offset)
    side = 'left' if offset >= 0.0 else 'right'
    half_hub = float(FIELD['hub'].get('halfSize', FIELD['bump'].get('yInner', 0.597)))
    if mag <= half_hub:
        return 'center'
    if mag <= float(FIELD['bump']['yOuter']):
        return f'{side} mound'
    return f'{side} trench'


def _xy_tuple(pt):
    if isinstance(pt, dict):
        return float(pt['x']), float(pt['y'])
    return float(pt[0]), float(pt[1])


def _polyline_length(points_xy):
    pts = [_xy_tuple(p) for p in (points_xy or ())]
    if len(pts) < 2:
        return 0.0
    return sum(math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]) for i in range(1, len(pts)))


def resample_by_arc(points_xy, n=32):
    """Resample a path to n points evenly spaced by cumulative Euclidean distance."""
    pts = [_xy_tuple(p) for p in (points_xy or ())]
    if n <= 0:
        return []
    if len(pts) < 2:
        return list(pts)
    if n == 1:
        return [pts[0]]
    cumulative = [0.0]
    for i in range(1, len(pts)):
        cumulative.append(cumulative[-1] + math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]))
    total = cumulative[-1]
    if total < 1e-6:
        return [pts[0] for _ in range(n)]
    out = []
    cursor = 0
    for k in range(n):
        target = (total * k) / (n - 1)
        while cursor < len(cumulative) - 2 and cumulative[cursor + 1] < target:
            cursor += 1
        span = (cumulative[cursor + 1] - cumulative[cursor]) or 1.0
        f = (target - cumulative[cursor]) / span
        x = pts[cursor][0] + (pts[cursor + 1][0] - pts[cursor][0]) * f
        y = pts[cursor][1] + (pts[cursor + 1][1] - pts[cursor][1]) * f
        out.append((float(x), float(y)))
    return out


def _resample_prefix_by_arc(points_xy, target_len_m, n=32):
    pts = [_xy_tuple(p) for p in (points_xy or ())]
    if not pts or target_len_m <= 0.0:
        return resample_by_arc(pts, n=n)
    prefix = [pts[0]]
    acc = 0.0
    for i in range(1, len(pts)):
        x0, y0 = pts[i - 1]
        x1, y1 = pts[i]
        seg = math.hypot(x1 - x0, y1 - y0)
        if seg <= 1e-9:
            continue
        if acc + seg >= target_len_m:
            f = (target_len_m - acc) / seg
            prefix.append((x0 + (x1 - x0) * f, y0 + (y1 - y0) * f))
            break
        acc += seg
        prefix.append((x1, y1))
    return resample_by_arc(prefix, n=n)


def _banded_dtw_dist(pts_a, pts_b, window=5):
    n, m = len(pts_a), len(pts_b)
    if not n or not m:
        return float('inf')
    w = max(int(window), abs(n - m))
    inf = float('inf')
    dp_cost = [[inf] * m for _ in range(n)]
    dp_steps = [[1] * m for _ in range(n)]
    for i in range(n):
        ax, ay = float(pts_a[i][0]), float(pts_a[i][1])
        j_lo, j_hi = max(0, i - w), min(m - 1, i + w)
        for j in range(j_lo, j_hi + 1):
            d = math.hypot(ax - float(pts_b[j][0]), ay - float(pts_b[j][1]))
            if i == 0 and j == 0:
                dp_cost[i][j] = d
                dp_steps[i][j] = 1
                continue
            best_prev_cost = inf
            best_prev_steps = 1
            for pi, pj in ((i - 1, j - 1), (i - 1, j), (i, j - 1)):
                if pi >= 0 and pj >= 0 and dp_cost[pi][pj] < best_prev_cost:
                    best_prev_cost = dp_cost[pi][pj]
                    best_prev_steps = dp_steps[pi][pj]
            if math.isfinite(best_prev_cost):
                dp_cost[i][j] = best_prev_cost + d
                dp_steps[i][j] = best_prev_steps + 1
    return dp_cost[n - 1][m - 1] / max(1, dp_steps[n - 1][m - 1])


def compute_auto_fingerprint(points, alliance='blue', offset_s=0.0, auto_end_ms=AUTO_END_MS):
    """Extract an alliance-normalized auto path fingerprint over [0, auto_end_ms]."""
    raw_pts = _extract_raw_points(points) if isinstance(points, dict) else (points or [])
    if isinstance(points, dict) and 'alliance' in points and alliance == 'blue':
        alliance = points.get('alliance') or alliance
    pts_ms = _points_to_ms(raw_pts, offset_s=offset_s, default_unit='auto')
    if not pts_ms or any(not p.get('calibrated', True) for p in pts_ms):
        return None
    in_window = [p for p in pts_ms if -250 <= p['t_ms'] <= int(auto_end_ms)]
    if len(in_window) < 5:
        return None

    kept = []
    for p in in_window:
        nx, ny = normalize_to_alliance(p['x'], p['y'], alliance)
        t_ms = max(0, int(p['t_ms']))
        if kept:
            dt = (t_ms - kept[-1]['t_ms']) / 1000.0
            if dt <= 0.0:
                continue
            speed = math.hypot(nx - kept[-1]['x'], ny - kept[-1]['y']) / dt
            if speed > 1.5 * VMAX and dt < 1.0:
                continue
        kept.append({'t_ms': t_ms, 'x': nx, 'y': ny})

    if len(kept) < 5:
        return None
    duration_ms = kept[-1]['t_ms'] - kept[0]['t_ms']
    if duration_ms < 1500 or kept[0]['t_ms'] > 3500:
        return None

    distance_m = 0.0
    for i in range(1, len(kept)):
        if kept[i]['t_ms'] - kept[i - 1]['t_ms'] <= 1500:
            distance_m += math.hypot(kept[i]['x'] - kept[i - 1]['x'], kept[i]['y'] - kept[i - 1]['y'])
    max_disp_m = max(math.hypot(p['x'] - kept[0]['x'], p['y'] - kept[0]['y']) for p in kept)

    xy = [(p['x'], p['y']) for p in kept]
    arc_points = [(round(x, 4), round(y, 4)) for x, y in resample_by_arc(xy, n=32)]

    # Interpolate at 1s steps (0, 1000, ..., auto_end_ms) across gaps <= 1500 ms
    time_points = {}
    cursor = 0
    for target_ms in range(0, int(auto_end_ms) + 1, 1000):
        if target_ms < kept[0]['t_ms'] or target_ms > kept[-1]['t_ms']:
            continue
        while cursor < len(kept) - 2 and kept[cursor + 1]['t_ms'] < target_ms:
            cursor += 1
        a, b = kept[cursor], kept[min(len(kept) - 1, cursor + 1)]
        span = b['t_ms'] - a['t_ms']
        if span > 1500:
            continue
        f = 0.0 if span <= 0 else (target_ms - a['t_ms']) / span
        if 0.0 <= f <= 1.0:
            ix = a['x'] + (b['x'] - a['x']) * f
            iy = a['y'] + (b['y'] - a['y']) * f
            time_points[str(target_ms)] = (round(ix, 4), round(iy, 4))

    start_xy = (round(kept[0]['x'], 4), round(kept[0]['y'], 4))
    finish_xy = (round(kept[-1]['x'], 4), round(kept[-1]['y'], 4))
    return {
        'start_xy': start_xy,
        'finish_xy': finish_xy,
        'start_position': start_position_at(start_xy[1]),
        'distance_m': round(distance_m, 4),
        'dead_auto': bool(distance_m < 0.75 or max_disp_m < 0.55),
        'arc_points': arc_points,
        'time_points': time_points,
        'duration_ms': duration_ms,
    }


def _mirror_fingerprint_y(fp):
    w = float(FIELD['width'])
    sa = fp.get('start_xy')
    fa = fp.get('finish_xy')
    arc = [(float(p[0]), round(w - float(p[1]), 4)) for p in (fp.get('arc_points') or ())]
    tp = {k: (float(v[0]), round(w - float(v[1]), 4)) for k, v in (fp.get('time_points') or {}).items()}
    start_xy = (float(sa[0]), round(w - float(sa[1]), 4)) if sa else None
    return {
        **fp,
        'start_xy': start_xy,
        'finish_xy': (float(fa[0]), round(w - float(fa[1]), 4)) if fa else None,
        'start_position': start_position_at(start_xy[1]) if start_xy else None,
        'arc_points': arc,
        'time_points': tp,
    }


def _auto_fingerprint_similarity_oriented(fp_a, fp_b, penalty_m=0.0):
    sa, sb = fp_a.get('start_xy'), fp_b.get('start_xy')
    arc_a, arc_b = fp_a.get('arc_points') or [], fp_b.get('arc_points') or []
    n = len(arc_a)
    start_dist_m = math.hypot(float(sa[0]) - float(sb[0]), float(sa[1]) - float(sb[1]))
    uniform_arc_dist_m = sum(math.hypot(float(pa[0]) - float(pb[0]), float(pa[1]) - float(pb[1])) for pa, pb in zip(arc_a, arc_b)) / n
    dtw_dist_m = _banded_dtw_dist(arc_a, arc_b, window=5)

    poly_a = _polyline_length(arc_a)
    poly_b = _polyline_length(arc_b)
    len_a = float(fp_a.get('distance_m') or poly_a)
    len_b = float(fp_b.get('distance_m') or poly_b)
    min_len = min(len_a, len_b)
    max_len = max(len_a, len_b)
    prefix_arc_dist_m = float('inf')
    is_prefix = False
    if min_len >= 3.0 and max_len > 1e-6 and (min_len / max_len) <= 0.85:
        is_prefix = True
        min_poly = min(poly_a, poly_b)
        if poly_a <= poly_b:
            sub_b = _resample_prefix_by_arc(arc_b, min_poly, n=n)
            u_pre = sum(math.hypot(float(pa[0]) - float(pb[0]), float(pa[1]) - float(pb[1])) for pa, pb in zip(arc_a, sub_b)) / n
            d_pre = _banded_dtw_dist(arc_a, sub_b, window=5)
            prefix_arc_dist_m = min(u_pre, d_pre)
        else:
            sub_a = _resample_prefix_by_arc(arc_a, min_poly, n=n)
            u_pre = sum(math.hypot(float(pa[0]) - float(pb[0]), float(pa[1]) - float(pb[1])) for pa, pb in zip(sub_a, arc_b)) / n
            d_pre = _banded_dtw_dist(sub_a, arc_b, window=5)
            prefix_arc_dist_m = min(u_pre, d_pre)

    arc_dist_m = min(uniform_arc_dist_m, dtw_dist_m, prefix_arc_dist_m)

    tp_a = fp_a.get('time_points') or {}
    tp_b = fp_b.get('time_points') or {}
    shared_t = sorted(set(tp_a.keys()).intersection(tp_b.keys()), key=lambda k: int(k))
    if shared_t:
        if is_prefix and prefix_arc_dist_m <= min(uniform_arc_dist_m, dtw_dist_m):
            cutoff_ms = max(3000, int(round(float(AUTO_END_MS) * (min_len / max_len))))
            active_t = [k for k in shared_t if int(k) <= cutoff_ms] or shared_t[:max(1, len(shared_t) // 2)]
            time_dist_m = sum(math.hypot(float(tp_a[k][0]) - float(tp_b[k][0]), float(tp_a[k][1]) - float(tp_b[k][1])) for k in active_t) / len(active_t)
        else:
            time_dist_m = sum(math.hypot(float(tp_a[k][0]) - float(tp_b[k][0]), float(tp_a[k][1]) - float(tp_b[k][1])) for k in shared_t) / len(shared_t)
    else:
        time_dist_m = arc_dist_m

    combined_dist = 0.55 * arc_dist_m + 0.25 * time_dist_m + 0.20 * start_dist_m + penalty_m
    similarity = math.exp(-combined_dist / 1.65)
    return {
        'similarity': float(similarity),
        'start_dist_m': float(start_dist_m),
        'arc_dist_m': float(arc_dist_m),
        'time_dist_m': float(time_dist_m),
    }


def auto_fingerprint_similarity(fp_a, fp_b, allow_mirror=True):
    """Compare two auto-path fingerprints; stationary (dead_auto) paths always return 0 similarity."""
    if not isinstance(fp_a, dict) or not isinstance(fp_b, dict):
        return {'similarity': 0.0, 'start_dist_m': float('inf'), 'arc_dist_m': float('inf'), 'time_dist_m': float('inf')}
    if fp_a.get('dead_auto') or fp_b.get('dead_auto'):
        return {'similarity': 0.0, 'start_dist_m': float('inf'), 'arc_dist_m': float('inf'), 'time_dist_m': float('inf')}
    sa, sb = fp_a.get('start_xy'), fp_b.get('start_xy')
    arc_a, arc_b = fp_a.get('arc_points') or [], fp_b.get('arc_points') or []
    if not sa or not sb or len(arc_a) != len(arc_b) or not arc_a:
        return {'similarity': 0.0, 'start_dist_m': float('inf'), 'arc_dist_m': float('inf'), 'time_dist_m': float('inf')}

    direct = _auto_fingerprint_similarity_oriented(fp_a, fp_b, penalty_m=0.0)
    if not allow_mirror:
        return direct
    mirrored = _auto_fingerprint_similarity_oriented(fp_a, _mirror_fingerprint_y(fp_b), penalty_m=0.15)
    return mirrored if mirrored['similarity'] > direct['similarity'] else direct


class EventAutoFingerprintLibrary:
    """Per-event library of trusted autonomous path fingerprints."""

    def __init__(self, event_key='default', cache_dir=None, entries=None):
        self.event_key = str(event_key or 'default').split('_')[0]
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.entries = list(entries or [])

    def default_path(self):
        base = self.cache_dir if self.cache_dir is not None else cache.root()
        return base / 'identity' / f'{self.event_key}_auto_fingerprints.json'

    def add_track(self, match_key, track, offset_s=0.0, min_identity_conf=0.75):
        if not isinstance(track, dict) or not _is_trusted_prior_track(track, min_prior_conf=min_identity_conf):
            return False
        team_key = _canon_team_key(track.get('team_key'))
        alliance = track.get('alliance') or 'blue'
        fp = track.get('fingerprint') if isinstance(track.get('fingerprint'), dict) else compute_auto_fingerprint(
            _extract_raw_points(track), alliance, offset_s=offset_s
        )
        if not fp or fp.get('dead_auto'):
            return False
        conf = float(track.get('identity_confidence') or 0.0)
        entry = {
            'match_key': str(match_key or ''),
            'team_key': team_key,
            'alliance': alliance,
            'identity_confidence': round(conf, 3),
            'fingerprint': fp,
        }
        for idx, prev in enumerate(self.entries):
            if prev.get('match_key') == entry['match_key'] and prev.get('team_key') == team_key:
                if conf >= float(prev.get('identity_confidence') or 0.0):
                    self.entries[idx] = entry
                return True
        self.entries.append(entry)
        return True

    def match_chain(
        self,
        fp,
        candidate_teams,
        exclude_match_key=None,
        max_start_dist_m=1.2,
        max_arc_dist_m=0.90,
        min_similarity=0.55,
        min_margin=0.12,
    ):
        if isinstance(fp, list):
            fp = compute_auto_fingerprint(fp, 'blue')
        if not isinstance(fp, dict) or fp.get('dead_auto'):
            return {'source': 'auto_fingerprint', 'team': None, 'confidence': 0.0, 'margin': 0.0}

        canon_to_cand = {}
        for t in candidate_teams or ():
            ck = _canon_team_key(t)
            if ck:
                canon_to_cand[ck] = str(t)
        if not canon_to_cand:
            return {'source': 'auto_fingerprint', 'team': None, 'confidence': 0.0, 'margin': 0.0}

        last_margin = 0.0
        for allow_mirror in (False, True):
            best_by_team = {}
            for entry in self.entries:
                if exclude_match_key and entry.get('match_key') == exclude_match_key:
                    continue
                tk = _canon_team_key(entry.get('team_key'))
                if tk not in canon_to_cand:
                    continue
                cmp_res = auto_fingerprint_similarity(fp, entry.get('fingerprint'), allow_mirror=allow_mirror)
                sim = cmp_res['similarity']
                prev = best_by_team.get(tk)
                if prev is None or sim > prev['similarity']:
                    best_by_team[tk] = {**cmp_res, 'match_key': entry.get('match_key')}

            if not best_by_team:
                return {'source': 'auto_fingerprint', 'team': None, 'confidence': 0.0, 'margin': 0.0}

            ranked = sorted(best_by_team.items(), key=lambda kv: kv[1]['similarity'], reverse=True)
            best_tk, best_info = ranked[0]
            runner_up_sim = ranked[1][1]['similarity'] if len(ranked) > 1 else 0.0
            margin = best_info['similarity'] - runner_up_sim
            last_margin = margin

            if (
                best_info['start_dist_m'] <= max_start_dist_m
                and best_info['arc_dist_m'] <= max_arc_dist_m
                and best_info['similarity'] >= min_similarity
                and margin >= min_margin
            ):
                return {
                    'source': 'auto_fingerprint',
                    'team': canon_to_cand[best_tk],
                    'confidence': round(best_info['similarity'], 3),
                    'similarity': round(best_info['similarity'], 3),
                    'arc_dist_m': round(best_info['arc_dist_m'], 3),
                    'start_dist_m': round(best_info['start_dist_m'], 3),
                    'margin': round(margin, 3),
                    'matched_match_key': best_info['match_key'],
                }
        return {'source': 'auto_fingerprint', 'team': None, 'confidence': 0.0, 'margin': round(max(0.0, last_margin), 3)}

    def save(self, path=None):
        target = Path(path) if path is not None else self.default_path()
        _write_json_atomic(target, {'event_key': self.event_key, 'entries': self.entries})
        return target

    @classmethod
    def load(cls, path_or_event_key, cache_dir=None):
        if isinstance(path_or_event_key, Path) or (isinstance(path_or_event_key, str) and path_or_event_key.endswith('.json')):
            path = Path(path_or_event_key)
            event_key = path.stem.replace('_auto_fingerprints', '')
        else:
            event_key = str(path_or_event_key or 'default').split('_')[0]
            base = Path(cache_dir) if cache_dir is not None else cache.root()
            path = base / 'identity' / f'{event_key}_auto_fingerprints.json'
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            entries = data.get('entries', []) if isinstance(data, dict) else data
            return cls(event_key=data.get('event_key', event_key) if isinstance(data, dict) else event_key, cache_dir=cache_dir, entries=entries)
        except (OSError, ValueError):
            return None


def collect_chain_priors(
    chains,
    tracklets,
    roster_alliance,
    alliance,
    offset_s=0.0,
    *,
    match_key=None,
    prior_tracks=None,
    appearance_classifier=None,
    fingerprint_lib=None,
):
    """Gather prior_run, appearance, and auto_fingerprint evidence items for each chain."""
    by_chain = {i: [] for i in range(len(chains))}
    if not chains:
        return by_chain

    event_key = str(match_key).split('_')[0] if match_key and '_' in str(match_key) else (str(match_key) if match_key else None)
    if prior_tracks is None and match_key and cache.enabled():
        prior_tracks = load_prior_run_tracks(match_key)
    if fingerprint_lib is None and event_key and cache.enabled():
        fingerprint_lib = EventAutoFingerprintLibrary.load(event_key)
    if appearance_classifier is None and event_key and cache.enabled():
        try:
            from appearance_classifier import EventAppearanceClassifier
            appearance_classifier = EventAppearanceClassifier.load(event_key)
        except Exception:
            appearance_classifier = None

    if prior_tracks:
        for item in match_prior_run_identities(chains, tracklets, prior_tracks, alliance, offset_s=offset_s):
            c_idx = item.get('chain_index')
            if c_idx in by_chain:
                matched_team = _match_candidate_team(item.get('team'), roster_alliance) or item.get('team')
                already_anchored = False
                for t_idx in chains[c_idx] or ():
                    if not (0 <= t_idx < len(tracklets)):
                        continue
                    tr = tracklets[t_idx]
                    ev_list = (tr.get('evidence') if isinstance(tr, dict) else getattr(tr, 'evidence', None)) or ()
                    for ev in ev_list:
                        if isinstance(ev, dict) and ev.get('source') == 'prior_run' and _canon_team_key(ev.get('team')) == _canon_team_key(matched_team):
                            ev['team'] = matched_team
                            ev['confidence'] = max(float(ev.get('confidence') or 0.0), float(item.get('confidence') or 0.0))
                            already_anchored = True
                if not already_anchored:
                    by_chain[c_idx].append({
                        **item,
                        'team': matched_team,
                    })

    if fingerprint_lib is not None and hasattr(fingerprint_lib, 'match_chain'):
        for c_idx, chain in enumerate(chains):
            if not chain:
                continue
            c_pts = _reconstruct_chain_points(chain, tracklets, offset_s=offset_s)
            fp = compute_auto_fingerprint(c_pts, alliance, offset_s=0.0)
            if fp and not fp.get('dead_auto'):
                pred = fingerprint_lib.match_chain(fp, roster_alliance, exclude_match_key=match_key)
                if pred and pred.get('team') and float(pred.get('confidence') or 0.0) > 0.0:
                    by_chain[c_idx].append(pred)

    if appearance_classifier is not None and hasattr(appearance_classifier, 'predict_tracklet'):
        for c_idx, chain in enumerate(chains):
            if not chain:
                continue
            crops = []
            for t_idx in chain:
                tr = tracklets[t_idx]
                tr_crops = tr.get('crops') if isinstance(tr, dict) else getattr(tr, 'crops', None)
                if tr_crops:
                    crops.extend(tr_crops)
            if not crops:
                continue
            pred = appearance_classifier.predict_tracklet(crops, roster_alliance, exclude_match_key=match_key)
            if pred and pred.get('team') and float(pred.get('confidence') or 0.0) > 0.0:
                by_chain[c_idx].append({
                    'source': 'appearance',
                    'team': _match_candidate_team(pred.get('team'), roster_alliance) or pred.get('team'),
                    'confidence': float(pred['confidence']),
                    'margin': pred.get('margin'),
                })

    return by_chain


def reconcile_tracks_with_priors(
    tracks,
    prior_tracks,
    roster,
    *,
    match_key=None,
    offset_s=0.0,
    min_prior_conf=0.60,
    appearance_classifier=None,
    fingerprint_lib=None,
):
    """Reconcile already-built tracks (e.g. a scouting run) against prior_tracks (e.g. a calibrated auto run) and the roster."""
    from copy import deepcopy
    from stitch import INDIRECT_PRIOR_SOURCES, solve_chain_identities

    out = deepcopy(list(tracks or []))
    if not out or not isinstance(roster, dict):
        return out

    for alliance in ('red', 'blue'):
        a_indices = [idx for idx, tr in enumerate(out) if isinstance(tr, dict) and tr.get('alliance') == alliance]
        a_roster = roster.get(alliance) or []
        if not a_indices or not a_roster:
            continue
        chains = [[i] for i in range(len(a_indices))]
        t_dicts = []
        for idx in a_indices:
            tr = out[idx]
            pts = tr.get('trajectory') or []
            tk = tr.get('team_key')
            conf = float(tr.get('identity_confidence') or 0.0)
            src = _track_source(tr)
            ev = []
            votes = {}
            if tk and conf > 0.0:
                votes[tk] = conf
                ev_src = 'wide_frame' if (src in ('weak_reads', 'reads', None) and len(pts) > 100) else (src or 'reads')
                ev.append({'source': ev_src, 'team': tk, 'confidence': conf})
            t_dicts.append({
                'start': (pts[0]['t'] / 1000.0) if pts else 0.0,
                'end': (pts[-1]['t'] / 1000.0) if pts else 0.0,
                'points': pts,
                'votes': votes,
                'evidence': ev,
                'crops': tr.get('crops'),
            })
        priors = collect_chain_priors(
            chains,
            t_dicts,
            a_roster,
            alliance,
            offset_s=offset_s,
            match_key=match_key,
            prior_tracks=prior_tracks,
            appearance_classifier=appearance_classifier,
            fingerprint_lib=fingerprint_lib,
        )
        assigned, completed, summaries = solve_chain_identities(chains, t_dicts, a_roster, chain_priors=priors)
        for local_i, idx in enumerate(a_indices):
            tr = out[idx]
            a_team, a_conf = assigned[local_i]
            c_team, c_conf, c_src = completed[local_i]
            ev_sum = summaries[local_i]
            team = a_team or c_team
            conf = a_conf if a_team else c_conf
            src = (ev_sum.get('primary_source') or 'reads') if a_team else c_src
            team_key = _canon_team_key(team) if team else None
            tr['team_key'] = team_key
            tr['identity_confidence'] = round(float(conf or 0.0), 2)
            tr['identity_source'] = src
            tr['identity_evidence'] = ev_sum
            if src in INDIRECT_PRIOR_SOURCES or src in ('weak_reads', 'elimination') or float(conf or 0.0) < 0.60:
                tr['needs_review'] = True
            if isinstance(tr.get('metrics'), dict):
                tr['metrics']['identity_source'] = src
                tr['metrics']['identity_evidence'] = ev_sum
    return out
