"""Counter v2: alliance totals from per-period track statistics at the HUB openings (video only).

The track counter misses a match-dependent share of the balls (the lit HUB washes them out, streams merge). A single
factor per HUB corrects the average miss but not the per-match one. The tracks carry evidence of the misses: tracks
lost just above an opening, rescued fragments, predicted entries, and balls released below an opening (scored balls
leave the HUB), so a linear model on those statistics, per HUB and scoring period, estimates the balls TBA scores.

The model is fitted on TBA's per-period counts of the 2026cc matches (offline, after inference) and stored with its
provenance in `models/counter_v2_period_model.json`; `eval_counter_event.py cv-model` reports its nested
leave-one-match-out accuracy. Inference uses video-derived statistics only.
"""
import json
from pathlib import Path

import cv2
import numpy as np

import fuel_count as fc

FEATURES = ('entries', 'predicted', 'stitched', 'near_tracks', 'born_below', 'seen_leaving', 'not_down', 'lost_above', 'near_obs')
PERIODS = ('auto', 'transition', 'shift1', 'shift2', 'shift3', 'shift4', 'endgame')
MODEL_PATH = Path(__file__).resolve().parent / 'models' / 'counter_v2_period_model.json'
NEAR_SIDE_PX = 15.0
NEAR_ABOVE_PX = 60.0


def period_of(t_ms):
    if t_ms < fc.TELEOP_START_MS:
        return 'auto'
    if t_ms < fc.TRANSITION_END_MS:
        return 'transition'
    if t_ms < fc.ENDGAME_START_MS:
        return f'shift{1 + int((t_ms - fc.TRANSITION_END_MS) // fc.SHIFT_MS)}'
    return 'endgame'


def hub_active(alliance, period, shift1_side):
    if period in ('auto', 'transition', 'endgame'):
        return True
    if shift1_side not in ('red', 'blue'):
        return None
    first = alliance == shift1_side
    return first if period in ('shift1', 'shift3') else not first


def opening_features(tracks, openings, offset_s, entries):
    """{alliance: {period: {feature: n}}}: counter entries and what happened to the other tracks near each opening."""
    feats = {c: {p: dict.fromkeys(FEATURES, 0) for p in PERIODS} for c in ('red', 'blue')}
    for e in entries:
        if 0 <= e['t_ms'] <= fc.MATCH_MS:
            f = feats[e['alliance']][period_of(e['t_ms'])]
            f['entries'] += 1
            f['predicted'] += bool(e.get('predicted'))
            f['stitched'] += bool(e.get('stitched'))
    counted = {e['track_id'] for e in entries if not e.get('stitched')}
    for name, poly in openings.items():
        alliance = fc.HUBS[name]
        p = poly.reshape(-1, 2)
        x0, x1 = p[:, 0].min() - NEAR_SIDE_PX, p[:, 0].max() + NEAR_SIDE_PX
        top, bottom = p[:, 1].min(), p[:, 1].max()
        for tid, track in enumerate(tracks):
            obs = fc._obs(track)
            near = [o for o in obs if x0 <= o[5] <= x1 and top - NEAR_ABOVE_PX <= o[6] <= bottom]
            if not near:
                continue
            t_ms = int((near[0][1] - offset_s) * 1000)
            if not 0 <= t_ms <= fc.MATCH_MS:
                continue
            f = feats[alliance][period_of(t_ms)]
            f['near_tracks'] += 1
            f['near_obs'] += len(near)
            if tid in counted:
                continue
            d = [cv2.pointPolygonTest(poly, (float(o[5]), float(o[6])), True) for o in obs]
            inside = [x >= 0 for x in d]
            if float(obs[0][6]) > bottom:
                f['born_below'] += 1
            elif any(inside) and any(x < -fc.EXIT_TOL_PX for x in d[inside.index(True) + 1:]):
                f['seen_leaving'] += 1
            elif float(obs[-1][6]) - float(obs[0][6]) < fc.ENTRY_DROP_PX:
                f['not_down'] += 1
            elif 0 < top - float(obs[-1][6]) <= fc.LOST_BAND_PX:
                f['lost_above'] += 1
    return feats


def fit(rows_by_hub, alpha=0.0):
    """{hub color: {'mu', 'sd', 'w'}} from {color: [(feature dict, TBA count)]}: standardized least squares."""
    model = {}
    for color, rows in rows_by_hub.items():
        X = np.array([[f[k] for k in FEATURES] for f, _ in rows], float)
        y = np.array([t for _, t in rows], float)
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Z = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
        R = alpha * np.eye(Z.shape[1])
        R[-1, -1] = 0
        w = np.linalg.lstsq(Z.T @ Z + R, Z.T @ y, rcond=None)[0]   # least squares: constant features get weight 0
        model[color] = {'mu': mu.tolist(), 'sd': sd.tolist(), 'w': w.tolist()}
    return model


RATIO_FEATURES = ('predicted', 'stitched', 'lost_above', 'born_below', 'seen_leaving', 'not_down', 'near_tracks')


def ratio_vector(f):
    """[1, feature / entries ...]: camera-independent shares of each statistic per counted entry."""
    e = max(float(f['entries']), 1.0)
    return [1.0] + [float(f[k]) / e for k in RATIO_FEATURES]


def two_ring_vector(base, upper):
    """ratio_vector of the base ring (1.233 m) plus, per base entry, the true-opening ring's (1.83 m) entries and statistics."""
    e = max(float(base['entries']), 1.0)
    return ratio_vector(base) + [float(upper['entries']) / e] + [float(upper[k]) / e for k in RATIO_FEATURES]


def fit_two_ring(rows):
    """rows: [(base features, upper features, TBA balls)] -> weights: balls = base entries x (w . two_ring_vector)."""
    X = np.array([two_ring_vector(b, u) for b, u, _ in rows])
    e = np.array([float(b['entries']) for b, _, _ in rows])
    y = np.array([float(t) for _, _, t in rows])
    A = X * e[:, None]
    return np.linalg.lstsq(A, y, rcond=None)[0].tolist()


def fit_scale(rows):
    """rows: [(features, TBA balls)] -> k: balls = k x entries, least squares on balls. With the chimney entry region
    one factor transfers across cameras: fitted without each event it was 0.714-0.738 on all five (2026cc, 2026arc,
    2026caclv, 2026cur, 2026gal), where the ring-only ratio and two-ring models missed 2026cur by 47-61%."""
    e = np.array([float(f['entries']) for f, _ in rows])
    y = np.array([float(t) for _, t in rows])
    return float(np.dot(e, y) / max(float(np.dot(e, e)), 1.0))


def fit_ratio(rows):
    """One multiplier model for both HUBs: balls = entries x (w . ratio_vector), least squares on balls. Fitted on
    shares rather than raw counts, so it transfers across cameras (2026arc 17.2% / 2026caclv 19.3% WAPE against 56% /
    23% for the raw-count model, with the same 7.5% leave-one-match-out on 2026cc)."""
    X = np.array([ratio_vector(f) for f, _ in rows])
    e = np.array([float(f['entries']) for f, _ in rows])
    y = np.array([float(t) for _, t in rows])
    A = X * e[:, None]
    return np.linalg.lstsq(A, y, rcond=None)[0].tolist()


def predict_period(model, color, f, upper=None):
    seen = f['entries'] > 0 or f['near_tracks'] > 0 or (upper is not None and (upper['entries'] > 0 or upper['near_tracks'] > 0))
    if not seen:
        return 0.0                                     # no ball seen near the opening: a model's intercept alone
                                                       # would invent balls (all-zero features: red 42, blue 16)
    if model.get('type') == 'scale':
        return max(0.0, float(f['entries']) * float(model['k']))
    if model.get('type') == 'two_ring':
        return max(0.0, float(f['entries']) * float(np.dot(two_ring_vector(f, upper), model['w'])))
    if model.get('type') == 'ratio':
        return max(0.0, float(f['entries']) * float(np.dot(ratio_vector(f), model['w'])))
    m = model[color]
    z = (np.array([f[k] for k in FEATURES], float) - np.array(m['mu'])) / np.array(m['sd'])
    return max(0.0, float(np.dot(np.append(z, 1.0), m['w'])))


def estimate_active(model, feats, shift1_side, upper_feats=None):
    """{alliance: active-HUB balls | None}: the model's estimate summed over each alliance's active periods.
    A two-ring model also takes the true-opening ring's features (`upper_feats`)."""
    out = {}
    for color in ('red', 'blue'):
        total, unknown = 0.0, False
        for period in PERIODS:
            act = hub_active(color, period, shift1_side)
            if act is None:
                unknown = True
            elif act:
                if model.get('type') == 'two_ring':
                    total += predict_period(model, color, feats[color][period], upper_feats[color][period])
                else:
                    total += predict_period(model if model.get('type') in ('ratio', 'scale') else model['hubs'], color, feats[color][period])
        out[color] = None if unknown else int(round(total))
    return out


def load(path=MODEL_PATH):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def apply_estimate(payload, tracks, openings, offset_s, entries, shift1_side, model=None, upper_openings=None):
    """Counter v2: replace each alliance's `active_scored` with the model's estimate (raw count kept as
    `active_counted`; features and model recorded in stats). Without a model file it falls back to the per-HUB
    factors (`fuel_count.calibrate_payload`). v1.0.x counters are left alone."""
    if fc.counter_mode() != fc.COUNTER_V2:
        return payload
    model = model if model is not None else load()
    if not model or not openings or len(openings) != 2:
        return fc.calibrate_payload(payload)
    if model.get('entry_region', 'ring') != fc.entry_region():
        return fc.calibrate_payload(payload)               # the model was fitted on entries through another region
    feats = opening_features(tracks, openings, offset_s, entries)
    upper = None
    if model.get('type') == 'two_ring':
        upper_entries = fc.hub_entries_strict(tracks, upper_openings, offset_s) if upper_openings else []
        upper = opening_features(tracks, upper_openings, offset_s, upper_entries) if upper_openings else None
        if upper is None:
            return fc.calibrate_payload(payload)
    est = estimate_active(model, feats, shift1_side, upper)
    for color, alliance in (payload.get('alliances') or {}).items():
        alliance['active_counted'] = alliance.get('active_scored')
        alliance['active_scored'] = est.get(color)
    stats = payload.setdefault('stats', {})
    stats['calibration'] = {'method': 'period model', 'model': model.get('version'), 'fitted_on': model.get('fitted_on'),
                            'cv_wape_pct': model.get('cv_wape_pct')}
    stats['opening_features'] = feats
    if upper is not None:
        stats['opening_features_true_opening'] = upper
    return fc.scale_robot_credits(payload)
