"""Infer the last alliance identity only for six distinct source-assisted bodies.

Uses the match roster, existing identities and simultaneous image boxes only.
Does not consume scores. Returns copies; never approves tracks or reviews.
"""
from copy import deepcopy
from math import hypot, isfinite

from identity import bumper_occluded


def _box_iou(a, b):
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = lambda box: (box[2] - box[0]) * (box[3] - box[1])
    return overlap / max(1e-12, area(a) + area(b) - overlap)


def _canon_team_key(t):
    if not isinstance(t, str):
        return None
    s = t.strip()
    if s.startswith('frc') and s[3:].isdigit():
        return s
    return f'frc{s}' if s.isdigit() else None


def _timeline(track, require_frame_index=True):
    rows = {}
    previous = None
    for point in sorted(track.get('trajectory', []), key=lambda p: p.get('t', -1)):
        if not isinstance(point, dict) or point.get('estimated') is True:
            continue
        if not all(isinstance(point.get(k), (int, float)) and not isinstance(point.get(k), bool) and isfinite(point[k]) for k in ['t', 'x', 'y']):
            return None
        if previous:
            dt = point['t'] - previous['t']
            if 0 <= dt <= 650 and hypot(point['x'] - previous['x'], point['y'] - previous['y']) > 8 * dt / 1000 + .4:
                return None  # Implausible within-segment jump; possible identity switch.
        previous = point
        box = point.get('box', [])
        frame = point.get('frame_index')
        if frame is None and not require_frame_index:
            frame = int(round(point['t'] / 20.0))
        if isinstance(frame, int) and not isinstance(frame, bool) and frame >= 0 and len(box) == 4 and all(isinstance(v, (int, float)) and not isinstance(v, bool) and isfinite(v) and 0 <= v <= 1 for v in box) and box[2] > box[0] and box[3] > box[1]:
            if frame in rows:
                return None
            rows[frame] = point
    return rows


def _has_duplicate_boxes(tl_a, tl_b):
    shared = [f for f in set(tl_a).intersection(tl_b) if abs(tl_a[f]['t'] - tl_b[f]['t']) <= 20]
    if not shared:
        return False
    dups = sum(1 for f in shared if _box_iou(tl_a[f]['box'], tl_b[f]['box']) > .5)
    return dups > max(1, len(shared) * .1)


def infer_roster_identities(tracks, roster):
    """Conservatively infer one unknown per alliance; return tracks + audit rows."""
    result = deepcopy(tracks)
    audit = []
    if len(result) != 6 or not isinstance(roster, dict):
        return result, audit
    if any(not isinstance(roster.get(a), list) or len(roster[a]) != 3 for a in ['red', 'blue']):
        return result, audit
    canon_roster = {a: [_canon_team_key(t) for t in roster[a]] for a in ['red', 'blue']}
    teams = [team for alliance in ['red', 'blue'] for team in canon_roster[alliance]]
    if any(t is None for t in teams) or len(set(teams)) != 6:
        return result, audit
    has_metrics = any('metrics' in t for t in result)
    source_keys = []
    for track in result:
        if track.get('tracking_confidence', 0) < .6:
            return result, audit
        if has_metrics:
            if track.get('swap_risks'):
                return result, audit
            m = track.get('metrics') or {}
            keys = m.get('source_track_keys') or []
            if not m.get('physical_body') or not m.get('source_assisted') or not keys or any(m.get(k) for k in ['identity_conflict', 'identity_collision', 'possible_identity_switch', 'possible_duplicate', 'swap_risks']):
                return result, audit
            source_keys.extend(keys)
    if has_metrics and len(set(source_keys)) != len(source_keys):
        return result, audit
    all_timelines = [_timeline(t, require_frame_index=has_metrics) for t in result]
    if any(t is None for t in all_timelines):
        return result, audit
    min_known_conf = .8 if has_metrics else .6
    for alliance in ['red', 'blue']:
        indices = [idx for idx, t in enumerate(result) if t.get('alliance') == alliance]
        members = [result[idx] for idx in indices]
        if len(members) != 3 or len(canon_roster.get(alliance, [])) != 3:
            continue
        known = [t for t in members if t.get('team_key')]
        unknown = [t for t in members if not t.get('team_key')]
        if len(known) != 2 or len(unknown) != 1 or len({t['team_key'] for t in known}) != 2:
            continue
        unknown_local_idx = next(i for i, t in enumerate(members) if not t.get('team_key'))
        if unknown[0].get('swap_risks') or any(not isinstance(r, dict) or r.get('with') == unknown_local_idx for t in known for r in (t.get('swap_risks') or [])):
            continue
        if any(t['team_key'] not in canon_roster[alliance] or t.get('identity_confidence', 0) < min_known_conf or (t.get('identity_source') or (t.get('metrics') or {}).get('identity_source')) in ('roster_inferred', 'weak_reads', 'elimination', 'appearance', 'auto_fingerprint') for t in known):
            continue
        timelines = [all_timelines[idx] for idx in indices]
        unknown_idx = next(idx for idx in indices if not result[idx].get('team_key'))
        if any(_has_duplicate_boxes(timelines[i], timelines[j]) for i, j in [(0, 1), (0, 2), (1, 2)]):
            continue
        if any(_has_duplicate_boxes(all_timelines[unknown_idx], all_timelines[other]) for other in range(len(result)) if other != unknown_idx):
            continue
        common = set(timelines[0]).intersection(*[set(t) for t in timelines[1:]])
        separate = []
        duplicates = 0
        simul = 0
        for frame in sorted(common):
            points = [t[frame] for t in timelines]
            if max(p['t'] for p in points) - min(p['t'] for p in points) > 20:
                continue
            simul += 1
            overlaps = [_box_iou(points[i]['box'], points[j]['box']) for i, j in [(0, 1), (0, 2), (1, 2)]]
            if max(overlaps) > .5:
                duplicates += 1
            if max(overlaps) <= .1 and not any(bumper_occluded(points[i]['box'], [points[j]['box'] for j in range(3) if j != i]) for i in range(3)):
                separate.append(points[0]['t'])
        if len(separate) < 5 or max(separate) - min(separate) < 500 or duplicates > max(1, simul * .1):
            continue
        missing = set(canon_roster[alliance]) - {t['team_key'] for t in known}
        if len(missing) != 1:
            continue
        team = missing.pop()
        inferred = unknown[0]
        proof = {'method': 'roster_elimination', 'alliance': alliance, 'supporting_team_keys': sorted(t['team_key'] for t in known), 'simultaneous_distinct_frames': len(separate), 'evidence_span_ms': max(separate) - min(separate), 'human_verified': False}
        prev_ev = inferred.get('identity_evidence') if isinstance(inferred.get('identity_evidence'), dict) else {}
        ev = {
            'winning_team': team,
            'winning_weight': prev_ev.get('winning_weight', 0.0),
            'total_weight': prev_ev.get('total_weight', 0.0),
            'share': prev_ev.get('share', 0.0),
            'margin': prev_ev.get('margin', 0.0),
            'runner_up_team': prev_ev.get('runner_up_team'),
            'runner_up_weight': prev_ev.get('runner_up_weight', 0.0),
            'decisive': False,
            'primary_source': 'roster_inferred',
            'by_team': prev_ev.get('by_team', {}),
            'proof': proof,
        }
        inferred['team_key'] = team
        inferred['identity_confidence'] = round(min(.7, *(t['identity_confidence'] for t in known)), 2)
        inferred['needs_review'] = True
        inferred['identity_evidence'] = ev
        if 'metrics' in inferred and isinstance(inferred['metrics'], dict):
            inferred['metrics'].update({'identity_source': 'roster_inferred', 'identity_inference': proof, 'identity_evidence': ev})
        audit.append({'team_key': team, **proof})
    return result, audit
