"""Where two robots' identities could have been swapped.

Tracking joins short pieces of path into one robot's track. When two same-alliance robots drive past or
into each other, a join made at that moment can hand one robot's later path to the other, and nothing in the
result shows it. The fix other vision scouting projects settled on is an offline pass that finds those
moments and sends them to a person. A close encounter alone is not a problem (robots meet all the time); it
is a problem only if a join in either track's chain happened around the same time, because that is where a
swap can occur.
"""
import math

RADIUS_M = 1.2          # robots this close (centres) are close enough to be confused
MIN_ENCOUNTER_S = 0.3   # shorter than this is a pass, not a tangle
LINK_WINDOW_S = 2.0     # a join this near an encounter makes a swap possible


def _position(points, t_ms):
    """Interpolated (x, y) at t_ms, or None when the track has no sample within 500 ms."""
    lo, hi = 0, len(points) - 1
    if not points or t_ms < points[0]['t'] - 500 or t_ms > points[-1]['t'] + 500:
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if points[mid]['t'] < t_ms:
            lo = mid + 1
        else:
            hi = mid
    after = points[lo]
    before = points[lo - 1] if lo > 0 else after
    if min(abs(t_ms - before['t']), abs(after['t'] - t_ms)) > 500:
        return None
    if after['t'] == before['t']:
        return after['x'], after['y']
    f = min(1.0, max(0.0, (t_ms - before['t']) / (after['t'] - before['t'])))
    return before['x'] + (after['x'] - before['x']) * f, before['y'] + (after['y'] - before['y']) * f


def close_encounters(a, b, radius=RADIUS_M, min_s=MIN_ENCOUNTER_S, step_ms=100):
    """[(start_ms, end_ms, closest_m)] while tracks a and b (lists of {t, x, y}) are within `radius` of each other."""
    if not a or not b:
        return []
    start, end = max(a[0]['t'], b[0]['t']), min(a[-1]['t'], b[-1]['t'])
    out, current = [], None
    t = start
    while t <= end:
        pa, pb = _position(a, t), _position(b, t)
        d = math.hypot(pa[0] - pb[0], pa[1] - pb[1]) if pa and pb else None
        if d is not None and d <= radius:
            if current is None:
                current = [t, t, d]
            else:
                current[1] = t
                current[2] = min(current[2], d)
        elif current is not None:
            if (current[1] - current[0]) / 1000 >= min_s:
                out.append(tuple(current))
            current = None
        t += step_ms
    if current is not None and (current[1] - current[0]) / 1000 >= min_s:
        out.append(tuple(current))
    return out


def swap_risks(tracks, join_times_ms, link_window_s=LINK_WINDOW_S):
    """For same-alliance `tracks` (dicts with 'trajectory'), the encounters where a join could have swapped identities.

    join_times_ms: for each track, the times (ms) at which its chain was joined from separate pieces.
    Returns, per track, [{'with': index of the other track, 't_ms', 'closest_m'}]; empty when nothing is at risk.
    """
    risks = [[] for _ in tracks]
    window = link_window_s * 1000
    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            traj_i = tracks[i]['trajectory']
            traj_j = tracks[j]['trajectory']
            for start, end, closest in close_encounters(traj_i, traj_j):
                joins = [t for t in list(join_times_ms[i]) + list(join_times_ms[j]) if start - window <= t <= end + window]
                if joins:
                    risks[i].append({'with': j, 't_ms': int(start), 'closest_m': round(closest, 2)})
                    risks[j].append({'with': i, 't_ms': int(start), 'closest_m': round(closest, 2)})
            if join_times_ms[i] and join_times_ms[j]:
                mutual_pairs = [
                    (ti, tj)
                    for ti in join_times_ms[i]
                    for tj in join_times_ms[j]
                    if abs(ti - tj) <= window
                ]
                mutual_pairs.sort(key=lambda pair: (abs(pair[0] - pair[1]), min(pair[0], pair[1])))
                for ti, tj in mutual_pairs:
                    pi = _position(traj_i, ti) or _position(traj_i, tj)
                    pj_at_ti = _position(traj_j, ti)
                    pj_at_tj = _position(traj_j, tj)
                    candidates_d = []
                    for p_a in (_position(traj_i, ti), _position(traj_i, tj), pi):
                        for p_b in (pj_at_ti, pj_at_tj):
                            if p_a is not None and p_b is not None:
                                candidates_d.append(math.hypot(p_a[0] - p_b[0], p_a[1] - p_b[1]))
                    if not candidates_d:
                        continue
                    d = min(candidates_d)
                    t_risk = int(min(ti, tj))
                    if d <= RADIUS_M and not any(r['with'] == j and abs(r['t_ms'] - t_risk) <= window for r in risks[i]):
                        risks[i].append({'with': j, 't_ms': t_risk, 'closest_m': round(d, 2)})
                        risks[j].append({'with': i, 't_ms': t_risk, 'closest_m': round(d, 2)})
    return risks
