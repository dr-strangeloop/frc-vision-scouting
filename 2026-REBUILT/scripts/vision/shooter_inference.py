"""Tie uncredited HUB entries to a physical robot from the ball's own flight, without an unbroken track to the body.

Most balls are first seen in mid-air, so a continuous path back to a robot rarely exists. Five rules use physical
evidence instead, in this order. None sees a score, a clock or a total, and none splits a count across robots: an entry
that no rule ties to exactly one body stays unattributed, and a body with no supported name earns no team credit.

- arc: fit the flight and follow it backward. It must reach exactly one launching body of the entry's alliance.
- launch_share: among the alliance's observed launches in the flight window, one body holds nearly all the weight.
  Timing alone is not enough: a looser share disagreed with arc too often to credit.
- stream: a robot firing a burst sends balls along nearly the same path, so an entry follows resolved neighbours
  when they all belong to one body.
- sole: exactly one body of the alliance shows observed exits in the flight-time window before the entry, and both of
  its partners are boxed apart from it at that moment. A partner out of view could be the real shooter, so a sole
  launcher among fewer than three visible robots proves nothing.
- burst: every resolved entry of the alliance within half a second belongs to one body, and no other robot shows an
  exit in the flight window. A robot emptying its hopper is the only source of that moment's balls.

On 2026cc qm2/qm3 held-out flights (track starts hidden) gave the right body 84 of 89 times for arc; leave-one-out gave
459 of 474 for stream; known sources gave 54 of 58 for sole. These are correlated development checks, not ground truth.
"""
import bisect

import numpy as np

METHOD = 'shooter_inference_v1'
ARC_FIT_S = 0.45                 # early flight used for the fit
ARC_MIN_OBS = 5
ARC_REACH_MS = 800               # how far back the fitted flight is followed
ARC_PAD = 0.05                   # share of a body's box around it
ACTIVE_MS = 800                  # the body must show an observed exit this close to where the arc reaches it
STREAM_WINDOW_S = 1.0
STREAM_RADIUS_1080P = 15         # median path distance, pixels at 1920 wide
STREAM_VOTES = 2
STREAM_ROUNDS = 2
LAUNCH_SHARE_CONFIDENCE = 0.75   # the flight-time stage's confidence; one body holds at least 79% of the window's weight
SOLE_WINDOW_MS = (300, 1600)     # exit-to-entry flight time
SOLE_PARTNERS = 2                # other alliance bodies that must be in view, apart from the launcher
BURST_WINDOW_MS = 500
BURST_VOTES = 2


def _arc_body(obs, alliance, index, activity, offset_s, width, height):
    """The one launching body of `alliance` the fitted flight reaches, or None."""
    from robot_launches import _held_box
    t0 = float(obs[0][1])
    fit = [p for p in obs if p[1] - t0 <= ARC_FIT_S]
    if len(fit) < ARC_MIN_OBS:
        return None
    t = np.array([p[1] - t0 for p in fit])
    fit_u, fit_v = np.polyfit(t, [p[2] / width for p in fit], 1), np.polyfit(t, [p[3] / height for p in fit], 2)
    if fit_v[0] <= 0:             # image v must curve downward under gravity
        return None
    bodies = set()
    for back_ms in range(33, ARC_REACH_MS + 1, 33):
        s = back_ms / 1000
        u, v = np.polyval(fit_u, -s), np.polyval(fit_v, -s)
        if not (0 <= u <= 1 and 0 <= v <= 1):
            break
        t_ms = round((t0 - s - offset_s) * 1000)
        for robot, times, boxes in index:
            key = robot.get('track_key')
            if robot.get('alliance') != alliance or key in bodies:
                continue
            box = _held_box(times, boxes, t_ms)
            if box is None:
                continue
            pad_u, pad_v = (box[2] - box[0]) * ARC_PAD, (box[3] - box[1]) * ARC_PAD
            if box[0] - pad_u <= u <= box[2] + pad_u and box[1] - pad_v <= v <= box[3] + pad_v:
                exits = activity.get(key) or []
                at = bisect.bisect_left(exits, t_ms - ACTIVE_MS)
                if at < len(exits) and exits[at] <= t_ms + ACTIVE_MS:
                    bodies.add(key)
    return next(iter(bodies)) if len(bodies) == 1 else None


def _apart(a, b):
    return a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]


def _partners_in_view(launcher, t_ms, alliance, index):
    """Do at least SOLE_PARTNERS other bodies of `alliance` have a measured box at `t_ms`, each apart from the launcher
    and from each other (fragments of one robot count once)?"""
    from robot_launches import _held_box
    own = next((_held_box(times, boxes, t_ms) for robot, times, boxes in index if robot.get('track_key') == launcher), None)
    if own is None:
        return False
    seen = [own]
    for robot, times, boxes in index:
        if robot.get('alliance') != alliance or robot.get('track_key') == launcher:
            continue
        box = _held_box(times, boxes, t_ms)
        if box is not None and all(_apart(box, other) for other in seen):
            seen.append(box)
    return len(seen) - 1 >= SOLE_PARTNERS


def _stream_votes(n, paths, labels, alliances, radius):
    """{body: neighbours} among resolved entries of the alliance whose path this entry follows."""
    path, votes = paths[n], {}
    if path is None:
        return votes
    start = path[0, 0]
    for m, body in labels.items():
        other = paths[m]
        if m == n or other is None or alliances[m] != alliances[n] or abs(other[0, 0] - start) > STREAM_WINDOW_S:
            continue
        gaps = np.hypot(path[:, 1][:, None] - other[:, 1][None, :], path[:, 2][:, None] - other[:, 2][None, :]).min(axis=1)
        if np.median(gaps) <= radius:
            votes[body] = votes.get(body, 0) + 1
    return votes


def _launch_share_body(entry, by_key):
    """The body the flight-time stage found dominant for this entry, or None."""
    key = entry.get('candidate_track_key')
    if entry.get('candidate_method') != 'flight_time' or key not in by_key:
        return None
    return key if float(entry.get('attribution_confidence') or 0) >= LAUNCH_SHARE_CONFIDENCE else None


def _burst_body(n, entries, labels, launching, team_of):
    """The one body behind every resolved neighbour of entry `n`, if no other robot launched in its flight window."""
    entry = entries[n]
    near = [body for m, body in labels.items() if entries[m].get('alliance') == entry.get('alliance')
            and abs(entries[m]['t_ms'] - entry['t_ms']) <= BURST_WINDOW_MS]
    if len(set(near)) != 1 or len(near) < BURST_VOTES:
        return None
    body = near[0]
    low, high = entry['t_ms'] - SOLE_WINDOW_MS[1], entry['t_ms'] - SOLE_WINDOW_MS[0]
    rivals = {key for t, key in launching.get(entry.get('alliance'), ()) if low <= t <= high and key != body}
    return None if any(team_of(key) is None or team_of(key) != team_of(body) for key in rivals) else body


def infer_shooters(entries, exits, tracks, robots, offset_s, width, height):
    """(entries, stats). A tied entry records its `shooter_body` and rule in `source_link`; one on a supported name also
    gets `attribution: inferred_path` and the team. Already credited entries only act as stream neighbours."""
    from fuel_count import _obs
    from robot_launches import body_index, exit_activity, supported_identity
    by_key = {r.get('track_key'): r for r in robots or ()}
    # Only exits that kept their body count as launching: one the nearer-body veto rejected shows no launch at all.
    kept = [{**e, 'body_key': e['track_key']} for e in exits if e.get('track_key') in by_key]
    index, activity = body_index(robots), exit_activity(kept)
    paths, labels, rule = {}, {}, {}
    alliances = [e.get('alliance') for e in entries]
    for n, entry in enumerate(entries):
        tid = entry.get('track_id')
        valid = isinstance(tid, int) and not isinstance(tid, bool) and 0 <= tid < len(tracks)
        obs = list(_obs(tracks[tid])) if valid else []
        paths[n] = np.array([(p[1], p[2], p[3]) for p in obs]) if obs else None
        if entry.get('team_key'):
            if entry.get('track_key') in by_key:
                labels[n] = entry['track_key']
        elif obs:
            body = _arc_body(obs, entry.get('alliance'), index, activity, offset_s, width, height)
            if body is not None:
                labels[n], rule[n] = body, 'arc'
    for n, entry in enumerate(entries):
        body = None if n in labels or entry.get('team_key') else _launch_share_body(entry, by_key)
        if body is not None:
            labels[n], rule[n] = body, 'launch_share'
    radius = STREAM_RADIUS_1080P * width / 1920
    for _ in range(STREAM_ROUNDS):
        found = {}
        for n, entry in enumerate(entries):
            if n in labels or entry.get('team_key'):
                continue
            votes = _stream_votes(n, paths, labels, alliances, radius)
            if len(votes) == 1 and sum(votes.values()) >= STREAM_VOTES:
                found[n] = next(iter(votes))
        labels.update(found)
        rule.update({n: 'stream' for n in found})
    launching = {}
    for event in kept:
        launching.setdefault(by_key[event['body_key']].get('alliance'), []).append((event['t_ms'], event['body_key']))
    for n, entry in enumerate(entries):
        if n in labels or entry.get('team_key'):
            continue
        low, high = entry['t_ms'] - SOLE_WINDOW_MS[1], entry['t_ms'] - SOLE_WINDOW_MS[0]
        shots = [(t, key) for t, key in launching.get(entry.get('alliance'), ()) if low <= t <= high]
        bodies = {key for _, key in shots}
        if len(bodies) == 1 and all(_partners_in_view(key, t, entry.get('alliance'), index) for t, key in shots):
            labels[n], rule[n] = next(iter(bodies)), 'sole'

    def team_of(key):
        robot = by_key.get(key) or {}
        return robot.get('team_key') if supported_identity(robot.get('identity_evidence')) else None
    found = {n: _burst_body(n, entries, labels, launching, team_of) for n, entry in enumerate(entries)
             if n not in labels and not entry.get('team_key')}
    found = {n: body for n, body in found.items() if body is not None}
    labels.update(found)
    rule.update({n: 'burst' for n in found})
    out, stats = [], {'method': METHOD, 'bodies_tied': 0, 'named': 0,
                      'by_rule': {'arc': 0, 'launch_share': 0, 'stream': 0, 'sole': 0, 'burst': 0}}
    for n, entry in enumerate(entries):
        if n not in rule:
            out.append(entry)
            continue
        robot = by_key.get(labels[n]) or {}
        named = bool(robot.get('team_key') and supported_identity(robot.get('identity_evidence')))
        stats['bodies_tied'] += 1
        stats['by_rule'][rule[n]] += 1
        stats['named'] += named
        out.append({**entry, 'shooter_body': labels[n], 'source_link': rule[n],
                    **({'attribution': 'inferred_path', 'team_key': robot['team_key'], 'track_key': labels[n]} if named else {})})
    return out, stats
