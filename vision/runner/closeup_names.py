"""Name wide-view robots from a broadcast's close-up panels.

Some broadcasts add low cameras beside the wide shot. A robot near its HUB fills those panels with a legible bumper,
while launches are only measured in the wide view. Each panel's floor maps to the wide view's floor by one homography,
because the cameras are fixed for an event, so a robot read in a panel at time t is the wide-view robot standing at that
point at t. These carried reads are literal numbers, so they outrank wide-view body recognition on the same tracklet.

On 2026cc qm8, carried names took named HUB entries from 37% to 54%, and agreed with 8 of 9 independent wide-view
bumper reads. Of 5 disagreements with wide-view names, the 4 checked by eye were all wrong wide-view names. These are
single-reviewer development checks, not ground truth. Only the floor mapping comes from the event's own tracks.

    python3 closeup_names.py <event.closeups.json> <wide_bottom> <full-frame inputs.json.gz> [more inputs ...]
"""
from __future__ import annotations

import gzip
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

MIN_CONFIDENCE = 0.8
GATE_PX = 30                     # carried foot point to the wide robot's foot
RIVAL_FACTOR = 2.5               # the next wide robot must be this much further...
RIVAL_MARGIN_PX = 15             # ...plus this
MAX_TIME_GAP_MS = 80
MIN_PANEL_BOX_WIDTH = 90         # smaller boxes are cut off or far away, and their feet are unreliable
EDGE_PX = 6
PANEL_MAJORITY = 3               # a panel tracklet's name must outvote any other by this factor
WIDE_MAJORITY = 4                # as must a wide tracklet's carried name
FIT_MAX_WIDE_BOXES = 3
FIT_NEAR_PX = 12
FIT_ROUNDS = 4
FIT_RANSAC_PX = (14.0, 10.0)
FIT_MIN_INLIERS = 500


def split_panels(bottom):
    """A new camera's two close-ups below the wide shot. With no fitted mapping their reads name no wide robot, but they
    are legible bumpers, so they can still build the event's gallery."""
    return [{'name': 'left', 'roi': [0.0, bottom, 0.5, 1.0], 'homography': None},
            {'name': 'right', 'roi': [0.5, bottom, 1.0, 1.0], 'homography': None}]


def load_closeups(folder, event_key):
    """The event's panels [{'name', 'roi', 'homography'}], or None when no mapping was fitted for it."""
    path = Path(folder or '') / f'{event_key}.closeups.json'
    if not folder or not event_key or not path.is_file():
        return None
    data = json.loads(path.read_text())
    if data.get('event_key') != event_key:
        return None
    return data.get('panels') or None


def foot(box):
    return (box[0] + box[2]) / 2, box[3]


def project(matrix, point):
    x, y = point
    u, v, w = (row[0] * x + row[1] * y + row[2] for row in matrix)
    return u / w, v / w


def _named(votes, majority):
    """{id: name} where the top name outvotes any other by `majority`."""
    names = {}
    for key, counts in votes.items():
        top = counts.most_common(2)
        if len(top) == 1 or top[0][1] >= majority * top[1][1]:
            names[key] = top[0][0]
    return names


def _measured(box, width, height):
    x0, _, x1, y1 = box
    return x1 - x0 >= MIN_PANEL_BOX_WIDTH and x0 > EDGE_PX and x1 < width - EDGE_PX and y1 < height - EDGE_PX


def _wide_match(point, t_ms, candidates):
    """(tracklet, t) of the one wide robot at `point` at `t_ms`, or None. `candidates`: [(tracklet, t, box)]."""
    closest = {}                 # each wide robot where it was nearest in time, not nearest in space
    for row in candidates:
        if row[0] not in closest or abs(row[1] - t_ms) < abs(closest[row[0]][1] - t_ms):
            closest[row[0]] = row
    ranked = []
    for tid, t, box in closest.values():
        fx, fy = foot(box)
        ranked.append((((fx - point[0]) ** 2 + (fy - point[1]) ** 2) ** .5, tid, t))
    ranked.sort()
    if not ranked or ranked[0][0] > GATE_PX:
        return None
    if len(ranked) > 1 and ranked[1][0] <= RIVAL_FACTOR * ranked[0][0] + RIVAL_MARGIN_PX:
        return None
    return ranked[0][1], ranked[0][2]


def carried_reads(panel_tracklets, reads, wide_tracklets, panels, size):
    """(rows, names). A panel tracklet takes the name its confident reads agree on; every measured point of it then
    names the one wide robot it maps to. `rows` are literal observations on wide tracklets, `names` {wide tracklet: team}.
    No name is carried across an ambiguous match, and a wide tracklet with disagreeing carried names keeps none."""
    if not all(panel.get('homography') for panel in panels):
        return [], {}
    votes = defaultdict(Counter)
    for read in reads:
        confidence = read.get('confidence')
        if read.get('team_key') and isinstance(confidence, (int, float)) and confidence >= MIN_CONFIDENCE:
            votes[read['tracklet']][read['team_key']] += 1
    panel_names = _named(votes, PANEL_MAJORITY)
    by_panel = {panel['name']: panel['homography'] for panel in panels}
    wide = defaultdict(list)
    for track in wide_tracklets:
        for point in track['points']:
            wide[(track['alliance'], point['t'] // 100)].append((track['id'], point['t'], point['box']))
    rows, carried = {}, defaultdict(Counter)
    for track in panel_tracklets:
        team = panel_names.get(track['id'])
        if not team:
            continue
        matrix = by_panel[track['panel']]
        for point in track['points']:
            if not _measured(point['box'], *size):
                continue
            nearby = [row for bucket in (point['t'] // 100 - 1, point['t'] // 100, point['t'] // 100 + 1)
                      for row in wide.get((track['alliance'], bucket), ()) if abs(row[1] - point['t']) <= MAX_TIME_GAP_MS]
            match = _wide_match(project(matrix, foot(point['box'])), point['t'], nearby)
            if match is None:
                continue
            record = f"closeup|{track['id']}|{point['t']}|{team}".encode()
            rows[match] = {'tracklet': match[0], 't': match[1], 'team_key': team, 'identity_source': 'bumper_read',
                           'confidence': MIN_CONFIDENCE, 'via': 'closeup',
                           'read_record_sha256': hashlib.sha256(record).hexdigest()}
    for row in rows.values():
        carried[row['tracklet']][row['team_key']] += 1
    names = _named(carried, WIDE_MAJORITY)
    return [row for row in rows.values() if names.get(row['tracklet']) == row['team_key']], names


def _tracklet_key(tracklets):
    """What must match for two runs to share tracklets. Detection repeats on the same video, so ids line up."""
    return [(track['id'], track['alliance'], [(point['t'], [round(float(v)) for v in point['box']])
                                              for point in track['points']]) for track in tracklets]


def earlier_reads(folder, run_id, tracklets):
    """(rows, runs): literal reads kept by earlier runs of these same tracklets. Which tracklets earn a name varies
    between runs of one video, and on 2026cc qm7 two runs never disagreed across 62 tracklets, so reads add up."""
    if not folder:
        return [], 0
    wanted, rows, runs = _tracklet_key(tracklets), {}, 0
    for path in sorted(Path(folder).glob('*.json.gz')):
        try:
            data = json.loads(gzip.decompress(path.read_bytes()))
        except (OSError, ValueError, EOFError):
            continue
        if data.get('run_id') == run_id or _tracklet_key(data.get('tracklets') or []) != wanted:
            continue
        runs += 1
        for row in data.get('observations') or ():
            if row.get('identity_source') == 'bumper_read' and row.get('team_key'):
                rows[(row.get('tracklet'), row.get('t'), row['team_key'])] = {**row, 'via': 'earlier_run'}
    return list(rows.values()), runs


def merge_earlier(rows, names, earlier):
    """(rows, names) with earlier reads added. A tracklet the runs name differently keeps no carried name."""
    teams = defaultdict(set)
    for tracklet, team in names.items():
        teams[tracklet].add(team)
    for row in earlier:
        teams[row['tracklet']].add(row['team_key'])
    contested = {tracklet for tracklet, found in teams.items() if len(found) > 1}
    seen = {(row['tracklet'], row['t'], row['team_key']) for row in rows}
    merged = [row for row in rows + [r for r in earlier if (r['tracklet'], r['t'], r['team_key']) not in seen]
              if row['tracklet'] not in contested]
    return merged, {tracklet: next(iter(found)) for tracklet, found in teams.items() if tracklet not in contested}


def _footing(frames, side, wide_bottom, width, height):
    """(panel foot, wide foot) candidates from frames with one panel robot and at most three of its alliance wide."""
    split = wide_bottom * height
    panel_box = lambda box: ((box[0] + box[2]) / 2 < width / 2) == (side == 'left') and (box[1] + box[3]) / 2 >= split
    pairs = []
    for rows in frames:
        for alliance in ('red', 'blue'):
            panel = [box for team, box in rows if team == alliance and panel_box(box) and box[1] > split + 4
                     and _measured(box, width, height) and abs((box[0] + box[2]) / 2 - width / 2) > 20]
            wide = [box for team, box in rows if team == alliance and (box[1] + box[3]) / 2 < split]
            if len(panel) == 1 and 1 <= len(wide) <= FIT_MAX_WIDE_BOXES:
                pairs += [(foot(panel[0]), foot(box)) for box in wide]
    return pairs


def fit_panel(pairs):
    """Panel floor to wide floor: robust fit, then refit on what lands close. Most candidate pairs are other robots."""
    import cv2
    import numpy as np
    source = np.float32([pair[0] for pair in pairs])
    target = np.float32([pair[1] for pair in pairs])
    fits = [cv2.findHomography(source, target, cv2.RANSAC, limit, maxIters=200000, confidence=.9999)
            for limit in FIT_RANSAC_PX]
    matrix = max(fits, key=lambda fit: int(fit[1].sum()))[0]
    near = None
    for _ in range(FIT_ROUNDS):
        moved = cv2.perspectiveTransform(source[None], matrix)[0]
        near = np.hypot(*(moved - target).T) < FIT_NEAR_PX
        matrix = cv2.findHomography(source[near], target[near], 0)[0]
    return matrix.tolist(), int(near.sum())


def main(argv):
    from identity_refinement import load_inputs
    out, wide_bottom, *paths = argv[1:]
    wide_bottom, frames, size, event_key = float(wide_bottom), [], None, None
    for path in paths:
        inputs = load_inputs(path)
        size, event_key = inputs['size'], inputs['event_key']
        at = defaultdict(list)
        for track in inputs['tracklets']:
            for point in track['points']:
                at[point['t']].append((track['alliance'], point['box']))
        frames += list(at.values())
    panels = []
    for name, roi in (('left', [0.0, wide_bottom, 0.5, 1.0]), ('right', [0.5, wide_bottom, 1.0, 1.0])):
        matrix, inliers = fit_panel(_footing(frames, name, wide_bottom, *size))
        print({'panel': name, 'inliers': inliers})
        if inliers < FIT_MIN_INLIERS:
            raise SystemExit(f'{name}: only {inliers} matched robots; not enough to trust a mapping')
        panels.append({'name': name, 'roi': roi, 'homography': matrix})
    Path(out).write_text(json.dumps({'event_key': event_key, 'panels': panels}))


if __name__ == '__main__':
    main(sys.argv)
