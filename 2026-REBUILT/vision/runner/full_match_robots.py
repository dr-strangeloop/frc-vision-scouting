"""Full-match observed robot boxes and source-bound identities for fuel attribution.

Production `identify_robots` uses conservative motion/appearance tracklets, literal bumper reads and an optional
private reference gallery. Ambiguous associations and lost views end a fragment rather than manufacture identity.
Older fixed-count/path assignment helpers remain for offline diagnostics only.
"""
from __future__ import annotations

import math
import hashlib
from collections import defaultdict
from itertools import permutations
from pathlib import Path

ALLIANCES = ("red", "blue")


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def size(box):
    return max(1.0, ((box[2] - box[0]) + (box[3] - box[1])) / 2.0)


def dedupe(detections, iou_threshold=0.5):
    """[(box, confidence, alliance[, read_score])] with at most one box per robot: of boxes overlapping by `iou_threshold` or more
    (the detector's red and blue classes on one robot), the more confident one is kept."""
    kept = []
    for det in sorted(detections, key=lambda d: d[1], reverse=True):
        if all(iou(det[0], other[0]) < iou_threshold for other in kept):
            kept.append(det)
    return kept


def _best_assignment(cost, n_rows, n_cols):
    """Maximum-cardinality, minimum-cost matching; None marks forbidden.

    Keep a bitmask of the smaller side rather than enumerating track permutations. Conservative fragments can leave
    more than three tracks open, while a frame still has only a few actual robot detections.
    """
    if not n_rows or not n_cols:
        return {}
    transpose = n_rows < n_cols
    outer, inner = (n_cols, n_rows) if transpose else (n_rows, n_cols)
    states = {0: (0.0, {})}
    for i in range(outer):
        next_states = dict(states)     # an observation/track may remain unmatched
        for mask, (total, pairs) in states.items():
            for j in range(inner):
                row, col = (j, i) if transpose else (i, j)
                value = cost[row][col]
                if mask & (1 << j) or value is None:
                    continue
                new_mask, new_total = mask | (1 << j), total + value
                if new_mask not in next_states or new_total < next_states[new_mask][0]:
                    next_states[new_mask] = (new_total, {**pairs, row: col})
        states = next_states
    best = max(states, key=lambda mask: (mask.bit_count(), -states[mask][0]))
    return states[best][1]


class FixedCountTracker:
    """Exactly `per_alliance` tracks per alliance over sampled frames (times in ms, boxes in pixels).

    gate: a box can continue a track when its center is within `gate_sizes` box sizes of the prediction, widened by
    `grow_per_s` sizes for every second the track has gone unseen (a robot moves while hidden), up to `max_gate_sizes`.
    """

    def __init__(self, per_alliance=3, gate_sizes=1.5, grow_per_s=2.0, max_gate_sizes=12.0, smoothing=0.6):
        self.per_alliance = per_alliance
        self.gate_sizes = gate_sizes
        self.grow_per_s = grow_per_s
        self.max_gate_sizes = max_gate_sizes
        self.smoothing = smoothing
        self.tracks = {a: [] for a in ALLIANCES}

    def _predict(self, track, t_ms):
        dt = (t_ms - track["t_ms"]) / 1000.0
        damp = min(1.0, dt)             # trust velocity for about a second, then hold position
        return (track["c"][0] + track["v"][0] * dt * (1 - damp / 2),
                track["c"][1] + track["v"][1] * dt * (1 - damp / 2))

    def step(self, t_ms, detections):
        """Feed one sampled frame: detections [(box, confidence, alliance[, read_score])]. Returns {alliance: [(track
        index, box, observed, read_score)]}: the detection when matched, else the predicted box (observed False, no
        score)."""
        out = {}
        for alliance in ALLIANCES:
            boxes = [d for d in detections if d[2] == alliance]
            tracks = self.tracks[alliance]
            cost = []
            for track in tracks:
                pred = self._predict(track, t_ms)
                unseen_s = (t_ms - track["seen_ms"]) / 1000.0
                gate = min(self.max_gate_sizes, self.gate_sizes + self.grow_per_s * unseen_s) * track["s"]
                row = []
                for det in boxes:
                    box, conf = det[0], det[1]
                    d = math.dist(pred, center(box))
                    row.append(d / track["s"] - 0.5 * conf if d <= gate else None)
                cost.append(row)
            matches = _best_assignment(cost, len(tracks), len(boxes)) if tracks and boxes else {}
            used = set(matches.values())
            for ti, bi in matches.items():
                track, box = tracks[ti], boxes[bi][0]
                c, dt = center(box), max(1e-3, (t_ms - track["t_ms"]) / 1000.0)
                a = self.smoothing
                v = ((c[0] - track["c"][0]) / dt, (c[1] - track["c"][1]) / dt)
                track.update(c=c, v=(a * v[0] + (1 - a) * track["v"][0], a * v[1] + (1 - a) * track["v"][1]),
                             s=a * size(box) + (1 - a) * track["s"], t_ms=t_ms, seen_ms=t_ms, box=box)
            # Unmatched boxes start a track while the alliance has fewer than three; after that a box nobody could
            # continue re-acquires the track that has been lost longest (a robot seen again far from where it hid).
            for bi, det in sorted(enumerate(boxes), key=lambda x: -x[1][1]):
                box = det[0]
                if bi in used:
                    continue
                if len(tracks) < self.per_alliance:
                    tracks.append({"c": center(box), "v": (0.0, 0.0), "s": size(box), "t_ms": t_ms, "seen_ms": t_ms,
                                   "box": box})
                    matches[len(tracks) - 1] = bi
                    continue
                lost = [ti for ti in range(len(tracks)) if ti not in matches and t_ms - tracks[ti]["seen_ms"] >= 1000]
                if lost:
                    ti = min(lost, key=lambda i: tracks[i]["seen_ms"])
                    tracks[ti].update(c=center(box), v=(0.0, 0.0), s=size(box), t_ms=t_ms, seen_ms=t_ms, box=box)
                    matches[ti] = bi
            rows = []
            for ti, track in enumerate(tracks):
                if ti in matches:
                    det = boxes[matches[ti]]
                    rows.append((ti, det[0], True, det[3] if len(det) > 3 else None))
                else:
                    pc = self._predict(track, t_ms)
                    w, h = track["box"][2] - track["box"][0], track["box"][3] - track["box"][1]
                    rows.append((ti, (pc[0] - w / 2, pc[1] - h / 2, pc[0] + w / 2, pc[1] + h / 2), False, None))
            out[alliance] = rows
        return out


def track_match(samples, **tracker_options):
    """samples: [(t_ms, [(box, confidence, alliance[, read_score])])] in time order. Returns {(alliance, index):
    [{'t', 'box', 'observed'[, 'read_score']}]} for every track the tracker made (at most three per alliance)."""
    tracker = FixedCountTracker(**tracker_options)
    out = defaultdict(list)
    for t_ms, detections in samples:
        for alliance, rows in tracker.step(t_ms, dedupe(detections)).items():
            for ti, box, observed, score in rows:
                point = {"t": int(t_ms), "box": [float(v) for v in box], "observed": observed}
                if score is not None:
                    point["read_score"] = float(score)
                out[(alliance, ti)].append(point)
    return dict(out)


def coverage(points, start_ms=0, end_ms=None):
    """Share of the window in which the track's robot was actually detected (sampled frames)."""
    window = [p for p in points if p["t"] >= start_ms and (end_ms is None or p["t"] <= end_ms)]
    return sum(p["observed"] for p in window) / len(window) if window else 0.0


def crop_plan(tracks, every_ms=500, max_per_track=12, min_gap_ms=4000, min_size_px=42, max_overlap=0.05):
    """{track id: [(t, box)]} in time order: the frames to read each robot's bumper from.

    Candidates are at most one frame per `every_ms`, only where the robot itself was detected, its box is at least
    `min_size_px` on both sides, and it overlaps no other robot's box. Each read costs seconds of Qwen time, so of
    those the largest boxes (closest to the camera, most readable digits) are taken first, at least `min_gap_ms`
    apart so they come from different moments of the match, up to `max_per_track`. A point's `read_score` (bumper digits
    visible and sharp, measured in the detector pass) ranks ahead of size when present."""
    by_time = defaultdict(list)
    for key, points in tracks.items():
        for p in points:
            if p["observed"]:
                by_time[p["t"]].append((key, p["box"], p.get("read_score")))
    candidates = defaultdict(list)
    last = {}
    for t in sorted(by_time):
        for key, box, score in by_time[t]:
            if t - last.get(key, -math.inf) < every_ms:
                continue
            if box[2] - box[0] < min_size_px or box[3] - box[1] < min_size_px:
                continue
            if any(iou(box, other) > max_overlap for k, other, _ in by_time[t] if k != key):
                continue
            candidates[key].append((t, box, score))
            last[key] = t
    plan = {}
    for key, rows in candidates.items():
        chosen = []
        # Ranked by the detection's bumper-readability score when the detector pass measured one (visible white digits
        # on the bumper band, sharpness), else by box area.
        def rank(r):
            area = (r[1][2] - r[1][0]) * (r[1][3] - r[1][1])
            return (-(r[2] if r[2] is not None else -1.0), -area)
        for t, box, _ in sorted(rows, key=rank):
            if len(chosen) >= max_per_track:
                break
            if all(abs(t - c[0]) >= min_gap_ms for c in chosen):
                chosen.append((t, box))
        plan[key] = sorted(chosen)
    return plan


def assign_teams(reads, roster, min_confidence=0.65):
    """reads: [{'track': (alliance, index), 'team_key', 'confidence', 't'}]; roster: {'red': [...], 'blue': [...]}.

    Within each alliance every roster team goes to exactly one track and every track to at most one team, maximizing
    the summed confidence of agreeing reads. A track with no read for its team keeps no name. Returns {track: {'team_key',
    'reads', 'agree', 'consistency'}} where consistency is the share of the track's reads naming its team."""
    weight = defaultdict(float)
    count = defaultdict(int)
    per_track = defaultdict(list)
    for r in reads:
        conf = r.get("confidence")
        if not isinstance(conf, (int, float)) or isinstance(conf, bool) or conf < min_confidence:
            continue
        track, team = tuple(r["track"]), r.get("team_key")
        if team not in (roster.get(track[0]) or []):
            continue
        weight[(track, team)] += float(conf)
        count[(track, team)] += 1
        per_track[track].append(team)
    result = {}
    for alliance in ALLIANCES:
        teams = list(roster.get(alliance) or [])
        tracks = sorted({t for t, _ in weight if t[0] == alliance} | {t for t in per_track if t[0] == alliance})
        if not teams or not tracks:
            continue
        cost = [[(-weight[(t, team)] if weight[(t, team)] > 0 else None) for team in teams] for t in tracks]
        for ti, ci in _best_assignment(cost, len(tracks), len(teams)).items():
            track, team = tracks[ti], teams[ci]
            n = len(per_track[track])
            result[track] = {"team_key": team, "reads": n, "agree": count[(track, team)],
                             "consistency": count[(track, team)] / n if n else 0.0}
    return result


def read_score(frame, box):
    """How likely a robot's bumper number is readable in this frame (0 to about 0.6): visible white digits on the
    bumper band and sharpness, the parts of team_identity.crop_robot's score that are not size. Cheap enough to run on
    every detection in the detector pass, so the Qwen budget goes to frames where digits can actually be seen."""
    import cv2
    from team_identity import bumper_contrast_bonus
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if crop.size == 0:
        return 0.0
    sharpness = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    return round(bumper_contrast_bonus(crop) + min(1.0, sharpness / 300) * 0.2, 4)


def name_segments(tracks, reads, roster, min_confidence=0.65, cover_share=0.8):
    """Name each track's segments from its bumper reads (roster teams only, confidence >= min_confidence).

    A track is cut where its reads change team (the tracker swapped two robots or re-acquired the wrong one), at the
    midpoint between the two reads. Two overlapping segments of one alliance cannot be the same robot: the one with
    fewer reads loses its name. A segment with no reads is named by elimination only when, over at least
    `cover_share` of its time, the alliance's two other tracks carry two different named teams; it is marked
    `by: 'elimination'` so a reviewer can tell; a segment whose reads name another track's robot stays unnamed
    (`by: 'conflict'`) and is never named by elimination. Returns {track: [{'start', 'end', 'team_key', 'reads', 'by'}]}.
    """
    valid = defaultdict(list)
    for r in reads:
        conf = r.get("confidence")
        track = tuple(r["track"])
        if (isinstance(conf, (int, float)) and not isinstance(conf, bool) and conf >= min_confidence
                and r.get("team_key") in (roster.get(track[0]) or [])):
            valid[track].append((r["t"], r["team_key"]))
    segments = {}
    for track, points in tracks.items():
        start, end = points[0]["t"], points[-1]["t"]
        rows = sorted(valid.get(track, []))
        cuts, segs, current = [], [], []
        for i, (t, team) in enumerate(rows):
            if current and current[-1][1] != team:
                cut = (current[-1][0] + t) // 2
                segs.append((cuts[-1] if cuts else start, cut, current))
                cuts.append(cut)
                current = []
            current.append((t, team))
        segs.append((cuts[-1] if cuts else start, end, current))
        segments[track] = [{"start": s, "end": e, "team_key": (c[0][1] if c else None), "reads": len(c),
                            "by": "reads" if c else None} for s, e, c in segs]
    overlap = lambda a, b: max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))  # noqa: E731
    for alliance in ALLIANCES:
        segs = [(t, s) for t, rows in segments.items() if t[0] == alliance for s in rows]
        for (ta, a), (tb, b) in ((x, y) for x in segs for y in segs if x[0] != y[0]):
            if a["team_key"] and a["team_key"] == b["team_key"] and overlap(a, b) > 0:
                loser = a if (a["reads"], ta) < (b["reads"], tb) else b
                loser.update(team_key=None, by="conflict")      # its reads name a robot another track holds
        teams = set(roster.get(alliance) or [])
        for t, seg in segs:
            if seg["team_key"] or seg["by"] == "conflict" or seg["end"] <= seg["start"]:
                continue
            others = [(ot, os_) for ot, os_ in segs if ot != t and os_["team_key"]]
            span = seg["end"] - seg["start"]
            named = {}
            for ot, os_ in others:
                named.setdefault(os_["team_key"], 0)
                named[os_["team_key"]] += overlap(seg, os_)
            covered = [team for team, ms in named.items() if ms >= cover_share * span]
            left = teams - set(covered)
            if len(covered) == 2 and len(left) == 1:
                seg.update(team_key=left.pop(), by="elimination")
    return segments


def histogram_agreement(a, b):
    """Bhattacharyya agreement of normalized body histograms, or None without comparable evidence."""
    if not a or not b or len(a) != len(b):
        return None
    total_a, total_b = sum(a), sum(b)
    if total_a <= 0 or total_b <= 0:
        return None
    return sum(math.sqrt(max(0.0, x) * max(0.0, y)) for x, y in zip(a, b)) / math.sqrt(total_a * total_b)


def tracklets(samples, gate_sizes=1.5, max_gap_ms=400, min_appearance_agreement=0.65, ambiguity_margin=0.12):
    """Short, conservative tracks: [{'id', 'alliance', 'points': [{'t', 'box', 'read_score'}]}].

    Each sampled frame's boxes are matched to the open tracklets of their alliance by predicted position (Hungarian,
    gate `gate_sizes` box sizes) and observed body appearance. An ambiguous crossing ends the competing tracklets;
    a new box never inherits their bumper identity. A box nobody continues starts a new tracklet, and a tracklet
    unseen for more than `max_gap_ms` ends. No predicted boxes are emitted as observations."""
    done, open_ = [], {a: [] for a in ALLIANCES}
    next_id = 0
    for t_ms, detections in samples:
        dets = dedupe(detections)
        for alliance in ALLIANCES:
            boxes = [d for d in dets if d[2] == alliance]
            still = []
            for tr in open_[alliance]:
                if t_ms - tr["points"][-1]["t"] > max_gap_ms:
                    done.append(tr)
                else:
                    still.append(tr)
            open_[alliance] = trs = still
            cost = []
            for tr in trs:
                last = tr["points"][-1]
                dt = (t_ms - last["t"]) / 1000.0
                pred = (center(last["box"])[0] + tr["v"][0] * dt, center(last["box"])[1] + tr["v"][1] * dt)
                s = size(last["box"])
                row = []
                for det in boxes:
                    d = math.dist(pred, center(det[0]))
                    agreement = histogram_agreement(last.get("hist"), det[4] if len(det) > 4 else None)
                    plausible = d <= gate_sizes * s and (agreement is None or agreement >= min_appearance_agreement)
                    row.append(d / s - 0.5 * det[1] + 0.75 * (1 - agreement if agreement is not None else 0)
                               if plausible else None)
                cost.append(row)
            matches = _best_assignment(cost, len(trs), len(boxes)) if trs and boxes else {}
            ambiguous = set()
            for ti, bi in list(matches.items()):
                chosen = cost[ti][bi]
                alternatives = [v for j, v in enumerate(cost[ti]) if j != bi and v is not None]
                alternatives += [row[bi] for j, row in enumerate(cost) if j != ti and row[bi] is not None]
                if alternatives and min(alternatives) - chosen < ambiguity_margin:
                    ambiguous.add(ti)
                    del matches[ti]
            for ti, bi in matches.items():
                tr, det = trs[ti], boxes[bi]
                last = tr["points"][-1]
                dt = max(1e-3, (t_ms - last["t"]) / 1000.0)
                c0, c1 = center(last["box"]), center(det[0])
                tr["v"] = (0.6 * (c1[0] - c0[0]) / dt + 0.4 * tr["v"][0], 0.6 * (c1[1] - c0[1]) / dt + 0.4 * tr["v"][1])
                tr["points"].append({"t": int(t_ms), "box": [float(v) for v in det[0]],
                                     "read_score": det[3] if len(det) > 3 else None,
                                     "hist": det[4] if len(det) > 4 else None})
            # Do not keep a competing pre-crossing identity open for a later frame to reacquire by proximity.
            done.extend(tr for ti, tr in enumerate(trs) if ti in ambiguous)
            open_[alliance] = [tr for ti, tr in enumerate(trs) if ti not in ambiguous]
            new_tracks = open_[alliance]
            for bi, det in enumerate(boxes):
                if bi not in matches.values():
                    new_tracks.append({"id": next_id, "alliance": alliance, "v": (0.0, 0.0),
                                "points": [{"t": int(t_ms), "box": [float(v) for v in det[0]],
                                            "read_score": det[3] if len(det) > 3 else None,
                                            "hist": det[4] if len(det) > 4 else None}]})
                    next_id += 1
    done.extend(tr for a in ALLIANCES for tr in open_[a])
    for tr in done:
        tr.pop("v", None)
        tr['association'] = {'method': 'hsv192-motion-v1', 'max_gap_ms': max_gap_ms,
                             'min_appearance_agreement': min_appearance_agreement,
                             'ambiguity_margin': ambiguity_margin}
    return sorted(done, key=lambda tr: (tr["points"][0]["t"], tr["id"]))


def link_robots(tracklet_list, reads, roster, min_confidence=0.65, link_gap_ms=3000, link_sizes=4.0, cover_share=0.8):
    """Name tracklets: {tracklet id: {'team_key', 'by'}} with by in 'reads', 'link', 'elimination', 'conflict'.

    reads: [{'tracklet': id, 'team_key', 'confidence'}]. Rules, applied until nothing changes:
    - reads: a tracklet's agreeing roster reads name it (a tracklet with reads for two teams keeps the majority);
    - conflict: two tracklets of one alliance overlapping in time cannot be one robot; the one with fewer reads for
      the shared team loses its name and is never named again by the other rules;
    - link: an unnamed tracklet starting within `link_gap_ms` of a named tracklet's end, within `link_sizes` box sizes
      (per second of gap, at least one) of where it ended, takes its name when that named tracklet is its only
      plausible predecessor and nothing overlapping already holds that name; the same backwards for successors;
    - elimination: over at least `cover_share` of an unnamed tracklet's time, the alliance's other robots carry two
      different named teams, so it is the third.
    """
    by_id = {tr["id"]: tr for tr in tracklet_list}
    span = {i: (tr["points"][0]["t"], tr["points"][-1]["t"]) for i, tr in by_id.items()}
    overlap = lambda a, b: max(0, min(span[a][1], span[b][1]) - max(span[a][0], span[b][0]))  # noqa: E731
    votes = defaultdict(lambda: defaultdict(int))
    for r in reads:
        conf, tid = r.get("confidence"), r.get("tracklet")
        if tid in by_id and isinstance(conf, (int, float)) and not isinstance(conf, bool) and conf >= min_confidence \
                and r.get("team_key") in (roster.get(by_id[tid]["alliance"]) or []):
            votes[tid][r["team_key"]] += 1
    names = {tid: {"team_key": max(v, key=v.get), "by": "reads", "reads": max(v.values())} for tid, v in votes.items()}
    for a in list(names):
        for b in list(names):
            if a < b and names.get(a, {}).get("team_key") and names[a]["team_key"] == names.get(b, {}).get("team_key") \
                    and by_id[a]["alliance"] == by_id[b]["alliance"] and overlap(a, b) > 0:
                loser = a if names[a]["reads"] < names[b]["reads"] else b
                names[loser] = {"team_key": None, "by": "conflict", "reads": names[loser]["reads"]}

    def free_for(tid, team):
        return all(not (names.get(o, {}).get("team_key") == team and overlap(tid, o) > 0)
                   for o in by_id if o != tid and by_id[o]["alliance"] == by_id[tid]["alliance"])

    def plausible(a_end, b_start):
        gap = b_start["t"] - a_end["t"]
        if not 0 < gap <= link_gap_ms:
            return False
        reach = link_sizes * max(1.0, gap / 1000.0) * size(a_end["box"])
        return math.dist(center(a_end["box"]), center(b_start["box"])) <= reach

    changed = True
    while changed:
        changed = False
        for tid, tr in by_id.items():
            if names.get(tid):
                continue
            first, last = tr["points"][0], tr["points"][-1]
            same = [o for o in by_id if o != tid and by_id[o]["alliance"] == tr["alliance"]]
            preds = [o for o in same if plausible(by_id[o]["points"][-1], first)]
            succs = [o for o in same if plausible(last, by_id[o]["points"][0])]
            for group in (preds, succs):
                if len(group) == 1 and (names.get(group[0]) or {}).get("team_key"):
                    team = names[group[0]]["team_key"]
                    if free_for(tid, team):
                        names[tid] = {"team_key": team, "by": "link", "reads": 0}
                        changed = True
                        break
            if names.get(tid):
                continue
            length = max(1, span[tid][1] - span[tid][0])
            held = defaultdict(int)
            for o in same:
                if (names.get(o) or {}).get("team_key"):
                    held[names[o]["team_key"]] += overlap(tid, o)
            covered = {team for team, ms in held.items() if ms >= cover_share * length}
            left = set(roster.get(tr["alliance"]) or []) - covered
            if len(covered) == 2 and len(left) == 1 and free_for(tid, next(iter(left))):
                names[tid] = {"team_key": left.pop(), "by": "elimination", "reads": 0}
                changed = True
    return names


def team_coverage(tracklet_list, names, samples_t):
    """{team: share of sampled frames in which a tracklet named for it was detected}."""
    frames = defaultdict(set)
    for tr in tracklet_list:
        team = (names.get(tr["id"]) or {}).get("team_key")
        if team:
            frames[team].update(p["t"] for p in tr["points"])
    total = max(1, len(samples_t))
    return {team: round(len(ts) / total, 3) for team, ts in frames.items()}


def _min_cost_paths(n_nodes, edges, source, sink, k):
    """Up to k unit flows source->sink of minimum total cost (successive shortest paths, Bellman-Ford on the residual
    graph; costs may be negative). edges: [(u, v, cost)] with capacity 1. Returns the set of used edge indices."""
    graph = [[] for _ in range(n_nodes)]
    arcs = []                                  # [to, cap, cost, rev index, edge index or None]
    for idx, (u, v, c) in enumerate(edges):
        graph[u].append(len(arcs)); arcs.append([v, 1, c, len(arcs) + 1, idx])
        graph[v].append(len(arcs)); arcs.append([u, 0, -c, len(arcs) - 1, None])
    for _ in range(k):
        dist = [math.inf] * n_nodes; prev = [None] * n_nodes; dist[source] = 0.0
        for _ in range(n_nodes):
            updated = False
            for u in range(n_nodes):
                if dist[u] == math.inf:
                    continue
                for ai in graph[u]:
                    to, cap, cost = arcs[ai][0], arcs[ai][1], arcs[ai][2]
                    if cap > 0 and dist[u] + cost < dist[to] - 1e-9:
                        dist[to] = dist[u] + cost; prev[to] = ai; updated = True
            if not updated:
                break
        if dist[sink] == math.inf or dist[sink] >= 0:
            break                                # no path left that lowers the cost
        v = sink
        while v != source:
            ai = prev[v]; arcs[ai][1] -= 1; arcs[arcs[ai][3]][1] += 1; v = arcs[arcs[ai][3]][0]
    return {arcs[ai][4] for ai in range(len(arcs)) if arcs[ai][4] is not None and arcs[ai][1] == 0}


def link_paths(tracklet_list, per_alliance=3, match_ms=165000, max_gap_ms=8000, sizes_per_s=3.0,
               point_reward=1.0, gap_cost_per_s=2.0, open_cost=30.0, edge_ms=3000):
    """{alliance: [[tracklet id, ...] per robot]}: the `per_alliance` paths through the alliance's tracklets that
    cover the most detections for the least cost (minimum-cost flow).

    Including a tracklet earns `point_reward` per detection. Linking tracklet a to a later b costs the distance between
    a's last box and b's first box in box sizes plus `gap_cost_per_s` per second of gap, and is allowed only within
    `max_gap_ms` and `sizes_per_s` box sizes per second of gap (at least one second's worth). A path starting after
    `edge_ms` into the match or ending more than `edge_ms` before its end pays `open_cost`, since every robot is on
    the field for the whole match. A tracklet on no path is left over (a false detection or an unlinkable piece)."""
    out = {}
    for alliance in ALLIANCES:
        trs = [tr for tr in tracklet_list if tr["alliance"] == alliance]
        n = len(trs)
        source, sink = 2 * n, 2 * n + 1
        edges = []
        for i, tr in enumerate(trs):
            first, last = tr["points"][0], tr["points"][-1]
            edges.append((2 * i, 2 * i + 1, -point_reward * len(tr["points"])))
            edges.append((source, 2 * i, open_cost if first["t"] > edge_ms else 0.0))
            edges.append((2 * i + 1, sink, open_cost if last["t"] < match_ms - edge_ms else 0.0))
            for j, other in enumerate(trs):
                b = other["points"][0]
                gap = b["t"] - last["t"]
                if not 0 < gap <= max_gap_ms:
                    continue
                s = size(last["box"])
                d = math.dist(center(last["box"]), center(b["box"])) / s
                if d > sizes_per_s * max(1.0, gap / 1000.0):
                    continue
                edges.append((2 * i + 1, 2 * j, d + gap_cost_per_s * gap / 1000.0))
        used = _min_cost_paths(2 * n + 2, edges, source, sink, per_alliance)
        nxt = {}
        starts = []
        for idx in used:
            u, v, _ = edges[idx]
            if u == source:
                starts.append(v // 2)
            elif v != sink and u % 2 == 1 and v % 2 == 0:
                nxt[u // 2] = v // 2
        paths = []
        for s0 in starts:
            path, cur = [], s0
            while cur is not None:
                path.append(trs[cur]["id"]); cur = nxt.get(cur)
            paths.append(path)
        out[alliance] = sorted(paths, key=lambda p: p[0])
    return out


def name_paths(paths, reads, roster, min_confidence=0.65):
    """{(alliance, path index): {'team_key', 'agree', 'reads', 'consistency'}}: within each alliance one roster team per
    path, maximizing agreeing reads (reads: [{'tracklet', 'team_key', 'confidence'}]). A path with no read agreeing
    with any free team stays unnamed."""
    path_of = {tid: (a, i) for a, ps in paths.items() for i, p in enumerate(ps) for tid in p}
    as_tracks = [{"track": list(path_of[r["tracklet"]]), "team_key": r.get("team_key"), "confidence": r.get("confidence")}
                 for r in reads if r.get("tracklet") in path_of]
    return {tuple(k): v for k, v in assign_teams(as_tracks, roster, min_confidence).items()}


def appearance(tracklet):
    """Mean body histogram of a tracklet's detections (None when none was measured)."""
    hists = [p["hist"] for p in tracklet["points"] if p.get("hist")]
    if not hists:
        return None
    return [sum(h[i] for h in hists) / len(hists) for i in range(len(hists[0]))]


def similarity(a, b):
    """Histogram intersection of two normalized histograms, 0 (nothing alike) to 1; None when either is missing."""
    if a is None or b is None:
        return None
    return sum(min(x, y) for x, y in zip(a, b))


def team_paths(tracklet_list, reads, roster, min_confidence=0.65, read_bonus=400.0, max_gap_ms=8000, sizes_per_s=3.0,
               gap_cost_per_s=2.0, appearance_cost=0.0):
    """{(alliance, team or None): [tracklet id, ...]}: one path per roster team through the tracklets, built from the
    bumper reads.

    Tracklets go forward in time, so for each team the best path is a longest path over the time-ordered tracklets:
    each detection earns 1, each tracklet read as that team earns `read_bonus` per read, a tracklet read as another
    team is excluded, and a link costs the jump in box sizes plus `gap_cost_per_s` per second of gap (allowed within
    `max_gap_ms` and `sizes_per_s` box sizes per second). Teams with more reads go first; their tracklets are then
    taken. A team with no reads gets no path of its own (key None holds the alliance's best remaining path when
    exactly one team is left unread: named by elimination by the caller, never silently). With `appearance_cost`, a
    link also costs that much times how unalike the two tracklets' robot bodies look (1 - histogram intersection)."""
    label = defaultdict(lambda: defaultdict(int))
    for r in reads:
        conf = r.get("confidence")
        if isinstance(conf, (int, float)) and not isinstance(conf, bool) and conf >= min_confidence and r.get("tracklet") is not None:
            label[r["tracklet"]][r.get("team_key")] += 1
    out = {}
    for alliance in ALLIANCES:
        teams = list(roster.get(alliance) or [])
        trs = sorted((tr for tr in tracklet_list if tr["alliance"] == alliance), key=lambda tr: tr["points"][0]["t"])
        read_count = {team: sum(label[tr["id"]].get(team, 0) for tr in trs) for team in teams}
        looks = {tr["id"]: appearance(tr) for tr in trs} if appearance_cost else {}
        read_teams = [tm for tm in teams if read_count[tm] > 0]
        # The paths are found one team at a time, so an earlier team can take a later team's tracklets. With three
        # robots there are at most six orders: keep the one whose paths score highest together.
        best_total, best_paths = -math.inf, {}
        for order in permutations(read_teams):
            paths_here, total_here = _paths_in_order(trs, label, list(order) + [None], looks, read_bonus, max_gap_ms,
                                                     sizes_per_s, gap_cost_per_s, appearance_cost)
            if total_here > best_total:
                best_total, best_paths = total_here, paths_here
        for team, path in best_paths.items():
            out[(alliance, team)] = path
    return out


def _paths_in_order(trs, label, order, looks, read_bonus, max_gap_ms, sizes_per_s, gap_cost_per_s, appearance_cost):
    """team_paths for one alliance with teams linked in `order` (None last: the best path through unread
    tracklets). Returns ({team: [tracklet ids]}, total score)."""
    taken, out, total = set(), {}, 0.0
    for team in order:
        best, prev = {}, {}
        for j, tr in enumerate(trs):
            if tr["id"] in taken:
                continue
            own = label[tr["id"]]
            if team is not None and any(t != team for t in own):
                continue                     # read as another team
            if team is None and own:
                continue
            gain = len(tr["points"]) + (read_bonus * own.get(team, 0) if team else 0)
            best[j], prev[j] = gain, None
            first = tr["points"][0]
            for i in range(j):
                if i not in best:
                    continue
                last = trs[i]["points"][-1]
                gap = first["t"] - last["t"]
                if not 0 < gap <= max_gap_ms:
                    continue
                s = size(last["box"])
                d = math.dist(center(last["box"]), center(first["box"])) / s
                if d > sizes_per_s * max(1.0, gap / 1000.0):
                    continue
                sim = similarity(looks[trs[i]["id"]], looks[tr["id"]]) if appearance_cost else None
                look = appearance_cost * (1.0 - sim) if sim is not None else 0.0
                score = best[i] - d - gap_cost_per_s * gap / 1000.0 - look + gain
                if score > best[j]:
                    best[j], prev[j] = score, i
        if not best:
            continue
        j = max(best, key=best.get)
        j_end = j
        path = []
        while j is not None:
            path.append(trs[j]["id"]); j = prev[j]
        path.reverse()
        taken.update(path)
        out[team] = path
        total += best[j_end]
    return out, total


def dense_crop_plan(tracklet_list, budget=240, every_ms=3000, min_size_px=30, min_height_px=88, max_overlap=0.05):
    """[(tracklet id, t, box)] to read, at most `budget` for the match: about one crop per `every_ms` of each tracklet,
    the best `read_score` frame (else the largest box) within each window, only where the robot is seen alone and its
    box is at least `min_height_px` tall (on 1080p broadcasts shorter boxes almost never read: 3 of 100 on 2026cc qm1,
    against 25 of 195 for taller ones). When the
    windows ask for more than the budget, the best-scoring crops win, spread across tracklets. Short tracklets get at
    least one crop each, so every piece of every robot has a chance to be named."""
    boxes_at = defaultdict(list)
    for tr in tracklet_list:
        for p in tr["points"]:
            boxes_at[p["t"]].append((tr["id"], p["box"]))

    def alone(tid, t, box):
        return all(iou(box, other) <= max_overlap for oid, other in boxes_at[t] if oid != tid)

    def score(p):
        area = (p["box"][2] - p["box"][0]) * (p["box"][3] - p["box"][1])
        return (p["read_score"] if p.get("read_score") is not None else 0.0, area)

    windows = []
    for tr in tracklet_list:
        by_window = defaultdict(list)
        for p in tr["points"]:
            b = p["box"]
            if (b[2] - b[0] < min_size_px or b[3] - b[1] < max(min_size_px, min_height_px)
                    or not alone(tr["id"], p["t"], b)):
                continue
            by_window[p["t"] // every_ms].append(p)
        for w, pts in by_window.items():
            best = max(pts, key=score)
            windows.append((score(best), tr["id"], best["t"], best["box"]))
    windows.sort(key=lambda x: x[0], reverse=True)
    chosen, per_tracklet = [], defaultdict(int)
    # First pass: one crop per tracklet (its best window), then the rest by score.
    for s, tid, t, box in windows:
        if per_tracklet[tid] == 0 and len(chosen) < budget:
            chosen.append((tid, t, box)); per_tracklet[tid] += 1
    taken = set((tid, t) for tid, t, _ in chosen)
    for s, tid, t, box in windows:
        if len(chosen) >= budget:
            break
        if (tid, t) not in taken:
            chosen.append((tid, t, box)); taken.add((tid, t))
    return sorted(chosen, key=lambda x: x[1])



def name_by_paths(tracklet_list, reads, roster, cover_share=0.8, **path_options):
    """{tracklet id: {'team_key', 'by'}} from team_paths: each read team's best path through the tracklets that agree
    with its reads ('reads' for a tracklet a read named, 'path' for one the path carried it through). When exactly one
    team of an alliance has no read, a remaining tracklet is that team ('elimination') only while the two other teams
    are each named by reads on another tracklet for `cover_share` of its time (a path through leftovers, or names
    carried along paths, proved wrong: 0 of 5 held-out reads on 2026cc qm1)."""
    paths = team_paths(tracklet_list, reads, roster, **path_options)
    read_tracklets = defaultdict(set)
    for r in reads:
        if r.get("team_key") and (r.get("confidence") or 0) >= 0.65:
            read_tracklets[r["tracklet"]].add(r["team_key"])
    names = {}
    for alliance in ALLIANCES:
        teams = list(roster.get(alliance) or [])
        unread = [t for t in teams if (alliance, t) not in paths]
        for team in teams:
            path = paths.get((alliance, team))
            if path:
                for tid in path:
                    names[tid] = {"team_key": team, "by": "reads" if team in read_tracklets.get(tid, ()) else "path"}
        if len(unread) != 1:
            continue
        # Elimination per tracklet, never along a path: an unnamed tracklet is the unread team only while the
        # alliance's two other teams are each named by reads on another tracklet for `cover_share` of its time.
        span = {tr["id"]: (tr["points"][0]["t"], tr["points"][-1]["t"]) for tr in tracklet_list}
        same = [tr for tr in tracklet_list if tr["alliance"] == alliance]
        for tr in same:
            if tr["id"] in names or read_tracklets.get(tr["id"]):
                continue
            s0, s1 = span[tr["id"]]
            length = max(1, s1 - s0)
            held = defaultdict(int)
            for other in same:
                team = (names.get(other["id"]) or {}).get("team_key")
                if team and team != unread[0] and names[other["id"]]["by"] == "reads":
                    o0, o1 = span[other["id"]]
                    held[team] += max(0, min(s1, o1) - max(s0, o0))
            if len([t for t, ms in held.items() if ms >= cover_share * length]) == 2:
                names[tr["id"]] = {"team_key": unread[0], "by": "elimination"}
    return names


def name_from_reads(tracklet_list, reads, roster, min_confidence=0.65):
    """Credit only tracklets with an unambiguous roster bumper read.

    Long path links and third-robot elimination are useful diagnostics but failed held-out Chezy reads. A conflicting
    read or simultaneous tracklet claiming the same team stays unnamed unless one claim has strictly more reads.
    This deliberately sacrifices coverage rather than assigning another robot's fuel to a named team.
    """
    by_id = {tr["id"]: tr for tr in tracklet_list}
    votes = defaultdict(lambda: defaultdict(int))
    for read in reads:
        tid, team, conf = read.get("tracklet"), read.get("team_key"), read.get("confidence")
        tr = by_id.get(tid)
        if (tr and isinstance(conf, (int, float)) and not isinstance(conf, bool) and conf >= min_confidence
                and team in (roster.get(tr["alliance"]) or [])):
            votes[tid][team] += 1

    names = {}
    for tid, counts in votes.items():
        if len(counts) == 1:
            team, read_count = next(iter(counts.items()))
            names[tid] = {"team_key": team, "by": "reads", "reads": read_count}

    # Two observed boxes at the same time cannot both be the named robot. Ties are unknown on both sides.
    rejected = set()
    for alliance in ALLIANCES:
        for team in roster.get(alliance) or []:
            tids = [tid for tid, name in names.items() if name["team_key"] == team and by_id[tid]["alliance"] == alliance]
            for i, a in enumerate(tids):
                for b in tids[i + 1:]:
                    pa, pb = by_id[a]["points"], by_id[b]["points"]
                    if min(pa[-1]["t"], pb[-1]["t"]) <= max(pa[0]["t"], pb[0]["t"]):
                        continue
                    na, nb = names[a]["reads"], names[b]["reads"]
                    if na <= nb:
                        rejected.add(a)
                    if nb <= na:
                        rejected.add(b)
    for tid in rejected:
        names.pop(tid, None)
    return names


def body_hist(frame, box):
    """Body appearance above the bumper band, including brightness to distinguish white and black chassis.

    Hue and saturation alone collapse white and black into the same bins; a 12x4x4 HSV histogram preserves that
    evidence for conservative association. This is a continuity gate, never a team-number classifier.
    """
    import cv2
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    crop = frame[max(0, y1):max(0, y1 + int((y2 - y1) * 0.65)), max(0, x1):max(0, x2)]
    if crop.size == 0:
        return None
    hist = cv2.calcHist([cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)], [0, 1, 2], None,
                       [12, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
    total = float(hist.sum()) or 1.0
    return [round(float(v) / total, 4) for v in hist]

PANEL_ID_STEP = 10_000_000       # above any tracklet id in a match
PANEL_CROP_EVERY_MS = 500        # a panel name needs three reads; 1500 left short tracklets unnamed


def detect_samples(model, video_path, window_ms=None, roi=None, stride=4, conf=0.15, imgsz=1920):
    """[(video ms, [(box px, confidence, alliance, read_score)])] for every `stride`-th frame inside `window_ms`, run
    on the calibration's field area (`roi` [x0, y0, x1, y1] as fractions; the whole frame without one) at full
    resolution, where broadcast robots are 40-150 px."""
    import cv2
    names = {int(k): str(v).lower() for k, v in model.names.items()}
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    x0, y0, x1, y1 = (0.0, 0.0, 1.0, 1.0) if not roi else roi
    X0, Y0, X1, Y1 = int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H)
    start = int((window_ms or [0, 0])[0] * fps / 1000) if window_ms else 0
    end = int(window_ms[1] * fps / 1000) if window_ms else None
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    samples, i = [], start
    while end is None or i <= end:
        if not cap.grab():
            break
        if (i - start) % stride == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            res = model.predict(frame[Y0:Y1, X0:X1], conf=conf, imgsz=imgsz, verbose=False)[0]
            dets = []
            for b in res.boxes:
                cls = names[int(b.cls)]
                alliance = {"robot_red": "red", "robot_blue": "blue"}.get(cls)
                if not alliance:
                    continue
                bx = b.xyxy[0].tolist()
                full = (bx[0] + X0, bx[1] + Y0, bx[2] + X0, bx[3] + Y0)
                dets.append((full, float(b.conf), alliance, read_score(frame, full), body_hist(frame, full)))
            samples.append((int(round(i * 1000 / fps)), dets))
        i += 1
    cap.release()
    return samples, (W, H)


def read_plan(video_path, plan, roster, qwen_url, qwen_token, batch=12, width=448, timeout=900,
              endpoint="read-bumper-grid", progress=None):
    """Qwen bumper reads for plan [(tracklet id, video ms, box, alliance)], by default as lettered grids of `batch`
    crops (/read-bumper-grid: about 1.3 s a crop on the Spark against 14 s for one-by-one reads with evidence):
    [{'tracklet', 't', 'team_key', 'confidence'}] (team_key None when unreadable). Roster-limited afterwards by the
    service; an outage or malformed answer leaves those crops unread, never guessed."""
    import base64
    import hashlib
    import cv2
    import numpy
    import requests
    from team_identity import crop_robot
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # One forward pass over the planned frames: a seek per crop decodes from the previous keyframe each time (2026cc
    # qm3, 226 crops: 140 s of seeks against 25 s for the pass, with identical crops).
    wanted = {}
    for k, (_, t, _, _) in enumerate(plan):
        wanted.setdefault(int(round(t * fps / 1000)), []).append(k)
    got_by_k, i, last = {}, 0, max(wanted, default=-1)
    while i <= last and cap.grab():
        if i in wanted:
            ok, frame = cap.retrieve()
            if ok:
                frame[:round(frame.shape[0] * .25)] = 0
            for k in wanted[i]:
                got_by_k[k] = crop_robot(frame, plan[k][2]) if ok else None
        i += 1
    cap.release()
    crops = []
    for k, (tid, t, _, alliance) in enumerate(plan):
        got = got_by_k.get(k)
        if not got:
            continue
        img = cv2.imdecode(numpy.frombuffer(got[0], "uint8"), cv2.IMREAD_COLOR)
        scale = min(1.0, width / img.shape[1])
        if scale < 1:
            img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        crops.append((tid, t, alliance, cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()))
    reads = []
    for i in range(0, len(crops), batch):
        part = crops[i:i + batch]
        payload = [{"crop_id": f"{tid}|{t}", "track_key": str(tid), "alliance": alliance,
                    "candidate_team_keys": sorted(roster.get(alliance) or [])[:3],
                    "jpeg_base64": base64.b64encode(jpg).decode("ascii")} for tid, t, alliance, jpg in part]
        source_hashes = {f"{tid}|{t}": hashlib.sha256(jpg).hexdigest() for tid, t, _, jpg in part}
        try:
            r = requests.post(f"{qwen_url.rstrip('/')}/{endpoint}", headers={"Authorization": f"Bearer {qwen_token}"},
                              json={"crops": payload}, timeout=timeout)
            r.raise_for_status()
            for read in r.json().get("reads", []):
                tid, t = read["crop_id"].split("|")
                reads.append({"tracklet": int(tid), "t": int(t), "team_key": read.get("team_key"),
                              "confidence": read.get("confidence"), "crop_sha256": source_hashes.get(read["crop_id"]),
                              "identity_source": "bumper_read"})
        except Exception:  # noqa: BLE001 - an unread crop stays unread
            continue
        finally:
            if progress:
                progress(i + len(part), len(crops))
    return reads


def closeup_panel_reads(model, video_path, roster, qwen_url, qwen_token, panels, wide_tracklets, window_ms, size_wh,
                        stride, conf, imgsz, budget=800):
    """(rows, names, stats): literal reads from the broadcast's close-up panels, carried onto wide tracklets."""
    from closeup_names import carried_reads
    area = [min(p["roi"][0] for p in panels), min(p["roi"][1] for p in panels),
            max(p["roi"][2] for p in panels), max(p["roi"][3] for p in panels)]
    samples, _ = detect_samples(model, video_path, window_ms, area, stride, conf, imgsz)
    width, height = size_wh
    panel_tracks = []
    for number, panel in enumerate(panels, start=1):
        x0, y0, x1, y1 = panel["roi"]

        def inside(det):
            box = det[0]
            return x0 * width <= (box[0] + box[2]) / 2 < x1 * width and y0 * height <= (box[1] + box[3]) / 2 < y1 * height
        # Each panel is its own camera. Offset ids so a read names one tracklet across panels and the wide view.
        panel_tracks += [{**track, "id": track["id"] + number * PANEL_ID_STEP, "panel": panel["name"]}
                         for track in tracklets([(t, [d for d in dets if inside(d)]) for t, dets in samples])]
    alliance_of = {track["id"]: track["alliance"] for track in panel_tracks}
    plan = [(tid, t, box, alliance_of[tid])
            for tid, t, box in dense_crop_plan(panel_tracks, budget=budget, every_ms=PANEL_CROP_EVERY_MS)]
    reads = read_plan(video_path, plan, roster, qwen_url, qwen_token)
    rows, names = carried_reads(panel_tracks, reads, wide_tracklets, panels, size_wh)
    legible = {"tracklets": panel_tracks,
               "observations": [{**r, "identity_source": "bumper_read"} for r in reads if r.get("team_key")]}
    return rows, names, {"panel_tracklets": len(panel_tracks), "crops": len(plan),
                         "readable": sum(1 for r in reads if r.get("team_key")), "wide_tracklets_named": len(names)}, legible


def grow_from_panels(gallery_path, event_key, match_key, roster, legible, video_path, stats):
    """Add legible panel bumpers to the gallery before body recognition, so a new camera's first match can use it. On a
    2026dal view with no setup, this took named HUB entries from 9% to 43%. A failure costs only those names."""
    if not gallery_path or not event_key or not legible["observations"]:
        return
    try:
        from grow_gallery import grow
        grow(gallery_path, {**legible, "event_key": event_key, "roster": roster}, str(video_path), match_key or f"{event_key}_panels")
    except Exception as error:  # noqa: BLE001 - the gallery is optional evidence, never a reason to fail scouting
        stats["gallery_error"] = type(error).__name__


def identify_robots(model, video_path, roster, qwen_url, qwen_token, window_ms=None, roi=None, budget=300,
                    stride=4, conf=0.15, imgsz=1920, min_unnamed_ms=0, first_round_share=1.0, appearance_cost=10.0,
                    gallery_path=None, event_key=None, panels=None, inputs_folder=None, run_id=None, match_key=None):
    """Robots for a whole match, named from Qwen bumper reads: {'robots': [{'team_key', 'alliance', 'by', 'reads',
    'points': [{'t' (video ms), 'box' (px), 'tracklet'}]}], 'size': (W, H), 'stats': {...}}.

    One entry per named team (all its tracklets), plus unnamed tracklets of at least `min_unnamed_ms` (team unknown).
    Nothing here is labeled by hand: boxes come from the detector, names from the reads and the linking rules."""
    import time
    began = time.time()
    samples, size_wh = detect_samples(model, video_path, window_ms, roi, stride, conf, imgsz)
    t_detect = time.time() - began
    tls = tracklets(samples)
    alliance_of = {tr["id"]: tr["alliance"] for tr in tls}
    can_read = bool(roster.get("red") and roster.get("blue"))
    # Round 1 spreads the budget over every tracklet. A second round on tracklets still not named by a read is
    # available (first_round_share < 1) but spent less in total on 2026cc qm1 (195 crops, 18 reads, against 300 and
    # 44 in one round), so one round is the default.
    first = int(budget * first_round_share)
    plan = [(tid, t, box, alliance_of[tid]) for tid, t, box in dense_crop_plan(tls, budget=first)]
    reads = read_plan(video_path, plan, roster, qwen_url, qwen_token) if can_read else []
    if can_read and budget > first:
        read_ids = {r["tracklet"] for r in reads if r.get("team_key")}
        tried = {(tid, t) for tid, t, _, _ in plan}
        pending = [tr for tr in tls if tr["id"] not in read_ids]
        second = [(tid, t, box, alliance_of[tid]) for tid, t, box in dense_crop_plan(pending, budget=budget - first,
                                                                                    every_ms=1000, min_height_px=75)
                  if (tid, t) not in tried]
        reads += read_plan(video_path, second, roster, qwen_url, qwen_token)
        plan += second
    # Held-out Chezy reads found path-carried names unreliable (14/27 correct across qm1-qm3); elimination had no
    # supporting bumper read. Only direct, nonconflicting reads can turn a robot track into a team fuel credit.
    carried, carried_names, closeup_stats = [], {}, None
    if panels and can_read:
        carried, carried_names, closeup_stats, legible = closeup_panel_reads(model, video_path, roster, qwen_url, qwen_token,
                                                                             panels, tls, window_ms, size_wh, stride, conf, imgsz)
        grow_from_panels(gallery_path, event_key, match_key, roster, legible, video_path, closeup_stats)
    earlier_stats = None
    if can_read:
        from closeup_names import earlier_reads, merge_earlier
        earlier, runs = earlier_reads(inputs_folder, run_id, tls)
        carried, carried_names = merge_earlier(carried, carried_names, earlier)
        earlier_stats = {"runs": runs, "reads": len(earlier), "tracklets_named": len(carried_names)}
    names = name_from_reads(tls, reads + carried, roster)
    robots = {}
    for tr in tls:
        info = names.get(tr["id"]) or {}
        team = info.get("team_key")
        if team:
            entry = robots.setdefault(team, {"team_key": team, "alliance": tr["alliance"], "by": set(), "reads": 0,
                                             "points": []})
            entry["by"].add(info.get("by"))
            entry["reads"] += sum(1 for r in reads if r["tracklet"] == tr["id"] and r.get("team_key") == team)
            entry["points"].extend({"t": p["t"], "box": p["box"], "tracklet": tr["id"]} for p in tr["points"])
        elif tr["points"][-1]["t"] - tr["points"][0]["t"] >= min_unnamed_ms:
            robots[f"unnamed-{tr['id']}"] = {"team_key": None, "alliance": tr["alliance"], "by": set(), "reads": 0,
                                             "points": [{"t": p["t"], "box": p["box"], "tracklet": tr["id"]}
                                                        for p in tr["points"]]}
    out = []
    for entry in robots.values():
        entry["points"].sort(key=lambda p: p["t"])
        entry["by"] = sorted(b for b in entry["by"] if b)
        out.append(entry)
    body_status = 'not_configured'
    body_reads, gallery_digest, identity_inputs = [], None, None
    if gallery_path and event_key:
        from identity_gallery import IdentityGallery
        from body_identity_client import identified_robots
        templates = IdentityGallery(event_key, gallery_path).templates(roster)
        if Path(gallery_path).is_file():
            gallery_digest = hashlib.sha256(Path(gallery_path).read_bytes()).hexdigest()
        from body_identity_client import complete_alliances
        body_status = 'missing_reference_reads' if templates['missing'] else 'complete_reference_gallery'
        if any(complete_alliances(roster, templates).values()):
            out, body_reads = identified_robots(video_path, tls, roster, templates, qwen_url, qwen_token,
                                                reads + carried, carried_names)
            body_status = 'bounded_body_observations'
            # Kept privately on the Spark for source-first refinement after counting (identity_refinement.py).
            identity_inputs = {'tracklets': tls, 'observations': body_reads}
        else:
            # An incomplete gallery still keeps the literal reads: they are how it grows (grow_gallery.py).
            body_reads = [{**row, 'identity_source': 'bumper_read'} for row in reads + carried if row.get('team_key')]
            identity_inputs = {'tracklets': tls, 'observations': body_reads}
    total = max(1, len(samples))
    teams = {entry['team_key'] for entry in out if entry['team_key']}
    coverage = {team: round(len({point['t'] for entry in out if entry['team_key'] == team
                               for point in entry['points']}) / total, 3) for team in teams}
    return {"robots": out, "size": size_wh, "reads": reads, "identity_inputs": identity_inputs,
            "tracklets": [{"id": tr["id"], "alliance": tr["alliance"],
                           "points": [{"t": p["t"], "box": p["box"]} for p in tr["points"]]} for tr in tls],
            "stats": {"frames": len(samples), "tracklets": len(tls), "crops": len(plan),
                      "readable": sum(1 for r in reads if r.get("team_key")), "detect_s": round(t_detect, 1),
                      "total_s": round(time.time() - began, 1),
                      "body_identity": body_status, "gallery_sha256": gallery_digest, "closeups": closeup_stats, "earlier_reads": earlier_stats,
                      "body_views": sum(row.get("identity_source") == "body_gallery_v1" for row in body_reads),
                      "recognized_body_views": sum(bool(row.get("team_key")) and row.get("identity_source") == "body_gallery_v1"
                                                   for row in body_reads), "coverage": coverage}}
