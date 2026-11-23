"""Source-first identity refinement after a ball count (private Spark step; nothing here reaches the Hub as input).

The counter requires a continuous observed ball path from a unique robot body before considering team credit.
When that source has no supported name, it remains unattributed. Source-directed views can provide identity
evidence without guessing a name or distributing alliance totals.

This step adds views at the proven launches of those fragments, through the same crop code and private model, keeps
only corroborated ones (`body_identity.corroborated`), and re-assembles identities with the unchanged rules
(`body_identity.assemble_robots`: verified physical association, +-2.5 s anchor bound, literal vetoes, duplicate-claim
removal). Only tracklets whose labels change are replaced in the robot tracks the counter re-scores. Scouting keeps
its identity inputs (tracklets with appearance histograms, association metadata and every view) privately on the
Spark for this; they are never uploaded."""
from __future__ import annotations

import gzip
import json
import math
import os
import sys
from collections import Counter, defaultdict

from body_identity import (METHOD, assemble_robots, bounded_identities, corroborated, distinct_recognition_frames,
                           launch_plan)

VERSION = 1
QUERY_PLAN = 'source_first_launch_v2'
EXIT_QUERY_PLAN = 'robot_exit_source_v1'
EXIT_FIRST_VIEWS = 24
EXIT_FOLLOWUP_VIEWS = 12
MIN_CLOCK_SUPPORT = 0.8
CLOCK_MATCH_MS = 20              # same point across clocks, as in segment_tracklets
SECOND_ROUND_VIEWS = 3          # extra views per fragment that has exactly one recognition after the first round
# Literal pass: compact bumper reads every half second along unnamed exit sources, most exits first. On 2026cc qm3
# the top unnamed sources were legible (581, 359, 6665) but no launch-time body view named them.
EXIT_READ_SPACING_MS = 500
EXIT_READ_CROPS = 300
EXIT_READS_PER_TRACKLET = 30
EXIT_READ_REACH_MS = 2500        # only where a name could reach an exit (the anchor bound)
# Literal precedence: on an unnamed exit source, at least this many distinct frames of literal bumper reads that all
# agree (and no literal read of another team) outrank body-gallery answers naming another team on that fragment only.
# On 2026cc qm3 such answers vetoed fragments whose bumpers read 581 x10, 694 x4 and 359 x2.
LITERAL_PRECEDENCE_FRAMES = 3
# Appearance: with the two-places rule, stored colour histograms named held-out segments 72 of 72 times at this margin
# on 2026cc qm1-qm3, abstaining on most; at 0.05 they were wrong 7 times in 123.
APPEARANCE_METHOD = 'appearance_match_v1'
APPEARANCE_MARGIN = 0.12
APPEARANCE_MIN_POINTS = 5
NEAR_LAUNCH_MS = (-600, 0, 600)  # second-round candidates: each launch and the observed points around it


def save_inputs(path, run_id, event_key, size, roster, tracklets, observations):
    """Write the scouting run's identity inputs (mode 600). Points keep only what the identity rules read."""
    data = {'version': VERSION, 'run_id': run_id, 'event_key': event_key, 'size': list(size),
            'roster': {k: list(roster.get(k) or []) for k in ('red', 'blue')},
            'tracklets': [{'id': tr['id'], 'alliance': tr['alliance'], 'association': tr.get('association'),
                           'points': [{'t': p['t'], 'box': [float(v) for v in p['box']], 'hist': p.get('hist')}
                                      for p in tr['points']]} for tr in tracklets],
            'observations': [{k: row.get(k) for k in ('tracklet', 't', 'team_key', 'identity_source', 'confidence',
                                                     'crop_sha256', 'read_record_sha256', 'via') if k in row}
                             for row in observations]}
    tmp = f'{path}.tmp'
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb') as out:
        out.write(json.dumps(data).encode())
    os.replace(tmp, path)


def load_inputs(path):
    with gzip.open(path, 'rb') as f:
        data = json.loads(f.read())
    if data.get('version') != VERSION:
        raise ValueError(f'unsupported identity inputs version {data.get("version")}')
    return data


def _supported(evidence):
    """Direct or gallery evidence only: appearance never trains on its own guesses."""
    return evidence == 'direct_reads' or (isinstance(evidence, dict) and evidence.get('method') == METHOD)


def _px_key(box):
    return tuple(int(round(v)) for v in box)


def clock_link(tracklets, robot_tracks, size, sample_every=10):
    """(offset ms, support): robot-track ms minus tracklet ms, from points whose boxes agree to the pixel, and the
    share of sampled robot points that agree at that offset. The same recording is required upstream; this proves the
    two clocks and box frames line up instead of trusting a recorded offset."""
    w, h = size
    at = defaultdict(list)
    for tr in tracklets:
        for p in tr['points']:
            at[_px_key(p['box'])].append(p['t'])
    votes, sampled = Counter(), 0
    samples = []
    for robot in robot_tracks:
        for p in (robot.get('trajectory') or [])[::sample_every]:
            box = p.get('box')
            if not box or len(box) != 4 or not isinstance(p.get('t'), (int, float)):
                continue
            sampled += 1
            key = _px_key((box[0] * w, box[1] * h, box[2] * w, box[3] * h))
            offsets = {round((p['t'] - t) / 10) * 10 for t in at.get(key, [])}
            samples.append(offsets)
            for o in offsets:
                votes[o] += 1
    if not votes:
        return None, 0.0
    offset = votes.most_common(1)[0][0]
    agree = sum(any(abs(o - offset) <= 20 for o in offsets) for offsets in samples)
    return offset, agree / max(1, sampled)


def segment_tracklets(robot_tracks, tracklets, offset, size):
    """{robot track key: tracklet id} for robot segments whose points are that tracklet's points (same box to the
    pixel, within 20 ms at the proven offset). A segment matching no or several tracklets is left unmapped."""
    w, h = size
    at = defaultdict(list)
    for tr in tracklets:
        for p in tr['points']:
            at[_px_key(p['box'])].append((p['t'], tr['id']))
    out = {}
    for robot in robot_tracks:
        found = Counter()
        for p in robot.get('trajectory') or []:
            box = p.get('box')
            if not box or len(box) != 4:
                continue
            key = _px_key((box[0] * w, box[1] * h, box[2] * w, box[3] * h))
            for t, tid in at.get(key, []):
                if abs(p['t'] - offset - t) <= CLOCK_MATCH_MS:
                    found[tid] += 1
        if len(found) == 1 and robot.get('track_key'):
            out[robot['track_key']] = next(iter(found))
    return out


def _single_recognitions(observations, new, done):
    """Tracklets (not yet corroborated) with exactly one distinct recognized frame and team, without any
    earlier anchor or veto naming another team."""
    earlier = defaultdict(set)
    for row in observations:
        if row.get('team_key'):
            earlier[row['tracklet']].add(row['team_key'])
    seen, frames = defaultdict(set), defaultdict(set)
    for row in new:
        if row.get('team_key'):
            seen[row['tracklet']].add(row['team_key'])
    for tid, t in distinct_recognition_frames(new, viewed=observations):
        frames[tid].add(t)
    return {tid for tid, teams in seen.items()
            if tid not in done and len(teams) == 1 and len(frames[tid]) == 1 and earlier[tid] <= teams}


def _labels(tracklets, observations, roster):
    labeled, _ = bounded_identities(tracklets, observations, roster)
    return {tid: [p['team_key'] for p in points] for tid, points in labeled.items()}


def literal_read_plan(tracklets, launches, viewed=(), spacing_ms=EXIT_READ_SPACING_MS, limit=EXIT_READ_CROPS,
                      per_tracklet=EXIT_READS_PER_TRACKLET, reach_ms=EXIT_READ_REACH_MS):
    """[(tracklet id, t, box, alliance)] for compact literal bumper reads: observed points at least `spacing_ms`
    apart within `reach_ms` of one of the tracklet's unnamed exits, tracklets with more exits first, at most
    `per_tracklet` each (spread evenly) and `limit` in all. Bumper digits are read, never inferred; a name still needs
    two agreeing distinct frames and no contrary evidence (`body_identity.corroborated`)."""
    by_id = {track['id']: track for track in tracklets}
    exits = defaultdict(list)
    for tid, t in launches:
        exits[tid].append(t)
    out, seen = [], set(viewed)
    for tid in sorted(exits, key=lambda tid: (-len(exits[tid]), str(tid))):
        track = by_id.get(tid)
        if track is None:
            continue
        chosen, last = [], None
        for point in track['points']:
            if last is not None and point['t'] - last < spacing_ms:
                continue
            if (tid, point['t']) in seen or min(abs(point['t'] - t) for t in exits[tid]) > reach_ms:
                continue
            chosen.append((tid, point['t'], point['box'], track['alliance']))
            last = point['t']
        if len(chosen) > per_tracklet:
            keep = {round(i * (len(chosen) - 1) / max(1, per_tracklet - 1)) for i in range(per_tracklet)}
            chosen = [row for i, row in enumerate(chosen) if i in keep]
        out += chosen[:max(0, limit - len(out))]
        if len(out) >= limit:
            break
    return out


def literal_precedence(observations, new, eligible, min_frames=LITERAL_PRECEDENCE_FRAMES):
    """(observations, new, {tracklet: team}, superseded count). On `eligible` tracklets whose literal bumper reads
    (earlier and new) name one team on at least `min_frames` distinct frames, body-gallery answers naming another team
    are dropped. Literal vetoes and every other row stay, so a veto naming another team still blocks it."""
    literal = [r for r in observations + new
               if r.get('identity_source') == 'bumper_read' and r.get('team_key') and r.get('tracklet') in eligible]
    teams, frames = defaultdict(set), defaultdict(set)
    for row in literal:
        teams[row['tracklet']].add(row['team_key'])
    for tid, t in distinct_recognition_frames(literal, viewed=()):
        frames[tid].add(t)
    strong = {tid: next(iter(names)) for tid, names in teams.items() if len(names) == 1 and len(frames[tid]) >= min_frames}

    def superseded(row):
        return (row.get('tracklet') in strong and row.get('identity_source') == METHOD and row.get('team_key')
                and row['team_key'] != strong[row['tracklet']])

    dropped = sum(1 for row in observations + new if superseded(row))
    return ([r for r in observations if not superseded(r)], [r for r in new if not superseded(r)], strong, dropped)


def _bounded_exit_plan(plan, launches, limit):
    """Favor physical sources with more observed exits, then give sources turns.

    Give each selected source two opportunities to corroborate before any gets
    a third view. No team name or score influences this ordering.
    """
    frequency = Counter(tid for tid, _ in launches)
    groups = defaultdict(list)
    for row in plan:
        groups[row[0]].append(row)
    ordered = sorted(groups, key=lambda tid: (-frequency[tid], str(tid)))
    out = []
    for start, end in ((0, 2), (2, 3)):
        for tid in ordered:
            for row in groups[tid][start:end]:
                out.append(row)
                if len(out) == limit:
                    return out
    return out


def refine(inputs, attributed, robot_tracks, templates, view_fn, read_fn=None, video_size=None, source_exits=None):
    """(robot tracks, stats). `attributed`: the count's entries (counter clock). `view_fn(plan_rows)` returns body
    observations for [(tracklet, t, box px in the scouting frame, alliance)]; `read_fn(plan_rows)` bumper reads
    (rows with 'tracklet', 't', 'team_key', 'confidence', 'crop_sha256'). Robot tracks come back unchanged whenever the
    clock is unproven, nothing is proven-unnamed, or no view is corroborated."""
    tracklets, roster, size = inputs['tracklets'], inputs['roster'], inputs['size']
    observations = inputs['observations']
    exit_mode = source_exits is not None
    stats = {'query_plan': EXIT_QUERY_PLAN if exit_mode else QUERY_PLAN, 'status': 'unchanged'}
    offset, support = clock_link(tracklets, robot_tracks, size)
    stats.update(clock_offset_ms=offset, clock_support=round(support, 3))
    if offset is None or support < MIN_CLOCK_SUPPORT:
        stats['status'] = 'clock_unproven'
        return robot_tracks, stats
    mapping = segment_tracklets(robot_tracks, tracklets, offset, size)
    unnamed = {r['track_key'] for r in robot_tracks if not r.get('team_key')}
    if exit_mode:
        launches = [(mapping[e['track_key']], e['source_ms'] - offset) for e in source_exits
                    if e.get('method') == 'observed_body_exit_v1' and e.get('source_continuous') is True
                    and e.get('identity') == 'unknown' and not e.get('team_key')
                    and e.get('track_key') in unnamed and e.get('track_key') in mapping
                    and isinstance(e.get('source_ms'), (int, float)) and not isinstance(e['source_ms'], bool)
                    and math.isfinite(e['source_ms'])]
        stats['unnamed_source_exits'] = len(launches)
        stats['view_budget'] = EXIT_FIRST_VIEWS + EXIT_FOLLOWUP_VIEWS
    else:
        launches = [(mapping[e['track_key']], e['launch_ms'] - offset) for e in attributed
                    if e.get('attribution') == 'same_track' and not e.get('team_key') and e.get('track_key') in unnamed
                    and e.get('track_key') in mapping and isinstance(e.get('launch_ms'), (int, float))]
        stats['unnamed_source_entries'] = len(launches)
    plan = launch_plan(tracklets, launches, viewed={(row['tracklet'], row['t']) for row in observations})
    if exit_mode:
        stats['eligible_views'] = len(plan)
        plan = _bounded_exit_plan(plan, launches, EXIT_FIRST_VIEWS)
        stats['views_limited'] = stats['eligible_views'] - len(plan)
    stats['views'] = len(plan)
    literal = exit_mode and read_fn is not None
    if not plan and not literal:
        return robot_tracks, stats
    def query(rows):
        views = view_fn(rows)
        reads = read_fn(rows) if read_fn else []
        return views, reads, list(views) + [{**row, 'identity_source': 'bumper_read'} for row in reads if row.get('team_key')]

    views, reads, new = query(plan) if plan else ([], [], [])
    stats['recognized_body_views'] = sum(bool(v.get('team_key')) for v in views)
    stats['readable_reads'] = sum(bool(r.get('team_key')) for r in reads)
    stats['first_round_recognized_frames'] = len(distinct_recognition_frames(new, viewed=observations))
    kept = corroborated(observations, new)
    stats['first_round_corroborated_tracklets'] = len({row['tracklet'] for row in kept})
    # Second round: a fragment with exactly one recognition (and no contradiction) gets a few more views, so a true
    # identity can reach the two agreeing views corroboration needs; unrecognized fragments cost nothing more.
    singles = _single_recognitions(observations, new, {row['tracklet'] for row in kept})
    if singles:
        candidates = [(tid, t + d) for tid, t in launches if tid in singles for d in NEAR_LAUNCH_MS]
        viewed = {(row['tracklet'], row['t']) for row in observations} | {(row[0], row[1]) for row in plan}
        plan2 = launch_plan(tracklets, candidates, viewed=viewed, max_per_tracklet=SECOND_ROUND_VIEWS)
        if exit_mode:
            eligible = len(plan2)
            plan2 = _bounded_exit_plan(plan2, launches, EXIT_FOLLOWUP_VIEWS)
            stats['second_round_views_limited'] = eligible - len(plan2)
        stats.update(second_round_tracklets=len(singles), second_round_views=len(plan2))
        if plan2:
            views2, reads2, new2 = query(plan2)
            stats['second_round_recognized_body_views'] = sum(bool(v.get('team_key')) for v in views2)
            stats['second_round_readable_reads'] = sum(bool(r.get('team_key')) for r in reads2)
            stats['second_round_recognized_frames'] = len(distinct_recognition_frames(new2, viewed=observations + new))
            new += new2
            kept = corroborated(observations, new)
    if literal:
        viewed = ({(row['tracklet'], row['t']) for row in observations} | {(row[0], row[1]) for row in plan}
                  | {(row['tracklet'], row['t']) for row in new})
        done = {row['tracklet'] for row in kept}
        rows = literal_read_plan(tracklets, [(tid, t) for tid, t in launches if tid not in done], viewed)
        stats['literal_read_views'] = len(rows)
        if rows:
            literal_reads = read_fn(rows)
            stats['literal_readable_reads'] = sum(bool(r.get('team_key')) for r in literal_reads)
            new += [{**row, 'identity_source': 'bumper_read'} for row in literal_reads if row.get('team_key')]
        observations, new, strong, dropped = literal_precedence(observations, new, {tid for tid, _ in launches} - done)
        stats.update(literal_precedence_tracklets=len(strong), superseded_body_answers=dropped)
        kept = corroborated(observations, new)
    stats['corroborated_tracklets'] = len({row['tracklet'] for row in kept})
    if not kept:
        stats['status'] = 'no_corroborated_views'
        return robot_tracks, stats
    before, after = _labels(tracklets, inputs['observations'], roster), _labels(tracklets, observations + kept, roster)
    changed = {tid for tid in after if after[tid] != before.get(tid)}
    stats['refined_tracklets'] = len(changed)
    if not changed:
        return robot_tracks, stats
    w, h = size
    # The clock link is rounded to 10 ms, so a rebuilt point keeps its original measured point (time, box, x, y): a
    # refinement names the body and must not nudge it, or observed exits on non-round offsets flip.
    measured = defaultdict(list)
    for r in robot_tracks:
        for p in r.get('trajectory') or []:
            if p.get('box') and len(p['box']) == 4:
                measured[_px_key((p['box'][0] * w, p['box'][1] * h, p['box'][2] * w, p['box'][3] * h))].append(p)
    replaced = defaultdict(list)
    for r in robot_tracks:
        if mapping.get(r.get('track_key')) in changed:
            replaced[mapping[r['track_key']]].append(r['track_key'])
    kept_tracks = [r for r in robot_tracks if mapping.get(r.get('track_key')) not in changed]
    segments = assemble_robots([tr for tr in tracklets if tr['id'] in changed], observations + kept, roster,
                               templates['references'])
    for i, segment in enumerate(segments):
        evidence = segment.get('identity_evidence')
        trajectory = []
        for p in segment['points']:
            t = p['t'] + offset
            near = [o for o in measured.get(_px_key(p['box']), []) if abs(o['t'] - t) <= CLOCK_MATCH_MS]
            trajectory.append(dict(min(near, key=lambda o: abs(o['t'] - t))) if near else
                              {'t': t, 'box': [p['box'][0] / w, p['box'][1] / h, p['box'][2] / w, p['box'][3] / h]})
        kept_tracks.append({'track_key': f'refined:{p["tracklet"]}:{i}', 'team_key': segment['team_key'],
                            'alliance': segment['alliance'],
                            'identity_evidence': {**evidence, 'query_plan': stats['query_plan']} if evidence else None,
                            'source_track_keys': replaced[p['tracklet']], 'trajectory': trajectory})
    stats['status'] = 'refined'
    return kept_tracks, stats


def _segment_look(robot, tracklet, offset):
    """The average appearance histogram of a robot segment, from its tracklet's stored points."""
    times = sorted(p['t'] - offset for p in robot.get('trajectory') or [] if p.get('t') is not None)
    if not times:
        return None
    hists = [p['hist'] for p in tracklet['points'] if times[0] - CLOCK_MATCH_MS <= p['t'] <= times[-1] + CLOCK_MATCH_MS
             and len(p.get('hist') or []) == 192]
    if len(hists) < APPEARANCE_MIN_POINTS:
        return None
    mean = [sum(values) / len(hists) for values in zip(*hists)]
    norm = math.sqrt(sum(v * v for v in mean))
    return [v / norm for v in mean] if norm else None


def _span(robot):
    times = [p['t'] for p in robot.get('trajectory') or [] if p.get('t') is not None]
    return (min(times), max(times)) if times else None


def appearance_names(inputs, robot_tracks, hidden=()):
    """(robot tracks, stats). Names an unnamed segment when it looks like one team's named segments of its alliance by
    a clear margin over every other team. A team already named on another body at the same moment is ruled out: one
    robot cannot be in two places. `hidden` segments lose their names first, for held-out checks.
    Named segments must carry supported evidence; a segment that looks ambiguous stays unnamed."""
    tracklets, size = {t['id']: t for t in inputs['tracklets']}, inputs['size']
    offset, support = clock_link(inputs['tracklets'], robot_tracks, size)
    stats = {'method': APPEARANCE_METHOD, 'named_segments': 0, 'teams': {}}
    if offset is None or support < MIN_CLOCK_SUPPORT:
        return robot_tracks, {**stats, 'status': 'clock_unproven'}
    mapping = segment_tracklets(robot_tracks, inputs['tracklets'], offset, size)
    looks, spans = {}, {}
    for i, robot in enumerate(robot_tracks):
        tid = mapping.get(robot.get('track_key'))
        look = _segment_look(robot, tracklets[tid], offset) if tid is not None else None
        if look is not None:
            looks[i] = look
        spans[i] = _span(robot)
    named = {i: r['team_key'] for i, r in enumerate(robot_tracks)
             if r.get('team_key') and r.get('track_key') not in hidden and _supported(r.get('identity_evidence'))}
    out = list(robot_tracks)
    for i, robot in enumerate(robot_tracks):
        teams = set(inputs['roster'].get(robot.get('alliance')) or ())
        if i in named or i not in looks or len(teams) != 3:
            continue
        ruled_out = {team for o, team in named.items() if spans[o] and spans[i]
                     and spans[o][0] <= spans[i][1] and spans[i][0] <= spans[o][1]}
        scores = {}
        for o, team in named.items():
            if o in looks and team in teams and team not in ruled_out and robot_tracks[o].get('alliance') == robot.get('alliance'):
                similarity = sum(a * b for a, b in zip(looks[i], looks[o]))
                scores[team] = max(scores.get(team, -1), similarity)
        if not scores:
            continue
        ranked = sorted(scores.items(), key=lambda row: row[1], reverse=True)
        beats = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else None
        if (beats is None and len(teams - ruled_out) != 1) or (beats is not None and beats < APPEARANCE_MARGIN):
            continue
        team = ranked[0][0]
        out[i] = {**robot, 'team_key': team, 'source_track_keys': [robot.get('track_key')],
                  'identity_evidence': {'method': APPEARANCE_METHOD, 'similarity': round(ranked[0][1], 3),
                                        'margin': round(beats, 3) if beats is not None else None,
                                        'references': sum(1 for t in named.values() if t == team)}}
        stats['named_segments'] += 1
        stats['teams'][team] = stats['teams'].get(team, 0) + 1
    return out, stats


def view_rows_for_video(plan, offset, size, video_size, video_ms_of):
    """The plan rows re-expressed for cropping the counted recording: (tracklet, video ms, box px, alliance)."""
    sx, sy = video_size[0] / size[0], video_size[1] / size[1]
    return [(tid, round(video_ms_of(t + offset)), [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy], alliance)
            for tid, t, box, alliance in plan]


def _view_progress(kind, planned):
    # Output represents actual phase/request progress, so a stuck individual request still reaches the watchdog.
    print(f'identity refinement: preparing {planned} {kind} views', file=sys.stderr, flush=True)
    def returned(done, total):
        print(f'identity refinement: {kind} requests returned {done}/{total} views', file=sys.stderr, flush=True)
    return returned


def body_views(video_path, plan, offset, size, video_size, video_ms_of, roster, templates, url, token):
    """Production view function: crop the counted recording at each planned point and ask the private model."""
    from body_identity_client import crop_plan, match_bodies
    rows = view_rows_for_video(plan, offset, size, video_size, video_ms_of)
    progress = _view_progress('body', len(plan))
    found = match_bodies(rows, crop_plan(video_path, rows), roster, templates, url, token, progress=progress)
    return [{**row, 'tracklet': plan[i][0], 't': plan[i][1]} for i, row in enumerate(found)
            if row.get('identity_source') == METHOD]


def bumper_reads(video_path, plan, offset, size, video_size, video_ms_of, roster, url, token):
    """Production read function: compact literal reads of the same crops, mapped back to tracklet time."""
    from full_match_robots import read_plan
    rows = view_rows_for_video(plan, offset, size, video_size, video_ms_of)
    back = {(tid, vt): t for (tid, vt, _, _), (_, t, _, _) in zip(rows, plan)}
    out = []
    progress = _view_progress('bumper', len(plan))
    for read in read_plan(video_path, rows, roster, url, token, progress=progress):
        t = back.get((read['tracklet'], read['t']))
        if t is not None:
            out.append({**read, 't': t})
    return out
