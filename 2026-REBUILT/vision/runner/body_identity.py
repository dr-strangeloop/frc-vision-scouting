"""Source-crop identities along short, observed physical trajectories."""
from __future__ import annotations

from collections import defaultdict
import re
import math

MODEL = 'nvidia/Cosmos-Reason2-8B'
REVISION = 'a9fae2cf89dc64db96b12860417f0eb403013bb9'
METHOD = 'body_gallery_v1'


def observation_plan(tracklets, every_ms=2000, min_separation_ms=400):
    """Observe both ends of each segment and every two seconds; no predicted boxes."""
    plan = []
    for track in tracklets:
        points = track['points']
        if not points:
            continue
        chosen = [points[0]]
        for point in points[1:]:
            if point['t'] - chosen[-1]['t'] >= every_ms:
                chosen.append(point)
        if points[-1]['t'] - chosen[-1]['t'] >= min_separation_ms:
            chosen.append(points[-1])
        for point in chosen:
            plan.append((track['id'], point['t'], point['box'], track['alliance']))
    return sorted(plan, key=lambda row: (row[1], row[0]))


def launch_plan(tracklets, launches, viewed=(), max_offset_ms=250, max_per_tracklet=3):
    """[(tracklet id, t, box, alliance)]: for each proven ball launch (tracklet id, tracklet-clock ms), the tracklet's
    observed point nearest the launch, within `max_offset_ms` and not already viewed. A launch is where a ball
    count needs the shooter's identity, so source-first refinement looks there first; no box is predicted.
    At most `max_per_tracklet` views per tracklet, spread evenly over its launches. Each costs a private model call;
    recognition and distinct-view corroboration are measured separately."""
    by_id = {track['id']: track for track in tracklets}
    seen, rows = set(viewed), defaultdict(list)
    for tid, t in launches:
        track = by_id.get(tid)
        if track is None or not track['points']:
            continue
        point = min(track['points'], key=lambda p: abs(p['t'] - t))
        if abs(point['t'] - t) > max_offset_ms or (tid, point['t']) in seen:
            continue
        seen.add((tid, point['t']))
        rows[tid].append((tid, point['t'], point['box'], track['alliance']))
    plan = []
    for found in rows.values():
        found.sort(key=lambda row: row[1])
        if max_per_tracklet and len(found) > max_per_tracklet:
            keep = {round(i * (len(found) - 1) / max(1, max_per_tracklet - 1)) for i in range(max_per_tracklet)}
            found = [row for i, row in enumerate(found) if i in keep]
        plan += found
    return sorted(plan, key=lambda row: (row[1], row[0]))


def distinct_recognition_frames(rows, viewed=()):
    """Recognized (tracklet, observed time) pairs with new crop pixels. Multiple methods on one frame count once;
    a crop repeated at another time (or already present in the source observations) is not another view."""
    def digest(row):
        value = row.get('crop_sha256')  # a read-record hash is not evidence of different source pixels
        return value if isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) else None

    seen = defaultdict(set)
    for row in viewed:
        if digest(row):
            seen[row.get('tracklet')].add(digest(row))
    frames = defaultdict(set)
    for row in rows:
        t, source = row.get('t'), row.get('identity_source')
        if (not row.get('team_key') or row.get('tracklet') is None
                or not isinstance(t, int) or isinstance(t, bool) or source not in (METHOD, 'bumper_read')):
            continue
        if source == 'bumper_read':
            confidence = row.get('confidence')
            if (not isinstance(confidence, (int, float)) or isinstance(confidence, bool)
                    or not math.isfinite(confidence) or not .8 <= confidence <= 1):
                continue
        if digest(row):
            frames[(row['tracklet'], t)].add(digest(row))
    distinct = set()
    for key in sorted(frames, key=lambda key: (key[1], str(key[0]))):
        tid, _ = key
        if frames[key] - seen[tid]:
            distinct.add(key)
        seen[tid].update(frames[key])
    return distinct


def corroborated(observations, refinement, min_views=2):
    """The refinement views kept for bounded_identities: at least `min_views` distinct frame times with fresh crop
    pixels recognize the same team, none recognizes another, and no earlier anchor or literal veto names
    another team. Stricter than a planned view: launch-time views are crowded by balls and the shooter, and on
    2026cc qm2/qm3 single launch-time views named robots that duplicated or vetoed existing identities."""
    earlier = defaultdict(set)
    for row in observations:
        if row.get('team_key'):
            earlier[row['tracklet']].add(row['team_key'])
    teams, frames = defaultdict(set), defaultdict(set)
    for row in refinement:
        if row.get('team_key'):
            teams[row['tracklet']].add(row['team_key'])
    for tid, t in distinct_recognition_frames(refinement, viewed=observations):
        frames[tid].add(t)
    keep = {tid for tid, names in teams.items()
            if len(names) == 1 and len(frames[tid]) >= min_views and earlier[tid] <= names}
    return [row for row in refinement if row['tracklet'] in keep]


def bounded_identities(tracklets, observations, roster, max_anchor_distance_ms=2500):
    """Use a recognized source crop only along its short, verified physical track.

    Two independent literal-read reference photos define each identity. Recognition uses the full robot crop;
    unreadable later views do not erase continuous motion evidence. No name crosses a lost/ambiguous association,
    extends more than 2.5 seconds from a recognized source, or survives a conflicting recognized body/bumper read.
    """
    from full_match_robots import histogram_agreement
    by_id = {track['id']: track for track in tracklets}
    anchors, vetoes = defaultdict(list), defaultdict(set)
    seen_pixels = set()
    for row in sorted(observations, key=lambda row: row.get('t', -1)):
        track = by_id.get(row.get('tracklet'))
        if (track is None or not isinstance(row.get('t'), int) or isinstance(row.get('t'), bool)
                or row['t'] not in {point['t'] for point in track['points']}):
            continue
        team = row.get('team_key')
        if team not in (roster.get(track['alliance']) or []):
            continue
        if row.get('identity_source') == 'literal_veto':
            vetoes[track['id']].add(team)
            continue
        if row.get('identity_source') not in (METHOD, 'bumper_read'):
            continue
        if row.get('identity_source') == 'bumper_read':
            confidence = row.get('confidence')
            if (not isinstance(confidence, (int, float)) or isinstance(confidence, bool)
                    or not math.isfinite(confidence) or not .8 <= confidence <= 1):
                continue
        digest = row.get('crop_sha256') or (row.get('read_record_sha256') if row.get('identity_source') == 'bumper_read' else None)
        if row.get('identity_source') in (METHOD, 'bumper_read'):
            if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
                continue
            if (track['id'], digest) in seen_pixels:
                continue
            seen_pixels.add((track['id'], digest))
        anchors[track['id']].append((row['t'], team))
    points, support = {}, defaultdict(set)
    for track in tracklets:
        association = track.get('association') or {}
        observed = track['points']
        valid = (association.get('method') == 'hsv192-motion-v1'
                 and 0 < association.get('max_gap_ms', 999999) <= 400
                 and association.get('min_appearance_agreement', 0) >= .65
                 and association.get('ambiguity_margin', 0) >= .12)
        for before, after in zip(observed, observed[1:]):
            agreement = histogram_agreement(before.get('hist'), after.get('hist'))
            valid = (valid and 0 < after['t'] - before['t'] <= 400
                     and len(before.get('hist') or []) == 192 and len(after.get('hist') or []) == 192
                     and agreement is not None and agreement >= .65)
        valid = valid and bool(observed) and all(len(point.get('hist') or []) == 192 for point in observed)
        rows = anchors[track['id']]
        teams = {team for _, team in rows} | vetoes[track['id']]
        team = next(iter(teams)) if valid and rows and len(teams) == 1 else None
        support[track['id']].update(t for t, _ in rows)
        points[track['id']] = [{**point, 'team_key': team if team and
                               min(abs(point['t'] - t) for t, _ in rows) <= max_anchor_distance_ms else None}
                              for point in observed]
    claims = defaultdict(list)
    for tid, rows in points.items():
        for point in rows:
            if point['team_key']:
                claims[(point['t'], point['team_key'])].append((tid, point))
    for rows in claims.values():
        if len({tid for tid, _ in rows}) > 1:
            for _, point in rows:
                point['team_key'] = None
    return points, support


def assemble_robots(tracklets, observations, roster, references):
    """Preserve each physical fragment, including unknown portions, for shooter attribution."""
    labeled, support = bounded_identities(tracklets, observations, roster)
    counts = defaultdict(int)
    for row in references:
        counts[row['team_key']] += 1
    out = []
    for track in tracklets:
        # Consecutive segments keep a stable physical ID, but change the named evidence independently.
        segments = []
        for point in labeled[track['id']]:
            team = point['team_key'] if counts[point['team_key']] >= 2 else None
            if not segments or segments[-1]['team_key'] != team:
                segments.append({'team_key': team, 'alliance': track['alliance'], 'by': [METHOD] if team else [],
                                 'reads': 0, 'points': [], 'identity_evidence': None})
            entry = segments[-1]
            entry['points'].append({**point, 'tracklet': track['id']})
            if team:
                entry['identity_evidence'] = {'method': METHOD, 'model': MODEL, 'revision': REVISION,
                                              'reference_reads': counts[team],
                                              'query_views': len(support[track['id']]),
                                              'anchor_sources': sorted({row.get('identity_source') for row in observations
                                                                        if row.get('tracklet') == track['id']
                                                                        and row.get('team_key') == team
                                                                        and row.get('identity_source') in (METHOD, 'bumper_read')}),
                                              'association': 'hsv192-motion-v1', 'max_gap_ms': 400,
                                              'max_anchor_distance_ms': 2500}
        out.extend(segments)
    return out
