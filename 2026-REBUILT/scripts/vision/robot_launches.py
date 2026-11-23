"""Source-local observed exits. A HUB entry or a fitted flight is never required.

These are partial observations, not validated complete shot totals. The older
fuel_count.launches() remains a broad candidate generator for entry diagnostics.
"""
import bisect
import math

METHOD = 'observed_body_exit_v1'
MAX_STEP_S = 0.10
MAX_EXIT_S = 0.35
MAX_ORIGIN_DEPTH = 0.65  # exclude the floor/bumper portion of a measured box
MIN_RELATIVE_RISE = 0.15
# A camera sees depth as height: a ball shot by a robot nearer the camera crosses farther robots' boxes on its way up,
# so "first seen inside this box" does not prove the source. On 2026cc qm3 the parked 6665 collected another robot's
# shots this way. The veto only ever removes a name, so it may use a body's last measured box for a moment: the tracker
# loses bodies around shooters, and a true shot sent to unknown costs less than a false credit.
NEARER_BODY_REACH_MS = 250       # how far back along the ball's early path to look
NEARER_BODY_HOLD_MS = 500        # a body's nearest measured box stays usable this long
NEARER_BODY_PAD = 0.1            # share of the nearer body's box around it
NEARER_BODY_ACTIVE_MS = 800      # the nearer body must itself show an observed exit this close to the ball's start;
                                 # a parked or defending robot in front of a true shooter does not veto it
PATH_OBSERVATIONS = 5            # early observations that set the path
# An unnamed body on the field with its two alliance partners, both named, is the third team on the roster. This names
# robots whose bumpers never read (6036 in 2026cc qm3). Held-out names on qm2/qm3: 8 right, 0 wrong; a small sample.
ELIMINATION_METHOD = 'alliance_elimination_v1'
ELIMINATION_BUCKET_MS = 250
ELIMINATION_MIN_VOTES = 4        # one second of moments with both partners named
ELIMINATION_JOIN_MS = 500        # a segment this close, box overlapping, is the same body
APPEARANCE_METHOD = 'appearance_match_v1'
# One team is one robot. Two bodies carrying the same team, measured apart at the same moments, cannot both be right.
CONFLICT_BUCKET_MS = 250
CONFLICT_MIN_BUCKETS = 2         # half a second seen apart; a brief tracker split of one body is not a conflict
WITHDRAWN_METHOD = 'roster_conflict_withdrawn_v1'


def supported_identity(evidence):
    """Same source evidence accepted by src/lib/vision/robotIdentity.js."""
    if evidence == 'direct_reads':
        return True
    if isinstance(evidence, dict) and evidence.get('method') == APPEARANCE_METHOD:
        references = evidence.get('references')
        return isinstance(references, int) and not isinstance(references, bool) and references >= 1
    if isinstance(evidence, dict) and evidence.get('method') == ELIMINATION_METHOD:
        votes, teammates = evidence.get('votes'), evidence.get('teammates')
        return (isinstance(votes, int) and not isinstance(votes, bool) and votes >= ELIMINATION_MIN_VOTES
                and isinstance(teammates, list) and len(set(teammates)) == 2)
    return isinstance(evidence, dict) and (
        evidence.get('method') == 'body_gallery_v1'
        and evidence.get('model') == 'nvidia/Cosmos-Reason2-8B'
        and evidence.get('revision') == 'a9fae2cf89dc64db96b12860417f0eb403013bb9'
        and evidence.get('association') == 'hsv192-motion-v1'
        and all(isinstance(evidence.get(k), int) and not isinstance(evidence[k], bool) and evidence[k] >= n
                for k, n in (('reference_reads', 2), ('query_views', 1)))
        and all(isinstance(evidence.get(k), int) and not isinstance(evidence[k], bool) and 0 < evidence[k] <= n
                for k, n in (('max_gap_ms', 400), ('max_anchor_distance_ms', 2500)))
    )


def identity_strength(evidence):
    """0 unsupported; 1 appearance; 2 alliance elimination; 3 a read of the robot itself (bumper or body gallery).
    Inferred names (1, 2) never name other robots and give way to a read when they disagree."""
    if not supported_identity(evidence):
        return 0
    method = evidence.get('method') if isinstance(evidence, dict) else None
    return {APPEARANCE_METHOD: 1, ELIMINATION_METHOD: 2}.get(method, 3)


def _bucket_boxes(robot):
    boxes = {}
    for p in robot.get('trajectory') or ():
        if p.get('t') is not None and p.get('box') is not None and len(p['box']) == 4:
            boxes.setdefault(int(p['t'] // CONFLICT_BUCKET_MS), p['box'])
    return boxes


def _apart(a, b):
    return a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]


def resolve_name_conflicts(robots):
    """(robots, stats). Where two bodies carry the same team while measured apart for at least CONFLICT_MIN_BUCKETS
    moments, the weaker inferred name is withdrawn; inferred names of equal weight are both withdrawn. Two read names
    are left alone: re-scoring may only add names, never move a supported read. Withdrawn segments keep their body but
    lose the team, so their balls stay with the alliance instead of going to the wrong robot."""
    named = [(i, r, identity_strength(r.get('identity_evidence'))) for i, r in enumerate(robots) if r.get('team_key')]
    named = [(i, r, strength, _bucket_boxes(r)) for i, r, strength in named if strength]
    withdraw = set()
    for n, (i, a, sa, boxes_a) in enumerate(named):
        for j, b, sb, boxes_b in named[n + 1:]:
            if a['team_key'] != b['team_key'] or (sa == 3 and sb == 3):
                continue
            apart = sum(_apart(boxes_a[k], boxes_b[k]) for k in boxes_a.keys() & boxes_b.keys())
            if apart < CONFLICT_MIN_BUCKETS:
                continue
            if sa <= sb and sa < 3:
                withdraw.add(i)
            if sb <= sa and sb < 3:
                withdraw.add(j)
    out, stats = list(robots), {'method': WITHDRAWN_METHOD, 'withdrawn_segments': len(withdraw), 'teams': {}}
    for i in sorted(withdraw):
        robot = robots[i]
        out[i] = {**robot, 'team_key': None,
                  'identity_evidence': {'method': WITHDRAWN_METHOD, 'team_key': robot['team_key'],
                                        'was': (robot.get('identity_evidence') or {}).get('method')}}
        stats['teams'][robot['team_key']] = stats['teams'].get(robot['team_key'], 0) + 1
    return out, stats


def withdraw_exit_names(exits, robots):
    """Counted exits lose a team their body no longer carries; the physical exits are unchanged."""
    unnamed = {r.get('track_key') for r in robots if not r.get('team_key')}
    return [{**e, 'team_key': None, 'identity': 'unknown'} if e.get('team_key') and e.get('track_key') in unnamed else e
            for e in exits]


def _inside(box, u, v):
    return box is not None and box[0] <= u <= box[2] and box[1] <= v <= box[3]


def _relative(box, u, v):
    return (u - box[0]) / (box[2] - box[0]), (v - box[1]) / (box[3] - box[1])


def _segment_times(robot):
    return sorted(p['t'] for p in robot.get('trajectory') or [] if p.get('t') is not None and p.get('box'))


def _named_neighbour_disagrees(robot, team, robots):
    """Does a named segment continuing this body carry another team?"""
    times = _segment_times(robot)
    points = sorted((p for p in robot['trajectory'] if p.get('t') is not None and p.get('box')), key=lambda p: p['t'])
    for other in robots:
        other_times = _segment_times(other)
        if other is robot or not other_times or not other.get('team_key') or other['team_key'] == team:
            continue
        others = sorted((p for p in other['trajectory'] if p.get('t') is not None and p.get('box')), key=lambda p: p['t'])
        for mine, theirs, gap in ((points[0], others[-1], times[0] - other_times[-1]),
                                  (points[-1], others[0], other_times[0] - times[-1])):
            a, b = mine['box'], theirs['box']
            if 0 <= gap <= ELIMINATION_JOIN_MS and a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                return True
    return False


def eliminate_names(robots, roster):
    """(robots, stats). Names an unnamed segment only when, at every moment both partners can be seen named, they leave
    the same third team, over at least a second. Moments with any other number of alliance bodies give no vote.
    Partners must be read names (strength 3): a name guessed from looks never names a third robot."""
    alive = {}
    for i, robot in enumerate(robots):
        for t in _segment_times(robot):
            alive.setdefault(robot.get('alliance'), {}).setdefault(int(t // ELIMINATION_BUCKET_MS), set()).add(i)
    out, stats = list(robots), {'method': ELIMINATION_METHOD, 'named_segments': 0, 'teams': {}}
    for i, robot in enumerate(robots):
        teams = set(roster.get(robot.get('alliance')) or ())
        if robot.get('team_key') or len(teams) != 3:
            continue
        votes, partners = {}, set()
        for bucket in {int(t // ELIMINATION_BUCKET_MS) for t in _segment_times(robot)}:
            others = alive[robot['alliance']][bucket] - {i}
            named = {robots[o].get('team_key') for o in others if identity_strength(robots[o].get('identity_evidence')) == 3}
            if len(others) == 2 and len(named) == 2 and named <= teams:
                third = next(iter(teams - named))
                votes[third] = votes.get(third, 0) + 1
                partners |= named
        team = next(iter(votes)) if len(votes) == 1 else None
        if (team is None or votes[team] < ELIMINATION_MIN_VOTES or len(partners) != 2
                or _named_neighbour_disagrees(robot, team, robots)):
            continue
        out[i] = {**robot, 'team_key': team, 'source_track_keys': [robot.get('track_key')],
                  'identity_evidence': {'method': ELIMINATION_METHOD, 'votes': votes[team], 'teammates': sorted(partners)}}
        stats['named_segments'] += 1
        stats['teams'][team] = stats['teams'].get(team, 0) + 1
    return out, stats


def body_index(robots):
    """[(robot, times, boxes)] of measured boxes, for nearer_body_path."""
    index = []
    for robot in robots or ():
        points = sorted((p['t'], p['box']) for p in (robot.get('trajectory') or [])
                        if p.get('t') is not None and p.get('box') is not None and len(p['box']) == 4)
        if points:
            index.append((robot, [t for t, _ in points], [box for _, box in points]))
    return index


def _held_box(times, boxes, t_ms):
    at = bisect.bisect_left(times, t_ms)
    near = [i for i in (at - 1, at) if 0 <= i < len(times)]
    best = min(near, key=lambda i: abs(times[i] - t_ms), default=None)
    return boxes[best] if best is not None and abs(times[best] - t_ms) <= NEARER_BODY_HOLD_MS else None


def nearer_body_path(track, robot, index, activity, offset_s, width, height):
    """Does the ball's early path, followed backward, reach a launching body nearer the camera than `robot`?
    Lower in the image is nearer. `activity`: {track key: sorted exit source ms}, from exit_activity."""
    from fuel_count import _box_at, _obs
    obs = _obs(track)[:PATH_OBSERVATIONS]
    if len(obs) < 3 or obs[-1][1] <= obs[0][1]:
        return False
    dt = obs[-1][1] - obs[0][1]
    du, dv = (obs[-1][2] - obs[0][2]) / width / dt, (obs[-1][3] - obs[0][3]) / height / dt
    t0, u0, v0 = float(obs[0][1]), obs[0][2] / width, obs[0][3] / height
    start_ms = round((t0 - offset_s) * 1000)
    source = _box_at(robot, start_ms, width=width, height=height, observed_only=True)
    if source is None:
        return False
    def launching(other):
        times = activity.get(other.get('track_key')) or []
        at = bisect.bisect_left(times, start_ms - NEARER_BODY_ACTIVE_MS)
        return at < len(times) and times[at] <= start_ms + NEARER_BODY_ACTIVE_MS
    for back_ms in range(33, NEARER_BODY_REACH_MS + 1, 33):
        s = back_ms / 1000
        t_ms = round((t0 - s - offset_s) * 1000)
        u, v = u0 - du * s, v0 - dv * s
        for other, times, boxes in index:
            box = None if other is robot else _held_box(times, boxes, t_ms)
            if box is None or box[3] <= source[3]:
                continue
            pad_u, pad_v = (box[2] - box[0]) * NEARER_BODY_PAD, (box[3] - box[1]) * NEARER_BODY_PAD
            if box[0] - pad_u <= u <= box[2] + pad_u and box[1] - pad_v <= v <= box[3] + pad_v and launching(other):
                return True
    return False


def exit_activity(exits, robots=()):
    """{track key: sorted exit source ms} per body, before any veto. A refined track inherits its sources' exits."""
    activity = {}
    for event in exits:
        if event.get('body_key'):
            activity.setdefault(event['body_key'], []).append(event['source_ms'])
    for robot in robots or ():
        inherited = [t for key in robot.get('source_track_keys') or () for t in activity.get(key, ())]
        if inherited:
            activity[robot.get('track_key')] = inherited
    return {key: sorted(times) for key, times in activity.items()}


def mark_nearer_body_entries(entries, tracks, robots, exits, offset_s, width, height):
    """Flag credited HUB entries whose ball path reaches a launching nearer body; evidenced_shooter_entries withholds them."""
    by_key = {r.get('track_key'): r for r in robots or ()}
    index, activity = body_index(robots), exit_activity(exits, robots)
    out = []
    for entry in entries:
        robot, tid = by_key.get(entry.get('track_key')), entry.get('track_id')
        flagged = (entry.get('attribution') == 'same_track' and entry.get('source_in_body') is True and robot is not None
                   and isinstance(tid, int) and not isinstance(tid, bool) and 0 <= tid < len(tracks)
                   and nearer_body_path(tracks[tid], robot, index, activity, offset_s, width, height))
        out.append({**entry, 'nearer_body_path': True} if flagged else entry)
    return out


def _sample(point):
    return round(float(point[1]), 3), round(float(point[2])), round(float(point[3]))


def link_same_ball_entries(entries, exits, tracks, pool):
    """Credit an unattributed HUB entry from a named exit of the same ball: its continuous track shares at least two
    observations with exactly one exit, on the same alliance. No gap is bridged; a gap link was wrong too often."""
    from fuel_count import _obs
    fragments = {(tid, index): native for tid, track in enumerate(pool) for index, native in observed_fragments(track)}
    samples = {}
    for i, event in enumerate(exits):
        native = fragments.get((event.get('track_id'), event.get('fragment')))
        for point in _obs(native) if native is not None else ():
            samples.setdefault(_sample(point), []).append(i)
    out = []
    for entry in entries:
        tid = entry.get('track_id')
        if (entry.get('team_key') or entry.get('source_continuous') is not True or entry.get('stitched')
                or entry.get('nearer_body_path') or not isinstance(tid, int) or isinstance(tid, bool)
                or not 0 <= tid < len(tracks)):
            out.append(entry)
            continue
        shared = {}
        for point in _obs(tracks[tid]):
            for i in samples.get(_sample(point), ()):
                shared[i] = shared.get(i, 0) + 1
        hits = [i for i, count in shared.items() if count >= 2]
        event = exits[hits[0]] if len(hits) == 1 else None
        if event is None or not event.get('team_key') or event.get('alliance') != entry.get('alliance'):
            out.append(entry)
            continue
        out.append({**entry, 'attribution': 'same_track', 'source_in_body': True, 'team_key': event['team_key'],
                    'track_key': event['track_key'], 'launch_ms': event['t_ms'], 'source_link': 'observed_exit_same_ball'})
    return out


def observed_fragments(track):
    """Native observed intervals, never a prefix inferred across a tracker's join.

    Joining two image tracks does not erase a later source-local exit. Treat
    each side independently, discarding observations inside declared joins.
    Missing/malformed provenance cannot establish native fragment boundaries.
    """
    from fuel import BallTrack
    from fuel_count import _obs
    breaks = getattr(track, 'source_breaks', None)
    if not isinstance(breaks, list) or any(
        not isinstance(pair, (list, tuple)) or len(pair) != 2
        or not all(isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t) for t in pair)
        or pair[1] <= pair[0] for pair in breaks
    ):
        return
    intervals = sorted(set(tuple(pair) for pair in breaks))
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    groups = {}
    for point in _obs(track):
        t = float(point[1])
        if any(start < t < end for start, end in merged):
            continue
        index = sum(end <= t for _, end in merged)
        groups.setdefault(index, []).append(point)
    for index, obs in groups.items():
        yield index, BallTrack(obs, source_breaks=[])


def observed_exits(tracks, robots, offset_s, width, height):
    """Return source events and diagnostics; no hub, score, clock overlay or roster prior.

    The ball must first be observed inside an upper measured body, then move
    upward out of it with two consecutive outside observations. No snapping or
    projected boxes prove an exit. Only this short prefix needs continuity:
    losing the ball later, missing the HUB or joining a later flight is immaterial.
    """
    from fuel import continuous_source_to
    from fuel_count import MATCH_MS, _box_at, _obs, _robot_time_buckets
    buckets = _robot_time_buckets(robots)
    index = body_index(robots)
    stats = dict(tracks=len(tracks), observed_fragments=0, body_origins=0, duplicate_tracks=0, nearer_body_paths=0,
                 unknown_provenance_tracks=sum(not isinstance(getattr(t, 'source_breaks', None), list) for t in tracks))
    events, samples, box_cache, event_tracks = [], {}, {}, []
    def measured_box(robot, t):
        key = (id(robot), t)
        if key not in box_cache:
            box_cache[key] = _box_at(robot, t, width=width, height=height, observed_only=True)
        return box_cache[key]
    fragments = ((tid, index, native) for tid, track in enumerate(tracks)
                 for index, native in observed_fragments(track))
    for tid, fragment, track in fragments:
        stats['observed_fragments'] += 1
        obs = _obs(track)
        if len(obs) < 3:
            continue
        t0 = float(obs[0][1])
        t_ms = round((t0 - offset_s) * 1000)
        if not -MAX_EXIT_S * 1000 <= t_ms <= MATCH_MS:
            continue
        u, v = obs[0][2] / width, obs[0][3] / height
        active = robots if buckets is None else buckets.get(t_ms // 1000, [])
        origins = []
        for robot in active:
            box = measured_box(robot, t_ms)
            if _inside(box, u, v) and box[2] > box[0] and box[3] > box[1]:
                origins.append((robot, box))
        if not origins:
            continue
        stats['body_origins'] += 1
        exits = []
        for robot, box in origins:
            _, depth = _relative(box, u, v)
            if depth > MAX_ORIGIN_DEPTH:
                continue
            outside, previous_y = 0, depth
            inside = [(0, depth)]
            for i in range(1, len(obs)):
                point = obs[i]
                dt = point[1] - obs[i - 1][1]
                if not 0 < dt <= MAX_STEP_S + 1e-6:
                    break
                at_ms = round((point[1] - offset_s) * 1000)
                current = measured_box(robot, at_ms)
                if current is None or current[2] <= current[0] or current[3] <= current[1]:
                    break
                pu, pv = point[2] / width, point[3] / height
                rx, ry = _relative(current, pu, pv)
                if _inside(current, pu, pv):
                    # A ball can be carried before it leaves. Keep only the
                    # recent measured body prefix; never restart after an
                    # outside pass-through or an interrupted flight.
                    if outside:
                        break
                    inside = [(j, y) for j, y in inside if point[1] - obs[j][1] <= MAX_EXIT_S + 1e-6]
                    inside.append((i, ry))
                    previous_y = ry
                    continue
                # Reject downward motion, stationary balls exposed by box motion,
                # and incoming/pass-through tracks whose first observation is outside.
                rising = ry < previous_y and obs[i - 1][3] - point[3] >= height / 720
                above_or_side = ry < -0.05 or (not 0 <= rx <= 1 and ry < 0.65)
                if not rising or not above_or_side:
                    break
                outside += 1
                previous_y = ry
                recent = [(j, y) for j, y in inside if point[1] - obs[j][1] <= MAX_EXIT_S + 1e-6
                          and y <= MAX_ORIGIN_DEPTH]
                if outside >= 2 and recent and max(y for _, y in recent) - ry >= MIN_RELATIVE_RISE:
                    if continuous_source_to(track, point[1]):
                        exits.append((robot, i, recent[0][0]))
                    break
        if not exits:
            continue
        # An overlapping opponent/unnamed body vetoes a name even if that body
        # does not itself satisfy the upward-motion rule.
        unique = len(origins) == 1 and len(exits) == 1
        robot, end, source = exits[0]
        # Re-check every body at the local source time, including bodies that
        # appeared while the ball was being carried.
        source_ms = round((obs[source][1] - offset_s) * 1000)
        su, sv = obs[source][2] / width, obs[source][3] / height
        local_robots = robots if buckets is None else buckets.get(source_ms // 1000, [])
        local_origins = [r for r in local_robots if _inside(measured_box(r, source_ms), su, sv)]
        unique = unique and len(local_origins) == 1 and local_origins[0] is robot
        colors = {r.get('alliance') for r, _ in origins} | {r.get('alliance') for r in local_origins}
        alliance = next(iter(colors)) if len(colors) == 1 and colors <= {'red', 'blue'} else None
        named = unique and robot.get('team_key') and supported_identity(robot.get('identity_evidence'))
        exit_ms = round((obs[end - 1][1] - offset_s) * 1000)
        if not 0 <= exit_ms <= MATCH_MS:
            continue
        event = {'t_ms': exit_ms, 'source_ms': source_ms,
                 'track_id': tid, 'fragment': fragment, 'track_key': robot.get('track_key') if unique else None,
                 'team_key': robot.get('team_key') if named else None, 'alliance': alliance,
                 'method': METHOD, 'source_continuous': True,
                 'body_key': robot.get('track_key') if unique else None,
                 'identity': 'supported' if named else 'unknown' if unique else 'ambiguous'}
        # Suppress only duplicated observed detections, not adjacent shots from
        # one stream. Two common timestamps/pixels are needed; no rate cap.
        prefix = [(round(o[1] * 1000), float(o[2]), float(o[3])) for o in obs[source:end + 1]]
        common = {}
        for t, x, y in prefix:
            seen = {eid for eid, px, py in samples.get(t, []) if math.hypot(x - px, y - py) <= 2 * height / 720}
            for eid in seen:
                common[eid] = common.get(eid, 0) + 1
        duplicate = any(n >= 2 for n in common.values())
        if duplicate:
            stats['duplicate_tracks'] += 1
            continue
        for t, x, y in prefix:
            samples.setdefault(t, []).append((tid, x, y))
        events.append(event)
        event_tracks.append((event, track, robot))
    activity = exit_activity(events)
    for event, track, robot in event_tracks:
        if event['track_key'] and nearer_body_path(track, robot, index, activity, offset_s, width, height):
            event.update(track_key=None, team_key=None, identity='ambiguous')
            stats['nearer_body_paths'] += 1
    return sorted(events, key=lambda e: (e['t_ms'], e['track_id'])), stats


def rename_exits(events, refined):
    """The same physical exits with names from refined robot tracks. Refinement never re-detects exits: splitting a
    body at a name boundary must not create or lose an observation. Only unique sources whose robot track was replaced
    change; a source time between refined segments stays unknown."""
    replaced = {}
    for robot in refined:
        for key in robot.get('source_track_keys') or ():
            replaced.setdefault(key, []).append(robot)
    out = []
    for event in events:
        rows = replaced.get(event.get('track_key'))
        if not rows:
            out.append(event)
            continue
        cover = [r for r in rows if any(p['t'] <= event['source_ms'] for p in r['trajectory'])
                 and any(p['t'] >= event['source_ms'] for p in r['trajectory'])]
        robot = cover[0] if len(cover) == 1 else None
        named = bool(robot and robot.get('team_key') and supported_identity(robot.get('identity_evidence')))
        out.append({**event, 'track_key': robot['track_key'] if robot else event['track_key'],
                    'team_key': robot['team_key'] if named else None, 'identity': 'supported' if named else 'unknown'})
    return out


def moved_sources(exits, renamed):
    """Track keys of supported exits that lost or changed their name, from `rename_exits` (same order as `exits`)."""
    return {old['track_key'] for old, new in zip(exits, renamed)
            if old.get('team_key') and new.get('team_key') != old['team_key']}


def restore_sources(refined, originals, keys):
    """`refined` with every refined track built from `keys` put back as it was: one fragment's conflict costs that
    fragment, not the whole refinement. Returns (robot tracks, restored track keys)."""
    bad = [r for r in refined if set(r.get('source_track_keys') or ()) & set(keys)]
    restored = {key for r in bad for key in r['source_track_keys']} | set(keys)
    kept = [r for r in refined if not any(r is b for b in bad)]
    present = {r.get('track_key') for r in kept}
    return kept + [r for r in originals if r.get('track_key') in restored and r.get('track_key') not in present], restored


def refinement_change(before, after):
    """Identity may only add names to the same observed physical exit set."""
    key = lambda e: (e['track_id'], e['source_ms'], e['t_ms'])
    old, new = {key(e): e for e in before}, {key(e): e for e in after}
    changed = len(old.keys() ^ new.keys()) + len(before) - len(old) + len(after) - len(new)
    moved = sum(bool(e.get('team_key')) and new.get(k, {}).get('team_key') != e['team_key'] for k, e in old.items())
    # An alliance change also changes the physical comparison's population.
    changed += sum(old[k].get('alliance') != new[k].get('alliance') for k in old.keys() & new.keys())
    added = sum(bool(e.get('team_key')) and not old.get(k, {}).get('team_key') for k, e in new.items())
    return {'named_launches_added': added, 'supported_launches_moved': moved, 'observed_exit_events_changed': changed}


def summarize_exits(events, robots, stats=None):
    from fuel_count import _phase, _timeline, tracked_share
    def timeline(rows):
        full = _timeline(rows)
        # Retain exact per-second steps and the final endpoint without spending
        # Hub snapshot space on repeated counts during long pauses.
        return [point for i, point in enumerate(full)
                if i == 0 or i == len(full)-1 or point[1] != full[i-1][1]]
    keys = dict.fromkeys(r['team_key'] for r in robots if r.get('team_key'))
    named_robots = []
    for key in keys:
        sources = [r for r in robots if r.get('team_key') == key and supported_identity(r.get('identity_evidence'))]
        if not sources:
            continue
        mine = [e for e in events if e.get('team_key') == key]
        named_robots.append({'team_key': key, 'alliance': sources[0].get('alliance'),
                             'observed': len(mine), 'auto': sum(_phase(e['t_ms']) == 'auto' for e in mine),
                             'teleop': sum(_phase(e['t_ms']) == 'teleop' for e in mine),
                             'timeline': timeline(mine), 'partial': True,
                             'tracked_share': tracked_share(sources), 'identity_evidence': sources[0].get('identity_evidence')})
    alliances = {}
    for color in ('red', 'blue'):
        mine = [e for e in events if e.get('alliance') == color]
        named = sum(bool(e.get('team_key')) for e in mine)
        alliances[color] = {'observed': len(mine), 'named': named, 'unattributed': len(mine) - named,
                            'timeline': timeline(mine), 'partial': True}
    return {'version': 1, 'method': METHOD, 'partial': True, 'validated_complete': False,
            'robots': named_robots, 'alliances': alliances,
            'unknown_alliance': sum(e.get('alliance') is None for e in events),
            'stats': stats or {}}
