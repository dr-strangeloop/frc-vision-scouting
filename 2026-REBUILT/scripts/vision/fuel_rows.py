"""Turn raw ball tracks and robot tracks into per-ball vision fuel observations.

Pure functions used by a scouting run (run_match.py) so it emits per-ball `fuel_scored` rows (`count: 1`,
`source: 'fuel_counter'`, with team attribution, auto/teleop phase, confidence, and `hub_active`) and per-robot
launch `fuel_attempt` rows directly from the ball tracks and robot tracks in a single pass.
"""
import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from camera3d import camera_from_pixel_homography  # noqa: E402
from common import AUTO_END_MS, AUTO_GRACE_MS  # noqa: E402
from fuel_count import (  # noqa: E402
    MATCH_MS,
    _phase,
    attribute_entries,
    auto_winner_from_entries,
    count_entries,
    default_hubs,
    hub_active_at,
    TRUE_OPENING_MODEL_Z,
    entry_openings,
    hub_openings,
    hub_polygons,
    lifted as fc_lifted,
    launches,
    summarize,
    evidenced_shooter_entries,
    _unique_observed_origin,
)

BALLISTIC_DEDUP_MS = 120


def _canon_team_key(team_key):
    if team_key is None:
        return None
    raw = str(team_key).strip()
    if not raw:
        return None
    low = raw.lower()
    if low.startswith('frc'):
        return f'frc{low[3:]}'
    if low.isdigit():
        return f'frc{low}'
    return low


def _normalize_roster_keys(roster_keys, robot_tracks=(), shots=()):
    """Return a set of canonical 'frcNNN' team keys, or None when unconstrained."""
    if roster_keys is None:
        return None
    out = set()
    if isinstance(roster_keys, dict):
        for k, v in roster_keys.items():
            if str(k).strip().lower() in ('red', 'blue') and isinstance(v, (list, tuple, set)):
                for item in v:
                    ck = _canon_team_key(item)
                    if ck:
                        out.add(ck)
            else:
                ck = _canon_team_key(k)
                if ck:
                    out.add(ck)
    else:
        for item in roster_keys:
            ck = _canon_team_key(item)
            if ck:
                out.add(ck)
    return out


def _resolve_camera_and_polygons(polygons=None, camera=None, calibration=None, h_ref_to_field=None, hubs=None, width=1280, height=720):
    hubs = hubs or default_hubs()
    if camera is None:
        if h_ref_to_field is None and isinstance(calibration, dict) and calibration.get('homography_norm'):
            h_norm = np.array(calibration['homography_norm'], float)
            h_ref_to_field = h_norm @ np.diag([1.0 / float(width), 1.0 / float(height), 1.0])
        if h_ref_to_field is not None:
            camera = camera_from_pixel_homography(np.asarray(h_ref_to_field, float), int(width), int(height))
    if polygons is not None:
        return polygons, camera
    if camera is not None:
        return hub_polygons(camera, hubs), camera
    return {}, None


def _sync_ballistic_shots_with_entries(shots, attributed_entries, auto_winner):
    """Update ballistic `shots` in-place/copy with hub-entry matches and `hub_active` flags."""
    if not shots:
        return []
    updated = [copy.deepcopy(s) for s in shots]
    claimed_entries = set()

    # 1. Exact track_id match
    tid_to_idx = {
        e['track_id']: i for i, e in enumerate(attributed_entries)
        if e.get('track_id') is not None
    }

    for s in updated:
        tid = s.get('track_id')
        if tid is not None and tid in tid_to_idx and tid_to_idx[tid] not in claimed_entries:
            e_idx = tid_to_idx[tid]
            e = attributed_entries[e_idx]
            if s.get('alliance') in (None, e['alliance']):
                claimed_entries.add(e_idx)
                s['result'] = 'scored'
                s['score_ms'] = e['t_ms']
                prev_entry = s.get('entry') if isinstance(s.get('entry'), dict) else {}
                s['entry'] = {
                    **prev_entry,
                    'decision': 'scored',
                    't_ms': e['t_ms'],
                    'hub_active': e.get('hub_active'),
                    'confidence': max(float(prev_entry.get('confidence') or 0.0), float(e.get('confidence') or 0.85)),
                }

    # 2. Match remaining ballistic shots to attributed entries that linked to their launch_ms
    for s in updated:
        if s.get('result') == 'scored' and s.get('score_ms') is not None:
            continue
        s_tk = _canon_team_key(s.get('team_key'))
        s_t = s.get('t_ms')
        if s_t is None:
            continue
        best = None
        for e_idx, e in enumerate(attributed_entries):
            if e_idx in claimed_entries:
                continue
            if s.get('alliance') and e.get('alliance') != s.get('alliance'):
                continue
            if s_tk and _canon_team_key(e.get('team_key')) != s_tk:
                continue
            l_ms = e.get('launch_ms')
            if l_ms is not None and abs(int(l_ms) - int(s_t)) <= BALLISTIC_DEDUP_MS:
                gap = abs(int(l_ms) - int(s_t))
                if best is None or gap < best[0]:
                    best = (gap, e_idx)
        if best is not None:
            e_idx = best[1]
            e = attributed_entries[e_idx]
            claimed_entries.add(e_idx)
            s['result'] = 'scored'
            s['score_ms'] = e['t_ms']
            prev_entry = s.get('entry') if isinstance(s.get('entry'), dict) else {}
            s['entry'] = {
                **prev_entry,
                'decision': 'scored',
                't_ms': e['t_ms'],
                'hub_active': e.get('hub_active'),
                'confidence': max(float(prev_entry.get('confidence') or 0.0), float(e.get('confidence') or 0.82)),
            }

    for s in updated:
        alliance = s.get('alliance')
        if not alliance and s.get('target'):
            alliance = 'blue' if s['target'] == 'blue_hub' else 'red'
        t_check = s.get('score_ms') if s.get('score_ms') is not None else s.get('t_ms')
        ha = hub_active_at(alliance, t_check, auto_winner)
        s['hub_active'] = ha
        if isinstance(s.get('entry'), dict) and 'hub_active' not in s['entry']:
            s['entry']['hub_active'] = ha
    return updated


def to_observations(
    entries,
    launch_list=(),
    robot_tracks=(),
    *,
    shots=(),
    roster_keys=None,
    auto_end_ms=AUTO_END_MS,
    grace_ms=AUTO_GRACE_MS,
    view_id=None,
    run_id=None,
    emit_attempts=True,
    auto_winner=None,
    width=1280,
    height=720,
    hubs=None,
    camera=None,
):
    """Convert hub `entries` and robot `launch_list` (plus optional ballistic `shots`) into `vision_observations` rows.

    Emits:
      * One `fuel_scored` row per individual ball into a hub (`count: 1`, `source: 'fuel_counter'`), with
        `team_key`, `track_key`, `alliance`, `phase`, `started_ms`, `ended_ms`, `confidence`, and `hub_active`.
      * When `emit_attempts=True`:
          - `fuel_attempt` rows (`source: 'fuel_ballistic'`) from fitted 3D `shots` (with `emit_scored=False`
            so no ball is counted twice).
          - `fuel_attempt` rows (`source: 'fuel_counter'`) for robot-box launches in `launch_list` not already
            covered by a `fuel_ballistic` shot.
    """
    roster_set = _normalize_roster_keys(roster_keys, robot_tracks=robot_tracks, shots=shots)
    track_key_by_team = {
        _canon_team_key(r.get('team_key')): r.get('track_key')
        for r in (robot_tracks or ())
        if r.get('team_key') and r.get('track_key')
    }

    if entries and ('attribution' not in entries[0] or 'hub_active' not in entries[0]):
        attributed_entries, _, _ = attribute_entries(
            entries, launch_list, robot_tracks=robot_tracks, width=width, height=height, hubs=hubs, ballistic_shots=shots, camera=camera
        )
    else:
        attributed_entries = [dict(e) for e in (entries or ())]
    attributed_entries = evidenced_shooter_entries(attributed_entries)

    if auto_winner is None:
        auto_winner = auto_winner_from_entries(attributed_entries, auto_end_ms=auto_end_ms, grace_ms=grace_ms)

    rows = []

    # 1. Emit fuel_attempt rows from ballistic 3D shots (without emitting duplicate fuel_scored rows)
    updated_shots = _sync_ballistic_shots_with_entries(shots, attributed_entries, auto_winner) if shots else []
    ballistic_tids = set()
    ballistic_times_by_team = {}
    if emit_attempts and updated_shots:
        import fuel as fuelmod
        effective_roster = roster_set if roster_set is not None else {
            _canon_team_key(s.get('team_key')) for s in updated_shots if s.get('team_key')
        } | {
            _canon_team_key(r.get('team_key')) for r in (robot_tracks or ()) if r.get('team_key')
        }
        attempt_rows = fuelmod.shots_to_observations(
            updated_shots, effective_roster, auto_end_ms, view_id=view_id, run_id=run_id, emit_scored=False
        )
        for row, s in zip(attempt_rows, updated_shots):
            ha = s.get('hub_active')
            row['hub_active'] = ha
            if isinstance(row.get('value'), dict):
                row['value']['hub_active'] = ha
            if isinstance(row.get('evidence'), dict):
                row['evidence']['hub_active'] = ha
            if not row.get('track_key') and row.get('team_key') in track_key_by_team:
                row['track_key'] = track_key_by_team[row['team_key']]
            rows.append(row)
            if s.get('track_id') is not None:
                ballistic_tids.add(s['track_id'])
            tk = _canon_team_key(s.get('team_key')) or s.get('alliance')
            for t_key in ('t_ms', 'first_seen_ms'):
                if s.get(t_key) is not None:
                    ballistic_times_by_team.setdefault(tk, []).append(int(s[t_key]))

    # Map launches to entries for attempt result / score_ms
    entry_by_launch = {}
    for e in attributed_entries:
        if e.get('track_id') is not None and e.get('attribution') == 'same_track':
            entry_by_launch[('tid', e['track_id'])] = e
        elif e.get('launch_ms') is not None and e.get('team_key'):
            key = ('time', _canon_team_key(e['team_key']), int(e['launch_ms']))
            entry_by_launch.setdefault(key, e)

    # 2. Emit fuel_attempt rows from fuel_counter launches not already covered by ballistic shots
    if emit_attempts and launch_list:
        for idx, launch in enumerate(launch_list):
            t_ms = int(launch['t_ms'])
            if t_ms < 0 or t_ms > MATCH_MS:
                continue
            tid = launch.get('track_id')
            if tid is not None and tid in ballistic_tids:
                continue
            raw_tk = _canon_team_key(launch.get('team_key'))
            alliance = launch.get('alliance')
            dedup_key = raw_tk or alliance
            if any(abs(t_ms - bt) <= BALLISTIC_DEDUP_MS for bt in ballistic_times_by_team.get(dedup_key, ())):
                continue
            valid_roster = (raw_tk is None) or (roster_set is None) or (raw_tk in roster_set)
            team_key = raw_tk if valid_roster else None
            track_key = (launch.get('track_key') or track_key_by_team.get(team_key)) if valid_roster else None
            linked_e = (
                entry_by_launch.get(('tid', tid))
                if tid is not None and ('tid', tid) in entry_by_launch
                else entry_by_launch.get(('time', raw_tk, t_ms))
            )
            score_ms = int(linked_e['t_ms']) if linked_e is not None else None
            result = 'scored' if linked_e is not None else 'unknown'
            flight_s = round(max(0.1, (score_ms - t_ms) / 1000.0), 2) if score_ms is not None and score_ms > t_ms else None
            ha = hub_active_at(alliance, score_ms if score_ms is not None else t_ms, auto_winner)
            conf = float(launch.get('confidence') or 0.80)
            launch_xyz = [launch['x'], launch['y'], 0.6] if (launch.get('x') is not None and launch.get('y') is not None) else None
            target = f'{alliance}_hub' if alliance in ('red', 'blue') else None
            started_ms = max(0, t_ms)
            ended_ms = max(started_ms + 300, score_ms) if score_ms is not None else started_ms + 700
            val = {
                'count': 1,
                'shot': idx,
                'source': 'fuel_counter',
                'method': 'robot_launch',
                'result': result,
                'score_ms': score_ms,
                'flight_s': flight_s,
                'hub_active': ha,
                'target': target,
                'launch': launch_xyz,
                'launch_uv': [launch['u'], launch['v']] if (launch.get('u') is not None and launch.get('v') is not None) else None,
                'distance_to_hub_m': launch.get('distance_to_hub_m'),
                'attribution': launch.get('attribution', 'confident'),
                'confidence': conf,
            }
            rows.append({
                'observation_type': 'fuel_attempt',
                'team_key': team_key,
                'track_key': track_key,
                'alliance': alliance,
                'phase': _phase(t_ms, scoring=False, auto_end_ms=auto_end_ms, grace_ms=grace_ms),
                'started_ms': started_ms,
                'ended_ms': ended_ms,
                'confidence': conf,
                'source': 'fuel_counter',
                'hub_active': ha,
                'view_id': view_id,
                'vision_run_id': run_id,
                'value': val,
                'evidence': {
                    'method': 'robot_launch',
                    'source': 'fuel_counter',
                    'hub_active': ha,
                    'attribution': launch.get('attribution', 'confident'),
                },
            })

    # 3. Emit ONE fuel_scored row per individual hub entry (count: 1, source: 'fuel_counter')
    for idx, entry in enumerate(attributed_entries):
        t_ms = int(entry['t_ms'])
        if t_ms < 0 or t_ms > MATCH_MS:
            continue
        alliance = entry.get('alliance')
        hub_name = entry.get('hub') or (f'{alliance}_hub' if alliance in ('red', 'blue') else None)
        raw_tk = _canon_team_key(entry.get('team_key'))
        valid_roster = (raw_tk is None) or (roster_set is None) or (raw_tk in roster_set)
        team_key = raw_tk if valid_roster else None
        track_key = (entry.get('track_key') or track_key_by_team.get(team_key)) if valid_roster else None
        launch_ms = entry.get('launch_ms')
        phase = _phase(t_ms, scoring=True, launch_ms=launch_ms, auto_end_ms=auto_end_ms, grace_ms=grace_ms)
        ha = entry['hub_active'] if 'hub_active' in entry else hub_active_at(alliance, t_ms, auto_winner)
        conf = float(entry.get('confidence') or 0.75)
        attr_method = entry.get('attribution') or ('unattributed' if (team_key is None and track_key is None) else 'flight_time')
        attr_conf = float(entry.get('attribution_confidence') if entry.get('attribution_confidence') is not None else (0.0 if (team_key is None and track_key is None) else conf))
        started_ms = max(0, t_ms)
        ended_ms = started_ms + 300
        val = {
            'count': 1,
            'entry_index': idx,
            'hub': hub_name,
            'source': 'fuel_counter',
            'method': 'hub_entry',
            'hub_active': ha,
            'launch_ms': launch_ms,
            'flight_s': entry.get('flight_s'),
            'attribution': attr_method,
            'attribution_confidence': attr_conf,
            'distance_to_hub_m': entry.get('distance_to_hub_m'),
            'source_continuous': entry.get('source_continuous'),
            'confidence': conf,
        }
        rows.append({
            'observation_type': 'fuel_scored',
            'team_key': team_key,
            'track_key': track_key,
            'alliance': alliance,
            'phase': phase,
            'started_ms': started_ms,
            'ended_ms': ended_ms,
            'confidence': conf,
            'source': 'fuel_counter',
            'hub_active': ha,
            'view_id': view_id,
            'vision_run_id': run_id,
            'value': val,
            'evidence': {
                'method': 'hub_entry',
                'source': 'fuel_counter',
                'hub': hub_name,
                'hub_active': ha,
                'attribution': attr_method,
                'attribution_confidence': attr_conf,
                'launch_ms': launch_ms,
                'source_continuous': entry.get('source_continuous'),
                'count': 1,
            },
        })

    return rows


def _fallback_rows_from_ballistic_shots(
    shots,
    robot_tracks=(),
    *,
    roster_keys=None,
    auto_end_ms=AUTO_END_MS,
    grace_ms=AUTO_GRACE_MS,
    view_id=None,
    run_id=None,
    emit_attempts=True,
    stats=None,
):
    """Fallback when raw `ball_tracks` are absent (e.g. unit tests mocking only ballistic `shots`)."""
    import fuel as fuelmod
    # A fit's nearest-robot label alone is not source evidence. Require the observed beginning of that same flight
    # inside one measured body, across all robots. Do not mutate the caller's diagnostic shots.
    shots = [copy.deepcopy(shot) for shot in shots or ()]
    for shot in shots:
        points = sorted((p for p in (shot.get('ball_track') or []) if isinstance(p, dict) and p.get('t') is not None),
                        key=lambda point: point['t'])
        box = points[0].get('box') if len(points) >= 3 else None
        observed = False
        if shot.get('source_continuous') is True and isinstance(box, (list, tuple)) and len(box) == 4 and all(isinstance(v, (int, float)) and np.isfinite(v) for v in box):
            u, v = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            observed = any(_canon_team_key(robot.get('team_key')) == _canon_team_key(shot.get('team_key'))
                           and _unique_observed_origin(robot, robot_tracks, points[0]['t'], u, v)
                           for robot in robot_tracks)
        shot['source_in_body'] = observed
        if not observed:
            shot['team_key'], shot['track_key'] = None, None
    roster_set = _normalize_roster_keys(roster_keys, robot_tracks=robot_tracks, shots=shots)
    effective_roster = roster_set if roster_set is not None else {
        _canon_team_key(s.get('team_key')) for s in (shots or ()) if s.get('team_key')
    } | {
        _canon_team_key(r.get('team_key')) for r in (robot_tracks or ()) if r.get('team_key')
    }
    track_key_by_team = {
        _canon_team_key(r.get('team_key')): r.get('track_key')
        for r in (robot_tracks or ())
        if r.get('team_key') and r.get('track_key')
    }

    scored_entries = []
    for s in shots or ():
        if s.get('result') != 'scored':
            continue
        alliance = s.get('alliance') or ('blue' if s.get('target') == 'blue_hub' else 'red')
        t_ms = int(s.get('score_ms') if s.get('score_ms') is not None else s.get('t_ms', 0))
        scored_entries.append({
            't_ms': t_ms,
            'launch_ms': int(s.get('t_ms', t_ms)),
            'alliance': alliance,
            'hub': f'{alliance}_hub',
            'team_key': _canon_team_key(s.get('team_key')),
            'track_key': s.get('track_key') or track_key_by_team.get(_canon_team_key(s.get('team_key'))),
            'attribution': 'same_track' if s.get('source_in_body') else 'unattributed',
            'source_in_body': s.get('source_in_body') is True,
            'source_continuous': s.get('source_continuous') is True,
        })
    auto_winner = auto_winner_from_entries(scored_entries, auto_end_ms=auto_end_ms, grace_ms=grace_ms)
    for entry in scored_entries:
        entry['hub_active'] = hub_active_at(entry['alliance'], entry['t_ms'], auto_winner)
        entry['phase'] = _phase(entry['t_ms'], scoring=True, launch_ms=entry['launch_ms'],
                                auto_end_ms=auto_end_ms, grace_ms=grace_ms)

    raw_rows = fuelmod.shots_to_observations(
        shots, effective_roster, auto_end_ms, view_id=view_id, run_id=run_id, emit_scored=True
    )
    rows = []
    for r in raw_rows:
        if not emit_attempts and r.get('observation_type') == 'fuel_attempt':
            continue
        t_check = r.get('value', {}).get('score_ms') if r.get('observation_type') == 'fuel_scored' else r.get('started_ms')
        if t_check is None:
            t_check = r.get('started_ms')
        ha = hub_active_at(r.get('alliance'), t_check, auto_winner)
        r['hub_active'] = ha
        if isinstance(r.get('value'), dict):
            r['value']['hub_active'] = ha
        if isinstance(r.get('evidence'), dict):
            r['evidence']['hub_active'] = ha
        if not r.get('track_key') and r.get('team_key') in track_key_by_team:
            r['track_key'] = track_key_by_team[r['team_key']]
        rows.append(r)

    shot_launches = [
        {
            't_ms': int(s.get('t_ms', 0)),
            'team_key': _canon_team_key(s.get('team_key')),
            'alliance': s.get('alliance') or ('blue' if s.get('target') == 'blue_hub' else 'red'),
        }
        for s in (shots or ())
    ]
    summary = summarize(scored_entries, shot_launches, robot_tracks, stats=stats, attributed=scored_entries)
    return rows, summary


def build_fuel_rows(
    ball_tracks=(),
    robot_tracks=(),
    *,
    source_tracks=None,
    calibration=None,
    camera=None,
    h_ref_to_field=None,
    hubs=None,
    polygons=None,
    offset_s=0.0,
    width=1280,
    height=720,
    shots=(),
    roster_keys=None,
    auto_end_ms=AUTO_END_MS,
    grace_ms=AUTO_GRACE_MS,
    view_id=None,
    run_id=None,
    emit_attempts=True,
    match_ms=MATCH_MS,
    stats=None,
    video=None,
    opening_mode=None,
    **_extra,
):
    """Turn `ball_tracks` + `robot_tracks` + calibration/camera (+ optional ballistic `shots`) into
    `(observations, counter_summary)`.

    When `ball_tracks` is non-empty, counts every hub entry (`fuel_count.count_entries`: counter v1.0.4b's
    height-projected openings and HUB-light schedule read from `video` when the calibration has a height projection,
    else v1's rings; `stats['counter']` says which) and every robot launch
    (`fuel_count.launches`), attributes entries to robots (same-track, solo launch, flight time + direction,
    proximity), marks `hub_active` per ball, and emits one `fuel_scored` row per ball (`source: 'fuel_counter'`,
    `count: 1`) plus `fuel_attempt` rows. When `ball_tracks` is empty and `shots` is provided, falls back to
    emitting observations from `shots` with `hub_active` populated.
    """
    ball_tracks = list(ball_tracks or ())
    robot_tracks = list(robot_tracks or ())
    shots = list(shots or ())
    hubs = hubs or default_hubs()

    if not ball_tracks and shots and not source_tracks:
        return _fallback_rows_from_ballistic_shots(
            shots,
            robot_tracks,
            roster_keys=roster_keys,
            auto_end_ms=auto_end_ms,
            grace_ms=grace_ms,
            view_id=view_id,
            run_id=run_id,
            emit_attempts=emit_attempts,
            stats=stats,
        )

    polys, resolved_camera = _resolve_camera_and_polygons(
        polygons=polygons,
        camera=camera,
        calibration=calibration,
        h_ref_to_field=h_ref_to_field,
        hubs=hubs,
        width=width,
        height=height,
    )
    if opening_mode is None:
        opening_mode = (stats or {}).get('opening_detection')
    entries, shift1_side, counter_info = count_entries(
        ball_tracks, offset_s, calibration=calibration, hubs=hubs, width=width, height=height, polygons=polys,
        video=video, opening_mode=opening_mode,
    )
    entries = [e for e in entries if 0 <= e['t_ms'] <= match_ms]
    stats = {**dict(stats or {}), **counter_info}
    launch_list = [
        s for s in launches(ball_tracks, robot_tracks, offset_s, width, height, hubs=hubs, camera=resolved_camera)
        if 0 <= s['t_ms'] <= match_ms
    ]

    import robot_launches
    exits, exit_stats = robot_launches.observed_exits(ball_tracks if source_tracks is None else source_tracks, robot_tracks, offset_s, width, height)
    if not entries and not launch_list and not exits and shots:
        return _fallback_rows_from_ballistic_shots(
            shots,
            robot_tracks,
            roster_keys=roster_keys,
            auto_end_ms=auto_end_ms,
            grace_ms=grace_ms,
            view_id=view_id,
            run_id=run_id,
            emit_attempts=emit_attempts,
            stats=stats,
        )

    attributed_entries, _, _ = attribute_entries(
        entries,
        launch_list,
        robot_tracks=robot_tracks,
        width=width,
        height=height,
        hubs=hubs,
        ballistic_shots=shots,
        camera=resolved_camera,
        shift1_side=shift1_side,
    )
    attributed_entries = robot_launches.mark_nearer_body_entries(attributed_entries, ball_tracks, robot_tracks, exits, offset_s, width, height)
    attributed_entries = evidenced_shooter_entries(attributed_entries)
    attributed_entries = robot_launches.link_same_ball_entries(attributed_entries, exits, ball_tracks,
                                                               ball_tracks if source_tracks is None else source_tracks)
    import shooter_inference
    attributed_entries, stats['shooter_inference'] = shooter_inference.infer_shooters(attributed_entries, exits, ball_tracks, robot_tracks, offset_s, width, height)
    summary = summarize(entries, launch_list, robot_tracks, stats=stats, attributed=attributed_entries, shift1_side=shift1_side)
    summary['stats']['launch_track_pool'] = 'linked_plus_short_v1' if source_tracks is not None else 'legacy_linked'
    exit_stats['track_pool'] = summary['stats']['launch_track_pool']
    summary['launches'] = robot_launches.summarize_exits(exits, robot_tracks, exit_stats)
    import period_model
    openings = entry_openings(calibration, hubs, width, height) if calibration else None
    openings = fc_lifted(openings, (stats or {}).get('ring_lift_px'))
    upper = hub_openings(calibration, hubs, width, height, rim_z=TRUE_OPENING_MODEL_Z) if calibration else None
    summary = period_model.apply_estimate(summary, ball_tracks, openings, offset_s, entries, shift1_side,
                                          upper_openings=upper)
    rows = to_observations(
        attributed_entries,
        launch_list,
        robot_tracks=robot_tracks,
        shots=shots,
        roster_keys=roster_keys,
        auto_end_ms=auto_end_ms,
        grace_ms=grace_ms,
        view_id=view_id,
        run_id=run_id,
        emit_attempts=emit_attempts,
        auto_winner=summary.get('auto_winner'),
        width=width,
        height=height,
        hubs=hubs,
        camera=resolved_camera,
    )
    return rows, summary


def fuel_observations(*args, **kwargs):
    """Same arguments as `build_fuel_rows`, returning only the `vision_observations` list."""
    rows, _ = build_fuel_rows(*args, **kwargs)
    return rows
