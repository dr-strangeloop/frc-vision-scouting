#!/usr/bin/env python3
"""Run vision scouting on one match, fully on a local GPU stack or fully on Gemini.

    # local GPU: local detector and local VLM (LOCAL_VLM_URL), no API cost
    python3 scripts/vision/run_match.py --match 2026cc_qm1 --youtube MBEj_crYoBA --engine local --weights best.pt --start 11

    # just the autonomous period (downloads ~25 s of video, a minute or two end to end)
    python3 scripts/vision/run_match.py --match 2026cc_qm1 --youtube MBEj_crYoBA --auto-only

    # no GPU, no local weights: Gemini boxes + classical bumper colour fallback not needed
    python3 scripts/vision/run_match.py --match 2026cc_qm1 --youtube MBEj_crYoBA --engine gemini

The field calibration is read from --calibration FILE. Results are written as JSON to --out (or
stdout); treat them as unreviewed.
"""
import argparse
import concurrent.futures as cf
import json
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import calibration_reuse  # noqa: E402
import contextlib  # noqa: E402
import events as ev  # noqa: E402
import fuel as fuelmod  # noqa: E402
import fuel_count as fuelcount  # noqa: E402
import identity  # noqa: E402
import media  # noqa: E402
import preflight  # noqa: E402
import vram  # noqa: E402
from common import AUTO_END_MS, AUTO_GRACE_MS, FIELD, env, log, roster_from_tba, tba_match  # noqa: E402
from detectors import ColorBumperDetector, GeminiBoxDetector, UltralyticsDetector  # noqa: E402
from pipeline import build_tracks, iter_frames, register_and_project, registration_problem  # noqa: E402
from camera3d import camera_from_pixel_homography  # noqa: E402
from roi import combined_mask  # noqa: E402
from vlm import BudgetExceeded, GEMINI_THINKING, GEMINI_TIERS, UsageMeter, make_client  # noqa: E402

def _load_fuel_rows():
    try:
        import fuel_rows as _fr
        return _fr
    except ImportError:
        import importlib.util
        for entry in sys.path:
            if not entry:
                continue
            fr_path = Path(entry) / 'fuel_rows.py'
            fc_path = Path(entry) / 'fuel_count.py'
            if fr_path.exists() and fc_path.exists():
                try:
                    spec_fc = importlib.util.spec_from_file_location('fuel_count', fc_path)
                    mod_fc = importlib.util.module_from_spec(spec_fc)
                    sys.modules['fuel_count'] = mod_fc
                    spec_fc.loader.exec_module(mod_fc)
                    spec_fr = importlib.util.spec_from_file_location('fuel_rows', fr_path)
                    mod_fr = importlib.util.module_from_spec(spec_fr)
                    sys.modules['fuel_rows'] = mod_fr
                    spec_fr.loader.exec_module(mod_fr)
                    return mod_fr
                except Exception:  # noqa: BLE001
                    return None
        return None


fuelrows = _load_fuel_rows()

# An FRC 2026 match is 20 s autonomous + 3 s transition + 140 s teleop = 163 s.
MATCH_SECONDS = 163
AUTO_WINDOW_S = (AUTO_END_MS + 3000) / 1000
# tracking_bench.run(FieldTracker, fps=2) vs fps=4 shows purity drops 0.05-0.10 and ID switches double at 2 fps
# because FieldTracker._predict fits velocity over the last 0.6 s and tentative tracks expire at 0.5 s; keep 4.
DEFAULT_TELEOP_FPS = 4.0


@contextlib.contextmanager
def _capture_linked_tracks(module, wrapper):
    """Capture linked ball tracks from `fuel.link_tracks` when `run_fuel_pass` returns a 2-tuple without `.tracks`."""
    if wrapper is None or not hasattr(module, 'link_tracks'):
        yield
        return
    orig = module.link_tracks
    module.link_tracks = wrapper
    try:
        yield
    finally:
        module.link_tracks = orig


def _hub_active_at(alliance, t_ms, auto_winner, auto_end_ms=AUTO_END_MS, grace_ms=AUTO_GRACE_MS):
    teleop_start = auto_end_ms + grace_ms
    trans_end = teleop_start + 10_000
    shift_ms = 25_000
    endgame_start = trans_end + 4 * shift_ms
    if t_ms < trans_end or t_ms >= endgame_start:
        return True
    if not auto_winner or alliance not in ('red', 'blue'):
        return None
    shift = int((t_ms - trans_end) // shift_ms)
    winner_active = (shift % 2) == 1
    return winner_active if alliance == auto_winner else (not winner_active)


def _fallback_build_fuel_rows(
    ball_tracks=(),
    robot_tracks=(),
    *,
    calibration=None,
    camera=None,
    h_ref_to_field=None,
    hubs=None,
    offset_s=0.0,
    width=1280,
    height=720,
    shots=(),
    roster_keys=None,
    auto_end_ms=AUTO_END_MS,
    grace_ms=AUTO_GRACE_MS,
    view_id=None,
    run_id=None,
    stats=None,
    video=None,
):
    """Build per-ball vision fuel rows when `fuel_rows.py` is not imported directly (same counter as `fuel_rows`)."""
    roster_set = set(roster_keys) if roster_keys is not None else None
    track_key_by_team = {t['team_key']: t.get('track_key') for t in (robot_tracks or []) if t.get('team_key')}
    hubs = hubs or fuelmod.hubs_from_field()
    if camera is None and h_ref_to_field is not None and width > 0 and height > 0:
        camera = camera_from_pixel_homography(h_ref_to_field, int(width), int(height))

    if ball_tracks and camera is not None:
        polygons = fuelcount.hub_polygons(camera, hubs)
        opening_mode = (stats or {}).get('opening_detection')
        entries = []
        for tr_idx, tr in enumerate(ball_tracks):
            found, _, _ = fuelcount.count_entries([tr], offset_s, calibration=calibration, hubs=hubs, width=int(width),
                                                  height=int(height), polygons=polygons, opening_mode=opening_mode)
            for e in found:
                entries.append({**e, '_track_idx': tr_idx})
        entries.sort(key=lambda e: e['t_ms'])
        # Counter v1.0.1+: the shift schedule comes from the HUB lights (read once), not from vision auto counts.
        _, shift1_side, counter_info = fuelcount.count_entries([], offset_s, calibration=calibration, hubs=hubs, width=int(width),
                                                               height=int(height), polygons=polygons, video=video,
                                                               opening_mode=opening_mode)
        stats = {**dict(stats or {}), **counter_info}

        launch_list = []
        for tr_idx, tr in enumerate(ball_tracks):
            found = fuelcount.launches([tr], robot_tracks, offset_s, int(width), int(height))
            for l_item in found:
                launch_list.append({**l_item, '_track_idx': tr_idx})
        launch_list.sort(key=lambda s: s['t_ms'])

        # Combine launches from robot boxes and ballistic shots for entry attribution
        cand_launches = [
            {
                't_ms': int(l_item['t_ms']),
                'team_key': l_item.get('team_key'),
                'alliance': l_item.get('alliance'),
                '_track_idx': l_item.get('_track_idx'),
                'source': 'fuel_counter',
            }
            for l_item in launch_list
        ]
        for s_idx, s in enumerate(shots or []):
            t_ms = int(s.get('t_ms', 0))
            tk = s.get('team_key')
            al = s.get('alliance') or ('blue' if s.get('target') == 'blue_hub' else ('red' if s.get('target') == 'red_hub' else None))
            if not any(c.get('team_key') == tk and abs(c['t_ms'] - t_ms) <= 180 for c in cand_launches):
                cand_launches.append({'t_ms': t_ms, 'team_key': tk, 'alliance': al, '_track_idx': None, '_shot_idx': s_idx, 'source': 'fuel_ballistic'})
        cand_launches.sort(key=lambda c: c['t_ms'])

        auto_cutoff = auto_end_ms + grace_ms
        red_auto = sum(1 for e in entries if e['alliance'] == 'red' and 0 <= e['t_ms'] <= auto_cutoff)
        blue_auto = sum(1 for e in entries if e['alliance'] == 'blue' and 0 <= e['t_ms'] <= auto_cutoff)
        if shift1_side is not None:
            auto_winner = fuelcount.auto_winner_from_shift1(shift1_side)
        else:
            auto_winner = ('red' if red_auto > blue_auto else 'blue') if abs(red_auto - blue_auto) > 3 else None

        used_launches = set()
        scored_rows = []
        for entry in entries:
            e_ms = int(entry['t_ms'])
            if e_ms < -2000 or e_ms > fuelcount.MATCH_MS:
                continue
            alliance = entry['alliance']
            tr_idx = entry.get('_track_idx')
            chosen = None
            attr_method = 'unattributed'
            attr_conf = 0.55

            # 1. Same ball track from robot launch to hub entry
            if tr_idx is not None:
                for c_idx, c in enumerate(cand_launches):
                    if c_idx not in used_launches and c.get('_track_idx') == tr_idx and c.get('alliance') == alliance and c.get('team_key'):
                        chosen = (c_idx, c)
                        attr_method = 'same_track'
                        attr_conf = 0.92
                        break

            # 2. Launch in [300ms, 3000ms] before hub entry
            if chosen is None:
                in_win = [
                    (c_idx, c)
                    for c_idx, c in enumerate(cand_launches)
                    if c_idx not in used_launches and c.get('alliance') == alliance and c.get('team_key') and 300 <= (e_ms - c['t_ms']) <= 3000
                ]
                if not in_win:
                    in_win = [
                        (c_idx, c)
                        for c_idx, c in enumerate(cand_launches)
                        if c.get('alliance') == alliance and c.get('team_key') and 300 <= (e_ms - c['t_ms']) <= 3000
                    ]
                if in_win:
                    distinct_teams = {c['team_key'] for _, c in in_win}
                    best_pair = min(in_win, key=lambda item: abs((e_ms - item[1]['t_ms']) - 1100))
                    chosen = best_pair
                    if len(distinct_teams) == 1:
                        attr_method = 'solo_launch'
                        attr_conf = 0.86
                    else:
                        attr_method = 'flight_time'
                        attr_conf = 0.72

            team_key = None
            launch_ms = None
            if chosen is not None:
                used_launches.add(chosen[0])
                raw_tk = chosen[1].get('team_key')
                team_key = raw_tk if (roster_set is None or raw_tk in roster_set) else None
                launch_ms = int(chosen[1]['t_ms'])
                shot_idx = chosen[1].get('_shot_idx')
                if shot_idx is not None and 0 <= shot_idx < len(shots):
                    shots[shot_idx]['result'] = 'scored'
                    shots[shot_idx]['score_ms'] = max(0, e_ms)

            started_ms = max(0, e_ms)
            phase = 'auto' if started_ms <= auto_cutoff and (launch_ms is None or launch_ms <= auto_end_ms) else 'teleop'
            hub_active = _hub_active_at(alliance, started_ms, auto_winner, auto_end_ms=auto_end_ms, grace_ms=grace_ms)
            flight_s = round((started_ms - launch_ms) / 1000.0, 2) if launch_ms is not None and started_ms >= launch_ms else None
            val = {
                'count': 1,
                'hub': entry['hub'],
                'source': 'fuel_counter',
                'method': 'hub_entry',
                'hub_active': hub_active,
                'launch_ms': launch_ms,
                'flight_s': flight_s,
                'attribution': attr_method,
                'attribution_confidence': attr_conf,
            }
            ev_dict = {
                'method': 'hub_entry',
                'source': 'fuel_counter',
                'hub': entry['hub'],
                'hub_active': hub_active,
                'attribution': attr_method,
                'launch_ms': launch_ms,
                'count': 1,
            }
            scored_rows.append({
                'observation_type': 'fuel_scored',
                'team_key': team_key,
                'track_key': track_key_by_team.get(team_key) if team_key else None,
                'alliance': alliance,
                'phase': phase,
                'value': val,
                'started_ms': started_ms,
                'ended_ms': started_ms + 300,
                'confidence': attr_conf,
                'evidence': ev_dict,
                'view_id': view_id,
                'vision_run_id': run_id,
                'source': 'fuel_counter',
                'hub_active': hub_active,
            })

        attempt_rows = []
        if shots:
            attempt_rows = fuelmod.shots_to_observations(shots, roster_set or set(), auto_end_ms, view_id, run_id, emit_scored=False)
            for r in attempt_rows:
                entry_t = (r.get('value') or {}).get('score_ms') or r.get('started_ms') or 0
                ha = _hub_active_at(r.get('alliance'), int(entry_t), auto_winner, auto_end_ms=auto_end_ms, grace_ms=grace_ms) if r.get('value', {}).get('result') == 'scored' else None
                r['hub_active'] = ha
                if isinstance(r.get('value'), dict):
                    r['value']['hub_active'] = ha
        for l_item in launch_list:
            l_ms = max(0, int(l_item['t_ms']))
            tk = l_item.get('team_key')
            tk_clean = tk if (roster_set is None or tk in roster_set) else None
            al = l_item.get('alliance')
            if any(r.get('team_key') == tk_clean and abs(int(r.get('started_ms', 0)) - l_ms) <= 180 for r in attempt_rows):
                continue
            ha = _hub_active_at(al, l_ms, auto_winner, auto_end_ms=auto_end_ms, grace_ms=grace_ms)
            attempt_rows.append({
                'observation_type': 'fuel_attempt',
                'team_key': tk_clean,
                'track_key': track_key_by_team.get(tk_clean) if tk_clean else None,
                'alliance': al,
                'phase': 'auto' if l_ms <= auto_end_ms else 'teleop',
                'value': {
                    'count': 1,
                    'source': 'fuel_counter',
                    'method': 'robot_launch',
                    'result': 'unknown',
                    'hub_active': ha,
                    'target': f'{al}_hub' if al in ('red', 'blue') else None,
                    'confidence': 0.8,
                },
                'started_ms': l_ms,
                'ended_ms': l_ms + 600,
                'confidence': 0.8,
                'evidence': {'method': 'robot_launch', 'source': 'fuel_counter'},
                'view_id': view_id,
                'vision_run_id': run_id,
                'source': 'fuel_counter',
                'hub_active': ha,
            })

        summary = fuelcount.summarize(entries, launch_list, robot_tracks, stats=stats, shift1_side=shift1_side)
        summary['auto_winner'] = auto_winner
        for color in ('red', 'blue'):
            a_sum = (summary.get('alliances') or {}).get(color)
            if isinstance(a_sum, dict):
                c_rows = [r for r in scored_rows if r.get('alliance') == color]
                a_sum.setdefault('balls_in_hub', len(c_rows))
                a_sum.setdefault('active_scored', sum(1 for r in c_rows if r.get('hub_active') is not False))
                a_sum.setdefault('inactive_scored', sum(1 for r in c_rows if r.get('hub_active') is False))
        return attempt_rows + scored_rows, summary

    if shots:
        auto_cutoff = auto_end_ms + grace_ms
        red_auto = sum(1 for s in shots if s.get('result') == 'scored' and (s.get('alliance') == 'red' or s.get('target') == 'red_hub') and int(s.get('score_ms') or s.get('t_ms') or 0) <= auto_cutoff)
        blue_auto = sum(1 for s in shots if s.get('result') == 'scored' and (s.get('alliance') == 'blue' or s.get('target') == 'blue_hub') and int(s.get('score_ms') or s.get('t_ms') or 0) <= auto_cutoff)
        auto_winner = ('red' if red_auto > blue_auto else 'blue') if abs(red_auto - blue_auto) > 3 else None
        rows = fuelmod.shots_to_observations(shots, roster_set or set(), auto_end_ms, view_id, run_id, emit_scored=True)
        for r in rows:
            t_check = int(r.get('started_ms') or 0)
            ha = _hub_active_at(r.get('alliance'), t_check, auto_winner, auto_end_ms=auto_end_ms, grace_ms=grace_ms)
            r['hub_active'] = ha
            if isinstance(r.get('value'), dict):
                r['value']['hub_active'] = ha
        return rows, None

    return [], None


def build_vision_fuel_rows(
    ball_tracks=(),
    robot_tracks=(),
    *,
    source_tracks=None,
    calibration=None,
    camera=None,
    h_ref_to_field=None,
    hubs=None,
    offset_s=0.0,
    width=1280,
    height=720,
    shots=(),
    roster_keys=None,
    auto_end_ms=AUTO_END_MS,
    view_id=None,
    run_id=None,
    stats=None,
    video=None,
):
    """Build per-ball fuel observations from `ball_tracks` + `shots`, delegating to `fuel_rows.build_fuel_rows` when present.

    `video` is the local match video the HUB lights are read from (counter v1.0.1+); without it the shift schedule
    stays unknown rather than guessed from vision."""
    global fuelrows
    if fuelrows is None:
        fuelrows = _load_fuel_rows()
    if fuelrows is not None and hasattr(fuelrows, 'build_fuel_rows'):
        res = fuelrows.build_fuel_rows(
            ball_tracks,
            robot_tracks,
            source_tracks=source_tracks,
            calibration=calibration,
            camera=camera,
            h_ref_to_field=h_ref_to_field,
            hubs=hubs,
            offset_s=offset_s,
            width=int(width),
            height=int(height),
            shots=shots,
            roster_keys=roster_keys,
            auto_end_ms=auto_end_ms,
            view_id=view_id,
            run_id=run_id,
            stats=stats,
            video=video,
        )
        return res if isinstance(res, tuple) and len(res) == 2 else (list(res or []), None)
    raise RuntimeError('The shared fuel_rows module is unavailable; refusing legacy heuristic robot credits')


def sample_frames(video, start_s, end_s, auto_fps, teleop_fps=None, auto_window_s=AUTO_WINDOW_S):
    """Yield (video_seconds, bgr) at auto_fps during auto (match time < 23 s) and teleop_fps in teleop."""
    if end_s <= start_s + auto_window_s:
        yield from iter_frames(video, start_s, end_s, auto_fps)
        return
    teleop_fps = teleop_fps or auto_fps
    split_s = start_s + auto_window_s
    last_t = None
    for t, frame in iter_frames(video, start_s, split_s, auto_fps):
        if t < split_s:
            last_t = t
            yield t, frame
    teleop_start = max(split_s, last_t + 0.5 / max(auto_fps, teleop_fps)) if last_t is not None else split_s
    for t, frame in iter_frames(video, teleop_start, end_s, teleop_fps):
        if last_t is None or t > last_t + 1e-3:
            last_t = t
            yield t, frame


def write_partial_tracks(path, tracks):
    """Write {'tracks': tracks} to path atomically via a temporary file and rename."""
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f'.{target.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps({'tracks': tracks}))
    os.replace(tmp, target)


def resolve_match_window(start, end, use_gemini=False, youtube=None, gemini_cheap=None):
    """Return (start, end) video seconds. When start is known, end defaults to start + MATCH_SECONDS without a VLM call."""
    if start is None:
        if not (use_gemini and youtube):
            sys.exit('pass --start (and --end) when not using Gemini with a YouTube id')
        found_start, found_end, evidence = media.find_match_window(gemini_cheap, youtube)
        start = max(0.0, float(found_start))
        if end is None:
            if not media.plausible_match(start, found_end):
                end = start + MATCH_SECONDS
                log(f'warning: timing read {found_start:.0f}-{found_end:.0f}s is not a plausible match ({evidence}); clamping end to {end:.0f}s (start + {MATCH_SECONDS}s)')
            else:
                end = found_end
    else:
        start = max(0.0, float(start))
    if end is None or end <= start:
        if end is not None and end <= start:
            log(f'warning: end ({end:.1f}s) <= start ({start:.1f}s); clamping end to {start + MATCH_SECONDS:.0f}s (start + {MATCH_SECONDS}s)')
        end = start + MATCH_SECONDS
    return start, end


def load_calibration(path):
    data = json.loads(Path(path).read_text()) if path else None
    if not data or not data.get('homography_norm'):
        return None
    return data


def read_frame_at(video_path, t_s):
    """Decode one BGR frame near `t_s` seconds into `video_path`, or None."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(t_s)) * 1000.0)
        ok, frame = cap.read()
        if not ok:
            n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            fallback_ms = max(0.0, (n_frames - 1) / fps * 1000.0) if n_frames > 1 and fps > 0 else 0.0
            cap.set(cv2.CAP_PROP_POS_MSEC, fallback_ms)
            ok, frame = cap.read()
            if not ok and fallback_ms > 0.0:
                cap.set(cv2.CAP_PROP_POS_MSEC, 0.0)
                ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def _download_url(url, target):
    import urllib.parse
    from common import redact_secrets
    parsed = urllib.parse.urlparse(str(url or ''))
    if parsed.scheme != 'https' or not parsed.hostname:
        raise ValueError(f'refusing non-HTTPS download URL: {redact_secrets(url)!r}')
    max_bytes = int(os.environ.get('VISION_MAX_DOWNLOAD_BYTES', 500 * 1024 * 1024))
    request = urllib.request.Request(url, headers={'User-Agent': 'spartans-vision-worker'})
    with urllib.request.urlopen(request, timeout=600) as response, open(target, 'wb') as handle:
        headers = getattr(response, 'headers', None)
        expected = headers.get('Content-Length') if headers else None
        if expected is not None and str(expected).strip().isdigit() and int(str(expected).strip()) > max_bytes:
            raise RuntimeError(f'download exceeds {max_bytes} byte limit')
        written = 0
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                raise RuntimeError(f'download exceeds {max_bytes} byte limit')
            handle.write(chunk)
    return target


def _fetch_candidate_frame(cand, t_s, tmp):
    """Fetch a reference BGR frame from a candidate calibration dict around `t_s` (cached via media.download_section)."""
    import re
    t_s = max(0.0, float(t_s or 0.0))
    yt = cand.get('youtube_id') or (str(cand.get('storage_path') or '')[len('youtube:'):] if str(cand.get('storage_path') or '').startswith('youtube:') else None)
    if yt:
        video_path, offset_s = media.download_section(yt, max(0.0, t_s - 1.0), t_s + 1.0, tmp)
        return read_frame_at(video_path, t_s - offset_s)
    if cand.get('signed_url'):
        safe_view_id = re.sub(r'[^A-Za-z0-9_-]+', '_', str(cand.get('view_id') or cand.get('id') or 'view'))[:64] or 'view'
        local = Path(tmp) / f'cand_{safe_view_id}.mp4'
        if not local.exists():
            _download_url(cand['signed_url'], local)
        return read_frame_at(local, t_s)
    if cand.get('file'):
        return read_frame_at(cand['file'], t_s - float(cand.get('file_offset_s') or 0.0))
    return None


def try_reuse_event_calibrations(event_calibrations_path, video, file_offset, start, tmp):
    """Try reusing a field calibration from `--event-calibrations PATH` after `start` is known.

    Returns (reused_calibration_dict, calibration_info_dict) or (None, None).
    """
    if not event_calibrations_path:
        return None, None
    data = json.loads(Path(event_calibrations_path).read_text())
    candidates = data.get('event_calibrations', data) if isinstance(data, dict) else data
    if not candidates:
        return None, None
    target_frames = {}
    for cand in candidates:
        try:
            cal = cand.get('field_calibration') or {}
            if not cal.get('homography_norm'):
                continue
            cand_start = cand.get('match_start_s')
            if cand_start is None and cand.get('sync_offset_ms') is not None:
                cand_start = float(cand.get('sync_offset_ms') or 0) / 1000.0
            ref_t_s = cal.get('ref_t_s')
            if ref_t_s is not None and cand_start is not None and float(cand_start) > 0:
                rel_t_s = float(ref_t_s) - float(cand_start)
            elif ref_t_s is not None and (cand_start is None or float(cand_start) == 0):
                rel_t_s = float(ref_t_s) if float(ref_t_s) <= 30.0 else 9.0
            else:
                rel_t_s = 9.0
            cand_ref_t_s = float(ref_t_s) if ref_t_s is not None else (float(cand_start or 0.0) + 9.0)
            target_t_s = max(0.0, float(start) + rel_t_s)
            cand_frame = _fetch_candidate_frame(cand, cand_ref_t_s, tmp)
            if cand_frame is None:
                continue
            key = round(target_t_s, 1)
            if key not in target_frames:
                target_frames[key] = read_frame_at(video, target_t_s - file_offset)
            targ_frame = target_frames[key]
            if targ_frame is None:
                continue
            match_key = cand.get('match_key') or cand.get('view_id') or cand.get('id')
            reused = calibration_reuse.reuse(cand_frame, targ_frame, cal, reused_from=match_key)
            if reused:
                reused = dict(reused, ref_t_s=target_t_s)
                meta = reused.get('reused_from') if isinstance(reused.get('reused_from'), dict) else {}
                info = {
                    'reused_from': match_key,
                    'view_id': cand.get('view_id') or cand.get('id'),
                    'residual_px': meta.get('residual_px'),
                    'inliers': meta.get('inliers'),
                }
                log(f'reusing calibration from {match_key}')
                return reused, info
        except BaseException as error:  # noqa: BLE001 - never block scouting if a candidate fails
            log(f'calibration reuse candidate failed: {error}')
    return None, None


def pixel_homography(calibration, width, height):
    """normalized image -> field meters, to pixel -> field meters for a frame of this size."""
    h_norm = np.array(calibration['homography_norm'], float)
    return h_norm @ np.diag([1.0 / width, 1.0 / height, 1.0])


def build_detector(args, gemini, red, blue, crop_box=None):
    if args.engine == 'gemini':
        return GeminiBoxDetector(gemini, red, blue, workers=args.workers, crop_box=crop_box), min(args.fps or 4, 5)
    if args.weights:
        return UltralyticsDetector(args.weights, device=args.device, batch=args.batch), args.fps or 10
    log('no --weights given: using the classical red/blue bumper detector (a baseline; train a YOLO/RF-DETR model for production)')
    return ColorBumperDetector(), args.fps or 10


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--match', required=True)
    ap.add_argument('--youtube', help='YouTube video id of the match broadcast')
    ap.add_argument('--file', help='local video (whole broadcast or a clip). With --file-start-s if it is a clip')
    ap.add_argument('--file-offset-s', type=float, default=0.0, help='seconds of the original video before --file starts')
    ap.add_argument('--engine', choices=['local', 'gemini'], default='gemini')
    ap.add_argument('--auto-only', action='store_true', help='only the autonomous period')
    ap.add_argument('--start', type=float, help='video second the match (auto) starts; found by the VLM if omitted')
    ap.add_argument('--end', type=float, help='video second the match ends')
    ap.add_argument('--calibration', help='field calibration JSON (homography_norm, optional height projection)')
    ap.add_argument('--event-calibrations', help='JSON list of candidate calibrations from other matches of the event for reuse after match start is known')
    ap.add_argument('--weights', help='Ultralytics weights for the local detector')
    ap.add_argument('--device', default=None)
    ap.add_argument('--fps', type=float)
    ap.add_argument('--teleop-fps', type=float, default=DEFAULT_TELEOP_FPS, help='teleop sampling rate for the Gemini box detector (kept at 4: 2 fps drops tracker purity > 0.02)')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--workers', type=int, default=_default_workers())
    ap.add_argument('--budget-gb', type=float, default=128.0, help='VRAM budget for the local stack')
    ap.add_argument('--no-verify', action='store_true', help='skip the strong-model re-check of doubtful events')
    ap.add_argument('--no-fuel', action='store_true', help='skip the fuel ball tracker (needs a field calibration)')
    ap.add_argument('--gemini-model', help='use this Gemini model for every Gemini step (e.g. gemini-4-argon) instead of the default tiers')
    ap.add_argument('--roster-json', help='{"red": [...], "blue": [...]} team numbers, instead of asking TBA')
    ap.add_argument('--fuel-serial', action='store_true', help='run the ball tracker in one process (the parallel one is the default)')
    ap.add_argument('--out', help='write the result JSON here')
    ap.add_argument('--partial-out', help='write {"tracks": tracks} atomically here right after build_tracks')
    args = ap.parse_args()
    if not (args.youtube or args.file):
        sys.exit('give --youtube and/or --file')

    started = time.time()
    os.environ.setdefault('VISION_RUN_LABEL', f"{args.match} {'auto' if args.auto_only else 'scouting'} ({args.engine})")
    timings, last_mark = {}, [started]

    def mark(stage):
        """Wall-clock seconds spent on each stage, stored with the result so slow steps are visible per run."""
        now = time.time()
        timings[stage] = round(timings.get(stage, 0) + now - last_mark[0], 1)
        last_mark[0] = now
    meter = UsageMeter()
    use_gemini = args.engine == 'gemini'
    # One model for every step when a run names one (the Gemini worker): the tiers still differ in how much they think.
    gemini = lambda tier: make_client('gemini', args.gemini_model or GEMINI_TIERS[tier], meter, thinking_level=_thinking(args.gemini_model, tier))
    gemini_bulk = gemini('bulk') if use_gemini else None
    gemini_cheap = gemini('cheap') if use_gemini else None
    gemini_strong = gemini('verify') if use_gemini and not args.no_verify else None
    local_vlm = make_client('local', None, meter) if args.engine == 'local' else None
    vlm = gemini_bulk or local_vlm

    if args.engine != 'gemini':
        plan = vram.plan(args.budget_gb, 'accuracy' if args.engine == 'local' else 'detector-only')
        log(f"VRAM plan: {plan['models']} = {plan['gb']} GB of {plan['usable_gb']} GB usable ({args.budget_gb:.0f} GB budget)")
        vram.enforce(args.budget_gb)

    roster = _roster(args)
    red, blue = roster['red'], roster['blue']
    alliance_of = {**{t: 'red' for t in red}, **{t: 'blue' for t in blue}}
    # Only the teams: the inference never sees the official score.
    log(f"{args.match}: red {red} vs blue {blue}")

    # --- when does the match start?
    start, end = resolve_match_window(args.start, args.end, use_gemini=use_gemini, youtube=args.youtube, gemini_cheap=gemini_cheap)
    log(f'match starts at {start:.0f}s in the video, ends {end:.0f}s')
    mark('find_start')

    # --- video: only what is needed
    t0 = start
    geo_end = min(end, start + AUTO_WINDOW_S) if args.auto_only else end
    tmp = media.tmpdir()
    if args.file:
        video, file_offset = Path(args.file).expanduser(), args.file_offset_s
    else:
        video, file_offset = media.download_section(args.youtube, t0 - 3, geo_end + 3, tmp)
    log(f'video ready ({time.time() - started:.0f}s elapsed)')
    mark('download')
    footage = None
    try:
        footage = preflight.assess_video(video, t0 - file_offset, geo_end - file_offset)
        log(preflight.format_report(footage))
    except ValueError as error:
        log(f'footage check skipped: {error}')

    event_key = args.match.split('_')[0]
    match_id = view_id = run_id = None
    config = {'engine': args.engine, 'auto_end_ms': AUTO_END_MS, 'match_start_s': start, 'match_end_s': end, 'auto_only': args.auto_only,
              'position_reference': 'floor-contact' if args.auto_only else 'footprint-center',
              'fps': args.fps, 'kind': 'auto' if args.auto_only else 'full', 'frame': 'field-meters',
              'footage': None if footage is None else {'verdict': footage['verdict'], 'score': footage['score'], 'advice': footage['advice']}}
    background = fuel_pool = None
    calibration_info = None
    try:
        calibration = load_calibration(args.calibration)
        if calibration is None and args.event_calibrations:
            calibration, calibration_info = try_reuse_event_calibrations(args.event_calibrations, video, file_offset, start, tmp)
            mark('calibration_reuse')
            if calibration is None and args.auto_only:
                raise RuntimeError('auto detection needs the view calibrated to the field (calibration reuse from event matches did not match this camera)')
        pseudo = calibration is None
        if pseudo:
            log('NO FIELD CALIBRATION for this view: boxes will be drawn on the video, but auto paths need a calibration (Vision Scouting > Auto, or --calibration).')
        cap = cv2.VideoCapture(str(video))
        width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        ref_frame = None
        if pseudo:
            # Stand-in field scaled from the picture so tracking and smoothing still have physical-ish units.
            h_ref_to_field = np.diag([FIELD['length'] / width, FIELD['width'] / height, 1.0])
        else:
            ref_t_s = calibration.get('ref_t_s')
            ref_t = (start if ref_t_s is None else float(ref_t_s)) - file_offset
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, ref_t) * 1000.0)
            ok, ref_frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, start - file_offset) * 1000.0)
                ok, ref_frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_MSEC, 0.0)
                ok, ref_frame = cap.read()
            if not ok:
                raise RuntimeError('could not read the calibration reference frame from the video')
            h_ref_to_field = pixel_homography(calibration, width, height)
        cap.release()
        camera = None
        mask_ref = None
        if not pseudo:
            # Only the wide camera's region, and only the volume the field occupies, is analysed.
            camera = camera_from_pixel_homography(h_ref_to_field, int(width), int(height))
            mask_ref = combined_mask(calibration, camera, int(width), int(height))
            log(f'analysing {100 * float((mask_ref > 0).mean()):.0f}% of the picture (region of interest and field volume)')

        # Action detection needs only the video and the roster, so it runs alongside the robot boxes and the fuel
        # tracker instead of after them. Fuel always comes from the ball counter / tracker, never from the VLM.
        will_track_fuel = calibration is not None and not args.no_fuel
        events_future, events_timer = None, {}
        if vlm:
            background = cf.ThreadPoolExecutor(1)
            event_windows = [(start, geo_end, 4)] if geo_end <= start + AUTO_WINDOW_S else ev.plan_windows(start, geo_end)
            local_frames = (video, file_offset) if (args.engine == 'local' or not args.youtube) else None

            def detect_in_background(include_fuel=False):
                began = time.time()
                found = _detect_events(args, vlm, local_video=local_frames, windows=event_windows, red=red, blue=blue, workers=args.workers, include_fuel=include_fuel)
                events_timer['seconds'] = round(events_timer.get('seconds', 0.0) + time.time() - began, 1)
                merged = ev.merge_events(found)
                if gemini_strong and args.youtube:
                    verify_began = time.time()
                    merged = _verify(gemini_strong, args.youtube, merged, red, blue, args.workers)
                    events_timer['verify_seconds'] = round(events_timer.get('verify_seconds', 0.0) + time.time() - verify_began, 1)
                return merged
            events_future = background.submit(detect_in_background, False)

        fuel_future, fuel_timer = None, {}
        hubs = fuelmod.hubs_from_field() if will_track_fuel else None
        if will_track_fuel:
            log(f'fuel tracker: {native_fps:.0f} fps over {geo_end - t0:.0f}s')

            def track_fuel_in_background():
                began = time.time()
                captured_tracks = []
                orig_link = getattr(fuelmod, 'link_tracks', None)

                def capture_link(raw_tracks, *a, **kw):
                    linked = orig_link(raw_tracks, *a, **kw)
                    captured_tracks.clear()
                    captured_tracks.extend(linked)
                    return linked

                with _capture_linked_tracks(fuelmod, capture_link if callable(orig_link) else None):
                    # Counter v1.0.4b: wider ball detection around the height-projected HUB openings (none without one).
                    _, opening_mode, opening_region = fuelcount.opening_setup(calibration, hubs, int(width), int(height))
                    if args.fuel_serial:
                        res = fuelmod.run_fuel_pass(iter_frames(video, t0 - file_offset, geo_end - file_offset, native_fps), h_ref_to_field, ref_frame, hubs,
                                                    robot_tracks=(), offset_s=t0 - file_offset, mask_ref=mask_ref,
                                                    opening_region_ref=opening_region, opening_mode=opening_mode)
                    else:
                        res = fuelmod.run_fuel_pass_parallel(video, t0 - file_offset, geo_end - file_offset, native_fps, h_ref_to_field, ref_frame, hubs,
                                                             robot_tracks=(), offset_s=t0 - file_offset, mask_ref=mask_ref,
                                                             opening_region_ref=opening_region, opening_mode=opening_mode)
                fuel_timer['seconds'] = round(time.time() - began, 1)
                fuel_timer['ball_tracks'] = captured_tracks
                return res

            fuel_pool = cf.ThreadPoolExecutor(1)
            fuel_future = fuel_pool.submit(track_fuel_in_background)

        detector, fps = build_detector(args, gemini_bulk, red, blue, crop_box=(calibration or {}).get('roi'))
        teleop_fps = min(args.fps or args.teleop_fps or DEFAULT_TELEOP_FPS, 5) if args.engine == 'gemini' and not args.auto_only else None
        frames = sample_frames(video, t0 - file_offset, geo_end - file_offset, fps, teleop_fps) if args.engine == 'gemini' else iter_frames(video, t0 - file_offset, geo_end - file_offset, fps)
        # Gemini detection is sent several batches at a time, so hand it enough frames per flush to fill the workers.
        flush_batch = detector.batch * detector.workers if args.engine == 'gemini' else args.batch
        tracker, stats = register_and_project(frames, detector, h_ref_to_field, ref_frame=ref_frame, batch=flush_batch, keep_crops=bool(vlm), mask_ref=mask_ref, contact_only=args.auto_only)
        log(f"geometry: {stats} ({time.time() - started:.0f}s elapsed)")
        mark('robot_boxes')
        if not pseudo:
            reg_problem = registration_problem(stats, ref_t_s=(calibration or {}).get('ref_t_s'), match_start_s=start)
            if reg_problem:
                raise RuntimeError(reg_problem)
        problem = getattr(detector, 'failure_problem', lambda: None)()
        if problem:
            raise RuntimeError(problem)
        if vlm:
            _read_numbers(tracker, vlm, roster, args.workers)
        tracks = build_tracks(tracker, roster, offset_s=t0 - file_offset, pseudo=pseudo, match_key=args.match)
        mark('identity')
        for track in tracks:
            track['engine'] = args.engine
        if args.partial_out:
            write_partial_tracks(args.partial_out, tracks)
        log(f"{len(tracks)} tracks: " + ', '.join(f"{t['team_key'] or '?'}({len(t['trajectory'])}pts, id {t['identity_confidence']:.0%})" for t in tracks))

        shots, ball_tracks, source_tracks, fuel_stats = [], [], None, {}
        if fuel_future is not None:
            try:
                fuel_res = fuel_future.result()
                if isinstance(fuel_res, tuple) and len(fuel_res) == 3:
                    shots, ball_tracks, raw_fuel_stats = fuel_res
                else:
                    shots, raw_fuel_stats = fuel_res
                    ball_tracks = (
                        getattr(raw_fuel_stats, 'tracks', None)
                        or getattr(raw_fuel_stats, 'ball_tracks', None)
                        or (raw_fuel_stats.get('ball_tracks') if isinstance(raw_fuel_stats, dict) else None)
                        or fuel_timer.get('ball_tracks')
                        or []
                    )
                source_tracks = getattr(raw_fuel_stats, 'source_tracks', None)
                fuel_stats = {k: v for k, v in dict(raw_fuel_stats or {}).items() if k != 'ball_tracks'}
                if any(s.get('team_key') is None for s in shots):
                    fuelmod.attribute_shots(shots, tracks)
                log(f'fuel: {len(shots)} shots tracked ({fuel_stats}) ({time.time() - started:.0f}s elapsed)')
            except BudgetExceeded:
                raise
            except Exception as error:  # noqa: BLE001 - continue run with non-fuel observations if ball tracker fails
                log(f'warning: fuel tracker failed ({error})')
                shots, ball_tracks, source_tracks, fuel_stats = [], [], None, {'error': str(error)}
            mark('fuel')
            if 'seconds' in fuel_timer:
                timings['fuel_parallel'] = fuel_timer['seconds']
        if fuel_pool is not None:
            fuel_pool.shutdown(wait=False)

        roster_keys = {f'frc{t}' for t in alliance_of}
        fuel_rows, fuel_count_payload = [], None
        if ball_tracks or shots or source_tracks:
            fuel_rows, fuel_count_payload = build_vision_fuel_rows(
                ball_tracks,
                tracks,
                source_tracks=source_tracks,
                calibration=calibration,
                camera=camera,
                h_ref_to_field=h_ref_to_field,
                hubs=hubs,
                offset_s=t0 - file_offset,
                width=int(width),
                height=int(height),
                shots=shots,
                roster_keys=roster_keys,
                auto_end_ms=AUTO_END_MS,
                view_id=view_id,
                run_id=run_id,
                stats=fuel_stats,
                video=video,
            )
            if ball_tracks and isinstance(fuel_count_payload, dict) and isinstance(fuel_count_payload.get('alliances'), dict):
                alliances_fc = fuel_count_payload['alliances']
                red_fc = alliances_fc.get('red') or {}
                blue_fc = alliances_fc.get('blue') or {}
                log(
                    f"counted: red {red_fc.get('scored', 0)} in, {red_fc.get('shots', 0)} shot; "
                    f"blue {blue_fc.get('scored', 0)} in, {blue_fc.get('shots', 0)} shot"
                )
                fuel_stats = {
                    **fuel_stats,
                    'fuel_counter': {
                        c: {
                            k: (alliances_fc.get(c) or {}).get(k)
                            for k in ('scored', 'balls_in_hub', 'active_scored', 'inactive_scored', 'auto_scored', 'teleop_scored', 'shots', 'unattributed')
                            if (alliances_fc.get(c) or {}).get(k) is not None
                        }
                        for c in ('red', 'blue')
                    },
                }
            else:
                fuel_count_payload = None

        reported = []
        if vlm:
            reported = events_future.result()
            mark('actions_wait')
            if 'seconds' in events_timer:
                timings['actions_parallel'] = events_timer['seconds']
            if 'verify_seconds' in events_timer:
                timings['verify_parallel'] = events_timer['verify_seconds']
        if background is not None:
            background.shutdown(wait=False)

        reported = [e for e in reported if e.get('type') not in ev.FUEL_TYPES]

        observations, skipped = [], 0
        if vlm:
            observations, skipped = ev.to_observations(reported, set(alliance_of), alliance_of, start, AUTO_END_MS, vlm.model, args.engine, view_id, run_id)
            log(f'{len(observations)} actions ({skipped} not stored)')
        observations = fuel_rows + observations
        track_key_by_team = {t['team_key']: t['track_key'] for t in tracks if t.get('team_key') and t.get('track_key')}
        for row in observations:
            if not row.get('track_key') and row.get('team_key') in track_key_by_team:
                row['track_key'] = track_key_by_team[row['team_key']]
        if meter.over_budget():
            # Calls stopped at the limit, so parts of this result are missing; say so instead of storing a partial run.
            raise RuntimeError(f'stopped at the Gemini spending limit (${meter.max_usd:.2f} per run, VISION_MAX_USD): spent ${meter.cost:.2f}. Check the video is one match, or raise the limit.')
        summary = {'seconds': round(time.time() - started), 'timings_s': timings, 'model': args.gemini_model, 'tracks': len(tracks), 'actions': len(observations), 'usage': meter.summary(), 'geometry': stats, 'fuel': fuel_stats}
        if calibration_info is not None:
            summary['calibration'] = calibration_info
        result = {'match': args.match, 'match_start_s': start, 'config': config, 'tracks': tracks, 'observations': observations, 'summary': summary}
        if fuel_count_payload is not None:
            result['fuel_count'] = fuel_count_payload
        if args.out:
            Path(args.out).write_text(json.dumps(result))
        else:
            print(json.dumps(result)[:4000])
        log(f"done in {summary['seconds']}s ({', '.join(f'{k} {v}s' for k, v in timings.items())}). Gemini usage: {meter.summary()}")
    except BaseException as error:
        raise
    finally:
        for pool in (background, fuel_pool):
            if pool is not None:
                pool.shutdown(wait=False, cancel_futures=True)
        shutil.rmtree(tmp, ignore_errors=True)


def _default_workers(default=8):
    raw = os.environ.get('VISION_GEMINI_WORKERS')
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return default


def _thinking(model, tier):
    """Thinking level for a tier. A named model gets 'low' for the cheap tier: not every model accepts 'minimal'."""
    level = GEMINI_THINKING[tier]
    return 'low' if (model and level == 'minimal') else level


def _roster(args):
    """Team numbers per alliance: from --roster-json when given (no TBA call), else from TBA."""
    from common import clean_team_list
    if args.roster_json:
        data = json.loads(args.roster_json)
        roster = {color: clean_team_list(data.get(color), lowercase=True) for color in ('red', 'blue')}
        if roster['red'] or roster['blue']:
            return roster
        log('the stored roster is empty; asking TBA')
    return roster_from_tba(tba_match(args.match, env('TBA_API_KEY')))


def _read_numbers(tracker, client, roster, workers):
    """Read bumper numbers for tracklets with missing, weak, or split identity votes, batched per alliance."""
    identity.read_tracker_numbers(
        tracker, client, roster,
        workers=max(1, int(workers or 1)),
        zoom_crops=getattr(tracker, 'zoom_crops', None),
        preprocess=True,
    )


def _detect_events(args, client, local_video, windows, red, blue, workers, include_fuel=False):
    # Gemini runs must never request fuel from the VLM; fuel comes strictly from vision ball tracking.
    if getattr(args, 'engine', 'gemini') == 'gemini':
        include_fuel = False
    def one(window):
        a, b, fps = window
        auto = window is windows[0]
        prompt = ev.window_prompt(a, b, red, blue, auto, include_fuel)
        if local_video is None:
            parts = [{'youtube': f'https://www.youtube.com/watch?v={args.youtube}', 'start': a, 'end': b, 'fps': fps}, {'text': prompt}]
        else:
            video, offset = local_video
            parts = []
            for t, frame in iter_frames(video, a - offset, b - offset, min(fps, 2)):
                ok, buf = cv2.imencode('.jpg', cv2.resize(frame, None, fx=0.6, fy=0.6), [cv2.IMWRITE_JPEG_QUALITY, 80])
                parts += [{'text': f'[t={t + offset:.1f}s]'}, {'image': buf.tobytes()}]
            parts.append({'text': prompt})
        data = client.generate_json(parts, ev.event_schema(include_fuel), max_tokens=12000)
        raw = (data.get('events') if isinstance(data, dict) else None) or []
        out = []
        for item in raw:
            if not isinstance(item, dict) or not item.get('type'):
                continue
            try:
                t_val = float(item['t'])
            except (KeyError, TypeError, ValueError):
                continue
            out.append({**item, 't': t_val})
        return out

    def progress(n, total, window, result, error):
        log(f'  window {n}/{total}: ' + (f'FAILED ({error})' if error else f'{len(result)} events'))

    events, failed = ev.run_windows(one, windows, workers, progress)
    if failed:
        log(f'warning: {len(failed)} of {len(windows)} event windows failed; the result is missing those seconds: {[f["window"] for f in failed]}')
    return events


def _verify(strong, youtube, reported, red, blue, workers):
    todo = [e for e in reported if ev.needs_verification(e)]
    log(f'verifying {len(todo)} of {len(reported)} events on the strong model')
    if not todo:
        return list(reported)

    def check(event):
        parts = [{'youtube': f'https://www.youtube.com/watch?v={youtube}', 'start': max(0, event['t'] - 4), 'end': event['t'] + 4, 'fps': 4}, {'text': ev.verify_prompt(event, red, blue)}]
        try:
            return event, strong.generate_json(parts, ev.VERIFY_SCHEMA, max_tokens=2500)
        except BudgetExceeded:
            raise
        except Exception as error:  # noqa: BLE001
            log(f'  verify failed: {error}')
            return event, None

    kept = [e for e in reported if e not in todo]
    with cf.ThreadPoolExecutor(max(1, min(int(workers or 1), len(todo)))) as pool:
        for event, verdict in pool.map(check, todo):
            if not isinstance(verdict, dict):
                kept.append(event)           # could not check: keep it, unverified, at its own confidence
            elif verdict.get('happened'):
                v_conf = verdict.get('confidence')
                conf = float(0.8 if v_conf is None else v_conf)
                event = dict(event, verified=True, confidence=max(float(event.get('confidence') or 0), conf))
                if verdict.get('team'):
                    event['team'] = verdict['team']
                if verdict.get('level') and event['type'] == 'climb_success':
                    event['detail'] = verdict['level']
                kept.append(event)
    kept.sort(key=lambda e: e['t'])
    return kept


if __name__ == '__main__':
    main()
