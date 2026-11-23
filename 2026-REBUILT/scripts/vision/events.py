"""Action detection (shots, scores, climbs, disabled robots) with a VLM, over overlapping windows.

Compared with the original single pass of 30 s windows on the Pro model:
  * the autonomous period gets its own window, sampled faster (fps 4) because that is where one missed
    second changes an auto, while the rest of the match can run at fps 2;
  * windows are 40 s with 3 s of overlap, so an action on a boundary is seen whole, and
    duplicates from the overlap are merged;
  * the bulk runs on a cheaper model; only low-confidence events and every climb are re-checked on the
    strongest model with a tight clip around the event (cascade), which is where accuracy is bought;
  * all timestamps are returned absolute (video seconds) and converted to match milliseconds here.
"""
import math
import re

from common import AUTO_END_MS, AUTO_GRACE_MS
from vlm import BudgetExceeded

EVENT_TYPES = ['fuel_shot', 'fuel_scored', 'climb_success', 'climb_attempt', 'defense', 'collision', 'disabled']
# Stored observation_type for each reported type (None = not stored; the check constraint on
# vision_observations allows only these). 'collision' and 'defense' need the migration in this PR.
STORED_TYPES = {'fuel_shot': 'fuel_attempt', 'fuel_scored': 'fuel_scored', 'climb_success': 'climb_success',
                'climb_attempt': 'climb_attempt', 'disabled': 'disabled', 'collision': 'collision', 'defense': 'defense'}

FUEL_TYPES = ('fuel_shot', 'fuel_scored')


def event_schema(include_fuel=False):
    """JSON schema for reported events. Fuel shots and scores are measured from ball tracks and flights,
    so the language model is not offered them by default."""
    types = EVENT_TYPES if include_fuel else [t for t in EVENT_TYPES if t not in FUEL_TYPES]
    return {'type': 'object', 'properties': {'events': {'type': 'array', 'items': {'type': 'object', 'properties': {
        't': {'type': 'number'}, 'team': {'type': 'string'}, 'type': {'type': 'string', 'enum': types},
        'count': {'type': 'integer'}, 'detail': {'type': 'string'}, 'confidence': {'type': 'number'}}, 'required': ['t', 'type']}}}, 'required': ['events']}


EVENT_SCHEMA = event_schema(False)

VERIFY_SCHEMA = {'type': 'object', 'properties': {'happened': {'type': 'boolean'}, 'team': {'type': 'string'}, 'level': {'type': 'string'},
                                                  'confidence': {'type': 'number'}, 'reason': {'type': 'string'}}, 'required': ['happened']}


def plan_windows(start_s, end_s, auto_end_ms=AUTO_END_MS, grace_ms=AUTO_GRACE_MS, window_s=40, overlap_s=3.0):
    """[(a, b, fps_hint)] in video seconds. First window is exactly the auto (+ scoring grace)."""
    auto_end = start_s + (auto_end_ms + grace_ms) / 1000.0
    windows = [(start_s, min(auto_end, end_s), 4)]
    if end_s <= auto_end:
        return windows
    a = auto_end - overlap_s
    while a < end_s - 1:
        b = min(a + window_s, end_s)
        windows.append((round(a, 2), round(b, 2), 2))
        a = b - overlap_s
        if b >= end_s:
            break
    return windows


def window_prompt(a, b, red, blue, auto_window, include_fuel=False):
    phase = ('This window is the AUTONOMOUS period (the first 20 seconds, robots drive on their own). ' if auto_window
             else 'This window is part of TELEOP. ')
    fuel = ('fuel_shot (a robot launches yellow foam fuel toward the hub; count = about how many in the burst), '
            'fuel_scored (fuel visibly enters the hub), ') if include_fuel else 'Do NOT report fuel shots or scores, and do NOT read any numbers from the broadcast overlay; they are measured separately. '
    return (f'FRC 2026 REBUILT broadcast. Analyse ONLY video seconds {a:.1f} to {b:.1f}. {phase}'
            f'Red alliance robots: {", ".join(red)}. Blue alliance robots: {", ".join(blue)}. '
            'Identify robots by the number printed on their bumper only; never read the broadcast overlay. List actions with the absolute video second t '
            '(seconds from the start of the whole clip, not the window) and the team: '
            f'{fuel}climb_success (put the rung level in detail, L1/L2/L3), climb_attempt, '
            'defense (a robot is pinning or blocking an opponent), collision (hard contact), disabled (a robot stops and stays stopped). '
            'Only list what you actually see on the field; if you cannot tell which robot, leave team empty. Give confidence 0-1 honestly: '
            'below 0.5 when partly occluded or you are guessing the team.')


def _clean_team(raw):
    return ''.join(c for c in str(raw or '') if c.isdigit())


_TEAM_CITATION_RE = re.compile(
    r'(?i)(?P<prefix>\b(?:team|robot|frc)\s*#?\s*|(?<!\w)#)(?P<pnum>\d{1,5})\b'
    r'|(?<!\.)\b(?P<bnum>\d{3,5})\b(?!\.\d|\s*(?:ms\b|s\b|sec\b|m\b|m/s|mps|fps|hz|px|deg))'
)


def _sanitize_detail_text(text, valid_team_nums, obs_type=None):
    """Replace off-roster team citations in a detail string when a non-empty match roster is known."""
    if not text or not valid_team_nums:
        return str(text or '')
    replacement = 'opponent robot' if obs_type == 'defense' else 'another robot'

    def _sub(match):
        raw_num = match.group('pnum') or match.group('bnum') or ''
        norm_num = raw_num.lstrip('0') or '0'
        if raw_num in valid_team_nums or norm_num in valid_team_nums:
            return match.group(0)
        return replacement

    return _TEAM_CITATION_RE.sub(_sub, str(text))


def _parse_climb_level(event):
    for raw in (event.get('level'), event.get('detail')):
        text = str(raw or '').upper()
        for lv in ('L3', 'L2', 'L1'):
            if lv in text:
                return lv
        m = re.search(r'(?:LEVEL|RUNG)\s*([123])\b|\b([123])\b', text)
        if m:
            return f"L{m.group(1) or m.group(2)}"
    return 'L1'


def merge_events(events, window_s=0.6):
    """Merge duplicates from overlapping windows: same type and team within window_s keeps the more
    confident one. Teams that are unknown match any team of that type nearby, then adopt a known one."""
    valid = []
    for e in events or []:
        if not isinstance(e, dict) or not e.get('type'):
            continue
        try:
            t = float(e['t'])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(t):
            continue
        item = dict(e, t=t)
        if 'team' in item:
            item['team'] = _clean_team(item.get('team'))
        valid.append(item)
    out = []
    for event in sorted(valid, key=lambda e: e['t']):
        ev_team = _clean_team(event.get('team'))
        exact_match = None
        wild_match = None
        for kept in reversed(out):
            if abs(event['t'] - kept['t']) > window_s:
                continue
            if kept['type'] != event['type']:
                continue
            kept_team = _clean_team(kept.get('team'))
            if ev_team and kept_team == ev_team:
                exact_match = kept
                break
            if wild_match is None and (not kept_team or not ev_team):
                wild_match = kept
        match = exact_match or wild_match
        if match is None:
            out.append(dict(event))
        else:
            kept_team = _clean_team(match.get('team'))
            detail = match.get('detail') or event.get('detail')
            if (event.get('confidence') or 0) > (match.get('confidence') or 0):
                detail = event.get('detail') or match.get('detail')
                match.update(event)
            if ev_team or kept_team:
                match['team'] = ev_team if (ev_team and (event.get('confidence') or 0) >= (match.get('confidence') or 0)) else (kept_team or ev_team)
            if detail:
                match['detail'] = detail
            match['count'] = max(match.get('count') or 0, event.get('count') or 0) or None
    return out


def to_observations(events, teams, alliance_of, offset_s, auto_end_ms=AUTO_END_MS, model='', source='vlm', view_id=None, run_id=None):
    """Reported events -> vision_observations rows in MATCH milliseconds."""
    rows, skipped = [], 0
    valid_nums = set()
    if isinstance(teams, dict):
        for side_items in teams.values():
            if isinstance(side_items, (list, tuple, set)):
                for item in side_items:
                    num = _clean_team(item)
                    if num:
                        valid_nums.add(num)
        for k in teams:
            num = _clean_team(k)
            if num:
                valid_nums.add(num)
    else:
        for item in teams or ():
            num = _clean_team(item)
            if num:
                valid_nums.add(num)
    alliance_map = alliance_of or {}

    for event in events or []:
        if not isinstance(event, dict):
            skipped += 1
            continue
        kind = STORED_TYPES.get(event.get('type'))
        if not kind:
            skipped += 1
            continue
        try:
            raw_t = float(event['t'])
        except (KeyError, TypeError, ValueError):
            skipped += 1
            continue
        if not math.isfinite(raw_t):
            skipped += 1
            continue
        t_ms = int(round((raw_t - offset_s) * 1000))
        if t_ms < -2000:
            skipped += 1
            continue
        t_ms = max(0, t_ms)
        team = _clean_team(event.get('team') or event.get('team_key'))
        in_roster = bool(team and team in valid_nums)
        value = dict(event.get('value') or {}) if isinstance(event.get('value'), dict) else {}
        if kind == 'climb_success':
            value['level'] = _parse_climb_level(event)
        elif kind in ('fuel_attempt', 'fuel_scored') and event.get('count'):
            try:
                count = int(event['count'])
                if count > 0:
                    value['count'] = count
            except (TypeError, ValueError):
                pass
        raw_target = value.get('target_team') or event.get('target_team')
        if raw_target is not None:
            target_num = _clean_team(raw_target)
            if target_num and (not valid_nums or target_num in valid_nums):
                value['target_team'] = f'frc{target_num}'
            else:
                value.pop('target_team', None)
        if 'detail' in value and value['detail']:
            value['detail'] = _sanitize_detail_text(value['detail'], valid_nums, obs_type=kind)
        raw_conf = event.get('confidence')
        try:
            conf = 0.6 if raw_conf is None else float(raw_conf)
        except (TypeError, ValueError):
            conf = 0.6
        if not math.isfinite(conf):
            conf = 0.6
        clean_detail = _sanitize_detail_text(event.get('detail') or '', valid_nums, obs_type=kind)
        rows.append({'observation_type': kind, 'team_key': f'frc{team}' if in_roster else None,
                     'alliance': (alliance_map.get(team) or alliance_map.get(f'frc{team}')) if in_roster else None,
                     'phase': 'auto' if t_ms <= auto_end_ms else 'teleop', 'value': value, 'started_ms': t_ms, 'ended_ms': t_ms + 500,
                     'confidence': min(1.0, max(0.0, conf)),
                     'evidence': {'detail': clean_detail, 'model': model, 'verified': bool(event.get('verified', False))},
                     'view_id': view_id, 'vision_run_id': run_id, 'source': source})
    return rows, skipped


def needs_verification(event, floor=0.5):
    return event['type'] == 'climb_success' or (event.get('confidence') or 0) < floor


def verify_prompt(event, red, blue):
    ev_type = str(event.get('type') or '') if event.get('type') in EVENT_TYPES else 'action'
    ev_team = _clean_team(event.get('team'))
    detail_raw = re.sub(r'[^A-Za-z0-9 _.,/-]+', '', str(event.get('detail') or '')).strip()[:80]
    return (f'FRC 2026 REBUILT broadcast clip centred on video second {float(event["t"]):.1f}. A previous pass reported: {ev_type}'
            f'{" by team " + ev_team if ev_team else ""}{" (" + detail_raw + ")" if detail_raw else ""}. '
            f'Red alliance: {", ".join(red)}. Blue alliance: {", ".join(blue)}. Did this really happen within this clip? '
            'Look closely; answer happened=false if it did not or you cannot tell. Give the team number and, for a climb, the level reached.')


def run_windows(call, windows, workers=6, on_done=None):
    """Run `call(window)` for every window in parallel. Returns (events, failed_windows).

    One window failing (a rate limit that outlasts the retries, a malformed answer) must not throw away the
    other windows' results and the minutes already spent on them: it is recorded and the rest carry on.
    """
    import concurrent.futures as cf
    events, failed = [], []

    def safe(window):
        try:
            res = call(window)
            return window, ([] if res is None else list(res)), None
        except BudgetExceeded:
            raise
        except Exception as error:  # noqa: BLE001 - isolate any per-window failure
            return window, [], str(error)[:200]

    with cf.ThreadPoolExecutor(workers) as pool:
        for n, (window, result, error) in enumerate(pool.map(safe, windows), 1):
            if error:
                failed.append({'window': window[:2], 'error': error})
            else:
                events += result
            if on_done:
                on_done(n, len(windows), window, result, error)
    return events, failed
