#!/usr/bin/env python3
"""Ball counter: how many fuel balls went into each hub, and how many each robot shot, counted from every ball track.

    python3 scripts/vision/fuel_count.py --youtube Ep-8ZaOWkQs --start 12.6 --calibration cal.json \\
        --robot-tracks tracks.json --out count.json

The ballistic tracker (fuel.py) fits a 3D flight to each ball and drops any ball it cannot fit, which on streams of
20-30 balls/s is most of them. The counter does not need a 3D flight fit. It runs the same per-frame ball detection
and image tracking (fuel.run_fuel_pass_parallel with tracks_only), then:

  * hub entries: a ball track that comes down into a hub's opening (the 1.83 m ring projected into the picture,
    padded by HUB_PAD for blur and calibration error) is one ball into that hub.
  * launches: a ball track that starts in or just above a robot's box (the scouting run's robot boxes) and climbs at
    least LAUNCH_CLIMB robot heights above it is a shot candidate, not a validated launch count.
  * observed exits: robot_launches.py counts a continuous, local inside-to-outside observation at a measured body.
    No HUB hit or complete flight is required. Partial observations are not complete totals or validated lower bounds.
  * scored per robot: each hub entry is attributed to a robot by linking full launch-to-hub tracks directly, matching
    flight time from robot distance to the hub and 2D approach velocity/direction over the launch window, and falling
    back to robot proximity in the shooting zone; entries with no plausible shooter stay unattributed.
  * hub_active: marks whether that alliance's hub was active (counting points) when each ball entered (auto, transition
    and endgame are active for both hubs; shifts 1-4 alternate with the vision auto winner inactive in shift 1; None
    when vision auto counts are within AUTO_WINNER_MARGIN balls).

Optional private source-image identity checks can name robot fragments. No official score, scoreboard overlay or human
score data is inference input.
"""
import argparse
import bisect
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ballistics import HUB_OPENING_RADIUS, HUB_Z  # noqa: E402
from camera3d import camera_from_pixel_homography  # noqa: E402
from common import AUTO_END_MS, AUTO_GRACE_MS, FIELD, log  # noqa: E402
import versions  # noqa: E402

HUB_PAD = 1.6               # the hub ring is drawn 1.6x its radius: ball blur, compression and calibration error
MIN_TRACK_OBS = 3
ENTRY_DROP_PX = 4.0         # a hub entry moves down the picture at least this much over its track
LAUNCH_HEAD_OBS = 4
LAUNCH_RISE_PX = 6.0        # rises at least this much over its first LAUNCH_HEAD_OBS observations
LAUNCH_CLIMB = 0.5          # and climbs this many robot-box heights above the box top
BOX_EXPAND_X = 0.25         # a launch starts within the robot box widened 25% on each side
BOX_ABOVE = 0.6             # or up to 0.6 box heights above it (up to 1.0 when moving toward hub at stream speed)
BOX_STREAM_ABOVE = 1.0
BOX_MAX_GAP_MS = 400
BOX_INTERP_SPAN_MS = 1000   # linearly interpolate robot box across bracketing samples up to 1.0 s apart
POS_INTERP_SPAN_MS = 1500   # linearly interpolate robot field (x, y) across bracketing samples up to 1.5 s apart
ENTRY_WINDOW_S = (0.3, 3.0)  # launched 0.3-3 s before the ball came down into the hub
FAST_FLIGHT_MIN_MS = 180    # close-range shots (<2 m) can enter the hub ~0.2 s after first seen leaving the robot
PROXIMITY_MAX_DIST_M = 4.5  # max distance to own hub for solo proximity fallback when launch track was occluded
VOLLEY_TAIL_MAX_MS = 4800   # extended burst window when a shooter remains stationed within shooting range
VOLLEY_MAX_DIST_M = 6.0
MATCH_MS = 163_000
# The resolution the counter and its period model were measured and fitted on. On the 6 canonical 2026cc matches,
# 720p copies of the same videos (same calibrations and starts) found 31% fewer raw entries, and the 1080p-fitted
# scale then under-counted by 30% (WAPE 30.3% against 14.1% at 1080p, 2026-10-22). Count at 1080p.
COUNT_MAX_HEIGHT = 1080
TELEOP_START_MS = AUTO_END_MS + AUTO_GRACE_MS
TRANSITION_END_MS = TELEOP_START_MS + 10_000
SHIFT_MS = 25_000
ENDGAME_START_MS = TRANSITION_END_MS + 4 * SHIFT_MS
AUTO_WINNER_MARGIN = 3      # if vision auto counts are within 3 balls, auto winner is uncertain (hub_active = None in shifts)
HUBS = {'red_hub': 'red', 'blue_hub': 'blue'}
_UNNAMED_PREFIX = '__unnamed_track__:'


def default_hubs():
    return {
        'blue_hub': (FIELD['hub']['x'], FIELD['hub']['y']),
        'red_hub': (FIELD['length'] - FIELD['hub']['x'], FIELD['width'] - FIELD['hub']['y']),
    }


def hub_polygons(camera, hubs, pad=HUB_PAD, n=48):
    """{hub name: (n, 2) picture polygon of the padded hub opening} in the calibration's reference frame."""
    out = {}
    for name, (x, y) in hubs.items():
        ring = np.array([[x + pad * HUB_OPENING_RADIUS * math.cos(a), y + pad * HUB_OPENING_RADIUS * math.sin(a), HUB_Z]
                         for a in np.linspace(0, 2 * math.pi, n, endpoint=False)])
        uv, front = camera.project(ring)
        if front.all():
            out[name] = uv.astype(np.float32).reshape(-1, 1, 2)
    return out


OPENING_PAD = 1.15          # the opening ring is 1.15x its radius (ball radius plus ~8 px calibration error)
TRUE_OPENING_MODEL_Z = 1.829 * 1.829 / 1.233   # the 72 in opening (1.83 m) in the counter's height convention
RIM_HEIGHT_M = 1.829        # HUB front rim height used by the calibration's vertical landmarks
EXIT_TOL_PX = 10.0          # v1.0.1: after entering, a ball that is seen this far outside the opening did not go in
LOST_BAND_PX = 50.0         # v1.0.3: a track that ends this close above an opening may have been lost against the HUB glow
PREDICT_HORIZON_S = 0.3     # v1.0.3: how far ahead its last motion is extrapolated
PREDICT_FIT_OBS = 5         # v1.0.3: last sightings used for the extrapolation
FRAGMENT_GAP_S = 0.15       # v1.0.3: a track starting this soon after another ends ...
FRAGMENT_DIST_PX = 25.0     # ... and this close to where it was heading is the same ball, still tracked
COUNTER_VERSION = 'v1.0.4b'
COUNTER_V2 = 'v2'
COUNTER_V2_ACCURACY_RELEASE = 'v2.1.0'
COUNTER_V2_ACCURACY_MODEL = 'v2-chimney-scale-3'
STITCH_GAP_S = 0.2          # v2: a fragment starting this soon after another ends near an opening ...
STITCH_DIST_PX = 15.0       # ... this close to where the first was heading ...
STITCH_BEHIND_PX = 4.0      # ... and not behind it along its path (the next ball in a stream is) is the same ball
STITCH_ABOVE_PX = 80.0      # v2: "near an opening": over it (widened by STITCH_SIDE_PX) and up to this far above it
STITCH_SIDE_PX = 20.0


# v2: per-HUB factors from counted to TBA-scored balls, fitted on the tune split of the 2026cc matches only
# (scripts/vision/eval_counter_event.py calibrate --save). The counter misses balls steadily per HUB (the lit blue
# HUB washes them out), so one factor per HUB corrects the totals; held-out matches are the reported accuracy.
HUB_CALIBRATION = {
    'hub_scales': {'red': 0.9985, 'blue': 1.2320},
    'fitted_on': 'tune split of 2026cc (26 matches), counter v2 run of commit 65f6c5a',
    'tune_loo_wape_pct': 9.1,
}


def calibrate_payload(payload, calibration=None):
    """v2: put the calibrated estimate in each alliance's `active_scored` (what the app shows as balls scored), keep
    the raw ball-by-ball count as `active_counted`, and record the factors in `stats.calibration`. Unknown stays
    unknown. Robots get the same correction in `estimated_scored`; their raw credit is unchanged."""
    calibration = calibration or HUB_CALIBRATION
    if not calibration or counter_mode() != COUNTER_V2:
        return payload
    factors = calibration.get('hub_scales') or {}
    for color, alliance in (payload.get('alliances') or {}).items():
        raw = alliance.get('active_scored')
        alliance['active_counted'] = raw
        f = factors.get(color)
        alliance['active_scored'] = None if raw is None or f is None else int(round(f * raw))
    payload.setdefault('stats', {})['calibration'] = {k: v for k, v in calibration.items() if k != 'note'}
    return scale_robot_credits(payload)


REFINEMENT_ROUNDS = 4           # fragments withheld one conflict at a time before the whole refinement is refused


def scale_robot_credits(payload):
    """Give each robot the correction its alliance got: raw entries over-count, so alliance totals are calibrated, and
    raw robot credit could add up to more than its own alliance. `estimated_scored` is the robot's credited active-HUB
    entries times the alliance's calibrated/raw ratio, rounded down, so robots can never exceed the estimate. The rest
    stays `estimated_unassigned`: nothing unattributed is split across robots. Unknown stays unknown."""
    for color, alliance in (payload.get('alliances') or {}).items():
        raw, estimate = alliance.get('active_counted'), alliance.get('active_scored')
        members = [r for r in payload.get('robots') or () if r.get('alliance') == color]
        if not raw or estimate is None or any(r.get('active_scored') is None for r in members):
            for robot in members:
                robot['estimated_scored'] = None
            alliance['estimated_unassigned'] = None
            continue
        for robot in members:
            robot['estimated_scored'] = math.floor(robot['active_scored'] * estimate / raw)
        alliance['estimated_unassigned'] = estimate - sum(r['estimated_scored'] for r in members)
    return payload


def counter_mode():
    """'v2' (default) or 'v1.0.4b', from VISION_FUEL_COUNTER. v2 joins ball-track fragments at the HUB openings
    before the entry test (see `stitch_fragments`) and splits touching balls in streams."""
    import os
    mode = os.environ.get('VISION_FUEL_COUNTER', '').strip().lower()
    return 'v1.0.4b' if mode in ('v1.0.4b', 'v1.0.4', 'v1') else COUNTER_V2
VANISH_WINDOW_S = 0.4       # vanished-at-HUB option: no track picks the ball up within this long ...
VANISH_DIST_PX = 80.0       # ... this close to where it was lost (anywhere around: above, beside or past the HUB)
VANISH_SIDE_PX = 10.0       # its last sighting is over the opening, widened this much on each side


LAUNCH_APEX_PX = 15.0       # v2: a ball born below the opening counts if its apex is this far above the entry region


def launched_rule_enabled():
    """v2 counts shots launched from below an opening (`_launched_entries`); VISION_FUEL_LAUNCHED_RULE=0 turns it off."""
    import os
    return os.environ.get('VISION_FUEL_LAUNCHED_RULE', '').strip().lower() not in ('0', 'false', 'no')


def vanished_rule_enabled():
    """VISION_FUEL_VANISHED_RULE=1 counts balls that vanish just above an opening (see `_vanished_entries`).
    Off by default until it is measured on the Spark."""
    import os
    return os.environ.get('VISION_FUEL_VANISHED_RULE', '').strip().lower() in ('1', 'true', 'yes')


def hub_openings(calibration, hubs, width, height, pad=OPENING_PAD, n=48, rim_z=None):
    """v1.0.1 {hub name: (n, 1, 2) picture polygon of the padded HUB opening}, placed through the calibration's measured
    height projection (rims within ~8 px), or None when the calibration has none. The homography-derived camera is
    not used: its error at HUB height is 50-90 px, which put v1's rings over the crowd behind the HUBs."""
    import hub_lights
    from hub_lights import height_project
    out = {}
    for name, (x, y) in hubs.items():
        z = RIM_HEIGHT_M if rim_z is None else rim_z
        ring = [(x + pad * HUB_OPENING_RADIUS * math.cos(a), y + pad * HUB_OPENING_RADIUS * math.sin(a), z)
                for a in np.linspace(0, 2 * math.pi, n, endpoint=False)]
        uv = height_project(hub_lights.for_hub(calibration, name), ring, width, height)
        if uv is None:
            return None
        out[name] = uv.astype(np.float32).reshape(-1, 1, 2)
    return out


def entry_region():
    """v2 entry region, from VISION_FUEL_ENTRY_REGION: 'chimney' (default) or 'ring' (the dark-face ring alone, as
    v2 counted before the 2026cur check)."""
    import os
    return 'ring' if os.environ.get('VISION_FUEL_ENTRY_REGION', '').strip().lower() == 'ring' else 'chimney'


def chimney_openings(calibration, hubs, width, height, pad=OPENING_PAD, n=48):
    """v2 {hub name: polygon} spanning the counter's ring (dark-face top) and the true opening (1.83 m): their convex
    hull. A falling ball vanishes behind the opening's near lip, which projects into the ring from a high camera
    (2026arc, ring 51 px tall) but 40-60 px above it from a lower one (2026cur, ring 24-30 px), where the ring alone
    counted 48% of TBA's balls against 90% at 2026cc. Through the chimney the four events count 1.22-1.63x TBA."""
    base = hub_openings(calibration, hubs, width, height, pad=pad, n=n)
    top = hub_openings(calibration, hubs, width, height, pad=pad, n=n, rim_z=TRUE_OPENING_MODEL_Z)
    if not base or not top:
        return base
    return {name: cv2.convexHull(np.concatenate([base[name].reshape(-1, 2), top[name].reshape(-1, 2)]).astype(np.float32))
            .reshape(-1, 1, 2) for name in base}


def entry_openings(calibration, hubs, width, height):
    """The polygons the entry test uses: the chimney for counter v2 (unless VISION_FUEL_ENTRY_REGION=ring), else the
    ring (`hub_openings`)."""
    if counter_mode() == COUNTER_V2 and entry_region() == 'chimney':
        return chimney_openings(calibration, hubs, width, height)
    return hub_openings(calibration, hubs, width, height)


def entry_lips(calibration, hubs, width, height):
    """v2 {hub name: picture row of the true opening's near lip} for the chimney, else None. A track first seen
    inside the entry region below this row is a ball already in the funnel (see `_strict_entries`)."""
    if counter_mode() != COUNTER_V2 or entry_region() != 'chimney':
        return None
    top = hub_openings(calibration, hubs, width, height, rim_z=TRUE_OPENING_MODEL_Z)
    return {name: float(poly[:, 0, 1].max()) for name, poly in top.items()} if top else None


def opening_region_mask(openings, width, height, above_px=None, side_px=None):
    """v1.0.4: uint8 mask of each HUB opening and the `above_px` above it (padded `side_px` sideways), where a ball
    is lost against the lit HUB and the detector accepts it by the opening detection mode."""
    import fuel
    above_px = fuel.OPENING_ABOVE_PX if above_px is None else above_px
    side_px = fuel.OPENING_SIDE_PX if side_px is None else side_px
    mask = np.zeros((height, width), np.uint8)
    for poly in (openings or {}).values():
        one = np.zeros((height, width), np.uint8)
        cv2.fillPoly(one, [np.int32(poly)], 255)
        x, y, w, h = cv2.boundingRect(np.int32(poly))
        cv2.rectangle(one, (x - side_px, max(0, y - above_px)), (x + w + side_px, y + h // 2), 255, -1)
        mask = cv2.bitwise_or(mask, one)
    return mask


def _near_opening(u, v, openings):
    for poly in openings.values():
        x0, x1 = float(poly[:, 0, 0].min()) - STITCH_SIDE_PX, float(poly[:, 0, 0].max()) + STITCH_SIDE_PX
        top, bottom = float(poly[:, 0, 1].min()), float(poly[:, 0, 1].max())
        if x0 <= u <= x1 and top - STITCH_ABOVE_PX <= v <= bottom:
            return True
    return False


def stitch_fragments(tracks, openings, gap_s=STITCH_GAP_S, dist_px=STITCH_DIST_PX, behind_px=STITCH_BEHIND_PX):
    """v2: join ball-track fragments near the HUB openings into one track per ball.

    The lit HUB washes balls out and the tracker re-acquires them, so one ball often leaves two or three fragments
    over an opening, none of which passes the entry test alone (2026cc qm1 blue: 87 rejected tracks were picked up
    again above the opening and still not counted). A fragment A that ends near an opening is joined to a fragment
    B when B starts within `gap_s` of A's end, within `dist_px` of where A was heading, and not behind A along its
    path (the next ball of a stream follows on the same path, behind). Each fragment joins at most one before and
    one after it, cheapest pairs first. Returns a new list of observation lists (unjoined tracks unchanged)."""
    return [chain for chain, _ in stitch_components(tracks, openings, gap_s, dist_px, behind_px)]


def stitch_components(tracks, openings, gap_s=STITCH_GAP_S, dist_px=STITCH_DIST_PX, behind_px=STITCH_BEHIND_PX):
    """[(joined observation list, [indices of the fragments in it])] for `stitch_fragments`. Only a fragment that is
    coming down at its end, and was not born below an opening (a ball the HUB released), is joined to a later one."""
    bottoms = [float(poly[:, 0, 1].max()) for poly in openings.values()]
    obs = [list(_obs(t)) for t in tracks]
    ends = []
    for i, o in enumerate(obs):
        if len(o) < 2:
            continue
        motion = _end_motion(o)
        if motion is None or motion[4] <= 0:
            continue                                       # not coming down at its end
        if float(o[0][6]) > max(bottoms):
            continue                                       # born below the openings: a released ball
        if _near_opening(motion[1], motion[2], openings):
            ends.append((i, motion))
    starts = sorted((float(o[0][1]), i) for i, o in enumerate(obs) if o)
    start_times = [s[0] for s in starts]
    pairs = []
    for i, (t1, u1, v1, du, dv) in ends:
        k = bisect.bisect_right(start_times, t1)
        while k < len(starts) and starts[k][0] <= t1 + gap_s:
            ts, j = starts[k]
            k += 1
            if j == i:
                continue
            u0, v0 = float(obs[j][0][5]), float(obs[j][0][6])
            if not _near_opening(u0, v0, openings):
                continue
            dt = ts - t1
            pu, pv = u1 + du * dt, v1 + dv * dt
            dist = math.hypot(u0 - pu, v0 - pv)
            if dist > dist_px:
                continue
            speed = math.hypot(du, dv)
            if speed > 1e-6 and ((u0 - u1) * du + (v0 - v1) * dv) / speed < -behind_px:
                continue                                   # behind the lost ball on its path: the next ball
            pairs.append((dist + 30.0 * dt, i, j))
    nxt, prev = {}, {}
    for _, i, j in sorted(pairs):
        if i not in nxt and j not in prev:
            nxt[i], prev[j] = j, i
    out = []
    for i, o in enumerate(obs):
        if i in prev:
            continue                                       # joined onto an earlier fragment
        chain, k, members = list(o), i, [i]
        while k in nxt and nxt[k] not in members:
            k = nxt[k]
            members.append(k)
            chain.extend(obs[k])
        out.append((chain, members))
    return out


def hub_entries_strict(tracks, openings, offset_s, min_obs=MIN_TRACK_OBS, drop_px=ENTRY_DROP_PX, exit_tol_px=EXIT_TOL_PX,
                       vanished=None, stitch=None, counter=None, lips=None):
    """v1.0.2 entries: a ball that comes down into a HUB opening and is not seen leaving it.

    The tracker often first picks a ball up at the opening itself, so a track may start above or inside it (v1.0.1
    required sightings above the opening and lost about half the real entries). Rejected, unlike v1: tracks that start
    below the opening (balls the HUB releases), tracks that do not move down (clutter in the funnel), and tracks that,
    after reaching the opening, are seen more than `exit_tol_px` outside it (arcs passing over or in front of the HUB,
    rebounds off the rim).

    Counter v2 (`counter_mode()`, or `counter='v2'`) adds balls rescued by joining fragments at the openings
    (`stitch_components`): a joined track counts only when none of its fragments counted alone, so joining never
    removes an entry (joining everything lost 65 qm1 entries to released balls and passers-by merged onto counted
    ones). `stitch` overrides that. Every entry records the counter that made it."""
    counter = counter or counter_mode()
    entries = _strict_entries(tracks, openings, offset_s, min_obs, drop_px, exit_tol_px, vanished,
                              launched=counter == COUNTER_V2 and launched_rule_enabled(), lips=lips)
    if openings and (stitch if stitch is not None else counter == COUNTER_V2):
        counted = {e['track_id'] for e in entries}
        for chain, members in stitch_components(tracks, openings):
            if len(members) < 2 or counted & set(members):
                continue
            for e in _strict_entries([chain], openings, offset_s, min_obs, drop_px, exit_tol_px, False, predict=False):
                e.update(track_id=members[0], stitched=len(members))
                entries.append(e)
    entries.sort(key=lambda e: (e['t_ms'], e.get('track_id', 0)))
    for e in entries:
        e['counter'] = counter
    return entry_source_provenance(entries, tracks, offset_s)


def _strict_entries(tracks, openings, offset_s, min_obs, drop_px, exit_tol_px, vanished, predict=True, launched=False,
                    lips=None):
    entries = []
    for track_id, track in enumerate(tracks):
        obs = _obs(track)
        if len(obs) < min_obs:
            continue
        for name, poly in openings.items():
            bottom = float(poly[:, 0, 1].max())
            dists = [cv2.pointPolygonTest(poly, (float(o[5]), float(o[6])), True) for o in obs]
            inside = [d >= 0.0 for d in dists]
            if not any(inside):
                continue
            first = inside.index(True)
            if float(obs[0][6]) > bottom:
                continue                                   # born below the opening: a ball the HUB released
            if lips and inside[0] and float(obs[0][6]) > lips[name]:
                continue                                   # v2: first seen in the funnel below the true opening's lip:
                                                           # a ball already in, picked up again (zoomed 2026gal views)
            if float(obs[-1][6]) - float(obs[0][6]) < drop_px:
                continue                                   # not moving down: clutter in the funnel
            if any(d < -exit_tol_px for d in dists[first + 1:]):
                continue                                   # seen leaving the opening: passed by or bounced out
            entries.append(_entry(name, track_id, obs, obs[first][1], obs[first][5], obs[first][6], offset_s))
            break
    if launched:
        entries.extend(_launched_entries(tracks, openings, offset_s, {e['track_id'] for e in entries}, min_obs, drop_px,
                                         exit_tol_px))
    if predict:
        entries.extend(_predicted_entries(tracks, openings, offset_s, {e['track_id'] for e in entries}, min_obs, drop_px))
    if vanished if vanished is not None else vanished_rule_enabled():
        entries.extend(_vanished_entries(tracks, openings, offset_s, {e['track_id'] for e in entries}, min_obs, drop_px))
    return entries


def _launched_entries(tracks, openings, offset_s, counted, min_obs, drop_px, exit_tol_px, apex_px=LAUNCH_APEX_PX):
    """v2: shots from a robot right in front of a HUB. Its shooter is below the opening in the picture, so the strict
    rule took the ball for one the HUB released and dropped it. A track born below an opening counts when it rises
    `apex_px` above the top of the entry region (a shot clearing the rim), comes down by `drop_px` after its apex,
    reaches the region and is not seen leaving it. Released balls only go down. Leave-one-event-out over 2026cc,
    2026arc, 2026caclv and 2026cur it cut the mean WAPE from 16.6% to 14.5%; with each event's own factor it lowered
    the scatter at all four (caclv 20.1% -> 13.3%)."""
    entries = []
    for track_id, track in enumerate(tracks):
        if track_id in counted:
            continue
        obs = _obs(track)
        if len(obs) < min_obs:
            continue
        v = [float(o[6]) for o in obs]
        apex = min(range(len(v)), key=v.__getitem__)
        for name, poly in openings.items():
            top, bottom = float(poly[:, 0, 1].min()), float(poly[:, 0, 1].max())
            if v[0] <= bottom or v[apex] > top - apex_px or v[-1] - v[apex] < drop_px:
                continue                                   # not born below, never over the rim, or not coming down
            dists = [cv2.pointPolygonTest(poly, (float(o[5]), float(o[6])), True) for o in obs[apex:]]
            inside = [d >= 0.0 for d in dists]
            if not any(inside):
                continue
            first = inside.index(True)
            if any(d < -exit_tol_px for d in dists[first + 1:]):
                continue                                   # seen leaving: passed by or bounced out
            o = obs[apex + first]
            entries.append({**_entry(name, track_id, obs, o[1], o[5], o[6], offset_s), 'launched': True})
            break
    return entries


def _entry(name, track_id, obs, t_s, ru, rv, offset_s, predicted=False):
    # Shooter matching needs the actual first pixel, not its registered field-reference coordinate. Derive climb
    # from registered observations so camera motion alone cannot manufacture an upward flight. Future sightings
    # (after this entry) cannot establish its source.
    seen = [o for o in obs if o[1] <= t_s + 1e-6]
    source = {}
    if len(seen) >= MIN_TRACK_OBS:
        rise = max(0.0, float(seen[0][6]) - min(float(o[6]) for o in seen))
        source = {'u0': float(seen[0][2]), 'v0': float(seen[0][3]),
                  'top_v': float(seen[0][3]) - rise, 'rise_px': rise, 'source_obs_count': len(seen)}
    return {
        **source,
        't_ms': int(round((t_s - offset_s) * 1000)),
        'hub': name,
        'alliance': HUBS[name],
        'track_id': track_id,
        't0_ms': int(round((obs[0][1] - offset_s) * 1000)),
        'ru': round(float(ru), 1),
        'rv': round(float(rv), 1),
        'ru0': round(float(obs[0][5]), 1),
        'rv0': round(float(obs[0][6]), 1),
        'obs_count': len(obs),
        'predicted': predicted,
        'counter': COUNTER_VERSION,
    }


def _end_motion(obs, n=PREDICT_FIT_OBS):
    """(t, u, v, du/dt, dv/dt) at a track's last sighting, from a straight-line fit of its last `n` sightings."""
    tail = obs[-n:]
    ts = np.array([float(o[1]) for o in tail])
    if len(tail) < 3 or ts[-1] - ts[0] <= 0:
        return None
    us = np.array([float(o[5]) for o in tail]); vs = np.array([float(o[6]) for o in tail])
    du, _ = np.polyfit(ts, us, 1); dv, _ = np.polyfit(ts, vs, 1)
    return float(ts[-1]), float(us[-1]), float(vs[-1]), float(du), float(dv)


def _predicted_entries(tracks, openings, offset_s, counted, min_obs, drop_px):
    """v1.0.3: balls lost just above an opening. The active blue HUB's lit top washes yellow balls out, so most
    blue tracks end 0-50 px above the opening (2026cc qm1: 69% of descending blue tracks vs 32% of red). A descending
    track whose last sighting is within LOST_BAND_PX above an opening, and whose extrapolated path drops into it within
    PREDICT_HORIZON_S, counts once, unless another track picks the same ball up again (it then counts on its own)."""
    starts = sorted((float(_obs(t)[0][1]), float(_obs(t)[0][5]), float(_obs(t)[0][6]), i)
                    for i, t in enumerate(tracks) if len(_obs(t)) >= 1)
    start_times = [s[0] for s in starts]
    out = []
    for track_id, track in enumerate(tracks):
        if track_id in counted:
            continue
        obs = _obs(track)
        if len(obs) < min_obs or float(obs[-1][6]) - float(obs[0][6]) < drop_px:
            continue
        motion = _end_motion(obs)
        if motion is None:
            continue
        t1, u1, v1, du, dv = motion
        if dv <= 0:
            continue                                       # not coming down at its last sighting
        for name, poly in openings.items():
            top, bottom = float(poly[:, 0, 1].min()), float(poly[:, 0, 1].max())
            x0, x1 = float(poly[:, 0, 0].min()), float(poly[:, 0, 0].max())
            if not (0 < top - v1 <= LOST_BAND_PX and x0 <= u1 <= x1) or float(obs[0][6]) > bottom:
                continue
            mid = 0.5 * (top + bottom)
            dt = (mid - v1) / dv                           # time to reach the opening's middle depth
            if dt > PREDICT_HORIZON_S:
                break
            pu, pv = u1 + du * dt, mid
            if cv2.pointPolygonTest(poly, (pu, pv), False) < 0:
                break
            # The same ball picked up again by another track: that track counts on its own.
            k = bisect.bisect_right(start_times, t1)
            continued = False
            while k < len(starts) and starts[k][0] <= t1 + FRAGMENT_GAP_S:
                ts, us, vs, other = starts[k]
                if other != track_id and math.hypot(us - (u1 + du * (ts - t1)), vs - (v1 + dv * (ts - t1))) <= FRAGMENT_DIST_PX:
                    continued = True
                    break
                k += 1
            if not continued:
                out.append(_entry(name, track_id, obs, t1 + dt, pu, pv, offset_s, predicted=True))
            break
    return out


def _vanished_entries(tracks, openings, offset_s, counted, min_obs, drop_px):
    """Option: balls that vanish at the HUB. On 2026cc qm1, of 238 blue tracks v1.0.3 rejected just above the opening,
    36 were never seen again anywhere near (28 'too slow', 8 'path misses'): most likely dropped in, lost against
    the lit opening. A descending, uncounted track whose last sighting is within LOST_BAND_PX above an opening and over
    it counts once (`vanished: True`) when no other track starts within VANISH_DIST_PX of that point in the next
    VANISH_WINDOW_S. A ball seen again past, beside or above the HUB is not counted."""
    starts = sorted((float(_obs(t)[0][1]), float(_obs(t)[0][5]), float(_obs(t)[0][6]), i)
                    for i, t in enumerate(tracks) if len(_obs(t)) >= 1)
    start_times = [st[0] for st in starts]
    out = []
    for track_id, track in enumerate(tracks):
        if track_id in counted:
            continue
        obs = _obs(track)
        if len(obs) < min_obs or float(obs[-1][6]) - float(obs[0][6]) < drop_px:
            continue
        motion = _end_motion(obs)
        if motion is None or motion[4] < 0:
            continue                                       # rising at its last sighting: not dropping in
        t1, u1, v1 = float(obs[-1][1]), float(obs[-1][5]), float(obs[-1][6])
        for name, poly in openings.items():
            top, bottom = float(poly[:, 0, 1].min()), float(poly[:, 0, 1].max())
            x0, x1 = float(poly[:, 0, 0].min()) - VANISH_SIDE_PX, float(poly[:, 0, 0].max()) + VANISH_SIDE_PX
            if not (0 < top - v1 <= LOST_BAND_PX and x0 <= u1 <= x1) or float(obs[0][6]) > bottom:
                continue
            k = bisect.bisect_right(start_times, t1)
            seen_again = False
            while k < len(starts) and starts[k][0] <= t1 + VANISH_WINDOW_S:
                _, us, vs, other = starts[k]
                if other != track_id and math.hypot(us - u1, vs - v1) <= VANISH_DIST_PX:
                    seen_again = True
                    break
                k += 1
            if not seen_again:
                e = _entry(name, track_id, obs, t1, u1, v1, offset_s, predicted=True)
                e['vanished'] = True
                out.append(e)
            break
    return out


RING_TARGET_PREDICTED = 0.35  # v2: share of entries that are predicted (lost just above) where the ring sits right
RING_MAX_LIFT_PX = 32
RING_STEP_PX = 4


def ring_lift_enabled():
    """VISION_FUEL_RING_LIFT=1 turns on `refine_openings`. Off by default: the two-ring period model (rings at 1.233 m and
    at the true 1.83 m opening) handled the cameras it was meant for better, and was evaluated without lifting."""
    import os
    return os.environ.get('VISION_FUEL_RING_LIFT', '').strip().lower() in ('1', 'true', 'yes')


def refine_openings(tracks, openings, offset_s):
    """v2 self-calibration from the video: lift each HUB's opening ring until descending balls reach it.

    When a ring sits too low, balls vanish above it and nearly every entry is a predicted one (2026arc red: 95-99%;
    on 2026cc, whose rings were checked by hand, 16% red and 33% blue). Each ring is raised in 4 px steps, up to
    32 px, until no more than 35% of its entries are predicted. Uses only the ball tracks. Returns
    (openings, {hub: lift in px})."""
    out, lifts = {}, {}
    for name, poly in openings.items():
        chosen = 0
        for lift in range(0, RING_MAX_LIFT_PX + 1, RING_STEP_PX):
            ents = hub_entries_strict(tracks, {name: poly - np.array([0.0, lift], np.float32)}, offset_s,
                                      counter='v2', stitch=False, vanished=False)
            share = sum(bool(e.get('predicted')) for e in ents) / max(1, len(ents))
            chosen = lift
            if share <= RING_TARGET_PREDICTED:
                break
        out[name] = poly - np.array([0.0, chosen], np.float32)
        lifts[name] = chosen
    return out, lifts


def lifted(openings, lifts):
    """`openings` raised by the self-calibrated lifts (`refine_openings`), or unchanged without them."""
    if not openings or not lifts:
        return openings
    return {name: poly - np.array([0.0, float(lifts.get(name, 0))], np.float32) for name, poly in openings.items()}


def counter_version(opening_mode):
    """The counter version for the opening detection the fuel pass used (stats['opening_detection']): 'v2' when the
    v2 counter runs with the default background detection, else the v1.0.x name of that detection."""
    if opening_mode == 'background' and counter_mode() == COUNTER_V2:
        return COUNTER_V2
    return {'wide': 'v1.0.4a', 'background': 'v1.0.4b'}.get(opening_mode, 'v1.0.3')


def stamp_accuracy_release(payload, height):
    """Name the measured 1080p counter only when its fitted model actually ran.

    A missing model, incompatible entry region, or lower-resolution video keeps the
    underlying v2 label so a fallback cannot masquerade as the evaluated release.
    """
    stats = payload.setdefault('stats', {})
    calibration = stats.get('calibration') or {}
    if (stats.get('counter') == COUNTER_V2 and height == COUNT_MAX_HEIGHT
            and calibration.get('method') == 'period model'
            and calibration.get('model') == COUNTER_V2_ACCURACY_MODEL
            and entry_region() == 'chimney' and true_opening_region_enabled()
            and launched_rule_enabled() and not vanished_rule_enabled() and not ring_lift_enabled()):
        stats['counter'] = COUNTER_V2_ACCURACY_RELEASE
    return payload


def true_opening_region_enabled():
    """v2 widens the opening detection region up to the true opening (see `opening_setup`);
    VISION_FUEL_TRUE_OPENING_REGION=0 turns it off."""
    import os
    return os.environ.get('VISION_FUEL_TRUE_OPENING_REGION', '').strip().lower() not in ('0', 'false', 'no')


def opening_setup(calibration, hubs, width, height, mode=None):
    """(openings, opening_mode, opening_region) for a fuel pass: the height-projected HUB openings, the detection mode
    (`fuel.opening_detection_mode()` unless given) and the region it applies to. All None without both openings."""
    openings = hub_openings(calibration, hubs, width, height) if calibration else None
    if not openings or len(openings) != 2:
        return None, None, None
    if mode is None:
        import fuel
        mode = fuel.opening_detection_mode()
    region = opening_region_mask(openings, width, height) if mode else None
    if region is not None and counter_mode() == COUNTER_V2 and true_opening_region_enabled():
        # v2: also cover the true opening (1.83 m). The counter's ring is the dark-face top (1.233 m); balls cross
        # the lit HUB's glare above it, outside the old region, and were lost there (2026cur red endgame: 65 of 162).
        # Together with the launched rule it lowered every event's leave-one-event-out WAPE (mean 16.4% -> 12.7%).
        true_rings = hub_openings(calibration, hubs, width, height, rim_z=TRUE_OPENING_MODEL_Z)
        if true_rings and len(true_rings) == 2:
            region = cv2.bitwise_or(region, opening_region_mask(true_rings, width, height))
    return openings, mode, region


def match_light_series(video, rois, offset_s):
    """The dense HUB-light series `count_entries` reads around a match that starts `offset_s` into `video`."""
    import hub_lights
    return hub_lights.light_series(video, rois, max(0.0, offset_s - 5.0), offset_s + MATCH_MS / 1000 + 5.0)


def count_entries(tracks, offset_s, *, calibration=None, hubs=None, width=1280, height=720, polygons=None,
                  video=None, opening_mode=None, lights=None):
    """HUB entries for `tracks` with the newest counter the calibration supports: `(entries, shift1_side, info)`.

    With a height projection: `hub_entries_strict` through the measured openings, and the HUB-light schedule read
    from `video` (`shift1_side` 'unknown' without a video, so `hub_active` stays unknown in the shifts). Without one:
    v1 `hub_entries` through `polygons` and `shift1_side` None (schedule from vision). `info` has `counter` (and
    `counter_note` / `hub_lights`) for the run's stats; `opening_mode` is the detection the fuel pass used. `lights` is
    the dense light series of `video` when it was read already (`match_light_series`, e.g. while the balls were tracked)."""
    hubs = hubs or default_hubs()
    openings = entry_openings(calibration, hubs, width, height) if calibration else None
    if openings and len(openings) == 2:
        lifts = None
        if counter_mode() == COUNTER_V2 and ring_lift_enabled():
            openings, lifts = refine_openings(tracks, openings, offset_s)
        entries = hub_entries_strict(tracks, openings, offset_s, lips=entry_lips(calibration, hubs, width, height))
        info = versions.stamp(counter_version(opening_mode))
        if lifts is not None:
            info['ring_lift_px'] = lifts
        if video is not None:
            import hub_lights
            rois = hub_lights.light_rois(calibration, hubs)
            shift1_side = 'unknown'
            if counter_mode() == COUNTER_V2 and rois:
                # v2: lights sampled every 0.25 s and matched to the whole teleop pattern. On 10 2026cc matches it
                # found the match start within 1 s and the shift-1 side in all 10, including 4 the 5 s reading could
                # not decide.
                series = lights if lights is not None else match_light_series(video, rois, offset_s)
                start, side, evidence = hub_lights.match_start_from_lights(series, search=(offset_s - 5.0, offset_s + 5.0))
                info['hub_lights_dense'] = {**evidence, 'match_start_s': start, 'shift1_side': side}
                if side in ('red', 'blue'):
                    shift1_side = side
            if shift1_side not in ('red', 'blue'):
                shift1_side, evidence = hub_lights.shift1_from_video(video, rois, match_start_s=offset_s)
                info['hub_lights'] = {k: v for k, v in evidence.items() if k != 'rois'}
        else:
            shift1_side = 'unknown'
            info['hub_lights'] = {'reason': 'no video given to read the HUB lights'}
        if shift1_side not in ('red', 'blue') and counter_mode() == COUNTER_V2:
            shift1_side, evidence = shift1_from_entries(entries)
            info['schedule'] = {**evidence, 'shift1_side': shift1_side}
        return entries, shift1_side, info
    entries = hub_entries(tracks, polygons, offset_s) if polygons else []
    return entries, None, {**versions.stamp('v1'), 'counter_note': 'calibration has no height projection; v1 HUB rings used'}


SCHEDULE_MIN_MARGIN = 15    # v2: entries must favour one schedule by at least this many balls ...
SCHEDULE_MIN_SHARE = 0.15   # ... and this share of the shift entries (2026cc: smallest margin 191 balls; low-scoring
                            # matches such as 2026caclv qm3, 104 / 47 balls, need the relative rule)


def shift1_from_entries(entries, min_margin=SCHEDULE_MIN_MARGIN, min_share=SCHEDULE_MIN_SHARE):
    """v2 fallback when the HUB lights cannot be read: (side, evidence) from where balls go in the shifts.

    Teams shoot mostly into the active HUB, so the HUB active in shift 1 takes more of its shift balls in shifts 1
    and 3, the other HUB in shifts 2 and 4. Agreed with the lights in all 40 2026cc matches where both were known
    (smallest margin 191 balls). Unlike v1's rule (auto counts), it does not guess the auto winner. Returns
    'unknown' when the margin is under `min_margin`."""
    first, step = TRANSITION_END_MS, SHIFT_MS
    n = {c: [0, 0] for c in ('red', 'blue')}
    for e in entries:
        t = e['t_ms']
        if first <= t < first + 4 * step:
            n[e['alliance']][int((t - first) // step) % 2] += 1      # [shifts 1 and 3, shifts 2 and 4]
    score = (n['red'][0] - n['red'][1]) - (n['blue'][0] - n['blue'][1])
    evidence = {'method': 'entries', 'margin': abs(score), 'counts': n}
    total = sum(sum(v) for v in n.values())
    if abs(score) < max(min_margin, min_share * total):
        return 'unknown', evidence
    return ('red' if score > 0 else 'blue'), evidence


def match_start_from_entries(entries, search_s=(0.0, 70.0), step_s=0.25, pre_window_s=20.0):
    """(match start video second, shift-1 side, evidence) from where balls go, for broadcasts whose HUB lights cannot
    be read (2026arc's panels clip to white). `entries` carry `t_ms` in video time. In each 25 s shift most balls go
    into the active HUB, so the start and side are the ones whose shift pattern the entries follow best; balls before
    the start count against it (no ball scores before the match), which also rules out starts one full HUB cycle
    (50 s) off. Video only. Returns (None, 'unknown', evidence) without enough balls."""
    t = np.array([e['t_ms'] / 1000.0 for e in entries])
    red = np.array([e['alliance'] == 'red' for e in entries])
    if len(t) < 50:
        return None, 'unknown', {'reason': 'too few HUB entries to read the shift pattern', 'entries': int(len(t))}
    first = TRANSITION_END_MS / 1000.0
    shift = SHIFT_MS / 1000.0
    best = (-np.inf, None, None, None)
    for t0 in np.arange(search_s[0], search_s[1] + 1e-9, step_s):
        pre = int(np.sum((t >= t0 - pre_window_s) & (t < t0)))
        for side in ('red', 'blue'):
            agree = 0
            for k in range(4):
                lo, hi = t0 + first + k * shift, t0 + first + (k + 1) * shift
                inside = (t >= lo) & (t < hi)
                active_red = (side == 'red') == (k % 2 == 0)
                n_red, n_blue = int(np.sum(inside & red)), int(np.sum(inside & ~red))
                agree += (n_red - n_blue) if active_red else (n_blue - n_red)
            score = agree - 2 * pre
            if score > best[0]:
                best = (score, float(t0), side, agree)
    score, t0, side, agree = best
    total = int(np.sum((t >= t0 + first) & (t < t0 + first + 4 * shift)))
    evidence = {'method': 'entries', 'agreement': round(agree / total, 3) if total else None, 'shift_entries': total}
    if not total or agree / total < 0.3:
        return None, 'unknown', {**evidence, 'reason': 'entries do not follow the shift pattern clearly'}
    return t0, side, evidence


def auto_winner_from_shift1(shift1_side):
    """The alliance that won auto, from whose HUB the lights show active in shift 1 (the auto loser's), else None."""
    return {'red': 'blue', 'blue': 'red'}.get(shift1_side)


def _obs(track):
    return track.obs if hasattr(track, 'obs') else track


def entry_source_provenance(entries, tracks, offset_s):
    from fuel import continuous_source_to
    for entry in entries:
        tid = entry.get('track_id')
        valid = isinstance(tid, int) and not isinstance(tid, bool) and 0 <= tid < len(tracks)
        entry['source_continuous'] = bool(valid and continuous_source_to(tracks[tid], offset_s + entry['t_ms'] / 1000))
    return entries


def _phase(t_ms, scoring=False, launch_ms=None, auto_end_ms=AUTO_END_MS, grace_ms=AUTO_GRACE_MS):
    if scoring:
        if launch_ms is not None and launch_ms > auto_end_ms:
            return 'teleop'
        return 'auto' if t_ms <= auto_end_ms + grace_ms else 'teleop'
    return 'auto' if t_ms <= auto_end_ms else 'teleop'


def auto_winner_from_entries(entries, auto_end_ms=AUTO_END_MS, grace_ms=AUTO_GRACE_MS, min_margin=AUTO_WINNER_MARGIN):
    """Alliance ('red' | 'blue') with more vision-counted auto balls, or None when tied / within `min_margin` balls."""
    counts = {'red': 0, 'blue': 0}
    for e in entries or ():
        color = e.get('alliance')
        if color not in counts:
            continue
        t_ms = e.get('t_ms')
        if t_ms is None or t_ms < 0:
            continue
        if _phase(t_ms, scoring=True, launch_ms=e.get('launch_ms'), auto_end_ms=auto_end_ms, grace_ms=grace_ms) == 'auto':
            counts[color] += int(e.get('count') or 1)
    diff = abs(counts['red'] - counts['blue'])
    if diff <= min_margin or (counts['red'] == 0 and counts['blue'] == 0):
        return None
    return 'red' if counts['red'] > counts['blue'] else 'blue'


def hub_active_at(alliance, t_ms, auto_winner):
    """Whether `alliance`'s hub counts points at match time `t_ms` (True | False | None)."""
    if alliance not in ('red', 'blue') or t_ms is None:
        return None
    if t_ms < TRANSITION_END_MS or t_ms >= ENDGAME_START_MS:
        return True
    if auto_winner not in ('red', 'blue'):
        return None
    shift = int((t_ms - TRANSITION_END_MS) // SHIFT_MS)
    winner_active = (shift % 2) == 1
    return winner_active if alliance == auto_winner else (not winner_active)


def hub_entries(tracks, polygons, offset_s, min_obs=MIN_TRACK_OBS, drop_px=ENTRY_DROP_PX):
    """[{'t_ms', 'hub', 'alliance', 'track_id', ...}] for each ball track that comes down into a hub opening.

    Tracks are in reference-frame pixels (obs = (frame, t, u, v, area, ru, rv)); a track that stays inside the
    opening the whole time is a ball sitting in the funnel, not an entry. Supports both descending tracks and
    full-arc launch-to-hub tracks that rise to an apex above the hub opening and descend into its core."""
    entries = []
    for track_id, track in enumerate(tracks):
        obs = _obs(track)
        if len(obs) < min_obs:
            continue
        descending_overall = (obs[-1][6] - obs[0][6]) >= drop_px
        apex_i = min(range(len(obs)), key=lambda i: obs[i][6])
        full_arc_descent = (
            not descending_overall
            and apex_i > 0
            and (obs[0][6] - obs[apex_i][6]) >= LAUNCH_RISE_PX
            and (obs[-1][6] - obs[apex_i][6]) >= max(drop_px, 8.0)
        )
        if not (descending_overall or full_arc_descent):
            continue
        for name, poly in polygons.items():
            dists = [cv2.pointPolygonTest(poly, (float(o[5]), float(o[6])), True) for o in obs]
            inside = [d >= 0.0 for d in dists]
            if not any(inside) or all(inside):
                continue
            if descending_overall:
                first = inside.index(True)
            else:
                inside_desc = inside[apex_i:]
                if inside[apex_i] or not inside[-1] or sum(inside_desc) < 2 or dists[-1] < 2.5:
                    continue
                first = apex_i + inside_desc.index(True)
                if obs[-1][6] < obs[first][6]:
                    continue
            k0 = max(0, first - 4) if first >= 2 else 0
            k1 = first if first >= 2 else min(len(obs) - 1, 3)
            dt_tail = max(0.02, float(obs[k1][1] - obs[k0][1]))
            du = float(obs[k1][5] - obs[k0][5])
            dv = float(obs[k1][6] - obs[k0][6])
            vu = du / dt_tail
            vv = dv / dt_tail
            head_end = min(len(obs), LAUNCH_HEAD_OBS + 2)
            rise_px = float(obs[0][3] - min(o[3] for o in obs[:head_end]))
            entries.append({
                't_ms': int(round((obs[first][1] - offset_s) * 1000)),
                'hub': name,
                'alliance': HUBS[name],
                'track_id': track_id,
                't0_ms': int(round((obs[0][1] - offset_s) * 1000)),
                'ru': round(float(obs[first][5]), 1),
                'rv': round(float(obs[first][6]), 1),
                'du': round(du, 2),
                'dv': round(dv, 2),
                'vu': round(vu, 1),
                'vv': round(vv, 1),
                'u0': round(float(obs[0][2]), 1),
                'v0': round(float(obs[0][3]), 1),
                'ru0': round(float(obs[0][5]), 1),
                'rv0': round(float(obs[0][6]), 1),
                'top_v': round(float(min(o[3] for o in obs)), 1),
                'rise_px': round(rise_px, 1),
                'obs_count': len(obs),
                'full_arc': full_arc_descent,
            })
            break
    entries.sort(key=lambda e: (e['t_ms'], e.get('track_id', 0)))
    return entry_source_provenance(entries, tracks, offset_s)


def _project_robot_box(xy, camera, width, height, radius_m=0.45, z_top_m=0.85):
    """Project a 3D robot volume centered at `xy` into a normalized `[x0, y0, x1, y1]` image box."""
    if xy is None or camera is None or width <= 0 or height <= 0:
        return None
    x, y = float(xy[0]), float(xy[1])
    corners = np.array([
        [x - radius_m, y - radius_m, 0.0],
        [x + radius_m, y - radius_m, 0.0],
        [x + radius_m, y + radius_m, 0.0],
        [x - radius_m, y + radius_m, 0.0],
        [x - radius_m, y - radius_m, z_top_m],
        [x + radius_m, y - radius_m, z_top_m],
        [x + radius_m, y + radius_m, z_top_m],
        [x - radius_m, y + radius_m, z_top_m],
    ], dtype=float)
    uv, front = camera.project(corners)
    if not np.any(front):
        return None
    uv = uv[front]
    x0 = max(0.0, min(1.0, float(np.min(uv[:, 0])) / float(width)))
    y0 = max(0.0, min(1.0, float(np.min(uv[:, 1])) / float(height)))
    x1 = max(0.0, min(1.0, float(np.max(uv[:, 0])) / float(width)))
    y1 = max(0.0, min(1.0, float(np.max(uv[:, 1])) / float(height)))
    return [x0, y0, x1, y1] if (x1 > x0 and y1 > y0) else None


def _identity_visible_at(robot, t_ms):
    evidence = robot.get('identity_evidence')
    if not isinstance(evidence, dict) or evidence.get('method') != 'body_gallery_v1':
        return True
    times = [point['t'] for point in (robot.get('trajectory') or []) if isinstance(point.get('t'), (int, float))]
    return bool(times) and min(times) <= t_ms <= max(times)


def _box_at(robot, t_ms, camera=None, width=1280, height=720, observed_only=False):
    if not _identity_visible_at(robot, t_ms):
        return None
    pts = [p for p in (robot.get('trajectory') or []) if p.get('box') is not None and len(p['box']) == 4 and p.get('t') is not None]
    if pts:
        before = max((p for p in pts if p['t'] <= t_ms), key=lambda p: p['t'], default=None)
        after = min((p for p in pts if p['t'] >= t_ms), key=lambda p: p['t'], default=None)
        # Source proof must be observed at this time, not snapped beyond a fragment
        # or bridged over an occlusion. Diagnostic matching keeps its wider fallback.
        if observed_only:
            if before is None or after is None:
                return None
            if before['t'] == after['t']:
                return [float(v) for v in before['box']]
            if after['t'] - before['t'] > BOX_MAX_GAP_MS:
                return None
        if before and after and before['t'] != after['t'] and (after['t'] - before['t']) <= BOX_INTERP_SPAN_MS:
            f = (t_ms - before['t']) / float(after['t'] - before['t'])
            return [float(before['box'][i]) + (float(after['box'][i]) - float(before['box'][i])) * f for i in range(4)]
        best = min(pts, key=lambda p: abs(p['t'] - t_ms))
        if abs(best['t'] - t_ms) <= BOX_MAX_GAP_MS:
            return [float(v) for v in best['box']]
    if camera is not None and not observed_only:
        xy = _pos_at(robot, t_ms)
        if xy is not None:
            return _project_robot_box(xy, camera, width, height)
    return None


def _unique_observed_origin(robot, robot_tracks, t_ms, u, v, camera=None, width=1280, height=720):
    """One measured body must contain the origin, considering opponents and rejected launch candidates too.

    A projected fallback box is useful for candidate diagnostics, but does not demonstrate the ball's source.
    """
    containing = []
    for other in robot_tracks:
        box = _box_at(other, t_ms, camera=camera, width=width, height=height, observed_only=True)
        if box is not None and box[0] <= u <= box[2] and box[1] <= v <= box[3]:
            containing.append(other)
    return len(containing) == 1 and containing[0] is robot


def _pos_at(robot, t_ms, max_gap_ms=600):
    if not _identity_visible_at(robot, t_ms):
        return None
    pts = [p for p in (robot.get('trajectory') or []) if p.get('x') is not None and p.get('y') is not None and p.get('t') is not None]
    if not pts:
        return None
    before = max((p for p in pts if p['t'] <= t_ms), key=lambda p: p['t'], default=None)
    after = min((p for p in pts if p['t'] >= t_ms), key=lambda p: p['t'], default=None)
    if before and after and before['t'] != after['t'] and (after['t'] - before['t']) <= POS_INTERP_SPAN_MS:
        f = (t_ms - before['t']) / float(after['t'] - before['t'])
        return float(before['x']) + (float(after['x']) - float(before['x'])) * f, float(before['y']) + (float(after['y']) - float(before['y'])) * f
    best = min(pts, key=lambda p: abs(p['t'] - t_ms))
    return (float(best['x']), float(best['y'])) if abs(best['t'] - t_ms) <= max_gap_ms else None


def expected_flight_ms(distance_m):
    """Empirical flight duration (ms) from launch to hub entry as a function of horizontal distance (m)."""
    if distance_m is None or not math.isfinite(float(distance_m)):
        return 700.0
    return min(1600.0, max(320.0, 200.0 + 170.0 * float(distance_m)))


def _robot_time_buckets(robot_tracks):
    """Conservative time index: retain the same ordering and every possible box/position snap."""
    buckets = {}
    pad = max(BOX_MAX_GAP_MS, 600)
    for robot in robot_tracks:
        times = [point['t'] for point in (robot.get('trajectory') or [])
                 if isinstance(point.get('t'), (int, float)) and math.isfinite(point['t'])]
        if not times:
            continue
        first, last = min(times), max(times)
        if last - first > 3_600_000:
            return None
        for second in range(math.floor((first - pad) / 1000), math.floor((last + pad) / 1000) + 1):
            buckets.setdefault(second, []).append(robot)
    return buckets


def launches(tracks, robot_tracks, offset_s, width, height, hubs=None, camera=None):
    """[{'t_ms', 'team_key', 'alliance', 'track_id', ...}] for each ball track that a robot shot."""
    hubs = hubs or default_hubs()
    hub_u_by_alliance = {}
    if camera is not None:
        for h_name, h_xy in hubs.items():
            uv, front = camera.project([[h_xy[0], h_xy[1], HUB_Z]])
            if bool(front[0]):
                hub_u_by_alliance[HUBS.get(h_name)] = float(uv[0, 0])

    out = []
    robot_buckets = _robot_time_buckets(robot_tracks)
    for track_id, track in enumerate(tracks):
        obs = _obs(track)
        if len(obs) < MIN_TRACK_OBS:
            continue
        head = obs[:LAUNCH_HEAD_OBS]
        rise_head = float(head[0][3] - head[-1][3])
        if rise_head < LAUNCH_RISE_PX:
            continue
        t_ms = int(round((obs[0][1] - offset_s) * 1000))
        u, v = obs[0][2] / width, obs[0][3] / height
        top = min(o[3] for o in obs) / height
        head_vel = obs[:min(len(obs), 5)]
        dt_h = max(0.02, float(head_vel[-1][1] - head_vel[0][1]))
        vu = float(head_vel[-1][5] - head_vel[0][5]) / dt_h
        vy = float(head_vel[-1][6] - head_vel[0][6]) / dt_h

        candidates = []
        active_robots = robot_tracks if robot_buckets is None else robot_buckets.get(t_ms // 1000, [])
        for robot in active_robots:
            box = _box_at(robot, t_ms, camera=camera, width=width, height=height)
            if box is None:
                continue
            w, h = box[2] - box[0], box[3] - box[1]
            if not (box[0] - BOX_EXPAND_X * w <= u <= box[2] + BOX_EXPAND_X * w):
                continue
            # Standard launch window: [box[1] - BOX_ABOVE * h, box[3]]
            in_standard_v = (box[1] - BOX_ABOVE * h <= v <= box[3])
            # High-stream pickup window: [box[1] - BOX_STREAM_ABOVE * h, box[1] - BOX_ABOVE * h) when moving rapidly upward toward own hub
            alliance = robot.get('alliance')
            hub_u = hub_u_by_alliance.get(alliance)
            toward_hub = (hub_u is None) or ((hub_u - float(obs[0][5])) * vu >= 0.0)
            in_stream_v = (
                not in_standard_v
                and (box[1] - BOX_STREAM_ABOVE * h <= v < box[1] - BOX_ABOVE * h)
                and len(obs) >= 4
                and vy <= -120.0
                and toward_hub
            )
            if not (in_standard_v or in_stream_v):
                continue
            if box[1] - top < LAUNCH_CLIMB * h:
                continue
            # Reject floor/bumper-level carpet bounces near the bottom of the box (v > box[1] + 0.65 * h)
            # unless the ball shows sustained upward climb across >= 4 frames and clears 0.6 * h above box[1].
            if v > box[1] + 0.65 * h:
                if len(obs) < 4 or not (obs[0][3] > obs[1][3] >= obs[2][3]) or (box[1] - top < 0.6 * h):
                    continue
            d = math.hypot(u - (box[0] + box[2]) / 2, v - (box[1] + box[3]) / 2)
            candidates.append((d, robot, box))
        if candidates:
            candidates.sort(key=lambda c: c[0])
            best_d, robot, best_box = candidates[0]
            margin = (candidates[1][0] - best_d) if len(candidates) > 1 else None
            box_diag = max(1e-3, math.hypot(best_box[2] - best_box[0], best_box[3] - best_box[1]))
            rel_d = best_d / box_diag
            if margin is not None and margin < 0.025:
                attr_label = 'ambiguous'
                conf = 0.58
            elif rel_d <= 0.45 and (margin is None or margin >= 0.05):
                attr_label = 'confident'
                conf = round(min(0.92, max(0.75, 0.90 - 0.25 * rel_d)), 2)
            else:
                attr_label = 'low_confidence'
                conf = 0.68
            alliance = robot.get('alliance')
            xy = _pos_at(robot, t_ms)
            hub_xy = hubs.get(f'{alliance}_hub') if alliance else None
            dist_m = math.hypot(xy[0] - hub_xy[0], xy[1] - hub_xy[1]) if (xy and hub_xy) else None
            out.append({
                't_ms': t_ms,
                'team_key': robot.get('team_key'),
                'track_key': robot.get('track_key'),
                'alliance': alliance,
                'track_id': track_id,
                'u': round(float(u), 4),
                'v': round(float(v), 4),
                'ru': round(float(obs[0][5]), 1),
                'rv': round(float(obs[0][6]), 1),
                'vu': round(vu, 1),
                'vy': round(vy, 1),
                'x': round(xy[0], 3) if xy else None,
                'y': round(xy[1], 3) if xy else None,
                'distance_to_hub_m': round(dist_m, 2) if dist_m is not None else None,
                'source_in_body': _unique_observed_origin(robot, active_robots, t_ms, u, v,
                                                         camera=camera, width=width, height=height),
                'attribution': attr_label,
                'confidence': conf,
            })
    out.sort(key=lambda e: (e['t_ms'], e.get('track_id', 0)))
    return out


def _pair_weight(entry, shot):
    """Compatibility weight between a hub `entry` and a candidate launch `shot` of the same alliance."""
    has_kinematics = (
        entry.get('ru') is not None
        or shot.get('ru') is not None
        or shot.get('distance_to_hub_m') is not None
    )
    if not has_kinematics:
        return 1.0
    dt = max(120.0, float(entry['t_ms'] - shot['t_ms']))
    dist_m = shot.get('distance_to_hub_m')
    if dist_m is not None:
        exp_dt = expected_flight_ms(dist_m)
        w_dt = math.exp(-0.5 * ((dt - exp_dt) / 350.0) ** 2) + 0.05 * (1.0 if dt <= 1500.0 else 0.25)
    else:
        # Flat plateau across the normal flight range [300, 1250 ms], decaying smoothly for older volley launches
        if dt < 300.0:
            w_dt = 0.65 + 0.35 * ((dt - 120.0) / 180.0)
        elif dt <= 1250.0:
            w_dt = 1.0
        else:
            w_dt = 0.1 + 0.9 * math.exp(-0.5 * ((dt - 1250.0) / 550.0) ** 2)
    w_dir = 1.0
    w_vel = 1.0
    if entry.get('ru') is not None and shot.get('ru') is not None:
        du_robot = float(entry['ru']) - float(shot['ru'])
        dt_s = max(0.15, dt / 1000.0)
        req_vu = du_robot / dt_s
        if entry.get('vu') is not None:
            w_vel *= math.exp(-0.5 * (abs(req_vu - float(entry['vu'])) / 180.0) ** 2)
        if shot.get('vu') is not None:
            w_vel *= math.exp(-0.5 * (abs(req_vu - float(shot['vu'])) / 220.0) ** 2)
        e_du = float(entry.get('du') or 0.0)
        if abs(e_du) >= 3.0 and abs(du_robot) >= 15.0:
            w_dir = 1.5 if ((e_du > 0) == (du_robot > 0)) else 0.1
    # Vertical entry speed consistency: farther shots enter from a higher arc with larger downward vv
    if entry.get('vv') is not None and dist_m is not None:
        exp_vv = min(260.0, max(60.0, 55.0 + 28.0 * float(dist_m)))
        w_vel *= 0.6 + 0.4 * math.exp(-0.5 * ((float(entry['vv']) - exp_vv) / 110.0) ** 2)
    conf_factor = float(shot.get('confidence') or 0.85)
    return max(1e-4, w_dt * w_dir * (0.25 + 0.75 * w_vel) * (0.5 + 0.5 * conf_factor))


def _entry_origin_robot(entry, robot_tracks, width, height, hubs, camera=None):
    """When a hub-entry track starts near a same-alliance robot box and rises before descending, link directly."""
    if not robot_tracks or entry.get('u0') is None or entry.get('v0') is None or entry.get('t0_ms') is None:
        return None
    dt_ms = entry['t_ms'] - entry['t0_ms']
    if not (50 <= dt_ms <= 2500):
        return None
    if float(entry.get('rise_px') or 0.0) < 3.0:
        return None
    u = float(entry['u0']) / max(1.0, float(width))
    v = float(entry['v0']) / max(1.0, float(height))
    top = float(entry.get('top_v', entry['v0'])) / max(1.0, float(height))
    t0_ms = entry['t0_ms']
    alliance = entry.get('alliance')
    candidates = []
    for robot in robot_tracks:
        if robot.get('alliance') != alliance:
            continue
        box = _box_at(robot, t0_ms, camera=camera, width=width, height=height, observed_only=True)
        if box is None:
            continue
        w, h = box[2] - box[0], box[3] - box[1]
        if not (box[0] - 0.35 * w <= u <= box[2] + 0.35 * w and box[1] - 0.7 * h <= v <= box[3] + 0.1 * h):
            continue
        if box[1] - top < 0.25 * h:
            continue
        d = math.hypot(u - 0.5 * (box[0] + box[2]), v - 0.5 * (box[1] + box[3]))
        supported = _unique_observed_origin(robot, robot_tracks, t0_ms, u, v,
                                            camera=camera, width=width, height=height)
        candidates.append((not supported, d, robot, box))
    if not candidates:
        return None
    candidates.sort(key=lambda c: (c[0], c[1]))
    robot = candidates[0][2]
    xy = _pos_at(robot, t0_ms)
    hub_xy = (hubs or {}).get(entry.get('hub'))
    dist_m = round(math.hypot(xy[0] - hub_xy[0], xy[1] - hub_xy[1]), 2) if (xy and hub_xy) else None
    conf = 0.86 if not candidates[0][0] or len(candidates) == 1 or (candidates[1][1] - candidates[0][1]) >= 0.04 else 0.72
    return {
        'team_key': robot.get('team_key'),
        'track_key': robot.get('track_key'),
        'launch_ms': t0_ms,
        'distance_to_hub_m': dist_m,
        'confidence': conf,
        'source_in_body': _unique_observed_origin(robot, robot_tracks, t0_ms, u, v,
                                                 camera=camera, width=width, height=height),
    }


def _robot_handle(robot, idx=None):
    tk = robot.get('team_key')
    if tk is not None:
        return tk
    trk = robot.get('track_key') or robot.get('id') or (idx if idx is not None else 'unlocated-launch')
    return f'{_UNNAMED_PREFIX}{trk}'


def _handle_to_team_key(handle):
    if handle is None or (isinstance(handle, str) and handle.startswith(_UNNAMED_PREFIX)):
        return None
    return handle


def _proximity_candidates(entry, shots, robot_tracks, width, height, hubs, hi_ms, camera=None):
    """Fallback attribution when no launch was detected inside the standard 3 s window.

    Includes both named and unnamed same-alliance robot tracks so an unnamed robot positioned near the hub
    keeps its share as `team_key = None` rather than letting a distant named robot steal proximity credit.
    """
    if not robot_tracks:
        return {}
    alliance = entry.get('alliance')
    hub_xy = (hubs or {}).get(entry.get('hub'))
    if not hub_xy:
        return {}
    t_query = entry['t_ms'] - 700
    e_du = float(entry.get('du') or 0.0)
    e_ru = entry.get('ru')

    positioned = []
    for idx, robot in enumerate(robot_tracks):
        if robot.get('alliance') != alliance:
            continue
        xy = _pos_at(robot, t_query)
        if xy is None:
            continue
        dist_m = math.hypot(xy[0] - hub_xy[0], xy[1] - hub_xy[1])
        box = _box_at(robot, t_query, camera=camera, width=width, height=height)
        robot_u = 0.5 * (box[0] + box[2]) * float(width) if box else None
        dir_ok = True
        if e_ru is not None and robot_u is not None and abs(e_du) >= 3.0 and abs(float(e_ru) - robot_u) >= 15.0:
            dir_ok = (e_du > 0) == ((float(e_ru) - robot_u) > 0)
        handle = _robot_handle(robot, idx)
        positioned.append((dist_m, dir_ok, robot, handle))

    if not positioned:
        return {}

    # 1. Extended volley continuation: robot launched in (hi_ms, VOLLEY_TAIL_MAX_MS] and is within VOLLEY_MAX_DIST_M
    recent_handles = {
        _robot_handle(s)
        for s in shots
        if s.get('alliance') == alliance and hi_ms < (entry['t_ms'] - s['t_ms']) <= VOLLEY_TAIL_MAX_MS
        and (s.get('team_key') or s.get('track_key') or s.get('id'))
    }
    volley_cands = [
        (dm, dir_ok, r, h) for dm, dir_ok, r, h in positioned
        if dm <= VOLLEY_MAX_DIST_M and h in recent_handles
    ]
    if volley_cands:
        weights = {}
        dist_by_handle = {}
        track_key_by_handle = {}
        for dm, dir_ok, r, h in volley_cands:
            w = (1.0 / max(1.0, dm)) * (1.4 if dir_ok else 0.35)
            weights[h] = weights.get(h, 0.0) + w
            dist_by_handle[h] = round(dm, 2)
            if r.get('track_key'):
                track_key_by_handle[h] = r['track_key']
        return {'weights': weights, 'dist_by_team': dist_by_handle, 'track_key_by_handle': track_key_by_handle, 'base_conf': 0.58}

    # 2. Close shooting-zone proximity (<= PROXIMITY_MAX_DIST_M) consistent with approach direction
    close_cands = [(dm, dir_ok, r, h) for dm, dir_ok, r, h in positioned if dm <= PROXIMITY_MAX_DIST_M and dir_ok]
    if close_cands:
        weights = {}
        dist_by_handle = {}
        track_key_by_handle = {}
        for dm, _, r, h in close_cands:
            weights[h] = weights.get(h, 0.0) + (1.0 / max(0.8, dm ** 1.5))
            dist_by_handle[h] = round(dm, 2)
            if r.get('track_key'):
                track_key_by_handle[h] = r['track_key']
        return {'weights': weights, 'dist_by_team': dist_by_handle, 'track_key_by_handle': track_key_by_handle, 'base_conf': 0.52}

    return {}


def _largest_remainder(fractions, total):
    floors = {k: int(math.floor(v)) for k, v in fractions.items()}
    left = total - sum(floors.values())
    for k in sorted(fractions, key=lambda k: (fractions[k] - floors[k], fractions[k]), reverse=True)[:max(0, left)]:
        floors[k] += 1
    return floors


def attribute_entries(entries, shots, robot_tracks=(), window_s=ENTRY_WINDOW_S, width=1280, height=720, hubs=None, ballistic_shots=(), camera=None, shift1_side=None):
    """Attribute each hub entry to a robot with method and confidence, and compute exact fractional credits.

    Returns (attributed_entries, credit, unattributed) where:
      * attributed_entries: list of dicts (one per entry in `entries`, in the same order), each with
        `team_key`, `track_key`, `launch_ms`, `flight_s`, `distance_to_hub_m`, `attribution`, `confidence`,
        `attribution_confidence`, and `hub_active`.
      * credit: {(alliance, team_key): fractional_balls} (where `team_key` is None for unnamed robot tracks)
      * unattributed: {'red': int, 'blue': int} (entries with no shooter in window or proximity)
    """
    hubs = hubs or default_hubs()
    lo_ms, hi_ms = int(round(window_s[0] * 1000)), int(round(window_s[1] * 1000))
    shots_sorted = [dict(s) for s in sorted(shots or (), key=lambda s: s['t_ms'])]

    robot_by_team = {}
    track_key_by_team = {}
    for r in robot_tracks or ():
        if r.get('team_key'):
            robot_by_team.setdefault(r['team_key'], r)
            if r.get('track_key') and r['team_key'] not in track_key_by_team:
                track_key_by_team[r['team_key']] = r['track_key']
    for s in shots_sorted:
        tk = s.get('team_key')
        if tk and s.get('track_key') and tk not in track_key_by_team:
            track_key_by_team[tk] = s['track_key']
        # Populate shot ru/rv/distance_to_hub_m from robot_tracks when missing and robot trajectory has box/(x,y)
        if tk and tk in robot_by_team:
            r = robot_by_team[tk]
            if s.get('ru') is None:
                box = _box_at(r, s['t_ms'], camera=camera, width=width, height=height)
                if box is not None:
                    s['ru'] = round(0.5 * (box[0] + box[2]) * float(width), 1)
                    s['rv'] = round(0.5 * (box[1] + box[3]) * float(height), 1)
            if s.get('distance_to_hub_m') is None:
                xy = _pos_at(r, s['t_ms'])
                hub_xy = hubs.get(f"{s.get('alliance')}_hub")
                if xy and hub_xy:
                    s['distance_to_hub_m'] = round(math.hypot(xy[0] - hub_xy[0], xy[1] - hub_xy[1]), 2)

    times = np.array([s['t_ms'] for s in shots_sorted], dtype=np.int64) if shots_sorted else np.zeros(0, dtype=np.int64)
    shots_by_tid = {s['track_id']: s for s in shots_sorted if s.get('track_id') is not None}
    ballistic_by_tid = {s['track_id']: s for s in (ballistic_shots or ()) if isinstance(s, dict) and s.get('track_id') is not None}

    credit = {}
    physical_credit = {}

    def add_credit(color, handle, amount):
        # Keep different unnamed shooters separate until after per-entry quotas are resolved.
        # Their public team credit still remains None; a missing name is never a shared physical identity.
        team = _handle_to_team_key(handle)
        credit[(color, team)] = credit.get((color, team), 0.0) + amount
        physical_credit[(color, handle)] = physical_credit.get((color, handle), 0.0) + amount

    unattributed = {'red': 0, 'blue': 0}
    staged = []

    for entry in entries:
        color = entry['alliance']
        t_ms = entry['t_ms']
        tid = entry.get('track_id')

        origin = _entry_origin_robot(entry, robot_tracks, width, height, hubs, camera=camera)
        observed_origin = origin is not None and origin.get('source_in_body') is True

        # 1. Exact same-track link (launch and hub entry on the same ball track)
        same_shot = shots_by_tid.get(tid) if tid is not None else None
        if (same_shot is not None and same_shot.get('alliance') == color and t_ms >= same_shot['t_ms']
                and (same_shot.get('source_in_body') is True or not observed_origin)):
            tk = same_shot.get('team_key')
            launch_ms = int(same_shot['t_ms'])
            flight_s = round(max(0.0, (t_ms - launch_ms) / 1000.0), 2)
            conf = round(min(0.95, max(0.84, float(same_shot.get('confidence') or 0.90))), 2)
            dist_m = same_shot.get('distance_to_hub_m')
            handle = _robot_handle(same_shot)
            add_credit(color, handle, 1.0)
            staged.append({
                **entry,
                'team_key': tk,
                'track_key': same_shot.get('track_key') or track_key_by_team.get(tk),
                'launch_ms': launch_ms,
                'flight_s': flight_s,
                'distance_to_hub_m': dist_m,
                'attribution': 'same_track',
                'source_in_body': same_shot.get('source_in_body') is True,
                'attribution_confidence': conf,
                'confidence': conf,
                '_fixed': True,
                '_handle': handle,
            })
            continue

        # Check ballistic shot on same track_id
        b_shot = ballistic_by_tid.get(tid) if tid is not None else None
        if (b_shot is not None and b_shot.get('alliance') == color and (b_shot.get('team_key') or b_shot.get('track_key'))
                and t_ms >= b_shot.get('t_ms', 0) and not observed_origin):
            tk = b_shot.get('team_key')
            launch_ms = int(b_shot['t_ms'])
            flight_s = round(max(0.0, (t_ms - launch_ms) / 1000.0), 2)
            conf = round(min(0.93, max(0.80, float(b_shot.get('confidence') or 0.86))), 2)
            dist_m = b_shot.get('distance_to_hub_m')
            handle = _robot_handle(b_shot)
            add_credit(color, handle, 1.0)
            staged.append({
                **entry,
                'team_key': tk,
                'track_key': b_shot.get('track_key') or track_key_by_team.get(tk),
                'launch_ms': launch_ms,
                'flight_s': flight_s,
                'distance_to_hub_m': dist_m,
                'attribution': 'same_track',
                'attribution_confidence': conf,
                'confidence': conf,
                '_fixed': True,
                '_handle': handle,
            })
            continue

        # Check if the entry track itself started at a same-alliance robot box (named or unnamed)
        if origin is not None:
            tk = origin.get('team_key')
            launch_ms = int(origin['launch_ms'])
            flight_s = round(max(0.0, (t_ms - launch_ms) / 1000.0), 2)
            conf = float(origin['confidence'])
            handle = _robot_handle(origin)
            add_credit(color, handle, 1.0)
            staged.append({
                **entry,
                'team_key': tk,
                'track_key': origin.get('track_key') or track_key_by_team.get(tk),
                'launch_ms': launch_ms,
                'flight_s': flight_s,
                'distance_to_hub_m': origin.get('distance_to_hub_m'),
                'attribution': 'same_track',
                'source_in_body': origin.get('source_in_body') is True,
                'attribution_confidence': conf,
                'confidence': conf,
                '_fixed': True,
                '_handle': handle,
            })
            continue

        # 2. Launches in the flight window before the entry
        has_rich = entry.get('track_id') is not None or entry.get('ru') is not None
        eff_lo = min(lo_ms, FAST_FLIGHT_MIN_MS) if has_rich else lo_ms
        if len(times):
            a = int(np.searchsorted(times, t_ms - hi_ms))
            b = int(np.searchsorted(times, t_ms - eff_lo, side='right'))
            near = [s for s in shots_sorted[a:b] if s.get('alliance') == color]
            if not near and eff_lo < lo_ms:
                a2 = int(np.searchsorted(times, t_ms - hi_ms))
                b2 = int(np.searchsorted(times, t_ms - lo_ms, side='right'))
                near = [s for s in shots_sorted[a2:b2] if s.get('alliance') == color]
        else:
            near = []

        if near:
            handle_weights = {}
            best_shot_by_handle = {}
            for s in near:
                handle = _robot_handle(s)
                w = _pair_weight(entry, s)
                handle_weights[handle] = handle_weights.get(handle, 0.0) + w
                prev = best_shot_by_handle.get(handle)
                if prev is None or w > prev[0]:
                    best_shot_by_handle[handle] = (w, s)
            total_w = sum(handle_weights.values()) or 1.0
            norm_weights = {handle: w / total_w for handle, w in handle_weights.items()}
            for handle, share in norm_weights.items():
                add_credit(color, handle, share)

            if len(norm_weights) == 1:
                handle = next(iter(norm_weights))
                tk = _handle_to_team_key(handle)
                best_s = best_shot_by_handle[handle][1]
                launch_ms = int(best_s['t_ms'])
                flight_s = round(max(0.0, (t_ms - launch_ms) / 1000.0), 2)
                s_conf = float(best_s.get('confidence') or 0.85)
                conf = round(min(0.90, max(0.70, 0.65 + 0.25 * s_conf)), 2)
                staged.append({
                    **entry,
                    'team_key': tk,
                    'track_key': best_s.get('track_key') or track_key_by_team.get(tk),
                    'launch_ms': launch_ms,
                    'flight_s': flight_s,
                    'distance_to_hub_m': best_s.get('distance_to_hub_m'),
                    'attribution': 'solo_launch',
                    'attribution_confidence': conf,
                    'confidence': conf,
                    '_fixed': True,
                    '_handle': handle,
                })
            else:
                staged.append({
                    **entry,
                    'team_key': None,
                    'track_key': None,
                    'launch_ms': None,
                    'flight_s': None,
                    'distance_to_hub_m': None,
                    'attribution': 'flight_time',
                    'attribution_confidence': round(min(0.84, max(0.48, 0.42 + 0.42 * max(norm_weights.values()))), 2),
                    'confidence': round(min(0.84, max(0.48, 0.42 + 0.42 * max(norm_weights.values()))), 2),
                    '_fixed': False,
                    '_weights': norm_weights,
                    '_best_shot': {handle: best_shot_by_handle[handle][1] for handle in norm_weights},
                })
            continue

        # 3. Proximity / extended burst fallback when robot trajectories are available
        prox = _proximity_candidates(entry, shots_sorted, robot_tracks, width, height, hubs, hi_ms, camera=camera)
        if prox and prox.get('weights'):
            raw_w = prox['weights']
            total_w = sum(raw_w.values()) or 1.0
            norm_weights = {h: w / total_w for h, w in raw_w.items()}
            dist_by_handle = prox['dist_by_team']
            trk_by_handle = prox.get('track_key_by_handle') or {}
            for handle, share in norm_weights.items():
                add_credit(color, handle, share)
            top_share = max(norm_weights.values())
            conf = round(min(0.65, max(0.40, float(prox['base_conf']) * (0.65 + 0.35 * top_share))), 2)
            if len(norm_weights) == 1:
                handle = next(iter(norm_weights))
                tk = _handle_to_team_key(handle)
                staged.append({
                    **entry,
                    'team_key': tk,
                    'track_key': trk_by_handle.get(handle) or track_key_by_team.get(tk),
                    'launch_ms': None,
                    'flight_s': None,
                    'distance_to_hub_m': dist_by_handle.get(handle),
                    'attribution': 'proximity',
                    'attribution_confidence': conf,
                    'confidence': conf,
                    '_fixed': True,
                    '_handle': handle,
                })
            else:
                staged.append({
                    **entry,
                    'team_key': None,
                    'track_key': None,
                    'launch_ms': None,
                    'flight_s': None,
                    'distance_to_hub_m': None,
                    'attribution': 'proximity',
                    'attribution_confidence': conf,
                    'confidence': conf,
                    '_fixed': False,
                    '_weights': norm_weights,
                    '_best_shot': {},
                    '_dist_by_handle': dist_by_handle,
                    '_trk_by_handle': trk_by_handle,
                })
            continue

        # 4. Unattributed
        unattributed[color] = unattributed.get(color, 0) + 1
        staged.append({
            **entry,
            'team_key': None,
            'track_key': None,
            'launch_ms': None,
            'flight_s': None,
            'distance_to_hub_m': None,
            'attribution': 'unattributed',
            'attribution_confidence': 0.0,
            'confidence': 0.55,
            '_fixed': True,
        })

    # Resolve multi-candidate entries into concrete per-ball team_key assignments (including `None` for unnamed
    # robot tracks so unnamed robot quota stays `None` rather than being forced onto named teammates).
    for color in ('red', 'blue'):
        all_fractions = {handle: v for (al, handle), v in physical_credit.items() if al == color}
        if not all_fractions:
            continue
        target_whole = _largest_remainder(all_fractions, int(round(sum(all_fractions.values()))))
        fixed_counts = {}
        for item in staged:
            if item['alliance'] == color and item.get('_fixed') and item.get('attribution') != 'unattributed':
                handle = item['_handle']
                fixed_counts[handle] = fixed_counts.get(handle, 0) + 1
        remaining = {tk: target_whole.get(tk, 0) - fixed_counts.get(tk, 0) for tk in all_fractions}

        multi_items = [item for item in staged if item['alliance'] == color and not item.get('_fixed')]
        multi_items.sort(key=lambda x: max((x.get('_weights') or {None: 0.0}).values()), reverse=True)
        for item in multi_items:
            weights = item.get('_weights') or {}
            cands = list(weights.keys())
            if not cands:
                continue
            chosen = max(
                cands,
                key=lambda tk: (
                    1 if remaining.get(tk, 0) > 0 else 0,
                    weights[tk] * (1.0 + 0.25 * max(0, remaining.get(tk, 0))),
                    remaining.get(tk, 0),
                ),
            )
            remaining[chosen] = remaining.get(chosen, 0) - 1
            item['team_key'] = _handle_to_team_key(chosen)
            best_s = (item.get('_best_shot') or {}).get(chosen)
            if best_s is not None:
                launch_ms = int(best_s['t_ms'])
                item['launch_ms'] = launch_ms
                item['flight_s'] = round(max(0.0, (item['t_ms'] - launch_ms) / 1000.0), 2)
                item['distance_to_hub_m'] = best_s.get('distance_to_hub_m')
                item['track_key'] = best_s.get('track_key') or track_key_by_team.get(chosen)
            else:
                item['distance_to_hub_m'] = (item.get('_dist_by_handle') or {}).get(chosen)
                item['track_key'] = (item.get('_trk_by_handle') or {}).get(chosen) or track_key_by_team.get(chosen)

    # v1.0.1: when the HUB lights were read, they decide the schedule ('unknown' leaves hub_active unknown in shifts).
    auto_winner = auto_winner_from_shift1(shift1_side) if shift1_side is not None else auto_winner_from_entries(staged)
    out = []
    for item in staged:
        clean = {k: v for k, v in item.items() if not k.startswith('_')}
        clean['phase'] = _phase(clean['t_ms'], scoring=True, launch_ms=clean.get('launch_ms'))
        clean['hub_active'] = hub_active_at(clean['alliance'], clean['t_ms'], auto_winner)
        out.append(clean)

    return out, credit, unattributed


def credit_entries(entries, shots, window_s=ENTRY_WINDOW_S, robot_tracks=(), width=1280, height=720, hubs=None, camera=None):
    """Share each hub entry among its alliance's robots that launched before it. Returns
    ({(alliance, team_key): fractional balls}, {alliance: unattributed balls}); team_key None is an unnamed robot."""
    _, credit, unattributed = attribute_entries(
        entries, shots, robot_tracks=robot_tracks, window_s=window_s, width=width, height=height, hubs=hubs, camera=camera
    )
    return credit, unattributed


def _timeline(events, step_ms=1000, end_ms=MATCH_MS):
    """Cumulative count per second: [[ms, count], ...]."""
    times = sorted(e['t_ms'] for e in events)
    out, k = [], 0
    for ms in range(0, end_ms + step_ms, step_ms):
        while k < len(times) and times[k] <= ms:
            k += 1
        out.append([ms, k])
    return out


def schedule_source(shift1_side, stats=None):
    """Where the HUB schedule came from: 'hub_lights', 'ball_entries' (v2 fallback, `shift1_from_entries`, recorded
    in stats['schedule']), 'unknown' when neither decided it, or 'vision_auto_counts' (v1, no schedule given)."""
    if shift1_side is None:
        return 'vision_auto_counts'
    if shift1_side not in ('red', 'blue'):
        return 'unknown'
    if ((stats or {}).get('schedule') or {}).get('shift1_side') == shift1_side:
        return 'ball_entries'
    return 'hub_lights'


def tracked_share(tracks):
    """Share of the match's seconds in which the team's robot tracks have a box or field position, or None without
    trajectories. A robot seen for a small share can only be credited for those seconds, so its count is a floor."""
    seconds = {int(p['t'] // 1000) for tr in tracks for p in (tr.get('trajectory') or [])
               if p.get('t') is not None and 0 <= p['t'] <= MATCH_MS
               and (p.get('box') is not None or (p.get('x') is not None and p.get('y') is not None))}
    if not any(tr.get('trajectory') for tr in tracks):
        return None
    return round(len(seconds) / -(-MATCH_MS // 1000), 3)


def refine_identities(path, attributed, robot_tracks, video, offset_s, width, height, source_exits=None):
    """Source-first identity refinement (vision/runner/identity_refinement.py) for this count: launch-time views of
    proven but unnamed sources, through the private model, with the unchanged identity rules. Never raises: any
    missing piece (inputs, gallery, model service) leaves the robot tracks as they were, with the reason recorded."""
    import os
    runner_dir = str(Path(__file__).resolve().parents[2] / 'vision' / 'runner')
    if runner_dir not in sys.path:
        sys.path.insert(0, runner_dir)
    try:
        import identity_refinement as ir
        from identity_gallery import IdentityGallery
        inputs = ir.load_inputs(path)
        gallery_dir, url, token = (os.environ.get('VISION_IDENTITY_GALLERY_DIR'), os.environ.get('VISION_QWEN_URL'),
                                   os.environ.get('VISION_QWEN_TOKEN'))
        if not (gallery_dir and url and token):
            return robot_tracks, {'query_plan': ir.QUERY_PLAN, 'status': 'not_configured'}
        templates = IdentityGallery(inputs['event_key'], Path(gallery_dir) / f"{inputs['event_key']}.gallery.json").templates(inputs['roster'])
        if templates['missing']:
            return robot_tracks, {'query_plan': ir.QUERY_PLAN, 'status': 'missing_reference_reads'}
        size, video_size = inputs['size'], (width, height)
        video_ms_of = lambda t: (offset_s + t / 1000.0) * 1000.0  # noqa: E731 - counter ms -> recording ms
        holder = {}

        def views(plan):
            holder['offset'] = ir.clock_link(inputs['tracklets'], robot_tracks, size)[0]
            return ir.body_views(str(video), plan, holder['offset'], size, video_size, video_ms_of, inputs['roster'],
                                 templates, url, token)

        def reads(plan):
            return ir.bumper_reads(str(video), plan, holder['offset'], size, video_size, video_ms_of, inputs['roster'],
                                   url, token)
        began = time.time()
        refined, stats = ir.refine(inputs, attributed, robot_tracks, templates, views, reads, source_exits=source_exits)
        stats['seconds'] = round(time.time() - began, 1)
        log(f"identity refinement: {json.dumps(stats, sort_keys=True)}")
        return refined, stats
    except Exception as error:  # noqa: BLE001 - refinement is optional evidence, never a reason to fail a count
        return robot_tracks, {'query_plan': 'source_first_launch_v2', 'status': 'failed', 'error': type(error).__name__}


def appearance_identities(path, robot_tracks, exits):
    """Name bodies that clearly look like one team, and carry the names onto their counted exits."""
    import robot_launches
    runner_dir = str(Path(__file__).resolve().parents[2] / 'vision' / 'runner')
    if runner_dir not in sys.path:
        sys.path.insert(0, runner_dir)
    try:
        import identity_refinement as ir
        named, stats = ir.appearance_names(ir.load_inputs(path), robot_tracks)
    except Exception as error:  # noqa: BLE001 - optional evidence, never a reason to fail a count
        return robot_tracks, exits, {'status': 'failed', 'error': type(error).__name__}
    exits = robot_launches.rename_exits(exits, [r for r in named if r.get('source_track_keys')])
    return [{k: v for k, v in r.items() if k != 'source_track_keys'} for r in named], exits, stats


def eliminate_identities(path, robot_tracks, exits):
    """Name bodies by elimination from their named alliance partners, and carry the names onto their counted exits."""
    import robot_launches
    runner_dir = str(Path(__file__).resolve().parents[2] / 'vision' / 'runner')
    if runner_dir not in sys.path:
        sys.path.insert(0, runner_dir)
    try:
        import identity_refinement as ir
        roster = ir.load_inputs(path)['roster']
    except Exception as error:  # noqa: BLE001 - optional evidence, never a reason to fail a count
        return robot_tracks, exits, {'status': 'failed', 'error': type(error).__name__}
    named, stats = robot_launches.eliminate_names(robot_tracks, roster)
    exits = robot_launches.rename_exits(exits, [r for r in named if r.get('source_track_keys')])
    return [{k: v for k, v in r.items() if k != 'source_track_keys'} for r in named], exits, stats


def evidenced_shooter_entries(attributed):
    """Confirmed robot credits need one continuous ball track beginning inside a unique observed robot box."""
    return [entry if entry.get('attribution') == 'inferred_path' or entry.get('attribution') == 'same_track' and entry.get('source_in_body') is True and not entry.get('stitched') and entry.get('source_continuous') is not False and not entry.get('nearer_body_path') else
            {**entry, 'candidate_team_key': entry.get('candidate_team_key', entry.get('team_key')),
             'candidate_track_key': entry.get('candidate_track_key', entry.get('track_key')),
             'candidate_method': entry.get('candidate_method', entry.get('attribution')), 'team_key': None, 'track_key': None,
             'attribution': 'unattributed'} for entry in attributed]


def summarize(entries, shots, robot_tracks, stats=None, attributed=None, shift1_side=None):
    """The fuel_counter snapshot payload."""
    entries = [e for e in entries if 0 <= e['t_ms'] <= MATCH_MS]
    shots = [s for s in shots if 0 <= s['t_ms'] <= MATCH_MS]
    # Attribution is a separate stage with known video geometry. A summary cannot invent that geometry or infer a
    # source from timing alone. This gate applies to full scouting and dedicated fuel jobs alike.
    attributed = evidenced_shooter_entries([e for e in (entries if attributed is None else attributed)
                                           if 0 <= e['t_ms'] <= MATCH_MS])
    auto_winner = auto_winner_from_shift1(shift1_side) if shift1_side is not None else auto_winner_from_entries(attributed)
    attributed = [{**e, 'phase': _phase(e['t_ms'], scoring=True, launch_ms=e.get('launch_ms')),
                   'hub_active': hub_active_at(e['alliance'], e['t_ms'], auto_winner)} for e in attributed]
    from robot_launches import identity_strength
    robots = []
    for robot in robot_tracks:
        key = robot.get('team_key')
        if key is None or any(r['team_key'] == key for r in robots):
            continue
        mine = [s for s in shots if s['team_key'] == key]
        # The team's strongest name: its first segment may be an unsupported carry while a later one was read, and the
        # site hides a robot whose evidence it does not accept, which would leave its credited balls unaccounted for.
        evidence = max((r.get('identity_evidence') for r in robot_tracks if r.get('team_key') == key),
                       key=identity_strength)
        robots.append({'team_key': key, 'alliance': robot.get('alliance'), 'shots': len(mine),
                       'auto_shots': sum(1 for s in mine if _phase(s['t_ms']) == 'auto'),
                       'teleop_shots': sum(1 for s in mine if _phase(s['t_ms']) == 'teleop'),
                       'identity_evidence': evidence,
                       'tracked_share': tracked_share([r for r in robot_tracks if r.get('team_key') == key])})
    alliances = {}
    for color in ('red', 'blue'):
        mine = [e for e in attributed if e['alliance'] == color]
        members = [r for r in robots if r['alliance'] == color]
        for r in members:
            my_entries = [e for e in mine if e.get('team_key') == r['team_key']]
            r['scored'] = len(my_entries)
            r['auto_scored'] = sum(1 for e in my_entries if e.get('phase') == 'auto')
            r['teleop_scored'] = r['scored'] - r['auto_scored']
            r['active_scored'] = None if any(e.get('hub_active') is None for e in my_entries) else sum(1 for e in my_entries if e.get('hub_active') is True)
            r['inactive_scored'] = sum(1 for e in my_entries if e.get('hub_active') is False)
            r['same_track_scored'] = sum(1 for e in my_entries if e.get('attribution') == 'same_track')
            r['inferred_scored'] = sum(1 for e in my_entries if e.get('attribution') == 'inferred_path')
            confidences = [float(e['attribution_confidence']) for e in my_entries
                           if isinstance(e.get('attribution_confidence'), (int, float))
                           and not isinstance(e['attribution_confidence'], bool) and math.isfinite(e['attribution_confidence'])]
            r['attribution_confidence'] = round(float(np.mean(confidences)), 2) if confidences else None
            r['timeline'] = _timeline(my_entries)           # when this robot's balls went in (Replay lights it up)
        attributed_sum = sum(r['scored'] for r in members)
        for robot in members:
            robot['attribution_complete'] = attributed_sum == len(mine)
        has_unknown_active = any(e.get('hub_active') is None for e in mine)
        active_scored = None if has_unknown_active else sum(1 for e in mine if e.get('hub_active') is True)
        inactive_scored = sum(1 for e in mine if e.get('hub_active') is False)
        alliances[color] = {
            'scored': len(mine),
            'balls_in_hub': len(mine),
            'active_scored': active_scored,
            'inactive_scored': inactive_scored,
            'auto_scored': sum(1 for e in mine if _phase(e['t_ms'], scoring=True, launch_ms=e.get('launch_ms')) == 'auto'),
            'teleop_scored': sum(1 for e in mine if _phase(e['t_ms'], scoring=True, launch_ms=e.get('launch_ms')) == 'teleop'),
            'shots': sum(1 for s in shots if s['alliance'] == color),
            'unattributed': max(0, len(mine) - attributed_sum),
            'timeline': _timeline(mine),
        }
    return {
        'version': 1,
        'method': 'ball tracks: hub entries and launches from robot boxes (fuel_count.py)',
        'auto_winner': auto_winner,
        'schedule_source': schedule_source(shift1_side, stats),
        'shift1_side': shift1_side,
        'alliances': alliances,
        'robots': robots,
        'stats': stats or {},
    }


def save_tracks(path, tracks, offset_s, width, height, stats=None, source_tracks=None):
    """Write the linked ball tracks (observations only) so counting rules can be re-scored without the video pass."""
    import gzip
    import pickle
    data = {'schema': 1, 'offset_s': float(offset_s), 'width': int(width), 'height': int(height),
            'stats': {k: v for k, v in (stats or {}).items() if isinstance(v, (int, float, str, bool, type(None)))},
            'source_breaks': [getattr(t, 'source_breaks', None) for t in tracks],
            'tracks': [[tuple(float(x) for x in o) for o in _obs(t)] for t in tracks]}
    if source_tracks is not None:
        data['launch_source_tracks'] = [[tuple(float(x) for x in o) for o in _obs(t)] for t in source_tracks]
        data['launch_source_breaks'] = [getattr(t, 'source_breaks', None) for t in source_tracks]
    with gzip.open(path, 'wb') as f:
        pickle.dump(data, f, protocol=4)


def load_tracks(path):
    """Read saved observations as BallTracks with join history (None for unknown legacy provenance)."""
    import gzip
    import pickle
    with gzip.open(path, 'rb') as f:
        data = pickle.load(f)
    # Keep the schema's observation lists readable by old evaluation tools; hydrate provenance for new callers.
    from fuel import BallTrack
    breaks = data.get('source_breaks')
    data['tracks'] = [BallTrack(obs, source_breaks=breaks[i] if isinstance(breaks, list) and i < len(breaks) else None)
                      for i, obs in enumerate(data['tracks'])]
    if 'launch_source_tracks' in data:
        provenance = data.get('launch_source_breaks')
        data['launch_source_tracks'] = [BallTrack(obs, source_breaks=provenance[i]
                                      if isinstance(provenance, list) and i < len(provenance) else None)
                                      for i, obs in enumerate(data['launch_source_tracks'])]
    return data


def unregistered_in_match(spans, offset_s, fps):
    """{'unregistered_in_match_s', 'unregistered_match_spans'}: how long, inside the match window (video seconds from
    `offset_s`), the camera could not be registered to the field, so no balls were seen. Unknown (None) without spans
    recorded; 0.0 when every frame of the match registered."""
    if spans is None:
        return {'unregistered_in_match_s': None}
    start, end, frame_s = float(offset_s), float(offset_s) + MATCH_MS / 1000.0, 1.0 / max(1.0, float(fps or 30.0))
    inside = [[max(a, start), min(b + frame_s, end)] for a, b in spans if b + frame_s > start and a < end]
    return {'unregistered_in_match_s': round(sum(b - a for a, b in inside), 2),
            'unregistered_match_spans': [[round(a - start, 2), round(b - start, 2)] for a, b in inside][:50]}


def align_robot_time(robot_tracks, source_start_s, counted_start_s):
    """Convert source match milliseconds via recording time, without changing the source trajectories."""
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
               for v in (source_start_s, counted_start_s)):
        raise ValueError('robot and counter recording origins must be finite seconds')
    shift_ms = (source_start_s - counted_start_s) * 1000
    return [{**track, 'trajectory': [{**p, 't': p['t'] + shift_ms} for p in track.get('trajectory', [])]}
            for track in robot_tracks]


def count_balls(video, start_s, end_s, file_offset, calibration, robot_tracks, workers=None, tracks_out=None,
                find_start=False, attribution_out=None, robot_match_start_s=None, identity_inputs=None):
    """Run the ball tracks over the match and count. `start_s`/`end_s` are video seconds; returns the payload.

    With `find_start`, the whole video is tracked and the match start is read from the video: the HUB lights' teleop
    pattern, or, where the lights cannot be read, the shift pattern of the balls entering the HUBs
    (`match_start_from_entries`); `start_s`/`end_s` are then ignored."""
    import fuel
    from pipeline import registration_problem
    from roi import combined_mask
    from run_match import pixel_homography, read_frame_at
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f'cannot open video {video}')
    width, height, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f'invalid video dimensions ({width}x{height}) for {video}')
    ref_t_s = calibration.get('ref_t_s')
    ref_t = float(ref_t_s if ref_t_s is not None else start_s) - file_offset
    ref_frame = read_frame_at(video, ref_t)
    if ref_frame is None:
        raise RuntimeError('could not read the calibration reference frame from the video')
    h_ref_to_field = pixel_homography(calibration, width, height)
    camera = camera_from_pixel_homography(h_ref_to_field, width, height)
    if camera is None:
        raise RuntimeError('could not recover a 3D camera from the field calibration')
    hubs = fuel.hubs_from_field()
    polygons = hub_polygons(camera, hubs)
    if len(polygons) < 2:
        raise RuntimeError('field calibration does not project both hubs in front of the camera')
    mask_ref = combined_mask(calibration, camera, width, height)
    offset_s = start_s - file_offset
    _, opening_mode, opening_region = opening_setup(calibration, hubs, width, height)
    began = time.time()
    lights_job = None
    if not find_start and counter_mode() == COUNTER_V2:
        # The HUB lights do not depend on the ball tracks: read them in a process of their own while the balls are
        # tracked (a process, not a thread: the tracking pool may fork, and a decoding thread must not be running then).
        import hub_lights
        rois = hub_lights.light_rois(calibration, hubs)
        if rois:
            lights_pool = fuel._process_pool(1)
            lights_job = lights_pool.submit(match_light_series, str(video), rois, offset_s)
            lights_pool.shutdown(wait=False)
    if find_start:
        cap = cv2.VideoCapture(str(video))
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        cap.release()
        offset_s, start_s, file_offset, end_s = 0.0, 0.0, 0.0, duration
    tracks, stats = fuel.run_fuel_pass_parallel(video, offset_s, end_s - file_offset, fps, h_ref_to_field, ref_frame, hubs,
                                                offset_s=offset_s, mask_ref=mask_ref, workers=workers, tracks_only=True,
                                                opening_region_ref=opening_region, opening_mode=opening_mode)
    if int((stats or {}).get('frames') or 0) <= 0:
        raise RuntimeError(f'no video frames decoded in match window {start_s:.1f}s-{end_s:.1f}s (file_offset={file_offset:.1f}s)')
    reg_err = registration_problem(stats, ref_t_s=ref_t_s, match_start_s=start_s)
    if reg_err:
        raise RuntimeError(reg_err)
    source_tracks = getattr(stats, 'source_tracks', None)
    source_pool = source_tracks if source_tracks is not None else tracks
    clean_stats = {k: v for k, v in (stats or {}).items() if k != 'rejections'}
    clean_stats['launch_track_pool'] = 'linked_plus_short_v1' if source_tracks is not None else 'legacy_linked'
    clean_stats = {**clean_stats, 'track_seconds': round(time.time() - began, 1), 'fps': round(fps, 2),
                   'track_fps': round(fuel.tracking_fps(fps), 2), 'video_size': [width, height],
                   **({'resolution_note': f'counted at {height}p; the counter was measured and its model fitted at {COUNT_MAX_HEIGHT}p'}
                      if height < COUNT_MAX_HEIGHT - 8 else {})}
    log(f"ball tracks: {len(tracks)} over {end_s - start_s:.0f}s ({clean_stats['track_seconds']}s)")
    lights_video = video
    if find_start:
        start_info = _find_match_start(video, tracks, calibration, hubs, width, height, end_s)
        if start_info.get('match_start_s') is None:
            raise RuntimeError(f"match start not found: {start_info}")
        offset_s = start_s = float(start_info['match_start_s'])
        clean_stats = {**clean_stats, 'match_start': start_info}
        if start_info['method'] != 'HUB lights':
            lights_video = None                 # unreadable lights: the schedule comes from the entries too
        log(f"match start {offset_s:.2f}s from {start_info['method']}")
    clean_stats = {**clean_stats, **unregistered_in_match(clean_stats.pop('unregistered_spans', None), offset_s, fps)}
    if tracks_out:
        save_tracks(tracks_out, tracks, offset_s=offset_s, width=width, height=height, stats=clean_stats, source_tracks=source_tracks)
    # v1.0.1+: openings placed through the measured height projection, the stricter entry test, and the HUB lights.
    entries, shift1_side, counter_info = count_entries(tracks, offset_s, calibration=calibration, hubs=hubs, width=width,
                                                       height=height, polygons=polygons, video=lights_video,
                                                       lights=lights_job.result() if lights_job is not None else None,
                                                       opening_mode=clean_stats.get('opening_detection', opening_mode))
    clean_stats = {**clean_stats, **counter_info}
    if robot_match_start_s is not None:
        robot_tracks = align_robot_time(robot_tracks, robot_match_start_s, offset_s + file_offset)
        clean_stats['robot_time_alignment'] = {'source_start_s': robot_match_start_s,
                                              'counted_start_s': offset_s + file_offset}
    shots = launches(tracks, robot_tracks, offset_s, width, height, hubs=hubs, camera=camera)
    attributed, _, _ = attribute_entries(entries, shots, robot_tracks=robot_tracks, width=width, height=height,
                                          hubs=hubs, camera=camera, shift1_side=shift1_side)
    diagnostic_assignments = attributed
    import robot_launches
    exits, exit_stats = robot_launches.observed_exits(source_pool, robot_tracks, offset_s, width, height)
    exit_stats['track_pool'] = clean_stats['launch_track_pool']
    attributed = evidenced_shooter_entries(robot_launches.mark_nearer_body_entries(attributed, tracks, robot_tracks, exits, offset_s, width, height))
    if identity_inputs and robot_tracks:
        refined, refinement = refine_identities(identity_inputs, attributed, robot_tracks, video, offset_s, width, height,
                                                source_exits=exits)
        if refinement.get('status') == 'refined':
            # Fail closed, one fragment at a time: refinement may only add names. A fragment that would move or drop a
            # supported credit keeps its original tracks; the rest of the refinement stands.
            withheld = set()
            for _ in range(REFINEMENT_ROUNDS):
                exits2 = robot_launches.rename_exits(exits, refined)
                shots2 = launches(tracks, refined, offset_s, width, height, hubs=hubs, camera=camera)
                raw2, _, _ = attribute_entries(entries, shots2, robot_tracks=refined, width=width, height=height,
                                               hubs=hubs, camera=camera, shift1_side=shift1_side)
                attributed2 = evidenced_shooter_entries(robot_launches.mark_nearer_body_entries(raw2, tracks, refined, exits, offset_s, width, height))
                moved = [a for a, b in zip(attributed, attributed2) if a.get('team_key') and b.get('team_key') != a.get('team_key')]
                bad = robot_launches.moved_sources(exits, exits2) | {a.get('track_key') for a in moved}
                if not bad:
                    break
                refined, restored = robot_launches.restore_sources(refined, robot_tracks, bad)
                withheld |= restored
            added = sum(1 for a, b in zip(attributed, attributed2) if b.get('team_key') and not a.get('team_key'))
            refinement.update(named_entries_added=added, supported_entries_moved=len(moved), withheld_sources=len(withheld))
            launch_check = robot_launches.refinement_change(exits, exits2)
            refinement.update(launch_check)
            if moved or launch_check['supported_launches_moved'] or launch_check['observed_exit_events_changed']:
                refinement['status'] = 'withheld_conflict'
            else:
                refined = [{k: v for k, v in r.items() if k != 'source_track_keys'} for r in refined]
                robot_tracks, shots, diagnostic_assignments, attributed = refined, shots2, raw2, attributed2
                exits = exits2
        clean_stats['identity_refinement'] = refinement
        robot_tracks, exits, clean_stats['appearance_names'] = appearance_identities(identity_inputs, robot_tracks, exits)
        robot_tracks, exits, clean_stats['alliance_elimination'] = eliminate_identities(identity_inputs, robot_tracks, exits)
        robot_tracks, clean_stats['roster_conflicts'] = robot_launches.resolve_name_conflicts(robot_tracks)
        exits = robot_launches.withdraw_exit_names(exits, robot_tracks)
    linked = robot_launches.link_same_ball_entries(attributed, exits, tracks, source_pool)
    clean_stats['same_ball_entries'] = sum(a is not b for a, b in zip(attributed, linked))
    import shooter_inference
    attributed, clean_stats['shooter_inference'] = shooter_inference.infer_shooters(linked, exits, tracks, robot_tracks, offset_s, width, height)
    clean_stats['shooter_attribution'] = 'continuous_ball_track_unique_body_origin'
    clean_stats['ball_source_provenance'] = 'tracker_fragment_intervals'
    clean_stats['unproven_ball_source_entries'] = sum(entry.get('source_continuous') is False for entry in attributed)
    clean_stats['physical_origin_entries'] = sum(entry.get('source_in_body') is True
                                                and entry.get('attribution') == 'same_track' for entry in attributed)
    payload = summarize(entries, shots, robot_tracks, clean_stats, attributed=attributed, shift1_side=shift1_side)
    payload['launches'] = robot_launches.summarize_exits(exits, robot_tracks, exit_stats)
    if attribution_out:
        # Private source evidence for a shooter audit, separate from the small Hub snapshot.
        import os
        with open(attribution_out, 'w') as audit:
            os.fchmod(audit.fileno(), 0o600)
            json.dump({'video_size': [width, height], 'offset_s': offset_s,
                       'entries': entries, 'shots': shots, 'observed_exits': exits, 'attributed': attributed,
                       'launch_track_pool': clean_stats['launch_track_pool'],
                       'diagnostic_assignments': diagnostic_assignments}, audit)
    import period_model
    payload = period_model.apply_estimate(payload, tracks, lifted(entry_openings(calibration, hubs, width, height),
                                                                  counter_info.get('ring_lift_px')), offset_s,
                                          entries, shift1_side,
                                          upper_openings=hub_openings(calibration, hubs, width, height, rim_z=TRUE_OPENING_MODEL_Z))
    payload = stamp_accuracy_release(payload, height)
    try:
        import fuel_features
        payload = fuel_features.from_fuel_count_payload(
            payload, robot_tracks, entries=attributed, shots=shots, tracks=tracks,
            calibration=calibration, offset_s=offset_s, width=width, height=height,
        )
    except Exception:  # noqa: BLE001 - optional feature module
        pass
    payload = scale_robot_credits(payload)                  # last: the feature module rebuilds the robot rows
    log(f"counted: red {payload['alliances']['red']['scored']} in, {payload['alliances']['red']['shots']} shot; "
        f"blue {payload['alliances']['blue']['scored']} in, {payload['alliances']['blue']['shots']} shot")
    return payload


def _find_match_start(video, tracks, calibration, hubs, width, height, duration_s):
    """{'match_start_s', 'method', ...}: the HUB lights' teleop pattern when readable, else the entries' shift pattern."""
    import hub_lights
    search = (0.0, max(0.0, duration_s - MATCH_MS / 1000.0))
    rois = hub_lights.light_rois(calibration, hubs)
    if rois:
        series = hub_lights.light_series(video, rois, 0.0, duration_s)
        t0, side, ev = hub_lights.match_start_from_lights(series, search=search)
        if t0 is not None:
            return {'match_start_s': round(t0, 2), 'method': 'HUB lights', 'shift1_side': side, **ev}
        light_reason = ev.get('reason')
    else:
        light_reason = 'no light boxes'
    openings = hub_openings(calibration, hubs, width, height)
    entries = hub_entries_strict(tracks, openings, 0.0) if openings else []
    t0, side, ev = match_start_from_entries(entries, search_s=search)
    return {'match_start_s': None if t0 is None else round(t0, 2), 'method': 'HUB entries', 'shift1_side': side,
            'lights': light_reason, **ev}


def main(argv=None):
    import media
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--youtube')
    ap.add_argument('--file')
    ap.add_argument('--file-offset-s', type=float, default=0.0)
    ap.add_argument('--start', type=float, default=0.0, help='video second the match starts')
    ap.add_argument('--find-start', action='store_true',
                    help='track the whole video and read the match start from it (HUB lights, else HUB entries)')
    ap.add_argument('--calibration', required=True, help='field calibration JSON (the view\'s)')
    ap.add_argument('--robot-tracks', help='JSON list of the scouting run\'s robot tracks (team_key, alliance, trajectory with boxes)')
    ap.add_argument('--robot-match-start-s', type=float, help='recording second used as the robot tracks time origin')
    ap.add_argument('--identity-inputs', help="the scouting run's private identity inputs (source-first refinement)")
    ap.add_argument('--workers', type=int)
    ap.add_argument('--out', required=True)
    ap.add_argument('--attribution-out', help='private JSON of entries, launches and shooter assignments for source-video audits')
    ap.add_argument('--tracks-out', help='also save the linked ball tracks (gzip pickle) for offline re-scoring')
    args = ap.parse_args(argv)
    if not (args.youtube or args.file):
        sys.exit('give --youtube or --file')
    if args.find_start and not args.file:
        sys.exit('--find-start reads the whole clip: give --file (--youtube downloads only the match window)')
    import shutil
    end = args.start + MATCH_MS / 1000
    tmp = None
    try:
        if args.file:
            video, file_offset = Path(args.file).expanduser(), args.file_offset_s
        else:
            log('download')
            tmp = media.tmpdir()
            video, file_offset = media.download_section(args.youtube, args.start - 1, end, tmp, max_height=COUNT_MAX_HEIGHT)
        calibration = json.loads(Path(args.calibration).read_text())
        robot_tracks = json.loads(Path(args.robot_tracks).read_text()) if args.robot_tracks else []
        log('counting balls')
        payload = count_balls(video, args.start, end, file_offset, calibration, robot_tracks, workers=args.workers,
                              tracks_out=args.tracks_out, find_start=args.find_start, attribution_out=args.attribution_out,
                              robot_match_start_s=args.robot_match_start_s, identity_inputs=args.identity_inputs)
        start = ((payload.get('stats') or {}).get('match_start') or {}).get('match_start_s', args.start)
        Path(args.out).write_text(json.dumps({'fuel_count': payload, 'match_start_s': start}))
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
