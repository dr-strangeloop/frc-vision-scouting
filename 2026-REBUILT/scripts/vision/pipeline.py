"""Frames -> per-robot field trajectories. Engine-agnostic: only the detector differs.

    frames --(cut check)--> camera motion --> detections --> ground points (field m) --> field tracker
        --> tracklets --> exact identity solve --> Kalman/RTS smoothing + heading --> tracks

Output tracks: trajectory points carry t in MATCH milliseconds, x/y in field meters (calibrated: true) and a normalized box for the
video overlay.
"""
import math
from dataclasses import dataclass

import cv2
import numpy as np

from camera_motion import CameraMotion, shot_changed
from common import FIELD
from detectors import infer_bumper_alliance
from identity import crop_clarity
from project import ground_point, in_field
from roi import warp_mask_to_frame
from roster_identity import infer_roster_identities
from smoothing import build_smooth_trajectory, resolve_heading, smooth_heading, smooth_track
from stitch import assign_teams, complete_teams, solve_alliance
from swaps import swap_risks

VMAX = FIELD['robot']['maxPlausibleSpeedMps']
CROPS_PER_TRACKLET = 6
MAX_UNREGISTERED_SHARE = 0.5


def registration_problem(stats, ref_t_s=None, match_start_s=None, max_unregistered_share=MAX_UNREGISTERED_SHARE):
    """Error message when most frames (> max_unregistered_share) cannot be registered to the calibration's reference frame."""
    frames = int((stats or {}).get('frames') or 0)
    unregistered = int((stats or {}).get('unregistered') or 0)
    if frames <= 0 or unregistered <= max_unregistered_share * frames:
        return None
    details = []
    if ref_t_s is not None:
        details.append(f'calibration taken at {float(ref_t_s):.1f}s')
    if match_start_s is not None:
        details.append(f'match starts at {float(match_start_s):.1f}s')
    prefix = f"{', '.join(details)}: " if details else ''
    return (
        f"{unregistered} of {frames} frames ({100 * unregistered / frames:.0f}%) could not be registered against "
        f"the calibration's reference frame ({prefix}recalibrate on a frame from the match)"
    )


def iter_frames(path, start_s, end_s, fps):
    """Yield (video_seconds, bgr) at about `fps` between start_s and end_s. Skipped frames are grabbed
    without decoding to pixels, which is several times faster than reading everything.

    The timestamp is read AFTER the frame is decoded: OpenCV's POS_MSEC is the time of the frame just
    read, so reading it before (as this once did) stamps every frame with its predecessor's time, one
    to two frame periods early. That does not change a speed, but it shifts every shot time and
    misaligns balls against robot positions. A keyframe seek can also land before start_s; frames
    earlier than the requested start are dropped instead of being passed on with early times.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f'cannot open video {path}')
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(native / fps))
    half_frame = 0.5 / native
    cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)
    index = 0
    while True:
        if not cap.grab():
            break
        if index % step == 0:
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if t > end_s:
                break
            if t >= start_s - half_frame:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                yield t, frame
        index += 1
    cap.release()


@dataclass
class Tracklet:
    alliance: str
    points: list          # dicts: t, x, y, sigma, box, angle, conf, team
    votes: dict
    crops: list = None    # best few (score, t, bgr crop) for reading the bumper number
    evidence: list = None

    @property
    def start(self): return self.points[0]['t']
    @property
    def end(self): return self.points[-1]['t']
    @property
    def first(self): return (self.points[0]['x'], self.points[0]['y'])
    @property
    def last(self): return (self.points[-1]['x'], self.points[-1]['y'])
    @property
    def confidence(self): return sum(p['conf'] for p in self.points) / len(self.points)


def _box_norm_iou(a, b):
    """IoU of two [x1, y1, x2, y2] normalized boxes when both have area > 1e-5."""
    if not (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == 4 and len(b) == 4):
        return 1.0
    try:
        ax1, ay1, ax2, ay2 = (float(v) for v in a)
        bx1, by1, bx2, by2 = (float(v) for v in b)
    except (TypeError, ValueError):
        return 1.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    if area_a <= 1e-5 or area_b <= 1e-5:
        return 1.0
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    if inter <= 0.0:
        return 0.0
    return inter / max(1e-12, area_a + area_b - inter)


def _select_tracklet_crops(candidates, t_start, t_end, k=CROPS_PER_TRACKLET):
    """Select up to k candidate crops (score, t, crop), spread across time bins when span >= 3.0 s."""
    if not candidates or k <= 0:
        return []
    ranked = sorted(candidates, key=lambda c: c[0], reverse=True)
    span = float(t_end) - float(t_start)
    if len(ranked) <= k or span < 3.0 or k < 2:
        return ranked[:k]
    bins = [[] for _ in range(k)]
    for c in ranked:
        b_idx = min(k - 1, max(0, int((float(c[1]) - float(t_start)) / span * k)))
        bins[b_idx].append(c)
    picked, picked_ids = [], set()
    for grp in bins:
        if grp:
            picked.append(grp[0])
            picked_ids.add(id(grp[0]))
    for c in ranked:
        if len(picked) >= k:
            break
        if id(c) not in picked_ids:
            picked.append(c)
            picked_ids.add(id(c))
    return sorted(picked, key=lambda c: c[0], reverse=True)[:k]


class FieldTracker:
    """Associate detections to tracklets in FIELD space, per alliance, within a physical speed gate.

    Working in meters (not pixels) is the point: the gate is physical (a robot cannot teleport) and it
    is independent of where the camera is pointing.

    predictive=True (the default) is a small multi-object tracker built for six robots on a field:
      - motion prediction: each tracklet's velocity is fit to its last half second, and a detection is scored
        by its distance to where the robot should be now, scaled by how uncertain that prediction is. Matching
        to the last position (the original rule, kept as predictive=False) swaps two robots that cross at speed.
      - optimal assignment: all pairs at once (Hungarian), not nearest-first, so one greedy pair cannot steal
        the detection another tracklet needed.
      - a matching cascade (the ByteTrack idea): established tracklets take confident detections first, then weak
        ones; a tracklet seen only once is tentative and gets what is left, so a stray box cannot steal a real
        robot's detections. Weak detections start a tracklet only while the alliance has fewer than three
        established robots, so a false positive cannot become a fourth red robot. Detector confidence scales
        differ (YOLO, Gemini, colour), so confirmation by a second sighting does the heavy lifting, not a threshold.
      - duplicate suppression: two same-alliance detections closer than a bumper's width are one robot seen
        twice (two boxes on one robot), never two robots.
      - bumper reads: a detection that read a team number is expensive to give to a tracklet that already
        voted for a different team.
    """

    SLOTS = 3                 # robots per alliance
    DUP_RADIUS_M = 0.5        # robot centres can never be closer than ~0.8 m (bumpers)
    STRONG_CONF = 0.5
    MAX_COST = 4.0            # gate in prediction standard deviations
    CONFIRM_HITS = 2          # a tracklet is established once it has been seen twice
    TENTATIVE_GAP_S = 0.5     # an unconfirmed tracklet not seen again this soon was a stray detection

    def __init__(self, max_gap_s=1.2, predictive=True):
        self.max_gap = max_gap_s
        self.predictive = predictive
        self.tracklets = []
        self.active = []

    def step(self, t, dets):
        if not self.predictive:
            return self._step_nearest(t, dets)
        dets = self._reconcile_bumper_alliance(t, dets)
        for alliance in ('red', 'blue'):
            live = [tr for tr in self.active if tr.alliance == alliance and t - tr.end <= (self.max_gap if self._confirmed(tr) else self.TENTATIVE_GAP_S)]
            confirmed = [tr for tr in live if self._confirmed(tr)]
            tentative = [tr for tr in live if not self._confirmed(tr)]
            mine = self._suppress_duplicates([d for d in dets if d['alliance'] == alliance], t=t, confirmed=confirmed)
            strong = [j for j, d in enumerate(mine) if d['conf'] >= self.STRONG_CONF]
            weak = [j for j, d in enumerate(mine) if d['conf'] < self.STRONG_CONF]
            # Cascade: established robots get first pick of confident detections, then of weak ones; only what is
            # left can extend a tentative tracklet (one seen once), so a stray box cannot steal a real robot.
            matched_t, used_d = self._associate(t, confirmed, mine, strong)
            more_t, more_d = self._associate(t, [tr for i, tr in enumerate(confirmed) if i not in matched_t], mine, weak)
            used_d |= more_d
            _, late_d = self._associate(t, tentative, mine, [j for j in range(len(mine)) if j not in used_d])
            used_d |= late_d
            alive = len(live)
            # New tracklets: confident leftovers, and weak ones only while fewer than three robots are being tracked.
            for j in sorted((j for j in range(len(mine)) if j not in used_d), key=lambda j: -mine[j]['conf']):
                if mine[j]['conf'] < self.STRONG_CONF and alive >= self.SLOTS:
                    continue
                tr = Tracklet(alliance, [], {})
                self._add(tr, t, mine[j])
                self.tracklets.append(tr)
                self.active.append(tr)
                alive += 1
        # A tentative tracklet that was not seen again soon is dropped from association (it stays in the list;
        # build_tracks ignores single points).
        self.active = [tr for tr in self.active if t - tr.end <= (self.max_gap if self._confirmed(tr) else self.TENTATIVE_GAP_S)]

    def _confirmed(self, tr):
        return len(tr.points) >= self.CONFIRM_HITS

    def _reconcile_bumper_alliance(self, t, dets):
        """When a detector inferred the wrong alliance from a misread cross-alliance team number, but the bumper
        colour and field continuity agree on the opposite alliance, restore the bumper alliance and drop the bad read."""
        if not any(d.get('bumper_alliance') and d['bumper_alliance'] != d.get('alliance') for d in dets):
            return dets
        confirmed_by = {
            a: [tr for tr in self.active if tr.alliance == a and self._confirmed(tr) and t - tr.end <= self.max_gap]
            for a in ('red', 'blue')
        }
        out = []
        for d in dets:
            ba = d.get('bumper_alliance')
            da = d.get('alliance')
            if ba in ('red', 'blue') and da in ('red', 'blue') and ba != da:
                near_claimed = any(math.hypot(d['x'] - tr.last[0], d['y'] - tr.last[1]) <= 1.5 for tr in confirmed_by[da])
                near_true = any(math.hypot(d['x'] - tr.last[0], d['y'] - tr.last[1]) <= 1.2 for tr in self.tracklets if tr.alliance == ba and len(tr.points) >= 2 and 0 <= t - tr.end <= 10.0)
                if not near_claimed and (near_true or len(confirmed_by[da]) >= self.SLOTS):
                    d = {**d, 'alliance': ba, 'team': None}
            out.append(d)
        return out

    def _associate(self, t, live, dets, candidates):
        """Assign detections (indices `candidates`) to `live` tracklets; extend the matched ones.
        Returns (matched tracklet indices into live, matched detection indices)."""
        if not live or not candidates:
            return set(), set()
        big = 1e6
        costs = [[big] * len(candidates) for _ in live]
        preds = [self._predict(tr, t) for tr in live]
        live_votes = []
        for tr in live:
            pos_votes = {tm: v for tm, v in (tr.votes or {}).items() if v > 0}
            tot = sum(pos_votes.values())
            top = max(pos_votes, key=pos_votes.get) if tot > 0 else None
            share = (pos_votes[top] / tot) if top else 0.0
            live_votes.append((tot, top, share))

        for i, tr in enumerate(live):
            px, py, spread = preds[i]
            dt = max(t - tr.end, 1e-3)
            total_votes, top_team, top_share = live_votes[i]
            for k, j in enumerate(candidates):
                d = dets[j]
                if math.hypot(d['x'] - tr.last[0], d['y'] - tr.last[1]) > VMAX * 1.25 * dt + 0.5 + 2 * d['sigma']:
                    continue
                raw_dist = math.hypot(d['x'] - px, d['y'] - py)
                gate_cost = raw_dist / math.hypot(spread, d['sigma'])
                if gate_cost > self.MAX_COST:
                    continue
                cost = gate_cost + 0.08 * max(0.0, dt - 0.20)
                d_team = d.get('team')
                if d_team:
                    if top_team and d_team == top_team:
                        cost = max(0.0, cost - 1.0)
                        gate_cost = max(0.0, gate_cost - 1.0)
                    elif top_team and d_team != top_team:
                        if tr.votes[top_team] >= 1.5 and top_share >= 0.70 and d.get('conf', 0.6) >= 0.70:
                            continue
                        elif tr.votes[top_team] >= 0.6:
                            cost += 2.5
                            gate_cost += 2.5
                    if not (top_team and d_team == top_team):
                        vetoed = False
                        penalized = False
                        for o_idx, other in enumerate(live):
                            if other is tr:
                                continue
                            _, other_top, _ = live_votes[o_idx]
                            if other_top == d_team:
                                o_votes = other.votes.get(d_team, 0.0)
                                if o_votes >= 1.2 and math.hypot(d['x'] - other.last[0], d['y'] - other.last[1]) <= 1.5:
                                    if o_votes >= 1.8:
                                        vetoed = True
                                        break
                                    if not penalized:
                                        cost += 2.5
                                        gate_cost += 2.5
                                        penalized = True
                                elif o_votes >= 1.0 and not penalized:
                                    cost += 2.0
                                    gate_cost += 2.0
                                    penalized = True
                        if vetoed:
                            continue
                if gate_cost <= self.MAX_COST:
                    costs[i][k] = cost
        pairs = _assign(costs, big)
        pair_by_i = dict(pairs)
        matched_t, matched_d = set(), set()
        for i, k in pairs:
            j = candidates[k]
            d = dets[j]
            if not d.get('team') and len(live) >= 2 and self._confirmed(live[i]):
                ambiguous = False
                for i2, tr2 in enumerate(live):
                    if i2 == i or not self._confirmed(tr2) or costs[i2][k] >= big or i2 in pair_by_i:
                        continue
                    if abs(costs[i2][k] - costs[i][k]) < 0.35 and math.hypot(preds[i][0] - preds[i2][0], preds[i][1] - preds[i2][1]) < 0.34:
                        ambiguous = True
                        break
                if ambiguous:
                    matched_t.add(i)
                    matched_d.add(j)
                    continue
            self._add(live[i], t, d)
            matched_t.add(i)
            matched_d.add(j)
        return matched_t, matched_d

    @staticmethod
    def _predict(tr, t):
        """Where the robot should be at t, and the 1-sigma spread of that guess (meters)."""
        last = tr.points[-1]
        dt = max(0.0, t - last['t'])
        recent = [p for p in tr.points[-6:] if last['t'] - p['t'] <= 0.6]
        vx = vy = 0.0
        if len(recent) >= 2 and recent[-1]['t'] - recent[0]['t'] > 1e-3:
            n = len(recent)
            ts = [p['t'] for p in recent]
            mt = sum(ts) / n
            den = sum((x - mt) ** 2 for x in ts)
            if den > 1e-9:
                vx = sum((p['t'] - mt) * p['x'] for p in recent) / den
                vy = sum((p['t'] - mt) * p['y'] for p in recent) / den
                speed = math.hypot(vx, vy)
                if speed > VMAX:
                    vx, vy = vx * VMAX / speed, vy * VMAX / speed
        # Extrapolate at most ~0.6 s of velocity: a robot hidden longer has probably turned or stopped.
        lead = min(dt, 0.6)
        spread = 0.25 + 1.2 * dt + (0.0 if len(recent) >= 3 else 0.3)
        return last['x'] + vx * lead, last['y'] + vy * lead, spread

    def _suppress_duplicates(self, dets, t=None, confirmed=None):
        kept = []
        for d in sorted(dets, key=lambda d: -d['conf']):
            twin = None
            for k in kept:
                dist = math.hypot(k['x'] - d['x'], k['y'] - d['y'])
                if dist >= self.DUP_RADIUS_M:
                    continue
                ba, bb = k.get('box_norm'), d.get('box_norm')
                if ba != bb and _box_norm_iou(ba, bb) < 0.20 and dist >= 0.30:
                    continue
                twin = k
                break
            if twin is None:
                kept.append(d)
            elif d.get('team') and not twin.get('team'):
                if _box_norm_iou(twin.get('box_norm'), d.get('box_norm')) >= 0.35 or math.hypot(twin['x'] - d['x'], twin['y'] - d['y']) < 0.25:
                    twin['team'] = d['team']          # keep the read only when genuinely overlapping or very close
        return kept

    def _step_nearest(self, t, dets):
        """The original association: nearest last position first, within the speed gate."""
        for alliance in ('red', 'blue'):
            mine = [d for d in dets if d['alliance'] == alliance]
            live = [tr for tr in self.active if tr.alliance == alliance and t - tr.end <= self.max_gap]
            pairs = []
            for i, tr in enumerate(live):
                dt = max(t - tr.end, 1e-3)
                for j, d in enumerate(mine):
                    dist = math.hypot(d['x'] - tr.last[0], d['y'] - tr.last[1])
                    if dist <= VMAX * 1.25 * dt + 0.5 + 2 * d['sigma']:
                        pairs.append((dist, i, j))
            used_t, used_d = set(), set()
            for _, i, j in sorted(pairs):
                if i in used_t or j in used_d:
                    continue
                used_t.add(i)
                used_d.add(j)
                self._add(live[i], t, mine[j])
            for j, d in enumerate(mine):
                if j not in used_d:
                    tr = Tracklet(alliance, [], {})
                    self._add(tr, t, d)
                    self.tracklets.append(tr)
                    self.active.append(tr)
        self.active = [tr for tr in self.active if t - tr.end <= self.max_gap]

    @staticmethod
    def _add(tr, t, d):
        tr.points.append({'t': t, 'x': d['x'], 'y': d['y'], 'sigma': d['sigma'], 'box': d['box_norm'], 'angle': d.get('angle'), 'conf': d['conf'], 'team': d.get('team')})
        crop = d.get('crop')
        if crop is not None and getattr(crop, 'size', 0):
            occluded = bool(d.get('bumper_occluded'))
            candidates = getattr(tr, '_candidate_crops', None)
            if candidates is None:
                candidates = list(tr.crops) if tr.crops else []
            if not (occluded and candidates):
                x1, y1, x2, y2 = d.get('box_norm') or (0.1, 0.1, 0.9, 0.9)
                edge = min(x1, y1, 1.0 - x2, 1.0 - y2)
                wall = min(d['x'], FIELD['length'] - d['x'], d['y'], FIELD['width'] - d['y'])
                edge_f = (1.0 if edge > 0.015 else 0.4) * (1.0 if wall > 0.3 else 0.5)
                score = crop.shape[0] * crop.shape[1] * d['conf'] * edge_f * crop_clarity(crop)
                if occluded:
                    score *= 0.10
                close_idx = next((i for i, c in enumerate(candidates) if abs(c[1] - t) <= 0.8), None)
                if close_idx is not None:
                    if score > candidates[close_idx][0]:
                        candidates[close_idx] = (score, t, crop)
                else:
                    candidates.append((score, t, crop))
                candidates = _select_tracklet_crops(candidates, tr.start, tr.end, 12)
                tr._candidate_crops = candidates
                tr.crops = _select_tracklet_crops(candidates, tr.start, tr.end, CROPS_PER_TRACKLET)
        if d.get('team'):
            tr.votes[d['team']] = tr.votes.get(d['team'], 0.0) + d['conf']
            if d.get('evidence_source') != 'none':
                tr.evidence = tr.evidence or []
                tr.evidence.append({'source': d.get('evidence_source') or 'wide_frame', 'team': str(d['team']), 'confidence': float(d['conf']), 't': float(t)})


def _assign(costs, big):
    """Minimum-cost pairs (row, col) of a cost matrix, skipping entries >= big. Hungarian via scipy when it is
    installed; otherwise exact search for the small case (a few robots) and greedy beyond it."""
    rows = len(costs)
    cols = len(costs[0]) if rows else 0
    if not rows or not cols:
        return []
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(np.array(costs))
        return [(i, k) for i, k in zip(r.tolist(), c.tolist()) if costs[i][k] < big]
    except ImportError:
        pass
    flat = sorted((costs[i][k], i, k) for i in range(rows) for k in range(cols) if costs[i][k] < big)
    used_r, used_c, out = set(), set(), []
    for _, i, k in flat:
        if i not in used_r and k not in used_c:
            used_r.add(i)
            used_c.add(k)
            out.append((i, k))
    return out


def register_and_project(frames, detector, h_ref_to_field, auto_start_s=0.0, min_inliers=25, batch=16, on_progress=None, ref_frame=None, keep_crops=False, mask_ref=None, contact_only=False, ref_t_s=None, match_start_s=None, max_unregistered_share=None):
    """Run detection + camera registration over frames. Returns (tracker, stats).

    frames is consumed in batches so a GPU detector sees batch-sized inputs (the big speed lever on the
    detector side), while camera motion, which is stateful, still runs frame by frame in order.
    """
    from identity import bumper_occluded
    try:
        from project import inside_hub_core
    except ImportError:
        try:
            from offline_auto import inside_hub_core
        except ImportError:
            def inside_hub_core(x, y):
                half = FIELD['hub']['halfSize'] - 0.15
                return any(
                    abs(x - cx) < half and abs(y - cy) < half
                    for cx, cy in (
                        (FIELD['hub']['x'], FIELD['hub']['y']),
                        (FIELD['length'] - FIELD['hub']['x'], FIELD['width'] - FIELD['hub']['y']),
                    )
                )

    motion = None
    prev_gray = None
    tracker = FieldTracker()
    stats = {'frames': 0, 'cuts': 0, 'unregistered': 0, 'static_frames': 0, 'detections': 0, 'dropped_out_of_field': 0}
    roi_box = cv2.boundingRect(mask_ref) if mask_ref is not None else None
    identity = np.eye(3)
    buffer = []

    def flush():
        if not buffer:
            return
        results = detector.detect([(t, img) for t, img, _, _ in buffer])
        for (t, img, h_cur_to_ref, _), dets in zip(buffer, results):
            img_h, img_w = img.shape[:2]
            projected = []
            h_cur = h_ref_to_field @ h_cur_to_ref
            inside = None
            if mask_ref is not None:
                inside = mask_ref if np.array_equal(h_cur_to_ref, identity) else warp_mask_to_frame(mask_ref, h_cur_to_ref, img_w, img_h)
            all_px_boxes = [det.box for det in dets if det.alliance is not None]
            for det in dets:
                if det.alliance is None:
                    continue
                if inside is not None:
                    mx, my = (det.box[0] + det.box[2]) / 2, det.box[3]
                    cx = min(max(int(mx), 0), img_w - 1) if 0 <= mx <= img_w else -1
                    cy = min(max(int(my), 0), img_h - 1) if 0 <= my <= img_h else -1    # bottom edge = where it meets the carpet
                    if not (0 <= cx < img_w and 0 <= cy < img_h and inside[cy, cx]):
                        stats['outside_roi'] = stats.get('outside_roi', 0) + 1
                        continue
                x, y, sigma = ground_point(det.box, h_cur, inset_m=0 if contact_only else .4)
                if not in_field(x, y) or inside_hub_core(x, y):
                    stats['dropped_out_of_field'] += 1
                    continue
                x1, y1, x2, y2 = det.box
                crop = None
                if keep_crops:
                    cx1, cy1, cx2, cy2 = (max(0, int(v)) for v in det.box)
                    crop = img[cy1:cy2, cx1:cx2].copy() if cx2 > cx1 and cy2 > cy1 else None
                projected.append({'alliance': det.alliance, 'bumper_alliance': infer_bumper_alliance(img, det.box, min_share=0.75, min_pixels=16),
                                  'team': det.team, 'x': x, 'y': y, 'sigma': sigma, 'conf': det.confidence, 'crop': crop,
                                  'bumper_occluded': bumper_occluded(det.box, [b for b in all_px_boxes if b != det.box]),
                                  'angle': _field_angle(det.angle, h_cur, det.box), 'box_norm': [x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h]})
            stats['detections'] += len(projected)
            tracker.step(t, projected)
        buffer.clear()

    for t, img in frames:
        stats['frames'] += 1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Only the wide camera's region is used to register the camera (a split-screen broadcast has
        # close-up panels whose motion has nothing to do with the field camera).
        if motion is None:
            motion = CameraMotion(min_inliers=min_inliers)
            if ref_frame is not None:
                # The field homography was solved on ref_frame, so everything is registered to it.
                motion.start(cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY), mask_ref)
                h_cur_to_ref, ok = motion.update(gray, ref_mask=mask_ref)
            else:
                motion.start(gray, mask_ref)
                h_cur_to_ref, ok = np.eye(3), True
        else:
            # A hard cut is only counted (the registration below handles it: it fails to match and the frame
            # is skipped). It is judged on the field camera's region, because in a split-screen broadcast the
            # close-up panels cut constantly and say nothing about the wide camera.
            if shot_changed(prev_gray, gray, roi=roi_box):
                stats['cuts'] += 1
            h_cur_to_ref, ok = motion.update(gray, ref_mask=mask_ref)
        prev_gray = gray
        if not ok:
            stats['unregistered'] += 1
            continue
        stats['static_frames'] = motion.static_frames
        buffer.append((t, img, h_cur_to_ref.copy(), None))
        if len(buffer) >= batch:
            flush()
        if on_progress:
            on_progress(t)
    flush()
    if max_unregistered_share is not None:
        problem = registration_problem(stats, ref_t_s=ref_t_s, match_start_s=match_start_s, max_unregistered_share=max_unregistered_share)
        if problem:
            raise RuntimeError(problem)
    return tracker, stats


def _field_angle(angle_px, h_cur_to_field, box):
    """Image-plane orientation of a box's long side -> field-plane angle, via the homography.

    Orientation in the image is foreshortened and sheared by perspective, so project a short segment
    along the box's direction from the box's ground point and read the angle in the field."""
    if angle_px is None or not math.isfinite(angle_px):
        return None
    x1, y1, x2, y2 = box
    p = np.array([(x1 + x2) / 2.0, y2])
    d = np.array([math.cos(angle_px), math.sin(angle_px)]) * 10.0
    a = h_cur_to_field @ np.array([p[0], p[1], 1.0])
    b = h_cur_to_field @ np.array([p[0] + d[0], p[1] + d[1], 1.0])
    if abs(a[2]) < 1e-9 or abs(b[2]) < 1e-9:
        return None
    a, b = a[:2] / a[2], b[:2] / b[2]
    angle = math.atan2(b[1] - a[1], b[0] - a[0])
    return angle if math.isfinite(angle) else None


def split_contaminated_tracklets(tracklets, min_half_votes=1.2, min_half_share=0.65):
    """Split tracklets whose timestamped team reads switch mid-tracklet from dominant team T1 to T2 != T1."""
    def _half_votes(pts, ev_items):
        votes = {}
        has_pt_team = False
        for p in pts:
            tm = p.get('team')
            if tm:
                has_pt_team = True
                s_tm = str(tm)
                votes[s_tm] = votes.get(s_tm, 0.0) + float(p.get('conf', 0.6))
        for ev in ev_items:
            if not isinstance(ev, dict) or not ev.get('team'):
                continue
            if has_pt_team and ev.get('source', 'wide_frame') == 'wide_frame':
                continue
            s_tm = str(ev['team'])
            votes[s_tm] = votes.get(s_tm, 0.0) + float(ev.get('confidence', 0.6))
        return votes

    def _dom(votes):
        tot = sum(v for v in votes.values() if v > 0)
        if tot <= 0:
            return None, 0.0, 0.0
        top = max(votes, key=votes.get)
        w = votes[top]
        share = w / tot
        if w >= min_half_votes and share >= min_half_share:
            return top, w, share
        return None, w, share

    out = []
    queue = list(tracklets or [])
    while queue:
        tr = queue.pop(0)
        pts = getattr(tr, 'points', None) or []
        ev_all = getattr(tr, 'evidence', None) or []
        pt_teams = {str(p['team']) for p in pts if p.get('team')}
        ev_teams = {str(e['team']) for e in ev_all if isinstance(e, dict) and e.get('team') and e.get('t') is not None}
        if len(pts) < 6 or len(pt_teams | ev_teams) < 2:
            out.append(tr)
            continue
        best_key = None
        best_split = None
        for k in range(2, len(pts) - 1):
            split_t = pts[k]['t']
            ev_1 = [e for e in ev_all if isinstance(e, dict) and e.get('t') is not None and float(e['t']) < split_t]
            ev_2 = [e for e in ev_all if isinstance(e, dict) and e.get('t') is not None and float(e['t']) >= split_t]
            v1 = _half_votes(pts[:k], ev_1)
            v2 = _half_votes(pts[k:], ev_2)
            t1, w1, _ = _dom(v1)
            t2, w2, _ = _dom(v2)
            if t1 and t2 and t1 != t2:
                jump = math.hypot(pts[k]['x'] - pts[k - 1]['x'], pts[k]['y'] - pts[k - 1]['y'])
                key = (round(w1 + w2, 6), round(jump, 6))
                if best_key is None or key > best_key:
                    best_key = key
                    best_split = (k, split_t, v1, v2, ev_1, ev_2)
        if best_split is None:
            out.append(tr)
            continue
        k_star, split_t, v1, v2, ev_1, ev_2 = best_split
        crops_all = getattr(tr, 'crops', None) or []
        crops_1 = [c for c in crops_all if len(c) >= 2 and float(c[1]) < split_t] or None
        crops_2 = [c for c in crops_all if len(c) >= 2 and float(c[1]) >= split_t] or None
        tr1 = Tracklet(tr.alliance, list(pts[:k_star]), v1, crops=crops_1, evidence=ev_1 if ev_all else None)
        tr2 = Tracklet(tr.alliance, list(pts[k_star:]), v2, crops=crops_2, evidence=ev_2 if ev_all else None)
        queue = [tr1, tr2] + queue
    return out


def build_tracks(tracker, roster, offset_s, min_track_s=0.5, slots=3, angle_period=math.pi / 2, pseudo=False, *,
                 match_key=None, prior_tracks=None, appearance_classifier=None, fingerprint_lib=None,
                 reconnect_events=None, reconnect_validations=None):
    """Tracklets -> one smoothed trajectory per robot, with team assignment and honest confidences.

    offset_s is the video second that is match time 0 (the auto start), subtracted so stored times
    are match milliseconds. `pseudo` means there was no field calibration: positions are in a stand-in
    field scaled from the picture, so they are written back as 0-1 picture coordinates with
    calibrated=false (good for drawing boxes on the video, refused for auto paths).
    Returns a list of track dicts (see module docstring)."""
    import identity_priors
    import stitch as _stitch
    from identity_priors import collect_chain_priors
    from stitch import INDIRECT_PRIOR_SOURCES, _link_cost, solve_chain_identities
    if not pseudo and prior_tracks is None and match_key and hasattr(identity_priors, 'load_prior_run_tracks'):
        prior_tracks = identity_priors.load_prior_run_tracks(match_key)
    out = []
    for alliance in ('red', 'blue'):
        tracklets = [t for t in tracker.tracklets if t.alliance == alliance and t.end - t.start >= 0.1 and len(t.points) >= 2]
        if not tracklets:
            continue
        tracklets = split_contaminated_tracklets(tracklets)
        dicts = [{'start': t.start, 'end': t.end, 'first': t.first, 'last': t.last, 'votes': dict(t.votes or {}), 'weight': t.confidence,
                  'points': t.points, 'evidence': list(t.evidence) if getattr(t, 'evidence', None) else None,
                  'crops': getattr(t, 'crops', None), 'fragment_id': getattr(t, 'fragment_id', f't{idx}')} for idx, t in enumerate(tracklets)]
        if not pseudo and prior_tracks and hasattr(identity_priors, 'anchor_early_tracklets_from_prior'):
            identity_priors.anchor_early_tracklets_from_prior(dicts, prior_tracks, alliance, offset_s=offset_s)
        if reconnect_events is not None and reconnect_validations is not None and hasattr(_stitch, 'apply_validated_reconnects'):
            alliance_events = [ev for ev in reconnect_events if isinstance(ev, dict) and ev.get('alliance') in (None, alliance)]
            chains, _, _ = _stitch.apply_validated_reconnects(dicts, alliance_events, reconnect_validations, slots=slots)
        else:
            chains = solve_alliance(dicts, slots=slots)
        claimed = {idx for c in chains for idx in c}
        early_unclaimed = [u for u in range(len(dicts)) if u not in claimed and dicts[u]['start'] - offset_s <= 2.0]
        if early_unclaimed:
            for c in chains:
                if not c or dicts[c[0]]['start'] - offset_s <= 2.0:
                    continue
                head = dicts[c[0]]
                feasible = [
                    (_link_cost(dicts[u], head), u)
                    for u in early_unclaimed
                    if u not in claimed and dicts[u]['end'] <= head['start'] + 0.15
                ]
                feasible = [(cost, u) for cost, u in feasible if cost is not None]
                if feasible:
                    _, best_u = min(feasible, key=lambda item: item[0])
                    c.insert(0, best_u)
                    claimed.add(best_u)
        chain_priors = collect_chain_priors(
            chains, dicts, roster[alliance], alliance, offset_s=offset_s,
            match_key=match_key, prior_tracks=prior_tracks,
            appearance_classifier=appearance_classifier, fingerprint_lib=fingerprint_lib,
        ) if (not pseudo and any(x is not None for x in (match_key, prior_tracks, appearance_classifier, fingerprint_lib))) else None
        teams, completed, chain_evidences = solve_chain_identities(chains, dicts, roster[alliance], chain_priors=chain_priors)
        # Names a decisive read does not back, kept aside: they are used only if roster elimination below cannot name the robot.
        fallbacks = [None if team else (t, s, src) for (team, _), (t, s, src) in zip(teams, completed)]
        first_of_alliance, joins = len(out), []
        for chain, (team, share), (_, _, comp_src), fallback, ev_summary in zip(chains, teams, completed, fallbacks, chain_evidences):
            if not chain:
                continue
            raw_points = sorted((p for i in chain for p in tracklets[i].points), key=lambda p: (round((p['t'] - offset_s) * 1000), -p['conf']))
            points = []
            for p in raw_points:
                if not points or round((p['t'] - offset_s) * 1000) != round((points[-1]['t'] - offset_s) * 1000):
                    points.append(p)
            if len(points) < 2 or points[-1]['t'] - points[0]['t'] < min_track_s:
                continue
            times = np.array([p['t'] for p in points])
            sm = smooth_track(times, [p['x'] for p in points], [p['y'] for p in points], [p['sigma'] for p in points])
            trajectory = build_smooth_trajectory(points, offset_s, sm, angle_period=angle_period, pseudo=pseudo)
            if len(trajectory) < 2 or (trajectory[-1]['t'] - trajectory[0]['t']) < min_track_s * 1000:
                continue
            kept_points = [p for i, p in enumerate(points) if sm['kept'][i]]
            tracking = round(float(sum(p['conf'] for p in kept_points) / len(kept_points) * (1 - 0.1 * (len(chain) - 1) / max(1, len(chain)))), 2)
            team_key = (str(team) if str(team).startswith('frc') else f'frc{team}') if team else None
            id_source = (comp_src or 'reads') if team else None
            ev = dict(ev_summary) if isinstance(ev_summary, dict) else {}
            ev['winning_team'] = team_key
            if not team_key:
                ev['decisive'] = False
                ev['primary_source'] = None
            out.append({'track_key': f'{alliance}-{len(out)}', 'team_key': team_key, 'alliance': alliance,
                        'started_ms': trajectory[0]['t'], 'ended_ms': trajectory[-1]['t'],
                        'identity_confidence': share if team else 0.0, 'tracking_confidence': tracking, 'trajectory': trajectory,
                        'needs_review': share < 0.6 or len(chain) > 3 or id_source in INDIRECT_PRIOR_SOURCES, 'tracklets': len(chain),
                        'identity_source': id_source, 'identity_evidence': ev, '_fallback': fallback})
            ordered = sorted((tracklets[i] for i in chain), key=lambda t: t.start)
            join_ts = [t.end for t in ordered[:-1]] + [t.start for t in ordered[1:]]
            joins.append(sorted({int(round((t - offset_s) * 1000)) for t in join_ts}))
        if not pseudo:
            # Two of this alliance's robots meeting while one of their chains was being joined is where a swap
            # can hide: send those tracks to a person rather than trusting them.
            mine = out[first_of_alliance:]
            for track, risk in zip(mine, swap_risks(mine, joins)):
                track['swap_risks'] = risk
                if risk:
                    track['needs_review'] = True
    if not pseudo and len(out) == 2 * slots and any(not t.get('team_key') for t in out):
        named = {id(t) for t in out if t.get('team_key')}
        inferred, _ = infer_roster_identities(out, roster)
        for before, after in zip(out, inferred):
            if after.get('team_key') and id(before) not in named:
                after['identity_source'] = 'roster_inferred'
        out = inferred
    out = fill_unnamed(out)
    if not pseudo and match_key:
        try:
            import cache
            identity_priors.save_run_identity_cache(match_key, out)
            if cache.enabled() and hasattr(identity_priors, 'EventAutoFingerprintLibrary'):
                event_key = str(match_key).split('_')[0]
                fp_lib = fingerprint_lib or identity_priors.EventAutoFingerprintLibrary.load(event_key) or identity_priors.EventAutoFingerprintLibrary(event_key=event_key)
                added = False
                for tr in out:
                    if fp_lib.add_track(match_key, tr, offset_s=0.0):
                        added = True
                if added:
                    fp_lib.save()
        except Exception:
            pass
    return out


def fill_unnamed(tracks):
    """Every robot gets a team number: a track still unnamed takes its weak-read or elimination name (stitch.complete_teams)
    unless another track of its alliance already carries that team. Filled-in names are marked for review."""
    for track in tracks:
        fallback = track.pop('_fallback', None)
        if track.get('team_key') or not fallback or not fallback[0]:
            continue
        team, confidence, source = fallback
        team_key = str(team) if str(team).startswith('frc') else f'frc{team}'
        if any(other.get('team_key') == team_key for other in tracks if other is not track and other.get('alliance') == track.get('alliance')):
            continue
        prev_ev = track.get('identity_evidence') if isinstance(track.get('identity_evidence'), dict) else {}
        ev = {
            'winning_team': team_key,
            'winning_weight': prev_ev.get('winning_weight', 0.0),
            'total_weight': prev_ev.get('total_weight', 0.0),
            'share': prev_ev.get('share', 0.0),
            'margin': prev_ev.get('margin', 0.0),
            'runner_up_team': prev_ev.get('runner_up_team'),
            'runner_up_weight': prev_ev.get('runner_up_weight', 0.0),
            'decisive': False,
            'primary_source': source,
            'by_team': prev_ev.get('by_team', {}),
        }
        track.update({'team_key': team_key, 'identity_confidence': confidence, 'identity_source': source, 'identity_evidence': ev, 'needs_review': True})
    return tracks
