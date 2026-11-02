"""Fuel (the yellow foam balls): detect, track, fit each flight in 3D, say what it did and who shot it.

    frames --> camera motion --> moving-yellow blobs --> image tracks --> ballistic fit per flight
           --> hub crossing test (scored / rim / miss / short) --> shooter attribution --> shots

What this can and cannot honestly give:
  * Launch speed, angle, apex, distance and where the ball crosses hub height come from the fit of a whole
    flight, with a pixel residual and a speed standard deviation, so unreliable fits say so.
  * A ball seen for too little of its flight, or lost in a volley, is dropped rather than guessed, so the
    per-shot list can undercount. That is why alliance TOTALS are anchored to the official match result
    (fuelAnalytics.js) and the shot list is used for who/when/how fast.
"""
import bisect
import itertools
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field

import cv2
import numpy as np

from ballistics import BALL_RADIUS, HUB_OPENING_RADIUS, HUB_Z, classify_flight, fit_flight, state_at_exit
from scipy.optimize import linear_sum_assignment
from camera3d import camera_from_pixel_homography, smooth_cameras
from camera_motion import CameraMotion
from common import AUTO_GRACE_MS, FIELD, log
from roi import warp_mask_to_frame


# The ball tracker is tuned for ~30 fps steps: at 60 fps a stream ball moves ~3 px per frame, less than its own radius, and
# neighbours in a 30 balls/s stream get confused (fuel_rate_bench: 16 of 60 balls tracked at 60 fps vs 47 at 30 fps).
TRACK_MAX_FPS = 30.0


OPENING_DETECTION_MODES = ('wide', 'background')
WIDE_LOW, WIDE_HIGH = (15, 60, 90), (50, 255, 255)  # v1.0.4a: near the openings, the lit HUB pushes yellow toward hue ~40-50
def _env_number(name, default):
    """An experiment knob: VISION_FUEL_<NAME> overrides a detection constant for A/B runs (recorded in the stats)."""
    try:
        return type(default)(os.environ.get(f'VISION_FUEL_{name}', '') or default)
    except ValueError:
        return default


BG_ALPHA = _env_number('BG_ALPHA', 0.05)              # v1.0.4b: running background near the openings (~20 frames)
BG_MIN_RG = _env_number('BG_MIN_RG', 20)              # v1.0.4b: a ball raises red and green by this much over it ...
BG_MIN_YELLOW_SHIFT = _env_number('BG_MIN_YELLOW_SHIFT', 30)  # ... and their mean this much more than blue
RED_GLOW = os.environ.get('VISION_FUEL_RED_GLOW', '').strip().lower() in ('1', 'true', 'yes')
RED_GLOW_BG_MIN = _env_number('RED_GLOW_BG_MIN', 200)
OPENING_ABOVE_PX = _env_number('OPENING_ABOVE_PX', 60)  # the opening region reaches this far above each opening ...
OPENING_SIDE_PX = _env_number('OPENING_SIDE_PX', 10)    # ... and this far to each side


def opening_detection_mode():
    """The detection used around the HUB openings: 'background' (v1.0.4b, the default), 'wide' (v1.0.4a), or None
    (v1.0.3 detector) when VISION_FUEL_OPENING_DETECTION is 'off'. Unknown values fall back to the default."""
    mode = os.environ.get('VISION_FUEL_OPENING_DETECTION', '').strip().lower()
    if mode in ('off', 'none', '0'):
        return None
    return mode if mode in OPENING_DETECTION_MODES else 'background'


DEFAULT_MOTION_THRESHOLD = 18
STREAM_SPLIT_AREA = 1.65     # a region this many single-ball areas big (with a camera) is split as a stream ...
STREAM_SPLIT_AREA_EXP = 1.3  # ... or this many with VISION_FUEL_SPLIT_STREAMS=1


def _env_flag(name):
    return os.environ.get(name, '').strip().lower() in ('1', 'true', 'yes')


def motion_threshold():
    """Grey-level change a moving ball must show (default 18). Only ~40% of balls lost against the lit blue HUB clear
    18; VISION_FUEL_MOTION_THRESHOLD lowers it for A/B runs.
    Off by default until it is measured on the Spark."""
    try:
        value = int(os.environ.get('VISION_FUEL_MOTION_THRESHOLD', '') or DEFAULT_MOTION_THRESHOLD)
    except ValueError:
        return DEFAULT_MOTION_THRESHOLD
    return min(40, max(6, value))


def split_streams_mode():
    """Split touching balls more eagerly (from 1.3 single-ball areas instead of 1.65), also inside the changed-pixel
    regions carved out of piles. On by default with counter v2 (2026cc qm1/qm3 A/B: WAPE 16.7% -> 12.1%, 4/4 within
    ±20%); VISION_FUEL_SPLIT_STREAMS=0/1 overrides, and VISION_FUEL_COUNTER=v1.0.4b turns it off."""
    raw = os.environ.get('VISION_FUEL_SPLIT_STREAMS', '').strip().lower()
    if raw:
        return raw in ('1', 'true', 'yes')
    return os.environ.get('VISION_FUEL_COUNTER', '').strip().lower() not in ('v1.0.4b', 'v1.0.4', 'v1')


def counter_options():
    """The experimental counter options switched on for this run (recorded in the run's stats), [] by default."""
    opts = []
    if motion_threshold() != DEFAULT_MOTION_THRESHOLD:
        opts.append(f'motion{motion_threshold()}')
    if split_streams_mode():
        opts.append('split_streams')
    if _env_flag('VISION_FUEL_VANISHED_RULE'):
        opts.append('vanished')
    if RED_GLOW:
        opts.append('red_glow')
    for name, default in (('BG_ALPHA', 0.05), ('BG_MIN_RG', 20), ('BG_MIN_YELLOW_SHIFT', 30),
                          ('OPENING_ABOVE_PX', 60), ('OPENING_SIDE_PX', 10)):
        value = globals()[name]
        if value != default:
            opts.append(f'{name.lower()}={value}')
    return opts


UNREGISTERED_SPAN_GAP_S = 0.5


def note_unregistered(stats, t_s, gap_s=UNREGISTERED_SPAN_GAP_S):
    """Record a frame that could not be registered to the field in `stats['unregistered_spans']` ([start, end] video
    seconds, merged across gaps up to `gap_s`), so the counter can say how much of the match itself went unseen."""
    spans = stats.setdefault('unregistered_spans', [])
    t_s = round(float(t_s), 2)
    if spans and t_s - spans[-1][1] <= gap_s:
        spans[-1][1] = max(spans[-1][1], t_s)
    else:
        spans.append([t_s, t_s])


def tracking_fps(video_fps):
    """Frames per second the ball tracker reads: the video's rate, or less with VISION_FUEL_TRACK_FPS (e.g. 30 to track
    every other frame of a 60 fps broadcast, about half the work) or the high-rate mode's 30 fps cap."""
    fps = float(video_fps)
    raw = os.environ.get('VISION_FUEL_TRACK_FPS', '').strip()
    try:
        cap = float(raw) if raw else None
    except ValueError:
        cap = None
    if cap is not None and cap > 0:
        fps = min(fps, cap)
    if high_rate_mode():
        fps = min(fps, TRACK_MAX_FPS)
    return fps


def high_rate_mode():
    """VISION_FUEL_HIGH_RATE=1 turns on the high fire-rate detector and tracker (camera-sized splitting of streams and dumps,
    optimal-assignment tracking, a 30 fps tracking cap). On the synthetic 30 balls/s bench it tracks 34-35 of 60 balls
    instead of 12-17, but on real 2026cc auto footage it credited fewer balls (qm7: 58 balls without a shooter vs 10), so it
    stays opt-in until it is tuned on labelled real video. Default runs use the original detector and tracker."""
    return os.environ.get('VISION_FUEL_HIGH_RATE', '').strip().lower() in ('1', 'true', 'yes')
MIN_SHOT_APEX = 1.4        # meters: lower than this is a roll or a bounce, not a shot at a hub 1.83 m up


@dataclass
class Blob:
    t: float
    u: float                 # pixel position in this frame (what the 3D fit uses)
    v: float
    area: float
    circularity: float
    ru: float = None         # position in the reference frame (what tracking and "did it move" use)
    rv: float = None

    def __post_init__(self):
        self.ru = self.u if self.ru is None else self.ru
        self.rv = self.v if self.rv is None else self.rv


def _split_touching_blobs(sub_mask, t, ox0, oy0, single_ball_area=85.0):
    """Split a merged moving-yellow region into individual ball centroids using distance-transform peaks."""
    dist = cv2.distanceTransform(sub_mask, cv2.DIST_L2, 3)
    dil = cv2.dilate(dist, np.ones((7, 7), np.uint8))
    peaks = ((dist >= 2.2) & (dist >= dil - 1e-5)).astype(np.uint8)
    n_peaks, peak_labels, _, peak_cent = cv2.connectedComponentsWithStats(peaks)
    if n_peaks <= 2:
        return None
    seeds = [
        (float(dist[peak_labels == k].max()), float(peak_cent[k][0]), float(peak_cent[k][1]))
        for k in range(1, n_peaks)
    ]
    seeds.sort(reverse=True)
    kept = []
    for r, x, y in seeds:
        ok = True
        for kr, kx, ky in kept:
            sep = math.hypot(x - kx, y - ky)
            if sep < max(5.2, 1.32 * max(r, kr)):
                ok = False
                break
            steps = max(2, int(round(sep)) + 1)
            xs = np.clip(np.round(np.linspace(x, kx, steps)).astype(int), 0, dist.shape[1] - 1)
            ys = np.clip(np.round(np.linspace(y, ky, steps)).astype(int), 0, dist.shape[0] - 1)
            if float(dist[ys, xs].min()) > 0.85 * min(r, kr):
                ok = False
                break
        if ok:
            kept.append((r, x, y))
    if len(kept) <= 1:
        return None
    return [
        Blob(t, x + ox0, y + oy0, min(single_ball_area, math.pi * max(2.5, r) ** 2), 0.85)
        for r, x, y in kept
    ]


def expected_ball_radius_px(camera, u, v, z_flight=1.35):
    """Radius in pixels a ball would have at picture point (u, v) at a typical flight height, from the 3D camera."""
    if camera is None:
        return None
    length, width = float(FIELD['length']), float(FIELD['width'])
    p = camera.point_at_height(float(u), float(v), z_flight)
    if p is None:
        p = np.array([0.5 * length, 0.5 * width, z_flight], float)
    else:
        p = np.array([np.clip(p[0], -1.0, length + 1.0), np.clip(p[1], -1.0, width + 1.0), z_flight], float)
    depth = float((camera.r @ p + camera.t)[2])
    return float(camera.f * BALL_RADIUS / depth) if depth > 1.0 else None


def _split_stream_blob(sub_mask, t, ox0, oy0, r_exp):
    """Split a moving yellow region that holds several balls, knowing the size one ball has at that spot.

    A 30 balls/s stream is a line of touching balls a few pixels apart; a dump is a 2-D cloud. Distance-transform peaks
    separated by a visible neck give the balls that can be seen apart; the rest are filled in along the stream's main axis
    (a line) or by greedy peak picking (a cloud), up to the count the region's length and area imply."""
    ys, xs = np.nonzero(sub_mask)
    if len(xs) < 6:
        return None
    area = float(len(xs))
    dist = cv2.distanceTransform(sub_mask, cv2.DIST_L2, 5)
    dmax = float(dist.max())

    pts = np.column_stack([xs, ys]).astype(float)
    mean = pts.mean(axis=0)
    centered = pts - mean
    if len(pts) >= 3:
        _, eigvecs = np.linalg.eigh(np.cov(centered.T))
        major, minor = eigvecs[:, 1], eigvecs[:, 0]
        proj_maj, proj_min = centered @ major, centered @ minor
        length = float(proj_maj.max() - proj_maj.min() + 1.0)
        width_min = float(proj_min.max() - proj_min.min() + 1.0)
    else:
        major, proj_maj = np.array([1.0, 0.0]), centered[:, 0]
        length, width_min = float(sub_mask.shape[1]), float(sub_mask.shape[0])

    # A curved parabolic stream is 1 ball thick along its arc even when its chord sagitta widens width_min.
    mean_thick = area / max(1.0, length)
    is_line = (mean_thick <= 2.55 * r_exp and dmax <= 1.45 * r_exp) or width_min <= 1.50 * (2.0 * r_exp)
    r_eff = float(np.clip(dmax - 0.2, 0.88 * r_exp, 1.22 * r_exp)) if is_line else float(r_exp)
    single_area = math.pi * r_eff ** 2
    diam = 2.0 * r_eff
    if width_min < 0.58 * diam and dmax < 0.58 * r_eff:
        return None                    # thinner than a ball: an edge sliver (a pile under a pan, a robot edge), not a line of balls

    ksize = 5 if r_eff <= 4.8 else 7
    dil = cv2.dilate(dist, np.ones((ksize, ksize), np.uint8))
    peaks = ((dist >= max(2.0, 0.58 * r_eff)) & (dist >= dil - 1e-5)).astype(np.uint8)
    n_peaks, peak_labels, _, peak_cent = cv2.connectedComponentsWithStats(peaks)
    seeds = sorted(((float(dist[peak_labels == k].max()), float(peak_cent[k][0]), float(peak_cent[k][1])) for k in range(1, n_peaks)), reverse=True)
    min_sep = max(4.6, 1.18 * r_eff)
    kept = []
    for r, x, y in seeds:
        ok = True
        for kr, kx, ky in kept:
            sep = math.hypot(x - kx, y - ky)
            if sep < max(min_sep, 1.22 * max(r, kr)):
                ok = False
                break
            steps = max(2, int(round(sep)) + 1)
            line_x = np.clip(np.round(np.linspace(x, kx, steps)).astype(int), 0, dist.shape[1] - 1)
            line_y = np.clip(np.round(np.linspace(y, ky, steps)).astype(int), 0, dist.shape[0] - 1)
            if float(dist[line_y, line_x].min()) > 0.87 * min(r, kr) and sep < 1.85 * diam:       # no neck between them: one blurred ball
                ok = False
                break
        if ok:
            kept.append((r, x, y))

    if is_line:
        if len(kept) >= 2 and area / len(kept) <= 1.32 * single_area:
            n_est = len(kept)
        elif len(kept) >= 2 or (length >= 2.65 * diam and area >= 2.15 * single_area):
            n_est = max(len(kept), int(round(area / max(8.0, 0.92 * single_area))))
        else:
            n_est = 1
    else:
        # A cloud (a dump) is split only when at least two balls are seen apart (a neck between peaks); a round blob with no
        # neck is one ball, however big it looks.
        if len(kept) >= 2 and area / len(kept) <= 1.32 * single_area:
            n_est = len(kept)
        else:
            n_est = max(len(kept), int(round(area / max(8.0, 0.92 * single_area)))) if (len(kept) >= 2 and area > 1.55 * single_area) else 1

    if n_est <= 1:
        return None
    if len(kept) > n_est:
        kept = kept[:n_est]
    if len(kept) < n_est:
        if is_line:                                        # a line: place balls along the curved ridge
            p_min, p_max = float(proj_maj.min()), float(proj_maj.max())
            margin = min(r_eff * 0.82, 0.23 * (p_max - p_min))
            band = max(2.0, (p_max - p_min) / (2.0 * n_est))
            out = []
            for tau in np.linspace(p_min + margin, p_max - margin, n_est):
                in_slice = np.abs(proj_maj - tau) <= band
                if np.any(in_slice):
                    w_s = dist[ys[in_slice], xs[in_slice]] ** 2 + 0.25
                    cx, cy = float(np.average(xs[in_slice], weights=w_s)), float(np.average(ys[in_slice], weights=w_s))
                else:
                    cx, cy = (float(v) for v in mean + tau * major)
                out.append(Blob(t, cx + ox0, cy + oy0, min(single_area, area / n_est), 0.85))
            return out
        else:                                              # a cloud: pick the next-deepest point away from the ones found
            work = dist.copy()
            yy, xx = np.ogrid[:work.shape[0], :work.shape[1]]
            suppress = max(4.2, 1.15 * r_eff)
            for _, kx, ky in kept:
                work[(xx - kx) ** 2 + (yy - ky) ** 2 <= suppress ** 2] = 0.0
            while len(kept) < n_est:
                best = float(work.max())
                if best < 1.1:
                    break
                py, px = np.unravel_index(int(np.argmax(work)), work.shape)
                kept.append((max(2.5, best), float(px), float(py)))
                work[(xx - px) ** 2 + (yy - py) ** 2 <= suppress ** 2] = 0.0
    if len(kept) <= 1:
        return None
    return [Blob(t, x + ox0, y + oy0, min(single_area, math.pi * max(2.5, r) ** 2), 0.85) for r, x, y in kept]


class MovingFuelDetector:
    """Yellow AND moving. Piles on the carpet, the depots and the outposts are yellow too, but static
    relative to the field; a ball in flight (or being fired) is not. The previous frame is warped by
    the camera motion before differencing so a pan does not make the whole pile look like it moves.

    Position and size come from the ball's whole yellow region, not from the pixels that changed: for a
    slow ball the changed pixels are a thin crescent, which shifts the centre and understates the size
    (and the size is a range cue in the 3D fit). The change mask decides WHICH blobs are balls; the
    yellow region decides where they are and how big. If the yellow region is much larger than the
    changed part (the ball is touching a pile or another ball) the changed part is used instead."""

    def __init__(self, low=(18, 110, 110), high=(38, 255, 255), min_area=6, max_area=2500, motion_threshold=None):
        if motion_threshold is None:
            motion_threshold = globals()['motion_threshold']()
        self.split_streams = split_streams_mode()
        self.low, self.high = np.array(low), np.array(high)
        self.soft_low = np.array([max(15, int(low[0]) - 3), max(80, int(low[1]) - 30), max(94, int(low[2]) - 16)], np.uint8)
        self.soft_high = np.array([min(41, int(high[0]) + 3), int(high[1]), int(high[2])], np.uint8)
        self.min_area, self.max_area, self.motion_threshold = min_area, max_area, motion_threshold
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # Set by the fuel pass to the reference camera: with it, regions holding several balls (streams, dumps) are split by
        # the size one ball has at that spot; without it the older pixel-size rules apply.
        self.camera = None
        self.ref_camera = None
        # Observed / predicted size of single balls (round, isolated, moving blobs). Motion blur, compression and the
        # broadcast's scaling make a real ball look bigger than its true size at that depth; splitting with the raw camera
        # prediction would cut one blurred ball into two. The median of recent ratios rescales the prediction.
        self._size_ratios = []
        self._obs_radii = []
        self._median_cache = {}
        self._prev_yellow = None
        self._prev_shape = None
        # v1.0.4: inside `near_mask` (crop coordinates; the region around the HUB openings) a ball is also accepted by a
        # wider colour window ('wide') or by its yellow shift over the local running background ('background').
        self.opening_mode = None
        self.near_mask = None
        self._bg = None

    def _near_opening_yellow(self, bgr, hsv):
        """uint8 mask of extra ball pixels inside `near_mask` for the opening detection mode, or None."""
        near = self.near_mask
        if self.opening_mode is None or near is None or near.shape[:2] != bgr.shape[:2] or not near.any():
            return None
        x, y, w, h = cv2.boundingRect(near)
        region = near[y:y + h, x:x + w] > 0
        extra = np.zeros(near.shape, np.uint8)
        if self.opening_mode == 'wide':
            wide = cv2.inRange(hsv[y:y + h, x:x + w], np.array(WIDE_LOW), np.array(WIDE_HIGH)) > 0
            extra[y:y + h, x:x + w][wide & region] = 255
            return extra
        patch = bgr[y:y + h, x:x + w].astype(np.float32)
        if self._bg is None or self._bg.shape != patch.shape:
            self._bg = patch.copy()
            return extra
        delta = patch - self._bg
        db, dg, dr = delta[..., 0], delta[..., 1], delta[..., 2]
        shift = (dr > BG_MIN_RG) & (dg > BG_MIN_RG) & ((dr + dg) / 2 - db > BG_MIN_YELLOW_SHIFT) & region
        if RED_GLOW:
            # Over a red-lit HUB the background's red channel is already near saturation, so a yellow ball cannot
            # raise it; accept a strong green rise over blue there instead (2026arc and 2026caclv: red short 25-45%).
            saturated = (self._bg[..., 2] > RED_GLOW_BG_MIN) & (patch[..., 2] > RED_GLOW_BG_MIN - 20)
            shift |= saturated & (dg > BG_MIN_RG) & (dg - db > BG_MIN_YELLOW_SHIFT) & region
        extra[y:y + h, x:x + w][shift] = 255
        # Learn the background (including the HUB light pulses) everywhere except where a ball is now.
        cv2.accumulateWeighted(patch, self._bg, BG_ALPHA, mask=(~shift).astype(np.uint8))
        return extra

    def _single_ball_radius(self, u, v, r=False):
        """`r` is expected_ball_radius_px at (u, v) when the caller already has it."""
        cam = self.camera or self.ref_camera
        if r is False:
            r = expected_ball_radius_px(cam, u, v) if cam is not None else None
        if r is not None:
            if len(self._size_ratios) >= 6:
                scale = self._median('_size_ratios', 0.95, 2.4)
            else:
                scale = 1.18                                 # until enough single balls are seen: assume slight blur
            return r * math.sqrt(scale)
        if len(self._obs_radii) >= 8:
            return self._median('_obs_radii', 4.10, 5.4)
        return 4.25

    def _median(self, name, lo, hi):
        """Clipped median of a size list, recomputed only when the list (replaced on every append) changes."""
        values = getattr(self, name)
        cached = self._median_cache.get(name)
        if cached is None or cached[0] is not values:
            cached = (values, float(np.clip(np.median(values), lo, hi)))
            self._median_cache[name] = cached
        return cached[1]

    def yellow_mask(self, bgr):
        return cv2.inRange(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV), self.low, self.high)

    def detect(self, t, bgr, prev_gray_warped=None, offset=(0, 0), gray=None, yellow=None, hsv=None):
        """Blobs in the coordinates of the ORIGINAL frame: `bgr` may be a crop, with `offset` = (x, y) of its corner.

        A ball is a whole yellow REGION that contains moving pixels (a slow ball's changed pixels alone split
        into a leading and a trailing crescent, which would be two detections). Regions too big to be one
        ball or a touching few (the piles) are not balls; a ball against a pile is recovered from the changed
        pixels inside that region instead.
        """
        if hsv is None and yellow is None:
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        yellow = cv2.inRange(hsv, self.low, self.high) if yellow is None else yellow
        ox, oy = offset
        if prev_gray_warped is None:                       # first frame: nothing to compare with
            self._prev_yellow = yellow
            self._prev_shape = yellow.shape
            return []
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if gray is None else gray
        diff_val = cv2.absdiff(gray, prev_gray_warped)
        diff = (diff_val > self.motion_threshold).astype(np.uint8) * 255
        if self.opening_mode is not None:
            if hsv is None:
                hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            extra = self._near_opening_yellow(bgr, hsv)
            if extra is not None and extra.any():
                yellow = cv2.bitwise_or(yellow, extra)
                if self.opening_mode == 'background':
                    diff = cv2.bitwise_or(diff, extra)       # a shift over the background is itself evidence of motion
        yh, yw = yellow.shape[:2]
        air_limit_y = int(0.62 * yh) if yh >= 200 else yh

        # Recover motion-blurred balls (whose saturation dips at fast shutter blur) and balls crossing bright/yellow-shirted
        # crowd backdrops behind the hubs where luminance diff is smaller (12..18) but chroma is newly yellow.
        if hsv is None:
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        soft_yellow = cv2.inRange(hsv[:air_limit_y], self.soft_low, self.soft_high)
        blur_motion = ((soft_yellow > 0) & (diff_val[:air_limit_y] > max(21, self.motion_threshold + 4))).astype(np.uint8) * 255
        if self._prev_yellow is not None and self._prev_shape == yellow.shape:
            prev_y_dil = cv2.dilate(self._prev_yellow, self.kernel, iterations=1)
            blur_motion = cv2.bitwise_and(blur_motion, 255 - prev_y_dil[:air_limit_y])
            newly_yellow = (yellow[:air_limit_y] > 0) & (prev_y_dil[:air_limit_y] == 0) & (hsv[:air_limit_y, :, 1] >= 140) & (hsv[:air_limit_y, :, 2] >= 135) & (diff_val[:air_limit_y] >= max(12, self.motion_threshold - 5))
            if np.any(newly_yellow):
                diff[:air_limit_y] = cv2.bitwise_or(diff[:air_limit_y], newly_yellow.astype(np.uint8) * 255)
        self._prev_yellow = yellow.copy()
        self._prev_shape = yellow.shape
        if blur_motion.any():
            yellow = yellow.copy()
            yellow[:air_limit_y] = cv2.bitwise_or(yellow[:air_limit_y], blur_motion)

        moving = cv2.dilate(diff, self.kernel, iterations=1)
        moving_yellow = cv2.bitwise_and(yellow, moving)
        if not moving_yellow.any():
            return []
        cam = self.camera or self.ref_camera
        # Closing first keeps one ball one region (its yellow can break into fragments); touching balls are split afterwards.
        regions = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, self.kernel)
        n, labels, stats, cent = cv2.connectedComponentsWithStats(regions)
        moved = np.bincount(labels[moving_yellow > 0], minlength=n)
        raw_moved = np.bincount(labels[(yellow > 0) & (diff > 0)], minlength=n)
        out = []
        too_big = []

        def add(area, w, h, cx, cy, sub_labels, k, x0, y0):
            r_raw = expected_ball_radius_px(cam, cx, cy) if cam is not None else None
            r_exp = self._single_ball_radius(cx, cy, r_raw)
            circ0 = area / max(1.0, math.pi * (max(w, h) / 2.0) ** 2)
            if sub_labels is labels and circ0 >= 0.72 and max(w, h) <= 15:
                if cam is not None:
                    if r_raw and 0.65 * math.pi * r_raw ** 2 <= area <= 2.2 * math.pi * r_raw ** 2:
                        self._size_ratios = (self._size_ratios + [area / (math.pi * r_raw ** 2)])[-200:]
                elif 25.0 <= area <= 105.0:
                    self._obs_radii = (self._obs_radii + [math.sqrt(area / math.pi)])[-200:]
            split_area = (STREAM_SPLIT_AREA_EXP if self.split_streams else STREAM_SPLIT_AREA) if cam is not None else 2.25
            if (sub_labels is labels or self.split_streams) and r_exp is not None and area > split_area * math.pi * r_exp ** 2 and max(w, h) >= 2.45 * r_exp:
                split = _split_stream_blob((sub_labels[y0:y0 + h, x0:x0 + w] == k).astype(np.uint8) * 255, t, ox + x0, oy + y0, r_exp)
                if split:
                    out.extend(split)
                    return
            if area > 85 and max(w, h) >= 12:
                split = _split_touching_blobs((sub_labels[y0:y0 + h, x0:x0 + w] == k).astype(np.uint8) * 255, t, ox + x0, oy + y0)
                if split:
                    out.extend(split)
                    return
            circ = area / max(1.0, math.pi * (max(w, h) / 2.0) ** 2)
            out.append(Blob(t, cx, cy, area, float(min(1.0, circ))))

        for i in range(1, n):
            if moved[i] < 3 or raw_moved[i] < 1:
                continue
            area = float(stats[i, cv2.CC_STAT_AREA])
            w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            cx, cy = float(cent[i][0]) + ox, float(cent[i][1]) + oy
            # A dense stream or volley in flight has most of its yellow pixels moving; keep its intact yellow silhouette so
            # overlapping balls in a 20-30 balls/s stream do not get hollowed out by frame differencing.
            is_moving_stream = area <= min(self.max_area, 850) and moved[i] >= 0.58 * area and min(w, h) <= 30
            pile = (not is_moving_stream and area > min(self.max_area, 220)) or moved[i] < max(3, 0.05 * area) or (area > 65 and max(w, h) > 16 and moved[i] < 0.45 * area)
            if pile:
                too_big.append(i)                          # a pile, or a ball touching one: only part of it moves
                continue
            if area < self.min_area:
                continue
            if yh >= 200 and int(cent[i][1]) >= air_limit_y and area >= 16 and moved[i] < 0.36 * area and raw_moved[i] < 0.16 * area:
                jx0, jy0 = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
                if int(diff_val[jy0:jy0 + h, jx0:jx0 + w].max()) < self.motion_threshold + 18:
                    continue
            add(area, w, h, cx, cy, labels, i, int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]))
        if too_big:
            # a ball touching a pile or a volley: use only the pixels that changed inside the big regions
            inside = np.isin(labels, too_big).astype(np.uint8) * 255
            raw_my = cv2.bitwise_and(yellow, diff)
            mask = cv2.morphologyEx(cv2.bitwise_and(raw_my, inside), cv2.MORPH_CLOSE, self.kernel)
            m, mlabels, mstats, mcent = cv2.connectedComponentsWithStats(mask)
            for j in range(1, m):
                area = float(mstats[j, cv2.CC_STAT_AREA])
                if not self.min_area <= area <= self.max_area:
                    continue
                w, h = mstats[j, cv2.CC_STAT_WIDTH], mstats[j, cv2.CC_STAT_HEIGHT]
                mx, my = float(mcent[j][0]), float(mcent[j][1])
                imy, imx = min(yh - 1, max(0, int(round(my)))), min(yw - 1, max(0, int(round(mx))))
                p_lbl = int(labels[imy, imx])
                if p_lbl > 0:
                    p_area = float(stats[p_lbl, cv2.CC_STAT_AREA])
                    p_mfrac = float(moved[p_lbl]) / max(1.0, p_area)
                    # Reject weak internal compression/jostling specks buried inside large mostly-static carpet piles and hoppers
                    if p_area > 160 and p_mfrac < 0.40:
                        jx0, jy0 = int(mstats[j, cv2.CC_STAT_LEFT]), int(mstats[j, cv2.CC_STAT_TOP])
                        max_d = int(diff_val[jy0:jy0 + h, jx0:jx0 + w].max())
                        if area < max(self.min_area + 5, 13.0) and max_d < self.motion_threshold + 14:
                            continue
                        if yh >= 200 and imy >= air_limit_y and p_area > 300 and p_mfrac < 0.30 and area < 28.0 and max_d < self.motion_threshold + 24:
                            continue
                add(area, w, h, mx + ox, my + oy, mlabels, j, int(mstats[j, cv2.CC_STAT_LEFT]), int(mstats[j, cv2.CC_STAT_TOP]))
        return out


@dataclass
class BallTrack:
    obs: list = field(default_factory=list)       # (frame_index, t, u, v, area, ru, rv)
    missed: int = 0
    source_breaks: list | None = field(default_factory=list)  # joined fragment intervals, in video seconds

    def predict(self, t):
        if len(self.obs) >= 2:
            pu, pv, _, _ = _predict_motion(self.obs, t)
            return pu, pv
        return self.obs[-1][5], self.obs[-1][6]


def continuous_source_to(track, through_s):
    """An upstream join cannot prove a continuous observed source-to-entry path. Legacy provenance is unknown."""
    breaks = getattr(track, 'source_breaks', None)
    if not isinstance(breaks, list) or not getattr(track, 'obs', None):
        return False
    start = float(track.obs[0][1])
    if not math.isfinite(through_s) or through_s < start:
        return False
    for interval in breaks:
        if (not isinstance(interval, (list, tuple)) or len(interval) != 2
                or not all(isinstance(t, (int, float)) and math.isfinite(t) for t in interval)
                or interval[1] <= interval[0]):
            return False
        if interval[0] < through_s and interval[1] > start:
            return False
    return True


class BallTracker:
    """Ball tracking in reference-frame pixels with gravity-aware prediction and optimal assignment."""

    def __init__(self, max_missed=3, base_gate_px=18.0, min_obs=6):
        self.max_missed, self.base_gate, self.min_obs = max_missed, base_gate_px, min_obs
        self.active, self.done = [], []

    def step(self, index, t, blobs):
        def new_track(b):
            return BallTrack([(index, t, b.u, b.v, b.area, b.ru, b.rv)])
        if not self.active or not blobs:
            survivors = []
            for track in self.active:
                track.missed += 1
                (survivors if track.missed <= self.max_missed else self.done).append(track)
            self.active = survivors + [new_track(b) for b in blobs]
            return
        n_t, n_b = len(self.active), len(blobs)
        bu = np.array([b.ru for b in blobs], float)
        bv = np.array([b.rv for b in blobs], float)
        samples = []
        for tr in self.active:
            if len(tr.obs) >= 3:
                o0, o1, o2 = tr.obs[-3], tr.obs[-2], tr.obs[-1]
                dt1, dt2 = max(o1[1] - o0[1], 1e-3), max(o2[1] - o1[1], 1e-3)
                vu1, vv1 = (o1[5] - o0[5]) / dt1, (o1[6] - o0[6]) / dt1
                vu2, vv2 = (o2[5] - o1[5]) / dt2, (o2[6] - o1[6]) / dt2
                s1, s2 = math.hypot(vu1, vv1), math.hypot(vu2, vv2)
                if s1 >= 35.0 and s2 >= 35.0 and (vu1 * vu2 + vv1 * vv2) >= 0.35 * s1 * s2:
                    samples.append((o2[5], o2[6], vu2, vv2))
            elif len(tr.obs) == 2:
                o1, o2 = tr.obs[0], tr.obs[1]
                dt_k = max(o2[1] - o1[1], 1e-3)
                vuk, vvk = (o2[5] - o1[5]) / dt_k, (o2[6] - o1[6]) / dt_k
                if vvk < -30.0 and math.hypot(vuk, vvk) >= 60.0:
                    samples.append((o1[5], o1[6], vuk, vvk))
        samples = np.array(samples, float).reshape(-1, 4)
        big = 1e6
        cost = np.full((n_t, n_b + n_t), big)
        for i, track in enumerate(self.active):
            n_obs = len(track.obs)
            u_cost = 28.0 if n_obs >= 3 else (22.0 if n_obs == 2 else 16.5)
            cost[i, n_b + i] = u_cost
            t_last, ru_last, rv_last = track.obs[-1][1], track.obs[-1][5], track.obs[-1][6]
            dt = max(t - t_last, 1e-3)
            if dt > 0.25:
                continue
            has_vel = True
            if n_obs >= 2:
                pu, pv, vu, vv = _predict_motion(track.obs, t)
            elif len(samples):
                d_s = np.hypot(samples[:, 0] - ru_last, samples[:, 1] - rv_last)
                k = int(np.argmin(d_s))
                if d_s[k] < 42.0:
                    vu, vv = samples[k, 2], samples[k, 3]
                    pu, pv = ru_last + vu * dt, rv_last + vv * dt
                else:
                    has_vel, vu, vv, pu, pv = False, 0.0, 0.0, ru_last, rv_last
            else:
                has_vel, vu, vv, pu, pv = False, 0.0, 0.0, ru_last, rv_last
            speed = math.hypot(vu, vv)
            exp_step = speed * dt
            gate = self.base_gate + 0.35 * exp_step
            if n_obs >= 3 and speed >= 40.0:
                gate = min(gate, max(10.0, 0.90 * exp_step + 4.0 * track.missed))
            d = np.hypot(bu - pu, bv - pv)
            ok = d <= gate
            du, dv = bu - ru_last, bv - rv_last
            if has_vel and speed >= 40.0:
                fwd = (du * vu + dv * vv) / speed
                lat = np.abs(-du * vv + dv * vu) / speed
                if exp_step >= 4.0:
                    ok &= (fwd >= 0.20 * exp_step - 2.0) & (fwd <= max(exp_step + 9.0, 1.75 * exp_step))
                c_row = d + 0.25 * lat + 0.30 * np.maximum(0.0, exp_step - fwd) + 2.5 * track.missed
            else:
                c_row = 0.75 * d + np.where(dv > 2.0, 4.5, 0.0) + 3.5 + 2.5 * track.missed
            cost[i, :n_b] = np.where(ok, np.minimum(u_cost - 0.2, c_row), big)
        used_t, used_b = set(), set()
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            if c >= n_b or cost[r, c] >= big:
                continue
            used_t.add(int(r))
            used_b.add(int(c))
            b = blobs[c]
            self.active[r].obs.append((index, t, b.u, b.v, b.area, b.ru, b.rv))
            self.active[r].missed = 0
        survivors = []
        for i, track in enumerate(self.active):
            if i not in used_t:
                track.missed += 1
            (survivors if track.missed <= self.max_missed else self.done).append(track)
        self.active = survivors + [new_track(b) for j, b in enumerate(blobs) if j not in used_b]

    def finish(self):
        tracks = self.done + self.active
        self.active, self.done = [], []
        return [tr for tr in tracks if len(tr.obs) >= self.min_obs]



def _predict_motion(obs, t):
    """(pu, pv, vu, vv) in reference pixels: velocity from the last few observations, with the measured vertical acceleration
    (gravity in the picture, capped) so a ball near its apex is still predicted where it is going."""
    n_obs = len(obs)
    t_last, ru_last, rv_last = obs[-1][1], obs[-1][5], obs[-1][6]
    dt = max(t - t_last, 1e-3)
    if n_obs >= 3:
        k = min(n_obs, 7)
        tail = obs[-k:]
        t0, ru0, rv0 = tail[0][1], tail[0][5], tail[0][6]
        dt_total = max(t_last - t0, 1e-3)
        mid = k // 2
        o_mid = tail[mid]
        dt1 = max(o_mid[1] - t0, 1e-3)
        dt2 = max(t_last - o_mid[1], 1e-3)
        vv1 = (o_mid[6] - rv0) / dt1
        vv2 = (rv_last - o_mid[6]) / dt2
        av_raw = (vv2 - vv1) / (0.5 * dt_total)
        vu_span = (ru_last - ru0) / dt_total
        vv_mean = (rv_last - rv0) / dt_total
        g_prior = 420.0 if math.hypot(vu_span, vv_mean) >= 30.0 else 0.0
        w = min(1.0, (dt_total / 0.12) ** 2)
        av = max(0.0, min(1400.0, w * av_raw + (1.0 - w) * g_prior))
        taus = [o[1] - t_last for o in tail]
        mean_tau = sum(taus) / k
        denom = sum((tau - mean_tau) ** 2 for tau in taus)
        if denom > 1e-9:
            us = [o[5] for o in tail]
            ys = [o[6] - 0.5 * av * tau * tau for o, tau in zip(tail, taus)]
            mean_u = sum(us) / k
            mean_y = sum(ys) / k
            vu = sum((tau - mean_tau) * (u - mean_u) for tau, u in zip(taus, us)) / denom
            vv_last = sum((tau - mean_tau) * (y - mean_y) for tau, y in zip(taus, ys)) / denom
            u0 = mean_u - vu * mean_tau
            v0 = mean_y - vv_last * mean_tau
            alpha = 0.55 if k >= 4 and dt <= 0.08 else 0.8
            ru_anc = alpha * ru_last + (1.0 - alpha) * u0
            rv_anc = alpha * rv_last + (1.0 - alpha) * v0
        else:
            vu = vu_span
            vv_last = vv_mean + 0.5 * av * dt_total
            ru_anc, rv_anc = ru_last, rv_last
        pu = ru_anc + vu * dt
        pv = rv_anc + vv_last * dt + 0.5 * av * dt * dt
        return pu, pv, vu, vv_last + 0.5 * av * dt
    if n_obs == 2:
        dt_prev = max(t_last - obs[-2][1], 1e-3)
        vu, vv = (ru_last - obs[-2][5]) / dt_prev, (rv_last - obs[-2][6]) / dt_prev
        av = 350.0 if math.hypot(vu, vv) >= 30.0 else 0.0
        return ru_last + vu * dt, rv_last + vv * dt + 0.5 * av * dt * dt, vu, vv + 0.5 * av * dt
    return ru_last, rv_last, 0.0, 0.0


class StreamBallTracker:
    """Frame-to-frame ball tracking by optimal assignment (high-rate mode, see high_rate_mode()).

    Each live track predicts where its ball is now (velocity and picture-space gravity from its own history; a track one
    frame old borrows the velocity of the nearest moving track, i.e. the stream it is part of). Every track-to-blob pair
    that passes the gates gets a cost, and one optimal assignment (scipy linear_sum_assignment) links them all at once.
    Greedy nearest-first linking let a ball in a 30 balls/s stream take its neighbour's detection; the forward-progress
    and lateral gates stop a track from stepping backwards or sideways onto the ball behind or beside it."""

    def __init__(self, max_missed=3, base_gate_px=18.0, min_obs=6):
        self.max_missed, self.base_gate, self.min_obs = max_missed, base_gate_px, min_obs
        self.active, self.done = [], []

    def step(self, index, t, blobs):
        def new_track(b):
            return BallTrack([(index, t, b.u, b.v, b.area, b.ru, b.rv)])
        if not self.active or not blobs:
            survivors = []
            for track in self.active:
                track.missed += 1
                (survivors if track.missed <= self.max_missed else self.done).append(track)
            self.active = survivors + [new_track(b) for b in blobs]
            return
        n_t, n_b = len(self.active), len(blobs)
        bu = np.array([b.ru for b in blobs], float)
        bv = np.array([b.rv for b in blobs], float)
        # velocity samples of moving tracks, lent to tracks that are one frame old
        samples = []
        for tr in self.active:
            for k in range(max(0, len(tr.obs) - 5), len(tr.obs) - 1):
                o1, o2 = tr.obs[k], tr.obs[k + 1]
                dt_k = max(o2[1] - o1[1], 1e-3)
                vu_k, vv_k = (o2[5] - o1[5]) / dt_k, (o2[6] - o1[6]) / dt_k
                if math.hypot(vu_k, vv_k) >= 35.0:
                    if len(tr.obs) == 2 and vv_k > 30.0:
                        continue
                    samples.append((o1[5], o1[6], vu_k, vv_k))
        samples = np.array(samples, float).reshape(-1, 4)
        big = 1e6
        cost = np.full((n_t, n_b + n_t), big)
        for i, track in enumerate(self.active):
            n_obs = len(track.obs)
            u_cost = 30.0 if n_obs >= 4 else (24.0 if n_obs == 3 else (20.0 if n_obs == 2 else 17.5))
            cost[i, n_b + i] = u_cost
            t_last, ru_last, rv_last = track.obs[-1][1], track.obs[-1][5], track.obs[-1][6]
            dt = max(t - t_last, 1e-3)
            if dt > 0.25:
                continue
            has_vel = True
            if n_obs >= 2:
                pu, pv, vu, vv = _predict_motion(track.obs, t)
            elif len(samples):
                d_s = np.hypot(samples[:, 0] - ru_last, samples[:, 1] - rv_last)
                k = int(np.argmin(d_s))
                if d_s[k] < 45.0:
                    vu, vv = samples[k, 2], samples[k, 3]
                    pu, pv = ru_last + vu * dt, rv_last + vv * dt
                else:
                    has_vel, vu, vv, pu, pv = False, 0.0, 0.0, ru_last, rv_last
            else:
                has_vel, vu, vv, pu, pv = False, 0.0, 0.0, ru_last, rv_last
            speed = math.hypot(vu, vv)
            exp_step = speed * dt
            gate = (self.base_gate + 0.25 * exp_step if has_vel else self.base_gate + 10.0) + 4.0 * track.missed
            du, dv = bu - ru_last, bv - rv_last
            step_dist = np.hypot(du, dv)
            if has_vel and speed >= 45.0:
                d_pred = np.hypot(bu - pu, bv - pv)
                fwd = (du * vu + dv * vv) / speed
                lat = np.abs(-du * vv + dv * vu) / speed
                ok = d_pred <= gate
                if exp_step >= 3.0:
                    ok &= (fwd >= 0.30 * exp_step) & (fwd <= max(exp_step + 8.0, 1.65 * exp_step))
                if exp_step >= 4.0:
                    ok &= lat <= max(6.0, 0.60 * exp_step)
                c_row = np.minimum(u_cost - 0.5, d_pred + 0.40 * lat + 0.45 * np.maximum(0.0, exp_step - fwd) + 3.5 * track.missed)
                cost[i, :n_b] = np.where(ok, c_row, big)
            else:
                step_speed = step_dist / dt
                ok = (step_dist <= gate) & (step_speed >= 35.0) & (step_speed <= 650.0)
                c_row = np.minimum(u_cost - 0.5, 0.45 * step_dist + np.where(dv > 2.0, 8.0, 0.0) + 3.0)
                cost[i, :n_b] = np.where(ok, c_row, big)
        used_t, used_b = set(), set()
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            if c >= n_b or cost[r, c] >= big:
                continue
            used_t.add(int(r))
            used_b.add(int(c))
            b = blobs[c]
            self.active[r].obs.append((index, t, b.u, b.v, b.area, b.ru, b.rv))
            self.active[r].missed = 0
        survivors = []
        for i, track in enumerate(self.active):
            if i not in used_t:
                track.missed += 1
            (survivors if track.missed <= self.max_missed else self.done).append(track)
        self.active = survivors + [new_track(b) for j, b in enumerate(blobs) if j not in used_b]

    def finish(self):
        tracks = self.done + self.active
        self.active, self.done = [], []
        return [tr for tr in tracks if len(tr.obs) >= self.min_obs]


def link_tracks(tracks, max_gap_s=0.5, base_tolerance_px=8.0):
    """Join track fragments that are the same ball.

    A ball hidden for a few frames (a robot, the hub) ends one track and starts another. Left alone it is
    counted twice, and each half is too short to fit well. A fragment B continues A when B's first sighting
    is where A's own motion, extrapolated, says the ball should be, and the combined observations fit a single
    2D ballistic arc with downward gravity. Each fragment is used at most once.
    """
    import bisect

    def fit_parabola_2d(obs):
        n = len(obs)
        if n < 3:
            return None
        t0 = float(obs[0][1])
        span_t = float(obs[-1][1]) - t0
        if span_t < 0.025:
            return None
        dts = [float(o[1]) - t0 for o in obs]
        rus = [float(o[5]) for o in obs]
        rvs = [float(o[6]) for o in obs]
        m_t = sum(dts) / n
        dt_c = [x - m_t for x in dts]
        s_tt = sum(x * x for x in dt_c)
        if s_tt < 1e-9:
            return None
        m_u = sum(rus) / n
        vx = sum(x * (u - m_u) for x, u in zip(dt_c, rus)) / s_tt
        u0 = m_u - vx * m_t
        q_raw = [0.5 * x * x for x in dts]
        m_q = sum(q_raw) / n
        q_c = [q - m_q for q in q_raw]
        s_tq = sum(x * q for x, q in zip(dt_c, q_c))
        s_qq = sum(q * q for q in q_c)
        m_v = sum(rvs) / n
        s_tv = sum(x * (v - m_v) for x, v in zip(dt_c, rvs))
        s_qv = sum(q * (v - m_v) for q, v in zip(q_c, rvs))
        det = s_tt * s_qq - s_tq * s_tq
        if det > 1e-12:
            vy = (s_qq * s_tv - s_tq * s_qv) / det
            g_v = (s_tt * s_qv - s_tq * s_tv) / det
            v0 = m_v - vy * m_t - g_v * m_q
        else:
            vy = s_tv / s_tt
            g_v = 0.0
            v0 = m_v - vy * m_t
        err_sq = 0.0
        for dt_i, q_i, ru_i, rv_i in zip(dts, q_raw, rus, rvs):
            du_i = ru_i - (u0 + vx * dt_i)
            dv_i = rv_i - (v0 + vy * dt_i + g_v * q_i)
            err_sq += du_i * du_i + dv_i * dv_i
        return {'t0': t0, 'u0': u0, 'vx': vx, 'v0': v0, 'vy': vy, 'g_v': g_v, 'rms_px': math.sqrt(err_sq / n)}

    valid = [tr for tr in tracks if tr and getattr(tr, 'obs', None)]
    if len(valid) <= 1:
        return valid
    tracks = sorted(valid, key=lambda tr: tr.obs[0][1])
    t0s = [float(tr.obs[0][1]) for tr in tracks]
    n_tracks = len(tracks)
    used = [False] * n_tracks
    merged = []
    for a_index, a in enumerate(tracks):
        if used[a_index]:
            continue
        used[a_index] = True
        chain_obs = list(a.obs)
        chain_breaks = getattr(a, 'source_breaks', None)
        chain_breaks = list(chain_breaks) if isinstance(chain_breaks, list) else None
        while len(chain_obs) >= 3:
            t_end = float(chain_obs[-1][1])
            start_b = bisect.bisect_right(t0s, t_end, lo=a_index + 1)
            if start_b >= n_tracks or (t0s[start_b] - t_end) > max_gap_s:
                break
            tail = chain_obs[-12:]
            fit_a = fit_parabola_2d(tail)
            if fit_a is None or fit_a['rms_px'] > 4.2:
                break
            ru_end, rv_end = float(chain_obs[-1][5]), float(chain_obs[-1][6])
            dt_end = t_end - fit_a['t0']
            vx_a = fit_a['vx']
            vy_a = fit_a['vy'] + fit_a['g_v'] * dt_end
            speed_px = math.hypot(vx_a, vy_a)
            best = None
            for b_index in range(start_b, n_tracks):
                if used[b_index]:
                    continue
                b_obs = tracks[b_index].obs
                gap = float(b_obs[0][1]) - t_end
                if gap <= 0:
                    continue
                if gap > max_gap_s:
                    break
                b_ru0, b_rv0 = float(b_obs[0][5]), float(b_obs[0][6])
                max_reach = (speed_px + 350.0 * gap) * gap + base_tolerance_px + 25.0
                if abs(b_ru0 - ru_end) > max_reach or abs(b_rv0 - rv_end) > max_reach:
                    continue
                dt_pred = float(b_obs[0][1]) - fit_a['t0']
                pred_u = fit_a['u0'] + fit_a['vx'] * dt_pred
                pred_v = fit_a['v0'] + fit_a['vy'] * dt_pred + 0.5 * fit_a['g_v'] * dt_pred * dt_pred
                err_start = math.hypot(b_ru0 - pred_u, b_rv0 - pred_v)
                tol = base_tolerance_px + 0.18 * gap * speed_px
                if err_start > max(22.0, tol):
                    continue
                if len(b_obs) >= 2 and abs(vx_a) > 25.0:
                    b_head = b_obs[min(len(b_obs) - 1, 5)]
                    dt_bh = max(float(b_head[1]) - float(b_obs[0][1]), 1e-3)
                    vx_b = (float(b_head[5]) - b_ru0) / dt_bh
                    if abs(vx_b) > 25.0 and vx_a * vx_b < 0:
                        continue
                comb = chain_obs[-15:] + b_obs[:15]
                fit_ab = fit_parabola_2d(comb)
                if fit_ab is None or fit_ab['rms_px'] > 3.2:
                    continue
                if len(comb) >= 6 and (fit_ab['g_v'] < -20.0 or fit_ab['g_v'] > 1100.0):
                    continue
                score = fit_ab['rms_px'] + 0.08 * err_start + 6.0 * gap if err_start <= tol else (fit_ab['rms_px'] + 0.15 * err_start + 6.0 * gap)
                if best is None or score < best[0]:
                    best = (score, b_index)
            if best is None:
                break
            used[best[1]] = True
            next_track = tracks[best[1]]
            next_breaks = getattr(next_track, 'source_breaks', None)
            chain_breaks = (chain_breaks + [[t_end, float(next_track.obs[0][1])]] + next_breaks
                            if chain_breaks is not None and isinstance(next_breaks, list) else None)
            chain_obs = chain_obs + list(tracks[best[1]].obs)
        merged.append(BallTrack(chain_obs, source_breaks=chain_breaks))
    return merged


GRAVITY_SHARE = 0.3        # a real flight shows at least this fraction of gravity's pull as vertical image acceleration


def quick_ballistic_check(track, vertical_px_per_m, min_obs=8, min_duration=0.25):
    """Cheap image-space test, run before the expensive 3D fit. Returns (passes, reason).

    On the real broadcast clip almost every track the detector made was a ball rolling or bouncing on the
    floor or a speck of noise (about 3,000 candidates for 200 shots), and fitting a non-flight costs
    about a second. A flight is, in the picture, smooth (a low-order polynomial in time) and accelerates
    downward by roughly gravity's share of the image scale; a roller does not accelerate, and noise is not
    smooth. Done in reference-frame pixels, so a camera pan does not matter.
    """
    n = len(track.obs)
    if n < min_obs:
        return False, 'too few sightings'
    t = np.array([o[1] for o in track.obs])
    duration = t[-1] - t[0]
    if duration < min_duration:
        return False, 'too short'
    t0 = t - t[0]
    u = np.array([o[5] for o in track.obs])
    v = np.array([o[6] for o in track.obs])
    # Smoothness: a flight is NOT a parabola in the picture (perspective and drag bend it; a quadratic left
    # 3.5 to 5 px of residual on good shots) but a 4th-order polynomial fits one to the noise floor
    # (0.5 px), while random-walk junk keeps 9 px or more. That 15x margin is why this can be strict.
    tolerance = 2.0 + 0.003 * math.hypot(u[-1] - u[0], v[-1] - v[0])
    for series in (u, v):
        residual = series - np.polyval(np.polyfit(t0, series, 4), t0)
        if float(np.sqrt(np.mean(residual ** 2))) > tolerance:
            return False, 'not smooth'
    # Gravity: the quadratic term is the acceleration. A roller has almost none.
    if 2 * np.polyfit(t0, v, 2)[0] < GRAVITY_SHARE * 9.81 * vertical_px_per_m:
        return False, 'does not accelerate downward like a thrown ball'
    return True, ''


def _fit_task(args):
    times, uv, cameras, radii = args
    return fit_flight(times, uv, cameras, radii_px=radii)


def _process_pool(workers):
    """A process pool with the start method in VISION_FUEL_MP_START ('spawn', 'forkserver' or 'fork'), else the
    platform default. CI sets 'spawn': on Linux the default 'fork' can deadlock when the parent already runs threads
    (OpenCV's, a test runner's), which hung the parallel fuel tests."""
    import multiprocessing
    method = os.environ.get('VISION_FUEL_MP_START', '').strip().lower()
    if method in multiprocessing.get_all_start_methods():
        return ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context(method))
    return ProcessPoolExecutor(workers)


def _env_fuel_workers():
    raw = os.environ.get('VISION_FUEL_WORKERS')
    if not raw:
        return None
    try:
        return max(1, int(raw))
    except ValueError:
        return None


def fit_tracks(tasks, workers=None):
    """Fit many tracks, across processes when there are enough to be worth it. Order is preserved.

    If the worker pool cannot run at all (a script that imports this without a __main__ guard makes spawned
    workers re-run it, a sandbox forbids processes, a worker is killed), the fits are done serially instead of
    failing the run: slower, never wrong.
    """
    if workers is None:
        workers = _env_fuel_workers()
    workers = workers if workers is not None else min(8, os.cpu_count() or 1)
    if workers <= 1 or len(tasks) < 6:
        return [_fit_task(t) for t in tasks]
    try:
        with _process_pool(workers) as pool:
            return list(pool.map(_fit_task, tasks, chunksize=max(1, len(tasks) // (workers * 4))))
    except (BrokenProcessPool, OSError, RuntimeError):
        return [_fit_task(t) for t in tasks]


def plausible_flight(track, min_travel_px=40.0):
    """A ball in flight travels across the picture; a speck of noise or a held ball does not."""
    first, last = track.obs[0], track.obs[-1]
    # Measured in the reference frame, so a static pile does not "travel" just because the camera panned.
    return math.hypot(last[5] - first[5], last[6] - first[6]) >= min_travel_px


def _interpolate_robot(track, t_ms):
    pts = track['trajectory']
    if not pts or t_ms < pts[0]['t'] - 500 or t_ms > pts[-1]['t'] + 500:
        return None
    before = max((p for p in pts if p['t'] <= t_ms), key=lambda p: p['t'], default=None)
    after = min((p for p in pts if p['t'] >= t_ms), key=lambda p: p['t'], default=None)
    if before and after and after['t'] != before['t']:
        f = (t_ms - before['t']) / (after['t'] - before['t'])
        return before['x'] + (after['x'] - before['x']) * f, before['y'] + (after['y'] - before['y']) * f
    near = before or after
    return near['x'], near['y']


def _interpolate_robot_box(track, t_ms):
    pts = [p for p in (track.get('trajectory') or []) if p.get('box') is not None and len(p['box']) == 4]
    if not pts or t_ms < pts[0]['t'] - 500 or t_ms > pts[-1]['t'] + 500:
        return None
    before = max((p for p in pts if p['t'] <= t_ms), key=lambda p: p['t'], default=None)
    after = min((p for p in pts if p['t'] >= t_ms), key=lambda p: p['t'], default=None)
    if before and after and after['t'] != before['t']:
        f = (t_ms - before['t']) / (after['t'] - before['t'])
        return [before['box'][i] + (after['box'][i] - before['box'][i]) * f for i in range(4)]
    near = before or after
    return [float(v) for v in near['box']]


def _launch_uv_from_shot(shot):
    uv = shot.get('launch_uv')
    if uv is not None and len(uv) >= 2:
        return float(uv[0]), float(uv[1])
    bt = shot.get('ball_track')
    if bt and isinstance(bt[0], dict) and bt[0].get('box') and len(bt[0]['box']) == 4:
        b = bt[0]['box']
        return 0.5 * (float(b[0]) + float(b[2])), 0.5 * (float(b[1]) + float(b[3]))
    return None


def attribute_shooter(launch_xy, launch_ms, robot_tracks, max_distance=1.6):
    """The robot nearest the launch point at launch time, or (None, None) when nobody is plausibly there."""
    best = None
    for track in robot_tracks:
        pos = _interpolate_robot(track, launch_ms)
        if pos is None:
            continue
        d = math.hypot(pos[0] - launch_xy[0], pos[1] - launch_xy[1])
        if best is None or d < best[0]:
            best = (d, track)
    if best is None or best[0] > max_distance:
        return None, None
    return best[1], best[0]


CONFIDENT_RANGE_M = 1.0      # a shooter this close, clearly nearer than anyone else, is a confident call
AMBIGUOUS_MARGIN_M = 0.3     # two robots this close to equally near cannot be told apart
CLEAR_MARGIN_M = 0.6
BURST_BONUS_M = 0.3          # a robot that fired a moment ago is treated as this much nearer
BURST_WINDOW_MS = 700
IMAGE_BOX_EXPAND = 0.15      # expand each robot's 0-1 normalized box by 15% when matching a ball's first pixel
IMAGE_CONFIDENCE_SCALE = 0.65


def _attribute_by_image_box(launch_uv, launch_ms, robot_tracks, shooter_alliance=None, expand=IMAGE_BOX_EXPAND):
    """Fallback when 3D field distance fails: match the ball's first normalized pixel `launch_uv` against each
    robot's normalized bounding box (expanded by `expand` = 15% on each side) at `launch_ms`, choosing the
    nearest box centre. Never links across alliances when `shooter_alliance` is known."""
    if launch_uv is None or len(launch_uv) < 2:
        return None
    u, v = float(launch_uv[0]), float(launch_uv[1])
    candidates = []
    for track in robot_tracks:
        if shooter_alliance and track.get('alliance') and track.get('alliance') != shooter_alliance:
            continue
        box = _interpolate_robot_box(track, launch_ms)
        if box is None:
            continue
        x0, y0, x1, y1 = box
        w, h = max(0.0, x1 - x0), max(0.0, y1 - y0)
        pad_x, pad_y = expand * w, expand * h
        if (x0 - pad_x) <= u <= (x1 + pad_x) and (y0 - pad_y) <= v <= (y1 + pad_y):
            cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
            candidates.append((math.hypot(u - cx, v - cy), track))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    best_d, best_track = candidates[0]
    margin = (candidates[1][0] - best_d) if len(candidates) > 1 else None
    return {'track': best_track, 'distance_m': None, 'margin_m': None if margin is None else round(margin, 2), 'label': 'image'}


def attribute_with_confidence(launch_xy, launch_ms, robot_tracks, max_distance=1.6, shooter_alliance=None, prior_team=None, launch_uv=None):
    """Who fired a ball, and how sure we are.

    The nearest robot within `max_distance` of the launch point wins, with three refinements taken from what
    other scouting vision projects found necessary: a ball aimed at a hub came from that hub's alliance (a hard
    rule, but never one that discards the only candidate); a robot that fired within the last
    BURST_WINDOW_MS gets a small head start, because volleys come from one robot; and the call carries a label
    so people review the shaky ones first. When field distance fails (single-camera depth error can project a
    launch several metres from the shooter), falls back to matching `launch_uv` against each robot's 0-1
    normalized box (expanded 15%) at `launch_ms`, with label 'image' and never crossing alliances.

    Returns {'track', 'distance_m', 'margin_m', 'label'}; label is confident, low_confidence, ambiguous, image
    or no_result. margin_m is how much farther the runner-up is (None when there is none within range).
    """
    found = []
    if launch_xy is not None:
        for track in robot_tracks:
            pos = _interpolate_robot(track, launch_ms)
            if pos is None:
                continue
            found.append((math.hypot(pos[0] - launch_xy[0], pos[1] - launch_xy[1]), track))
    in_range = [c for c in found if c[0] <= max_distance]
    ruled_out = False
    if shooter_alliance:
        same = [c for c in in_range if c[1].get('alliance') == shooter_alliance]
        if same:
            in_range = same
        elif in_range:
            ruled_out = True                      # only robots of the wrong alliance are near: keep them, flag it
    if not in_range:
        if launch_uv is not None:
            img_match = _attribute_by_image_box(launch_uv, launch_ms, robot_tracks, shooter_alliance=shooter_alliance)
            if img_match is not None:
                return img_match
        return {'track': None, 'distance_m': None, 'margin_m': None, 'label': 'no_result'}

    def score(item):
        d, track = item
        return d - BURST_BONUS_M if prior_team and track.get('team_key') == prior_team else d

    ranked = sorted(in_range, key=score)
    best = ranked[0]
    helped = best is not min(in_range, key=lambda c: c[0])        # the burst bonus changed the winner
    margin = None
    if len(ranked) > 1:
        margin = score(ranked[1]) - score(best)
    elif any(c[1] is not best[1] for c in found):
        others = [c[0] for c in found if c[1] is not best[1]]
        margin = min(others) - best[0]            # nobody else in range, but how far is the next robot
    distance = best[0]
    if ruled_out or helped or distance > CONFIDENT_RANGE_M:
        label = 'low_confidence'
    elif margin is not None and margin < AMBIGUOUS_MARGIN_M:
        label = 'ambiguous'
    elif margin is not None and margin < CLEAR_MARGIN_M:
        label = 'low_confidence'
    else:
        label = 'confident'
    return {'track': best[1], 'distance_m': distance, 'margin_m': None if margin is None else round(margin, 2), 'label': label}


def attribute_in_order(items, robot_tracks):
    """Attribute a list of (launch_ms, launch_xy, shooter_alliance[, launch_uv]) in firing order.

    Each ball can lean on who fired the one before it (within BURST_WINDOW_MS), but only a CONFIDENT call
    passes that along: a guess must not snowball down a whole volley. Returns the attribution dicts in the
    same order as `items`.
    """
    order = sorted(range(len(items)), key=lambda i: items[i][0])
    results = [None] * len(items)
    robots = list(robot_tracks)
    last_team, last_ms = None, -1e9
    for i in order:
        item = items[i]
        launch_ms, xy, alliance = item[0], item[1], item[2]
        launch_uv = item[3] if len(item) > 3 else None
        prior = last_team if launch_ms - last_ms <= BURST_WINDOW_MS else None
        results[i] = attribute_with_confidence(xy, launch_ms, robots, shooter_alliance=alliance, prior_team=prior, launch_uv=launch_uv)
        if results[i]['label'] == 'confident':
            last_team, last_ms = results[i]['track'].get('team_key'), launch_ms
    return results


def attribute_shots(shots, robot_tracks):
    """Attribute a list of tracker shot dicts to `robot_tracks` in firing order.

    Returns `shots` with `team_key`, `track_key`, `alliance`, `attribution_m`, `attribution` and
    `attribution_margin_m` populated identically to passing `robot_tracks` into `run_fuel_pass` /
    `run_fuel_pass_parallel`.
    """
    if not shots:
        return shots
    items = []
    for s in shots:
        if '_fit_confidence' not in s and s.get('confidence') is not None:
            s['_fit_confidence'] = float(s['confidence'])
        launch_ms = float(s.get('t0_ms') if s.get('t0_ms') is not None else s['t_ms'])
        xy = s['launch'][:2] if s.get('launch') is not None else None
        target = s.get('target')
        alliance = target.split('_')[0] if target else None
        items.append((launch_ms, xy, alliance, _launch_uv_from_shot(s)))
    attributions = attribute_in_order(items, list(robot_tracks))
    for s, attr in zip(shots, attributions):
        robot, distance = attr['track'], attr['distance_m']
        s['team_key'] = robot.get('team_key') if robot else None
        s['track_key'] = robot.get('track_key') if robot else None
        s['alliance'] = robot.get('alliance') if robot else None
        s['attribution_m'] = None if distance is None else round(distance, 2)
        s['attribution'] = attr['label']
        s['attribution_margin_m'] = attr['margin_m']
        base_conf = s.get('_fit_confidence')
        if base_conf is not None:
            if attr['label'] == 'image':
                s['confidence'] = round(max(0.05, base_conf * IMAGE_CONFIDENCE_SCALE), 2)
            else:
                s['confidence'] = round(base_conf, 2)
    return shots


MAX_BALL_TRACK_POINTS = 90
ENTRY_RESULT = {'scored': 'scored', 'rim': 'rim', 'missed': 'miss', 'unknown': 'short'}


def build_ball_track(track, width, height, offset_s=0.0, max_points=MAX_BALL_TRACK_POINTS):
    """Sampled observed detection boxes along a flight, normalized to [0, 1] by frame width/height.

    Uses the observed detections (not the fitted curve), sorted in time and evenly subsampled to at most
    `max_points`. Timestamps use the same match-millisecond base as `t_ms`: round((t - offset_s) * 1000).
    """
    if not track or not getattr(track, 'obs', None) or width <= 0 or height <= 0:
        return []
    obs = sorted(track.obs, key=lambda o: o[1])
    if len(obs) > max_points:
        indices = np.unique(np.linspace(0, len(obs) - 1, max_points).round().astype(int))
        obs = [obs[i] for i in indices]
    w, h = float(width), float(height)
    out = []
    for o in obs:
        t_ms = int(round((o[1] - offset_s) * 1000.0))
        u, v, area = float(o[2]), float(o[3]), max(1.0, float(o[4]))
        r = max(2.0, math.sqrt(area / math.pi))
        x0 = max(0.0, min(1.0, (u - r) / w))
        y0 = max(0.0, min(1.0, (v - r) / h))
        x1 = max(0.0, min(1.0, (u + r) / w))
        y1 = max(0.0, min(1.0, (v + r) / h))
        if x1 <= x0:
            x1 = min(1.0, x0 + 1.0 / w)
        if y1 <= y0:
            y1 = min(1.0, y0 + 1.0 / h)
        out.append({'t': t_ms, 'box': [round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4)]})
    return out


def _project_hub_opening(camera, hub_xy, n_samples=24):
    """Projected image region of a hub's hexagonal opening at HUB_Z: center, semi-axes and ball radius in px."""
    hx, hy = float(hub_xy[0]), float(hub_xy[1])
    angles = np.linspace(0.0, 2.0 * math.pi, n_samples, endpoint=False)
    ring = np.column_stack([hx + HUB_OPENING_RADIUS * np.cos(angles), hy + HUB_OPENING_RADIUS * np.sin(angles), np.full(n_samples, HUB_Z)])
    pts = np.vstack([[[hx, hy, HUB_Z]], ring])
    uv, front = camera.project(pts)
    if not bool(front[0]):
        return None
    center = uv[0]
    ring_uv = uv[1:][front[1:]]
    if len(ring_uv) < 4:
        return None
    du = np.abs(ring_uv[:, 0] - center[0])
    dv = np.abs(ring_uv[:, 1] - center[1])
    rx = max(4.0, float(np.max(du)))
    ry = max(3.0, float(np.max(dv)))
    ball_px = max(2.0, rx * (BALL_RADIUS / HUB_OPENING_RADIUS))
    return {'center': (float(center[0]), float(center[1])), 'rx': rx, 'ry': ry, 'ball_px': ball_px}


class FuelStats(dict):
    """JSON stats plus private track attributes: linked `.tracks` / `.ball_tracks`,
    and `.source_tracks` with the same linked tracks followed by short native observations.
    None means an older caller did not retain the separate launch source pool.
    """

    def __init__(self, *args, tracks=None, source_tracks=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracks = list(tracks) if tracks is not None else []
        self.ball_tracks = self.tracks
        # Private native observations for local exits; never serialized into Hub stats.
        self.source_tracks = list(source_tracks) if source_tracks is not None else None


def _trim_post_entry_obs(obs, projs):
    """Drop false link_tracks continuations (a >90 ms gap or >45 px jump) after a track has reached a hub opening."""
    if not projs or len(obs) < 4:
        return obs
    for proj in projs.values():
        cx, cy = proj['center']
        rx, ry = proj['rx'], proj['ry']
        ds = [math.hypot((float(o[2]) - cx) / rx, (float(o[3]) - cy) / ry) for o in obs]
        if min(ds) <= 1.85:
            min_i = ds.index(min(ds))
            for k in range(max(1, min_i + 1), len(obs)):
                dt = float(obs[k][1]) - float(obs[k - 1][1])
                step = math.hypot(float(obs[k][2]) - float(obs[k - 1][2]), float(obs[k][3]) - float(obs[k - 1][3]))
                if dt > 0.09 or step > 45.0:
                    return obs[:k]
            break
    return obs


def _hub_polygon_entry(obs, camera, hubs, offset_s=0.0, polygons=None):
    """Check whether `obs` (or its descending branch from image apex) enters a padded hub opening polygon."""
    if len(obs) < 2:
        return None
    import fuel_count as fc
    if polygons is None:
        if camera is None or not hubs:
            return None
        polygons = fc.hub_polygons(camera, hubs, pad=fc.HUB_PAD)
    if not polygons:
        return None
    he = fc.hub_entries([obs], polygons, offset_s=offset_s)
    if he:
        return he[0]
    apex_idx = int(np.argmin([float(o[3]) for o in obs]))
    if len(obs) - apex_idx >= 2:
        he_desc = fc.hub_entries([obs[apex_idx:]], polygons, offset_s=offset_s, min_obs=2)
        if he_desc:
            return he_desc[0]
    return None


def _extrapolate_to_hub(obs, fit, camera, hubs, projs, polygons=None, duration_s=0.0):
    """Extrapolate a track lost before the hub using its projected 3D ballistic arc and 2D image parabola.

    Returns {'target', 'd_norm', 'in_poly', 'cross_t_s', 'gap_s', 'method'} or None.
    """
    if camera is None or not projs or len(obs) < 6 or duration_s < 0.24:
        return None
    from ballistics import simulate
    times = np.array([float(o[1]) for o in obs], float)
    us = np.array([float(o[2]) for o in obs], float)
    vs = np.array([float(o[3]) for o in obs], float)
    min_cy = min(p['center'][1] for p in projs.values())
    # Ignore tracks that start and fall purely below every hub opening (chute exits or floor bounces)
    if float(np.min(vs)) > min_cy + 25.0 and float(vs[-1]) > float(vs[0]) + 8.0:
        return None

    best = None
    # 1. Project the fitted 3D ballistic trajectory forward into the camera
    if fit and fit.get('launch') is not None and fit.get('velocity') is not None:
        t_grid = np.arange(max(0.0, duration_s - 0.05), duration_s + 0.85, 1.0 / 60.0)
        if len(t_grid) >= 2:
            pts_3d = simulate(fit['launch'], fit['velocity'], t_grid)
            uv_3d, front = camera.project(pts_3d)
            for name, proj in projs.items():
                cx, cy = proj['center']
                rx, ry = proj['rx'], proj['ry']
                poly = (polygons or {}).get(name)
                for k in range(1, len(t_grid)):
                    if not (bool(front[k]) and bool(front[k - 1])) or uv_3d[k, 1] <= uv_3d[k - 1, 1]:
                        continue
                    u_k, v_k = float(uv_3d[k, 0]), float(uv_3d[k, 1])
                    d_norm = math.hypot((u_k - cx) / rx, (v_k - cy) / ry)
                    in_poly = poly is not None and cv2.pointPolygonTest(poly, (u_k, v_k), False) >= 0
                    gap = float(t_grid[k] - duration_s)
                    if (in_poly or d_norm <= 1.65) and -0.05 <= gap <= 0.80:
                        cand = {
                            'target': name, 'd_norm': d_norm, 'in_poly': bool(in_poly),
                            'cross_t_s': float(t_grid[k]), 'gap_s': max(0.0, gap), 'method': '3d_proj',
                        }
                        if best is None or d_norm < best['d_norm']:
                            best = cand
                        break

    # 2. 2D quadratic image-space extrapolation (u linear, v quadratic with downward gravity)
    t_rel = times - times[-1]
    cu = np.polyfit(t_rel, us, 1)
    cv = np.polyfit(t_rel, vs, 2)
    if float(cv[0]) > 15.0:
        dt_grid = np.linspace(0.0, 0.75, 46)
        u_ext = np.polyval(cu, dt_grid)
        v_ext = np.polyval(cv, dt_grid)
        for name, proj in projs.items():
            cx, cy = proj['center']
            rx, ry = proj['rx'], proj['ry']
            poly = (polygons or {}).get(name)
            for k in range(1, len(dt_grid)):
                if v_ext[k] <= v_ext[k - 1]:
                    continue
                u_k, v_k = float(u_ext[k]), float(v_ext[k])
                d_norm = math.hypot((u_k - cx) / rx, (v_k - cy) / ry)
                in_poly = poly is not None and cv2.pointPolygonTest(poly, (u_k, v_k), False) >= 0
                if in_poly or d_norm <= 1.60:
                    gap = float(dt_grid[k])
                    cand = {
                        'target': name, 'd_norm': d_norm, 'in_poly': bool(in_poly),
                        'cross_t_s': duration_s + gap, 'gap_s': gap, 'method': '2d_poly',
                    }
                    if best is None or d_norm < best['d_norm']:
                        best = cand
                    break
    return best


def _select_hub(track, fit, verdict, hubs, camera):
    """Pick the target hub and its projected opening from hub-polygon entry, image-space descent, or 3D verdict."""
    if not hubs:
        return None
    projs = {}
    if camera is not None:
        for name, xy in hubs.items():
            proj = _project_hub_opening(camera, xy)
            if proj is not None:
                projs[name] = proj
    best_info = None
    target = verdict.get('target') if verdict else None
    if target and target in projs:
        best_info = (target, hubs[target], projs[target])

    if camera is not None and track and getattr(track, 'obs', None) and projs:
        obs = _trim_post_entry_obs(sorted(track.obs, key=lambda o: o[1]), projs)
        he = _hub_polygon_entry(obs, camera, hubs, offset_s=0.0)
        if he and he['hub'] in projs:
            hname = he['hub']
            return (hname, hubs[hname], projs[hname])

        apex_idx = int(np.argmin([float(o[3]) for o in obs]))
        desc_obs = obs[apex_idx:] if apex_idx < len(obs) else obs
        by_px = []
        for name, proj in projs.items():
            cx, cy = proj['center']
            rx, ry = proj['rx'], proj['ry']
            min_desc_d = min(math.hypot((float(o[2]) - cx) / rx, (float(o[3]) - cy) / ry) for o in desc_obs)
            last_d = math.hypot((float(obs[-1][2]) - cx) / rx, (float(obs[-1][3]) - cy) / ry)
            by_px.append((min(min_desc_d, last_d), name, hubs[name], proj))
        if by_px:
            by_px.sort(key=lambda item: item[0])
            if best_info is None or by_px[0][0] < 2.5:
                _, name, xy, proj = by_px[0]
                best_info = (name, xy, proj)
            elif len(obs) >= 6:
                duration_s = float(obs[-1][1] - obs[0][1])
                ext = _extrapolate_to_hub(obs, fit, camera, hubs, projs, duration_s=duration_s)
                if ext and ext['target'] in projs:
                    ename = ext['target']
                    best_info = (ename, hubs[ename], projs[ename])
    return best_info


def evaluate_hub_entry(track, fit, hubs, cameras=None, verdict=None, offset_s=0.0, hub_entries_list=None, polygons=None):
    """Decide whether a tracked shot entered the hub, combining image-space hub-polygon evidence,
    ray-depth-corrected 3D crossing, and ballistic extrapolation for tracks lost before the hub.
    """
    from ballistics import ray_corrected_crossing
    verdict = classify_flight(fit, hubs) if verdict is None else verdict
    obs = sorted(track.obs, key=lambda o: o[1]) if track and getattr(track, 'obs', None) else []
    n_total = len(obs) if obs else int(fit.get('observations', 0))
    rms_px = float(fit.get('rms_px', 2.0))
    fit_res = verdict.get('result', 'short')
    miss_m = verdict.get('miss_m')
    cross_t = verdict.get('cross_t_s')
    t0_s = float(obs[0][1]) if obs else float(fit.get('t0_s', 0.0))

    cam_last = None
    if isinstance(cameras, dict) and obs:
        cam_last = cameras.get(obs[-1][0]) or next(iter(cameras.values()), None)
    elif isinstance(cameras, (list, tuple)) and cameras:
        cam_last = cameras[-1]
    elif cameras is not None and hasattr(cameras, 'project'):
        cam_last = cameras

    projs = {}
    if cam_last is not None and hubs:
        for name, xy in hubs.items():
            proj = _project_hub_opening(cam_last, xy)
            if proj is not None:
                projs[name] = proj
    if projs and len(obs) >= 4:
        obs = _trim_post_entry_obs(obs, projs)
    duration_s = float(obs[-1][1] - obs[0][1]) if len(obs) >= 2 else float(fit.get('duration_s', 0.0))

    if polygons is None and cam_last is not None and hubs:
        import fuel_count as fc
        polygons = fc.hub_polygons(cam_last, hubs, pad=fc.HUB_PAD)

    hub_info = _select_hub(BallTrack(obs) if obs else track, fit, verdict, hubs or {}, cam_last)
    inner_m = HUB_OPENING_RADIUS - BALL_RADIUS
    rim_outer_m = inner_m + 2.0 * BALL_RADIUS
    if hub_info is not None and hub_info[0] != verdict.get('target'):
        cross_xy = verdict.get('cross_xy')
        if cross_xy is not None and len(cross_xy) >= 2 and hub_info[1] is not None:
            miss_m = float(math.hypot(cross_xy[0] - hub_info[1][0], cross_xy[1] - hub_info[1][1]))
            fit_res = 'scored' if miss_m <= inner_m else 'rim' if miss_m <= rim_outer_m else 'miss'
        else:
            miss_m = None
            fit_res = 'short'

    ray_cross = None
    if cam_last is not None and hasattr(cam_last, 'center') and fit and fit.get('launch') is not None and fit.get('velocity') is not None:
        target_hubs = {hub_info[0]: hub_info[1]} if (hub_info is not None and hub_info[1] is not None) else (hubs or {})
        ray_cross = ray_corrected_crossing(fit, target_hubs, camera_center=cam_last.center)

    gap_s = max(0.0, float(cross_t) - duration_s) if cross_t is not None else 0.35
    n_near = 0
    decision = None
    evidence = ''
    f_agree = 0.85
    entry_time_s = None

    if hub_info is not None and len(obs) >= 2:
        target_name, _, proj = hub_info
        cx, cy = proj['center']
        rx, ry, ball_px = proj['rx'], proj['ry'], proj['ball_px']
        pad_x = rx * 1.6 + 1.5 * ball_px
        pad_y = ry * 1.6 + 1.5 * ball_px

        pts_uv = [(float(o[2]), float(o[3])) for o in obs]
        times = [float(o[1]) for o in obs]
        t_rel = [t - times[0] for t in times]
        d_norms = [math.hypot((u - cx) / rx, (v - cy) / ry) for u, v in pts_uv]
        n_near = sum(1 for d in d_norms if d <= 2.5)

        apex_idx = int(np.argmin([v for _, v in pts_uv]))
        k_tail = min(len(obs) - 1, max(1, min(4, len(obs) // 3)))
        dt_tail = max(1e-3, times[-1] - times[-1 - k_tail])
        vy_tail = (pts_uv[-1][1] - pts_uv[-1 - k_tail][1]) / dt_tail
        descending_end = vy_tail > 5.0 or (pts_uv[-1][1] - pts_uv[apex_idx][1] >= 4.0)

        last_u, last_v = pts_uv[-1]
        last_d = d_norms[-1]
        desc_start = min(apex_idx, len(obs) - 1)
        desc_d = d_norms[desc_start:]
        min_desc_offset = int(np.argmin(desc_d))
        min_desc_idx = desc_start + min_desc_offset
        min_desc_d = desc_d[min_desc_offset]

        he_hit = _hub_polygon_entry(obs, cam_last, {target_name: hub_info[1]}, offset_s=offset_s, polygons=polygons)
        in_opening = (last_d <= 1.25 and abs(last_v - cy) <= pad_y and abs(last_u - cx) <= pad_x) or (
            he_hit is not None and min_desc_d <= 1.65
        )
        on_rim_px = 0.82 < last_d <= 1.45 and abs(last_v - cy) <= pad_y * 1.15

        # A ball that drops into the semi-transparent upper funnel remains visible for 15-45 px below cy;
        # only a track that continues well below the funnel (>= 3 sightings below cy + 2*ry and past below_thresh_v)
        # without a clean 3D scoring arc is a pass-by miss.
        below_thresh_v = cy + max(3.5 * ry + 1.5 * ball_px, 52.0)
        n_below_funnel = sum(1 for _, v in pts_uv[desc_start:] if v > cy + 2.0 * ry)
        ended_below = descending_end and last_v > below_thresh_v and n_below_funnel >= 3 and min_desc_d <= 2.5
        passed_opening_on_descent = (
            min_desc_d <= 1.6
            and min_desc_idx < len(obs) - 1
            and ended_below
        )

        # Effective gap combines time to crossing and image distance to opening
        dist_gap_s = max(0.0, (min(last_d, min_desc_d) - 1.0) * 0.12)
        if cross_t is not None:
            time_gap = max(0.0, float(cross_t) - duration_s)
            gap_s = min(time_gap, dist_gap_s) if in_opening else (min(time_gap, dist_gap_s) if last_d <= 1.8 else max(time_gap, dist_gap_s))
        else:
            gap_s = dist_gap_s

        if (in_opening or he_hit is not None) and descending_end and not passed_opening_on_descent:
            if fit_res == 'rim' or (miss_m is not None and inner_m < miss_m <= rim_outer_m and min_desc_d > 0.85 and last_d > 0.85):
                decision = 'rim'
                f_agree = 0.9
                evidence = f'observed descending at {target_name} rim (miss={miss_m:.2f}m)'
            else:
                decision = 'scored'
                if fit_res == 'scored' and cross_t is not None:
                    entry_time_s = float(fit.get('t0_s', t0_s)) + float(cross_t)
                elif he_hit is not None and last_d > 1.35:
                    entry_time_s = float(he_hit['t_ms']) / 1000.0 + offset_s
                else:
                    entry_time_s = times[-1]
                gap_s = min(gap_s, 0.08)
                n_near = max(n_near, 4)
                if fit_res == 'scored':
                    f_agree = 1.0
                    evidence = f'descending into {target_name} opening; 3D fit agrees (miss={miss_m:.2f}m)'
                elif ray_cross is not None and ray_cross['miss_m'] <= HUB_OPENING_RADIUS * 1.25 and (miss_m is None or miss_m > 1.5):
                    # When raw 3D fit was only offset in depth along the ray, report honest moderate agreement
                    f_agree = max(0.58, 0.82 - 0.06 * min(4.0, max(0.0, (miss_m or 2.0) - HUB_OPENING_RADIUS)))
                    evidence = f'observed descending into {target_name} opening (d={min_desc_d:.2f}); 3D fit offset {miss_m or ray_cross["miss_m"]:.2f}m'
                elif miss_m is not None:
                    f_agree = max(0.58, 0.82 - 0.06 * min(4.0, max(0.0, miss_m - HUB_OPENING_RADIUS)))
                    evidence = f'observed descending into {target_name} opening (d={min_desc_d:.2f}); 3D fit offset {miss_m:.2f}m'
                else:
                    f_agree = 0.68
                    evidence = f'observed descending into {target_name} opening (d={min_desc_d:.2f})'
        elif passed_opening_on_descent or ended_below:
            cross_idx = int(np.argmin([abs(tr - float(cross_t)) for tr in t_rel])) if cross_t is not None else min_desc_idx
            d_at_cross = d_norms[cross_idx]
            extra_after_cross = duration_s - float(cross_t) if cross_t is not None else 0.0
            if (
                fit_res == 'scored'
                and cross_t is not None
                and d_at_cross <= 1.25
                and extra_after_cross >= 0.2
                and rms_px <= 1.5
                and float(fit.get('outlier_fraction', 0.0)) == 0.0
            ):
                decision = 'scored'
                entry_time_s = float(fit.get('t0_s', t0_s)) + float(cross_t)
                f_agree = 0.95
                gap_s = 0.0
                evidence = f'crossed {target_name} opening on fitted arc (miss={miss_m:.2f}m)'
            else:
                decision = 'missed'
                f_agree = 1.0 if fit_res in ('miss', 'short') else 0.78
                evidence = f'observed continuing below/past {target_name} opening (d={last_d:.2f})'
        else:
            # Track ended before or beside the hub opening: test ballistic extrapolation & ray-corrected 3D crossing
            ext = _extrapolate_to_hub(obs, fit, cam_last, {target_name: hub_info[1]}, {target_name: proj}, polygons=polygons, duration_s=duration_s)
            if ext is None and len(projs) > 1:
                ext = _extrapolate_to_hub(obs, fit, cam_last, hubs, projs, polygons=polygons, duration_s=duration_s)
                if ext is not None and ext['target'] != target_name:
                    target_name = ext['target']
                    proj = projs[target_name]
            ray_ok = (
                ray_cross is not None
                and duration_s >= 0.24
                and ray_cross['miss_m'] <= HUB_OPENING_RADIUS * 1.35
                and -0.05 <= (ray_cross['cross_t_s'] - duration_s) <= 0.80
                and not (min(v for _, v in pts_uv) > cy + 25.0 and pts_uv[-1][1] > pts_uv[0][1] + 8.0)
            )
            if ext is not None or ray_ok:
                pred_cross_t = ext['cross_t_s'] if ext is not None else ray_cross['cross_t_s']
                gap_s = max(0.0, float(pred_cross_t) - duration_s)
                entry_time_s = t0_s + float(pred_cross_t)
                if hub_entries_list:
                    pred_ms = int(round((entry_time_s - offset_s) * 1000.0))
                    near_fc = [e['t_ms'] for e in hub_entries_list if e.get('hub') == target_name and abs(int(e['t_ms']) - pred_ms) <= 300]
                    if near_fc:
                        best_fc_ms = min(near_fc, key=lambda tm: abs(int(tm) - pred_ms))
                        entry_time_s = float(best_fc_ms) / 1000.0 + offset_s
                        gap_s = min(gap_s, 0.12)
                if fit_res == 'rim' or (ray_cross is not None and inner_m < ray_cross['miss_m'] <= rim_outer_m and (ext is None or ext['d_norm'] > 1.15)):
                    decision = 'rim'
                    f_agree = 0.82
                    evidence = f'extrapolated arc grazes {target_name} rim (gap={gap_s:.2f}s)'
                else:
                    decision = 'scored'
                    n_near = max(n_near, 3)
                    f_agree = 0.78 if (ext is not None and ext.get('in_poly')) else 0.7
                    d_rep = ext['d_norm'] if ext is not None else (ray_cross['miss_m'] / HUB_OPENING_RADIUS)
                    evidence = f'extrapolated into {target_name} opening (d={d_rep:.2f}, gap={gap_s:.2f}s)'
            elif last_d > 2.0 and (vy_tail <= 0.0 or cross_t is None or duration_s < float(cross_t) - 0.15):
                decision = 'unknown'
                gap_s = max(gap_s, 0.25 if cross_t is None else max(0.18, float(cross_t) - duration_s))
                f_agree = 0.6
                evidence = f'track lost before reaching {target_name} (d={last_d:.2f}, gap={gap_s:.2f}s)'
            elif on_rim_px and descending_end and (fit_res == 'rim' or (miss_m is not None and miss_m <= rim_outer_m + BALL_RADIUS)):
                decision = 'rim'
                f_agree = 0.9 if fit_res == 'rim' else 0.78
                evidence = f'observed near {target_name} rim (d={last_d:.2f})'
            elif last_d <= 2.2 and descending_end:
                if fit_res == 'rim' or (miss_m is not None and inner_m < miss_m <= rim_outer_m):
                    decision = 'rim'
                    f_agree = 0.88
                    evidence = f'fit grazes {target_name} rim (miss={miss_m:.2f}m)'
                else:
                    decision = 'missed'
                    f_agree = 0.95 if fit_res == 'miss' else 0.72
                    evidence = f'observed descending beside {target_name} opening (d={last_d:.2f})'
            else:
                if cross_t is None or duration_s < float(cross_t) - 0.18:
                    decision = 'unknown'
                    gap_s = max(gap_s, 0.25)
                    f_agree = 0.6
                    evidence = f'flight lost early before {target_name} crossing (gap={gap_s:.2f}s)'
                elif fit_res == 'rim':
                    decision = 'rim'
                    f_agree = 0.8
                    evidence = f'3D fit crosses {target_name} rim (miss={miss_m:.2f}m)'
                elif fit_res == 'miss':
                    decision = 'missed'
                    f_agree = 0.85
                    evidence = f'3D fit misses {target_name} (miss={miss_m:.2f}m)'
                else:
                    decision = 'unknown'
                    f_agree = 0.6
                    evidence = f'unconfirmed at {target_name} opening (d={last_d:.2f})'
    else:
        if cross_t is None:
            decision = 'unknown'
            gap_s = 0.4
            f_agree = 0.6
            evidence = 'fitted arc never reaches hub height'
        elif duration_s < float(cross_t) - 0.18 or n_total < 8:
            decision = 'unknown'
            gap_s = max(0.2, float(cross_t) - duration_s)
            f_agree = 0.6
            evidence = f'track lost {gap_s:.2f}s before hub crossing'
        else:
            gap_s = max(0.0, float(cross_t) - duration_s)
            n_near = max(1, int(round(n_total * max(0.2, 1.0 - gap_s / max(0.3, float(cross_t))))))
            if fit_res == 'scored':
                decision = 'scored'
                entry_time_s = float(fit.get('t0_s', t0_s)) + float(cross_t)
                f_agree = 0.9
                evidence = f'3D fit crosses inside {verdict.get("target")} opening (miss={miss_m:.2f}m)'
            elif fit_res == 'rim':
                decision = 'rim'
                f_agree = 0.85
                evidence = f'3D fit crosses {verdict.get("target")} rim (miss={miss_m:.2f}m)'
            else:
                decision = 'missed'
                f_agree = 0.9
                evidence = f'3D fit misses {verdict.get("target")} by {miss_m:.2f}m'

    f_obs = max(0.25, min(1.0, (n_near + 0.35 * min(n_total, 20)) / 14.0))
    f_rms = max(0.25, min(1.0, 1.0 - rms_px / 6.0))
    f_gap = max(0.2, min(1.0, 1.0 - gap_s / 0.45))
    base = 0.95 if decision in ('scored', 'missed') else 0.82 if decision == 'rim' else 0.55
    conf = round(min(0.95, max(0.05, base * f_obs * f_rms * f_gap * f_agree)), 2)
    entry_t_ms = int(round((entry_time_s - offset_s) * 1000.0)) if (decision == 'scored' and entry_time_s is not None) else None
    return {'decision': decision, 'confidence': conf, 'evidence': evidence, 't_ms': entry_t_ms}


def shot_confidence(fit):
    """0-1: how much to trust one shot's numbers. Penalises a poor pixel fit, a short observed arc, and a
    wide speed uncertainty. A heuristic for ranking and flagging, not a calibrated probability."""
    score = 1.0
    score *= max(0.0, 1.0 - fit['rms_px'] / 8.0)
    score *= max(0.0, 1.0 - 1.5 * fit.get('outlier_fraction', 0.0))
    score *= min(1.0, fit['observations'] / 14.0)
    if fit.get('speed_sigma') is not None:
        score *= max(0.0, 1.0 - fit['speed_sigma'] / (0.25 * fit['speed']))     # +-25% or worse counts as no confidence
    return round(max(0.0, min(1.0, score)), 2)


def run_fuel_pass(frames, h_ref_to_field, ref_frame, hubs, robot_tracks=(), offset_s=0.0, on_progress=None, mask_ref=None, workers=None, tracks_only=False, return_tracks=False,
                  opening_region_ref=None, opening_mode=None):
    """Process frames (an iterable of (seconds, bgr) at native fps) and return (shots, stats) (or (shots, tracks, stats) when return_tracks=True).

    `stats` is a `FuelStats(dict)` carrying `.tracks` (and `.ball_tracks`) with the raw linked `BallTrack`s so callers
    can consume both fitted shots and linked ball tracks from a single pass while `stats` remains JSON-serializable.

    hubs: {'blue_hub': (x, y), 'red_hub': (x, y)} in the field frame; robot_tracks: output of
    build_tracks (match-time trajectories) for attribution; offset_s: video second that is match time 0.

    Cost per frame, on a static 720p broadcast camera: about 3 ms (decode 0.5, region-of-interest detection
    0.7, static-gate registration 1.4). It was about 14 ms before registration was gated and everything was
    cropped to the region. Fitting is the other cost: candidate tracks are pre-filtered in the picture
    (tracks that cannot be thrown balls never reach the 3D fit) and the rest are fitted in parallel.
    """
    height, width = ref_frame.shape[:2]
    motion = CameraMotion(min_inliers=25)
    # mask_ref (see roi.py) limits everything to the wide camera's region and the field volume, so the
    # close-up panels, scoreboard and crowd are never registered against or searched for balls.
    motion.start(cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY), mask_ref)
    ref_camera = camera_from_pixel_homography(h_ref_to_field, width, height)
    if ref_camera is None:
        raise RuntimeError('could not recover a 3D camera from the field calibration (is the view calibrated on a clear frame?)')
    bx, by, bw, bh = cv2.boundingRect(mask_ref) if mask_ref is not None else (0, 0, width, height)
    t_crop = np.array([[1.0, 0.0, -bx], [0.0, 1.0, -by], [0.0, 0.0, 1.0]])
    high_rate = high_rate_mode()
    detector, tracker = MovingFuelDetector(), (StreamBallTracker if high_rate else BallTracker)(min_obs=4)
    detector.camera = ref_camera if high_rate else None
    detector.ref_camera = ref_camera
    near_ref = np.asarray(opening_region_ref, np.uint8) if (opening_region_ref is not None and opening_mode) else None
    detector.opening_mode = opening_mode if near_ref is not None else None
    near_crop_static = near_ref[by:by + bh, bx:bx + bw] if near_ref is not None else None
    cameras = {}
    identity = np.eye(3)
    prev_gray, prev_h = None, identity
    stats = {'frames': 0, 'unregistered': 0, 'unregistered_spans': [], 'static_frames': 0, 'blobs': 0, 'tracks': 0, 'linked': 0, 'prefiltered_out': 0, 'fitted': 0, 'rejected_fit': 0,
             'ref_focal_px': round(ref_camera.f, 1), 'opening_detection': detector.opening_mode, 'counter_options': counter_options()}
    index = -1
    last_t = None
    for t, img in frames:
        if high_rate and last_t is not None and t - last_t < 1.0 / TRACK_MAX_FPS - 1e-3:
            continue                                       # 60 fps video: track every other frame (see TRACK_MAX_FPS)
        last_t = t
        index += 1
        stats['frames'] += 1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        crop = img[by:by + bh, bx:bx + bw]
        crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        yellow = cv2.inRange(crop_hsv, detector.low, detector.high)       # = detector.yellow_mask(crop)

        def feature_mask():
            # Register on the field, not on the balls: yellow regions (flying fuel, piles) are masked out of
            # the feature search, otherwise a ball crossing the picture drags the camera estimate with it.
            usable = np.full((height, width), 255, np.uint8)
            usable[by:by + bh, bx:bx + bw] = 255 - cv2.dilate(yellow, detector.kernel, iterations=4)
            return usable

        h_cur_to_ref, ok = motion.update(gray, prefer_reference=True, ref_mask=mask_ref, mask_factory=feature_mask)
        if not ok:
            stats['unregistered'] += 1
            note_unregistered(stats, t)
            prev_gray = None
            continue
        static = bool(np.array_equal(h_cur_to_ref, identity))
        stats['static_frames'] += static
        if prev_gray is None:
            warped = None
        elif (static and prev_h is identity) or np.array_equal(h_cur_to_ref, prev_h):
            warped = prev_gray[by:by + bh, bx:bx + bw]
        else:
            warped = cv2.warpPerspective(prev_gray, t_crop @ np.linalg.inv(h_cur_to_ref) @ prev_h, (bw, bh))
        if near_ref is not None:
            detector.near_mask = near_crop_static if static else warp_mask_to_frame(near_ref, h_cur_to_ref, width, height)[by:by + bh, bx:bx + bw]
        blobs = detector.detect(t, crop, warped, offset=(bx, by), gray=gray[by:by + bh, bx:bx + bw], yellow=yellow, hsv=crop_hsv) if warped is not None else []
        if blobs:
            if static:
                for b in blobs:
                    b.ru, b.rv = b.u, b.v
            else:
                ref_pts = cv2.perspectiveTransform(np.array([[[b.u, b.v] for b in blobs]], np.float32), h_cur_to_ref)[0]
                for b, (ru, rv) in zip(blobs, ref_pts):
                    b.ru, b.rv = float(ru), float(rv)
            if mask_ref is not None:
                blobs = [b for b in blobs if 0 <= int(b.rv) < height and 0 <= int(b.ru) < width and mask_ref[int(b.rv), int(b.ru)]]
        stats['blobs'] += len(blobs)
        tracker.step(index, t, blobs)
        if not tracks_only:
            cameras[index] = ref_camera if static else (camera_from_pixel_homography(h_ref_to_field @ h_cur_to_ref, width, height, f_prior=ref_camera.f) or ref_camera)
        prev_gray, prev_h = gray, (identity if static else h_cur_to_ref)
        if on_progress:
            on_progress(t)
    if not tracks_only and cameras and stats['static_frames'] < len(cameras):
        order = sorted(cameras)
        cameras = dict(zip(order, smooth_cameras([cameras[i] for i in order], window=11)))

    # ---- tracks -> flights: bridge occlusion gaps, drop what cannot be a thrown ball, fit the rest
    short_sources = [tr for tr in tracker.done + tracker.active if len(tr.obs) == 3]
    raw = tracker.finish()
    stats['tracks'] = len(raw)
    tracks = link_tracks(raw)
    stats['linked'] = len(raw) - len(tracks)
    stats = FuelStats(stats, tracks=tracks, source_tracks=tracks + short_sources)
    if tracks_only:                                     # the ball counter (fuel_count.py) needs every ball track, not fitted flights
        return tracks, stats
    import fuel_count as fc
    polygons = fc.hub_polygons(ref_camera, hubs, pad=fc.HUB_PAD) if hubs else {}
    all_hub_entries = fc.hub_entries(tracks, polygons, offset_s) if polygons else []
    vertical_px_per_m = float(np.linalg.norm(ref_camera.project([[8.27, 4.0, 1.0]])[0][0] - ref_camera.project([[8.27, 4.0, 0.0]])[0][0]))
    candidates = []
    for track in tracks:
        passes, _ = quick_ballistic_check(track, vertical_px_per_m)
        if passes and plausible_flight(track):
            candidates.append(track)
    stats['prefiltered_out'] = len(tracks) - len(candidates)
    tasks = [([o[1] for o in tr.obs], [(o[2], o[3]) for o in tr.obs], [cameras[o[0]] for o in tr.obs], [math.sqrt(o[4] / math.pi) for o in tr.obs]) for tr in candidates]
    fits = fit_tracks(tasks, workers)
    shots = []
    staged = []
    for track, fit in zip(candidates, fits):
        times = [o[1] for o in track.obs]
        if fit is None or fit['rms_px'] > 3.5 or fit['outlier_fraction'] > 0.25:
            stats['rejected_fit'] += 1
            stats.setdefault('rejections', []).append({'t': round(times[0], 2), 'obs': len(times), 'fit': None if fit is None else round(fit['rms_px'], 2)})
            continue
        verdict = classify_flight(fit, hubs)
        # A shot has to get up toward hub height (1.83 m). A ball that rolls, bounces or is nudged along the
        # carpet also moves and can even fit a low arc, but it is not a shot: on real broadcast footage these
        # outnumber real flights by an order of magnitude.
        if verdict['apex_m'] < MIN_SHOT_APEX:
            stats['ground_balls'] = stats.get('ground_balls', 0) + 1
            continue
        stats['fitted'] += 1
        point, exit_velocity, back_s = state_at_exit(fit)
        staged.append((((fit['t0_s'] - offset_s + back_s) * 1000.0), track, fit, verdict, point, exit_velocity))
    staged.sort(key=lambda item: item[0])
    for launch_ms, track, fit, verdict, point, exit_velocity in staged:
        launch_speed = float(np.linalg.norm(exit_velocity))
        horizontal = math.hypot(exit_velocity[0], exit_velocity[1])
        areas = [o[4] for o in track.obs]
        entry = evaluate_hub_entry(track, fit, hubs, cameras=cameras, verdict=verdict, offset_s=offset_s, hub_entries_list=all_hub_entries, polygons=polygons)
        result = ENTRY_RESULT.get(entry['decision'], verdict['result'])
        ball_track = build_ball_track(track, width, height, offset_s=offset_s)
        target = verdict['target']
        if hubs and cameras and track and getattr(track, 'obs', None):
            last_obs = max(track.obs, key=lambda o: o[1])
            cam_last = cameras.get(last_obs[0], ref_camera)
            sel = _select_hub(track, fit, verdict, hubs, cam_last)
            if sel is not None:
                target = sel[0]
        hub_xy = hubs.get(target) if target else None
        score_ms = None
        if result == 'scored':
            score_ms = entry.get('t_ms')
            if score_ms is None and verdict.get('cross_t_s') is not None:
                score_ms = int(round((fit['t0_s'] - offset_s + verdict['cross_t_s']) * 1000))
            if score_ms is None and ball_track:
                score_ms = ball_track[-1]['t']
            entry['t_ms'] = score_ms
        shots.append({
            't_ms': int(round(launch_ms)), 'team_key': None, 'track_key': None, 'alliance': None,
            'speed_mps': round(launch_speed, 2), 'speed_sigma_mps': None if fit['speed_sigma'] is None else round(fit['speed_sigma'], 2),
            'elevation_deg': round(math.degrees(math.atan2(exit_velocity[2], horizontal)), 1), 'azimuth_deg': round(math.degrees(math.atan2(exit_velocity[1], exit_velocity[0])), 1),
            'launch': [round(float(v), 3) for v in point], 'velocity': [round(float(v), 3) for v in exit_velocity], 't0_ms': int(round(launch_ms)),
            'first_seen_ms': int(round((fit['t0_s'] - offset_s) * 1000)),
            'score_ms': score_ms,
            'source_continuous': continuous_source_to(track, offset_s + score_ms / 1000 if score_ms is not None else float(track.obs[-1][1])),
            'distance_to_hub_m': None if hub_xy is None else round(math.hypot(point[0] - hub_xy[0], point[1] - hub_xy[1]), 2),
            'apex_m': round(verdict['apex_m'], 2), 'flight_s': round(fit['duration_s'], 2), 'result': result, 'target': target,
            'entry': entry, 'ball_track': ball_track,
            'miss_m': None if verdict['miss_m'] is None else round(verdict['miss_m'], 2), 'cross_xy': verdict.get('cross_xy'),
            'rms_px': round(fit['rms_px'], 2), 'outlier_fraction': round(fit['outlier_fraction'], 2), 'observations': fit['observations'], 'confidence': shot_confidence(fit),
            'attribution_m': None, 'attribution': 'no_result', 'attribution_margin_m': None,
            # area growth well above one ball = a merged blob, i.e. several balls tracked as one
            'merged': bool(np.median(areas) > 2.2 * np.percentile(areas, 10) and np.median(areas) > 30),
        })
    shots.sort(key=lambda s: s['t_ms'])
    attribute_shots(shots, robot_tracks)
    return (shots, tracks, stats) if return_tracks else (shots, stats)


CHUNK_TARGET_S = 8.0
CHUNK_OVERLAP_S = 1.0


def fixed_chunks_mode():
    """Counter v2: chunk boundaries depend only on the time window (ceil(duration / 8 s) chunks), never on the number
    of workers, so a match counts the same on any machine. Before, N = min(workers, ...) moved the boundaries and the
    same video gave different counts with 12 and 14 workers (2026cc qm1 blue 366 vs 355). VISION_FUEL_FIXED_CHUNKS=0
    restores the old plan."""
    return os.environ.get('VISION_FUEL_FIXED_CHUNKS', '1').strip().lower() not in ('0', 'false', 'no')


def plan_fuel_chunks(start_s, end_s, workers=None, target_s=CHUNK_TARGET_S, overlap_s=CHUNK_OVERLAP_S, fixed=None):
    """Split [start_s, end_s] into N equal slices, each extended by `overlap_s` on both sides.

    N = ceil(duration / 8 s) with fixed chunks (the default, see `fixed_chunks_mode`), else
    min(cpu_count, ceil(duration / 8 s)) (or `workers` when given). Returns a list of
    (core_start_s, core_end_s, ext_start_s, ext_end_s) tuples, clamped at 0 and end_s.
    """
    start_s, end_s = float(start_s), float(end_s)
    duration = max(0.0, end_s - start_s)
    if duration <= 0.0:
        return []
    if workers is None:
        workers = _env_fuel_workers()
    by_time = math.ceil(duration / max(1e-3, float(target_s)))
    if fixed if fixed is not None else fixed_chunks_mode():
        n_chunks = max(1, by_time)
    else:
        cpus = max(1, (int(workers) if workers is not None else (os.cpu_count() or 1)))
        n_chunks = max(1, min(cpus, by_time))
    step = duration / n_chunks
    chunks = []
    for i in range(n_chunks):
        c0 = start_s + i * step
        c1 = end_s if i == n_chunks - 1 else start_s + (i + 1) * step
        e0 = max(0.0, start_s if i == 0 else max(start_s, c0 - overlap_s))
        e1 = end_s if i == n_chunks - 1 else min(end_s, c1 + overlap_s)
        chunks.append((c0, c1, e0, e1))
    return chunks


def seed_chunk_rng(core_start_s):
    """Seed OpenCV's and NumPy's random generators from the chunk's start time.

    RANSAC draws from OpenCV's generator, whose state otherwise carries over between chunks that happen to share a
    pool worker, so the same chunk could give slightly different tracks from run to run."""
    seed = 971_000 + int(round(float(core_start_s) * 1000))
    cv2.setRNGSeed(seed)
    np.random.seed(seed % (2**32))


def _track_chunk_worker(args):
    """Run the frame loop of one video chunk in a worker process using only picklable inputs."""
    video_path, core_start_s, core_end_s, ext_start_s, ext_end_s, fps, h_ref_to_field, ref_frame, mask_ref = args[:9]
    near_ref, opening_mode = (args[9], args[10]) if len(args) > 9 else (None, None)
    from pipeline import iter_frames
    cv2.setNumThreads(1)
    seed_chunk_rng(core_start_s)
    h_ref_to_field = np.asarray(h_ref_to_field, float)
    ref_frame = np.asarray(ref_frame, np.uint8)
    mask_ref = np.asarray(mask_ref, np.uint8) if mask_ref is not None else None
    height, width = ref_frame.shape[:2]
    motion = CameraMotion(min_inliers=25)
    motion.start(cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY), mask_ref)
    ref_camera = camera_from_pixel_homography(h_ref_to_field, width, height)
    if ref_camera is None:
        raise RuntimeError('could not recover a 3D camera from the field calibration (is the view calibrated on a clear frame?)')
    bx, by, bw, bh = cv2.boundingRect(mask_ref) if mask_ref is not None else (0, 0, width, height)
    t_crop = np.array([[1.0, 0.0, -bx], [0.0, 1.0, -by], [0.0, 0.0, 1.0]])
    high_rate = high_rate_mode()
    detector, tracker = MovingFuelDetector(), (StreamBallTracker if high_rate else BallTracker)(min_obs=4)
    detector.camera = ref_camera if high_rate else None
    detector.ref_camera = ref_camera
    near_ref = np.asarray(near_ref, np.uint8) if near_ref is not None else None
    detector.opening_mode = opening_mode if near_ref is not None else None
    near_crop_static = near_ref[by:by + bh, bx:bx + bw] if near_ref is not None else None
    identity = np.eye(3)
    prev_gray, prev_h = None, identity
    frame_records = []          # (local_idx, t_s, in_core, is_static, h_cur_to_ref)
    blobs_core = 0
    local_idx = -1
    last_t = None
    for t, img in iter_frames(video_path, ext_start_s, ext_end_s, fps):
        if high_rate and last_t is not None and t - last_t < 1.0 / TRACK_MAX_FPS - 1e-3:
            continue
        last_t = t
        local_idx += 1
        in_start = True if core_start_s <= ext_start_s + 1e-9 else (t >= core_start_s)
        in_end = (t <= core_end_s) if core_end_s >= ext_end_s - 1e-9 else (t < core_end_s)
        in_core = in_start and in_end
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        crop = img[by:by + bh, bx:bx + bw]
        crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        yellow = cv2.inRange(crop_hsv, detector.low, detector.high)       # = detector.yellow_mask(crop)

        def feature_mask():
            usable = np.full((height, width), 255, np.uint8)
            usable[by:by + bh, bx:bx + bw] = 255 - cv2.dilate(yellow, detector.kernel, iterations=4)
            return usable

        h_cur_to_ref, ok = motion.update(gray, prefer_reference=True, ref_mask=mask_ref, mask_factory=feature_mask)
        if not ok:
            frame_records.append((local_idx, float(t), in_core, False, None))
            prev_gray = None
            continue
        static = bool(np.array_equal(h_cur_to_ref, identity))
        if prev_gray is None:
            warped = None
        elif (static and prev_h is identity) or np.array_equal(h_cur_to_ref, prev_h):
            warped = prev_gray[by:by + bh, bx:bx + bw]
        else:
            warped = cv2.warpPerspective(prev_gray, t_crop @ np.linalg.inv(h_cur_to_ref) @ prev_h, (bw, bh))
        if near_ref is not None:
            detector.near_mask = near_crop_static if static else warp_mask_to_frame(near_ref, h_cur_to_ref, width, height)[by:by + bh, bx:bx + bw]
        blobs = detector.detect(t, crop, warped, offset=(bx, by), gray=gray[by:by + bh, bx:bx + bw], yellow=yellow, hsv=crop_hsv) if warped is not None else []
        if blobs:
            if static:
                for b in blobs:
                    b.ru, b.rv = b.u, b.v
            else:
                ref_pts = cv2.perspectiveTransform(np.array([[[b.u, b.v] for b in blobs]], np.float32), h_cur_to_ref)[0]
                for b, (ru, rv) in zip(blobs, ref_pts):
                    b.ru, b.rv = float(ru), float(rv)
            if mask_ref is not None:
                blobs = [b for b in blobs if 0 <= int(b.rv) < height and 0 <= int(b.ru) < width and mask_ref[int(b.rv), int(b.ru)]]
        if in_core:
            blobs_core += len(blobs)
        tracker.step(local_idx, float(t), blobs)
        frame_records.append((local_idx, float(t), in_core, static, None if static else h_cur_to_ref.tolist()))
        prev_gray, prev_h = gray, (identity if static else h_cur_to_ref)
    # Keep exactly-three-observation tracks for local exits without changing the
    # existing >=4-observation cohort used by HUB entries and flight fitting.
    short = [[list(o) for o in tr.obs] for tr in tracker.done + tracker.active if len(tr.obs) == 3]
    raw = [[list(o) for o in tr.obs] for tr in tracker.finish()]
    return {'frames': frame_records, 'tracks': raw, 'short_source_tracks': short, 'blobs_core': blobs_core}


def _median_of(values):
    """np.median of a short list of floats without the numpy round trip (same value: middle, or mean of the two middle)."""
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    return float(ordered[mid]) if n % 2 else float((ordered[mid - 1] + ordered[mid]) / 2)


def deduplicate_overlap_tracks(tracks, fps=60.0, max_dist_px=6.0):
    """Merge ball tracks from overlapping video chunks when they observe the same ball at the same time.

    Two tracks that overlap in time and whose shared observations lie within `max_dist_px` in reference
    frame pixels are merged into a single BallTrack, keeping one observation per frame timestamp.
    """
    if not tracks:
        return []
    fps = max(1.0, float(fps))
    dt_tol = 0.6 / fps
    max_interp_gap = 4.5 / fps
    bbox_pad = max_dist_px + 8.0
    abort_dist = max(24.0, 4.0 * max_dist_px)
    valid = [BallTrack(sorted(tr.obs, key=lambda o: o[1]), source_breaks=getattr(tr, 'source_breaks', None))
             for tr in tracks if tr and getattr(tr, 'obs', None)]
    ordered = sorted(valid, key=lambda tr: (tr.obs[0][1], -len(tr.obs)))

    def _bbox(obs):
        us = [o[5] for o in obs]
        vs = [o[6] for o in obs]
        return min(us), max(us), min(vs), max(vs)

    def _shared_dists(a_obs, b_obs, a_times=None, b_times=None):
        b_t0, b_t1 = b_obs[0][1], b_obs[-1][1]
        bn = len(b_obs)
        bi = 0
        seek = b_times is not None
        dists = []
        # a_obs is sorted by time: start at its first observation inside b's span (the ones before it are skipped anyway).
        first = bisect.bisect_left(a_times, b_t0 - dt_tol) if a_times is not None else 0
        for ao in itertools.islice(a_obs, first, None):
            at = ao[1]
            if at < b_t0 - dt_tol:
                continue
            if at > b_t1 + dt_tol:
                break
            if seek:
                # The walk below passes every b observation at or before `at`: jump to the last of them first.
                bi, seek = max(0, bisect.bisect_right(b_times, at) - 1), False
            while bi + 1 < bn and abs(b_obs[bi + 1][1] - at) <= abs(b_obs[bi][1] - at):
                bi += 1
            bt = b_obs[bi][1]
            if abs(bt - at) <= dt_tol:
                d_ref = math.hypot(ao[5] - b_obs[bi][5], ao[6] - b_obs[bi][6])
                d_uv = math.hypot(ao[2] - b_obs[bi][2], ao[3] - b_obs[bi][3])
                d = min(d_ref, d_uv)
                if not dists and d > abort_dist:
                    return []
                dists.append(d)
            elif b_t0 <= at <= b_t1:
                k0 = bi - 1 if bt > at and bi > 0 else bi
                k1 = k0 + 1
                if 0 <= k0 and k1 < bn:
                    t0, t1 = b_obs[k0][1], b_obs[k1][1]
                    if t0 <= at <= t1 and (t1 - t0) <= max_interp_gap and (t1 - t0) > 1e-9:
                        f = (at - t0) / (t1 - t0)
                        ru = b_obs[k0][5] + f * (b_obs[k1][5] - b_obs[k0][5])
                        rv = b_obs[k0][6] + f * (b_obs[k1][6] - b_obs[k0][6])
                        d = math.hypot(ao[5] - ru, ao[6] - rv)
                        if not dists and d > abort_dist:
                            return []
                        dists.append(d)
        return dists

    merged = []
    merged_times = []
    meta = []        # (e_t0, e_t1, min_u, max_u, min_v, max_v)
    active_idxs = []
    for cand in ordered:
        c_obs = cand.obs
        c_times = [o[1] for o in c_obs]
        c_t0, c_t1 = c_obs[0][1], c_obs[-1][1]
        c_min_u, c_max_u, c_min_v, c_max_v = _bbox(c_obs)
        active_idxs = [m_idx for m_idx in active_idxs if meta[m_idx][1] >= c_t0 - dt_tol]
        matched_idx = None
        best_med = None
        for m_idx in active_idxs:
            e_t0, e_t1, e_min_u, e_max_u, e_min_v, e_max_v = meta[m_idx]
            if c_t1 < e_t0 - dt_tol:
                continue
            if c_max_u < e_min_u - bbox_pad or c_min_u > e_max_u + bbox_pad:
                continue
            if c_max_v < e_min_v - bbox_pad or c_min_v > e_max_v + bbox_pad:
                continue
            existing = merged[m_idx]
            shared_dists = _shared_dists(c_obs, existing.obs, c_times, merged_times[m_idx])
            if not shared_dists and len(existing.obs) <= 12:
                shared_dists = _shared_dists(existing.obs, c_obs, merged_times[m_idx], c_times)
            if shared_dists:
                med = _median_of(shared_dists)
                if med <= max_dist_px and (best_med is None or med < best_med):
                    best_med = med
                    matched_idx = m_idx
        if matched_idx is None:
            new_idx = len(merged)
            merged.append(BallTrack(list(c_obs), source_breaks=cand.source_breaks))
            merged_times.append(c_times)
            meta.append((c_t0, c_t1, c_min_u, c_max_u, c_min_v, c_max_v))
            active_idxs.append(new_idx)
        else:
            combined = list(merged[matched_idx].obs)
            for co in c_obs:
                ct = co[1]
                if not any(abs(eo[1] - ct) <= dt_tol for eo in combined):
                    combined.append(co)
            combined.sort(key=lambda o: o[1])
            before_breaks = merged[matched_idx].source_breaks
            breaks = before_breaks + cand.source_breaks if isinstance(before_breaks, list) and isinstance(cand.source_breaks, list) else None
            merged[matched_idx] = BallTrack(combined, source_breaks=breaks)
            merged_times[matched_idx] = [o[1] for o in combined]
            e_t0, e_t1, e_min_u, e_max_u, e_min_v, e_max_v = meta[matched_idx]
            meta[matched_idx] = (
                min(e_t0, c_t0), max(e_t1, c_t1),
                min(e_min_u, c_min_u), max(e_max_u, c_max_u),
                min(e_min_v, c_min_v), max(e_max_v, c_max_v),
            )
    return merged


def run_fuel_pass_parallel(video_path, start_s, end_s, fps, h_ref_to_field, ref_frame, hubs,
                           robot_tracks=(), offset_s=0.0, on_progress=None, mask_ref=None, workers=None, parallel=True, tracks_only=False, return_tracks=False,
                           opening_region_ref=None, opening_mode=None):
    """Parallel multi-core entry point for the fuel ball tracker over `[start_s, end_s]` of `video_path`.

    Signature for `run_match`:
        run_fuel_pass_parallel(video_path, start_s, end_s, fps, h_ref_to_field, ref_frame, hubs,
                               robot_tracks=(), offset_s=0.0, on_progress=None, mask_ref=None,
                               workers=None, parallel=True, return_tracks=False) -> (shots, stats) or (shots, tracks, stats)
    With tracks_only=True it stops after linking and returns (ball tracks, stats) for the ball counter.
    Otherwise `stats` is a `FuelStats(dict)` carrying `.tracks` / `.ball_tracks` (`list[BallTrack]`).

    Splits `[start_s, end_s]` into `N = min(cpu_count, ceil(duration / 8 s))` chunks, each extended by
    1.0 s of overlap on both sides, runs the per-frame detection/registration/tracking loop of each chunk
    in a `ProcessPoolExecutor` (each worker opens `video_path` with `iter_frames` and starts `CameraMotion`
    from `ref_frame` with picklable numpy/list arguments), deduplicates tracks across the chunk overlaps,
    and runs `link_tracks` / `fit_tracks` / shooter attribution once on the merged set. If `parallel=False`,
    `N == 1`, or the process pool is unavailable, falls back cleanly to `run_fuel_pass`.
    """
    from pipeline import iter_frames
    start_s, end_s, fps = float(start_s), float(end_s), tracking_fps(fps)
    if workers is None:
        workers = _env_fuel_workers()
    chunks = plan_fuel_chunks(start_s, end_s, workers=workers) if parallel else []
    if not parallel or len(chunks) <= 1:
        return run_fuel_pass(
            iter_frames(video_path, start_s, end_s, fps),
            h_ref_to_field, ref_frame, hubs,
            robot_tracks=robot_tracks, offset_s=offset_s, on_progress=on_progress, mask_ref=mask_ref, workers=workers, tracks_only=tracks_only, return_tracks=return_tracks,
            opening_region_ref=opening_region_ref, opening_mode=opening_mode,
        )

    height, width = ref_frame.shape[:2]
    ref_camera = camera_from_pixel_homography(h_ref_to_field, width, height)
    if ref_camera is None:
        raise RuntimeError('could not recover a 3D camera from the field calibration (is the view calibrated on a clear frame?)')

    h_arr = np.asarray(h_ref_to_field, float)
    ref_arr = np.asarray(ref_frame, np.uint8)
    mask_arr = np.asarray(mask_ref, np.uint8) if mask_ref is not None else None
    near_arr = np.asarray(opening_region_ref, np.uint8) if (opening_region_ref is not None and opening_mode) else None
    worker_args = [
        (str(video_path), c0, c1, e0, e1, fps, h_arr, ref_arr, mask_arr, near_arr, opening_mode if near_arr is not None else None)
        for (c0, c1, e0, e1) in chunks
    ]
    pool_workers = min(len(chunks), max(1, int(workers) if workers is not None else (os.cpu_count() or 1)))
    try:
        if pool_workers == 1:
            chunk_outputs = [_track_chunk_worker(a) for a in worker_args]   # same chunks, same result, no pool
        else:
            with _process_pool(pool_workers) as pool:
                # Report each chunk as it finishes (in any order), so a long whole-video pass is never silent long enough
                # for the worker to take it for stalled; the results are still assembled in chunk order.
                futures = [pool.submit(_track_chunk_worker, a) for a in worker_args]
                for done, _ in enumerate(as_completed(futures), 1):
                    log(f'tracked {done}/{len(futures)} chunks')
                chunk_outputs = [f.result() for f in futures]
    except (BrokenProcessPool, OSError, RuntimeError):
        return run_fuel_pass(
            iter_frames(video_path, start_s, end_s, fps),
            h_ref_to_field, ref_frame, hubs,
            robot_tracks=robot_tracks, offset_s=offset_s, on_progress=on_progress, mask_ref=mask_ref, workers=workers, tracks_only=tracks_only, return_tracks=return_tracks,
            opening_region_ref=opening_region_ref, opening_mode=opening_mode,
        )

    # Merge frame records and cameras across chunks (deduplicating overlap frames by timestamp)
    import bisect
    dt_tol = 0.6 / max(1.0, fps)
    global_frames = []          # sorted list of (t_s, is_static, h_cur_to_ref, in_core)
    global_times = []
    stats = {
        'frames': 0, 'unregistered': 0, 'unregistered_spans': [], 'static_frames': 0, 'blobs': 0,
        'tracks': 0, 'linked': 0, 'prefiltered_out': 0, 'fitted': 0, 'rejected_fit': 0,
        'ref_focal_px': round(ref_camera.f, 1), 'chunks': len(chunks),
        'opening_detection': opening_mode if near_arr is not None else None,
        'counter_options': counter_options(),
    }
    for out in chunk_outputs:
        stats['blobs'] += int(out['blobs_core'])
        for idx_local, t_s, in_core, is_static, h_list in out['frames']:
            if in_core:
                stats['frames'] += 1
                if h_list is None and not is_static:
                    stats['unregistered'] += 1
                    note_unregistered(stats, t_s)
                elif is_static:
                    stats['static_frames'] += 1
            if h_list is None and not is_static:
                continue
            if on_progress and in_core:
                on_progress(t_s)
            h_mat = None if is_static else np.asarray(h_list, float)
            if not global_frames or t_s - global_times[-1] > dt_tol:
                global_frames.append((t_s, is_static, h_mat, in_core))
                global_times.append(t_s)
            else:
                pos = bisect.bisect_left(global_times, t_s)
                near = None
                for k in (pos - 1, pos):
                    if 0 <= k < len(global_times) and abs(global_times[k] - t_s) <= dt_tol:
                        near = k
                        break
                if near is None:
                    global_frames.insert(pos, (t_s, is_static, h_mat, in_core))
                    global_times.insert(pos, t_s)
                elif in_core and not global_frames[near][3]:
                    global_frames[near] = (t_s, is_static, h_mat, in_core)
                    global_times[near] = t_s

    if not global_frames:
        stats = FuelStats(stats, tracks=[])
        return ([], [], stats) if (return_tracks and not tracks_only) else ([], stats)

    frame_times = np.array(global_times, float)
    cameras = {}
    if not tracks_only:
        all_static = True
        for g_idx, (_, is_static, h_cur_to_ref, *_) in enumerate(global_frames):
            if is_static:
                cameras[g_idx] = ref_camera
            else:
                all_static = False
                cameras[g_idx] = camera_from_pixel_homography(h_arr @ h_cur_to_ref, width, height, f_prior=ref_camera.f) or ref_camera
        if cameras and not all_static:
            order = sorted(cameras)
            cameras = dict(zip(order, smooth_cameras([cameras[i] for i in order], window=11)))

    # Remap each chunk track's frame index to the global frame index and deduplicate overlap tracks
    chunk_tracks, short_source_tracks = [], []
    n_global = len(frame_times)
    for out in chunk_outputs:
        for is_short, raw_obs in ([(False, row) for row in out['tracks']]
                                  + [(True, row) for row in out.get('short_source_tracks', [])]):
            if not raw_obs:
                continue
            obs_times = np.array([float(o[1]) for o in raw_obs], float)
            g_indices = np.searchsorted(frame_times, obs_times)
            g_indices = np.clip(g_indices, 0, n_global - 1)
            prev_indices = np.maximum(0, g_indices - 1)
            choose_prev = np.abs(frame_times[prev_indices] - obs_times) < np.abs(frame_times[g_indices] - obs_times)
            g_indices = np.where(choose_prev, prev_indices, g_indices)
            remapped = [
                (int(g_idx), float(o[1]), float(o[2]), float(o[3]), float(o[4]), float(o[5]), float(o[6]))
                for g_idx, o in zip(g_indices, raw_obs)
            ]
            (short_source_tracks if is_short else chunk_tracks).append(BallTrack(remapped))

    raw = deduplicate_overlap_tracks(chunk_tracks, fps=fps)
    stats['tracks'] = len(raw)
    tracks = link_tracks(raw)
    stats['linked'] = len(raw) - len(tracks)
    stats = FuelStats(stats, tracks=tracks, source_tracks=tracks + deduplicate_overlap_tracks(short_source_tracks, fps=fps))
    if tracks_only:
        return tracks, stats
    import fuel_count as fc
    polygons = fc.hub_polygons(ref_camera, hubs, pad=fc.HUB_PAD) if hubs else {}
    all_hub_entries = fc.hub_entries(tracks, polygons, offset_s) if polygons else []
    vertical_px_per_m = float(np.linalg.norm(ref_camera.project([[8.27, 4.0, 1.0]])[0][0] - ref_camera.project([[8.27, 4.0, 0.0]])[0][0]))
    candidates = []
    for track in tracks:
        passes, _ = quick_ballistic_check(track, vertical_px_per_m)
        if passes and plausible_flight(track):
            candidates.append(track)
    stats['prefiltered_out'] = len(tracks) - len(candidates)
    tasks = [([o[1] for o in tr.obs], [(o[2], o[3]) for o in tr.obs], [cameras[o[0]] for o in tr.obs], [math.sqrt(o[4] / math.pi) for o in tr.obs]) for tr in candidates]
    fits = fit_tracks(tasks, workers)
    shots = []
    staged = []
    for track, fit in zip(candidates, fits):
        times = [o[1] for o in track.obs]
        if fit is None or fit['rms_px'] > 3.5 or fit['outlier_fraction'] > 0.25:
            stats['rejected_fit'] += 1
            stats.setdefault('rejections', []).append({'t': round(times[0], 2), 'obs': len(times), 'fit': None if fit is None else round(fit['rms_px'], 2)})
            continue
        verdict = classify_flight(fit, hubs)
        if verdict['apex_m'] < MIN_SHOT_APEX:
            stats['ground_balls'] = stats.get('ground_balls', 0) + 1
            continue
        stats['fitted'] += 1
        point, exit_velocity, back_s = state_at_exit(fit)
        staged.append((((fit['t0_s'] - offset_s + back_s) * 1000.0), track, fit, verdict, point, exit_velocity))
    staged.sort(key=lambda item: item[0])
    for launch_ms, track, fit, verdict, point, exit_velocity in staged:
        launch_speed = float(np.linalg.norm(exit_velocity))
        horizontal = math.hypot(exit_velocity[0], exit_velocity[1])
        areas = [o[4] for o in track.obs]
        entry = evaluate_hub_entry(track, fit, hubs, cameras=cameras, verdict=verdict, offset_s=offset_s, hub_entries_list=all_hub_entries, polygons=polygons)
        result = ENTRY_RESULT.get(entry['decision'], verdict['result'])
        ball_track = build_ball_track(track, width, height, offset_s=offset_s)
        target = verdict['target']
        if hubs and cameras and track and getattr(track, 'obs', None):
            last_obs = max(track.obs, key=lambda o: o[1])
            cam_last = cameras.get(last_obs[0], ref_camera)
            sel = _select_hub(track, fit, verdict, hubs, cam_last)
            if sel is not None:
                target = sel[0]
        hub_xy = hubs.get(target) if target else None
        score_ms = None
        if result == 'scored':
            score_ms = entry.get('t_ms')
            if score_ms is None and verdict.get('cross_t_s') is not None:
                score_ms = int(round((fit['t0_s'] - offset_s + verdict['cross_t_s']) * 1000))
            if score_ms is None and ball_track:
                score_ms = ball_track[-1]['t']
            entry['t_ms'] = score_ms
        shots.append({
            't_ms': int(round(launch_ms)), 'team_key': None, 'track_key': None, 'alliance': None,
            'speed_mps': round(launch_speed, 2), 'speed_sigma_mps': None if fit['speed_sigma'] is None else round(fit['speed_sigma'], 2),
            'elevation_deg': round(math.degrees(math.atan2(exit_velocity[2], horizontal)), 1), 'azimuth_deg': round(math.degrees(math.atan2(exit_velocity[1], exit_velocity[0])), 1),
            'launch': [round(float(v), 3) for v in point], 'velocity': [round(float(v), 3) for v in exit_velocity], 't0_ms': int(round(launch_ms)),
            'first_seen_ms': int(round((fit['t0_s'] - offset_s) * 1000)),
            'score_ms': score_ms,
            'source_continuous': continuous_source_to(track, offset_s + score_ms / 1000 if score_ms is not None else float(track.obs[-1][1])),
            'distance_to_hub_m': None if hub_xy is None else round(math.hypot(point[0] - hub_xy[0], point[1] - hub_xy[1]), 2),
            'apex_m': round(verdict['apex_m'], 2), 'flight_s': round(fit['duration_s'], 2), 'result': result, 'target': target,
            'entry': entry, 'ball_track': ball_track,
            'miss_m': None if verdict['miss_m'] is None else round(verdict['miss_m'], 2), 'cross_xy': verdict.get('cross_xy'),
            'rms_px': round(fit['rms_px'], 2), 'outlier_fraction': round(fit['outlier_fraction'], 2), 'observations': fit['observations'], 'confidence': shot_confidence(fit),
            'attribution_m': None, 'attribution': 'no_result', 'attribution_margin_m': None,
            'merged': bool(np.median(areas) > 2.2 * np.percentile(areas, 10) and np.median(areas) > 30),
        })
    shots.sort(key=lambda s: s['t_ms'])
    attribute_shots(shots, robot_tracks)
    return (shots, tracks, stats) if return_tracks else (shots, stats)


def hubs_from_field():
    return {'blue_hub': (FIELD['hub']['x'], FIELD['hub']['y']), 'red_hub': (FIELD['length'] - FIELD['hub']['x'], FIELD['width'] - FIELD['hub']['y'])}


def shots_to_observations(shots, roster_keys, auto_end_ms, view_id=None, run_id=None, emit_scored=True):
    """Tracker shots -> vision_observations rows (match milliseconds).

    emit_scored=False when fuel_counter emits the fuel_scored rows: the shots then carry their entry call on
    fuel_attempt only, so balls are not counted twice.

    One `fuel_attempt` per shot carrying every measured detail in `value`, and a `fuel_scored` per shot
    judged to have dropped into the hub (timed at the hub crossing), so existing totals and release logic
    keep working while the Fuel tab reads the detail. `source` marks them as tracker output, which is how
    the app knows to prefer them over language-model guesses.
    """
    rows = []
    for n, shot in enumerate(shots):
        alliance = shot['alliance']
        if shot['team_key'] is None and shot.get('target'):
            alliance = 'blue' if shot['target'] == 'blue_hub' else 'red'
        detail = {k: shot[k] for k in ('result', 'entry', 'ball_track', 'score_ms', 'speed_mps', 'speed_sigma_mps', 'elevation_deg', 'azimuth_deg', 'distance_to_hub_m', 'apex_m', 'flight_s',
                                       'target', 'miss_m', 'launch', 'velocity', 'rms_px', 'outlier_fraction', 'observations', 'merged', 'attribution_m', 'attribution', 'attribution_margin_m', 'confidence') if k in shot}
        detail.update({'count': 1, 'shot': n})
        base = {'team_key': shot['team_key'] if shot['team_key'] in roster_keys else None, 'track_key': shot.get('track_key'), 'alliance': alliance, 'view_id': view_id, 'vision_run_id': run_id,
                'source': 'fuel_ballistic', 'confidence': float(shot['confidence'])}
        evidence = {'rms_px': shot['rms_px'], 'method': 'ballistic fit with drag from a camera recovered from the field calibration'}
        rows.append({**base, 'observation_type': 'fuel_attempt', 'phase': 'auto' if shot['t_ms'] <= auto_end_ms else 'teleop', 'value': detail,
                     'started_ms': max(0, shot['t_ms']), 'ended_ms': max(0, shot['t_ms']) + int(round(shot['flight_s'] * 1000)), 'evidence': evidence})
        if emit_scored and shot['result'] == 'scored':
            t = max(0, shot['score_ms'] if shot.get('score_ms') is not None else shot['t_ms'])
            rows.append({**base, 'observation_type': 'fuel_scored', 'phase': 'auto' if shot['t_ms'] <= auto_end_ms and t <= auto_end_ms + AUTO_GRACE_MS else 'teleop', 'value': {**detail, 'count': 1},
                         'started_ms': t, 'ended_ms': t + 300, 'evidence': evidence})
    return rows
