"""Robot detectors behind one interface: detect(frames) -> list of Detection per frame.

  ColorBumperDetector  - classical: red / blue bumper blobs by HSV. Needs no training, gives oriented
                         boxes, and is the baseline the learned detectors must beat. It also runs the
                         synthetic end-to-end test.
  UltralyticsDetector  - YOLO26 / YOLO11 weights (axis-aligned or OBB) fine-tuned on FRC bumpers.
  GeminiBoxDetector    - Gemini boxes on still frames (slow and approximate; the no-GPU fallback).
"""
import hashlib
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from vlm import BudgetExceeded


def verify_weights_path(weights, expected_sha256=None):
    """Ensure `weights` is an existing local `.pt` checkpoint (never a URL or remote model name) and optionally verify SHA-256."""
    raw = str(weights or '').strip()
    if not raw or '://' in raw:
        raise ValueError(f'Local .pt weights file required (no remote downloads): {weights!r}')
    weights_path = Path(raw).expanduser().resolve()
    if not weights_path.is_file() or weights_path.suffix.lower() != '.pt':
        raise ValueError(f'Local .pt weights file required (no remote downloads): {weights!r}')
    expected = str(expected_sha256 if expected_sha256 is not None else os.environ.get('VISION_MODEL_SHA256', '')).strip().lower()
    if expected:
        actual = hashlib.sha256(weights_path.read_bytes()).hexdigest().lower()
        if actual != expected:
            raise ValueError(f'Model weights SHA-256 mismatch for {weights_path}: expected {expected}, got {actual}')
    os.environ.setdefault('YOLO_OFFLINE', 'True')
    return weights_path


@dataclass
class Detection:
    t: float                 # video seconds
    box: tuple               # (x1, y1, x2, y2) pixels, axis aligned
    alliance: str = None     # 'red' | 'blue' | None
    team: str = None         # bare team number if the detector read one
    angle: float = None      # radians, image-plane orientation of the long side (None if unknown)
    confidence: float = 0.5
    extra: dict = field(default_factory=dict)


class ColorBumperDetector:
    """Find robots from the red or blue bumper band.

    Bumpers are saturated, uniformly coloured and low on the robot, which makes colour a surprisingly
    strong cue on a green carpet. Weak against: red/blue field graphics, alliance-coloured clothing in
    the stands, and robots with black bumpers. Fine as a baseline and as a fallback, not as the
    production detector.
    """
    RED = [((0, 120, 80), (8, 255, 255)), ((172, 120, 80), (180, 255, 255))]
    BLUE = [((100, 140, 60), (130, 255, 255))]

    def __init__(self, min_area=250, max_area_frac=0.08, kernel=5):
        self.min_area = min_area
        self.max_area_frac = max_area_frac
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))

    def _mask(self, hsv, ranges):
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for low, high in ranges:
            mask |= cv2.inRange(hsv, np.array(low), np.array(high))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)

    def detect_frame(self, t, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        limit = self.max_area_frac * bgr.shape[0] * bgr.shape[1]
        out = []
        for alliance, ranges in (('red', self.RED), ('blue', self.BLUE)):
            contours, _ = cv2.findContours(self._mask(hsv, ranges), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if not self.min_area <= area <= limit:
                    continue
                (cx, cy), (w, h), angle = cv2.minAreaRect(contour)
                x, y, bw, bh = cv2.boundingRect(contour)
                fill = area / max(1.0, w * h)
                theta = math.radians(angle if w >= h else angle + 90)
                out.append(Detection(t, (x, y, x + bw, y + bh), alliance, None, theta, round(min(0.95, 0.4 + fill * 0.55), 2)))
        return out

    def detect(self, frames):
        return [self.detect_frame(t, img) for t, img in frames]


class UltralyticsDetector:
    """Wraps an Ultralytics model. Class names containing 'red' / 'blue' set the alliance."""

    def __init__(self, weights, conf=0.35, imgsz=1280, device=None, batch=16):
        weights_path = verify_weights_path(weights)
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise SystemExit('pip install ultralytics to use the YOLO detector') from error
        self.model = YOLO(str(weights_path))
        self.conf, self.imgsz, self.device, self.batch = conf, imgsz, device, batch

    def detect(self, frames):
        out = []
        for i in range(0, len(frames), self.batch):
            chunk = frames[i:i + self.batch]
            results = self.model.predict([img for _, img in chunk], conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)
            for (t, _), result in zip(chunk, results):
                names = result.names
                dets = []
                if getattr(result, 'obb', None) is not None and len(result.obb):
                    for xywhr, cls, conf, xyxy in zip(result.obb.xywhr.cpu().numpy(), result.obb.cls.cpu().numpy(), result.obb.conf.cpu().numpy(), result.obb.xyxy.cpu().numpy()):
                        dets.append(self._det(t, names[int(cls)], float(conf), xyxy, float(xywhr[4])))
                elif result.boxes is not None:
                    for xyxy, cls, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                        dets.append(self._det(t, names[int(cls)], float(conf), xyxy, None))
                out.append(suppress_frame_detections(dets))
        return out

    @staticmethod
    def _det(t, name, conf, xyxy, angle):
        lowered = name.lower()
        alliance = 'red' if 'red' in lowered else 'blue' if 'blue' in lowered else None
        return Detection(t, tuple(float(v) for v in xyxy), alliance, None, angle, conf)


FRAME_SCHEMA = {'type': 'object', 'properties': {'robots': {'type': 'array', 'items': {'type': 'object', 'properties': {
    't': {'type': 'number'}, 'team': {'type': 'string'}, 'box_2d': {'type': 'array', 'items': {'type': 'integer'}}, 'confidence': {'type': 'number'}},
    'required': ['t', 'team', 'box_2d']}}}, 'required': ['robots']}

BUMPER_RED_HSV = [((0, 110, 70), (11, 255, 255)), ((169, 110, 70), (180, 255, 255))]
BUMPER_BLUE_HSV = [((98, 120, 55), (132, 255, 255))]


def valid_box_2d(box, min_span=8.0, max_area=450_000.0):
    """True when `box = [y1, x1, y2, x2]` in 0..1000 coordinates has finite bounds, positive span, and bounded area."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return False
    vals = []
    for v in box:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return False
        fv = float(v)
        if not math.isfinite(fv) or fv < 0.0 or fv > 1000.0:
            return False
        vals.append(fv)
    y1, x1, y2, x2 = vals
    dx = x2 - x1
    dy = y2 - y1
    return dx >= min_span and dy >= min_span and (dx * dy) <= max_area


def _box_area(box):
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def _box_intersection(a, b):
    return max(0.0, min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0]))) * max(0.0, min(float(a[3]), float(b[3])) - max(float(a[1]), float(b[1])))


def box_iou(a, b):
    inter = _box_intersection(a, b)
    if inter <= 0.0:
        return 0.0
    union = _box_area(a) + _box_area(b) - inter
    return inter / union if union > 0.0 else 0.0


def suppress_frame_detections(dets, iou_threshold=0.55):
    """Remove enclosing two-robot composite boxes and suppress high-IoU same-alliance duplicate boxes."""
    dets = list(dets or [])
    if len(dets) <= 1:
        return dets
    remaining = []
    for idx, d in enumerate(dets):
        d_area = _box_area(d.box)
        if d_area <= 0.0:
            continue
        inner = [
            m for j, m in enumerate(dets)
            if j != idx and 0.20 * d_area <= _box_area(m.box) <= 0.85 * d_area
            and _box_intersection(d.box, m.box) / max(_box_area(m.box), 1e-9) >= 0.80 - 1e-9
        ]
        d_diag = math.hypot(float(d.box[2]) - float(d.box[0]), float(d.box[3]) - float(d.box[1]))
        is_composite = any(
            box_iou(m1.box, m2.box) < 0.40
            and math.hypot(
                0.5 * (float(m1.box[0]) + float(m1.box[2]) - float(m2.box[0]) - float(m2.box[2])),
                0.5 * (float(m1.box[1]) + float(m1.box[3]) - float(m2.box[1]) - float(m2.box[3])),
            ) >= 0.25 * d_diag - 1e-9
            for i, m1 in enumerate(inner)
            for m2 in inner[i + 1:]
        )
        if not is_composite:
            remaining.append(d)

    kept = []
    for d in sorted(remaining, key=lambda det: (float(det.confidence), _box_area(det.box)), reverse=True):
        twin = next(
            (
                k for k in kept
                if (k.alliance is None or d.alliance is None or k.alliance == d.alliance)
                and box_iou(d.box, k.box) >= iou_threshold - 1e-9
            ),
            None,
        )
        if twin is None:
            kept.append(d)
        else:
            if twin.team is None and d.team is not None and twin.alliance in (None, d.alliance):
                twin.team = d.team
            if twin.alliance is None and d.alliance is not None:
                twin.alliance = d.alliance
    return kept


def _match_chunk_timestamp(raw_t, by_t):
    """Map model-returned timestamp `raw_t` to a frame key in `by_t` within tolerance, or `None` if out of window."""
    if not by_t or raw_t is None:
        return None
    try:
        rt = float(raw_t)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(rt):
        return None
    if len(by_t) == 1:
        k0 = next(iter(by_t))
        return k0 if (abs(k0 - rt) <= 1.0 or abs(rt) <= 1e-6) else None
    sorted_ts = sorted(by_t)
    step = (sorted_ts[-1] - sorted_ts[0]) / (len(sorted_ts) - 1)
    tol = max(0.35, 0.6 * step)
    best = min(by_t, key=lambda k: abs(k - rt))
    if abs(best - rt) <= tol + 1e-9:
        return best
    if sorted_ts[0] > tol:
        rel_t = sorted_ts[0] + rt
        best_rel = min(by_t, key=lambda k: abs(k - rel_t))
        if abs(best_rel - rel_t) <= tol + 1e-9:
            return best_rel
    return None


def _crop_bounds(crop_box, w, h, margin=0.03):
    """Pixel (x0, y0, x1, y1) of a normalised [x0, y0, x1, y1] crop box plus margin, clamped to the frame."""
    if not crop_box:
        return 0, 0, w, h
    x0, y0, x1, y1 = (float(v) for v in crop_box)
    cx0 = int(round(max(0.0, x0 - margin) * w))
    cy0 = int(round(max(0.0, y0 - margin) * h))
    cx1 = int(round(min(1.0, x1 + margin) * w))
    cy1 = int(round(min(1.0, y1 + margin) * h))
    if cx1 <= cx0 or cy1 <= cy0:
        return 0, 0, w, h
    return cx0, cy0, cx1, cy1


def infer_bumper_alliance(bgr, box, min_share=0.6, min_pixels=8):
    """Alliance from red vs blue HSV pixel share in the lower third of `box` (where the bumper sits), or None."""
    if bgr is None or not bgr.size:
        return None
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = box
    bx1, bx2 = max(0, int(round(x1))), min(w, int(round(x2)))
    by1, by2 = max(0, int(round(y1 + (y2 - y1) * 2.0 / 3.0))), min(h, int(round(y2)))
    if bx2 <= bx1 or by2 <= by1:
        return None
    hsv = cv2.cvtColor(bgr[by1:by2, bx1:bx2], cv2.COLOR_BGR2HSV)
    counts = {}
    for color, ranges in (('red', BUMPER_RED_HSV), ('blue', BUMPER_BLUE_HSV)):
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for low, high in ranges:
            mask |= cv2.inRange(hsv, np.array(low), np.array(high))
        counts[color] = int(np.count_nonzero(mask))
    total = counts['red'] + counts['blue']
    if total < min_pixels:
        return None
    for color in ('red', 'blue'):
        if counts[color] / total >= min_share:
            return color
    return None


class GeminiBoxDetector:
    """Gemini finds robots (and reads team numbers off the broadcast overlay) in batches of stills.

    Batches are independent, so they are sent `workers` at a time: a whole match is ~70 calls, and sending them one
    after another made box detection most of a Gemini run's wall-clock time."""

    def __init__(self, client, red, blue, batch=None, workers=8, crop_box=None, max_width=None, jpeg_quality=85, media_resolution=None, min_team_conf=None):
        self.crop_box = tuple(float(v) for v in crop_box) if crop_box else None
        self.client, self.red, self.blue = client, red, blue
        self.batch = batch if batch is not None else (16 if self.crop_box else 10)
        self.workers = max(1, workers)
        self.max_width = int(max_width) if max_width else None
        self.jpeg_quality = int(jpeg_quality)
        self.media_resolution = media_resolution
        self.min_team_conf = float(min_team_conf) if min_team_conf is not None else None
        self.batches = 0
        self.failures = []

    def failure_problem(self, limit=0.3):
        """A message when too many batches failed for the result to mean anything (e.g. the network was down)."""
        if self.batches and len(self.failures) > limit * self.batches:
            return f'Gemini robot detection failed for {len(self.failures)} of {self.batches} frame batches (last error: {self.failures[-1]}). Try again when the connection is stable.'
        return None

    def detect(self, frames):
        chunks = [frames[i:i + self.batch] for i in range(0, len(frames), self.batch)]
        if len(chunks) <= 1 or self.workers == 1:
            return [dets for chunk in chunks for dets in self._detect_chunk(chunk)]
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(min(self.workers, len(chunks))) as pool:
            return [dets for result in pool.map(self._detect_chunk, chunks) for dets in result]

    def _detect_chunk(self, chunk):
        h, w = chunk[0][1].shape[:2]
        cx0, cy0, cx1, cy1 = _crop_bounds(self.crop_box, w, h)
        cw, ch = cx1 - cx0, cy1 - cy0
        parts = []
        for t, img in chunk:
            view = img[cy0:cy1, cx0:cx1] if self.crop_box else img
            vh, vw = view.shape[:2]
            if self.max_width and vw > self.max_width:
                view = cv2.resize(view, (self.max_width, max(1, int(round(vh * self.max_width / vw)))), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode('.jpg', view, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            parts += [{'text': f'Frame at t={t:.2f}s:'}, {'image': buf.tobytes()}]
        region = 'that field crop' if self.crop_box else 'that whole frame'
        parts.append({'text': (
            f'FRC 2026 REBUILT broadcast frames. Red alliance: {", ".join(self.red)}. Blue alliance: {", ".join(self.blue)}. '
            f'Find every ROBOT (large boxy machines with bumpers; not people or referees). Give its team number (digits), '
            f'box_2d [ymin, xmin, ymax, xmax] normalised 0-1000 over {region}, and confidence 0-1. '
            'Report a robot once per frame, in the widest field view. t is the frame label time.')})
        self.batches += 1
        try:
            call_kwargs = {'max_tokens': 8000}
            if self.media_resolution is not None:
                call_kwargs['media_resolution'] = self.media_resolution
            data = self.client.generate_json(parts, FRAME_SCHEMA, **call_kwargs)
        except BudgetExceeded:
            raise
        except Exception as error:  # noqa: BLE001 - one failed batch leaves a gap the tracker bridges, not a failed run
            self.failures.append(str(error)[:200])
            return [[] for _ in chunk]
        by_t = {round(t, 2): [] for t, _ in chunk}
        img_by_t = {round(t, 2): img for t, img in chunk}
        for robot in (data or {}).get('robots') or [] if isinstance(data, dict) else []:
            if not isinstance(robot, dict):
                continue
            try:
                raw_t = float(robot.get('t'))
                box = [float(v) for v in (robot.get('box_2d') or [])]
                raw_conf = robot.get('confidence')
                conf = 0.6 if raw_conf is None else float(raw_conf)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(raw_t) or not math.isfinite(conf):
                continue
            if not valid_box_2d(box, max_area=1_000_000.0 if (self.crop_box and box == [0.0, 0.0, 1000.0, 1000.0]) else 450_000.0):
                continue
            t = _match_chunk_timestamp(raw_t, by_t)
            if t is None:
                continue
            team = re.sub(r'\D', '', str(robot.get('team') or '')) or None
            y1, x1, y2, x2 = box
            px_box = (cx0 + x1 / 1000 * cw, cy0 + y1 / 1000 * ch, cx0 + x2 / 1000 * cw, cy0 + y2 / 1000 * ch)
            frame_img = img_by_t.get(t)
            team_alliance = 'red' if team in self.red else 'blue' if team in self.blue else None
            bumper_col = infer_bumper_alliance(frame_img, px_box, min_share=0.72, min_pixels=16)
            if team_alliance is not None and bumper_col is not None and bumper_col != team_alliance:
                alliance = bumper_col
                team = None
            else:
                alliance = team_alliance or bumper_col or infer_bumper_alliance(frame_img, px_box)
            if self.min_team_conf is not None and conf < self.min_team_conf:
                team = None
                if bumper_col is not None:
                    alliance = bumper_col
            by_t[t].append(Detection(t, px_box, alliance, team, None, conf))
        return [suppress_frame_detections(by_t[round(t, 2)]) for t, _ in chunk]
