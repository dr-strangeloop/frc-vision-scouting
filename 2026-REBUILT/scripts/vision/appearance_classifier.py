"""Per-event robot appearance classifier (William Jojarth's approach).

Across an event, each robot plays ~10-12 matches and accumulates dozens of high-confidence crops where
its bumper number was clearly read. A robot's upper chassis (above the bumper band: white vs dark side
panels, anodized mechanisms, hopper/turret/intake geometry) stays visually consistent across matches
even when its bumper cover switches between Red and Blue.

This module extracts a 144-D bumper-color-invariant upper-chassis + structure feature vector per crop
and fits a regularized shrinkage-LDA + k-NN classifier constrained to the 3 teams on an alliance roster.
"""
import hashlib
import math
import os
import re
import threading
from pathlib import Path

import cv2
import numpy as np

import cache
import detectors
from detectors import ColorBumperDetector
from identity import bumper_contrast_bonus, bumper_occluded, crop_clarity

FEATURE_DIM = 144

_HOG_ROW_CELL = (np.arange(36, dtype=np.int32)[:, None] >= 18).astype(np.int32)
_HOG_COL_IDX = np.arange(52, dtype=np.int32)[None, :]
_HOG_RADIAL_CELL = ((_HOG_COL_IDX < 13) | (_HOG_COL_IDX >= 39)).astype(np.int32)
_HOG_CELL_OFFSET = ((_HOG_ROW_CELL * 2 + _HOG_RADIAL_CELL) * 8).ravel()
_HIST_COL_WEIGHTS = np.broadcast_to(
    np.where((_HOG_COL_IDX >= 13) & (_HOG_COL_IDX < 39), 1.6, 0.4).astype(np.float32),
    (36, 52),
).copy()
_HIST_COL_WEIGHTS_FLAT = _HIST_COL_WEIGHTS.ravel()
_HIST_WEIGHT_SUM = float(_HIST_COL_WEIGHTS_FLAT.sum())
_RADIAL_MOMENT_WEIGHTS = np.array([1.15, 0.65], dtype=np.float32)[None, :, None]


def canonical_team_key(team):
    """Normalize any team identifier ('971', 971, 'frc971') to 'frc971', or None."""
    if team is None or isinstance(team, bool):
        return None
    s = str(team).strip()
    digits = re.sub(r'\D', '', s)
    return f'frc{digits}' if digits else None


def _neutralize_bumper_bleed(upper_bgr):
    """Neutralize saturated red/blue bumper pixels that bleed into the bottom of the upper-chassis slice
    so features remain invariant when a robot switches between Red and Blue bumpers across matches."""
    out = upper_bgr.copy()
    sub = out[18:, :]
    hsv_sub = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    h, s, v = hsv_sub[:, :, 0], hsv_sub[:, :, 1], hsv_sub[:, :, 2]
    red = ((h <= 8) | (h >= 172)) & (s >= 120) & (v >= 80)
    blue = (h >= 100) & (h <= 130) & (s >= 140) & (v >= 60)
    bleed = red | blue
    if np.any(bleed):
        val = np.clip((np.max(sub, axis=2).astype(np.int32) * 5) // 10, 80, 120).astype(np.uint8)
        sub[bleed] = np.stack([val[bleed], val[bleed], val[bleed]], axis=-1)
    return out


def extract_appearance_features(crop):
    """Extract a 144-D L2-normalized appearance feature vector from a BGR robot crop, or None if invalid.

    Designed for cross-match and cross-alliance invariance:
      - Focuses on the upper chassis (rows 4..36, cols 6..58 of 64x64) to exclude both the lower
        Red/Blue bumper band and outer field-carpet background margins.
      - Horizontally reflection-invariant (symmetric center vs outer columns and folded [0, pi/2] HOG
        orientations) so a robot facing +x on Red matches the same robot facing -x on Blue.

    Layout (144 floats):
      [0:72]    Spatial 3x2 radial-vertical grid color moments (mean & std of L, a, b, H, S, V)
      [72:104]  Global upper-chassis color histograms (16 Hue, 8 Saturation, 8 Value), centered
      [104:136] Reflection-invariant 2x2x8 magnitude-weighted Sobel HOG descriptor
      [136:144] Whole-box geometry & upper-chassis structural contrast (8 floats)
    """
    if crop is None or not isinstance(crop, np.ndarray) or crop.ndim != 3 or crop.shape[2] < 3:
        return None
    h, w = crop.shape[:2]
    if h < 12 or w < 12:
        return None

    bgr = crop[:, :, :3]
    resized = cv2.resize(bgr, (64, 64), interpolation=cv2.INTER_AREA)
    upper = _neutralize_bumper_bleed(resized[:38, :])
    # Crop out top 2px and outer 6px left/right border where field carpet/wall background leaks in
    core = upper[2:38, 6:58]  # shape (36, 52, 3)

    lab = cv2.cvtColor(core, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(core, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Center channels around neutral midpoints so white vs dark chassis and distinct colors
    # point in opposite directions (low/negative cosine similarity) rather than sharing a positive offset.
    l_ch = (lab[:, :, 0] - 115.0) * (1.0 / 128.0)
    a_ch = (lab[:, :, 1] - 128.0) * (1.0 / 64.0)
    b_ch = (lab[:, :, 2] - 128.0) * (1.0 / 64.0)
    h_rad = hsv[:, :, 0] * (2.0 * math.pi / 180.0)
    sat_w = np.clip(hsv[:, :, 1] * (1.0 / 255.0), 0.0, 1.0)
    h_ch = np.cos(h_rad) * sat_w
    s_ch = (hsv[:, :, 1] - 85.0) * (1.0 / 128.0)
    v_ch = (hsv[:, :, 2] - 115.0) * (1.0 / 128.0)
    channels = np.stack([l_ch, a_ch, b_ch, h_ch, s_ch, v_ch], axis=-1)  # (36, 52, 6)

    # 1. Spatial 3x2 reflection-invariant grid color moments (3 vertical bands x 2 radial zones: inner 26px vs outer 26px)
    inner = channels[:, 13:39, :].reshape(3, 12, 26, 6)
    outer = np.concatenate([channels[:, :13, :], channels[:, 39:, :]], axis=1).reshape(3, 12, 26, 6)
    radial_grid = np.stack([inner, outer], axis=1)  # (3, 2, 12, 26, 6)
    means = radial_grid.mean(axis=(2, 3)) * _RADIAL_MOMENT_WEIGHTS           # (3, 2, 6)
    stds = (radial_grid.std(axis=(2, 3)) - 0.15) * _RADIAL_MOMENT_WEIGHTS    # (3, 2, 6)
    moments = np.concatenate([means, stds], axis=-1).reshape(72).astype(np.float32)

    # 2. Global upper-chassis color histograms (16 Hue + 8 Sat + 8 Val = 32 floats), center-weighted via fast bincount
    h_bins = np.clip((hsv[:, :, 0] * (16.0 / 180.0)).astype(np.int32), 0, 15).ravel()
    h_weights = (sat_w * _HIST_COL_WEIGHTS).ravel()
    h_hist = np.bincount(h_bins, weights=h_weights, minlength=16).astype(np.float32)
    h_sum = float(h_hist.sum())
    h_hist = (h_hist / h_sum - (1.0 / 16.0)) if h_sum > 1e-5 else np.zeros(16, dtype=np.float32)

    s_bins = np.clip((hsv[:, :, 1] * (8.0 / 256.0)).astype(np.int32), 0, 7).ravel()
    s_hist = np.bincount(s_bins, weights=_HIST_COL_WEIGHTS_FLAT, minlength=8).astype(np.float32) / _HIST_WEIGHT_SUM - (1.0 / 8.0)

    v_bins = np.clip((hsv[:, :, 2] * (8.0 / 256.0)).astype(np.int32), 0, 7).ravel()
    v_hist = np.bincount(v_bins, weights=_HIST_COL_WEIGHTS_FLAT, minlength=8).astype(np.float32) / _HIST_WEIGHT_SUM - (1.0 / 8.0)

    hists = np.concatenate([h_hist, s_hist, v_hist]) * np.float32(1.5)

    # 3. Reflection-invariant structural HOG descriptor (2 vertical x 2 radial cells x 8 folded orientation bins = 32 floats)
    gray = cv2.cvtColor(core, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    mag = np.hypot(gx, gy)
    ang_folded = np.arctan2(gy, gx)  # in [0, pi/2], invariant to horizontal/vertical mirror
    ang_bins = np.clip((ang_folded * (16.0 / math.pi)).astype(np.int32), 0, 7).ravel()
    flat_hog_idx = _HOG_CELL_OFFSET + ang_bins
    hog = np.bincount(flat_hog_idx, weights=mag.ravel(), minlength=32).astype(np.float32)
    hog_norm = float(np.linalg.norm(hog))
    if hog_norm > 1e-6:
        hog = hog / hog_norm - float(np.mean(hog / hog_norm))
    else:
        hog.fill(0.0)

    # 4. Whole-box geometry & upper-chassis contrast (8 floats, bumper-color & flip invariant)
    aspect = float(np.clip(w / float(max(1, h)), 0.4, 2.5)) - 1.0
    top_lum = float(l_ch[:18, :].mean())
    mid_lum = float(l_ch[18:36, :].mean())
    inner_lum = float(l_ch[:, 13:39].mean())
    outer_lum = float(0.5 * (l_ch[:, :13].mean() + l_ch[:, 39:].mean()))
    mag_mean = float(mag.mean())
    horiz_vs_vert = float((gy.mean() - gx.mean()) / max(1.0, mag_mean))
    edge_density = float(np.clip(mag_mean / 80.0, 0.0, 1.5)) - 0.4
    contrast_bonus = float(bumper_contrast_bonus(resized)) - 0.1
    geom = np.array([
        aspect * 0.25,
        (top_lum - mid_lum) * 0.5,
        (inner_lum - outer_lum) * 0.5,
        horiz_vs_vert * 0.5,
        edge_density * 0.3,
        float(sat_w.mean() - 0.35) * 0.5,
        float(l_ch.std() - 0.2) * 0.5,
        contrast_bonus * 0.15,
    ], dtype=np.float32)

    vec = np.concatenate([moments, hists, hog, geom]).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8 or not np.all(np.isfinite(vec)):
        return None
    return (vec / norm).astype(np.float32)


def crop_similarity(crop_a, crop_b):
    """Cosine similarity between two BGR crops mapped to [0.0, 1.0], or None if either crop is invalid."""
    fa = crop_a if (isinstance(crop_a, np.ndarray) and crop_a.ndim == 1 and crop_a.shape[0] == FEATURE_DIM) else extract_appearance_features(crop_a)
    fb = crop_b if (isinstance(crop_b, np.ndarray) and crop_b.ndim == 1 and crop_b.shape[0] == FEATURE_DIM) else extract_appearance_features(crop_b)
    if fa is None or fb is None:
        return None
    return float(np.clip((float(np.dot(fa, fb)) + 1.0) / 2.0, 0.0, 1.0))


class EventAppearanceClassifier:
    """Event-scoped robot appearance classifier trained on confidently named crops across matches."""

    def __init__(self, event_key='default', target_crops_per_team=100, min_crops_per_team=5, shrinkage=0.15):
        self.event_key = str(event_key or 'default')
        self.target_crops_per_team = max(5, int(target_crops_per_team))
        self.min_crops_per_team = max(1, int(min_crops_per_team))
        self.shrinkage = float(np.clip(shrinkage, 0.01, 0.99))
        self.exemplars = {}  # {team_key: list[dict]}
        self.centroids = {}  # {team_key: np.ndarray(144,)}
        self.whitening = np.eye(FEATURE_DIM, dtype=np.float32)
        self._fitted = False
        self._lomo_cache = {}

    def add_crop(self, team_key, crop, confidence=1.0, match_key=None, t_ms=None, crop_sha256=None):
        """Add one confidently named robot crop to the team's exemplar bank. Returns True if stored."""
        canon = canonical_team_key(team_key)
        if not canon:
            return False
        try:
            conf = float(confidence)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(conf) or conf < 0.70:
            return False
        if crop is None or not isinstance(crop, np.ndarray) or crop.ndim != 3:
            return False
        clarity = float(crop_clarity(crop))
        if clarity < 0.45:
            return False
        feats = extract_appearance_features(crop)
        if feats is None:
            return False

        if not crop_sha256:
            crop_sha256 = hashlib.sha256(np.ascontiguousarray(crop).tobytes()).hexdigest()
        else:
            crop_sha256 = str(crop_sha256)

        bank = self.exemplars.setdefault(canon, [])
        if any(ex.get('crop_sha256') == crop_sha256 for ex in bank):
            return False

        quality = float(conf * clarity)
        item = {
            'features': feats,
            'confidence': conf,
            'clarity': clarity,
            'quality': quality,
            'match_key': str(match_key) if match_key else None,
            't_ms': int(round(float(t_ms))) if t_ms is not None else None,
            'crop_sha256': crop_sha256,
        }

        if len(bank) >= self.target_crops_per_team:
            # Evict the lowest-quality exemplar from the most over-represented match_key to preserve cross-match diversity.
            counts = {}
            for ex in bank:
                mk = ex.get('match_key') or ''
                counts[mk] = counts.get(mk, 0) + 1
            max_count = max(counts.values())
            dominant_keys = {mk for mk, cnt in counts.items() if cnt == max_count}
            candidates = [(i, ex) for i, ex in enumerate(bank) if (ex.get('match_key') or '') in dominant_keys]
            evict_idx, evict_ex = min(candidates, key=lambda pair: pair[1]['quality'])
            if len(counts) == 1 and item['quality'] <= evict_ex['quality']:
                return False
            bank[evict_idx] = item
        else:
            bank.append(item)

        self._fitted = False
        self._lomo_cache.clear()
        return True

    def harvest_from_tracks(self, tracks, video_path, match_start_s=0.0, file_offset_s=0.0, match_key=None, min_identity_conf=0.75, max_per_track=25):
        """Harvest unoccluded crops in a single forward pass over `video_path` from high-confidence tracks."""
        if not tracks or not video_path or not Path(video_path).exists():
            return {'added': 0, 'teams': {k: len(v) for k, v in self.exemplars.items()}}

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {'added': 0, 'teams': {k: len(v) for k, v in self.exemplars.items()}}

        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
            frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

            def _to_px_box(box):
                if not box or len(box) != 4:
                    return None
                x1, y1, x2, y2 = (float(v) for v in box)
                if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
                    x1, x2 = x1 * frame_w, x2 * frame_w
                    y1, y2 = y1 * frame_h, y2 * frame_h
                return (
                    max(0, min(frame_w, int(round(x1)))),
                    max(0, min(frame_h, int(round(y1)))),
                    max(0, min(frame_w, int(round(x2)))),
                    max(0, min(frame_h, int(round(y2)))),
                )

            def _traj_in_ms(track, traj):
                if track.get('started_ms') is not None or track.get('ended_ms') is not None:
                    return True
                ts = [abs(float(p.get('t', 0.0))) for p in traj if p.get('t') is not None]
                return bool(ts and max(ts) > 200.0)

            # Index all robot boxes per frame so occlusion is checked against all robots on the field
            all_boxes_by_frame = {}
            eligible_targets_by_frame = {}

            for track_idx, track in enumerate(tracks):
                traj = track.get('trajectory') or track.get('points') or []
                if not traj:
                    continue
                in_ms = _traj_in_ms(track, traj)
                for pt in traj:
                    px_box = _to_px_box(pt.get('box'))
                    if px_box is None or px_box[2] - px_box[0] < 12 or px_box[3] - px_box[1] < 12:
                        continue
                    t_raw = float(pt.get('t', 0.0))
                    t_s = t_raw / 1000.0 if in_ms else t_raw
                    file_s = float(match_start_s) - float(file_offset_s) + t_s
                    if file_s < 0.0:
                        continue
                    f_idx = int(round(file_s * fps))
                    if total_frames > 0 and f_idx >= total_frames:
                        continue
                    all_boxes_by_frame.setdefault(f_idx, []).append((track_idx, px_box))

                team_key = canonical_team_key(track.get('team_key') or track.get('team'))
                track_alliance = str(track.get('alliance') or '').strip().lower() or None
                id_conf = float(track.get('identity_confidence') or 0.0)
                metrics = track.get('metrics') or {}
                id_source = track.get('identity_source') or metrics.get('identity_source')
                swaps = track.get('swap_risks') or metrics.get('swap_risks')
                if not team_key or id_conf < min_identity_conf:
                    continue
                if id_source not in ('reads', 'closeup_crop', 'zoom_crop', None):
                    continue

                # Reject short broken fragments when no explicit direct-read source is recorded on the track
                started_ms = track.get('started_ms')
                ended_ms = track.get('ended_ms')
                if id_source is None and started_ms is not None and ended_ms is not None:
                    if int(started_ms) > 10000 or int(ended_ms) - int(started_ms) < 15000:
                        continue

                max_safe_t_ms = None
                if swaps:
                    if not isinstance(swaps, (list, tuple)) or any(not isinstance(s, dict) or 't_ms' not in s for s in swaps):
                        continue
                    tight_swaps = [int(s['t_ms']) for s in swaps if float(s.get('closest_m', 0.0)) < 0.55]
                    if tight_swaps and id_conf < 0.95:
                        earliest_tight = min(tight_swaps)
                        if earliest_tight < 4000:
                            continue
                        max_safe_t_ms = earliest_tight - 500
                    elif id_conf < 0.85:
                        earliest_any = min(int(s['t_ms']) for s in swaps)
                        if earliest_any < 4000:
                            continue
                        max_safe_t_ms = earliest_any - 500

                # Select up to max_per_track points spread across the track's clean window (preferring first 45s)
                valid_pts = []
                for p in traj:
                    if not (p.get('box') and len(p['box']) == 4):
                        continue
                    t_raw = float(p.get('t', 0.0))
                    t_ms_val = int(round(t_raw if in_ms else t_raw * 1000.0))
                    if max_safe_t_ms is not None and t_ms_val > max_safe_t_ms:
                        continue
                    if t_ms_val <= 45000:
                        valid_pts.append(p)
                if not valid_pts:
                    continue
                if len(valid_pts) > max_per_track:
                    step = len(valid_pts) / float(max_per_track)
                    chosen_pts = []
                    for b in range(max_per_track):
                        chunk = valid_pts[int(round(b * step)):max(int(round(b * step)) + 1, int(round((b + 1) * step)))]
                        chosen_pts.append(max(chunk, key=lambda p: float(p.get('confidence') or p.get('conf') or 0.5)))
                else:
                    chosen_pts = valid_pts

                for pt in chosen_pts:
                    px_box = _to_px_box(pt.get('box'))
                    if px_box is None or px_box[2] - px_box[0] < 12 or px_box[3] - px_box[1] < 12:
                        continue
                    t_raw = float(pt.get('t', 0.0))
                    t_ms = int(round(t_raw if in_ms else t_raw * 1000.0))
                    t_s = t_raw / 1000.0 if in_ms else t_raw
                    file_s = float(match_start_s) - float(file_offset_s) + t_s
                    if file_s < 0.0:
                        continue
                    f_idx = int(round(file_s * fps))
                    if total_frames > 0 and f_idx >= total_frames:
                        continue
                    pt_conf = float(pt.get('confidence') or pt.get('conf') or 0.8)
                    crop_conf = max(0.70, min(1.0, id_conf * max(0.75, pt_conf)))
                    eligible_targets_by_frame.setdefault(f_idx, []).append(
                        (track_idx, team_key, track_alliance, px_box, crop_conf, t_ms, file_s * 1000.0)
                    )

            if not eligible_targets_by_frame:
                return {'added': 0, 'teams': {k: len(v) for k, v in self.exemplars.items()}}

            tol_ms = max(60.0, 1200.0 / max(1.0, fps))
            half_step_ms = 500.0 / max(1.0, fps)
            frame_step_ms = 1000.0 / max(1.0, fps)
            target_frames = sorted(eligible_targets_by_frame.keys())
            cur_frame = -1
            cur_pos_msec = -1e9
            cur_img = None
            added = 0

            def _read_pos_msec(frame_idx):
                raw_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                pos_msec = float(raw_msec) if (raw_msec is not None and math.isfinite(float(raw_msec))) else 0.0
                if pos_msec <= 0.0 and frame_idx > 0:
                    pos_msec = frame_idx / fps * 1000.0
                return pos_msec

            for f_idx in target_frames:
                targets = eligible_targets_by_frame[f_idx]
                target_ms = float(targets[0][6])

                if cur_img is None or cur_pos_msec < target_ms - half_step_ms:
                    if (cur_frame < 0 and f_idx > 90) or (cur_frame >= 0 and target_ms - cur_pos_msec > max(3000.0, 90.0 * frame_step_ms)):
                        if not cap.set(cv2.CAP_PROP_POS_MSEC, target_ms):
                            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                        cur_frame = f_idx - 1
                        cur_img = None
                        ok, frame = cap.read()
                        if not ok or frame is None:
                            break
                        cur_frame += 1
                        cur_img = frame
                        cur_pos_msec = _read_pos_msec(cur_frame)

                    while cur_frame < 0 or cur_pos_msec < target_ms - half_step_ms:
                        if cur_frame >= 0 and (target_ms - cur_pos_msec) > 1.5 * frame_step_ms:
                            if not cap.grab():
                                cur_img = None
                                break
                            cur_frame += 1
                            cur_img = None
                            cur_pos_msec = _read_pos_msec(cur_frame)
                        else:
                            ok, frame = cap.read()
                            if not ok or frame is None:
                                cur_img = None
                                break
                            cur_frame += 1
                            cur_img = frame
                            cur_pos_msec = _read_pos_msec(cur_frame)

                    if cur_img is None and abs(cur_pos_msec - target_ms) <= tol_ms:
                        ok, frame = cap.read()
                        if ok and frame is not None:
                            cur_frame += 1
                            cur_img = frame
                            cur_pos_msec = _read_pos_msec(cur_frame)

                if cur_img is None:
                    break
                if abs(cur_pos_msec - target_ms) > tol_ms:
                    continue

                frame = cur_img
                fh, fw = frame.shape[:2]
                all_in_frame = all_boxes_by_frame.get(f_idx, ())
                for track_idx, team_key, track_alliance, px_box, crop_conf, t_ms, item_target_ms in targets:
                    if abs(cur_pos_msec - float(item_target_ms)) > tol_ms:
                        continue
                    others = [b for t_i, b in all_in_frame if t_i != track_idx]
                    if bumper_occluded(px_box, others, minimum_overlap=0.35):
                        continue
                    x1, y1, x2, y2 = (
                        max(0, min(fw, px_box[0])),
                        max(0, min(fh, px_box[1])),
                        max(0, min(fw, px_box[2])),
                        max(0, min(fh, px_box[3])),
                    )
                    if x2 - x1 < 12 or y2 - y1 < 12:
                        continue
                    det_alliance = detectors.infer_bumper_alliance(frame, (x1, y1, x2, y2), min_share=0.65, min_pixels=12)
                    if track_alliance in ('red', 'blue') and det_alliance in ('red', 'blue') and det_alliance != track_alliance:
                        continue
                    crop = frame[y1:y2, x1:x2].copy()
                    if self.add_crop(team_key, crop, confidence=crop_conf, match_key=match_key, t_ms=t_ms):
                        added += 1
        finally:
            cap.release()

        if added > 0:
            self.fit()
        return {'added': added, 'teams': {k: len(v) for k, v in self.exemplars.items()}}

    def _compute_model(self, exclude_match_key=None):
        """Compute L2-normalized team centroids and shrinkage-LDA whitening matrix, optionally excluding one match."""
        feats_by_team = {}
        centroids = {}
        residuals = []
        for team_key, bank in self.exemplars.items():
            filtered = [ex['features'] for ex in bank if exclude_match_key is None or ex.get('match_key') != exclude_match_key]
            if not filtered:
                continue
            mat = np.stack(filtered, axis=0).astype(np.float64)
            feats_by_team[team_key] = mat.astype(np.float32)
            mean_vec = mat.mean(axis=0)
            norm = float(np.linalg.norm(mean_vec))
            centroids[team_key] = (mean_vec / max(1e-8, norm)).astype(np.float32)
            if mat.shape[0] >= 2:
                residuals.append(mat - mean_vec)

        if residuals:
            all_res = np.vstack(residuals)
            dof = max(1, all_res.shape[0] - len(residuals))
            s_w = (all_res.T @ all_res) / float(dof)
            lam = self.shrinkage
            cov = (1.0 - lam) * s_w + lam * np.eye(FEATURE_DIM, dtype=np.float64)
            eigvals, eigvecs = np.linalg.eigh(cov)
            inv_sqrt = 1.0 / np.sqrt(np.maximum(eigvals, 1e-6))
            whitening = ((eigvecs * inv_sqrt) @ eigvecs.T).astype(np.float32)
        else:
            whitening = np.eye(FEATURE_DIM, dtype=np.float32)

        return centroids, whitening, feats_by_team

    def fit(self):
        """Fit per-team centroids and pooled shrinkage-LDA whitening transform across all stored exemplars."""
        self.centroids, self.whitening, _ = self._compute_model(exclude_match_key=None)
        self._fitted = True
        self._lomo_cache.clear()
        return {
            'teams': len(self.centroids),
            'exemplars': sum(len(v) for v in self.exemplars.values()),
            'counts': {k: len(v) for k, v in self.exemplars.items()},
        }

    def _model_for(self, exclude_match_key=None):
        if exclude_match_key is None:
            if not self._fitted:
                self.fit()
            feats_by_team = {
                k: np.stack([ex['features'] for ex in v], axis=0).astype(np.float32)
                for k, v in self.exemplars.items()
                if v
            }
            return self.centroids, self.whitening, feats_by_team
        key = str(exclude_match_key)
        cached = self._lomo_cache.get(key)
        if cached is None:
            cached = self._compute_model(exclude_match_key=key)
            self._lomo_cache[key] = cached
        return cached

    def predict_crop(self, crop, candidate_teams, exclude_match_key=None, temperature=0.20):
        """Predict alliance-constrained team probabilities for a single crop."""
        canon_candidates = []
        for t in candidate_teams or ():
            c = canonical_team_key(t)
            if c and c not in canon_candidates:
                canon_candidates.append(c)

        if not canon_candidates:
            return {
                'source': 'appearance',
                'team': None,
                'confidence': 0.0,
                'margin': 0.0,
                'probabilities': {},
                'n_train': {},
            }

        centroids, whitening, feats_by_team = self._model_for(exclude_match_key=exclude_match_key)
        n_train = {c: int(feats_by_team[c].shape[0]) if c in feats_by_team else 0 for c in canon_candidates}
        eligible = [c for c in canon_candidates if n_train[c] >= self.min_crops_per_team]

        uniform = {c: round(1.0 / len(canon_candidates), 4) for c in canon_candidates}
        if len(eligible) < 2:
            return {
                'source': 'appearance',
                'team': None,
                'confidence': 0.0,
                'margin': 0.0,
                'probabilities': uniform,
                'n_train': n_train,
            }

        f = crop if (isinstance(crop, np.ndarray) and crop.ndim == 1 and crop.shape[0] == FEATURE_DIM) else extract_appearance_features(crop)
        if f is None:
            return {
                'source': 'appearance',
                'team': None,
                'confidence': 0.0,
                'margin': 0.0,
                'probabilities': uniform,
                'n_train': n_train,
            }

        wf = whitening @ f
        wf_norm = float(np.linalg.norm(wf))
        wf = wf / max(1e-8, wf_norm)

        scores = {}
        for c in eligible:
            mu = centroids[c]
            wmu = whitening @ mu
            wmu = wmu / max(1e-8, float(np.linalg.norm(wmu)))
            s_lda = float(np.dot(wf, wmu))

            mat = feats_by_team[c]
            w_mat = mat @ whitening.T
            w_norms = np.maximum(1e-8, np.linalg.norm(w_mat, axis=1, keepdims=True))
            w_mat_normed = w_mat / w_norms
            sims = w_mat_normed @ wf
            k = min(5, int(sims.shape[0]))
            s_knn = float(np.mean(np.sort(sims)[-k:]))
            scores[c] = 0.6 * s_lda + 0.4 * s_knn

        temp = max(1e-3, float(temperature))
        max_s = max(scores.values())
        exps = {c: math.exp((scores[c] - max_s) / temp) if c in scores else 0.0 for c in canon_candidates}
        z = sum(exps.values()) or 1.0
        probs = {c: exps[c] / z for c in canon_candidates}

        ranked = sorted(canon_candidates, key=lambda c: probs[c], reverse=True)
        top1 = ranked[0]
        top2 = ranked[1] if len(ranked) > 1 else None
        p1 = probs[top1]
        p2 = probs[top2] if top2 else 0.0
        margin = p1 - p2

        min_n = min(n_train[c] for c in eligible)
        maturity = min(1.0, float(min_n) / 20.0)
        winner = top1 if margin >= 0.10 else None
        conf = (p1 * maturity) if winner else 0.0

        return {
            'source': 'appearance',
            'team': winner,
            'confidence': round(float(conf), 4),
            'margin': round(float(margin), 4),
            'probabilities': {c: round(float(probs[c]), 4) for c in canon_candidates},
            'n_train': n_train,
        }

    def predict_tracklet(self, crops, candidate_teams, exclude_match_key=None, min_margin=0.12, temperature=0.12):
        """Predict team identity across all crops of a tracklet (returns AppearancePrediction §3.2)."""
        canon_candidates = []
        for t in candidate_teams or ():
            c = canonical_team_key(t)
            if c and c not in canon_candidates:
                canon_candidates.append(c)

        if hasattr(crops, 'crops'):
            raw_crops = crops.crops or []
        elif isinstance(crops, dict) and 'crops' in crops:
            raw_crops = crops.get('crops') or []
        else:
            raw_crops = list(crops or ())

        _, _, feats_by_team = self._model_for(exclude_match_key=exclude_match_key)
        n_train = {c: int(feats_by_team[c].shape[0]) if c in feats_by_team else 0 for c in canon_candidates}
        eligible = [c for c in canon_candidates if n_train[c] >= self.min_crops_per_team]
        uniform = {c: round(1.0 / max(1, len(canon_candidates)), 4) for c in canon_candidates}

        if not canon_candidates or len(eligible) < 2 or not raw_crops:
            return {
                'source': 'appearance',
                'team': None,
                'confidence': 0.0,
                'margin': 0.0,
                'probabilities': uniform,
                'crops_used': 0,
                'n_train': n_train,
            }

        log_probs = {c: 0.0 for c in canon_candidates}
        total_w = 0.0
        crops_used = 0

        for item in raw_crops:
            if isinstance(item, (tuple, list)) and len(item) >= 3:
                crop_img = item[2]
            elif isinstance(item, dict):
                crop_img = item.get('crop')
            else:
                crop_img = item

            pred = self.predict_crop(crop_img, canon_candidates, exclude_match_key=exclude_match_key, temperature=temperature)
            probs = pred.get('probabilities') or {}
            if not probs or all(probs.get(c, 0.0) == probs.get(canon_candidates[0], 0.0) for c in canon_candidates):
                continue
            w = float(crop_clarity(crop_img)) if ( isinstance(crop_img, np.ndarray) and crop_img.ndim == 3 ) else 1.0
            for c in canon_candidates:
                log_probs[c] += w * math.log(max(1e-6, float(probs.get(c, 1e-6))))
            total_w += w
            crops_used += 1

        if crops_used == 0 or total_w <= 0.0:
            return {
                'source': 'appearance',
                'team': None,
                'confidence': 0.0,
                'margin': 0.0,
                'probabilities': uniform,
                'crops_used': 0,
                'n_train': n_train,
            }

        avg_log = {c: log_probs[c] / total_w for c in canon_candidates}
        max_lp = max(avg_log[c] for c in eligible)
        exps = {c: math.exp(avg_log[c] - max_lp) if c in eligible else 0.0 for c in canon_candidates}
        z = sum(exps.values()) or 1.0
        final_probs = {c: exps[c] / z for c in canon_candidates}

        ranked = sorted(canon_candidates, key=lambda c: final_probs[c], reverse=True)
        top1 = ranked[0]
        top2 = ranked[1] if len(ranked) > 1 else None
        p1 = final_probs[top1]
        p2 = final_probs[top2] if top2 else 0.0
        margin = p1 - p2

        min_n = min(n_train[c] for c in eligible)
        maturity = min(1.0, float(min_n) / 20.0)
        winner = top1 if margin >= float(min_margin) else None
        conf = (p1 * maturity) if winner else 0.0

        return {
            'source': 'appearance',
            'team': winner,
            'confidence': round(float(conf), 4),
            'margin': round(float(margin), 4),
            'probabilities': {c: round(float(final_probs[c]), 4) for c in canon_candidates},
            'crops_used': int(crops_used),
            'n_train': n_train,
        }

    @staticmethod
    def _default_cache_path(event_key):
        key = str(event_key or 'default').strip()
        return cache.root() / 'identity' / f'{key}_appearance.npz'

    def save(self, path=None):
        """Save exemplar features and fitted model parameters to a compressed .npz in the local cache."""
        if not self._fitted:
            self.fit()
        target = Path(path) if path is not None else self._default_cache_path(self.event_key)
        target.parent.mkdir(parents=True, exist_ok=True)

        features_list, confs, clarities, qualities, teams, matches, t_ms_list, digests = [], [], [], [], [], [], [], []
        for team_key, bank in sorted(self.exemplars.items()):
            for ex in bank:
                features_list.append(ex['features'])
                confs.append(float(ex['confidence']))
                clarities.append(float(ex['clarity']))
                qualities.append(float(ex['quality']))
                teams.append(str(team_key))
                matches.append(str(ex.get('match_key') or ''))
                t_ms_list.append(int(ex['t_ms']) if ex.get('t_ms') is not None else -1)
                digests.append(str(ex.get('crop_sha256') or ''))

        feats_arr = np.stack(features_list, axis=0).astype(np.float32) if features_list else np.empty((0, FEATURE_DIM), dtype=np.float32)
        tmp = target.with_name(f'{target.stem}.{os.getpid()}.{threading.get_ident()}.tmp.npz')
        np.savez_compressed(
            tmp,
            event_key=np.array(self.event_key),
            target_crops_per_team=np.array(self.target_crops_per_team, dtype=np.int32),
            min_crops_per_team=np.array(self.min_crops_per_team, dtype=np.int32),
            shrinkage=np.array(self.shrinkage, dtype=np.float32),
            whitening=self.whitening.astype(np.float32),
            features=feats_arr,
            confidence=np.array(confs, dtype=np.float32),
            clarity=np.array(clarities, dtype=np.float32),
            quality=np.array(qualities, dtype=np.float32),
            team_key=np.array(teams),
            match_key=np.array(matches),
            t_ms=np.array(t_ms_list, dtype=np.int64),
            crop_sha256=np.array(digests),
        )
        tmp.replace(target)
        return target

    @classmethod
    def load(cls, path_or_event_key):
        """Load a saved EventAppearanceClassifier from a file path, event_key ('2026cc'), or match_key ('2026cc_qm1')."""
        if path_or_event_key is None:
            return None
        if isinstance(path_or_event_key, Path) or str(path_or_event_key).endswith('.npz') or '/' in str(path_or_event_key) or '\\' in str(path_or_event_key):
            path = Path(path_or_event_key)
        else:
            key = str(path_or_event_key).strip()
            path = cls._default_cache_path(key)
            if not path.exists() and '_' in key:
                path = cls._default_cache_path(key.split('_', 1)[0])
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                clf = cls(
                    event_key=str(data['event_key'].item()) if 'event_key' in data else 'default',
                    target_crops_per_team=int(data['target_crops_per_team'].item()) if 'target_crops_per_team' in data else 100,
                    min_crops_per_team=int(data['min_crops_per_team'].item()) if 'min_crops_per_team' in data else 5,
                    shrinkage=float(data['shrinkage'].item()) if 'shrinkage' in data else 0.15,
                )
                feats = data['features']
                confs = data['confidence']
                clarities = data['clarity']
                qualities = data['quality']
                teams = data['team_key']
                matches = data['match_key']
                t_ms_arr = data['t_ms']
                digests = data['crop_sha256']
                for i in range(feats.shape[0]):
                    t_key = str(teams[i])
                    mk = str(matches[i]) or None
                    t_val = int(t_ms_arr[i])
                    clf.exemplars.setdefault(t_key, []).append({
                        'features': feats[i].astype(np.float32),
                        'confidence': float(confs[i]),
                        'clarity': float(clarities[i]),
                        'quality': float(qualities[i]),
                        'match_key': mk,
                        't_ms': None if t_val < 0 else t_val,
                        'crop_sha256': str(digests[i]),
                    })
                clf.fit()
                if 'whitening' in data and data['whitening'].shape == (FEATURE_DIM, FEATURE_DIM):
                    clf.whitening = data['whitening'].astype(np.float32)
                return clf
        except (OSError, ValueError, KeyError):
            return None
