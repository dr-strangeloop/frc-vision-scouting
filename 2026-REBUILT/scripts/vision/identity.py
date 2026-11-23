"""Who is this robot? Bumper numbers read from a handful of crops, voted across the track.

Reading every frame is wasteful and noisy. Per tracklet we take the few best crops (large, confident,
not at the frame edge), ask a VLM to read the bumper number constrained to the six roster teams, and
turn the answers into votes. stitch.assign_teams then combines votes across the chain and the roster.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import math
import re

import cv2
import numpy as np

NUMBER_SCHEMA = {'type': 'object', 'properties': {'reads': {'type': 'array', 'items': {'type': 'object', 'properties': {
    'index': {'type': 'integer'}, 'team': {'type': 'string'}, 'confidence': {'type': 'number'}}, 'required': ['index', 'team']}}}, 'required': ['reads']}

_ZOOM_RED_RANGES = (((0, 120, 80), (8, 255, 255)), ((172, 120, 80), (180, 255, 255)))
_ZOOM_BLUE_RANGES = (((100, 140, 60), (130, 255, 255)),)


def bumper_occluded(box, others=(), minimum_overlap=0.35, min_overlap=None):
    """Return True if another robot box covers at least minimum_overlap (35%) of box's lower bumper band."""
    threshold = minimum_overlap if min_overlap is None else min_overlap
    l, t, r, b = box
    w, h = r - l, b - t
    band = (l + 0.05 * w, t + 0.42 * h, r - 0.05 * w, b)
    area = (band[2] - band[0]) * (band[3] - band[1])
    if area <= 0:
        return True
    return any(
        max(0.0, min(band[2], q[2]) - max(band[0], q[0]))
        * max(0.0, min(band[3], q[3]) - max(band[1], q[1]))
        / area >= threshold
        for q in (others or ())
        if q is not None and len(q) == 4
    )


def _band_white_digit_bonus(band, max_white_share=0.35):
    if band is None or not getattr(band, 'size', 0) or getattr(band, 'ndim', 0) != 3 or band.shape[0] < 2 or band.shape[1] < 4:
        return 0.0
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    white = (hsv[:, :, 1] <= 100) & (hsv[:, :, 2] >= 145)
    white_share = float(white.mean())
    if not 0.01 <= white_share <= max_white_share:
        return 0.0
    bumper_detail = float(cv2.Laplacian((white.astype('uint8') * 255), cv2.CV_64F).var())
    return min(1.0, bumper_detail / 300.0) * 0.4


def bumper_contrast_bonus(crop):
    """Bonus (0.0..0.4) when bright, low-saturation white digits appear in the lower bumper band (0.42..0.91 height)."""
    if crop is None or not getattr(crop, 'size', 0) or getattr(crop, 'ndim', 0) != 3 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return 0.0
    band = crop[round(crop.shape[0] * 0.42):round(crop.shape[0] * 0.91)]
    return _band_white_digit_bonus(band, max_white_share=0.35)


def crop_clarity(crop):
    """Relative sharpness factor (0.2..3.0) from Laplacian variance plus white-digit bumper contrast bonus."""
    if crop is None or not getattr(crop, 'size', 0) or crop.shape[0] < 4 or crop.shape[1] < 4:
        return 1.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    base = min(3.0, max(0.2, math.sqrt(max(0.0, lap) / 200.0)))
    return base * (1.0 + bumper_contrast_bonus(crop))


def prepare_bumper_closeup(crop, box=None, neighbor_boxes=(), target_width=320, bumper_band_norm=None):
    """Build a VLM-ready close-up image by masking overlapping neighbor pixels in the bumper band and
    stacking the full robot crop above a bicubic-upscaled lower bumper band. Returns None if occluded."""
    if isinstance(crop, dict):
        if bumper_band_norm is None:
            bumper_band_norm = crop.get('bumper_band_norm')
        crop = crop.get('crop')
    if crop is None or not isinstance(crop, np.ndarray) or crop.ndim != 3 or crop.shape[0] < 8 or crop.shape[1] < 8:
        return None
    h, w = crop.shape[:2]
    img = crop.copy()
    has_band_norm = (
        isinstance(bumper_band_norm, (list, tuple))
        and len(bumper_band_norm) == 2
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)) for v in bumper_band_norm)
    )
    if box is not None and neighbor_boxes:
        if bumper_occluded(box, neighbor_boxes, minimum_overlap=0.35):
            return None
        bx0, by0, bx1, by1 = (float(v) for v in box)
        bw = max(1e-6, bx1 - bx0)
        bh = max(1e-6, by1 - by0)
        scale_x = w / bw
        scale_y = h / bh
        is_pixel = max(abs(bx0), abs(by0), abs(bx1), abs(by1)) > 2.0
        band_y0 = max(0, int(round(h * float(bumper_band_norm[0])))) if has_band_norm else int(round(h * 0.42))
        band_y1 = min(h, int(round(h * float(bumper_band_norm[1])))) if has_band_norm else int(round(h * 0.95))
        for other in neighbor_boxes:
            if other is None or len(other) != 4:
                continue
            ox0, oy0, ox1, oy1 = (float(v) for v in other)
            ow, oh = max(0.0, ox1 - ox0), max(0.0, oy1 - oy0)
            px = max(4.0, ow * 0.06) if is_pixel else max(4.0 / max(1, w) * bw, ow * 0.06)
            py = max(4.0, oh * 0.12) if is_pixel else max(4.0 / max(1, h) * bh, oh * 0.12)
            cx0 = max(0, int(round((ox0 - px - bx0) * scale_x)))
            cx1 = min(w, int(round((ox1 + px - bx0) * scale_x)))
            cy0 = max(band_y0, int(round((oy0 - py - by0) * scale_y)))
            cy1 = min(band_y1, int(round((oy1 + py - by0) * scale_y)))
            if cx1 > cx0 and cy1 > cy0:
                img[cy0:cy1, cx0:cx1] = 0

    if has_band_norm:
        by0_norm, by1_norm = float(bumper_band_norm[0]), float(bumper_band_norm[1])
        band = img[max(0, round(h * by0_norm)):min(h, round(h * by1_norm)), :]
    else:
        band = img[round(h * 0.40):round(h * 0.96), :]
    if w < 16 or band.shape[0] < 6:
        return img

    desired_w = max(w, min(480, int(target_width)))
    out_w = min(desired_w, int(round(w * 4.0)))
    scale = out_w / float(w)
    interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    full_h = max(1, int(round(h * scale)))
    band_h = max(1, int(round(band.shape[0] * scale)))
    full_resized = cv2.resize(img, (out_w, full_h), interpolation=interp)
    band_resized = cv2.resize(band, (out_w, band_h), interpolation=interp)
    divider = np.zeros((2, out_w, img.shape[2]), dtype=img.dtype)
    return np.vstack([full_resized, divider, band_resized])


def combine_reader_votes(reads, min_confidence=0.65):
    """Disagreement between readers on one source frame invalidates that vote."""
    groups = {}
    for read in reads or ():
        if not isinstance(read, dict):
            continue
        groups.setdefault((read.get('track_key'), read.get('frame_index')), []).append(read)
    result = []
    for values in groups.values():
        valid = [
            r for r in values
            if (r.get('team_key') or r.get('team'))
            and isinstance(r.get('confidence'), (int, float))
            and not isinstance(r['confidence'], bool)
            and math.isfinite(r['confidence'])
            and min_confidence <= r['confidence'] <= 1.0
        ]
        if len({str(r.get('team_key') or r.get('team')).strip() for r in valid}) == 1:
            result.append(max(valid, key=lambda r: float(r['confidence'])))
    return result


def best_crops(observations, k=4, frame_w=1920, frame_h=1080, spread_bins=True):
    """Pick up to k observations to read: biggest unoccluded boxes with good confidence and clarity, away from the
    frame edge, spread over time. observations: dicts with box (x1,y1,x2,y2), conf, t, frame (ndarray) or crop."""
    if k <= 0 or not observations:
        return []
    eligible = [
        o for o in observations
        if not o.get('bumper_occluded')
        and not bumper_occluded(o['box'], o.get('other_boxes') or o.get('neighbors') or ())
    ]
    if not eligible:
        return []

    def score(o):
        x1, y1, x2, y2 = o['box']
        edge = min(x1, y1, frame_w - x2, frame_h - y2)
        crop = o.get('crop')
        if crop is None and o.get('frame') is not None:
            img = o['frame']
            cx1, cy1, cx2, cy2 = max(0, int(x1)), max(0, int(y1)), min(img.shape[1], int(x2)), min(img.shape[0], int(y2))
            crop = img[cy1:cy2, cx1:cx2] if cx2 > cx1 and cy2 > cy1 else None
        clarity = crop_clarity(crop) if crop is not None else 1.0
        return max(0.0, x2 - x1) * max(0.0, y2 - y1) * o.get('conf', 0.5) * (1.0 if edge > 20 else 0.4) * clarity

    ranked = sorted(eligible, key=score, reverse=True)
    picked, times, frames, digests, picked_ids = [], [], set(), set(), set()

    def _try_accept(o):
        if id(o) in picked_ids:
            return False
        frame_idx = o.get('frame_index')
        digest = o.get('crop_sha256')
        if frame_idx is not None and frame_idx in frames:
            return False
        if digest is not None and digest in digests:
            return False
        if not all(abs(o['t'] - t) > 0.8 for t in times):
            return False
        picked.append(o)
        picked_ids.add(id(o))
        times.append(o['t'])
        if frame_idx is not None:
            frames.add(frame_idx)
        if digest is not None:
            digests.add(digest)
        return True

    t_min = min(float(o['t']) for o in eligible)
    t_max = max(float(o['t']) for o in eligible)
    span = t_max - t_min

    if spread_bins and k >= 2 and len(eligible) > k and span >= 3.0:
        bins = [[] for _ in range(k)]
        for o in ranked:
            b_idx = min(k - 1, max(0, int((float(o['t']) - t_min) / span * k)))
            bins[b_idx].append(o)
        for group in bins:
            for o in group:
                if _try_accept(o):
                    break
            if len(picked) >= k:
                break

    for o in ranked:
        if len(picked) >= k:
            break
        _try_accept(o)

    return picked


def needs_closeup_reads(votes, min_total=1.8, min_share=0.72, *, evidence=None, shared_top_team=False):
    """Return True when a tracklet's existing votes are empty, weak (< min_total), split (< min_share),
    share a top team with another tracklet on the same alliance, or rely solely on weak/noisy wide-frame reads."""
    if shared_top_team:
        return True
    valid = {
        str(k): float(v)
        for k, v in (votes or {}).items()
        if k is not None and isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0
    }
    if not valid:
        return True
    total = sum(valid.values())
    top = max(valid.values())
    share = top / total
    if evidence:
        ev_items = [e for e in evidence if isinstance(e, dict) and e.get('source')]
        if ev_items:
            has_direct_closeup = any(e.get('source') in ('closeup_crop', 'zoom_crop') for e in ev_items)
            all_wide = all(e.get('source') == 'wide_frame' for e in ev_items)
            if not has_direct_closeup and all_wide:
                return bool(total < max(float(min_total), 4.5) or share < max(float(min_share), 0.78))
    return bool(total < min_total or share < min_share)


def read_bumper_numbers(client, crops, roster, preprocess=False):
    """crops: list of ndarray BGR images. Returns [(team or None, confidence)] per crop, restricted to the roster."""
    if not crops:
        return []
    roster_map = {}
    for t in roster:
        s = str(t).strip()
        roster_map[s] = s
        digits = re.sub(r'\D', '', s)
        if digits:
            roster_map.setdefault(digits, s)
            roster_map.setdefault(f'frc{digits}', s)
    parts = []
    jpeg_quality = 92 if preprocess else 90
    for i, crop in enumerate(crops):
        band_norm = None
        if isinstance(crop, dict):
            band_norm = crop.get('bumper_band_norm')
            crop = crop.get('crop')
        if crop is None or not getattr(crop, 'size', 0):
            continue
        img_to_send = crop
        if preprocess and getattr(crop, 'ndim', 0) == 3 and crop.shape[0] >= 24 and crop.shape[1] >= 24:
            closeup = prepare_bumper_closeup(crop, bumper_band_norm=band_norm)
            if closeup is not None:
                img_to_send = closeup
        ok, buf = cv2.imencode('.jpg', img_to_send, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if ok:
            parts += [{'text': f'Crop {i}:'}, {'image': buf.tobytes()}]
    if not parts:
        return [(None, 0.0)] * len(crops)
    parts.append({'text': (f'Each crop is one FRC robot. Read the number on its bumper. It must be one of: {", ".join(str(t) for t in roster)}. '
                           'If you cannot read it, answer team "" with confidence 0. Return one read per crop, using its index.')})
    data = client.generate_json(parts, NUMBER_SCHEMA, max_tokens=1500)
    out = [(None, 0.0)] * len(crops)
    by_index = {}
    for read in (data or {}).get('reads') or [] if isinstance(data, dict) else []:
        if not isinstance(read, dict):
            continue
        i = read.get('index')
        raw_team = str(read.get('team') or '').strip()
        team = roster_map.get(raw_team) or roster_map.get(re.sub(r'\D', '', raw_team))
        raw_conf = read.get('confidence')
        try:
            conf = 0.6 if raw_conf is None else float(raw_conf)
        except (TypeError, ValueError):
            continue
        if isinstance(i, int) and not isinstance(i, bool) and 0 <= i < len(crops) and team is not None and math.isfinite(conf) and conf > 0:
            by_index.setdefault(i, []).append((team, min(1.0, conf)))
    for i, candidates in by_index.items():
        if len({t for t, _ in candidates}) == 1:
            out[i] = max(candidates, key=lambda item: item[1])
    return out


def detect_zoom_panels(calibration=None, frame_shape=(720, 1280)):
    """Identify bottom split-screen broadcast zoom panels when the wide field ROI ends at y <= 0.82."""
    roi = (calibration or {}).get('roi') if isinstance(calibration, dict) else None
    if not roi or len(roi) < 4:
        return []
    try:
        y1 = float(roi[3])
    except (TypeError, ValueError):
        return []
    if not math.isfinite(y1) or y1 > 0.82:
        return []
    y0 = round(min(0.95, y1 + 0.01), 4)
    return [
        {'name': 'left_zoom', 'roi': [0.0, y0, 0.5, 1.0], 'side': 'left'},
        {'name': 'right_zoom', 'roi': [0.5, y0, 1.0, 1.0], 'side': 'right'},
    ]


def collect_zoom_panel_crops(frames, calibration=None, max_crops_per_alliance=6, min_area_px=1500):
    """Harvest high-clarity unoccluded robot bumper crops from broadcast zoom panels without field projection."""
    panels = detect_zoom_panels(calibration)
    if not panels or not frames or max_crops_per_alliance <= 0:
        return []

    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 7))
    candidates_by_alliance = {'red': [], 'blue': []}

    for t, frame_bgr in frames:
        if frame_bgr is None or not getattr(frame_bgr, 'size', 0) or getattr(frame_bgr, 'ndim', 0) != 3:
            continue
        fh, fw = frame_bgr.shape[:2]
        for panel in panels:
            rx0, ry0, rx1, ry1 = panel['roi']
            px0, py0 = max(0, int(round(rx0 * fw))), max(0, int(round(ry0 * fh)))
            px1, py1 = min(fw, int(round(rx1 * fw))), min(fh, int(round(ry1 * fh)))
            ph, pw = py1 - py0, px1 - px0
            if pw < 24 or ph < 24:
                continue
            panel_bgr = frame_bgr[py0:py1, px0:px1]
            hsv = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2HSV)
            panel_boxes = []
            for alliance, ranges in (('red', _ZOOM_RED_RANGES), ('blue', _ZOOM_BLUE_RANGES)):
                mask = np.zeros((ph, pw), dtype=np.uint8)
                for low, high in ranges:
                    mask |= cv2.inRange(hsv, np.array(low, dtype=np.uint8), np.array(high, dtype=np.uint8))
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    area = float(cv2.contourArea(contour))
                    if area < min_area_px or area > 0.80 * ph * pw:
                        continue
                    x, y, bw, bh = cv2.boundingRect(contour)
                    if bw < 16 or bh < 8 or (bw / max(1, bh)) > 4.5:
                        continue
                    rx1_p = max(0, int(round(x - 0.06 * bw)))
                    rx2_p = min(pw, int(round(x + bw + 0.06 * bw)))
                    ry1_p = max(0, int(round(y - 1.5 * bh)))
                    ry2_p = min(ph, int(round(y + bh + 0.12 * bh)))
                    panel_boxes.append((alliance, area, (x, y, x + bw, y + bh), (rx1_p, ry1_p, rx2_p, ry2_p)))

            for idx, (alliance, area, b_box, r_box) in enumerate(panel_boxes):
                other_boxes = [other_r for j, (_, _, _, other_r) in enumerate(panel_boxes) if j != idx]
                if bumper_occluded(r_box, other_boxes, minimum_overlap=0.35):
                    continue
                x1_r, y1_r, x2_r, y2_r = r_box
                crop = panel_bgr[y1_r:y2_r, x1_r:x2_r].copy()
                if crop.shape[0] < 12 or crop.shape[1] < 16:
                    continue
                bx1, by1, bx2, by2 = b_box
                bumper_patch = panel_bgr[max(0, by1):min(ph, by2), max(0, bx1):min(pw, bx2)]
                bonus = max(bumper_contrast_bonus(crop), _band_white_digit_bonus(bumper_patch, max_white_share=0.45))
                if bonus <= 0.02:
                    continue
                score = float(area * crop_clarity(crop) * (1.0 + bonus))
                digest = hashlib.sha256(np.ascontiguousarray(crop).tobytes()).hexdigest()
                span_y = max(1, y2_r - y1_r)
                bumper_band_norm = [
                    round((by1 - y1_r) / span_y, 3),
                    round((by2 - y1_r) / span_y, 3),
                ]
                candidates_by_alliance[alliance].append({
                    't': float(t),
                    'alliance': alliance,
                    'panel': panel['name'],
                    'side': panel['side'],
                    'crop': crop,
                    'crop_sha256': digest,
                    'score': round(score, 2),
                    'bumper_band_norm': bumper_band_norm,
                })

    selected = []
    for alliance in ('red', 'blue'):
        ranked = sorted(candidates_by_alliance[alliance], key=lambda item: item['score'], reverse=True)
        picked_times = []
        picked_digests = set()
        count = 0
        for item in ranked:
            if item['crop_sha256'] in picked_digests:
                continue
            if not all(abs(item['t'] - pt) >= 0.8 for pt in picked_times):
                continue
            selected.append(item)
            picked_times.append(item['t'])
            picked_digests.add(item['crop_sha256'])
            count += 1
            if count >= max_crops_per_alliance:
                break
    return sorted(selected, key=lambda item: (item['t'], item['alliance']))


def _tracklet_get(tr, key, default=None):
    return tr.get(key, default) if isinstance(tr, dict) else getattr(tr, key, default)


def _tracklet_set(tr, key, value):
    if isinstance(tr, dict):
        tr[key] = value
    else:
        setattr(tr, key, value)


def _seed_tracklet_evidence(tr):
    evidence = _tracklet_get(tr, 'evidence')
    if evidence is None:
        evidence = []
        for k, v in (_tracklet_get(tr, 'votes') or {}).items():
            if k is not None and isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0:
                evidence.append({'source': 'wide_frame', 'team': str(k), 'confidence': float(v)})
        _tracklet_set(tr, 'evidence', evidence)
    return evidence


def match_zoom_reads_to_tracklets(zoom_reads, tracklets, time_window_s=1.0, min_conf=0.70, min_margin=0.08, similarity_fn=None):
    """Link high-confidence zoom-panel bumper reads to concurrent wide-view tracklets via appearance & screen side."""
    if not zoom_reads or not tracklets:
        return []
    if similarity_fn is None:
        from appearance_classifier import crop_similarity as similarity_fn

    linked = []
    for zr in zoom_reads:
        if not isinstance(zr, dict):
            continue
        team = str(zr.get('team') or '').strip()
        try:
            conf = float(zr.get('confidence') or 0.0)
            t_zr = float(zr.get('t', 0.0))
        except (TypeError, ValueError):
            continue
        if not team or not math.isfinite(conf) or conf < min_conf:
            continue
        alliance = zr.get('alliance')
        z_crop = zr.get('crop')
        z_side = zr.get('side')

        candidates = []
        for idx, tr in enumerate(tracklets):
            tr_alliance = _tracklet_get(tr, 'alliance')
            if alliance and tr_alliance != alliance:
                continue
            pts = _tracklet_get(tr, 'points') or []
            near_pts = [pt for pt in pts if abs(float(pt.get('t', 0.0)) - t_zr) <= time_window_s]
            if not near_pts:
                continue
            candidates.append((idx, tr, near_pts))

        if not candidates:
            continue

        scored = []
        for idx, tr, near_pts in candidates:
            # Visual similarity to temporally closest crop on the tracklet
            tr_crops = _tracklet_get(tr, 'crops') or []
            best_sim = None
            if z_crop is not None and tr_crops:
                best_crop_img = None
                best_dt = float('inf')
                for c_item in tr_crops:
                    if isinstance(c_item, (tuple, list)) and len(c_item) >= 3:
                        c_t, c_img = float(c_item[1]), c_item[2]
                    elif isinstance(c_item, dict):
                        c_t, c_img = float(c_item.get('t', t_zr)), c_item.get('crop')
                    else:
                        c_t, c_img = t_zr, c_item
                    dt = abs(c_t - t_zr)
                    if c_img is not None and dt < best_dt:
                        best_dt = dt
                        best_crop_img = c_img
                if best_crop_img is not None:
                    best_sim = similarity_fn(z_crop, best_crop_img)
            if best_sim is not None:
                vis_sim = float(best_sim)
            else:
                other_has_crops = any(
                    bool(_tracklet_get(other_tr, 'crops'))
                    for other_idx, other_tr, _ in candidates
                    if other_idx != idx
                )
                vis_sim = 0.25 if other_has_crops else 0.50
            score = vis_sim

            # Screen-side correspondence bonus (+0.10)
            closest_pt = min(near_pts, key=lambda pt: abs(float(pt.get('t', 0.0)) - t_zr))
            box = closest_pt.get('box') or closest_pt.get('box_norm')
            if box is not None and len(box) == 4:
                u = (float(box[0]) + float(box[2])) / 2.0
                u_norm = u if max(abs(float(v)) for v in box) <= 1.5 else u / 1920.0
                if (z_side == 'left' and u_norm < 0.5) or (z_side == 'right' and u_norm >= 0.5):
                    score += 0.10

            scored.append((score, idx, tr))

        if not scored:
            continue
        scored.sort(key=lambda item: item[0], reverse=True)
        top_score, top_idx, top_tr = scored[0]
        runner_up_score = scored[1][0] if len(scored) > 1 else 0.0
        margin = top_score - runner_up_score

        if (len(scored) == 1 and top_score >= 0.35) or (len(scored) > 1 and margin >= min_margin and top_score >= 0.45):
            evidence = _seed_tracklet_evidence(top_tr)
            ev_item = {
                'source': 'zoom_crop',
                'team': team,
                'confidence': float(conf),
                't': float(t_zr),
                'crop_sha256': zr.get('crop_sha256'),
                'margin': round(float(margin), 3),
            }
            evidence.append(ev_item)
            votes = _tracklet_get(top_tr, 'votes')
            if votes is None:
                votes = {}
                _tracklet_set(top_tr, 'votes', votes)
            votes[team] = votes.get(team, 0.0) + float(conf)
            linked.append({**ev_item, 'tracklet_index': top_idx, 'score': round(float(top_score), 3)})

    return linked


def _top_vote_team(votes):
    valid = {
        str(k): float(v)
        for k, v in (votes or {}).items()
        if k is not None and isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0
    }
    if not valid:
        return None
    return max(valid.items(), key=lambda kv: kv[1])[0]


def read_tracker_numbers(tracker, client, roster, workers=8, zoom_crops=None, preprocess=True):
    """Read close-up bumper crops on every tracklet whose votes are empty, weak, or split, plus optional zoom crops."""
    tracklets = getattr(tracker, 'tracklets', None) or []
    substantial = [
        tr for tr in tracklets
        if len(_tracklet_get(tr, 'points') or ()) >= 3
        and _tracklet_get(tr, 'crops')
    ]
    top_team_counts = {}
    for tr in substantial:
        alliance = _tracklet_get(tr, 'alliance')
        top_t = _top_vote_team(_tracklet_get(tr, 'votes'))
        if alliance and top_t:
            top_team_counts[(alliance, top_t)] = top_team_counts.get((alliance, top_t), 0) + 1

    to_read = []
    for tr in substantial:
        alliance = _tracklet_get(tr, 'alliance')
        votes = _tracklet_get(tr, 'votes')
        top_t = _top_vote_team(votes)
        shared = bool(alliance and top_t and top_team_counts.get((alliance, top_t), 0) >= 2)
        if needs_closeup_reads(votes, evidence=_tracklet_get(tr, 'evidence'), shared_top_team=shared):
            to_read.append(tr)

    def _prep_crop(crop_img, bumper_band_norm=None):
        if not preprocess or crop_img is None:
            return crop_img
        closeup = prepare_bumper_closeup(crop_img, bumper_band_norm=bumper_band_norm)
        return closeup if closeup is not None else crop_img

    def _read_one(tr):
        prepared_crops = []
        for c in _tracklet_get(tr, 'crops') or ():
            if isinstance(c, dict):
                c_img, band_norm = c.get('crop'), c.get('bumper_band_norm')
            elif isinstance(c, (tuple, list)):
                c_img = c[2] if len(c) >= 3 else None
                band_norm = None
            else:
                c_img, band_norm = c, None
            prepared_crops.append(_prep_crop(c_img, bumper_band_norm=band_norm))
        try:
            return tr, read_bumper_numbers(client, prepared_crops, roster[_tracklet_get(tr, 'alliance')])
        except Exception as error:  # noqa: BLE001
            if type(error).__name__ == 'BudgetExceeded':
                raise
            return tr, []

    if len(to_read) <= 1 or workers <= 1:
        results = [_read_one(tr) for tr in to_read]
    else:
        with ThreadPoolExecutor(min(max(1, int(workers)), len(to_read))) as pool:
            results = list(pool.map(_read_one, to_read))

    closeup_reads = 0
    for tr, reads in results:
        evidence = _seed_tracklet_evidence(tr)
        votes = _tracklet_get(tr, 'votes')
        if votes is None:
            votes = {}
            _tracklet_set(tr, 'votes', votes)
        for crop_tuple, (team, conf) in zip(_tracklet_get(tr, 'crops') or (), reads):
            if team and conf > 0:
                if isinstance(crop_tuple, dict):
                    t_val = float(crop_tuple['t']) if crop_tuple.get('t') is not None else None
                    crop_img = crop_tuple.get('crop')
                elif isinstance(crop_tuple, (tuple, list)):
                    t_val = float(crop_tuple[1]) if len(crop_tuple) >= 2 else None
                    crop_img = crop_tuple[2] if len(crop_tuple) >= 3 else None
                else:
                    t_val = None
                    crop_img = crop_tuple
                digest = hashlib.sha256(np.ascontiguousarray(crop_img).tobytes()).hexdigest() if isinstance(crop_img, np.ndarray) else None
                evidence.append({
                    'source': 'closeup_crop',
                    'team': str(team),
                    'confidence': float(conf),
                    't': t_val,
                    'crop_sha256': digest,
                })
                votes[team] = votes.get(team, 0.0) + float(conf)
                closeup_reads += 1

    zoom_reads_count = 0
    zoom_linked_count = 0
    if zoom_crops:
        for alliance in ('red', 'blue'):
            alliance_zooms = [z for z in zoom_crops if z.get('alliance') == alliance and z.get('crop') is not None]
            if not alliance_zooms or not roster.get(alliance):
                continue
            prepared_zooms = [
                _prep_crop(z['crop'], bumper_band_norm=z.get('bumper_band_norm'))
                for z in alliance_zooms
            ]
            try:
                reads = read_bumper_numbers(client, prepared_zooms, roster[alliance])
            except Exception as error:  # noqa: BLE001
                if type(error).__name__ == 'BudgetExceeded':
                    raise
                reads = []
            resolved_zooms = []
            for z, (team, conf) in zip(alliance_zooms, reads):
                if team and conf > 0:
                    zoom_reads_count += 1
                    resolved_zooms.append({**z, 'team': team, 'confidence': conf})
            if resolved_zooms:
                linked = match_zoom_reads_to_tracklets(resolved_zooms, tracklets)
                zoom_linked_count += len(linked)

    return {
        'tracklets_read': len(to_read),
        'closeup_reads': closeup_reads,
        'zoom_reads': zoom_reads_count,
        'zoom_linked': zoom_linked_count,
    }
