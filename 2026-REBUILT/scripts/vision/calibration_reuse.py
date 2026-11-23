"""Reuse an event's field calibration across matches when the broadcast wide camera is the same.

Broadcast wide shots at an event are usually identical from match to match, save for a pixel or two
of camera settle or framing. Given a candidate match's calibrated reference frame and a target frame
from the uncalibrated match, we match ORB features inside the calibration ROI, fit a homography with
RANSAC, and accept it only when it is near identity at 1280 wide (< 6 px median reprojection, >= 60
inliers, scale 0.97-1.03). When accepted, the candidate's normalized homography is composed with that
small target->candidate correction so landmarks map from the target frame to field meters.
"""
import math

import cv2
import numpy as np

from field_landmarks import solve_field_calibration
from roi import roi_rect_mask

CANONICAL_WIDTH = 1280
MAX_MEDIAN_REPROJ_PX = 6.0
MIN_INLIERS = 60
MIN_SCALE = 0.97
MAX_SCALE = 1.03
MAX_SOURCE_LOO_P95_M = 0.45


def _resize_to_width(frame, width):
    h, w = frame.shape[:2]
    if w == width:
        return frame
    target_h = max(1, int(round(h * (width / float(w)))))
    interp = cv2.INTER_AREA if w > width else cv2.INTER_LINEAR
    return cv2.resize(frame, (width, target_h), interpolation=interp)


def _gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame


def _fit_homography_to_candidate(cand_gray, targ_gray, roi):
    """Fit H mapping target pixels -> candidate pixels (at canonical width) inside roi."""
    h_c, w_c = cand_gray.shape[:2]
    h_t, w_t = targ_gray.shape[:2]
    mask_c = roi_rect_mask(roi, w_c, h_c)
    mask_t = roi_rect_mask(roi, w_t, h_t)
    orb = cv2.ORB_create(nfeatures=4000, fastThreshold=10)
    kp_c, desc_c = orb.detectAndCompute(cand_gray, mask_c)
    kp_t, desc_t = orb.detectAndCompute(targ_gray, mask_t)
    if desc_c is None or desc_t is None or len(kp_c) < MIN_INLIERS or len(kp_t) < MIN_INLIERS:
        return None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(desc_t, desc_c)
    if len(matches) < MIN_INLIERS:
        return None
    pts_t = np.float32([kp_t[m.queryIdx].pt for m in matches])
    pts_c = np.float32([kp_c[m.trainIdx].pt for m in matches])
    h_mat, inlier_mask = cv2.findHomography(pts_t, pts_c, cv2.RANSAC, 4.0, maxIters=3000, confidence=0.995)
    if h_mat is None or inlier_mask is None:
        return None
    keep = inlier_mask.ravel().astype(bool)
    inliers = int(keep.sum())
    if inliers < MIN_INLIERS:
        return None
    refined, _ = cv2.findHomography(pts_t[keep], pts_c[keep], 0)
    if refined is not None and np.isfinite(refined).all() and abs(refined[2, 2]) > 1e-9:
        h_mat = refined / refined[2, 2]
    else:
        h_mat = h_mat / h_mat[2, 2]
    return h_mat, pts_t[keep], pts_c[keep], inliers


def _near_identity(h_mat, pts_t, pts_c, width, height):
    """Check whether h_mat (target -> candidate at canonical width) is near identity."""
    projected = cv2.perspectiveTransform(pts_t.reshape(-1, 1, 2), h_mat).reshape(-1, 2)
    reproj_err = np.linalg.norm(projected - pts_c, axis=1)
    if float(np.median(reproj_err)) >= MAX_MEDIAN_REPROJ_PX:
        return None
    # Scale of the linear part and perspective terms
    a = h_mat[:2, :2]
    det = float(np.linalg.det(a))
    if det <= 0:
        return None
    sv = np.linalg.svd(a, compute_uv=False)
    scale = math.sqrt(det)
    if not (MIN_SCALE <= scale <= MAX_SCALE and MIN_SCALE <= float(sv[0]) <= MAX_SCALE and MIN_SCALE <= float(sv[1]) <= MAX_SCALE):
        return None
    # Off-diagonal rotation/shear must be small for a stationary broadcast mount
    if abs(float(a[0, 1])) > 0.03 or abs(float(a[1, 0])) > 0.03:
        return None
    # Near identity: where inlier features actually move between target and candidate frames
    identity_disp = np.linalg.norm(projected - pts_t, axis=1)
    residual_px = float(np.median(identity_disp))
    if residual_px >= MAX_MEDIAN_REPROJ_PX:
        return None
    return residual_px, float(np.median(reproj_err)), scale


def reuse(candidate_frame_bgr, target_frame_bgr, calibration, reused_from=None, trust_source=False):
    """Reuse `calibration` on `target_frame_bgr` when the camera matches `candidate_frame_bgr`.

    Returns a copy of `calibration` with `homography_norm` composed with the small target->candidate
    correction and `reused_from` metadata, or None when the frames do not match closely enough.

    `trust_source` skips the source-quality gates (leave-one-out error, landmark geometry) for a calibration a person
    has checked by other means, e.g. a hand-marked one whose height projection sits on the measured HUB rims. The
    camera-match checks always apply, and the result records that the gate was skipped.
    """
    if candidate_frame_bgr is None or target_frame_bgr is None or not calibration:
        return None
    h_norm = calibration.get('homography_norm')
    if h_norm is None:
        return None
    h_norm = np.asarray(h_norm, dtype=float)
    if h_norm.shape != (3, 3) or not np.isfinite(h_norm).all():
        return None

    cal_loo_p95 = None if trust_source else calibration.get('loo_p95_m')
    if cal_loo_p95 is not None:
        try:
            if not math.isfinite(float(cal_loo_p95)) or float(cal_loo_p95) > MAX_SOURCE_LOO_P95_M:
                return None
        except (TypeError, ValueError):
            return None

    landmarks = calibration.get('landmarks')
    if isinstance(landmarks, list) and len(landmarks) >= 6 and not trust_source:
        solved = solve_field_calibration(landmarks)
        if not solved.get('ok') or solved.get('two_line_cross') or int(solved.get('n_general_position') or 0) < 4:
            return None
        loo_p95 = solved.get('loo_p95_m')
        if loo_p95 is not None and (not math.isfinite(float(loo_p95)) or float(loo_p95) > MAX_SOURCE_LOO_P95_M):
            return None

    cand = _resize_to_width(candidate_frame_bgr, CANONICAL_WIDTH)
    targ = _resize_to_width(target_frame_bgr, CANONICAL_WIDTH)
    cand_gray = _gray(cand)
    targ_gray = _gray(targ)
    h_c, w_c = cand_gray.shape[:2]
    h_t, w_t = targ_gray.shape[:2]
    if abs(h_c - h_t) > 4:
        return None

    roi = calibration.get('roi')
    fitted = _fit_homography_to_candidate(cand_gray, targ_gray, roi)
    if fitted is None:
        return None
    h_t2c_px, pts_t, pts_c, inliers = fitted
    checked = _near_identity(h_t2c_px, pts_t, pts_c, w_t, h_t)
    if checked is None:
        return None
    residual_px, reproj_px, scale = checked

    # Convert pixel homography (target_px -> cand_px) to normalized coordinates (target_norm -> cand_norm)
    to_cand_norm = np.diag([1.0 / w_c, 1.0 / h_c, 1.0])
    from_targ_norm = np.diag([float(w_t), float(h_t), 1.0])
    h_t2c_norm = to_cand_norm @ h_t2c_px @ from_targ_norm
    composed = h_norm @ h_t2c_norm
    if abs(composed[2, 2]) > 1e-12:
        composed = composed / composed[2, 2]

    source_meta = reused_from if isinstance(reused_from, dict) else ({'view_id': reused_from} if reused_from else {})
    meta = {
        **source_meta,
        'inliers': inliers,
        'residual_px': round(residual_px, 3),
        'reprojection_px': round(reproj_px, 3),
        'scale': round(scale, 4),
        **({'source_quality_gate': 'skipped (trusted source)'} if trust_source else {}),
    }
    out = dict(calibration)
    out['homography_norm'] = composed.tolist()
    hp = carry_height_projection(calibration, h_t2c_px, (w_c, h_c), (w_t, h_t))
    if hp is not None:
        out['height_projection'] = hp
        meta['height_projection'] = 'carried'
    elif calibration.get('height_projection'):
        out.pop('height_projection', None)               # cannot be moved: v1 HUB rings rather than misplaced openings
        meta['height_projection'] = 'dropped'
    # Picture-space marks from the source view are not valid on the target view.
    out.pop('vertical_landmarks', None)
    out['reused_from'] = meta
    return out


def carry_height_projection(calibration, h_t2c_px, cand_size, targ_size):
    """The source calibration's `height_projection` moved onto the target view, or None when it has none.

    The height projection maps field (x, y, z) to source-picture pixels (`source_width` x `source_height`). The target
    view is the same camera up to `h_t2c_px` (target -> candidate pixels at the canonical width), so target pixels are
    `inv(K_t) @ inv(h_t2c_px) @ K_c` of source pixels, with K scaling source pixels to each canonical frame. Both its
    floor homography and its height column are multiplied by that, which keeps HUB openings and light boxes on the rims.
    """
    hp = (calibration or {}).get('height_projection')
    if not hp or not hp.get('floor_to_image') or not hp.get('height_column'):
        return None
    g = np.asarray(hp['floor_to_image'], float)
    col = np.asarray(hp['height_column'], float)
    if g.shape != (3, 3) or col.shape != (3,) or not (np.isfinite(g).all() and np.isfinite(col).all()):
        return None
    sw = float(calibration.get('source_width') or 1920)
    sh = float(calibration.get('source_height') or 1080)
    k_c = np.diag([cand_size[0] / sw, cand_size[1] / sh, 1.0])
    k_t = np.diag([targ_size[0] / sw, targ_size[1] / sh, 1.0])
    try:
        a = np.linalg.inv(k_t) @ np.linalg.inv(np.asarray(h_t2c_px, float)) @ k_c
    except np.linalg.LinAlgError:
        return None
    g2, col2 = a @ g, a @ col
    scale = g2[2, 2] if abs(g2[2, 2]) > 1e-12 else 1.0
    return {**hp, 'floor_to_image': (g2 / scale).tolist(), 'height_column': (col2 / scale).tolist(), 'carried_over': True}
