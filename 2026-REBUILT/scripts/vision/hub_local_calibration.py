#!/usr/bin/env python3
"""Calibrate the picture around each HUB from the two AprilTags on its front face (no hand marks).

Broadcast field cameras are wide and strongly barrel-distorted (at 2026arc no single camera model fits the field's
tags: 34 px RMS even for a general 3x4 projection), and the HUB side-face tags are too foreshortened to detect. But
the v2 counter only needs the geometry at each HUB: the opening ring at rim height and the lit band on the front face.
Over one HUB the distortion is small, so each HUB gets its own local camera:

  * The two front-face tags (eight corners, all in the face plane) give the face-plane homography, which places the
    lit band and the front rim exactly.
  * Both HUBs' front faces lie in one plane; a homography fitted across all four tags, 7 m apart, carries enough
    perspective to fix the focal length (two small tags alone do not). With it, each HUB's own face homography
    becomes a local 3x4 projection (square pixels, distortion centre at the picture centre): rotation from the face's
    two in-plane directions, depth from their cross product. That places the opening ring 0.6 m behind the face.

The calibration stores one height projection per HUB (`height_projection_by_hub`, used by fuel_count.hub_openings and
hub_lights.light_rois) and a floor homography fitted to floor points near both HUBs (used by the fuel pass for
registration and the field mask).

    python3 scripts/vision/hub_local_calibration.py --video match.mp4 --layout 2026-rebuilt-welded.json \\
        --times 10,20,30,40 --roi 0,0.15,1,0.65 --out calibration.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import apriltag_camera as ac  # noqa: E402

HUB_CENTRES = {'blue_hub': (4.625, 4.035), 'red_hub': (11.916, 4.035)}
FACE_HALF_M = 0.597
MAX_FACE_RMS_PX = 1.5
# The counter was tuned with Chezy Champs' hand calibration, whose "rim" (z = 1.829 m) is the top of the HUB's dark
# front face, which the field's AprilTags place at 1.233 m (1.19-1.26 m over the four marked corners). Heights here
# are scaled so that 1.233 m maps to the model's 1.829 m, putting openings and light boxes where they were tuned.
HAND_RIM_PHYSICAL_M = 1.233
MODEL_RIM_M = 1.829
# The lit panels run from the dark-face top (1.233 m) up to the opening (1.83 m). The hand calibration's light band
# (model 1.75-1.85 m) only just touches their bottom edge, which was enough at Chezy Champs but misses them at 2026arc,
# so tag calibrations sample the panels' lower part, physically 1.30-1.50 m (above the face, below balls on top).
LIGHT_PANEL_PHYSICAL_M = (1.30, 1.50)


def _detector():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    params = cv2.aruco.DetectorParameters()
    params.minMarkerPerimeterRate = 0.003
    params.polygonalApproxAccuracyRate = 0.05
    params.adaptiveThreshWinSizeMin, params.adaptiveThreshWinSizeMax, params.adaptiveThreshWinSizeStep = 3, 63, 4
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(dictionary, params)


def _detect_in(gray, x0, y0, detector, variants):
    """{id: corners} in a gray patch at offset (x0, y0), trying each (scale, clahe) variant; first decode wins."""
    clahe = cv2.createCLAHE(2.0, (8, 8))
    found = {}
    for scale, use_clahe in variants:
        g = clahe.apply(gray) if use_clahe else gray
        big = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        corners, ids, _ = detector.detectMarkers(big)
        ids = [] if ids is None else [int(i) for i in ids.ravel()]
        for i, c in zip(ids, corners):
            if ids.count(i) == 1 and i not in found:
                found[i] = c.reshape(4, 2) / scale + [x0, y0]
    return found


def detect_tags(frames, roi, upscale=3.0, locate_frames=8, pad_px=60, keep_ids=None):
    """{tag id: (4, 2) median corners} over `frames`, in the region of interest.

    Small tags decode only under some preprocessing, which is slow on a whole upscaled frame. So the tags are first
    located with a cheap 2x pass over a few frames, then every frame is searched only in small crops around them with
    several variants (3x and 4x, with and without contrast enhancement). Ids seen twice (close-up panels) are dropped.
    The wide camera is static, so a tag seen in a tenth of the frames is enough."""
    detector = _detector()
    boxes = {}
    for frame in frames[:locate_frames]:
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = int(roi[0] * w), int(roi[1] * h), int(roi[2] * w), int(roi[3] * h)
        gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        for tid, c in _detect_in(gray, x0, y0, detector, ((2.0, True), (2.0, False))).items():
            if keep_ids is not None and tid not in keep_ids:
                continue                                   # only the tags the calibration uses get the slow search
            bx = (int(c[:, 0].min()) - pad_px, int(c[:, 1].min()) - pad_px, int(c[:, 0].max()) + pad_px, int(c[:, 1].max()) + pad_px)
            old = boxes.get(tid, bx)
            boxes[tid] = (min(old[0], bx[0]), min(old[1], bx[1]), max(old[2], bx[2]), max(old[3], bx[3]))
    # merge the boxes of neighbouring tags (a HUB face) into one crop each
    crops = []
    for b in sorted(boxes.values()):
        for i, c in enumerate(crops):
            if b[0] <= c[2] and c[0] <= b[2] and b[1] <= c[3] and c[1] <= b[3]:
                crops[i] = (min(c[0], b[0]), min(c[1], b[1]), max(c[2], b[2]), max(c[3], b[3]))
                break
        else:
            crops.append(b)
    variants = ((upscale, True), (upscale, False), (upscale + 1.0, True), (upscale + 1.0, False))
    dets = []
    for frame in frames:
        h, w = frame.shape[:2]
        found = {}
        for x0, y0, x1, y1 in crops:
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
            gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
            for tid, c in _detect_in(gray, x0, y0, detector, variants).items():
                found.setdefault(tid, c)
        dets.append(found)
    return ac.median_corners(dets, min_frames=max(2, len(dets) // 10))


def all_front_tags(layout):
    """Every HUB front-face tag id, on either face (the camera may be on either side)."""
    ids = set()
    for hub, (_, cy) in HUB_CENTRES.items():
        for face_y in (cy - FACE_HALF_M - 0.007, cy + FACE_HALF_M + 0.006):
            ids.update(front_tags(layout, hub, face_y))
    return ids


def front_tags(layout, hub, face_y):
    """Layout ids of the tags on `hub`'s face at y = `face_y` (the face toward the camera)."""
    cx, _ = HUB_CENTRES[hub]
    return sorted(t for t, (pos, _) in layout.items()
                  if abs(pos[1] - face_y) < 0.05 and abs(pos[0] - cx) < FACE_HALF_M + 0.05 and 0.8 < pos[2] < 1.5)


def face_homography(corners, layout, tags):
    """(3x3 homography from face coordinates (x, z) to pixels, RMS px, corner roll) from the face tags' corners.

    The detector's corner order on real broadcast frames comes out rotated 180 degrees from the layout's (2026cc:
    11.8 px misfit as modelled, a clean fit rotated), so both orientations are tried for all tags together and the
    one that fits is kept."""
    best = None
    for roll in (0, 2, 1, 3):
        xz = np.concatenate([np.roll(ac.tag_corners(*layout[t]), -roll, axis=0)[:, [0, 2]] for t in tags])
        uv = np.concatenate([corners[t] for t in tags])
        H, _ = cv2.findHomography(xz, uv, 0)
        q = (H @ np.hstack([xz, np.ones((len(xz), 1))]).T).T
        rms = float(np.sqrt(np.mean(np.linalg.norm(q[:, :2] / q[:, 2:] - uv, axis=1) ** 2)))
        if best is None or rms < best[1]:
            best = (H, rms, roll)
    return best


def local_projection(H, face_y, centre_px, camera_side_y, f):
    """3x4 projection from the face-plane homography (x, z) -> pixels for focal length `f` (square pixels, distortion
    centre at `centre_px`): the face's x and z directions are the rotation's first and third columns, depth their
    cross product, signed so the camera is on the side of `camera_side_y` (negative: the y = 0 side).
    Returns (P, camera centre)."""
    cx, cy = centre_px
    T = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1.0]])
    K = np.array([[f, 0, 0], [0, f, 0], [0, 0, 1.0]])
    M = np.linalg.inv(K) @ T @ H
    lam = 2.0 / (np.linalg.norm(M[:, 0]) + np.linalg.norm(M[:, 1]))
    r1, r3, t = lam * M[:, 0], lam * M[:, 1], lam * M[:, 2]
    if t[2] < 0:                                       # the face must be in front of the camera
        r1, r3, t = -r1, -r3, -t
    u, _, vt = np.linalg.svd(np.stack([r1, r3], axis=1), full_matrices=False)
    r1, r3 = (u @ vt).T
    R = np.stack([r1, np.cross(r3, r1), r3], axis=1)   # world Y = Z x X: a proper rotation (det +1), never mirrored
    tt = t - face_y * R[:, 1]
    centre = -R.T @ tt
    if (centre[1] - face_y) * camera_side_y <= 0:
        raise ValueError('the solved camera is behind the HUB face it sees (corner order or layout mismatch)')
    P = np.linalg.inv(T) @ K @ np.hstack([R, tt.reshape(3, 1)])
    return P, centre


def focal_from_plane(H, centre_px):
    """Focal length (px) from one plane's homography (plane coordinates -> pixels) with square pixels and the
    principal point at `centre_px`: the plane's two directions must map to orthogonal, equal-length rotation columns.
    Least squares over both conditions; None when they give no real focal length."""
    cx, cy = centre_px
    Hc = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1.0]]) @ H
    a1, a3, g1, g3 = Hc[:2, 0], Hc[:2, 1], Hc[2, 0], Hc[2, 1]
    A = np.array([a1 @ a3, a1 @ a1 - a3 @ a3])        # coefficients of w = 1 / f^2
    b = -np.array([g1 * g3, g1 * g1 - g3 * g3])
    w = float(A @ b / (A @ A))
    return 1.0 / math.sqrt(w) if w > 0 else None


def projection_to_height_model(P, height_scale=HAND_RIM_PHYSICAL_M / MODEL_RIM_M):
    """Floor homography and height column; model heights are physical heights / `height_scale` (see above)."""
    F = P[:, [0, 1, 3]]
    s = F[2, 2]
    return {'floor_to_image': (F / s).tolist(), 'height_column': (height_scale * P[:, 2] / s).tolist(),
            'height_convention': f'model z {MODEL_RIM_M} m = physical {HAND_RIM_PHYSICAL_M} m (top of the HUB dark face)'}


def project(P, pts):
    q = (P @ np.hstack([np.asarray(pts, float), np.ones((len(pts), 1))]).T).T
    return q[:, :2] / q[:, 2:]


def calibrate(corners, layout, width, height, roi, ref_t_s=0.0, source=''):
    """The calibration dict, or raises ValueError when a HUB's face tags are missing or fit poorly."""
    # which side is the camera on? the face whose tags were found
    sides = {}
    for hub in HUB_CENTRES:
        for face_y, side in ((HUB_CENTRES[hub][1] - FACE_HALF_M - 0.007, -1.0), (HUB_CENTRES[hub][1] + FACE_HALF_M + 0.006, 1.0)):
            tags = [t for t in front_tags(layout, hub, face_y) if t in corners]
            if len(tags) >= 2:
                sides[hub] = (face_y, side, tags)
    if set(sides) != set(HUB_CENTRES):
        raise ValueError(f'need both front-face tags on each HUB; found {sides}')
    by_hub, info, floor_pts = {}, {}, []
    centre_px = (width / 2.0, height / 2.0)
    homographies = {}
    for hub, (face_y, side, tags) in sides.items():
        H, rms, roll = face_homography(corners, layout, tags)
        if rms > MAX_FACE_RMS_PX:
            raise ValueError(f'{hub} face tags disagree ({rms:.2f} px)')
        homographies[hub] = (H, rms, roll)
    # Both HUBs' front faces lie in one plane (y = 3.431 or 4.638 m). Fitted across both, 7 m apart, its homography
    # carries enough perspective to fix the focal length; each HUB keeps its own local face homography.
    all_tags = sorted(t for _, _, tags in sides.values() for t in tags)
    joint, joint_rms, _ = face_homography(corners, layout, all_tags)
    f = focal_from_plane(joint, centre_px)
    if f is None:
        raise ValueError('the front faces do not give a real focal length')
    face_ys = {round(fy, 3) for fy, _, _ in sides.values()}
    if len(face_ys) != 1:
        raise ValueError(f'the HUB faces seen are not one plane: {face_ys}')
    face_y, side, _ = next(iter(sides.values()))
    # One camera from the joint face plane: on real broadcasts it fits all four tags to ~0.4 px, while two 13 px tags
    # per HUB leave each local camera's depth too noisy to use.
    P_joint, cam = local_projection(joint, face_y, centre_px, side, f)
    for hub, (face_y, side, tags) in sides.items():
        H, rms, roll = homographies[hub]
        P = P_joint
        by_hub[hub] = projection_to_height_model(P)
        info[hub] = {'tags': tags, 'face_rms_px': round(rms, 3), 'corner_roll': roll, 'focal_px': round(f, 1),
                     'joint_face_rms_px': round(joint_rms, 2),
                     'camera_m': [round(float(c), 2) for c in cam]}
        hx, hy = HUB_CENTRES[hub]
        for dx in (-2.0, -1.0, 0.0, 1.0, 2.0):
            for dy in (-2.5, -1.2, 0.0, 1.2, 2.5):
                floor_pts.append(((hx + dx, hy + dy), project(P, [(hx + dx, hy + dy, 0.0)])[0]))
    xy = np.array([p for p, _ in floor_pts], float)
    uv = np.array([q for _, q in floor_pts], float)
    h_px, _ = cv2.findHomography(uv, xy, 0)
    h_norm = h_px @ np.diag([float(width), float(height), 1.0])
    h_norm = h_norm / h_norm[2, 2]
    side = np.mean([s for _, s, _ in sides.values()])
    first = next(iter(by_hub.values()))
    return {
        'homography_norm': h_norm.tolist(),
        'height_projection': first,                 # a fallback for code that takes one projection
        'height_projection_by_hub': by_hub,
        'source_width': int(width), 'source_height': int(height),
        'ref_t_s': float(ref_t_s), 'roi': list(roi),
        'light_face_y_sign': 1.0 if side > 0 else -1.0,
        'light_band_z': [round(z * MODEL_RIM_M / HAND_RIM_PHYSICAL_M, 3) for z in LIGHT_PANEL_PHYSICAL_M],
        'hub_local': info,
        'calibration_source': source or 'HUB front-face AprilTags (local per HUB)',
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--video', required=True)
    ap.add_argument('--layout', required=True)
    ap.add_argument('--times', required=True)
    ap.add_argument('--roi', default='0,0,1,1')
    ap.add_argument('--out', required=True)
    ap.add_argument('--corners-cache', help='reuse (or save) the detected tag corners here')
    args = ap.parse_args(argv)
    times = [float(t) for t in args.times.split(',')]
    roi = [float(v) for v in args.roi.split(',')]
    cache = Path(args.corners_cache) if args.corners_cache else None
    if cache and cache.is_file():
        data = json.loads(cache.read_text())
        w, h = data['size']
        corners = {int(k): np.array(v, float) for k, v in data['corners'].items()}
    else:
        frames = [f for _, f in ac.frames_at(args.video, times)]
        if not frames:
            sys.exit('could not read frames')
        h, w = frames[0].shape[:2]
        corners = detect_tags(frames, roi)
        if cache:
            cache.write_text(json.dumps({'size': [w, h], 'corners': {str(k): v.tolist() for k, v in corners.items()}}))
    cal = calibrate(corners, ac.load_layout(args.layout), w, h, roi, times[0],
                    f'HUB front-face AprilTags in {Path(args.video).name} at {args.times}s')
    Path(args.out).write_text(json.dumps(cal, indent=1) + '\n', encoding='utf-8')
    print(args.out, json.dumps(cal['hub_local']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
