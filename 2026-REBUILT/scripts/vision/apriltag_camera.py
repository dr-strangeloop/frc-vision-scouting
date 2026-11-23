#!/usr/bin/env python3
"""Calibrate a broadcast camera from the field's AprilTags: floor homography and height projection, no hand marks.

The HUBs, trenches and alliance walls carry AprilTags at surveyed 3D positions (WPILib's AprilTagFieldLayout, e.g.
`2026-rebuilt-welded.json`). Their corners are detected in several frames of the static wide shot (median per corner,
so passing robots and noise drop out), the camera's field of view and pose are solved with solvePnP (scanning the
field of view for the lowest reprojection error, then dropping tags that do not fit), and the 3x4 projection gives
both the floor homography and the height projection the v2 counter needs for the HUB openings and lights.

    python3 scripts/vision/apriltag_camera.py --video match.mp4 --layout 2026-rebuilt-welded.json \\
        --times 8,10,12,14 --roi 0,0.15,1,0.65 --out calibration.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

TAG_SIZE_M = 0.1651          # FRC 6.5 in tags
FOV_RANGE_DEG = (20.0, 100.0)
MAX_RMS_PX = 3.0             # accept a solve only when the tags reproject this well
MIN_TAGS = 3


def load_layout(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    tags = {}
    for tag in payload.get('tags') or payload.get('field-tags') or []:  # WPILib renamed 'tags' to 'field-tags'
        pose = tag.get('pose') or {}
        t, q = pose.get('translation') or {}, (pose.get('rotation') or {}).get('quaternion') or {}
        tags[int(tag['ID'])] = (np.array([t['x'], t['y'], t['z']], float),
                                (float(q.get('W', 1)), float(q.get('X', 0)), float(q.get('Y', 0)), float(q.get('Z', 0))))
    return tags


def _rotation(w, x, y, z):
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def tag_corners(translation, quaternion, size=TAG_SIZE_M):
    """Field-frame corners in the detector's order (top-left, top-right, bottom-right, bottom-left, seen from in
    front of the tag: image right is the tag's +Y, image up is +Z)."""
    h = size / 2.0
    local = np.array([[0.0, -h, h], [0.0, h, h], [0.0, h, -h], [0.0, -h, -h]])
    return translation + local @ _rotation(*quaternion).T


def detect(frame, upscale=2.0):
    """{tag id: (4, 2) corner pixels} for AprilTag 36h11 tags in `frame`. Broadcast tags are only 20-30 px wide, so
    the frame is upscaled and the detector allows small markers. A tag id seen twice (a close-up panel showing the
    same tag) is ambiguous and left out."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    params = cv2.aruco.DetectorParameters()
    params.minMarkerPerimeterRate = 0.005
    params.adaptiveThreshWinSizeMin, params.adaptiveThreshWinSizeMax, params.adaptiveThreshWinSizeStep = 3, 53, 4
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if upscale and upscale != 1.0:
        gray = cv2.resize(gray, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    corners, ids, _ = cv2.aruco.ArucoDetector(dictionary, params).detectMarkers(gray)
    if ids is None:
        return {}
    ids = [int(i) for i in ids.ravel()]
    return {i: c.reshape(4, 2).astype(float) / (upscale or 1.0) for i, c in zip(ids, corners) if ids.count(i) == 1}


def median_corners(detections, min_frames=2):
    """Per tag, the median of each corner over the frames it was seen in (at least `min_frames`)."""
    seen = {}
    for frame_dets in detections:
        for tid, c in frame_dets.items():
            seen.setdefault(tid, []).append(c)
    return {tid: np.median(np.stack(cs), axis=0) for tid, cs in seen.items() if len(cs) >= min_frames}


def camera_matrix(width, height, fov_deg):
    f = (width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    return np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1.0]])


def _solve(obj, img, width, height, fov):
    K = camera_matrix(width, height, fov)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
    err = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)
    return {'K': K, 'rvec': rvec, 'tvec': tvec, 'errors': err, 'rms': float(np.sqrt(np.mean(err ** 2))), 'fov': fov}


def _best_fov(obj, img, width, height):
    """The field of view (and its solution) with the lowest reprojection error: coarse scan, then a bounded search."""
    from scipy.optimize import minimize_scalar
    coarse = [s for s in (_solve(obj, img, width, height, f) for f in np.arange(*FOV_RANGE_DEG, 2.0)) if s]
    if not coarse:
        return None
    f0 = min(coarse, key=lambda s: s['rms'])['fov']
    res = minimize_scalar(lambda f: (_solve(obj, img, width, height, f) or {'rms': 1e9})['rms'],
                          bounds=(max(FOV_RANGE_DEG[0], f0 - 2.0), min(FOV_RANGE_DEG[1], f0 + 2.0)), method='bounded',
                          options={'xatol': 0.01})
    return _solve(obj, img, width, height, float(res.x))


def _fit(corners, layout, tags, width, height):
    obj = np.concatenate([tag_corners(*layout[t]) for t in tags])
    img = np.concatenate([corners[t] for t in tags])
    return _best_fov(obj, img, width, height)


def solve(corners, layout, width, height, good_rms_px=1.0):
    """Best camera for the detected corners. While the fit is worse than `good_rms_px`, the tag whose removal helps
    most is dropped if that removal more than halves the error (a misread or moved tag). Returns the solution with
    `tags`, `dropped` and per-tag errors, or raises ValueError."""
    use = sorted(t for t in corners if t in layout)
    dropped = sorted(set(corners) - set(use))
    if len(use) < MIN_TAGS:
        raise ValueError(f'only {len(use)} layout tags usable (need {MIN_TAGS})')
    best = _fit(corners, layout, use, width, height)
    while best and best['rms'] > good_rms_px and len(use) > MIN_TAGS:
        trials = [(t, _fit(corners, layout, [u for u in use if u != t], width, height)) for t in use]
        t, cand = min(((t, c) for t, c in trials if c), key=lambda tc: tc[1]['rms'])
        if cand['rms'] > 0.5 * best['rms']:
            break
        use.remove(t)
        dropped.append(t)
        best = cand
    if best is None:
        raise ValueError('solvePnP failed')
    per_tag = best['errors'].reshape(len(use), 4).mean(axis=1)
    best.update(tags=use, dropped=sorted(dropped), per_tag_px={t: round(float(e), 2) for t, e in zip(use, per_tag)})
    return best


def calibration_from_solution(sol, width, height, ref_t_s, roi, source):
    """A counter-ready calibration dict from a solvePnP solution."""
    R, _ = cv2.Rodrigues(sol['rvec'])
    P = sol['K'] @ np.hstack([R, sol['tvec'].reshape(3, 1)])
    F = P[:, [0, 1, 3]]
    F, col = F / F[2, 2], P[:, 2] / P[2, 3]
    h_norm = np.linalg.inv(F) @ np.diag([float(width), float(height), 1.0])
    h_norm = h_norm / h_norm[2, 2]
    centre = (-R.T @ sol['tvec']).ravel()
    return {
        'homography_norm': h_norm.tolist(),
        'height_projection': {'floor_to_image': F.tolist(), 'height_column': col.tolist(), 'rms_px': sol['rms'],
                              'approximate': False, 'source': 'apriltag'},
        'source_width': int(width), 'source_height': int(height),
        'ref_t_s': float(ref_t_s), 'roi': list(roi),
        'light_face_y_sign': 1.0 if centre[1] > 4.035 else -1.0,
        'camera': {'fov_deg': round(sol['fov'], 2), 'position_m': [round(float(c), 3) for c in centre]},
        'apriltag': {'tags': sol['tags'], 'dropped': sol['dropped'], 'rms_px': round(sol['rms'], 3), 'per_tag_px': sol['per_tag_px']},
        'calibration_source': source,
    }


def frames_at(video, times):
    cap = cv2.VideoCapture(str(video))
    try:
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if ok:
                yield t, frame
    finally:
        cap.release()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--video', required=True)
    ap.add_argument('--layout', required=True)
    ap.add_argument('--times', required=True, help='comma-separated video seconds of the static wide shot')
    ap.add_argument('--roi', default='0,0,1,1', help='x0,y0,x1,y1 of the wide field view (normalized)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    times = [float(t) for t in args.times.split(',')]
    roi = [float(v) for v in args.roi.split(',')]
    layout = load_layout(args.layout)
    dets, size = [], None
    for _, frame in frames_at(args.video, times):
        h, w = frame.shape[:2]
        size = (w, h)
        x0, y0, x1, y1 = int(roi[0] * w), int(roi[1] * h), int(roi[2] * w), int(roi[3] * h)
        found = detect(frame[y0:y1, x0:x1])
        dets.append({t: c + [x0, y0] for t, c in found.items()})
    if not size:
        sys.exit('could not read any frame')
    corners = median_corners(dets, min_frames=max(1, len(dets) // 2))
    sol = solve(corners, layout, *size)
    if sol['rms'] > MAX_RMS_PX:
        sys.exit(f"tags reproject at {sol['rms']:.2f} px RMS (limit {MAX_RMS_PX}); not saving")
    cal = calibration_from_solution(sol, *size, times[0], roi, f'AprilTags {sol["tags"]} in {Path(args.video).name} at {args.times}s')
    Path(args.out).write_text(json.dumps(cal, indent=1) + '\n', encoding='utf-8')
    print(f"{args.out}: {len(sol['tags'])} tags {sol['tags']} (dropped {sol['dropped']}), RMS {sol['rms']:.2f} px, "
          f"fov {sol['fov']:.1f} deg, camera at {cal['camera']['position_m']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
