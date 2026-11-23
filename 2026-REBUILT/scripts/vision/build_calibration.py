#!/usr/bin/env python3
"""Turn hand marks from tools/calibration_marker.html into a counter-ready field calibration.

    python3 scripts/vision/build_calibration.py --marks marks.json --out calibration.json

Marks are picture pixels of named field landmarks (floor, z = 0) and of HUB rim corners (`rim:<base landmark id>`,
z = 1.829 m, straight above that base corner). Floor marks give the floor homography (field_landmarks.py, the same
solver as the app); rim marks give the height projection the v2 counter needs for the HUB openings and lights
(terrain_projection.fit_height_column). The build refuses a calibration whose fits are poor rather than guess.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import field_landmarks as fl  # noqa: E402
from fuel_count import RIM_HEIGHT_M  # noqa: E402
from terrain_projection import fit_height_column  # noqa: E402

MAX_FLOOR_RMS_M = 0.35
MIN_FLOOR_POINTS = 5
MIN_RIM_POINTS = 4


def _camera_y(height_projection):
    """Field y (m) of the camera centre of the fitted height projection (its 3x4 matrix's null vector), or None.
    It decides which HUB face is lit toward the camera; the marked rims' mean y cannot, since rims marked on both
    faces of a HUB average to its centre line."""
    f = np.asarray(height_projection['floor_to_image'], float)
    p = np.column_stack([f[:, 0], f[:, 1], np.asarray(height_projection['height_column'], float), f[:, 2]])
    c = np.linalg.svd(p)[2][-1]
    return float(c[1] / c[3]) if abs(c[3]) > 1e-12 else None


def build(marks):
    width, height = int(marks['width']), int(marks['height'])
    points = marks.get('points') or {}
    floor, rims = [], []
    for pid, (u, v) in points.items():
        if pid.startswith('rim:'):
            base = fl.LANDMARK_BY_ID.get(pid[4:])
            if base is None:
                raise ValueError(f'unknown rim landmark {pid}')
            rims.append({'id': pid, 'x': base['x'], 'y': base['y'], 'z': RIM_HEIGHT_M, 'u': float(u), 'v': float(v)})
        elif pid in fl.LANDMARK_BY_ID:
            lm = fl.LANDMARK_BY_ID[pid]
            floor.append({'id': pid, 'name': lm.get('name'), 'x': lm['x'], 'y': lm['y'],
                          'u': float(u) / width, 'v': float(v) / height})
        else:
            raise ValueError(f'unknown landmark {pid}')
    if len(floor) < MIN_FLOOR_POINTS:
        raise ValueError(f'mark at least {MIN_FLOOR_POINTS} floor landmarks spread across the field (got {len(floor)})')
    if len(rims) < MIN_RIM_POINTS:
        raise ValueError(f'mark at least {MIN_RIM_POINTS} HUB rim corners (got {len(rims)})')
    solved = fl.solve_field_calibration(floor)
    if not solved.get('ok'):
        raise ValueError(f"floor homography failed: {solved.get('message') or solved.get('reason')}")
    rms_m = float(solved.get('rms_m') or 0.0)
    if rms_m > MAX_FLOOR_RMS_M:
        raise ValueError(f'floor marks disagree by {rms_m:.2f} m RMS; re-check them (limit {MAX_FLOOR_RMS_M} m)')
    h_norm = np.asarray(solved['homography_norm'], float)
    h_px = h_norm @ np.diag([1.0 / width, 1.0 / height, 1.0])
    hp = fit_height_column(h_px, rims)
    camera_y = _camera_y(hp)
    face_y = camera_y if camera_y is not None else np.mean([r['y'] for r in rims])
    calibration = {
        'homography_norm': h_norm.tolist(),
        'landmarks': floor,
        'vertical_landmarks': [{k: r[k] for k in ('x', 'y', 'z', 'u', 'v')} for r in rims],
        'height_projection': hp,
        'source_width': width, 'source_height': height,
        'ref_t_s': float(marks.get('frame_t_s') or 0.0),
        'roi': marks.get('roi') or [0.0, 0.0, 1.0, 1.0],
        'light_face_y_sign': 1.0 if face_y > 4.035 else -1.0,
        'rms_m': rms_m, 'loo_p95_m': solved.get('loo_p95_m'),
        'calibration_source': f"hand marks ({marks.get('marked_by') or 'unknown'}) on {marks.get('frame') or 'a frame'}",
    }
    return calibration


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--marks', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    cal = build(json.loads(Path(args.marks).read_text(encoding='utf-8')))
    Path(args.out).write_text(json.dumps(cal, indent=1) + '\n', encoding='utf-8')
    hp = cal['height_projection']
    print(f"{args.out}: floor {len(cal['landmarks'])} marks, RMS {cal['rms_m']:.3f} m; rims {len(cal['vertical_landmarks'])} "
          f"marks, RMS {hp['rms_px']:.1f} px (max {hp['max_px']:.1f}); light face y sign {cal['light_face_y_sign']:+.0f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
