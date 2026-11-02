"""Ground-plane field landmarks and DLT calibration solver ported from src/lib/vision/fieldCalibration.js.

Coordinates are in meters in the WPILib blue-origin frame (rebuiltField.json):
  - Origin (0, 0) is corner-blue-right (blue alliance wall, blue-driver right).
  - +x runs from the blue alliance wall (0) to the red alliance wall (FIELD['length'] = 16.541 m).
  - +y runs left as seen from the blue driver station (0 to FIELD['width'] = 8.07 m).
"""
import math

import numpy as np

from common import FIELD


def _round_mm(value):
    """Match JavaScript's Math.round(v * 1000) / 1000 (half rounds toward +infinity)."""
    fval = float(value)
    if not math.isfinite(fval):
        return fval
    scaled = fval * 1000.0
    rounded = math.floor(scaled + 0.5)
    # Return int when exact integer (0, etc.) or float otherwise; in Python 0.0 == 0,
    # and round(rounded / 1000.0, 3) avoids float representation noise.
    val = round(rounded / 1000.0, 3)
    return int(val) if val == int(val) else val


def landmarks(field=FIELD):
    """Ground landmarks with their field coordinates in meters (WPILib blue-origin frame)."""
    length = float(field['length'])
    width = float(field['width'])
    side_left = float(field['hub']['y']) + float(field['hub']['halfSize'])
    side_right = float(field['hub']['y']) - float(field['hub']['halfSize'])
    hub_near = float(field['startLineX'])
    hub_far = float(field['trench']['x1'])
    tower = field['tower']

    out = []

    def add(lid, name, x, y, group):
        out.append({
            'id': lid,
            'name': name,
            'x': _round_mm(x),
            'y': _round_mm(y),
            'group': group,
        })

    add('corner-blue-right', 'Field corner: blue wall, blue-driver right', 0, 0, 'Field')
    add('corner-blue-left', 'Field corner: blue wall, blue-driver left', 0, width, 'Field')
    add('corner-red-right', 'Field corner: red wall, red-driver right', length, width, 'Field')
    add('corner-red-left', 'Field corner: red wall, red-driver left', length, 0, 'Field')
    add('center-line-right', 'Center line meets the guardrail (blue-driver right)', length / 2.0, 0, 'Lines')
    add('center-line-left', 'Center line meets the guardrail (blue-driver left)', length / 2.0, width, 'Lines')
    add('blue-start-right', 'Blue starting line meets the guardrail (right)', hub_near, 0, 'Lines')
    add('blue-start-left', 'Blue starting line meets the guardrail (left)', hub_near, width, 'Lines')
    add('red-start-right', 'Red starting line meets the guardrail (blue-driver right)', length - hub_near, 0, 'Lines')
    add('red-start-left', 'Red starting line meets the guardrail (blue-driver left)', length - hub_near, width, 'Lines')
    # Hub bases. "wall side" faces the alliance's own wall; left/right as that alliance's driver sees it.
    add('blue-hub-wall-left', 'Blue hub base: wall side, left', hub_near, side_left, 'Blue hub')
    add('blue-hub-wall-right', 'Blue hub base: wall side, right', hub_near, side_right, 'Blue hub')
    add('blue-hub-center-left', 'Blue hub base: center side, left', hub_far, side_left, 'Blue hub')
    add('blue-hub-center-right', 'Blue hub base: center side, right', hub_far, side_right, 'Blue hub')
    add('red-hub-wall-left', 'Red hub base: wall side, left', length - hub_near, width - side_left, 'Red hub')
    add('red-hub-wall-right', 'Red hub base: wall side, right', length - hub_near, width - side_right, 'Red hub')
    add('red-hub-center-left', 'Red hub base: center side, left', length - hub_far, width - side_left, 'Red hub')
    add('red-hub-center-right', 'Red hub base: center side, right', length - hub_far, width - side_right, 'Red hub')
    add('blue-tower-left', 'Blue tower base: front, left', float(tower['x1']), float(tower['y1']), 'Towers')
    add('blue-tower-right', 'Blue tower base: front, right', float(tower['x1']), float(tower['y0']), 'Towers')
    add('red-tower-left', 'Red tower base: front, left', length - float(tower['x1']), width - float(tower['y1']), 'Towers')
    add('red-tower-right', 'Red tower base: front, right', length - float(tower['x1']), width - float(tower['y0']), 'Towers')
    return out


LANDMARKS = landmarks()
LANDMARK_BY_ID = {lm['id']: lm for lm in LANDMARKS}
LANDMARK_IDS = [lm['id'] for lm in LANDMARKS]


def overlay_outlines(field=FIELD):
    """Outlines (in field meters) to draw over the picture, matching overlayOutlines() in JS."""
    length = float(field['length'])
    width = float(field['width'])

    def rect(x0, y0, x1, y1):
        return [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]

    def mirror(poly):
        return [[length - x, width - y] for x, y in poly]

    hub = rect(
        float(field['startLineX']),
        float(field['hub']['y']) - float(field['hub']['halfSize']),
        float(field['trench']['x1']),
        float(field['hub']['y']) + float(field['hub']['halfSize']),
    )
    tower = rect(
        float(field['tower']['x0']),
        float(field['tower']['y0']),
        float(field['tower']['x1']),
        float(field['tower']['y1']),
    )
    return [
        {'id': 'field', 'points': rect(0.0, 0.0, length, width)},
        {'id': 'center', 'points': [[length / 2.0, 0.0], [length / 2.0, width]]},
        {'id': 'blue-start', 'points': [[float(field['startLineX']), 0.0], [float(field['startLineX']), width]]},
        {'id': 'red-start', 'points': [[length - float(field['startLineX']), 0.0], [length - float(field['startLineX']), width]]},
        {'id': 'blue-hub', 'points': hub},
        {'id': 'red-hub', 'points': mirror(hub)},
        {'id': 'blue-tower', 'points': tower},
        {'id': 'red-tower', 'points': mirror(tower)},
    ]


def apply_homography(matrix, point):
    """Apply a 3x3 homography to a 2D point [u, v] -> [x, y], or None when on the vanishing line."""
    if matrix is None or point is None or len(point) < 2:
        return None
    u, v = float(point[0]), float(point[1])
    if not (math.isfinite(u) and math.isfinite(v)):
        return None
    m = np.asarray(matrix, dtype=float)
    denom = m[2, 0] * u + m[2, 1] * v + m[2, 2]
    if abs(denom) < 1e-12:
        return None
    return [float((m[0, 0] * u + m[0, 1] * v + m[0, 2]) / denom),
            float((m[1, 0] * u + m[1, 1] * v + m[1, 2]) / denom)]


def invert3(matrix):
    """3x3 inverse as a list of lists, or None when singular."""
    m = np.asarray(matrix, dtype=float)
    if m.shape != (3, 3) or not np.isfinite(m).all():
        return None
    det = float(np.linalg.det(m))
    if abs(det) < 1e-14:
        return None
    return np.linalg.inv(m).tolist()


def field_to_image(homography, x, y):
    """Field point (x, y) in meters -> normalized image point [u, v], or None."""
    inv = invert3(homography)
    return apply_homography(inv, [x, y]) if inv is not None else None


def _solve_linear_8x8(normal, rhs):
    """Gaussian elimination with partial pivoting matching solve() in fieldCalibration.js."""
    n = len(rhs)
    a = [list(normal[i]) + [float(rhs[i])] for i in range(n)]
    for c in range(n):
        pivot = c
        for r in range(c + 1, n):
            if abs(a[r][c]) > abs(a[pivot][c]):
                pivot = r
        if abs(a[pivot][c]) < 1e-12:
            return None
        a[c], a[pivot] = a[pivot], a[c]
        for r in range(n):
            if r == c:
                continue
            f = a[r][c] / a[c][c]
            for k in range(c, n + 1):
                a[r][k] -= f * a[c][k]
    return [a[i][n] / a[i][i] for i in range(n)]


def _max_collinear_on_xy(xy, tol_m=0.15):
    """Return (max_count, best_line_sets) of points in `xy` lying within `tol_m` of a single line."""
    n = len(xy)
    if n <= 2:
        return n, [list(range(n))]
    best_count = 2
    best_sets = []
    for i in range(n):
        xi, yi = xy[i]
        for j in range(i + 1, n):
            dx = xy[j][0] - xi
            dy = xy[j][1] - yi
            length = math.hypot(dx, dy)
            if length <= 1e-6:
                members = [k for k in range(n) if math.hypot(xy[k][0] - xi, xy[k][1] - yi) <= tol_m]
            else:
                members = [
                    k for k in range(n)
                    if abs(dx * (xy[k][1] - yi) - dy * (xy[k][0] - xi)) / length <= tol_m
                ]
            cnt = len(members)
            if cnt > best_count:
                best_count = cnt
                best_sets = [members]
            elif cnt == best_count:
                best_sets.append(members)
    if not best_sets:
        best_sets = [[0, 1]]
    return best_count, best_sets


def collinear_stats(points, tol_m=0.15):
    """Compute field-coordinate collinearity and projective general-position rank statistics."""
    xy = []
    for p in points or ():
        try:
            if isinstance(p, dict):
                x, y = float(p['x']), float(p['y'])
            elif len(p) >= 2:
                x, y = float(p[0]), float(p[1])
            else:
                continue
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            xy.append((x, y))

    n = len(xy)
    if n == 0:
        return {
            'max_collinear_count': 0,
            'n_general_position': 0,
            'two_line_cross': False,
        }
    max_collinear, best_sets = _max_collinear_on_xy(xy, tol_m=tol_m)
    n_general_position = n - max(0, max_collinear - 2)
    two_line_cross = False
    if max_collinear >= 4:
        for line1 in best_sets:
            line1_set = set(line1)
            rem = [xy[k] for k in range(n) if k not in line1_set]
            rem_count = len(rem)
            if rem_count <= 2:
                two_line_cross = True
                break
            rem_max, _ = _max_collinear_on_xy(rem, tol_m=tol_m)
            if rem_max == rem_count:
                two_line_cross = True
                break
    return {
        'max_collinear_count': int(max_collinear),
        'n_general_position': int(n_general_position),
        'two_line_cross': bool(two_line_cross),
    }


def _solve_dlt_only(valid):
    """Solve DLT homography on validated tuples `[(p, u, v, raw_x, raw_y), ...]`, or return None."""
    if len(valid) < 4:
        return None
    scale = 10.0  # solve in decameters so the system is well conditioned
    normal = [[0.0] * 8 for _ in range(8)]
    rhs = [0.0] * 8
    for _, u, v, raw_x, raw_y in valid:
        x = raw_x / scale
        y = raw_y / scale
        rows = (
            ([u, v, 1.0, 0.0, 0.0, 0.0, -x * u, -x * v], x),
            ([0.0, 0.0, 0.0, u, v, 1.0, -y * u, -y * v], y),
        )
        for row, target in rows:
            for i in range(8):
                rhs[i] += row[i] * target
                for j in range(8):
                    normal[i][j] += row[i] * row[j]

    h = _solve_linear_8x8(normal, rhs)
    if not h or any(not math.isfinite(val) for val in h):
        return None
    return [
        [h[0] * scale, h[1] * scale, h[2] * scale],
        [h[3] * scale, h[4] * scale, h[5] * scale],
        [h[6], h[7], 1.0],
    ]


def solve_field_calibration(points):
    """Least-squares DLT homography from n >= 4 correspondences {u, v, x, y}, matching fieldCalibration.js."""
    valid = []
    for p in points or ():
        try:
            u, v, x, y = float(p['u']), float(p['v']), float(p['x']), float(p['y'])
        except (KeyError, TypeError, ValueError):
            continue
        if all(math.isfinite(val) for val in (u, v, x, y)):
            valid.append((p, u, v, x, y))
    if len(valid) < 4:
        return {'ok': False, 'reason': 'need_four', 'message': 'Mark at least four landmarks.'}

    homography = _solve_dlt_only(valid)
    if homography is None:
        return {
            'ok': False,
            'reason': 'degenerate',
            'message': 'Those landmarks are in a line or on top of each other. Pick points spread across the picture.',
        }

    errors = []
    for _, u, v, raw_x, raw_y in valid:
        out = apply_homography(homography, [u, v])
        errors.append(math.hypot(out[0] - raw_x, out[1] - raw_y) if out is not None else float('inf'))
    rms = math.sqrt(sum(e * e for e in errors) / len(errors))
    max_error = max(errors)
    worst = errors.index(max_error)

    stats = collinear_stats([(raw_x, raw_y) for _, _, _, raw_x, raw_y in valid])

    loo_errors = []
    loo_rms_m = None
    loo_p95_m = None
    loo_max_error_m = None
    loo_worst_index = None
    if len(valid) >= 5:
        for i in range(len(valid)):
            h_loo = _solve_dlt_only(valid[:i] + valid[i + 1:])
            _, u_i, v_i, x_i, y_i = valid[i]
            out_loo = apply_homography(h_loo, [u_i, v_i]) if h_loo is not None else None
            loo_errors.append(
                math.hypot(out_loo[0] - x_i, out_loo[1] - y_i) if out_loo is not None else float('inf')
            )
        if all(math.isfinite(e) for e in loo_errors):
            loo_rms = math.sqrt(sum(e * e for e in loo_errors) / len(loo_errors))
            s_loo = sorted(loo_errors)
            pos = 0.95 * (len(s_loo) - 1)
            lo = int(math.floor(pos))
            hi = int(math.ceil(pos))
            loo_p95 = s_loo[lo] + (pos - lo) * (s_loo[hi] - s_loo[lo])
            loo_max = max(loo_errors)
            loo_rms_m = _round_mm(loo_rms)
            loo_p95_m = _round_mm(loo_p95)
            loo_max_error_m = _round_mm(loo_max)
            loo_worst_index = loo_errors.index(loo_max)
        else:
            loo_rms_m = None
            loo_p95_m = float('inf')
            loo_max_error_m = float('inf')
            loo_worst_index = next((idx for idx, e in enumerate(loo_errors) if not math.isfinite(e)), 0)

    warnings = []
    if len(valid) == 4:
        warnings.append('Four landmarks solve exactly, so there is no error estimate. Add a fifth or sixth to see how accurate the clicks are.')
    if rms > 0.3:
        warnings.append(f'The landmarks disagree by {rms:.2f} m on average. Re-click the one with the largest error.')
    if stats['n_general_position'] < 4 or stats['two_line_cross']:
        warnings.append('Too many landmarks lie on the same line; add corner or starting-line landmarks on both sides of the field.')
    if loo_max_error_m is not None and loo_max_error_m > 0.45:
        if math.isfinite(loo_max_error_m):
            warnings.append(f'Leave-one-out check shows up to {loo_max_error_m:.2f} m error when holding out one landmark.')
        else:
            warnings.append('Leave-one-out check shows up to Infinity m error when holding out one landmark.')

    return {
        'ok': True,
        'homography_norm': homography,
        'rms_m': _round_mm(rms),
        'max_error_m': _round_mm(max_error),
        'worst_index': worst,
        'errors': errors,
        'loo_rms_m': loo_rms_m,
        'loo_p95_m': loo_p95_m,
        'loo_max_error_m': loo_max_error_m,
        'loo_worst_index': loo_worst_index,
        'loo_errors': loo_errors,
        'max_collinear_count': stats['max_collinear_count'],
        'n_general_position': stats['n_general_position'],
        'two_line_cross': stats['two_line_cross'],
        'warnings': warnings,
    }
