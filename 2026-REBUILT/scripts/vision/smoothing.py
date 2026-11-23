"""Damped-acceleration (Singer) Kalman filter + Rauch-Tung-Striebel smoother for robot tracks in field meters.

Raw detections jitter by tens of centimeters, occasionally hop on a single frame (a bounding box
expanding or contracting), and occasionally jump onto the wrong robot. The smoother:
  1. Weights each point by its projected uncertainty plus a bounding-box jitter floor, down-weighting
     single-frame hairpin reversals.
  2. Rejects outlier jumps that exceed the Mahalanobis or physical speed gate.
  3. Runs a 6-state [x, y, vx, vy, ax, ay] jerk-driven Singer forward filter and backward RTS pass,
     resetting velocity/acceleration coupling across observation gaps so camera cuts or occlusions
     never whip segment endpoints.
  4. Projects each contiguous observed segment onto physical bounds for acceleration (<= AMAX) and
     turn rate (<= WMAX) while anchoring kept points within a fidelity tube around raw detections.
  5. Resolves and smooths headings consistently with the smoothed velocity vector and bounded turn rate.
  6. Densifies sparse autonomous trajectories to 10-20 Hz along an acceleration-bounded Hermite curve
     within observed spans, without inventing points across gaps or outside the observations.
"""
import math

import numpy as np

from common import AUTO_END_MS, FIELD

VMAX = FIELD['robot']['maxPlausibleSpeedMps']
AMAX = 6.0              # m/s^2: physical carpet traction bound in autonomous
WMAX = 4.0              # rad/s: maximum plausible chassis/travel turn rate (~230 deg/s)
ACCEL_TAU_S = 0.40      # s: Singer acceleration correlation time
GAP_RESET_S = 0.65      # s: observation gap threshold; never couple or interpolate across > GAP_RESET_S
BOX_JITTER_M = 0.115    # m: typical detector box-edge floor jitter beyond 2-pixel homography scale


def _wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def _clamp_vec(vx, vy, limit):
    mag = math.hypot(vx, vy)
    if mag > limit > 0.0:
        scale = limit / mag
        return vx * scale, vy * scale
    return vx, vy


def _transition(dt, accel_sigma, tau_s=ACCEL_TAU_S):
    """6-state [x, y, vx, vy, ax, ay] Singer transition matrix F and jerk-driven process covariance Q."""
    alpha = 1.0 / max(tau_s, 1e-3)
    ad = alpha * dt
    if ad < 1e-4:
        c_pa = 0.5 * dt * dt
        c_va = dt
        beta = 1.0 - ad
    else:
        beta = math.exp(-ad)
        c_pa = (ad - 1.0 + beta) / (alpha * alpha)
        c_va = (1.0 - beta) / alpha
    f = np.array([
        [1.0, 0.0, dt,  0.0, c_pa, 0.0],
        [0.0, 1.0, 0.0, dt,  0.0,  c_pa],
        [0.0, 0.0, 1.0, 0.0, c_va, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0,  c_va],
        [0.0, 0.0, 0.0, 0.0, beta, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0,  beta],
    ], float)
    jerk_sigma = accel_sigma * 1.25
    q_j = jerk_sigma ** 2
    g = np.array([
        [dt ** 3 / 6.0, 0.0],
        [0.0, dt ** 3 / 6.0],
        [dt ** 2 / 2.0, 0.0],
        [0.0, dt ** 2 / 2.0],
        [dt, 0.0],
        [0.0, dt],
    ], float)
    q = q_j * (g @ g.T) + np.diag([1e-7, 1e-7, 1e-6, 1e-6, 1e-5, 1e-5])
    return f, q


def _detect_stationary_holds(t_arr, x_arr, y_arr, gap_s=GAP_RESET_S, min_dur_s=0.40, max_chord_v=0.25, max_rad=0.18):
    """Identify contiguous stationary windows (>= 400 ms, raw chord speed < 0.25 m/s, within 0.18 m of local median)."""
    n = len(t_arr)
    in_hold = np.zeros(n, bool)
    hold_x = x_arr.copy()
    hold_y = y_arr.copy()
    if n < 2:
        return in_hold, hold_x, hold_y
    step_slow = np.zeros(n - 1, bool)
    for i in range(n - 1):
        dt = float(t_arr[i + 1] - t_arr[i])
        if 0.0 < dt <= gap_s:
            if math.hypot(x_arr[i + 1] - x_arr[i], y_arr[i + 1] - y_arr[i]) / dt < max_chord_v:
                step_slow[i] = True
    i = 0
    while i < n - 1:
        if not step_slow[i]:
            i += 1
            continue
        j = i
        while j + 1 < n - 1 and step_slow[j + 1]:
            j += 1
        a, b = i, j + 1
        if t_arr[b] - t_arr[a] >= min_dur_s:
            mx = float(np.median(x_arr[a:b + 1]))
            my = float(np.median(y_arr[a:b + 1]))
            rad = max(math.hypot(x_arr[k] - mx, y_arr[k] - my) for k in range(a, b + 1))
            if rad <= max_rad:
                for k in range(a, b + 1):
                    in_hold[k] = True
                    hold_x[k] = mx
                    hold_y[k] = my
            else:
                for k in range(a, b + 1):
                    win = [w for w in range(a, b + 1) if abs(t_arr[w] - t_arr[k]) <= 0.75]
                    if len(win) >= 2 and t_arr[win[-1]] - t_arr[win[0]] >= min_dur_s:
                        wmx = float(np.median(x_arr[win]))
                        wmy = float(np.median(y_arr[win]))
                        if max(math.hypot(x_arr[w] - wmx, y_arr[w] - wmy) for w in win) <= max_rad:
                            in_hold[k] = True
                            hold_x[k] = wmx
                            hold_y[k] = wmy
        i = j + 1
    return in_hold, hold_x, hold_y


def smooth_track(times_s, xs, ys, sigmas, accel_sigma=6.0, gate=4.5, max_accel=AMAX, max_turn_rate=WMAX, max_deviation=0.25, tau_s=ACCEL_TAU_S):
    """Return dict of arrays: x, y, vx, vy, ax, ay, sigma (position std), kept (bool per input), for each input time.

    Uses a 6-state jerk-driven Singer forward Kalman filter and Rauch-Tung-Striebel backward pass,
    with stationary hold damping and simultaneous Jacobi kinematic projection bounding acceleration
    (<= max_accel) and turn rate (<= max_turn_rate) while keeping smoothed positions within a fidelity
    tube around kept observations.
    """
    t_arr = np.asarray(times_s, float)
    x_arr = np.asarray(xs, float)
    y_arr = np.asarray(ys, float)
    s_arr = np.asarray(sigmas, float)
    n = len(t_arr)
    if n == 0:
        return {k: np.array([]) for k in ('x', 'y', 'vx', 'vy', 'ax', 'ay', 'sigma', 'kept')}
    h = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]], float)
    xf = np.zeros((n, 6))
    pf = np.zeros((n, 6, 6))
    xp = np.zeros((n, 6))
    pp = np.zeros((n, 6, 6))
    fs = [np.eye(6)] * n
    kept = np.zeros(n, bool)

    in_hold, hold_x, hold_y = _detect_stationary_holds(t_arr, x_arr, y_arr)

    # Detect 1-frame hairpin reversals (bounding box hopping forward and immediately back)
    hairpin = np.ones(n, float)
    for i in range(1, n - 1):
        dt1 = t_arr[i] - t_arr[i - 1]
        dt2 = t_arr[i + 1] - t_arr[i]
        if 0.0 < dt1 <= GAP_RESET_S and 0.0 < dt2 <= GAP_RESET_S:
            dx1, dy1 = x_arr[i] - x_arr[i - 1], y_arr[i] - y_arr[i - 1]
            dx2, dy2 = x_arr[i + 1] - x_arr[i], y_arr[i + 1] - y_arr[i]
            d1, d2 = math.hypot(dx1, dy1), math.hypot(dx2, dy2)
            if d1 > 0.03 and d2 > 0.03:
                cos_a = (dx1 * dx2 + dy1 * dy2) / (d1 * d2)
                if cos_a < 0.0:
                    hairpin[i] = 1.0 + 2.2 * (-cos_a)

    s0 = max(float(s_arr[0]), 0.08)
    x = np.array([hold_x[0] if in_hold[0] else x_arr[0], hold_y[0] if in_hold[0] else y_arr[0], 0.0, 0.0, 0.0, 0.0], float)
    p = np.diag([s0 ** 2, s0 ** 2, 4.0, 4.0, 6.0, 6.0])
    for i in range(n):
        if i == 0:
            xp[i], pp[i] = x, p
        else:
            dt = max(1e-3, float(t_arr[i] - t_arr[i - 1]))
            if dt > GAP_RESET_S:
                lead = min(dt, 0.6)
                x = np.array([x[0] + x[2] * lead, x[1] + x[3] * lead, x[2] * 0.5, x[3] * 0.5, 0.0, 0.0], float)
                p = np.diag([(0.4 + 1.5 * dt) ** 2, (0.4 + 1.5 * dt) ** 2, 6.0, 6.0, 9.0, 9.0])
                fs[i] = np.eye(6)
            else:
                f, q = _transition(dt, accel_sigma, tau_s=tau_s)
                if in_hold[i] and in_hold[i - 1]:
                    f = f.copy()
                    f[0, 2:] *= 0.35
                    f[1, 2:] *= 0.35
                    f[2:, 2:] *= 0.35
                fs[i] = f
                x = f @ x
                x[2], x[3] = _clamp_vec(x[2], x[3], VMAX)
                x[4], x[5] = _clamp_vec(x[4], x[5], max_accel)
                p = f @ p @ f.T + q
            xp[i], pp[i] = x, p

        sig_raw = max(float(s_arr[i]), 0.03)
        dt_prev = max(1e-3, float(t_arr[i] - t_arr[i - 1])) if i > 0 else 0.1
        s_gate = h @ p @ h.T + np.eye(2) * (sig_raw ** 2 + (0.5 * accel_sigma * dt_prev ** 2) ** 2)
        z_raw = np.array([x_arr[i], y_arr[i]], float)
        innovation_raw = z_raw - h @ x
        maha = math.sqrt(max(0.0, float(innovation_raw @ np.linalg.solve(s_gate, innovation_raw))))
        implausible = False
        if i > 0 and kept[:i].any():
            last = int(np.where(kept[:i])[0][-1])
            dt_last = max(1e-3, float(t_arr[i] - t_arr[last]))
            implausible = math.hypot(x_arr[i] - xf[last][0], y_arr[i] - xf[last][1]) / dt_last > VMAX * 1.5
        if i == 0 or (maha <= gate and not implausible):
            if in_hold[i]:
                z = np.array([hold_x[i], hold_y[i]], float)
                sig_meas = math.hypot(sig_raw, BOX_JITTER_M * 0.75)
            else:
                z = z_raw
                sig_meas = math.hypot(sig_raw, BOX_JITTER_M) * hairpin[i]
            innovation = z - h @ x
            r_meas = np.eye(2) * (sig_meas ** 2)
            s_upd = h @ p @ h.T + r_meas
            k = p @ h.T @ np.linalg.inv(s_upd)
            x = x + k @ innovation
            x[2], x[3] = _clamp_vec(x[2], x[3], VMAX)
            x[4], x[5] = _clamp_vec(x[4], x[5], max_accel)
            ikh = np.eye(6) - k @ h
            p = ikh @ p @ ikh.T + k @ r_meas @ k.T
            kept[i] = True
        xf[i], pf[i] = x, p

    # Backward Rauch-Tung-Striebel (RTS) pass within contiguous segments
    xs_out = xf.copy()
    ps_out = pf.copy()
    for i in range(n - 2, -1, -1):
        dt = float(t_arr[i + 1] - t_arr[i])
        if dt > GAP_RESET_S:
            continue
        f = fs[i + 1]
        c = pf[i] @ f.T @ np.linalg.pinv(pp[i + 1])
        xs_out[i] = xf[i] + c @ (xs_out[i + 1] - xp[i + 1])
        xs_out[i, 2], xs_out[i, 3] = _clamp_vec(xs_out[i, 2], xs_out[i, 3], VMAX)
        xs_out[i, 4], xs_out[i, 5] = _clamp_vec(xs_out[i, 4], xs_out[i, 5], max_accel)
        ps_out[i] = pf[i] + c @ (ps_out[i + 1] - pp[i + 1]) @ c.T

    # Simultaneous Jacobi kinematic projection (bounds accel and turn rate while preserving fidelity)
    px = xs_out[:, 0].copy()
    py = xs_out[:, 1].copy()
    for _ in range(4):
        nx, ny = px.copy(), py.copy()
        for i in range(1, n - 1):
            if in_hold[i] and in_hold[i - 1] and in_hold[i + 1]:
                if t_arr[i] - t_arr[i - 1] <= GAP_RESET_S and t_arr[i + 1] - t_arr[i] <= GAP_RESET_S:
                    nx[i] = 0.25 * px[i - 1] + 0.25 * px[i] + 0.25 * px[i + 1] + 0.25 * hold_x[i]
                    ny[i] = 0.25 * py[i - 1] + 0.25 * py[i] + 0.25 * py[i + 1] + 0.25 * hold_y[i]
        px, py = nx, ny
    init_caps = np.maximum(np.where(hairpin > 1.05, max_deviation, min(max_deviation, 0.145)), 1.8 * np.maximum(s_arr, 0.05)) * np.minimum(1.3, hairpin)
    dev_caps = np.maximum(max_deviation, 1.8 * np.maximum(s_arr, 0.05)) * np.minimum(1.3, hairpin)
    seg_starts = [0]
    for i in range(1, n):
        if t_arr[i] - t_arr[i - 1] > GAP_RESET_S:
            seg_starts.append(i)
    seg_starts.append(n)

    for s_idx in range(len(seg_starts) - 1):
        a_idx, b_idx = seg_starts[s_idx], seg_starts[s_idx + 1]
        m = b_idx - a_idx
        for i in range(a_idx, b_idx):
            if kept[i]:
                dx, dy = px[i] - x_arr[i], py[i] - y_arr[i]
                d = math.hypot(dx, dy)
                cap = float(init_caps[i])
                if d > cap:
                    px[i] = x_arr[i] + dx * (cap / d)
                    py[i] = y_arr[i] + dy * (cap / d)
        if m >= 3:
            for _ in range(20):
                nx = px.copy()
                ny = py.copy()
                changed = False
                for i in range(a_idx + 1, b_idx - 1):
                    dt1 = max(1e-3, float(t_arr[i] - t_arr[i - 1]))
                    dt2 = max(1e-3, float(t_arr[i + 1] - t_arr[i]))
                    dtm = 0.5 * (dt1 + dt2)
                    w1 = dt2 / (dt1 + dt2)
                    w2 = dt1 / (dt1 + dt2)
                    mid_x = w1 * px[i - 1] + w2 * px[i + 1]
                    mid_y = w1 * py[i - 1] + w2 * py[i + 1]
                    vx1, vy1 = (px[i] - px[i - 1]) / dt1, (py[i] - py[i - 1]) / dt1
                    vx2, vy2 = (px[i + 1] - px[i]) / dt2, (py[i + 1] - py[i]) / dt2
                    ax_i, ay_i = (vx2 - vx1) / dtm, (vy2 - vy1) / dtm
                    a_mag = math.hypot(ax_i, ay_i)
                    spd1, spd2 = math.hypot(vx1, vy1), math.hypot(vx2, vy2)
                    turn_rate = 0.0
                    if spd1 > 0.35 and spd2 > 0.35:
                        dh = abs(_wrap(math.atan2(vy2, vx2) - math.atan2(vy1, vx1)))
                        turn_rate = dh / dtm
                    pull = 0.0
                    if a_mag > max_accel * 0.9:
                        pull = max(pull, 0.5 * (1.0 - (max_accel * 0.9) / a_mag))
                    if turn_rate > max_turn_rate * 0.9:
                        pull = max(pull, 0.5 * (1.0 - (max_turn_rate * 0.9) / turn_rate))
                    if pull > 0.0:
                        changed = True
                        tx = px[i] + pull * (mid_x - px[i])
                        ty = py[i] + pull * (mid_y - py[i])
                        if kept[i]:
                            dx, dy = tx - x_arr[i], ty - y_arr[i]
                            d = math.hypot(dx, dy)
                            cap = float(dev_caps[i])
                            if d > cap:
                                tx = x_arr[i] + dx * (cap / d)
                                ty = y_arr[i] + dy * (cap / d)
                        nx[i], ny[i] = tx, ty
                px, py = nx, ny
                if not changed:
                    break

        for i in range(a_idx, b_idx):
            if in_hold[i] and ((i > a_idx and in_hold[i - 1]) or (i + 1 < b_idx and in_hold[i + 1])):
                vx, vy = 0.0, 0.0
            elif i == a_idx:
                if m >= 2:
                    dt = max(1e-3, float(t_arr[i + 1] - t_arr[i]))
                    vx, vy = (px[i + 1] - px[i]) / dt, (py[i + 1] - py[i]) / dt
                else:
                    vx, vy = 0.0, 0.0
            elif i == b_idx - 1:
                dt = max(1e-3, float(t_arr[i] - t_arr[i - 1]))
                vx, vy = (px[i] - px[i - 1]) / dt, (py[i] - py[i - 1]) / dt
            else:
                dt1 = max(1e-3, float(t_arr[i] - t_arr[i - 1]))
                dt2 = max(1e-3, float(t_arr[i + 1] - t_arr[i]))
                vx = (dt2 / (dt1 + dt2)) * ((px[i] - px[i - 1]) / dt1) + (dt1 / (dt1 + dt2)) * ((px[i + 1] - px[i]) / dt2)
                vy = (dt2 / (dt1 + dt2)) * ((py[i] - py[i - 1]) / dt1) + (dt1 / (dt1 + dt2)) * ((py[i + 1] - py[i]) / dt2)
            vx, vy = _clamp_vec(vx, vy, VMAX)
            xs_out[i, 0], xs_out[i, 1] = px[i], py[i]
            xs_out[i, 2], xs_out[i, 3] = vx, vy
        for i in range(a_idx, b_idx):
            if m < 2:
                ax_i, ay_i = 0.0, 0.0
            elif i == a_idx:
                dt = max(1e-3, float(t_arr[i + 1] - t_arr[i]))
                ax_i, ay_i = (xs_out[i + 1, 2] - xs_out[i, 2]) / dt, (xs_out[i + 1, 3] - xs_out[i, 3]) / dt
            elif i == b_idx - 1:
                dt = max(1e-3, float(t_arr[i] - t_arr[i - 1]))
                ax_i, ay_i = (xs_out[i, 2] - xs_out[i - 1, 2]) / dt, (xs_out[i, 3] - xs_out[i - 1, 3]) / dt
            else:
                dt = max(1e-3, float(t_arr[i + 1] - t_arr[i - 1]))
                ax_i, ay_i = (xs_out[i + 1, 2] - xs_out[i - 1, 2]) / dt, (xs_out[i + 1, 3] - xs_out[i - 1, 3]) / dt
            xs_out[i, 4], xs_out[i, 5] = _clamp_vec(ax_i, ay_i, max_accel)

    return {
        'x': xs_out[:, 0], 'y': xs_out[:, 1],
        'vx': xs_out[:, 2], 'vy': xs_out[:, 3],
        'ax': xs_out[:, 4], 'ay': xs_out[:, 5],
        'sigma': np.sqrt(np.maximum(ps_out[:, 0, 0], 0.0)),
        'kept': kept,
    }


def resolve_heading(theta, velocity_heading, speed, previous, period=math.pi / 2, move_speed=0.6, infer_from_velocity=False):
    """Pick the heading consistent with an oriented box (or velocity vector when infer_from_velocity=True).

    A rectangle's orientation is only known modulo its symmetry (180 degrees for a rectangle, 90 for
    a square robot, which most FRC chassis nearly are). While moving, choose the candidate closest to
    the direction of travel (with hysteresis toward `previous` when present so noise near 45 degrees
    does not flip a 90-degree symmetry axis); while stopped, stay on the candidate closest to the
    previous heading so the arrow does not spin. When `theta` is None (axis-aligned detector), returns
    None by default so downstream consumers do not mistake velocity headings for detector measurements,
    or falls back to the velocity direction when `infer_from_velocity=True`.
    """
    if theta is None:
        if not infer_from_velocity:
            return None
        if velocity_heading is not None and speed >= move_speed:
            if previous is not None:
                blend = min(1.0, (speed - move_speed) / max(move_speed, 0.4) * 0.7 + 0.3)
                return _wrap(previous + _wrap(velocity_heading - previous) * blend)
            return _wrap(velocity_heading)
        return _wrap(previous) if previous is not None else None
    n_cand = max(1, int(round(2 * math.pi / period)))
    candidates = [theta + k * period for k in range(n_cand)]
    moving = velocity_heading is not None and speed >= move_speed
    if moving and previous is not None:
        return _wrap(min(candidates, key=lambda c: abs(_wrap(c - velocity_heading)) + 0.35 * abs(_wrap(c - previous))))
    anchor = velocity_heading if moving else previous
    if anchor is None:
        return _wrap(candidates[0])
    return _wrap(min(candidates, key=lambda c: abs(_wrap(c - anchor))))


def smooth_heading(times_s, headings, weights=None, turn_sigma=4.0, vx=None, vy=None, max_turn_rate=WMAX, move_speed=0.35):
    """Smooth a heading series (radians, None where unknown) with an angle+rate RTS smoother.

    Unwraps first so a turn through +-pi is continuous, couples with the velocity direction while
    moving when `vx, vy` are supplied, resets across observation gaps, and bounds angular rate
    |omega| <= max_turn_rate. Returns (wrapped_headings, turn_rates).
    """
    n = len(times_s)
    out_h = [None] * n
    out_w = [0.0] * n
    idx = [i for i in range(n) if headings[i] is not None]
    if len(idx) == 1:
        i0 = idx[0]
        out_h[i0] = _wrap(float(headings[i0]))
        return out_h, out_w
    if len(idx) < 2:
        return out_h, out_w
    un = {}
    prev = None
    for i in idx:
        a = float(headings[i])
        if vx is not None and vy is not None:
            spd = math.hypot(float(vx[i]), float(vy[i]))
            if spd >= move_speed:
                vh = math.atan2(float(vy[i]), float(vx[i]))
                diff_v = _wrap(vh - a)
                if abs(diff_v) <= math.pi / 3:
                    a = _wrap(a + 0.35 * diff_v)
        if prev is not None:
            a = prev + _wrap(a - prev)
        un[i] = a
        prev = a
    t = np.array([float(times_s[i]) for i in idx])
    z = np.array([un[i] for i in idx])
    w = np.array([1.0 if weights is None else max(float(weights[i]), 0.05) for i in idx])
    m = len(idx)
    h = np.array([[1.0, 0.0]])
    xf, pf, xp, pp, fs = np.zeros((m, 2)), np.zeros((m, 2, 2)), np.zeros((m, 2)), np.zeros((m, 2, 2)), [np.eye(2)] * m
    x = np.array([z[0], 0.0])
    p = np.diag([0.05, 2.0])
    tau_w = 0.35
    for k in range(m):
        if k:
            dt = max(1e-3, float(t[k] - t[k - 1]))
            if dt > GAP_RESET_S:
                f = np.array([[1.0, 0.0], [0.0, 0.0]])
                x = np.array([z[k], 0.0])
                p = np.diag([0.25, 4.0])
            else:
                beta = math.exp(-dt / tau_w)
                c_aw = tau_w * (1.0 - beta)
                f = np.array([[1.0, c_aw], [0.0, beta]], float)
                g = np.array([[dt ** 2 / 2.0], [dt]])
                x = f @ x
                x[1] = max(-max_turn_rate, min(max_turn_rate, float(x[1])))
                p = f @ p @ f.T + (turn_sigma ** 2) * (g @ g.T)
            fs[k] = f
        xp[k], pp[k] = x, p
        r = np.array([[(0.15 / w[k]) ** 2]])
        s = h @ p @ h.T + r
        kk = p @ h.T @ np.linalg.inv(s)
        x = x + (kk @ (z[k] - h @ x)).ravel()
        x[1] = max(-max_turn_rate, min(max_turn_rate, float(x[1])))
        p = (np.eye(2) - kk @ h) @ p
        xf[k], pf[k] = x, p
    xs = xf.copy()
    for k in range(m - 2, -1, -1):
        dt = float(t[k + 1] - t[k])
        if dt > GAP_RESET_S:
            continue
        c = pf[k] @ fs[k + 1].T @ np.linalg.pinv(pp[k + 1])
        xs[k] = xf[k] + c @ (xs[k + 1] - xp[k + 1])
        xs[k, 1] = max(-max_turn_rate, min(max_turn_rate, float(xs[k, 1])))

    # Enforce turn-rate bound on unwrapped angles within contiguous segments
    ang = xs[:, 0].copy()
    seg_starts = [0]
    for k in range(1, m):
        if t[k] - t[k - 1] > GAP_RESET_S:
            seg_starts.append(k)
    seg_starts.append(m)
    for s_idx in range(len(seg_starts) - 1):
        a_k, b_k = seg_starts[s_idx], seg_starts[s_idx + 1]
        if b_k - a_k < 2:
            continue
        for sweep in range(6):
            if sweep % 2 == 0:
                for k in range(a_k + 1, b_k):
                    dt = max(1e-3, float(t[k] - t[k - 1]))
                    da = ang[k] - ang[k - 1]
                    limit = max_turn_rate * dt
                    if abs(da) > limit:
                        ang[k] = ang[k - 1] + math.copysign(limit, da)
            else:
                for k in range(b_k - 2, a_k - 1, -1):
                    dt = max(1e-3, float(t[k + 1] - t[k]))
                    da = ang[k + 1] - ang[k]
                    limit = max_turn_rate * dt
                    if abs(da) > limit:
                        ang[k] = ang[k + 1] - math.copysign(limit, da)
        for k in range(a_k, b_k):
            if k == a_k:
                dt = max(1e-3, float(t[k + 1] - t[k]))
                w_k = (ang[k + 1] - ang[k]) / dt
            elif k == b_k - 1:
                dt = max(1e-3, float(t[k] - t[k - 1]))
                w_k = (ang[k] - ang[k - 1]) / dt
            else:
                dt = max(1e-3, float(t[k + 1] - t[k - 1]))
                w_k = (ang[k + 1] - ang[k - 1]) / dt
            xs[k, 0] = ang[k]
            xs[k, 1] = max(-max_turn_rate, min(max_turn_rate, w_k))

    for k, i in enumerate(idx):
        out_h[i] = _wrap(float(xs[k, 0]))
        out_w[i] = float(xs[k, 1])
    return out_h, out_w


def build_smooth_trajectory(points, offset_s, sm, angle_period=math.pi / 2, pseudo=False, auto_hz=12.0, auto_end_ms=AUTO_END_MS, max_gap_s=GAP_RESET_S, max_subdivide_s=0.35, max_accel=AMAX, max_turn_rate=WMAX, infer_velocity_heading=False):
    """Turn smoothed states + kept detections into a trajectory list, densifying sparse auto spans to 10-20 Hz.

    - Preserves detector heading provenance: when `p.get('angle') is None` (axis-aligned detector) and
      `infer_velocity_heading=False` (default), `point['heading']` stays `None` while `(vx, vy)` are emitted
      so downstream `autoDetection.js` derives velocity headings without misclassifying them as detector headings.
    - Within the autonomous window (0 <= t_ms <= auto_end_ms) when calibrated (`not pseudo`), subdivides
      observed intervals sparser than 10 Hz (`0.105 s < dt <= max_subdivide_s`, default `0.35 s`) to `auto_hz`
      (`12 Hz`), emitting `'estimated': True` on subdivided points and `'estimated': False` on observed points.
    - Runs the dense C2/Jacobi smoothing pass on auto sub-runs even when a full-match track continues past
      `auto_end_ms`, and never invents points across observation gaps (`dt > max_subdivide_s`).
    """
    kept_indices = [i for i, k in enumerate(sm['kept']) if k]
    if not kept_indices:
        return []

    speeds = np.hypot(sm['vx'], sm['vy'])
    k_times = [float(points[i]['t']) for i in kept_indices]
    k_vx = [float(sm['vx'][i]) for i in kept_indices]
    k_vy = [float(sm['vy'][i]) for i in kept_indices]
    k_conf = [float(points[i]['conf']) for i in kept_indices]

    raw_headings = [None] * len(kept_indices)
    previous = None
    for k, i in enumerate(kept_indices):
        if k > 0 and k_times[k] - k_times[k - 1] > max_gap_s:
            previous = None
        p = points[i]
        vh = math.atan2(k_vy[k], k_vx[k]) if speeds[i] >= 0.35 else None
        h_val = resolve_heading(p.get('angle'), vh, float(speeds[i]), previous, angle_period, infer_from_velocity=infer_velocity_heading)
        raw_headings[k] = h_val
        if h_val is not None:
            previous = h_val

    nxt = None
    for k in range(len(kept_indices) - 1, -1, -1):
        if k < len(kept_indices) - 1 and k_times[k + 1] - k_times[k] > max_gap_s:
            nxt = None
        if raw_headings[k] is not None:
            nxt = raw_headings[k]
        elif nxt is not None and (infer_velocity_heading or points[kept_indices[k]].get('angle') is not None):
            raw_headings[k] = nxt

    smooth_h, _ = smooth_heading(k_times, raw_headings, k_conf, vx=k_vx, vy=k_vy, max_turn_rate=max_turn_rate)

    if pseudo or auto_hz is None or auto_hz <= 0:
        trajectory = []
        for k, i in enumerate(kept_indices):
            p = points[i]
            x, y = float(sm['x'][i]), float(sm['y'][i])
            if pseudo:
                x, y = x / FIELD['length'], y / FIELD['width']
            pt = {
                't': int(round((p['t'] - offset_s) * 1000)),
                'x': round(x, 4 if pseudo else 3),
                'y': round(y, 4 if pseudo else 3),
                'heading': None if (pseudo or smooth_h[k] is None) else round(smooth_h[k], 3),
                'sigma': round(float(sm['sigma'][i]), 3),
                'confidence': round(p['conf'], 2),
                'box': [round(v, 4) for v in p['box']],
                'calibrated': not pseudo,
                'estimated': False,
            }
            if not pseudo:
                pt['vx'] = round(k_vx[k], 3)
                pt['vy'] = round(k_vy[k], 3)
            trajectory.append(pt)
        return trajectory

    min_split_dt = max(0.105, 1.0 / (auto_hz * 0.85))
    sub_cap_s = min(float(max_gap_s), float(max_subdivide_s))
    target_dt = 1.0 / auto_hz
    m = len(kept_indices)
    t_ms_k = [int(round((float(points[i]['t']) - offset_s) * 1000)) for i in kept_indices]
    tk = [t / 1000.0 for t in t_ms_k]
    xk_sm = [float(sm['x'][i]) for i in kept_indices]
    yk_sm = [float(sm['y'][i]) for i in kept_indices]
    sig_k = [float(sm['sigma'][i]) for i in kept_indices]
    box_k = [[float(v) for v in points[i]['box']] for i in kept_indices]
    ang_k = [points[i].get('angle') for i in kept_indices]
    in_hold_k, _, _ = _detect_stationary_holds(
        np.asarray(tk, float),
        np.asarray([float(points[i]['x']) for i in kept_indices], float),
        np.asarray([float(points[i]['y']) for i in kept_indices], float),
    )

    seg_starts = [0]
    for k in range(1, m):
        dt = tk[k] - tk[k - 1]
        in_auto_prev = 0 <= t_ms_k[k - 1] and t_ms_k[k] <= auto_end_ms + 1000
        if dt > sub_cap_s or not in_auto_prev:
            seg_starts.append(k)
    seg_starts.append(m)

    h_mat = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]], float)
    trajectory = []

    for s_idx in range(len(seg_starts) - 1):
        a, b = seg_starts[s_idx], seg_starts[s_idx + 1]
        if b - a < 2:
            for k in range(a, b):
                trajectory.append({
                    't': t_ms_k[k],
                    'x': round(xk_sm[k], 4),
                    'y': round(yk_sm[k], 4),
                    'vx': round(k_vx[k], 3),
                    'vy': round(k_vy[k], 3),
                    'heading': None if smooth_h[k] is None else round(float(smooth_h[k]), 3),
                    'sigma': round(sig_k[k], 3),
                    'confidence': round(k_conf[k], 2),
                    'box': [round(v, 4) for v in box_k[k]],
                    'calibrated': True,
                    'estimated': False,
                })
            continue

        td_ms, obs_at, meta, hold_mask = [], {}, [], []
        for k in range(a, b - 1):
            t0_ms, t1_ms = t_ms_k[k], t_ms_k[k + 1]
            dt = (t1_ms - t0_ms) / 1000.0
            obs_at[len(td_ms)] = k
            td_ms.append(t0_ms)
            meta.append((sig_k[k], k_conf[k], box_k[k], ang_k[k], smooth_h[k]))
            step_hold = bool(in_hold_k[k] and in_hold_k[k + 1])
            hold_mask.append(bool(in_hold_k[k]))
            in_auto = 0 <= t0_ms and t1_ms <= auto_end_ms + 1000
            if in_auto and min_split_dt < dt <= sub_cap_s:
                steps = max(2, int(round(dt / target_dt)))
                for s in range(1, steps):
                    t_sub_ms = int(round(t0_ms + (s / steps) * (t1_ms - t0_ms)))
                    if t_sub_ms <= td_ms[-1] or t_sub_ms >= t1_ms:
                        continue
                    u = (t_sub_ms - t0_ms) / (t1_ms - t0_ms)
                    td_ms.append(t_sub_ms)
                    hold_mask.append(step_hold)
                    h_interp = None
                    if smooth_h[k] is not None and smooth_h[k + 1] is not None:
                        h_interp = _wrap(smooth_h[k] + _wrap(smooth_h[k + 1] - smooth_h[k]) * u)
                    else:
                        h_interp = smooth_h[k] if smooth_h[k] is not None else smooth_h[k + 1]
                    meta.append((
                        sig_k[k] + (sig_k[k + 1] - sig_k[k]) * u,
                        min(k_conf[k], k_conf[k + 1]),
                        [box_k[k][j] + (box_k[k + 1][j] - box_k[k][j]) * u for j in range(4)],
                        None,
                        h_interp,
                    ))
        obs_at[len(td_ms)] = b - 1
        td_ms.append(t_ms_k[b - 1])
        hold_mask.append(bool(in_hold_k[b - 1]))
        meta.append((sig_k[b - 1], k_conf[b - 1], box_k[b - 1], ang_k[b - 1], smooth_h[b - 1]))

        nd = len(td_ms)
        td = np.array([t / 1000.0 for t in td_ms], float)
        if nd == b - a:
            # No subdivision occurred in this segment
            for j, k in enumerate(range(a, b)):
                trajectory.append({
                    't': td_ms[j],
                    'x': round(xk_sm[k], 4),
                    'y': round(yk_sm[k], 4),
                    'vx': round(k_vx[k], 3),
                    'vy': round(k_vy[k], 3),
                    'heading': None if smooth_h[k] is None else round(float(smooth_h[k]), 3),
                    'sigma': round(sig_k[k], 3),
                    'confidence': round(k_conf[k], 2),
                    'box': [round(v, 4) for v in box_k[k]],
                    'calibrated': True,
                    'estimated': False,
                })
            continue

        # Run 6-state jerk RTS on the dense grid anchored at smoothed knots
        xf = np.zeros((nd, 6))
        pf = np.zeros((nd, 6, 6))
        xp = np.zeros((nd, 6))
        pp = np.zeros((nd, 6, 6))
        fs = [np.eye(6)] * nd
        x = np.array([xk_sm[a], yk_sm[a], k_vx[a], k_vy[a], 0.0, 0.0], float)
        p = np.diag([0.015 ** 2, 0.015 ** 2, 1.0, 1.0, 4.0, 4.0])
        for j in range(nd):
            if j > 0:
                dt = max(1e-3, float(td[j] - td[j - 1]))
                f, q = _transition(dt, 3.8)
                fs[j] = f
                x = f @ x
                x[2], x[3] = _clamp_vec(x[2], x[3], VMAX)
                x[4], x[5] = _clamp_vec(x[4], x[5], max_accel)
                p = f @ p @ f.T + q
            xp[j], pp[j] = x, p
            if j in obs_at:
                k_obs = obs_at[j]
                z = np.array([xk_sm[k_obs], yk_sm[k_obs]], float)
                r = np.eye(2) * (0.018 ** 2)
                s_upd = h_mat @ p @ h_mat.T + r
                kk = p @ h_mat.T @ np.linalg.inv(s_upd)
                x = x + kk @ (z - h_mat @ x)
                x[2], x[3] = _clamp_vec(x[2], x[3], VMAX)
                x[4], x[5] = _clamp_vec(x[4], x[5], max_accel)
                ikh = np.eye(6) - kk @ h_mat
                p = ikh @ p @ ikh.T + kk @ r @ kk.T
            xf[j], pf[j] = x, p
        xs_d = xf.copy()
        for j in range(nd - 2, -1, -1):
            c = pf[j] @ fs[j + 1].T @ np.linalg.pinv(pp[j + 1])
            xs_d[j] = xf[j] + c @ (xs_d[j + 1] - xp[j + 1])

        px = xs_d[:, 0].copy()
        py = xs_d[:, 1].copy()
        if nd >= 3:
            for _ in range(36):
                nx, ny = px.copy(), py.copy()
                changed = False
                for j in range(1, nd - 1):
                    dt1 = max(1e-3, float(td[j] - td[j - 1]))
                    dt2 = max(1e-3, float(td[j + 1] - td[j]))
                    dtm = 0.5 * (dt1 + dt2)
                    w1, w2 = dt2 / (dt1 + dt2), dt1 / (dt1 + dt2)
                    mid_x = w1 * px[j - 1] + w2 * px[j + 1]
                    mid_y = w1 * py[j - 1] + w2 * py[j + 1]
                    vx1, vy1 = (px[j] - px[j - 1]) / dt1, (py[j] - py[j - 1]) / dt1
                    vx2, vy2 = (px[j + 1] - px[j]) / dt2, (py[j + 1] - py[j]) / dt2
                    ax_j, ay_j = (vx2 - vx1) / dtm, (vy2 - vy1) / dtm
                    a_mag = math.hypot(ax_j, ay_j)
                    spd1, spd2 = math.hypot(vx1, vy1), math.hypot(vx2, vy2)
                    turn_rate = 0.0
                    if spd1 > 0.35 and spd2 > 0.35:
                        dh = abs(_wrap(math.atan2(vy2, vx2) - math.atan2(vy1, vx1)))
                        turn_rate = dh / dtm
                    pull = 0.0
                    if a_mag > max_accel * 0.88:
                        pull = max(pull, 0.45 * (1.0 - (max_accel * 0.88) / a_mag))
                    if turn_rate > max_turn_rate * 0.88:
                        pull = max(pull, 0.45 * (1.0 - (max_turn_rate * 0.88) / turn_rate))
                    if pull > 0.0:
                        changed = True
                        nx[j] = px[j] + pull * (mid_x - px[j])
                        ny[j] = py[j] + pull * (mid_y - py[j])
                px, py = nx, ny
                if not changed:
                    break

        # Smooth dense headings consistently with dense velocities
        vx_d = np.zeros(nd, float)
        vy_d = np.zeros(nd, float)
        for j in range(nd):
            if hold_mask[j] and ((j > 0 and hold_mask[j - 1]) or (j + 1 < nd and hold_mask[j + 1])):
                vx_d[j], vy_d[j] = 0.0, 0.0
            elif j == 0:
                dt = max(1e-3, float(td[1] - td[0]))
                vx_d[j], vy_d[j] = _clamp_vec((px[1] - px[0]) / dt, (py[1] - py[0]) / dt, VMAX)
            elif j == nd - 1:
                dt = max(1e-3, float(td[-1] - td[-2]))
                vx_d[j], vy_d[j] = _clamp_vec((px[-1] - px[-2]) / dt, (py[-1] - py[-2]) / dt, VMAX)
            else:
                dt = max(1e-3, float(td[j + 1] - td[j - 1]))
                vx_d[j], vy_d[j] = _clamp_vec((px[j + 1] - px[j - 1]) / dt, (py[j + 1] - py[j - 1]) / dt, VMAX)
        h_in = [m_item[4] for m_item in meta]
        c_in = [m_item[1] for m_item in meta]
        h_dense, _ = smooth_heading(td.tolist(), h_in, c_in, vx=vx_d.tolist(), vy=vy_d.tolist(), max_turn_rate=max_turn_rate)

        for j in range(nd):
            sig_j, conf_j, box_j, _, _ = meta[j]
            trajectory.append({
                't': td_ms[j],
                'x': round(float(px[j]), 4),
                'y': round(float(py[j]), 4),
                'vx': round(float(vx_d[j]), 3),
                'vy': round(float(vy_d[j]), 3),
                'heading': None if h_dense[j] is None else round(float(h_dense[j]), 3),
                'sigma': round(float(sig_j), 3),
                'confidence': round(float(conf_j), 2),
                'box': [round(v, 4) for v in box_j],
                'calibrated': True,
                'estimated': j not in obs_at,
            })
    return trajectory


densify_trajectory = build_smooth_trajectory


def evaluate_track_quality(times_s, xs, ys, raw_xs=None, raw_ys=None, headings=None, kept=None, max_gap_s=GAP_RESET_S, move_speed=0.4):
    """Compute smoothness (jerk RMS, max turn rate, accel) and fidelity (max/p95 deviation from raw detections) metrics."""
    t = np.asarray(times_s, float)
    x = np.asarray(xs, float)
    y = np.asarray(ys, float)
    n = len(t)
    if n < 3:
        return {
            'samples': n, 'sample_hz': None,
            'jerk_rms': 0.0, 'jerk_p95': 0.0, 'jerk_max': 0.0,
            'accel_rms': 0.0, 'accel_p95': 0.0, 'accel_max': 0.0,
            'turn_rate_rms': 0.0, 'turn_rate_p95': 0.0, 'turn_rate_max': 0.0,
            'heading_turn_rate_max': 0.0,
            'deviation_median': 0.0, 'deviation_p95': 0.0, 'deviation_max': 0.0,
            'kept_ratio': 1.0,
        }
    dt = np.diff(t)
    valid_dt = dt[(dt > 1e-4) & (dt <= max_gap_s)]
    sample_hz = float(1.0 / np.median(valid_dt)) if len(valid_dt) else None
    step_gap_s = min(max_gap_s, 0.35 if sample_hz and sample_hz >= 8.0 else max_gap_s)
    vx = np.diff(x) / np.maximum(dt, 1e-3)
    vy = np.diff(y) / np.maximum(dt, 1e-3)
    accels, jerks, turn_rates = [], [], []
    for i in range(len(dt) - 1):
        if dt[i] > step_gap_s or dt[i + 1] > step_gap_s:
            continue
        dtm = 0.5 * (dt[i] + dt[i + 1])
        ax = (vx[i + 1] - vx[i]) / max(dtm, 1e-3)
        ay = (vy[i + 1] - vy[i]) / max(dtm, 1e-3)
        accels.append(math.hypot(ax, ay))
        s1, s2 = math.hypot(vx[i], vy[i]), math.hypot(vx[i + 1], vy[i + 1])
        if s1 >= move_speed and s2 >= move_speed:
            dh = abs(_wrap(math.atan2(vy[i + 1], vx[i + 1]) - math.atan2(vy[i], vx[i])))
            turn_rates.append(dh / max(dtm, 1e-3))
    for i in range(len(dt) - 2):
        if dt[i] > step_gap_s or dt[i + 1] > step_gap_s or dt[i + 2] > step_gap_s:
            continue
        dtm1 = 0.5 * (dt[i] + dt[i + 1])
        dtm2 = 0.5 * (dt[i + 1] + dt[i + 2])
        ax1 = (vx[i + 1] - vx[i]) / max(dtm1, 1e-3)
        ay1 = (vy[i + 1] - vy[i]) / max(dtm1, 1e-3)
        ax2 = (vx[i + 2] - vx[i + 1]) / max(dtm2, 1e-3)
        ay2 = (vy[i + 2] - vy[i + 1]) / max(dtm2, 1e-3)
        dtj = 0.5 * (dtm1 + dtm2)
        jerks.append(math.hypot((ax2 - ax1) / max(dtj, 1e-3), (ay2 - ay1) / max(dtj, 1e-3)))

    h_turn_rates = []
    if headings is not None:
        for i in range(n - 1):
            if dt[i] <= max_gap_s and headings[i] is not None and headings[i + 1] is not None:
                h_turn_rates.append(abs(_wrap(float(headings[i + 1]) - float(headings[i]))) / max(dt[i], 1e-3))

    devs = []
    kept_ratio = 1.0
    if raw_xs is not None and raw_ys is not None:
        rx = np.asarray(raw_xs, float)
        ry = np.asarray(raw_ys, float)
        if kept is not None:
            k_mask = np.asarray(kept, bool)
            kept_ratio = float(k_mask.mean()) if len(k_mask) else 1.0
            if len(x) == len(rx):
                devs = np.hypot(x[k_mask] - rx[k_mask], y[k_mask] - ry[k_mask]).tolist()
            elif len(x) == int(k_mask.sum()):
                devs = np.hypot(x - rx[k_mask], y - ry[k_mask]).tolist()
        elif len(x) == len(rx):
            devs = np.hypot(x - rx, y - ry).tolist()

    def _rms(vals):
        return float(math.sqrt(np.mean(np.square(vals)))) if vals else 0.0

    def _pct(vals, p):
        return float(np.percentile(vals, p)) if vals else 0.0

    def _max(vals):
        return float(np.max(vals)) if vals else 0.0

    return {
        'samples': n,
        'sample_hz': sample_hz,
        'jerk_rms': _rms(jerks),
        'jerk_p95': _pct(jerks, 95),
        'jerk_max': _max(jerks),
        'accel_rms': _rms(accels),
        'accel_p95': _pct(accels, 95),
        'accel_max': _max(accels),
        'turn_rate_rms': _rms(turn_rates),
        'turn_rate_p95': _pct(turn_rates, 95),
        'turn_rate_max': _max(turn_rates),
        'heading_turn_rate_max': _max(h_turn_rates),
        'deviation_median': _pct(devs, 50),
        'deviation_p95': _pct(devs, 95),
        'deviation_max': _max(devs),
        'kept_ratio': kept_ratio,
    }
