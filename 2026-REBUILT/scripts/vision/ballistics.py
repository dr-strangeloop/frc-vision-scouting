"""Fit a ball's flight from its track in the picture: 3D launch position and velocity, from pixels.

A single camera cannot place one point in 3D, but a *ballistic arc* is determined: gravity is a known
acceleration, so the curvature of the path in the image fixes the scale along the viewing ray. We fit
the six numbers (launch position, launch velocity) that minimise the reprojection error of the whole
flight, including quadratic air drag (a 15 cm foam ball at 10 m/s loses roughly 2 m/s^2 to drag,
about a fifth of gravity, which is not negligible over a 1 s flight).

The fit also returns how sure it is: reprojection RMS in pixels, and a standard deviation for speed from
the Jacobian, so an unreliable fit is reported as unreliable instead of as a confident number.
"""
import math

import numpy as np
from scipy.optimize import least_squares

from common import FIELD

G = 9.81
# Fuel: 15.0 cm foam ball, about 0.23 kg. Quadratic drag coefficient k = rho * Cd * A / (2 m).
BALL_RADIUS = 0.075
BALL_MASS = 0.23
DRAG_K = 1.2 * 0.47 * (math.pi * BALL_RADIUS ** 2) / (2 * BALL_MASS)
HUB_Z = FIELD['hub']['openingHeight']
HUB_OPENING_RADIUS = 0.53      # 41.7 in across the hexagon's flats


def simulate(x0, v0, times, drag=DRAG_K, dt=1 / 120):
    """Positions at `times` (seconds after launch, sorted, >= 0) with drag, by RK4 on plain floats
    (several times faster than numpy for a system this small, which matters inside the fit)."""
    x, y, z = (float(c) for c in x0)
    vx, vy, vz = (float(c) for c in v0)
    out = []
    t = 0.0
    for target in times:
        while t < target - 1e-12:
            h = min(dt, target - t)

            def acc(ax, ay, az):
                s = math.sqrt(ax * ax + ay * ay + az * az)
                return -drag * s * ax, -drag * s * ay, -G - drag * s * az

            a1 = acc(vx, vy, vz)
            v2 = (vx + 0.5 * h * a1[0], vy + 0.5 * h * a1[1], vz + 0.5 * h * a1[2]); a2 = acc(*v2)
            v3 = (vx + 0.5 * h * a2[0], vy + 0.5 * h * a2[1], vz + 0.5 * h * a2[2]); a3 = acc(*v3)
            v4 = (vx + h * a3[0], vy + h * a3[1], vz + h * a3[2]); a4 = acc(*v4)
            x += h / 6 * (vx + 2 * v2[0] + 2 * v3[0] + v4[0])
            y += h / 6 * (vy + 2 * v2[1] + 2 * v3[1] + v4[1])
            z += h / 6 * (vz + 2 * v2[2] + 2 * v3[2] + v4[2])
            vx += h / 6 * (a1[0] + 2 * a2[0] + 2 * a3[0] + a4[0])
            vy += h / 6 * (a1[1] + 2 * a2[1] + 2 * a3[1] + a4[1])
            vz += h / 6 * (a1[2] + 2 * a2[2] + 2 * a3[2] + a4[2])
            t += h
        out.append((x, y, z))
    return np.array(out)


FIT_DT = 1 / 60          # RK4 step inside the fit: smooth flights need no more, and it halves the cost of every residual
GOOD_ENOUGH_PX = 1.2     # stop trying further starting points once one fits this well
GIVE_UP_AFTER = 2        # starting points to try before concluding a track is not a flight
GIVE_UP_PX = 6.0         # ... if the best so far still misses by more than this
INLIER_PX = 4.0          # a sighting further than this from the fitted arc is an outlier (a track that jumped onto another ball)
ROBUST_SCALE_PX = 1.5    # width of the Cauchy loss: errors beyond a couple of pixels stop pulling on the fit
def simulate_batch(x0, v0, times, drag=DRAG_K, dt=1 / 60):
    """Like simulate(), for many launch states at once: x0, v0 are (m, 3); returns (m, len(times), 3).

    The fit needs the trajectory for the current parameters AND for each parameter nudged (a numerical
    Jacobian), seven simulations that are identical but for their start. Running them as one numpy array makes
    the Python-level loop run once instead of seven times, which is where nearly all the fit time went.
    """
    pos = np.array(x0, float).reshape(-1, 3)
    vel = np.array(v0, float).reshape(-1, 3)
    out = np.empty((pos.shape[0], len(times), 3))
    def acc(v):
        # np.linalg.norm costs more in call overhead than the arithmetic here (it was most of the fit time)
        speed = np.sqrt(v[:, 0] * v[:, 0] + v[:, 1] * v[:, 1] + v[:, 2] * v[:, 2])[:, None]
        a = -drag * speed * v
        a[:, 2] -= G
        return a

    t = 0.0
    for j, target in enumerate(times):
        while t < target - 1e-12:
            h = min(dt, target - t)
            k1 = acc(vel)
            v2 = vel + 0.5 * h * k1; k2 = acc(v2)
            v3 = vel + 0.5 * h * k2; k3 = acc(v3)
            v4 = vel + h * k3; k4 = acc(v4)
            pos = pos + h / 6 * (vel + 2 * v2 + 2 * v3 + v4)
            vel = vel + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
        out[:, j] = pos
    return out


RADIUS_WEIGHT = 0.6      # how much the apparent size counts relative to a pixel of position error


def _residual_batch(thetas, t_rel, uv, cameras, drag, radii):
    """Residual vectors for m parameter sets at once -> (m, n * (2 or 3)), per observation [dx, dy, (dr)].

    Pixel position error, plus (optionally) apparent-radius error. The ball's apparent size is a direct range
    cue: radius_px = f * R / depth. Position alone leaves depth to be inferred from the parabola's curvature,
    which is poorly conditioned when the ball flies nearly along the camera's line of sight. The size term pins
    depth down, at lower weight because motion blur and merged blobs inflate a measured radius.
    """
    thetas = np.atleast_2d(thetas)
    if drag == 0.0:
        # No drag: the trajectory is a closed-form parabola, no integration loop needed (used to screen tracks).
        t = np.asarray(t_rel, float)[None, :, None]
        xyz = thetas[:, None, :3] + thetas[:, None, 3:] * t + 0.5 * np.array([0.0, 0.0, -G]) * t * t
    else:
        xyz = simulate_batch(thetas[:, :3], thetas[:, 3:], t_rel, drag)       # (m, n, 3)
    m, n, _ = xyz.shape
    static = all(c is cameras[0] for c in cameras)
    if static:
        camera = cameras[0]
        flat = xyz.reshape(-1, 3)
        projected, front = camera.project(flat)
        depth = (flat @ camera.r.T + camera.t)[:, 2].reshape(m, n)
        projected, front = projected.reshape(m, n, 2), front.reshape(m, n)
        focal = np.full(n, camera.f)
    else:
        projected, front, depth = np.empty((m, n, 2)), np.empty((m, n), bool), np.empty((m, n))
        for j, camera in enumerate(cameras):
            projected[:, j], front[:, j] = camera.project(xyz[:, j])
            depth[:, j] = (xyz[:, j] @ camera.r.T + camera.t)[:, 2]
        focal = np.array([c.f for c in cameras])
    position = projected - uv[None]
    position[~front] = 1e3
    if radii is None:
        return position.reshape(m, -1)
    size = RADIUS_WEIGHT * (focal[None] * BALL_RADIUS / np.maximum(depth, 0.5) - radii[None])
    return np.concatenate([position, size[..., None]], axis=2).reshape(m, -1)


def _residuals(theta, t_rel, uv, cameras, drag, radii=None):
    return _residual_batch(np.asarray(theta, float), t_rel, uv, cameras, drag, radii)[0]


def _jacobian(theta, t_rel, uv, cameras, drag, radii):
    """Forward-difference Jacobian from ONE batched simulation (the base state and six nudged ones)."""
    steps = 1e-5 * np.maximum(1.0, np.abs(theta))
    thetas = np.vstack([theta] + [theta + np.eye(6)[i] * steps[i] for i in range(6)])
    r = _residual_batch(thetas, t_rel, uv, cameras, drag, radii)
    return ((r[1:] - r[0]) / steps[:, None]).T


def _position_errors(res, has_radii):
    """Pixel distance between predicted and observed position for each observation (radius rows excluded)."""
    per = 3 if has_radii else 2
    rows = np.asarray(res).reshape(-1, per)[:, :2]
    return np.sqrt(np.sum(rows ** 2, axis=1))


def _position_rms(res, has_radii):
    """RMS pixel error over the INLIERS: sightings within INLIER_PX of the arc. A few wild sightings (a blob
    that swapped onto a neighbouring ball in a volley) are reported separately as the outlier fraction
    instead of inflating the residual and getting a good shot thrown out."""
    errors = _position_errors(res, has_radii)
    inliers = errors[errors <= INLIER_PX]
    return float(math.sqrt(np.mean(inliers ** 2))) if inliers.size else float(math.sqrt(np.mean(errors ** 2)))


def _initial_guesses(t_rel, uv, cameras):
    """Back-project the first and last observation to a few plausible heights and solve the straight
    line (plus gravity) between them; the best-fitting candidate seeds the optimiser."""
    guesses = []
    first, last = cameras[0], cameras[-1]
    span = max(t_rel[-1], 1e-3)
    min_v_idx = int(np.argmin(uv[:, 1]))
    descending_only = (min_v_idx <= max(1, len(t_rel) // 4)) and (uv[-1][1] > uv[0][1] + 4.0)
    z_starts = (1.8, 2.3, 2.8, 3.3, 1.2) if descending_only else (0.5, 1.2, 2.2, 2.8)
    z_ends = (HUB_Z - 0.2, HUB_Z, 2.2, 0.8) if descending_only else (0.0, 0.8, HUB_Z, 2.6, 3.4)
    for z_start in z_starts:
        for z_end in z_ends:
            if descending_only and z_start < z_end - 0.1:
                continue
            a = first.point_at_height(uv[0][0], uv[0][1], z_start)
            b = last.point_at_height(uv[-1][0], uv[-1][1], z_end)
            if a is None or b is None:
                continue
            v = (b - a) / span + np.array([0.0, 0.0, 0.5 * G * span])
            guesses.append(np.concatenate([a, v]))
    return guesses


SCREEN_POINTS = 20       # observations used to screen starting points (the full track is used to refine the winner)
SCREEN_NFEV = 25         # iterations a screening fit may spend; a real flight converges in well under this


def _solve(guess, t_rel, uv, cameras, drag, radii, max_nfev, bounds=(-np.inf, np.inf)):
    # Robust loss: plain least squares lets one or two bad sightings drag the whole fit. Cauchy stops errors
    # beyond a couple of pixels from pulling (speed stayed within 0.2% with six 25 to 40 px outliers; linear
    # loss drifted to 2%), and its cost is the same on clean tracks.
    g_clamped = np.clip(guess, bounds[0] + 1e-4, bounds[1] - 1e-4) if np.isfinite(bounds[0]).any() else guess
    return least_squares(_residuals, g_clamped, jac=_jacobian, args=(t_rel, uv, cameras, drag, radii),
                         bounds=bounds, method='trf', loss='cauchy', f_scale=ROBUST_SCALE_PX, max_nfev=max_nfev)


def _plausible(x):
    """Speed and height are at the FIRST OBSERVATION, which can be well after the shot left the robot."""
    speed = float(np.linalg.norm(x[3:]))
    eff_speed = math.sqrt(speed * speed + 2.0 * G * max(0.0, float(x[2]) - 0.75))
    return 1.6 < speed < 22.0 and 2.5 < eff_speed < 24.0 and -0.3 < x[2] < 4.5


def fit_flight(times_s, pixels, cameras, drag=DRAG_K, radii_px=None):
    """Fit one ball. times_s: observation times; pixels: (n,2); cameras: one Camera per observation.

    Returns a dict (launch, velocity, speed, speed_sigma, elevation_deg, azimuth_deg, rms_px, ok, ...) or
    None when there are too few observations or no physically plausible fit.

    Two stages, because most tracks that reach here are not flights and used to cost about a second each:
    candidate starting points are SCREENED on about 20 observations with a small iteration budget (a real
    flight converges in a handful of iterations; junk does not, and is dropped after four starts), and only
    the best start is REFINED on every observation.
    """
    t = np.asarray(times_s, float)
    uv = np.asarray(pixels, float)
    if len(t) < 4:
        return None
    if np.any(np.diff(t) < 0):
        order = np.argsort(t)
        t = t[order]
        uv = uv[order]
        cameras = [cameras[i] for i in order]
        if radii_px is not None:
            radii_px = np.asarray(radii_px, float)[order]
    t_rel = t - t[0]
    radii = None if radii_px is None else np.asarray(radii_px, float)
    # Motion blur and morphological closing on real broadcast volleys inflate blob radii above single-ball size,
    # which would pull the depth estimate into the foreground floor basin.
    min_v_idx = int(np.argmin(uv[:, 1]))
    descending_only = (min_v_idx <= max(1, len(t) // 4)) and (uv[-1, 1] > uv[0, 1] + 4.0)
    if radii is not None and (descending_only or float(np.median(radii)) > 5.8):
        radii = None
    z_min = 1.45 if descending_only else -0.25
    bounds = (
        np.array([-5.0, -5.0, z_min, -20.0, -20.0, -18.0]),
        np.array([FIELD['length'] + 6.0, FIELD['width'] + 5.0, 4.4, 20.0, 20.0, 20.0]),
    )
    n = len(t)
    idx = np.arange(n) if n <= SCREEN_POINTS + 4 else np.unique(np.linspace(0, n - 1, SCREEN_POINTS).round().astype(int))
    screen = (t_rel[idx], uv[idx], [cameras[i] for i in idx], None if radii is None else radii[idx])
    best = None
    tried = 0
    # Choose among starts by TOTAL cost (position AND the apparent-size term), not by position residual alone:
    # two basins can fit the positions equally and differ in depth, which only the size term separates.
    guesses = _initial_guesses(t_rel, uv, cameras)
    guesses.sort(key=lambda g: abs(float(np.linalg.norm(g[3:])) - 9.5))      # plausible launch speeds first
    for guess in guesses:
        try:
            result = _solve(guess, *screen[:3], 0.0, screen[3], SCREEN_NFEV, bounds=bounds)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if not _plausible(result.x):
            continue
        rms = _position_rms(result.fun, radii is not None)
        if best is None or result.cost < best[0].cost:
            best = (result, rms)
        if rms < GOOD_ENOUGH_PX and result.cost <= best[0].cost:
            break
        tried += 1
        # Not a flight: after a few starting points all fit badly, further tries only burn time.
        if tried >= GIVE_UP_AFTER and best[1] > GIVE_UP_PX:
            break
    if best is not None and (best[1] > GIVE_UP_PX or float(np.mean(_position_errors(best[0].fun, radii is not None) > INLIER_PX)) > 0.5):
        return None                      # the screen says this is not a flight: do not pay for a full refit
    if best is not None:
        try:
            refined = _solve(best[0].x, t_rel, uv, cameras, drag, radii, 60, bounds=bounds)
            if _plausible(refined.x):
                best = (refined, _position_rms(refined.fun, radii is not None))
        except (ValueError, np.linalg.LinAlgError):
            pass
    if best is None:
        return None
    result, rms = best
    theta = result.x
    outlier_fraction = float(np.mean(_position_errors(result.fun, radii is not None) > INLIER_PX))
    # Standard deviation of speed from the Jacobian: cov = s^2 (J^T J)^-1, s^2 from the residual.
    sigma_speed = None
    try:
        dof = max(1, len(result.fun) - 6)
        # Floor the noise at a pixel: a fit can be near-perfect and still be one of a family of nearly
        # equally good fits (depth along the line of sight), and scaling by a tiny residual would
        # report false certainty.
        s2 = max(float(np.sum(result.fun ** 2) / dof), 1.0)
        cov = s2 * np.linalg.inv(result.jac.T @ result.jac)
        v = theta[3:]
        grad = v / np.linalg.norm(v)
        sigma_speed = float(math.sqrt(max(0.0, grad @ cov[3:, 3:] @ grad)))
    except np.linalg.LinAlgError:
        pass
    v = theta[3:]
    horizontal = math.hypot(v[0], v[1])
    cam_last = cameras[-1] if cameras else None
    cam_center = cam_last.center.tolist() if (cam_last is not None and hasattr(cam_last, 'center')) else None
    return {
        't0_s': float(t[0]), 'launch': theta[:3].tolist(), 'velocity': v.tolist(),
        'speed': float(np.linalg.norm(v)), 'speed_sigma': sigma_speed,
        'elevation_deg': math.degrees(math.atan2(v[2], horizontal)), 'azimuth_deg': math.degrees(math.atan2(v[1], v[0])),
        'rms_px': rms, 'outlier_fraction': outlier_fraction, 'observations': int(len(t)), 'duration_s': float(t[-1] - t[0]),
        'camera_center': cam_center,
    }


def state_at_exit(fit, exit_height=0.75, max_back_s=0.7, drag=DRAG_K):
    """Integrate the fitted arc BACKWARDS to where the ball was at shooter-exit height.

    The fit's start is the first frame the ball was seen, usually a few frames after it left the
    robot, so its speed is already lower (gravity and drag act the whole time). Scouts want launch
    speed, so go back in time. Returns (position, velocity, seconds_before_first_observation).
    """
    pos = np.array(fit['launch'], float)
    vel = np.array(fit['velocity'], float)
    best = (pos.copy(), vel.copy(), 0.0)
    h = -1 / 240
    t = 0.0
    while -t < max_back_s and pos[2] > exit_height:
        # one RK4 step backwards in time
        def f(p, v):
            sp = float(np.linalg.norm(v))
            return v, np.array([0.0, 0.0, -G]) - drag * sp * v
        k1p, k1v = f(pos, vel)
        k2p, k2v = f(pos + 0.5 * h * k1p, vel + 0.5 * h * k1v)
        k3p, k3v = f(pos + 0.5 * h * k2p, vel + 0.5 * h * k2v)
        k4p, k4v = f(pos + h * k3p, vel + h * k3v)
        pos = pos + h / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
        vel = vel + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        t += h
        if pos[2] < best[0][2]:
            best = (pos.copy(), vel.copy(), t)
    return best


def trajectory_after_launch(fit, horizon_s=3.0, step=1 / 60, drag=DRAG_K):
    times = np.arange(0.0, horizon_s, step)
    pts = simulate(fit['launch'], fit['velocity'], times, drag)
    return times, pts


def ray_corrected_crossing(fit, hubs, camera_center=None, drag=DRAG_K):
    """3D hub-opening crossing corrected for single-camera depth scale along the camera-to-hub viewing ray.

    A single camera constrains the direction from the camera to the ball far more tightly than depth along the ray.
    Rescaling the simulated 3D arc along the viewing ray to each hub's 3D distance tests whether the arc crosses
    the hub opening plane at HUB_Z when placed at the hub's depth.
    """
    if not hubs:
        return None
    if camera_center is None:
        camera_center = fit.get('camera_center')
    if camera_center is None:
        return None
    c_xyz = np.asarray(camera_center, float)
    times, pts = trajectory_after_launch(fit, drag=drag)
    inner = HUB_OPENING_RADIUS - BALL_RADIUS
    best = None
    for name, (hx, hy) in hubs.items():
        h_vec = np.array([float(hx) - c_xyz[0], float(hy) - c_xyz[1], HUB_Z - c_xyz[2]], float)
        h_dist = float(np.linalg.norm(h_vec))
        if h_dist < 1.0:
            continue
        u_ray = h_vec / h_dist
        rel = pts - c_xyz[None, :]
        proj_len = rel @ u_ray
        if not np.any(proj_len > 1.0):
            continue
        scaled = c_xyz[None, :] + rel * (h_dist / np.maximum(proj_len, 1.0))[:, None]
        for i in range(1, len(scaled)):
            if scaled[i - 1, 2] >= HUB_Z > scaled[i, 2] and (scaled[i, 2] - scaled[i - 1, 2]) < 0:
                f = (scaled[i - 1, 2] - HUB_Z) / (scaled[i - 1, 2] - scaled[i, 2])
                t_c = float(times[i - 1] + f * (times[i] - times[i - 1]))
                at_s = scaled[i - 1] + f * (scaled[i] - scaled[i - 1])
                dist = math.hypot(float(at_s[0]) - float(hx), float(at_s[1]) - float(hy))
                scale_at = h_dist / float(max(1.0, proj_len[i - 1] + f * (proj_len[i] - proj_len[i - 1])))
                if 0.35 <= scale_at <= 2.6:
                    res = 'scored' if dist <= inner else 'rim' if dist <= inner + 2.0 * BALL_RADIUS else 'miss'
                    cand = {
                        'result': res, 'target': name, 'miss_m': float(dist),
                        'cross_t_s': t_c, 'cross_xy': [float(at_s[0]), float(at_s[1])],
                        'depth_scale': float(scale_at),
                    }
                    if best is None or dist < best['miss_m']:
                        best = cand
                break
    return best


def classify_flight(fit, hubs, drag=DRAG_K, camera=None):
    """What the ball did, from the fitted arc: scored, rim, or missed which hub, or landed short.

    Integrates the arc forward until it comes down through hub height and measures how far from each hub
    center it crosses. `hubs` maps a name to (x, y) of the hub center. A ball crossing within
    HUB_OPENING_RADIUS - BALL_RADIUS of the center drops in; one crossing within a ball diameter outside
    that is a rim-out or a coin flip; further is a miss. When the first observation is already descending
    just below HUB_Z, integrates backwards up to 0.35 s to recover the descending HUB_Z crossing and apex.
    """
    times, pts = trajectory_after_launch(fit, drag=drag)
    descending = None
    for i in range(1, len(pts)):
        if pts[i - 1][2] >= HUB_Z > pts[i][2] and (pts[i][2] - pts[i - 1][2]) < 0:
            f = (pts[i - 1][2] - HUB_Z) / (pts[i - 1][2] - pts[i][2])
            descending = (times[i - 1] + f * (times[i] - times[i - 1]), pts[i - 1] + f * (pts[i] - pts[i - 1]))
            break
    apex = float(np.max(pts[:, 2]))
    if descending is None and len(pts) > 0 and 1.4 <= float(pts[0][2]) < HUB_Z and float(fit['velocity'][2]) < -0.2:
        pos = np.array(fit['launch'], float)
        vel = np.array(fit['velocity'], float)
        h = -1 / 120
        t_b = 0.0
        while -t_b < 0.35 and vel[2] < 0.0:
            prev_pos, prev_t = pos.copy(), t_b
            sp = float(np.linalg.norm(vel))
            acc = np.array([0.0, 0.0, -G]) - drag * sp * vel
            pos = pos + h * vel
            vel = vel + h * acc
            t_b += h
            apex = max(apex, float(pos[2]))
            if prev_pos[2] < HUB_Z <= pos[2] and descending is None:
                f = (HUB_Z - prev_pos[2]) / max(1e-9, pos[2] - prev_pos[2])
                descending = (prev_t + f * (t_b - prev_t), prev_pos + f * (pos - prev_pos))
    cam_center = camera.center if (camera is not None and hasattr(camera, 'center')) else fit.get('camera_center')
    ray_cross = ray_corrected_crossing(fit, hubs, camera_center=cam_center, drag=drag) if cam_center is not None else None
    if descending is None or not hubs:
        landing = next((p for p in pts[1:] if p[2] <= 0.0), pts[-1])
        out = {'result': 'short', 'target': None, 'apex_m': apex, 'miss_m': None, 'landing': [float(landing[0]), float(landing[1])]}
        if ray_cross is not None:
            out['ray_crossing'] = ray_cross
        return out
    t_cross, at = descending
    name, dist = min(((n, math.hypot(at[0] - h[0], at[1] - h[1])) for n, h in hubs.items()), key=lambda x: x[1])
    inner = HUB_OPENING_RADIUS - BALL_RADIUS
    result = 'scored' if dist <= inner else 'rim' if dist <= inner + 2 * BALL_RADIUS else 'miss'
    out = {'result': result, 'target': name, 'apex_m': apex, 'miss_m': float(dist), 'cross_t_s': float(t_cross), 'cross_xy': [float(at[0]), float(at[1])]}
    if ray_cross is not None:
        out['ray_crossing'] = ray_cross
    return out
