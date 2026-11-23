"""A 3D camera recovered from the same ground-plane calibration the robot tracker uses.

The field calibration is a homography between the carpet and the picture. With the usual assumptions
(square pixels, no skew, principal point at the image center) one homography is enough to recover the
focal length and the camera's pose (Zhang's method). That turns a pixel into a ray in field
coordinates, which is what lets a ball in the air be placed in 3D instead of being treated as if it
were on the floor. Lens distortion is ignored; a broadcast lens has some, which is one of the
reasons reprojection residuals are reported with every fit.
"""
import math

import numpy as np


def focal_candidates(g_field_to_pixel, cx, cy):
    """Two independent closed-form estimates of the focal length (pixels) from one homography.

    Each follows from the constraints on the first two columns of K[r1 r2 t]: r1 and r2 are orthogonal,
    and have equal length. They degenerate in different poses (r1.r2 when the camera looks along a
    field axis, the length constraint when it is far above), so both are returned and the caller keeps
    the sane ones.
    """
    g = np.array(g_field_to_pixel, float)
    g = g / g[2, 2]
    a = np.array([g[0, 0] - cx * g[2, 0], g[1, 0] - cy * g[2, 0], g[2, 0]])
    b = np.array([g[0, 1] - cx * g[2, 1], g[1, 1] - cy * g[2, 1], g[2, 1]])
    out = []
    denom = a[2] * b[2]
    if abs(denom) > 1e-12:
        f2 = -(a[0] * b[0] + a[1] * b[1]) / denom
        if f2 > 0:
            out.append(math.sqrt(f2))
    denom = a[2] ** 2 - b[2] ** 2
    if abs(denom) > 1e-12:
        f2 = ((b[0] ** 2 + b[1] ** 2) - (a[0] ** 2 + a[1] ** 2)) / denom
        if f2 > 0:
            out.append(math.sqrt(f2))
    return out


def pose_from_homography(g_field_to_pixel, f, cx, cy):
    """(K, R, t) with x_pixel ~ K (R X + t) for field point X, or None if the pose is not physical."""
    g = np.array(g_field_to_pixel, float)
    g = g / g[2, 2]
    k = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    a = np.linalg.inv(k) @ g
    scale = 1.0 / np.linalg.norm(a[:, 0])
    r1, r2, t = a[:, 0] * scale, a[:, 1] * scale, a[:, 2] * scale
    # The field must be in front of the camera. The overall sign of a homography is arbitrary.
    if (r1 * 8.0 + r2 * 4.0 + t)[2] < 0:
        r1, r2, t = -r1, -r2, -t
    r3 = np.cross(r1, r2)
    u, _, vt = np.linalg.svd(np.column_stack([r1, r2, r3]))
    r = u @ vt
    if np.linalg.det(r) < 0:
        r = -r
    return k, r, t


class Camera:
    """Pinhole camera: project field points to pixels, back-project pixels to rays."""

    def __init__(self, k, r, t):
        self.k, self.r, self.t = k, r, t
        self.p = k @ np.hstack([r, t.reshape(3, 1)])
        self.center = -r.T @ t                   # camera position in field coordinates
        self.f = float(k[0, 0])
        self._k_inv, self._r_t = np.linalg.inv(k), r.T   # reused by every ray(); same arithmetic as inverting each time

    def project(self, xyz):
        """Field xyz (n,3) -> pixels (n,2), and a mask of points in front of the camera."""
        xyz = np.atleast_2d(np.asarray(xyz, float))
        cam = xyz @ self.r.T + self.t
        z = cam[:, 2]
        uv = (cam[:, :2] / np.where(np.abs(z) < 1e-9, 1e-9, z)[:, None]) * self.f + self.k[:2, 2]
        return uv, z > 0.1

    def ray(self, u, v):
        """Unit direction in field coordinates of the ray through pixel (u, v), from `center`."""
        d = self._r_t @ (self._k_inv @ np.array([u, v, 1.0]))
        return d / np.linalg.norm(d)

    def point_at_height(self, u, v, z):
        """Where the ray through (u, v) is at height z above the carpet, or None."""
        d = self.ray(u, v)
        if abs(d[2]) < 1e-6:
            return None
        s = (z - self.center[2]) / d[2]
        return None if s <= 0 else self.center + s * d

    def ground_pixel_scale(self, xyz):
        """Pixels per meter near a field point, along the image's horizontal axis."""
        uv0, _ = self.project(xyz)
        uv1, _ = self.project(np.asarray(xyz, float) + self.r[0] * 0.1)
        return float(np.linalg.norm(uv1 - uv0) / 0.1)


def camera_from_pixel_homography(h_pixel_to_field, width, height, f_prior=None):
    """Camera for one frame from its pixel->field homography. f_prior (pixels) is used when the
    homography alone does not pin the focal length down (a near-degenerate view)."""
    cx, cy = width / 2.0, height / 2.0
    g = np.linalg.inv(np.array(h_pixel_to_field, float))
    candidates = [f for f in focal_candidates(g, cx, cy) if 0.3 * width < f < 6 * width]
    if candidates:
        f = float(np.median(candidates))
        if f_prior and not (0.6 * f_prior < f < 1.7 * f_prior):
            f = f_prior          # an estimate far from the reference zoom is more likely noise than a zoom
    elif f_prior:
        f = f_prior
    else:
        return None
    pose = pose_from_homography(g, f, cx, cy)
    return Camera(*pose) if pose else None


def smooth_cameras(cameras, window=9):
    """Temporally smooth a per-frame list of Cameras (focal length, rotation, position).

    Each frame's camera comes from a homography estimated from image features, which jitters by a
    fraction of a pixel; that becomes pose noise, amplified near the picture's edge. A broadcast camera
    pans and zooms smoothly, so a short moving median removes the jitter without lagging the motion.

    Rotations are filtered as QUATERNIONS with their signs made consistent (q and -q are the same rotation).
    A high camera looking down is rotated close to 180 degrees from the field frame, exactly where rotation
    vectors flip sign and an average of neighbours can land on the opposite rotation.
    """
    if len(cameras) < 3:
        return list(cameras)
    from scipy.ndimage import median_filter
    from scipy.spatial.transform import Rotation
    fs = np.array([c.f for c in cameras])
    quats = Rotation.from_matrix(np.array([c.r for c in cameras])).as_quat()
    for i in range(len(quats)):                      # one hemisphere: every quaternion near the first
        if float(quats[i] @ quats[0]) < 0:
            quats[i] = -quats[i]
    centers = np.array([c.center for c in cameras])
    w = min(window, len(cameras) // 2 * 2 - 1) or 1
    fs = median_filter(fs, size=w, mode='nearest')
    quats = median_filter(quats, size=(w, 1), mode='nearest')
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    centers = median_filter(centers, size=(w, 1), mode='nearest')
    cx, cy = cameras[0].k[0, 2], cameras[0].k[1, 2]
    out = []
    for f, q, center in zip(fs, quats, centers):
        r = Rotation.from_quat(q).as_matrix()
        out.append(Camera(np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]]), r, -r @ center))
    return out
