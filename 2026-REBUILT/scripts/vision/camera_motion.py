"""Keep a panning broadcast camera registered to one reference frame.

The calibrated homography maps pixels of a REFERENCE frame to the field. A broadcast camera pans,
tilts and zooms, so the pixel under a robot at time t is not the pixel it would have had in the
reference frame. This module estimates, frame by frame, the homography from the current frame
back to the reference frame (ORB features on the carpet, robots masked out, RANSAC), chains them,
and re-anchors straight to the reference whenever the view is similar enough to cancel accumulated
drift. A cut (a new camera) shows up as too few inliers; the caller then ends the segment instead
of projecting garbage.

Accuracy: features are taken at 0.75x scale. At 0.5x, registration inside a region of interest was
0.9 px off on average (1.4 px worst); at 0.75x it is 0.18 px, for about 2 ms more, and the static gate
means that path is the exception.

Speed: registration was the dominant per-frame cost (about 11 of 14 ms on 720p), and most broadcast
cameras that matter here are mounted and do not move at all. So there are two shortcuts:
  * features are searched only inside the region of interest (the bounding box of the reference mask);
  * a "static gate" compares a small thumbnail of that region with the reference by phase
    correlation, on its left and right halves separately (a zoom shifts the halves in opposite
    directions, a pan shifts both). If neither moved by a fraction of a pixel, the camera is static
    and the answer is the identity, for about 1 ms instead of 11.
"""
import cv2
import numpy as np

THUMB_SCALE = 0.5            # the static gate works on half-resolution thumbnails
STATIC_SHIFT_PX = 0.3        # in thumbnail pixels, so about 0.6 px in the full frame
STATIC_MIN_RESPONSE = 0.12   # phase-correlation peak strength; low means the content changed, not just moved


def _even_dft_dim(n):
    """Size near `n` whose OpenCV optimal DFT size is even, avoiding phaseCorrelate's 0.5 px odd-DFT shift."""
    n = max(24, int(round(n)))
    while n > 24 and cv2.getOptimalDFTSize(n) % 2 == 1:
        n -= 1
    while cv2.getOptimalDFTSize(n) % 2 == 1:
        n += 1
    return n


class CameraMotion:
    def __init__(self, features=1500, ransac_px=3.0, min_inliers=25, reanchor_every=15, reanchor_min_inliers=60, scale=0.75, static_gate=True):
        self.scale = scale
        self.orb = cv2.ORB_create(nfeatures=features, fastThreshold=12)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.ransac_px = ransac_px
        self.min_inliers = min_inliers
        self.reanchor_every = reanchor_every
        self.reanchor_min_inliers = reanchor_min_inliers
        self.static_gate = static_gate
        self.ref = None
        self.prev = None
        self.crop = None            # (x, y, w, h) of the region features are taken from
        self.frame_area = 1.0
        self.ref_thumb = None
        self._thumb_size = None
        self._hanning = None
        self._ref_halves = None
        self.h_prev_to_ref = np.eye(3)
        self.frames = 0
        self.static_frames = 0

    # -- feature plumbing -------------------------------------------------------------
    def _bbox(self, shape, mask):
        if mask is None:
            return (0, 0, shape[1], shape[0])
        x, y, w, h = cv2.boundingRect(mask)
        return (x, y, w, h) if w > 16 and h > 16 else (0, 0, shape[1], shape[0])

    def _thumb(self, gray):
        x, y, w, h = self.crop
        if self._thumb_size is None:
            half_w = _even_dft_dim(w * THUMB_SCALE / 2.0)
            th_h = _even_dft_dim(h * THUMB_SCALE)
            self._thumb_size = (2 * half_w, th_h)
        return cv2.resize(gray[y:y + h, x:x + w], self._thumb_size, interpolation=cv2.INTER_AREA).astype(np.float32)

    def _features(self, gray, mask):
        """(points in FULL-frame pixels, descriptors) from the cropped region."""
        x, y, w, h = self.crop
        small = cv2.resize(gray[y:y + h, x:x + w], None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        small_mask = None
        if mask is not None:
            small_mask = cv2.resize(mask[y:y + h, x:x + w], (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
        keypoints, descriptors = self.orb.detectAndCompute(small, small_mask)
        points = np.array([kp.pt for kp in keypoints], np.float32).reshape(-1, 2) / self.scale + np.array([x, y], np.float32)
        return points, descriptors

    def _homography(self, a, b):
        """Homography mapping frame-b pixels -> frame-a pixels (full resolution), and inlier count."""
        (pa, da), (pb, db) = a, b
        if da is None or db is None or len(pa) < 8 or len(pb) < 8:
            return None, 0
        matches = self.matcher.match(db, da)
        if len(matches) < 8:
            return None, 0
        src = pb[[m.queryIdx for m in matches]]
        dst = pa[[m.trainIdx for m in matches]]
        h, inliers = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac_px)
        if h is None:
            return None, 0
        count = int(inliers.sum())
        # RANSAC picks a good consensus but its estimate comes from a minimal sample, so it is only accurate
        # to about a pixel. Refit by least squares over all inliers: measured 1.3 px -> 0.5 px mean corner
        # error for about 0.3 ms. (A second, tighter reselection pass was tried and made the worst case
        # worse, and ECC refinement diverged outright, so neither is used.)
        chosen = inliers.ravel().astype(bool)
        if chosen.sum() >= 8:
            refined = self._fit_model(src[chosen], dst[chosen])
            if refined is not None and np.isfinite(refined).all():
                h = refined
        return h, count

    def _fit_model(self, src, dst):
        """Least-squares motion model over inlier matches.

        A full homography (8 parameters) is right when the matches span the picture, but fitted over a small
        region it is ill-conditioned: the translation can trade off against the perspective terms, and the
        mapping inside the region (the only part used) came out 1 to 2 px off. When the matches cover less
        than half of the frame a 6-parameter affine model is used instead: it is well conditioned, exact for
        pan, zoom and roll of a distant camera, and the perspective terms it drops are negligible there.
        """
        x0, y0 = src.min(axis=0)
        x1, y1 = src.max(axis=0)
        spread = ((x1 - x0) * (y1 - y0)) / max(1.0, self.frame_area)
        if spread >= 0.5:
            refined, _ = cv2.findHomography(src, dst, 0)
            return refined
        design = np.hstack([src, np.ones((len(src), 1))])
        coeffs, *_ = np.linalg.lstsq(design, dst, rcond=None)           # (3, 2): dst = [x y 1] @ coeffs
        return np.array([[coeffs[0, 0], coeffs[1, 0], coeffs[2, 0]], [coeffs[0, 1], coeffs[1, 1], coeffs[2, 1]], [0.0, 0.0, 1.0]])

    def is_static(self, gray):
        """True when the region looks like the reference: no translation and no zoom."""
        if not self.static_gate or self.ref_thumb is None:
            return False
        thumb = self._thumb(gray)
        half = thumb.shape[1] // 2
        if half < 24:
            return False
        if self._hanning is None or self._hanning.shape != (thumb.shape[0], half):
            self._hanning = cv2.createHanningWindow((half, thumb.shape[0]), cv2.CV_32F)
            self._ref_halves = (
                np.ascontiguousarray(self.ref_thumb[:, :half] * self._hanning),
                np.ascontiguousarray(self.ref_thumb[:, half:2 * half] * self._hanning),
            )
        dxs, dys = [], []
        for idx, sl in enumerate((slice(0, half), slice(half, 2 * half))):
            cur_win = np.ascontiguousarray(thumb[:, sl] * self._hanning)
            (dx, dy), response = cv2.phaseCorrelate(self._ref_halves[idx], cur_win)
            if response < STATIC_MIN_RESPONSE:
                return False
            dxs.append(dx)
            dys.append(dy)
        max_dx = max(abs(dxs[0]), abs(dxs[1]))
        max_dy = max(abs(dys[0]), abs(dys[1]))
        if max_dx <= STATIC_SHIFT_PX and max_dy <= STATIC_SHIFT_PX:
            return True
        # Moving robots/balls inside the ROI can pull the subpixel phase-correlation centroid slightly
        # (< 0.78 thumb px) on a mounted camera while the static field background (>=35% of pixels) is unchanged.
        if max_dx <= 0.78 and max_dy <= 0.40 and not (dxs[0] * dxs[1] < -0.02):
            if float(np.percentile(np.abs(thumb[::2, ::2] - self.ref_thumb[::2, ::2]), 35)) <= 6.5:
                return True
        return False

    # -- public API -------------------------------------------------------------------
    def start(self, gray, mask=None):
        """Declare `gray` the reference frame (the one the field homography was solved on)."""
        self.crop = self._bbox(gray.shape, mask)
        self.frame_area = float(gray.shape[0] * gray.shape[1])
        self._thumb_size = None
        features = self._features(gray, mask)
        self.ref = features
        self.prev = features
        self.ref_thumb = self._thumb(gray)
        half = self.ref_thumb.shape[1] // 2
        if half >= 24:
            self._hanning = cv2.createHanningWindow((half, self.ref_thumb.shape[0]), cv2.CV_32F)
            self._ref_halves = (
                np.ascontiguousarray(self.ref_thumb[:, :half] * self._hanning),
                np.ascontiguousarray(self.ref_thumb[:, half:2 * half] * self._hanning),
            )
        else:
            self._hanning = None
            self._ref_halves = None
        self.h_prev_to_ref = np.eye(3)
        self.frames = 0
        self.static_frames = 0
        return np.eye(3)

    def update(self, gray, mask=None, prefer_reference=False, ref_mask=None, mask_factory=None):
        """Return (H current->reference, ok). ok is False when the view cannot be registered (a cut).

        prefer_reference registers straight against the reference frame whenever it still matches well,
        which cannot accumulate drift the way chaining frame-to-frame can; chaining is the fallback once
        the view has moved too far from the reference to match it directly. mask_factory builds the per-frame
        feature mask lazily, so frames the static gate settles never pay for it.
        """
        self.frames += 1
        if self.is_static(gray):
            self.static_frames += 1
            self.h_prev_to_ref = np.eye(3)
            return self.h_prev_to_ref, True
        if mask is None and mask_factory is not None:
            mask = mask_factory()                  # only built when ORB actually has to run
        if ref_mask is not None:
            # ref_mask is in REFERENCE-frame pixels; bring it into this frame using the last known motion.
            if np.array_equal(self.h_prev_to_ref, np.eye(3)):
                warped = ref_mask
            else:
                h, w = gray.shape[:2]
                warped = cv2.warpPerspective(ref_mask, np.linalg.inv(self.h_prev_to_ref), (w, h), flags=cv2.INTER_NEAREST)
            mask = warped if mask is None else cv2.bitwise_and(mask, warped)
        features = self._features(gray, mask)
        if prefer_reference:
            h_ref, inliers_ref = self._homography(self.ref, features)
            if h_ref is not None and inliers_ref >= self.reanchor_min_inliers:
                self.h_prev_to_ref, self.prev = h_ref / h_ref[2, 2], features
                return self.h_prev_to_ref, True
        h_step, inliers = self._homography(self.prev, features)
        if h_step is None or inliers < self.min_inliers:
            # Try the reference directly before giving up; a brief occlusion should not end a segment.
            h_ref, inliers_ref = self._homography(self.ref, features)
            if h_ref is None or inliers_ref < self.min_inliers:
                return self.h_prev_to_ref, False
            self.h_prev_to_ref, self.prev = h_ref, features
            return h_ref, True
        h = self.h_prev_to_ref @ h_step
        if self.frames % self.reanchor_every == 0:
            h_ref, inliers_ref = self._homography(self.ref, features)
            if h_ref is not None and inliers_ref >= self.reanchor_min_inliers:
                h = h_ref
        self.h_prev_to_ref = h / h[2, 2]
        self.prev = features
        return self.h_prev_to_ref, True


def robot_mask(shape, boxes_px, pad=6):
    """255 everywhere except inside robot boxes (so features come from the static field)."""
    mask = np.full(shape[:2], 255, np.uint8)
    for x1, y1, x2, y2 in boxes_px:
        cv2.rectangle(mask, (int(x1) - pad, int(y1) - pad), (int(x2) + pad, int(y2) + pad), 0, -1)
    return mask


def shot_changed(prev_gray, gray, threshold=38.0, roi=None):
    """Cheap cut detector: mean absolute difference of 64x36 thumbnails (of the ROI when given, so close-up
    panels that cut constantly in a split-screen broadcast do not look like cuts of the field camera)."""
    if roi is not None:
        x, y, w, h = roi
        prev_gray, gray = prev_gray[y:y + h, x:x + w], gray[y:y + h, x:x + w]
    a = cv2.resize(prev_gray, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
    b = cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
    return float(np.abs(a - b).mean()) > threshold
