"""Where in the picture the wide field camera is, so nothing else is ever analysed.

A broadcast is rarely one full-frame camera. The 2026 stream splits the picture: a high wide shot on top,
hub close-ups below, a top overlay banner over it, and a crowd in yellow shirts behind the field. Projecting a
detection from a close-up panel through the wide camera's homography puts it somewhere on the field that
is simply wrong, and yellow shirts look like fuel. So the calibration carries a region of interest (a
rectangle of the picture, normalised 0-1), and on top of it the FIELD VOLUME is projected into the
picture: the carpet plus the air above it up to where a ball can fly. Only pixels inside both are used,
both for finding robots and balls and for registering the camera.
"""
import cv2
import numpy as np

from common import FIELD

MAX_BALL_HEIGHT = 4.2     # meters: the apex of the highest shots seen, with margin


def roi_rect_mask(roi, width, height):
    """255 inside the normalised rectangle [x0, y0, x1, y1], or the whole frame when roi is None."""
    mask = np.zeros((height, width), np.uint8)
    if not roi:
        mask[:] = 255
        return mask
    x0, y0, x1, y1 = (float(v) for v in roi)
    mask[int(max(0, y0) * height):int(min(1, y1) * height) + 1, int(max(0, x0) * width):int(min(1, x1) * width) + 1] = 255
    return mask


def field_volume_mask(camera, width, height, margin=0.6):
    """255 inside the picture region the field's volume projects to (carpet to MAX_BALL_HEIGHT, plus margin).

    Built from the 3D camera recovered from the calibration: the eight corners of the field box are
    projected and their convex hull filled. Balls outside it (the stands) can never be on the field.
    """
    xs = (-margin, FIELD['length'] + margin)
    ys = (-margin, FIELD['width'] + margin)
    corners = np.array([[x, y, z] for x in xs for y in ys for z in (0.0, MAX_BALL_HEIGHT)])
    uv, front = camera.project(corners)
    mask = np.zeros((height, width), np.uint8)
    if front.sum() < 4:
        mask[:] = 255
        return mask
    hull = cv2.convexHull(np.round(uv[front]).astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


def combined_mask(calibration, camera, width, height):
    """Reference-frame mask: inside the ROI rectangle AND inside the projected field volume."""
    roi = (calibration or {}).get('roi')
    mask = roi_rect_mask(roi, width, height)
    if camera is not None:
        mask = cv2.bitwise_and(mask, field_volume_mask(camera, width, height))
    return mask


def warp_mask_to_frame(mask_ref, h_cur_to_ref, width, height):
    """The reference-frame mask as it sits in the current frame (the camera may have panned)."""
    h_ref_to_cur = np.linalg.inv(h_cur_to_ref)
    return cv2.warpPerspective(mask_ref, h_ref_to_cur, (width, height), flags=cv2.INTER_NEAREST)
