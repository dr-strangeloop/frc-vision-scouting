"""Find a split-screen broadcast's wide camera from the picture itself.

Panes meet at a fixed horizontal seam: one row that stays a sharp edge across the picture through the whole match, while
everything else moves. A seam is thin, with quiet rows on both sides, so the row that stands out most from the rows just
beside it in the middle band is the wide shot's bottom edge: 2026cc and 2026cacg give 0.595-0.597 (0.595 was first set by
hand) and 2026dal and 2026hop 0.651. The strongest edge is not safe, because a wide view's near guardrail is as strong as
a seam but busy beside it (0.506 at 2026cacg, 0.592 at 2026hop), and a band ending at 0.72 leaves out 2026cc's panel edge
at 0.79. A single-camera picture has no seam, and keeps the whole frame.
"""
from __future__ import annotations

import numpy as np

TOP = 0.2                 # above the scoreboard's lower edge, and above any robot on the far side of the field
SEARCH = (0.5, 0.72)      # share of the picture height
EDGE_STEP = 18            # grey levels between neighbouring rows
MIN_EDGE_SHARE = 0.35     # share of the picture width that must be an edge
BESIDE = 0.003            # share of the picture height between a row and the rows compared with it
MIN_LEAD = 0.15           # how much more of the width must be an edge than in the rows beside it
SAMPLE_S = range(20, 180, 12)


def seam_row(frames):
    """The thinnest strong seam in SEARCH as a share of the height, or None. `frames`: grey arrays, one per moment."""
    if not frames:
        return None
    height = frames[0].shape[0]
    edges = np.median([(np.abs(frame[1:].astype(np.float32) - frame[:-1].astype(np.float32)) > EDGE_STEP).mean(axis=1)
                       for frame in frames], axis=0)
    beside = max(1, round(BESIDE * height))
    lead = edges - np.maximum(np.roll(edges, beside), np.roll(edges, -beside))
    lead[edges < MIN_EDGE_SHARE] = -1
    low, high = int(SEARCH[0] * height), int(SEARCH[1] * height)
    row = low + int(np.argmax(lead[low:high]))
    return round(row / height, 3) if lead[row] >= MIN_LEAD else None


def wide_pane(video_path):
    """The wide camera's detection area [x0, y0, x1, y1] as picture shares, or None for a single-camera picture."""
    import cv2
    capture = cv2.VideoCapture(str(video_path))
    frames = []
    for second in SAMPLE_S:
        capture.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
        ok, frame = capture.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    capture.release()
    bottom = seam_row(frames)
    return None if bottom is None else [0, TOP, 1, bottom]
