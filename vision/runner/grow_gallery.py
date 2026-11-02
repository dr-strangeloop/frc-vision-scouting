"""Add a match's literal bumper reads to the private event gallery, so body identity can run on its roster.

Scouting keeps its reads (identity inputs) even when the gallery lacks a roster team. This crops each confident read from
the same recording and offers it to `IdentityGallery.add`, which applies every gallery rule: literal reads only, source
bound, no repeated pixels, a bounded number per team. Predictions never enter. Private Spark step; nothing is uploaded.

    python3 grow_gallery.py <event.gallery.json> <inputs.json.gz> <recording.mp4> <match_key>
"""
from __future__ import annotations

import base64
import hashlib
import os
import sys

from identity_gallery import IdentityGallery
from identity_refinement import load_inputs

MIN_CONFIDENCE = 0.8
BOX_PAD = 0.15
READER = 'qwen-compact-bumper-read'


def gallery_rows(inputs, frame_at, recording_sha256, match_key, encode, reader_revision):
    """Gallery rows for confident literal reads. `frame_at(t_ms)` returns (image array) or None; `encode(crop)` JPEG bytes."""
    points = {(tr['id'], p['t']): (p['box'], tr['alliance']) for tr in inputs['tracklets'] for p in tr['points']}
    rows = []
    for read in inputs['observations']:
        team, confidence = read.get('team_key'), read.get('confidence')
        found = points.get((read.get('tracklet'), read.get('t')))
        # A close-up read names this tracklet, but its pixels are in the panel, not in this box.
        if (read.get('identity_source') != 'bumper_read' or read.get('via') or not team or found is None
                or not isinstance(confidence, (int, float)) or isinstance(confidence, bool)
                or not MIN_CONFIDENCE <= confidence <= 1):
            continue
        frame = frame_at(read['t'])
        if frame is None:
            continue
        (x0, y0, x1, y1), alliance = found
        pad_x, pad_y = (x1 - x0) * BOX_PAD, (y1 - y0) * BOX_PAD
        height, width = frame.shape[:2]
        left, top = max(0, int(x0 - pad_x)), max(0, int(y0 - pad_y))
        right, bottom = min(width, int(x1 + pad_x)), min(height, int(y1 + pad_y))
        if right <= left or bottom <= top:
            continue
        jpeg = encode(frame[top:bottom, left:right])
        rows.append({'team_key': team, 'digits': team[3:], 'identity_source': 'bumper_read', 'confidence': float(confidence),
                     'quality': float((right - left) * (bottom - top)), 'crop_sha256': hashlib.sha256(jpeg).hexdigest(),
                     'jpeg_base64': base64.b64encode(jpeg).decode(),
                     'source': {'match_key': match_key, 't_ms': int(read['t']), 'box': [left, top, right, bottom],
                                'alliance': alliance, 'recording_sha256': recording_sha256, 'reader': READER,
                                'reader_revision': reader_revision}})
    return rows


def main(argv):
    gallery_path, inputs_path, video_path, match_key = argv[1:5]
    grow(gallery_path, load_inputs(inputs_path), video_path, match_key)


def grow(gallery_path, inputs, video_path, match_key):
    """Offer `inputs`' confident literal reads to the event gallery, and save it."""
    import cv2
    digest = hashlib.sha256()
    with open(video_path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    capture = cv2.VideoCapture(video_path)

    def frame_at(t_ms):
        capture.set(cv2.CAP_PROP_POS_MSEC, t_ms)
        ok, frame = capture.read()
        return frame if ok else None

    def encode(crop):
        return cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()
    gallery = IdentityGallery(inputs['event_key'], gallery_path)
    rows = gallery_rows(inputs, frame_at, digest.hexdigest(), match_key, encode, os.environ.get('VISION_QWEN_REVISION') or 'unknown')
    added = sum(gallery.add(row) for row in rows)
    gallery.save()
    missing = gallery.templates(inputs['roster'])['missing']
    print({'match': match_key, 'confident_reads': len(rows), 'added': added, 'still_missing': missing})


if __name__ == '__main__':
    main(sys.argv)
