"""Is this footage worth processing? A quick check before spending minutes on a whole match.

Every team that has tried automated scouting from broadcast video reports the same thing: the camera angle,
zoom, resolution and compression of the footage decide the result far more than the model does, and a bad
video wastes a long processing run. This samples a few moments of a video and reports, per property, whether
it is fine, worth a warning, or poor, with what to do about it. It never blocks a run; it tells a person.
"""
import math

import cv2
import numpy as np

OK, WARN, POOR = 'ok', 'warn', 'poor'
WEIGHT = {OK: 1.0, WARN: 0.6, POOR: 0.2}
WORK_WIDTH = 960                    # all pixel measures are made at this width so they do not depend on resolution


def _resize(frame):
    h, w = frame.shape[:2]
    if w == WORK_WIDTH:
        return frame
    return cv2.resize(frame, (WORK_WIDTH, max(1, round(h * WORK_WIDTH / w))), interpolation=cv2.INTER_AREA)


def _check(name, level, value, text, advice=None):
    return {'name': name, 'level': level, 'value': value, 'text': text, 'advice': advice}


def check_resolution(height):
    if height >= 720:
        return _check('resolution', OK, height, f'{height}p')
    if height >= 480:
        return _check('resolution', WARN, height, f'{height}p', 'Small balls are a few pixels wide at this size. Download 720p or better if the broadcast offers it.')
    return _check('resolution', POOR, height, f'{height}p', 'Too small to see balls reliably. Find a higher-resolution copy of the broadcast.')


def check_fps(fps):
    if fps >= 29.5:
        return _check('frame rate', OK, round(fps, 1), f'{fps:.0f} fps')
    if fps >= 23.5:
        return _check('frame rate', WARN, round(fps, 1), f'{fps:.0f} fps', 'A shot moves a long way between frames at this rate, so speeds are less precise. 30 fps or more is better.')
    return _check('frame rate', POOR, round(fps, 1), f'{fps:.0f} fps', 'Too few frames per second to follow a ball in flight.')


def check_sharpness(frames):
    """Median variance of the Laplacian: low means blur or heavy compression smearing small objects."""
    values = [cv2.Laplacian(cv2.cvtColor(_resize(f), cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() for f in frames]
    v = float(np.median(values))
    if v >= 60:
        return _check('sharpness', OK, round(v, 1), 'sharp')
    if v >= 20:
        return _check('sharpness', WARN, round(v, 1), 'soft', 'The picture is soft (compression or blur). Balls will be harder to separate from the background.')
    return _check('sharpness', POOR, round(v, 1), 'very soft', 'The video is heavily blurred or compressed. Try another download or a different stream quality.')


def check_exposure(frames):
    v = float(np.median([cv2.cvtColor(_resize(f), cv2.COLOR_BGR2GRAY).mean() for f in frames]))
    if 60 <= v <= 190:
        return _check('exposure', OK, round(v, 1), 'well exposed')
    if 40 <= v <= 215:
        return _check('exposure', WARN, round(v, 1), 'dark' if v < 60 else 'bright', 'The picture is on the dark or bright side; detection may miss balls.')
    return _check('exposure', POOR, round(v, 1), 'too dark' if v < 40 else 'washed out', 'The picture is badly under or over exposed.')


def _centre(frame):
    """The middle of the picture. Broadcast overlays (top banner, bug, lower-third graphics) sit at the edges and never move,
    so measuring the whole frame would report a steady camera while the field pans underneath them."""
    img = _resize(frame)
    h, w = img.shape[:2]
    return img[int(h * 0.25):int(h * 0.75), int(w * 0.2):int(w * 0.8)]


def _shift(a, b):
    ga = np.float32(cv2.cvtColor(_centre(a), cv2.COLOR_BGR2GRAY))
    gb = np.float32(cv2.cvtColor(_centre(b), cv2.COLOR_BGR2GRAY))
    window = cv2.createHanningWindow(ga.shape[::-1], cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(ga, gb, window)
    return math.hypot(dx, dy)


def check_camera_motion(pairs):
    """Median image shift between two moments half a second apart: a tripod camera barely moves."""
    if not pairs:
        return _check('camera motion', WARN, None, 'not measured', 'Could not read two frames close together to see whether the camera moves.')
    v = float(np.median([_shift(a, b) for a, b in pairs]))
    if v < 1.5:
        return _check('camera motion', OK, round(v, 1), 'steady')
    if v < 8:
        return _check('camera motion', WARN, round(v, 1), 'slight movement', 'The camera drifts a little. Tracking will register each frame to the field, which costs a little accuracy.')
    return _check('camera motion', POOR, round(v, 1), 'panning or zooming', 'The camera pans or zooms. Expect lower accuracy; a fixed wide shot of the field works far better.')


def yellow_blobs(frame):
    """Count yellow blobs of ball size. Yellow clothing and signage show up here too, so a high count means clutter."""
    hsv = cv2.cvtColor(_resize(frame), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (18, 110, 120), (38, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    return sum(1 for i in range(1, n) if 12 <= stats[i, cv2.CC_STAT_AREA] <= 900)


def check_fuel_visibility(frames):
    counts = [yellow_blobs(f) for f in frames]
    median = float(np.median(counts))
    if max(counts) == 0:
        return _check('fuel visibility', WARN, 0, 'no yellow balls seen', 'No fuel was visible in the sampled moments. The window may be before the match starts, or the balls are hidden or off screen.')
    if median > 200:
        return _check('fuel visibility', WARN, median, 'cluttered', 'Many yellow objects are in view (spectators, signs), so the detector will see false balls. Mark the field region of interest to cut them out.')
    return _check('fuel visibility', OK, median, 'balls visible')


def assess_frames(width, height, fps, frames, pairs=()):
    """The report for already-read frames. `frames` are BGR arrays; `pairs` are (earlier, later) frames about half a second apart."""
    checks = [check_resolution(height), check_fps(fps)]
    if frames:
        checks += [check_sharpness(frames), check_exposure(frames), check_fuel_visibility(frames)]
    checks.append(check_camera_motion(list(pairs)))
    score = float(np.mean([WEIGHT[c['level']] for c in checks]))
    worst = min(checks, key=lambda c: WEIGHT[c['level']])
    if score < 0.55 or (worst['level'] == POOR and score < 0.75):
        verdict = 'poor'
    elif score >= 0.85 and worst['level'] == OK:
        verdict = 'good'
    else:
        verdict = 'usable'
    return {'verdict': verdict, 'score': round(score, 2), 'checks': checks, 'advice': [c['advice'] for c in checks if c['advice']]}


def assess_video(path, start_s=0.0, end_s=None, samples=6, gap_s=0.5):
    """Sample `samples` moments between start_s and end_s and assess the video. Raises ValueError if it cannot be opened."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f'cannot open video {path}')
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None
        end = min(end_s if end_s is not None else (total or start_s + 60), total or 1e9)
        frames, pairs = [], []
        for k in range(samples):
            t = start_s + (end - start_s) * (k + 0.5) / samples
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, a = cap.read()
            if not ok:
                continue
            frames.append(a)
            cap.set(cv2.CAP_PROP_POS_MSEC, (t + gap_s) * 1000)
            ok, b = cap.read()
            if ok:
                pairs.append((a, b))
    finally:
        cap.release()
    if not frames:
        raise ValueError(f'no frames could be read from {path} between {start_s}s and {end}s')
    return assess_frames(width, height, fps, frames, pairs)


def format_report(report):
    marks = {OK: 'ok  ', WARN: 'warn', POOR: 'POOR'}
    lines = [f"Footage check: {report['verdict']} ({report['score']:.2f})"]
    lines += [f"  [{marks[c['level']]}] {c['name']}: {c['text']}" for c in report['checks']]
    lines += [f'  -> {a}' for a in report['advice']]
    return '\n'.join(lines)
