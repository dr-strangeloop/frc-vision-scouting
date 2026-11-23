#!/usr/bin/env python3
"""Calibrate a venue's broadcast camera from one match video, with no hand marks.

    python3 vision.py calibrate --youtube <video id> --out cal.json

1. Gets the match video (YouTube id or --file).
2. Finds the wide camera on a split-screen broadcast (or uses the whole picture).
3. Calibrates the picture from the AprilTags on each HUB's front face (WPILib's 2026 layout in data/).
4. Tunes the HUB light boxes so the lights show two clear states, which is how the match start and each shift's
   active HUB are found; broadcasts differ, and a band a few centimetres off reads nothing.
5. Checks that the match start is found, and writes a preview image (cal.preview.jpg) to look at.

One calibration serves every match filmed from the same camera position at an event.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path[:0] = [str(HERE), str(ROOT / 'vision' / 'runner')]

import fuel  # noqa: E402
import hub_lights  # noqa: E402
import hub_local_calibration  # noqa: E402
import media  # noqa: E402
import wide_pane  # noqa: E402

LAYOUT = ROOT / 'data' / '2026-rebuilt-welded.json'
TIMES = '20,35,50,70,90,110,130,150,170'   # spread over the match so robots in front of a tag move out of the way


def get_video(args, folder):
    if args.file:
        return Path(args.file)
    return Path(media.download_section(args.youtube, 0, 200, folder, max_height=1080)[0])


def frames_every(video, step_s=2.0, end_s=190.0):
    capture = cv2.VideoCapture(str(video))
    frames, t = [], 0.0
    while t <= end_s:
        capture.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
        t += step_s
    capture.release()
    return frames


def separation(cal, frames):
    """How clearly each HUB's light box splits into lit and unlit across the match (grey levels, red + blue)."""
    rois = hub_lights.light_rois(cal, fuel.hubs_from_field())
    if not rois:
        return 0.0
    total = 0.0
    for side in ('red', 'blue'):
        scores = np.array([hub_lights._light_score(f, rois[side], side) for f in frames])
        low, high = np.percentile(scores, (20, 80))
        total += high - low
    return float(total)


def tune_light_band(cal, frames):
    """Shift the light band up or down to where the lights read most clearly, and return (calibration, gain)."""
    band = cal.get('light_band_z')
    if not band:
        return cal, 0.0
    base = separation(cal, frames)
    best = (base, 0.0)
    for dz in np.arange(-0.3, 0.81, 0.05):
        trial = {**cal, 'light_band_z': [band[0] + dz, band[1] + dz]}
        best = max(best, (separation(trial, frames), float(dz)))
    if best[1]:
        cal = {**cal, 'light_band_z': [round(band[0] + best[1], 3), round(band[1] + best[1], 3)],
               'light_band_note': f'raised {best[1]:+.2f} m so the HUB lights read clearly'}
    return cal, best[0] - base


def preview(cal, video, out):
    """A frame with the field outline and the HUB light boxes drawn on it, to check the calibration by eye."""
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_MSEC, 60000)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        return None
    h, w = frame.shape[:2]
    inverse = np.linalg.inv(np.array(cal['homography_norm'], float))
    def px(x, y):
        p = inverse @ np.array([x, y, 1.0])
        return int(p[0] / p[2] * w), int(p[1] / p[2] * h)
    L, W = fuel.FIELD['length'], fuel.FIELD['width']
    for a, b in [((0, 0), (L, 0)), ((L, 0), (L, W)), ((L, W), (0, W)), ((0, W), (0, 0)), ((L / 2, 0), (L / 2, W))]:
        cv2.line(frame, px(*a), px(*b), (0, 255, 255), 2)
    for (x0, y0, x1, y1) in (hub_lights.light_rois(cal, fuel.hubs_from_field()) or {}).values():
        cv2.rectangle(frame, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)), (255, 0, 255), 2)
    cv2.imwrite(str(out), frame)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument('--youtube', help='YouTube id of a match video from the venue')
    source.add_argument('--file', help='a local match video instead')
    ap.add_argument('--out', default='cal.json', help='calibration file to write (default cal.json)')
    ap.add_argument('--roi', help='x0,y0,x1,y1 of the wide camera as picture shares (found automatically if omitted)')
    ap.add_argument('--times', default=TIMES, help='video seconds to read the tags at')
    args = ap.parse_args(argv)

    with tempfile.TemporaryDirectory() as folder:
        video = get_video(args, folder)
        print(f'video: {video.name}')
        roi = [float(v) for v in args.roi.split(',')] if args.roi else (wide_pane.wide_pane(video) or [0, 0, 1, 1])
        print(f'wide camera area: {[round(v, 3) for v in roi]}')
        out = Path(args.out)
        hub_local_calibration.main(['--video', str(video), '--layout', str(LAYOUT), '--times', args.times,
                                    '--roi', ','.join(str(v) for v in roi), '--out', str(out)])
        cal = json.loads(out.read_text())
        frames = frames_every(video)
        cal, gain = tune_light_band(cal, frames)
        out.write_text(json.dumps(cal, indent=1) + '\n')
        print(f'HUB lights: band {cal["light_band_z"]} m' + (f' (tuned, +{gain:.0f} clearer)' if gain > 0 else ''))
        rois = hub_lights.light_rois(cal, fuel.hubs_from_field())
        series = hub_lights.light_series(video, rois, 0.0, 190.0)
        start, side, evidence = hub_lights.match_start_from_lights(series, search=(-10.0, 60.0))
        if start is None:
            print(f'match start: not found ({evidence.get("reason")}); fuel timing will need --start on each run')
        else:
            print(f'match start: {start:.2f} s into this video (agreement {evidence["agreement"]:.0%})')
        image = preview(cal, video, out.with_suffix('.preview.jpg'))
        print(f'wrote {out}' + (f' and {image} (check the yellow field outline and pink light boxes)' if image else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
