#!/usr/bin/env python3
"""FRC vision scouting: one entry point.

    python3 vision.py check                                   # is everything installed and configured?
    python3 vision.py calibrate --youtube <id> --out cal.json  # once per venue camera
    python3 vision.py run --match 2026cc_qm1 --youtube <id> --calibration cal.json --out qm1.json
    python3 vision.py fuel --youtube <id> --start 12.6 --calibration cal.json --out count.json

Add --help after any command for its options.
"""
import importlib
import importlib.util
import os
import platform
import runpy
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMMANDS = {
    'calibrate': 'scripts/vision/calibrate.py',
    'run': 'scripts/vision/run_match.py',
    'fuel': 'scripts/vision/fuel_count.py',
}


def env(name):
    """A setting from the environment, else from .env next to this file."""
    if os.environ.get(name):
        return os.environ[name]
    path = ROOT / '.env'
    if path.is_file():
        for line in path.read_text().splitlines():
            key, _, value = line.partition('=')
            if key.strip() == name and value.strip():
                return value.strip().strip('"').strip("'")
    return ''


def check():
    """Report what works on this machine and what each missing piece would enable."""
    ok = True
    def line(good, label, detail=''):
        nonlocal ok
        shown = detail if (detail and (not good or not detail.startswith(('pip ', 'install ', 'set ')))) else ''
        print(f"  [{'ok' if good else '--'}] {label}{(': ' + shown) if shown else ''}")
        return good

    print('Required')
    ok &= line(sys.version_info >= (3, 10), 'Python 3.10 or newer', platform.python_version())
    for module, package in (('numpy', 'numpy'), ('scipy', 'scipy'), ('requests', 'requests'), ('cv2', 'opencv-python-headless')):
        try:
            importlib.import_module(module)
            ok &= line(True, package)
        except ImportError:
            ok &= line(False, package, 'pip install -r requirements.txt')
    try:
        import cv2
        ok &= line(hasattr(cv2, 'aruco'), 'OpenCV AprilTag detector (for calibrate)', cv2.__version__)
    except ImportError:
        pass
    ok &= line(bool(shutil.which('ffmpeg')), 'ffmpeg', 'install it with your package manager (brew, apt, winget)')
    ytdlp = shutil.which('yt-dlp') or importlib.util.find_spec('yt_dlp')
    ok &= line(bool(ytdlp), 'yt-dlp (downloads match videos)', 'pip install -r requirements.txt')
    line((ROOT / 'data' / '2026-rebuilt-welded.json').is_file(), 'WPILib 2026 AprilTag layout (data/)')

    print('\nRobot detection: pick one')
    gemini = line(bool(env('GEMINI_API_KEY')), 'Gemini (--engine gemini)', 'set GEMINI_API_KEY in .env' if not env('GEMINI_API_KEY') else 'key set')
    try:
        importlib.import_module('ultralytics')
        has_yolo = True
    except ImportError:
        has_yolo = False
    local = line(has_yolo, 'Your own YOLO detector (--engine local --weights your.pt)',
                 'ultralytics installed' if has_yolo else 'pip install ultralytics, then train a detector (see README)')
    if has_yolo:
        try:
            import torch
            line(torch.cuda.is_available(), 'GPU for the local detector', 'CUDA available' if torch.cuda.is_available() else 'none found: it runs, slowly, on the CPU')
        except ImportError:
            pass
    if not (gemini or local):
        ok = False
        print('  Neither is set up: robot tracking needs one of them.')

    print('\nOptional')
    line(bool(env('TBA_API_KEY')), 'The Blue Alliance key (match rosters)', 'or pass --roster-json on each run')
    line(bool(env('LOCAL_VLM_URL')), 'Local vision-language model for bumper reads (LOCAL_VLM_URL)',
         'Gemini mode does not need it' if not env('LOCAL_VLM_URL') else env('LOCAL_VLM_URL'))
    print('\n' + ('Ready.' if ok else 'Fix the [--] items marked required above.'))
    return 0 if ok else 1


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help', 'help'):
        print(__doc__)
        return 0
    command, rest = sys.argv[1], sys.argv[2:]
    if command == 'check':
        return check()
    if command not in COMMANDS:
        print(f'unknown command {command!r}\n{__doc__}')
        return 2
    script = ROOT / COMMANDS[command]
    sys.argv = [str(script), *rest]
    runpy.run_path(str(script), run_name='__main__')
    return 0


if __name__ == '__main__':
    sys.exit(main())
