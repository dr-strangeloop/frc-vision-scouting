"""One versioning scheme for vision scouting.

- The **release version** (`vision/VERSION`, `MAJOR.MINOR.PATCH`) names the vision code. It is the only version a run,
  report or scorecard is filed under.
- The **counter version** (v1, v1.0.1 ... v1.0.4b, v2, v2.1.0) names the ball-counting method inside a release, so
  fuel results can be compared across counters. v2.1.0 specifically means the chimney-scale-3 model ran at 1080p;
  a lower-resolution run or model fallback remains v2. It is recorded next to the release version, never instead of it.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def app_version():
    """The release version in vision/VERSION, or 'unknown' outside a checkout."""
    try:
        return (REPO_ROOT / 'vision' / 'VERSION').read_text(encoding='utf-8').strip() or 'unknown'
    except OSError:
        return 'unknown'


def stamp(counter=None):
    """{'version': release, 'counter': counter version} for a run's stats or a measurements record."""
    return {'version': app_version(), 'counter': counter}
