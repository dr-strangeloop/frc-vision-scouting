# FRC vision scouting (2026 REBUILT)

Turns a match video into per-robot field paths and FUEL counts for the 2026 FRC game, REBUILT. Built by FRC Team 971
for its own scouting and shared so other teams can use, check and improve it.

What it does, for one match video:

- **Robot tracks.** Detects robots each frame, projects them onto the field through a camera calibration, links them
  into tracklets and smooths each robot's path. Alliances come from bumper colour.
- **Robot identity.** Reads bumper numbers with a vision-language model (a local Qwen service or Gemini) and solves which
  tracklet is which team. Ambiguous stretches stay unnamed rather than guessed.
- **Match start and HUB schedule.** Finds the match start and each shift's active HUB from the physical HUB lights,
  not the broadcast scoreboard.
- **FUEL.** Tracks balls, counts HUB entries and attributes observed shots to the robot a ball left.

Everything is evidence for a human to review, not an official result. It never reads scores, the match clock or
scouting data as input.

## Layout

| Path | What |
| --- | --- |
| `scripts/vision/run_match.py` | Run one match end to end; writes JSON to `--out` |
| `scripts/vision/fuel_count.py` | FUEL counter on its own |
| `scripts/vision/hub_lights.py` | Match start and active HUB from the HUB lights |
| `scripts/vision/pipeline.py` | Frames to per-robot field trajectories |
| `vision/runner/full_match_robots.py` | Robot boxes and identities across a full match |
| `vision/qwen/` | Optional local Qwen vision-language service for bumper reads |
| `src/lib/vision/data/rebuiltField.json` | 2026 field geometry (from the Game Manual and the WPILib AprilTag layout) |

## Setup

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

You also need `ffmpeg`, a **YOLO detector** trained to find robots (classes `robot_red` and `robot_blue`; no weights are
included), and a **camera calibration** for the venue's broadcast view: a JSON file with `homography_norm` (normalized
picture coordinates to field metres) and, for the HUB lights, a height projection.

## Run a match

```bash
# local GPU detector and a local OpenAI-compatible VLM (LOCAL_VLM_URL)
python3 scripts/vision/run_match.py --match 2026cc_qm1 --youtube <video id> --engine local \
    --weights best.pt --calibration cal.json --out qm1.json

# autonomous period only
python3 scripts/vision/run_match.py --match 2026cc_qm1 --youtube <video id> --engine local \
    --weights best.pt --calibration cal.json --auto-only --out qm1-auto.json

# no GPU: Gemini for detection and reads (GEMINI_API_KEY)
python3 scripts/vision/run_match.py --match 2026cc_qm1 --youtube <video id> --engine gemini \
    --calibration cal.json --out qm1.json
```

The roster comes from The Blue Alliance (`TBA_API_KEY`) or `--roster-json`. Run `--help` on each script for its options.

## Notes

- Positions are only as good as the calibration; robots behind HUBs or near the near wall are often unseen, and
  unseen time is left empty, not filled in.
- Broadcast layouts differ by event (split screens, close-up panes); calibrate each venue's wide view, and expect to
  adjust region-of-interest settings for a new broadcast.

## Changelog

Every release, newest first: [CHANGELOG.md](CHANGELOG.md). This code is release 1.21.1 (`vision/VERSION`).

## License

MIT. See [LICENSE](LICENSE).
