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

## Quick start

```bash
git clone https://github.com/dr-strangeloop/frc-vision-scouting && cd frc-vision-scouting
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # add requirements-local.txt for your own detector
cp .env.example .env                     # fill in GEMINI_API_KEY and/or TBA_API_KEY
python3 vision.py check                  # tells you what is missing
```

Or with Docker: `docker build -t frc-vision . && docker run --rm frc-vision check`.

You also need `ffmpeg` installed.

### 1. Calibrate the venue (once per camera position)

```bash
python3 vision.py calibrate --youtube <match video id> --out cal.json
```

It finds the broadcast's wide camera, calibrates from the HUB AprilTags (WPILib's layout is in `data/`), tunes the HUB
light boxes and checks it can find the match start. Open `cal.preview.jpg`: the yellow outline should sit on the field
and the pink boxes on the HUB lights. If the tags can't be read, mark landmarks by hand with
`tools/calibration_marker.html` and build the file with `scripts/vision/build_calibration.py`.

### 2. Choose how robots are detected

- **Gemini** (`--engine gemini`): no GPU or detector needed. Set `GEMINI_API_KEY`; each run is billed by Google, capped
  by `VISION_MAX_USD`.
- **Your own YOLO detector** (`--engine local --weights your.pt`): train an Ultralytics YOLO model on broadcast frames
  with two classes, `robot_red` and `robot_blue`. Runs free on your own GPU (or slowly on a CPU). Bumper reads use a
  local vision-language model at `LOCAL_VLM_URL` if you have one.

### 3. Run a match

```bash
python3 vision.py run --match 2026cc_qm1 --youtube <video id> --calibration cal.json --engine gemini --out qm1.json
python3 vision.py run --match 2026cc_qm1 --youtube <video id> --calibration cal.json --engine local --weights your.pt --auto-only --out qm1-auto.json
```

Rosters come from The Blue Alliance (`TBA_API_KEY`) or `--roster-json '{"red":[...],"blue":[...]}'`. The result JSON
has each robot's trajectory (field metres, match milliseconds), identities with confidence, and FUEL counts. Count
FUEL on its own with `python3 vision.py fuel --help`.

## Notes

- Positions are only as good as the calibration; robots behind HUBs or near the near wall are often unseen, and
  unseen time is left empty, not filled in.
- Broadcast layouts differ by event (split screens, close-up panes); calibrate each venue's wide view, and expect to
  adjust region-of-interest settings for a new broadcast.

## Changelog

Every release, newest first: [CHANGELOG.md](CHANGELOG.md). This code is release 1.21.1 (`vision/VERSION`).

## License

MIT. See [LICENSE](LICENSE).
