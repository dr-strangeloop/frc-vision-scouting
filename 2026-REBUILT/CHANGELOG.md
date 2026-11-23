# Changelog

Each release of the vision pipeline, newest first. Versions follow `MAJOR.MINOR.PATCH`: PATCH is a fix, MINOR is a new
feature or noticeably different results, MAJOR is a breaking change. The current version is in `vision/VERSION`.

This repository is the public release of **1.21.1**. Earlier versions were developed privately; the entries below
record what each one changed in the pipeline. Work began on 2026-08-28, before releases had version numbers; that
period is summarized at the end.

## 1.21.1 — 2026-11-01

- The wide-camera finder picks a split-screen broadcast's thin pane seam instead of a nearby guardrail.

## 1.21.0 — 2026-11-01

- Body recognition turns off only for the alliance with a team missing from the gallery, not the whole match.

## 1.20.0 — 2026-11-01

- Legible close-up bumper reads seed the event gallery before body recognition runs, so a new camera can use
  recognition from its first match.

## 1.19.0 — 2026-11-01

- With no saved calibration area, the wide camera of a split-screen broadcast is found from its fixed pane seam.

## 1.18.0 — 2026-11-01

- The event gallery grows itself after each scouted match (`grow_gallery`), with no manual step.

## 1.17.1 — 2026-11-01

- Slow automatic field calibrations are no longer cut off as stalled.

## 1.17.0 — 2026-10-31

- Literal bumper reads from earlier runs of the same tracklets are added, since which tracklets earn a name varies
  between runs.

## 1.16.1 — 2026-10-31

- Close-up panels are read every 0.5 s instead of every 1.5 s, within the same crop budget.

## 1.16.0 — 2026-10-31

- Shooter inference gains two rules, a dominant launcher and shot bursts, which name more HUB entries.

## 1.15.0 — 2026-10-31

- On split-screen broadcasts, each close-up panel is read and its name carried to the wide-view robot at the same floor
  point (`closeup_names.py`).

## 1.14.4 — 2026-10-31

- Robot scouting stays on the wide camera instead of the whole frame when no calibration exists yet.

## 1.14.3 — 2026-10-31

- Match video downloads prefer H.264, which OpenCV decodes everywhere.

## 1.14.2 — 2026-10-30

- Identity refinement withholds only the fragment a disputed credit came from, not every name in the match.

## 1.14.1 — 2026-10-30

- Robot credits use the alliance's calibration.

## 1.14.0 — 2026-10-30

- Shot attribution from ball flight, robot appearance and elimination by alliance partners.

## 1.13.2 — 2026-10-30

- The event gallery covers the whole roster, and the FUEL pass has a time limit.

## 1.13.1 — 2026-10-30

- A match start of zero from scouting is treated as unknown, so the HUB-lights search still runs.

## 1.13.0 — 2026-10-30

- Bumpers are read every 0.5 s along unnamed ball-exit sources, near each observed exit.

## 1.12.0 — 2026-10-29

- `robot_launches.py`: counts balls seen leaving a robot's body, with no fitted flight or HUB hit required.

## 1.11.0 — 2026-10-27

- Identity follow-up counts distinct source views, and its extra reads are bounded.

## 1.10.1 — 2026-10-26

- Identity refinement runs from the FUEL counter as well as robot scouting.

## 1.10.0 — 2026-10-26

- Source-first identity refinement after counting: proven ball sources with no name get targeted bumper reads.

## 1.9.5 — 2026-10-26

- Ball tracks keep the intervals the tracker joined (re-links, de-duplicated overlaps) for audits.

## 1.9.4 — 2026-10-26

- A shot's source needs a measured robot box at the launch time, or two boxes close on either side of it.

## 1.9.3 — 2026-10-26

- Entries keep their first observed picture position and their rise in camera-registered coordinates.

## 1.9.2 — 2026-10-26

- Robot-source boxes are limited to the counted recording and keep their own time origin.

## 1.9.1 — 2026-10-26

- The unique-origin check considers every measured robot box, opponents and unnamed robots included.

## 1.9.0 — 2026-10-26

- Robot FUEL keeps its source evidence (track IDs, identity evidence, coverage, counts and timelines), and 1080p
  attribution geometry is fixed.

## 1.8.1 — 2026-10-25

- Falls back to the system `yt-dlp` when a managed copy can't be read.

## 1.8.0 — 2026-10-25

- Robot FUEL is credited only to identities from unambiguous, roster-constrained bumper reads.

## 1.7.0 — 2026-10-26

- The FUEL counter reports how much of the match each robot was tracked.

## 1.6.3 — 2026-10-26

- Robot identity reads all its crops in one pass (faster, same results).

## 1.6.2 — 2026-10-26

- A match can be counted from its other linked video when the first is gone.

## 1.6.1 — 2026-10-26

- An interrupted run can be started again.

## 1.6.0 — 2026-10-26

- Every robot is named from bumper reads across the whole match (`full_match_robots.py`).

## 1.5.1 — 2026-10-25

- Robot trajectory points carry their normalized picture box, so the FUEL counter can match balls to robots.

## 1.5.0 — 2026-10-25

- Automatic field calibration from landmarks proposed by a local vision-language model, fitted and gated before use.

## 1.4.1 — 2026-10-25

- Each credited robot keeps its timeline in the FUEL-count snapshot.

## 1.3.2 — 2026-10-23

- Restores an older command-line entry point.

## 1.3.1 — 2026-10-23

- Configuration fixes; counting is unchanged.

## 1.3.0 — 2026-10

- The height-projected HUB openings, stricter entry test and HUB-light shift schedule are used in every FUEL path.
- Optional counting rules (balls vanishing at the HUB, splitting touching balls, motion threshold), off by default.
- Calibrations carried to other matches keep their height projection.

## 1.2.0 — 2026-10

- Every accuracy figure comes from a recorded run; nothing is typed in, and unmeasured metrics read as unknown.

## 1.1.0 — 2026-10

- Robot identity from close-up and zoom-panel bumper reads, a per-event appearance classifier, names reused across runs
  of a match, and stitching and swap fixes.
- Auto accuracy: calibration quality, floor-contact projection, and match start from robot motion.
- The broadcast match clock is no longer read.

## 1.0.0 — 2026-10-17

- First versioned release: robot tracking and identity, and FUEL counting.

## Before versioning (2026-08-28 to 2026-10-16)

### October 2026 (to the 16th)

- Gemini path: robot boxes from still frames and robot actions from video, drawn over the match video.
- Automatic robot detection with a region of interest, rejection of balls on the ground, and a ballistic FUEL
  tracker that fits 3D flights.
- Pipeline review: faster camera registration, a batched robust ballistic fit, an exact identity solve, and fixes for
  counting a ball twice.
- Conflicting bumper identities are rejected, appearance matching keeps brightness, and autonomous motion is kept
  separate from the scoring grace period.
- Automatic ground-plane field calibration, once per event; video and model-answer caching, and parallel downloads.
- Reading the broadcast scoreboard was tried briefly and later removed (see 1.1.0).

### September 2026

- The local vision-language model moved to a dense 27B Qwen model on a local GPU.
- FUEL attribution fixed and camera tracking isolated per view.
- A private, team-labelled training set for the robot detector.
- Robot motion events and reviewed suggestions for scouts, plus a review and evaluation workflow.

### August 2026 (from the 28th)

- First pipeline: robot detection and re-identification, hybrid FUEL detection, and scoring attribution, with a local
  vision-language model reviewing short clips.
- Field calibration from the field's AprilTags, with gating, a recording preflight and a configurable tag family.
- Robot identities constrained to the match roster.
