"""Pure parsing and validation for Qwen Vision responses."""
from __future__ import annotations

import json
import math
import re

ALLOWED_EVENT_TYPES = {
    "robot", "fuel_scored", "climb_attempt", "climb_success",
    "disabled_or_immobile",
}
ALLOWED_ALLIANCES = {"red", "blue", "unknown"}
# Must stay in step with VALID_CLIMB_POS in src/routes/api/vision/+server.js.
# A climb_level outside this set is dropped back to null here rather than
# carried all the way to release-run, which would silently discard it.
ALLOWED_CLIMB_LEVELS = {"N/A", "Failed", "L1", "L2", "L3"}

SYSTEM_PROMPT = """You assist human reviewers with FRC match-video analysis.
Never invent an event hidden by blur or occlusion. Return only valid JSON.
Coordinates use [x1,y1,x2,y2] normalized to 0..1000. Model output is always
provisional and must be reviewed by a human."""

TASK_PROMPT = """Analyze this chronological sequence of timestamped frames.
The detector already handles robot presence; omit robot-presence events.
Report at most three clearly supported action events per clip, with at most
twelve words per evidence string and review_notes. This is sparse evidence
review, not an exhaustive fuel count. Never infer from scoreboard numbers.
Report only clearly visible evidence for fuel entering a scoring target,
climb attempts, completed climbs, and disabled or immobile robots. Do not infer a
score merely because a robot shoots. Do not count one event twice across adjacent
frames. Use the absolute timestamp_ms printed before each frame.

For a climb event, set climb_level to exactly one of "L1", "L2", "L3",
"Failed" or "N/A" - never any other wording, and null if the level is not
clearly visible. Any other value is discarded.

Return exactly this JSON shape:
{"events":[{"type":"robot|fuel_scored|climb_attempt|climb_success|disabled_or_immobile",
"timestamp_ms":0,"alliance":"red|blue|unknown","box":[0,0,0,0],
"confidence":0.0,"evidence":"brief visible reason",
"climb_level":"L1|L2|L3|Failed|N/A|null"}],
"clip_quality":"good|limited|unusable","review_notes":"brief text"}"""


def _reject_constant(value):
    raise ValueError(f"Non-JSON numeric constant: {value}")


def parse_json_response(text: str):
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I | re.S)
    try:
        return json.loads(stripped, parse_constant=_reject_constant), None
    except (json.JSONDecodeError, ValueError) as error:
        return None, f"Invalid model JSON: {error}"


def normalize_result(parsed, clip_start_ms: int, clip_end_ms: int):
    # Complete root arrays are a real model response variant. They still go
    # through every event validator; absent quality remains limited evidence.
    if isinstance(parsed, list):
        parsed = {"events": parsed, "clip_quality": "limited"}
    if not isinstance(parsed, dict) or not isinstance(parsed.get("events"), list):
        return None, "Response lacks an events array"
    cleaned = []
    for event in parsed["events"]:
        if not isinstance(event, dict) or event.get("type") not in ALLOWED_EVENT_TYPES:
            continue
        try:
            timestamp = int(event.get("timestamp_ms"))
            confidence = float(event.get("confidence", 0))
            if not math.isfinite(confidence):
                continue
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            continue
        box = event.get("box")
        if not isinstance(box, list) or len(box) != 4:
            box = None
        else:
            try:
                box = [max(0, min(1000, round(float(value)))) for value in box]
            except (TypeError, ValueError):
                box = None
        cleaned.append({
            "type": event["type"],
            "timestamp_ms": max(clip_start_ms, min(clip_end_ms, timestamp)),
            "alliance": event.get("alliance") if event.get("alliance") in ALLOWED_ALLIANCES else "unknown",
            "box_0_1000": box,
            "confidence": confidence,
            "evidence": str(event.get("evidence", ""))[:500],
            "climb_level": str(event["climb_level"]) if str(event.get("climb_level")) in ALLOWED_CLIMB_LEVELS else None,
            "review_status": "unreviewed",
        })
    quality = parsed.get("clip_quality")
    return {
        "events": cleaned,
        "clip_quality": quality if quality in {"good", "limited", "unusable"} else "limited",
        "review_notes": str(parsed.get("review_notes", ""))[:1000],
    }, None
