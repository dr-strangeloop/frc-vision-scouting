"""Strict contract for private, roster-constrained team-number reads."""
from __future__ import annotations

import json
import math
import re


TEAM_KEY = re.compile(r"^frc(\d{1,5})$", re.I)

SYSTEM_PROMPT = """You read FRC team numbers from robot crops for a private
scouting system. A crop can be blurry, obstructed, or show a referee. Never
guess a number. Return only JSON matching the requested shape."""

TASK_PROMPT = """Read the complete team number literally printed on the bumper
in each image. You are not given a roster: never choose a plausible team number
or infer identity from appearance. If digits are hidden, blurred, or incomplete,
return null. If multiple different team numbers appear, return null rather than
choosing one. Return exactly:
{"reads":[{"crop_id":"input crop id","team_key":"frc####|null",
"confidence":0.0,"evidence":"brief visible text or why unreadable"}]}.
Use confidence 0 when the number is unreadable; do not infer from alliance,
robot appearance, or another image."""


def normalize_team_key(value: object) -> str | None:
    text = str(value or "").strip().lower()
    if text.isdigit():
        text = f"frc{text}"
    match = TEAM_KEY.fullmatch(text)
    return f"frc{match.group(1)}" if match else None


def _reject_constant(value):
    raise ValueError(f"Non-JSON numeric constant: {value}")


def parse_json_response(text: str):
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(text or "").strip(), flags=re.I | re.S)
    try:
        return json.loads(stripped, parse_constant=_reject_constant), None
    except (json.JSONDecodeError, ValueError) as error:
        return None, f"Invalid model JSON: {error}"


def normalize_reads(parsed: object, permitted: dict[str, set[str]]):
    if isinstance(parsed, list):
        parsed = {"reads": parsed}
    if not isinstance(parsed, dict) or not isinstance(parsed.get("reads"), list):
        return None, "Response lacks a reads array"
    output = []
    seen = set()
    for item in parsed["reads"]:
        if not isinstance(item, dict):
            continue
        crop_id = str(item.get("crop_id") or "")
        if crop_id not in permitted or crop_id in seen:
            continue
        team_key = normalize_team_key(item.get("team_key"))
        if team_key not in permitted[crop_id]:
            team_key = None
        try:
            confidence = float(item.get("confidence", 0))
            confidence = max(0.0, min(1.0, confidence)) if math.isfinite(confidence) else 0.0
        except (TypeError, ValueError):
            confidence = 0.0
        if team_key is None:
            confidence = 0.0
        output.append({
            "crop_id": crop_id,
            "team_key": team_key,
            "confidence": confidence,
            "evidence": str(item.get("evidence") or "")[:300],
        })
        seen.add(crop_id)
    return output, None


# Compact reads (/read-bumper-digits). Same rules as TASK_PROMPT, but the answer is only the digits per crop: on the
# Spark the dense 27B model generates about 5 tokens a second, so a per-crop evidence sentence, not the image, is what
# made a read take 10-14 s. The roster is still never shown to the model and is applied afterwards.
COMPACT_TASK_PROMPT = """For each crop id above, read the complete team number
literally printed on the robot bumper. Never guess: if the digits are hidden,
blurred, cut off, or more than one different number is visible, use null. Do
not infer from color, appearance, or another crop. Return only one JSON object
mapping every crop id to its digits or null, for example {"c0":"254","c1":null}."""


def normalize_compact_reads(parsed: object, permitted: dict[str, set[str]]):
    """[{'crop_id', 'team_key', 'confidence'}] for every permitted crop id. A literal read on its crop's roster gets
    confidence 0.8 (the model reports none); anything else, including a missing id, is null with confidence 0."""
    if not isinstance(parsed, dict):
        return None, "Response is not a JSON object of crop ids"
    output = []
    for crop_id, allowed in permitted.items():
        value = parsed.get(crop_id)
        team_key = normalize_team_key(value) if isinstance(value, (str, int)) and not isinstance(value, bool) else None
        if team_key not in allowed:
            team_key = None
        output.append({"crop_id": crop_id, "team_key": team_key, "confidence": 0.8 if team_key else 0.0,
                       "evidence": "compact read" if team_key else ""})
    return output, None


GRID_TASK_PROMPT = """The image is a grid of robot crops. Each cell has a letter
label in a black box at its top-left corner; the label is not a team number.
For each cell letter listed above, read the complete team number literally
printed on that cell's robot bumper. Never guess: if the digits are hidden,
blurred, cut off, or more than one different number is visible in the cell, use
null. Read each cell on its own; never copy a number from a neighbouring cell.
Answer on one line only: each listed letter in order, immediately followed by
its digits, or by - when unreadable, separated by single spaces. Example for
three cells: A254 B- C1678"""

GRID_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"   # no I or O, which read like 1 and 0


GRID_CELL = re.compile(r"\b([A-Z])(\d{1,5}|-)(?=\s|$)")


def parse_grid_line(text: str, letters: str):
    """{letter: digits or None} from a grid answer such as 'A254 B- C1678'. Only listed letters count; a letter given
    twice with different answers is unreadable. A response with none of the letters is an error."""
    found = {}
    for letter, value in GRID_CELL.findall(str(text or "").replace("`", " ")):
        if letter not in letters:
            continue
        value = None if value == "-" else value
        if letter in found and found[letter] != value:
            found[letter] = None
        else:
            found[letter] = value
    if not found:
        return None, "Response lists none of the grid cells"
    return found, None
