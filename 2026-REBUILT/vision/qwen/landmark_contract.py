"""Strict contract for private field-landmark proposals (automatic field calibration on the Spark)."""
from __future__ import annotations

import json
import math
import re

SYSTEM_PROMPT = """You locate known field landmarks on the carpet in one frame
of an FRC match broadcast for a private calibration tool. The top band of the
frame is blacked out; never read or use any scoreboard or overlay text. Only
report a landmark you can see; mark anything hidden or unsure as not visible.
Return only JSON matching the requested shape."""

LANDMARK_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


def _reject_constant(value):
    raise ValueError(f"Non-JSON numeric constant: {value}")


def parse_json_response(text: str):
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(text or "").strip(), flags=re.I | re.S)
    try:
        return json.loads(stripped, parse_constant=_reject_constant), None
    except (json.JSONDecodeError, ValueError) as error:
        return None, f"Invalid model JSON: {error}"


def valid_landmark_ids(ids) -> bool:
    return bool(ids) and all(isinstance(i, str) and LANDMARK_ID.fullmatch(i) for i in ids) and len(set(ids)) == len(ids)


def normalize_landmarks(parsed: object, allowed: set[str]):
    """Keep one finite proposal per requested landmark id. Coordinates are left as returned (the caller rescales
    pixel or 0..1000 answers); anything else is dropped, never guessed."""
    if isinstance(parsed, list):
        parsed = {"landmarks": parsed}
    if not isinstance(parsed, dict) or not isinstance(parsed.get("landmarks"), list):
        return None, "Response lacks a landmarks array"
    out, seen = [], set()
    for item in parsed["landmarks"]:
        if not isinstance(item, dict):
            continue
        lid = str(item.get("id") or "").strip()
        if lid not in allowed or lid in seen:
            continue
        try:
            u, v = float(item.get("u")), float(item.get("v"))
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(n) for n in (u, v, confidence)) or u < 0 or v < 0:
            continue
        seen.add(lid)
        out.append({"id": lid, "u": u, "v": v, "visible": item.get("visible") is not False,
                    "confidence": min(1.0, max(0.0, confidence))})
    return out, None
