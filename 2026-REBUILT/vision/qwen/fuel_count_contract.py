"""Compact, provisional alliance fuel count from a short wide-camera clip."""
from __future__ import annotations

from qwen_contract import parse_json_response


FUEL_COUNT_PROMPT = """Review these timestamped frames from the main wide field camera.
For each alliance, estimate the number of distinct yellow FUEL balls visibly entering
its HUB between the first and last frame. Count an entry once, at the moment it
crosses the HUB opening. A shot, loose ball, or ball already inside is not an
entry. Use only the images; no official score, human tally, or scoreboard is
available. The frames are sampled, so admit uncertainty. If a HUB entrance is
not visible enough to estimate, use null for that alliance; do not use zero
for missing evidence. Return only compact JSON, with no events or explanation:
{"red":0,"blue":0,"red_visible":true,"blue_visible":true,
"quality":"good|limited|unusable"}
"""


def normalize_fuel_count(raw: str):
    parsed, error = parse_json_response(raw)
    if error:
        return None, error
    if not isinstance(parsed, dict):
        return None, 'Expected an object'
    result = {}
    for side in ('red', 'blue'):
        visible = parsed.get(f'{side}_visible')
        count = parsed.get(side)
        if not isinstance(visible, bool):
            return None, f'{side}_visible must be a boolean'
        if visible:
            if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= 200:
                return None, f'{side} must be an integer from 0 to 200 when visible'
            result[side] = count
        else:
            if count is not None:
                return None, f'{side} must be null when not visible'
            result[side] = None
        result[f'{side}_visible'] = visible
    quality = parsed.get('quality')
    if quality not in ('good', 'limited', 'unusable'):
        return None, 'quality must be good, limited, or unusable'
    result['quality'] = quality
    return result, None
