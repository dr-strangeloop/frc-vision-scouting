"""Single-target robot body matching against literal-read reference photos."""
from __future__ import annotations

import json
import math
import re

REFERENCE_IDS = frozenset('ABCDEF')
TARGET_ID = re.compile(r'^[A-Za-z0-9_-]{1,32}$')
REFERENCE_PROMPT = ('REFERENCE ROBOTS. Each letter names one physical robot. The two views with the same letter '
                    'show that same robot. Match chassis, hopper and mechanism structure. Bumper cover color and fuel '
                    'load can change. Printed digits may be unreadable in the TARGET. Never guess by image order or '
                    'alliance color.')
TARGET_PROMPT = ('Which reference robot, if any, is visibly the same physical robot as TARGET? Use UNKNOWN when '
                 'distinctive shared structure is not visible or it matches none. Return only JSON: '
                 '{"reference":"A|B|C|D|E|F|UNKNOWN","confidence":0.0}.')


def match_content(reference_images, reference_ids, target_image):
    """Keep each reference as its own image. Multi-target/reference grids failed validation."""
    if (not reference_ids or len(reference_images) != len(reference_ids) or len(reference_ids) > 6
            or any(ref not in REFERENCE_IDS for ref in reference_ids)
            or any(reference_ids.count(ref) != 2 for ref in set(reference_ids))):
        raise ValueError('two photos of each of at most three reference robots required')
    def fitted(image):
        result = image.convert('RGB').copy()
        result.thumbnail((448, 448))
        return result
    content = [{'type': 'text', 'text': REFERENCE_PROMPT}]
    for image, ref in zip(reference_images, reference_ids):
        content.extend([{'type': 'text', 'text': 'Reference ' + ref + ':'},
                        {'type': 'image', 'image': fitted(image)}])
    content.extend([{'type': 'text', 'text': 'TARGET: this crop is independent. It may show a fourth robot, an obstruction or a person.'},
                    {'type': 'image', 'image': fitted(target_image)},
                    {'type': 'text', 'text': TARGET_PROMPT}])
    return content


def parse_match(text, reference_ids):
    """Reject malformed, duplicate, nonfinite, unsupported and low-confidence answers."""
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate response key')
            result[key] = value
        return result
    def invalid_constant(value):
        raise ValueError(value)
    stripped = re.sub(r'^```(?:json)?\s*|\s*```$', '', str(text).strip(), flags=re.I | re.S)
    try:
        parsed = json.loads(stripped, object_pairs_hook=unique_object, parse_constant=invalid_constant)
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    confidence = parsed.get('confidence')
    if (not isinstance(confidence, (int, float)) or isinstance(confidence, bool)
            or not math.isfinite(confidence) or not .95 <= confidence <= 1):
        return None
    ref = parsed.get('reference')
    return ref if isinstance(ref, str) and ref in set(reference_ids) else None
