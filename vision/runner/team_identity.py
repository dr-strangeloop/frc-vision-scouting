"""Private, confidence-gated FRC robot identity from bumper-number crops.

This deliberately never treats alliance colour as identity.  It sends only a
small number of in-memory robot crops to the loopback Qwen service and accepts
a team only after repeated roster-constrained reads.
"""
from __future__ import annotations

import base64
import hashlib
import math
from collections import Counter, defaultdict


def team_key(value):
    text = str(value or "").strip().lower()
    if text.isdigit():
        text = f"frc{text}"
    return text if text.startswith("frc") and text[3:].isdigit() else None


def roster_for(team_roster, alliance):
    values = (team_roster or {}).get(alliance, []) if alliance in {"red", "blue"} else []
    return {key for value in values if (key := team_key(value))}


def bumper_occluded(box, others=(), minimum_overlap=.35):
    l, t, r, b = box
    w, h = r - l, b - t
    band = (l + .05 * w, t + .42 * h, r - .05 * w, b)
    area = (band[2] - band[0]) * (band[3] - band[1])
    if area <= 0:
        return True
    return any(
        max(0, min(band[2], q[2]) - max(band[0], q[0]))
        * max(0, min(band[3], q[3]) - max(band[1], q[1]))
        / area >= minimum_overlap
        for q in (others or ())
        if q is not None and len(q) == 4
    )


def bumper_contrast_bonus(crop):
    import cv2
    if crop is None or not getattr(crop, "size", 0) or getattr(crop, "ndim", 0) != 3:
        return 0.0
    band = crop[round(crop.shape[0] * .42):round(crop.shape[0] * .91)]
    if band.size == 0:
        return 0.0
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    white = (hsv[:, :, 1] <= 100) & (hsv[:, :, 2] >= 145)
    white_share = float(white.mean())
    if not .01 <= white_share <= .35:
        return 0.0
    bumper_detail = float(cv2.Laplacian((white.astype("uint8") * 255), cv2.CV_64F).var())
    return min(1.0, bumper_detail / 300) * .4


def crop_robot(frame, coords, other_coords=()):
    """Return an enlarged robot crop and quality score, or None if too small or occluded."""
    import cv2  # runtime dependency of the runner; keep consensus tests pure
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in coords]
    robot_width, robot_height = x2 - x1, y2 - y1
    if robot_width < 42 or robot_height < 42:
        return None
    if other_coords and bumper_occluded((x1, y1, x2, y2), other_coords):
        return None
    # Bumper numbers can sit at either edge. Preserve the whole detection plus
    # a small margin rather than assuming a fixed bumper orientation.
    x1 = max(0, int(x1 - robot_width * .12)); x2 = min(width, int(x2 + robot_width * .12))
    y1 = max(0, int(y1 - robot_height * .12)); y2 = min(height, int(y2 + robot_height * .12))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    sharpness = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    bumper_bonus = bumper_contrast_bonus(crop)
    scale = min(4.0, max(1.0, 720 / max(crop.shape[:2])))
    if scale > 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    encoded, jpeg = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 94])
    if not encoded:
        return None
    # Area rejects distant robots; sharpness and white-digit bumper contrast
    # break ties between similarly sized candidates. The number is only used
    # for sampling, never trust.
    score = min(1.0, (robot_width * robot_height) / 50_000) + min(1.0, sharpness / 300) * .2 + bumper_bonus
    return jpeg.tobytes(), score


class CropCollector:
    def __init__(self, max_per_track=3, min_frame_gap=30, min_time_gap_ms=1000):
        self.max_per_track = max_per_track
        self.min_frame_gap = max(1, min_frame_gap)
        self.min_time_gap_ms = max(0, min_time_gap_ms)
        self._last_probe = {}
        self.crops = defaultdict(list)

    def offer(self, track_key, alliance, frame_index, timestamp_ms, frame, coords, other_coords=()):
        if other_coords and coords is not None and bumper_occluded(coords, other_coords):
            return
        existing = self.crops[track_key]
        # Probe quality at bounded intervals; a poor first frame should not
        # prevent a clearer frame in the same sampling window from replacing it.
        last_probe = self._last_probe.get(track_key)
        if last_probe is not None and frame_index - last_probe < max(1, self.min_frame_gap // 5):
            return
        self._last_probe[track_key] = frame_index
        nearby = [row for row in existing
                  if abs(frame_index - row['frame_index']) < self.min_frame_gap
                  or abs(timestamp_ms - row['timestamp_ms']) < self.min_time_gap_ms]
        if len(nearby) > 1:
            return
        result = crop_robot(frame, coords)
        if not result:
            return
        jpeg, score = result
        if nearby:
            if score <= nearby[0]['score']:
                return
            existing.remove(nearby[0])
        existing.append({"track_key": track_key, "alliance": alliance, "frame_index": frame_index,
                         "timestamp_ms": timestamp_ms, "jpeg": jpeg, "score": score})
        existing.sort(key=lambda row: row["score"], reverse=True)
        del existing[self.max_per_track:]

    def flattened(self):
        return [crop for values in self.crops.values() for crop in values]


def combine_reader_votes(reads, min_confidence=.65):
    """Disagreement between readers on one source frame invalidates that vote."""
    groups = {}
    for read in reads or ():
        if not isinstance(read, dict):
            continue
        groups.setdefault((read.get('track_key'), read.get('frame_index')), []).append(read)
    result = []
    for values in groups.values():
        valid = []
        for r in values:
            raw_conf = r.get('confidence')
            if isinstance(raw_conf, bool) or not isinstance(raw_conf, (int, float)):
                continue
            conf = float(raw_conf)
            team = team_key(r.get('team_key'))
            if team and math.isfinite(conf) and min_confidence <= conf <= 1:
                valid.append({**r, 'team_key': team, 'confidence': conf})
        if len({r['team_key'] for r in valid}) == 1:
            result.append(max(valid, key=lambda r: r['confidence']))
    return result


def decide_identity(reads, min_reads=3, min_confidence=.65, min_share=.70):
    """Return a confidence-gated identity decision from OCR reads for one track."""
    reads = combine_reader_votes(reads, min_confidence=min_confidence)
    valid = []
    seen_frames, seen_images = set(), set()
    for row in reads:
        if not isinstance(row, dict):
            continue
        try:
            confidence = float(row.get('confidence', 0))
        except (TypeError, ValueError):
            continue
        frame = row.get('frame_index')
        digest = row.get('crop_sha256')
        team = team_key(row.get('team_key'))
        if (not team or isinstance(row.get('confidence'), bool) or not math.isfinite(confidence) or not min_confidence <= confidence <= 1
                or isinstance(frame, bool) or not isinstance(frame, int) or frame < 0
                or not isinstance(digest, str) or len(digest) != 64
                or any(c not in '0123456789abcdef' for c in digest)
                or frame in seen_frames or digest in seen_images):
            continue
        seen_frames.add(frame); seen_images.add(digest)
        valid.append({**row, 'team_key': team, 'confidence': confidence})
    if not valid:
        return None, {"read_count": 0, "reason": "no independent high-confidence bumper-number reads"}
    weighted = Counter()
    counts = Counter()
    for row in valid:
        weighted[row["team_key"]] += float(row["confidence"])
        counts[row["team_key"]] += 1
    team, weight = weighted.most_common(1)[0]
    share = weight / sum(weighted.values())
    if counts[team] < min_reads:
        return None, {"read_count": counts[team], "candidate": team, "share": share, "reason": "not enough repeated reads"}
    if share < min_share:
        return None, {"read_count": counts[team], "candidate": team, "share": share, "reason": "conflicting reads"}
    return team, {"read_count": counts[team], "candidate": team, "share": share,
                  "confidence": min(1.0, share * (weight / counts[team]))}


def resolve_identities(tracks, collector, team_roster, qwen_url, qwen_token, *, timeout=180,
                       min_reads=3, min_confidence=.65, min_share=.70, batch_size=2):
    """Ask local Qwen to read crops and apply only consensus assignments.

    This function is safe to call on every completed view: missing roster,
    disabled endpoint, malformed replies, or an OCR outage leave tracks
    unidentified and reviewable.
    """
    import requests  # runner dependency; keep decision-rule tests dependency-free
    by_key = {track["track_key"]: track for track in tracks}
    crops = [row for row in collector.flattened() if roster_for(team_roster, row["alliance"])]
    evidence = defaultdict(list)
    batch_size = max(1, min(8, int(batch_size)))
    for offset in range(0, len(crops), batch_size):
        batch = crops[offset:offset + batch_size]
        payload_crops = []
        permitted = {}
        for index, crop in enumerate(batch):
            crop_id = f"{crop['track_key']}:{crop['frame_index']}:{index}"
            crop["crop_id"] = crop_id
            candidates = sorted(roster_for(team_roster, crop["alliance"]))
            permitted[crop_id] = candidates
            payload_crops.append({"crop_id": crop_id, "track_key": crop["track_key"], "alliance": crop["alliance"],
                                  "candidate_team_keys": candidates,
                                  "jpeg_base64": base64.b64encode(crop["jpeg"]).decode("ascii")})
        try:
            response = requests.post(f"{qwen_url.rstrip('/')}/read-team-numbers",
                headers={"Authorization": f"Bearer {qwen_token}", "Content-Type": "application/json"},
                json={"crops": payload_crops}, timeout=timeout)
            response.raise_for_status()
            reads = response.json().get("reads", [])
            if not isinstance(reads, list):
                raise ValueError('Identity response reads must be an array')
        except Exception as error:
            for crop in batch:
                by_key[crop["track_key"]].setdefault("metrics", {}).setdefault("teamIdentity", {})["error"] = str(error)[:300]
            continue
        crop_by_id = {crop["crop_id"]: crop for crop in batch}
        seen_crop_ids = set()
        for read in reads:
            if not isinstance(read, dict) or not isinstance(read.get('crop_id'), str) or read['crop_id'] in seen_crop_ids:
                continue
            crop = crop_by_id.get(read['crop_id'])
            if crop and read.get("team_key") in permitted[crop["crop_id"]]:
                seen_crop_ids.add(crop['crop_id'])
                evidence[crop["track_key"]].append({**read,
                    'track_key': crop['track_key'], 'reader': 'qwen_bumper',
                    'frame_index': crop['frame_index'], 'timestamp_ms': crop['timestamp_ms'],
                    'crop_sha256': hashlib.sha256(crop['jpeg']).hexdigest()})
    for key, track in by_key.items():
        if track.get("team_key"):  # a human identity_map always wins
            continue
        votes = combine_reader_votes(evidence[key], min_confidence=min_confidence)
        identity, details = decide_identity(votes, min_reads, min_confidence, min_share)
        track.setdefault("metrics", {})["teamIdentity"] = {**details, "reads": evidence[key], "distinct_frame_votes": len(votes)}
        if identity:
            track["team_key"] = identity
            track["identity_confidence"] = details["confidence"]
            track["needs_review"] = True  # still never auto-released
    return tracks
