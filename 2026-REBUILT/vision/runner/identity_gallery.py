"""Private event reference images anchored only by literal bumper reads.

Appearance predictions never train this gallery. Two different source images
at different times are required for every roster team before body matching.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path

TEAM = re.compile(r"^frc[0-9]{1,5}$")
EVENT = re.compile(r"^[A-Za-z0-9_-]{3,40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def independent(a, b):
    if a["crop_sha256"] == b["crop_sha256"]:
        return False
    x, y = a["source"], b["source"]
    if x["recording_sha256"] != y["recording_sha256"]:
        return True
    return abs(x["t_ms"] - y["t_ms"]) >= 1000


class IdentityGallery:
    def __init__(self, event_key, path=None):
        if not EVENT.fullmatch(event_key):
            raise ValueError("valid event key required")
        self.event_key, self.path = event_key, Path(path) if path else None
        self.examples = []
        self.rejected_digests = set()
        if self.path and self.path.exists():
            data = json.loads(self.path.read_text())
            if data.get("schema") != 1 or data.get("event_key") != event_key:
                raise ValueError("gallery event/schema does not match")
            self.rejected_digests.update(data.get("rejected_digests") or [])
            for row in data.get("examples") or []:
                self.add(row)

    def add(self, row):
        """Accept a source-bound literal read, never a path/appearance label or repeated pixels."""
        if not isinstance(row, dict) or not TEAM.fullmatch(str(row.get("team_key") or "")):
            return False
        confidence = row.get("confidence")
        source = row.get("source") or {}
        if (row.get("identity_source") != "bumper_read" or row.get("digits") != row["team_key"][3:]
                or not isinstance(confidence, (int, float)) or isinstance(confidence, bool)
                or not math.isfinite(confidence) or not .8 <= confidence <= 1
                or not SHA256.fullmatch(str(source.get("recording_sha256") or ""))
                or not isinstance(source.get("t_ms"), int) or isinstance(source["t_ms"], bool) or source["t_ms"] < 0
                or not str(source.get("match_key") or "").startswith(self.event_key + "_")
                or not source.get("reader") or not source.get("reader_revision")):
            return False
        try:
            jpeg = base64.b64decode(row.get("jpeg_base64") or "", validate=True)
        except (ValueError, TypeError):
            return False
        if not jpeg or len(jpeg) > 2 * 1024 * 1024:
            return False
        digest = hashlib.sha256(jpeg).hexdigest()
        if row.get("crop_sha256") != digest or digest in self.rejected_digests:
            return False
        old = [item for item in self.examples if item["crop_sha256"] == digest]
        if old:
            if any(item["team_key"] != row["team_key"] for item in old):
                self.examples = [item for item in self.examples if item["crop_sha256"] != digest]
                self.rejected_digests.add(digest)
            return False
        self.examples.append(dict(row))
        # Bound private storage while retaining different matches before adding more of one view.
        team_rows = [item for item in self.examples if item["team_key"] == row["team_key"]]
        if len(team_rows) > 8:
            ranked = sorted(team_rows, key=lambda item: (-float(item.get("quality") or 0), item["source"]["t_ms"]))
            chosen, match_counts = [], defaultdict(int)
            while ranked and len(chosen) < 8:
                best = min(range(len(ranked)), key=lambda i: match_counts[ranked[i]["source"]["match_key"]])
                candidate = ranked.pop(best)
                if any(not independent(candidate, item) for item in chosen):
                    continue
                chosen.append(candidate)
                match_counts[candidate["source"]["match_key"]] += 1
            self.examples = [item for item in self.examples if item["team_key"] != row["team_key"]] + chosen
        return True

    def templates(self, roster):
        teams = list(dict.fromkeys((roster.get("red") or []) + (roster.get("blue") or [])))
        if not 1 <= len(teams) <= 6 or any(not TEAM.fullmatch(team) for team in teams):
            raise ValueError("bounded match roster required")
        references, aliases, missing = [], {}, []
        for alias, team in zip("ABCDEF", teams):
            desired_alliance = next((alliance for alliance in ('red', 'blue') if team in (roster.get(alliance) or [])), None)
            rows = sorted((row for row in self.examples if row["team_key"] == team),
                          key=lambda row: -float(row.get("quality") or 0))
            pairs = [(a, b) for i, a in enumerate(rows) for b in rows[i + 1:] if independent(a, b)]
            def variety(pair):
                a, b = pair
                return (sum(row['source'].get('alliance') == desired_alliance for row in (a, b)),
                        a['source']['match_key'] != b['source']['match_key'],
                        min(10000, abs(a['source']['t_ms'] - b['source']['t_ms'])),
                        min(float(a.get('quality') or 0), float(b.get('quality') or 0)))
            pair = max(pairs, key=variety) if pairs else None
            if pair is None:
                missing.append(team)
                continue
            aliases[alias] = team
            for example in pair:
                references.append({**example, "reference_id": alias})
        return {"references": references, "aliases": aliases, "missing": missing}

    def save(self):
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        data = {"schema": 1, "event_key": self.event_key, "examples": self.examples,
                "rejected_digests": sorted(self.rejected_digests)}
        with tempfile.NamedTemporaryFile(mode="w", dir=self.path.parent, prefix=".identity-", delete=False) as file:
            temporary = Path(file.name)
            try:
                os.chmod(temporary, 0o600)
                json.dump(data, file)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        temporary.replace(self.path)
