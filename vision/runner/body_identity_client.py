"""Private model client: score-free cropped robot images and source-bound reference aliases only."""
from __future__ import annotations

import base64
import hashlib
from collections import defaultdict

from body_identity import MODEL, REVISION, METHOD, assemble_robots, observation_plan


def crop_plan(video_path, plan, mask_top_fraction=0.25):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    wanted = defaultdict(list)
    for index, (_, t, _, _) in enumerate(plan):
        wanted[round(t * fps / 1000)].append(index)
    crops, i = {}, 0
    while i <= max(wanted, default=-1) and cap.grab():
        if i in wanted:
            ok, frame = cap.retrieve()
            if ok:
                h, w = frame.shape[:2]
                # The broadcast overlay (including its clock) never enters an inference crop.
                frame[:round(h * mask_top_fraction)] = 0
                for index in wanted[i]:
                    box = plan[index][2]
                    x0, y0, x1, y1 = map(round, box)
                    image = frame[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
                    if image.size and min(image.shape[:2]) >= 20:
                        scale = min(1, 448 / max(image.shape[:2]))
                        if scale < 1:
                            image = cv2.resize(image, (max(1, round(image.shape[1] * scale)),
                                                      max(1, round(image.shape[0] * scale))))
                        encoded, jpeg = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 94])
                        if encoded:
                            crops[index] = base64.b64encode(jpeg.tobytes()).decode('ascii')
        i += 1
    cap.release()
    return crops


def match_bodies(plan, crops, roster, templates, url, token, timeout=900, batch=8, progress=None):
    """One observation per plan row [(tracklet, t, box, alliance)]: the gallery team the private model recognizes in
    that row's crop (`crops`: row index -> base64 JPEG), or None. Only the model and revision the evidence names are
    accepted; a failed or malformed answer leaves those views unrecognized."""
    import requests
    if templates.get('missing'):
        raise ValueError('every roster team needs two independent literal-read reference images')
    observations = [{'tracklet': tid, 't': t, 'team_key': None} for tid, t, _, _ in plan]
    completed = 0
    total = sum(i in crops and row[3] in ('red', 'blue') for i, row in enumerate(plan))
    for alliance in ('red', 'blue'):
        references = [{'reference_id': ref['reference_id'], 'jpeg_base64': ref['jpeg_base64']}
                      for ref in templates['references'] if ref['team_key'] in (roster.get(alliance) or [])]
        aliases = {key: team for key, team in templates['aliases'].items() if team in (roster.get(alliance) or [])}
        indices = [i for i, row in enumerate(plan) if row[3] == alliance and i in crops]
        for start in range(0, len(indices), batch):
            part = indices[start:start + batch]
            ids = {f't{plan[i][0]}_{plan[i][1]}': i for i in part}
            targets = [{'crop_id': key, 'jpeg_base64': crops[i]} for key, i in ids.items()]
            try:
                response = requests.post(url.rstrip('/') + '/match-robot-bodies',
                                         headers={'Authorization': 'Bearer ' + token},
                                         json={'references': references, 'targets': targets}, timeout=timeout)
                response.raise_for_status()
                data = response.json()
                if data.get('model') != MODEL or data.get('revision') != REVISION:
                    continue
                matches = data.get('matches')
                if not isinstance(matches, dict) or any(key not in ids for key in matches):
                    continue
                for key, index in ids.items():
                    observations[index]['team_key'] = aliases.get(matches.get(key))
                    observations[index]['identity_source'] = METHOD
                    observations[index]['crop_sha256'] = hashlib.sha256(base64.b64decode(crops[index])).hexdigest()
            except (requests.RequestException, ValueError, TypeError):
                # A failed view breaks a bracket; it never becomes an invented identity.
                continue
            finally:
                completed += len(part)
                if progress:
                    progress(completed, total)  # a request returned; not a fabricated periodic heartbeat
    return observations


def recognize(video_path, tracklets, roster, templates, url, token, timeout=900, batch=8, progress=None):
    if templates.get('missing'):
        raise ValueError('every roster team needs two independent literal-read reference images')
    plan = observation_plan(tracklets)
    return match_bodies(plan, crop_plan(video_path, plan), roster, templates, url, token, timeout=timeout, batch=batch, progress=progress)


def complete_alliances(roster, templates):
    """The roster's alliances with references for all three teams. With a team hidden from the gallery, 24% of views of
    its robots took a partner's name on 2026dal qm20, so an alliance missing a team is not recognized at all."""
    missing = set(templates.get('missing') or ())
    return {alliance: [] if missing & set(teams or ()) else list(teams or ()) for alliance, teams in roster.items()}


def identified_robots(video_path, tracklets, roster, templates, url, token, direct_reads=(), literal_names=None):
    complete = complete_alliances(roster, templates)
    observations = recognize(video_path, [track for track in tracklets if complete.get(track['alliance'])], complete,
                             {**templates, 'missing': []}, url, token)
    # A number read in a close-up panel outranks a body answer for another team on that tracklet: wide-view answers
    # were wrong on all 4 such 2026cc qm8 conflicts checked by eye.
    literal_names = literal_names or {}
    observations = [row for row in observations
                    if not row.get('team_key') or literal_names.get(row['tracklet'], row['team_key']) == row['team_key']]
    # Literal numbers veto disagreeing body identities at their exact source time.
    observations.extend({**row, 'identity_source': 'bumper_read'} for row in direct_reads if row.get('team_key'))
    return assemble_robots(tracklets, observations, roster, templates['references']), observations
