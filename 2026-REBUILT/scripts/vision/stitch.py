"""Exact identity solving: turn broken tracklets into three robots per alliance.

A tracker breaks a robot's track at every occlusion, so one robot becomes many tracklets, and
which tracklet is which team is only known from a few OCR / VLM reads of bumper numbers. Greedy
linking compounds early mistakes. Instead, per alliance, find the cheapest way to explain all
tracklets as at most `slots` (3) robots, each a time-ordered chain whose consecutive tracklets
can be linked at a feasible speed, with unexplained tracklets allowed at a cost. The search is
exact dynamic programming over "last tracklet in each slot", which is tiny (a few dozen tracklets
over 23 seconds) and cannot be fooled by a locally attractive wrong link. Teams are then assigned
to chains by maximising summed vote weight over the roster permutations.
"""
import itertools
import math
from functools import lru_cache

from common import FIELD

VMAX = FIELD['robot']['maxPlausibleSpeedMps']


def _effective_votes(tracklet):
    if not isinstance(tracklet, dict):
        ev = getattr(tracklet, 'evidence', None)
        votes = getattr(tracklet, 'votes', None)
        tracklet = {'votes': votes or {}, 'evidence': ev or []}
    if tracklet.get('evidence'):
        return dict(aggregate_tracklet_votes(tracklet)['totals'])
    return {
        str(k): float(v)
        for k, v in (tracklet.get('votes') or {}).items()
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0
    }


def _dominant_team(tracklet):
    if not isinstance(tracklet, dict):
        tracklet = {'votes': getattr(tracklet, 'votes', None) or {}, 'evidence': getattr(tracklet, 'evidence', None) or []}
    if tracklet.get('evidence'):
        agg = aggregate_tracklet_votes(tracklet)
        votes = agg['totals']
        if not votes:
            return None
        top = max(votes, key=votes.get)
        total = sum(votes.values())
        if total <= 0:
            return None
        share = votes[top] / total
        t_info = agg['by_team'].get(top, {})
        ht_w = sum(w for src, w in (t_info.get('sources') or {}).items() if src in HIGH_TRUST_DIRECT_SOURCES)
        if ht_w >= 1.5 and share >= 0.58:
            return top
        runner_up = max((v for k, v in votes.items() if k != top), default=0.0)
        min_share = 0.75 if runner_up >= 3.0 else 0.7
        return top if share >= min_share else None
    votes = _effective_votes(tracklet)
    if not votes:
        return None
    top = max(votes, key=votes.get)
    total = sum(votes.values())
    runner_up = max((v for k, v in votes.items() if k != top), default=0.0)
    min_share = 0.75 if runner_up >= 3.0 else 0.7
    return top if total > 0 and votes[top] / total >= min_share else None


def _fit_point_velocity(tracklet, segment_pts):
    if not isinstance(segment_pts, (list, tuple)) or len(segment_pts) < 2:
        return None
    try:
        raw_ts = [float(p['t_s'] if 't_s' in p else p['t']) for p in segment_pts]
        xs = [float(p['x']) for p in segment_pts]
        ys = [float(p['y']) for p in segment_pts]
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(t) and math.isfinite(x) and math.isfinite(y) for t, x, y in zip(raw_ts, xs, ys)):
        return None
    t_scale = 1.0
    if 't_s' not in segment_pts[-1] and tracklet.get('end') is not None and tracklet.get('start') is not None:
        dur_s = float(tracklet['end']) - float(tracklet['start'])
        if dur_s > 0 and (raw_ts[-1] - raw_ts[0]) > 100.0 * dur_s:
            t_scale = 0.001
    ts_all = [t * t_scale for t in raw_ts]
    return ts_all, xs, ys


def _endpoint_velocity(tracklet):
    """Estimate `(vx, vy)` in m/s at the end of `tracklet`, clamped to `VMAX`, or `None` if unavailable."""
    if not isinstance(tracklet, dict):
        return None
    ev = tracklet.get('end_velocity')
    if isinstance(ev, (list, tuple)) and len(ev) == 2:
        try:
            vx, vy = float(ev[0]), float(ev[1])
            if not isinstance(ev[0], bool) and not isinstance(ev[1], bool) and math.isfinite(vx) and math.isfinite(vy):
                speed = math.hypot(vx, vy)
                if speed > VMAX:
                    vx, vy = vx * VMAX / speed, vy * VMAX / speed
                return vx, vy
        except (TypeError, ValueError):
            pass
    pts = tracklet.get('points')
    if not isinstance(pts, (list, tuple)) or len(pts) < 2:
        return None
    parsed = _fit_point_velocity(tracklet, pts[-6:])
    if parsed is None:
        return None
    ts_all, xs, ys = parsed
    last_t = ts_all[-1]
    indices = [i for i, t in enumerate(ts_all) if last_t - t <= 0.6 + 1e-6]
    if len(indices) < 2 or ts_all[indices[-1]] - ts_all[indices[0]] <= 1e-3:
        return None
    ts = [ts_all[i] for i in indices]
    rxs = [xs[i] for i in indices]
    rys = [ys[i] for i in indices]
    n = len(ts)
    mt = sum(ts) / n
    den = sum((t - mt) ** 2 for t in ts)
    if den <= 1e-9:
        return None
    vx = sum((t - mt) * x for t, x in zip(ts, rxs)) / den
    vy = sum((t - mt) * y for t, y in zip(ts, rys)) / den
    speed = math.hypot(vx, vy)
    if speed > VMAX:
        vx, vy = vx * VMAX / speed, vy * VMAX / speed
    return vx, vy


def _startpoint_velocity(tracklet):
    """Estimate `(vx, vy)` in m/s at the start of `tracklet`, clamped to `VMAX`, or `None` if unavailable."""
    if not isinstance(tracklet, dict):
        return None
    sv = tracklet.get('start_velocity')
    if isinstance(sv, (list, tuple)) and len(sv) == 2:
        try:
            vx, vy = float(sv[0]), float(sv[1])
            if not isinstance(sv[0], bool) and not isinstance(sv[1], bool) and math.isfinite(vx) and math.isfinite(vy):
                speed = math.hypot(vx, vy)
                if speed > VMAX:
                    vx, vy = vx * VMAX / speed, vy * VMAX / speed
                return vx, vy
        except (TypeError, ValueError):
            pass
    pts = tracklet.get('points')
    if not isinstance(pts, (list, tuple)) or len(pts) < 2:
        return None
    parsed = _fit_point_velocity(tracklet, pts[:6])
    if parsed is None:
        return None
    ts_all, xs, ys = parsed
    first_t = ts_all[0]
    indices = [i for i, t in enumerate(ts_all) if t - first_t <= 0.6 + 1e-6]
    if len(indices) < 2 or ts_all[indices[-1]] - ts_all[indices[0]] <= 1e-3:
        return None
    ts = [ts_all[i] for i in indices]
    rxs = [xs[i] for i in indices]
    rys = [ys[i] for i in indices]
    n = len(ts)
    mt = sum(ts) / n
    den = sum((t - mt) ** 2 for t in ts)
    if den <= 1e-9:
        return None
    vx = sum((t - mt) * x for t, x in zip(ts, rxs)) / den
    vy = sum((t - mt) * y for t, y in zip(ts, rys)) / den
    speed = math.hypot(vx, vy)
    if speed > VMAX:
        vx, vy = vx * VMAX / speed, vy * VMAX / speed
    return vx, vy


def _vote_link_adjustment(a, b, slot_team=None):
    """Return `(multiplier, additive_penalty)` from dominant/soft votes, or `None` on hard team conflict."""
    ta = _dominant_team(a) or (str(slot_team) if slot_team is not None else None)
    tb = _dominant_team(b)
    if ta is not None and tb is not None and ta != tb:
        return None
    if ta is not None and tb is not None and ta == tb:
        return 0.5, 0.0
    t_ref = ta or tb
    if t_ref is not None:
        other = b if ta is not None else a
        v = _effective_votes(other)
        if v:
            canon = _canon_team_map(raw_teams=list(v.keys()) + [t_ref])
            c_ref = canon.get(str(t_ref).strip(), str(t_ref).strip())
            tot = sum(v.values())
            ref_w = sum(val for k, val in v.items() if canon.get(str(k).strip(), str(k).strip()) == c_ref)
            share = ref_w / tot if tot > 0 else 0.0
            if share >= 0.40:
                return 0.75, 0.0
            if tot >= 1.5 and share < 0.10:
                return 1.0, 0.25
    return 1.0, 0.0


def _link_cost(a, b, slot_team=None):
    """Cost of continuing a robot from tracklet a into b, or None if impossible."""
    gap = (b['start'] - a['end'])
    if gap < -0.15:
        return None
    gap_pos = max(gap, 0.0)
    dist = math.hypot(b['first'][0] - a['last'][0], b['first'][1] - a['last'][1])
    reachable = VMAX * 1.25 * gap_pos + 0.9
    if dist > reachable:
        return None
    vote_adj = _vote_link_adjustment(a, b, slot_team=slot_team)
    if vote_adj is None:
        return None
    mult, pen = vote_adj
    va = _endpoint_velocity(a)
    vb = _startpoint_velocity(b)
    if 0.0 < gap_pos <= 2.0 and (va is not None or vb is not None):
        lead = min(gap_pos, 1.5)
        if va is not None and vb is not None:
            fwd_dist = math.hypot(b['first'][0] - (a['last'][0] + va[0] * lead), b['first'][1] - (a['last'][1] + va[1] * lead))
            bwd_dist = math.hypot((b['first'][0] - vb[0] * lead) - a['last'][0], (b['first'][1] - vb[1] * lead) - a['last'][1])
            pred_dist = 0.5 * fwd_dist + 0.5 * bwd_dist
        elif va is not None:
            pred_dist = math.hypot(b['first'][0] - (a['last'][0] + va[0] * lead), b['first'][1] - (a['last'][1] + va[1] * lead))
        else:
            pred_dist = math.hypot((b['first'][0] - vb[0] * lead) - a['last'][0], (b['first'][1] - vb[1] * lead) - a['last'][1])
        eff_dist = 0.6 * pred_dist + 0.4 * dist
    else:
        eff_dist = dist
    return (0.05 * eff_dist + 0.075 * gap_pos) * mult + pen


STALE_AFTER_S = 4.0        # a slot idle this long no longer constrains where its robot can be next
EARLY_ANCHOR_STALE_S = 12.0
MAX_TRACKLETS = 48         # more than this and the weakest (shortest, least confident) are left unexplained


def _coerce_tracklet_dict(tr, index):
    if isinstance(tr, dict):
        if 'start' in tr and 'end' in tr and 'first' in tr and 'last' in tr:
            return tr
        return {
            'start': float(tr.get('start', tr.get('start_s', 0.0))),
            'end': float(tr.get('end', tr.get('end_s', 0.0))),
            'first': tuple(tr.get('first') or tr.get('first_xy') or (0.0, 0.0)),
            'last': tuple(tr.get('last') or tr.get('last_xy') or (0.0, 0.0)),
            'votes': dict(tr.get('votes') or {}),
            'evidence': tr.get('evidence'),
            'crops': tr.get('crops'),
            'weight': float(tr.get('weight', 1.0)),
            'points': tr.get('points'),
            'start_velocity': tr.get('start_velocity'),
            'end_velocity': tr.get('end_velocity'),
            'fragment_id': tr.get('fragment_id', f't{index}'),
        }
    return {
        'start': float(getattr(tr, 'start', 0.0)),
        'end': float(getattr(tr, 'end', 0.0)),
        'first': tuple(getattr(tr, 'first', (0.0, 0.0))),
        'last': tuple(getattr(tr, 'last', (0.0, 0.0))),
        'votes': dict(getattr(tr, 'votes', None) or {}),
        'evidence': getattr(tr, 'evidence', None),
        'crops': getattr(tr, 'crops', None),
        'weight': float(getattr(tr, 'confidence', 1.0)),
        'points': getattr(tr, 'points', None),
        'start_velocity': getattr(tr, 'start_velocity', None),
        'end_velocity': getattr(tr, 'end_velocity', None),
        'fragment_id': getattr(tr, 'fragment_id', f't{index}'),
    }


def _resolve_fragment_index(ref, id_to_idx, total):
    if ref is None or isinstance(ref, bool):
        return None
    if isinstance(ref, int) and 0 <= ref < total:
        return ref
    if ref in id_to_idx:
        return id_to_idx[ref]
    s = str(ref).strip()
    if s in id_to_idx:
        return id_to_idx[s]
    if s.isdigit() and 0 <= int(s) < total:
        return int(s)
    if s.startswith('t') and s[1:].isdigit() and 0 <= int(s[1:]) < total:
        return int(s[1:])
    return None


def solve_alliance(tracklets, slots=3, skip_per_second=2.5, new_chain=0.3, forced_links=None, forbidden_links=None):
    """Assign tracklets to `slots` chains. tracklets: dicts with start, end (s), first, last ((x, y)),
    duration is end-start. Returns a list of `slots` lists of tracklet indices (time ordered).

    Exact dynamic programming over "the last tracklet in each slot". That state space grows as n^slots, which
    took 0.9 s at 40 tracklets and 5.7 s at 60 and would not finish on real footage (hundreds of fragments).
    Two reductions keep it exact for what matters: a slot whose last tracklet ended more than STALE_AFTER_S
    before the next one starts is treated as free (a robot unseen that long could be anywhere reachable, so the
    link tells us nothing), which bounds the states by what happened RECENTLY; and the list is capped at
    MAX_TRACKLETS by keeping the longest, most confident fragments (the rest are noise or too short to matter).
    """
    raw_tracklets = [_coerce_tracklet_dict(t, i) for i, t in enumerate(tracklets)]
    total_raw = len(raw_tracklets)
    id_to_idx = {}
    for i, t in enumerate(raw_tracklets):
        fid = t.get('fragment_id')
        if fid is not None:
            id_to_idx[str(fid)] = i
        id_to_idx.setdefault(f't{i}', i)

    norm_forced = {}
    for src_ref, dst_ref in (forced_links or {}).items():
        si = _resolve_fragment_index(src_ref, id_to_idx, total_raw)
        di = _resolve_fragment_index(dst_ref, id_to_idx, total_raw)
        if si is not None and di is not None and si != di:
            if raw_tracklets[di]['start'] - raw_tracklets[si]['end'] >= -0.15:
                ta, tb = _dominant_team(raw_tracklets[si]), _dominant_team(raw_tracklets[di])
                if ta is None or tb is None or ta == tb:
                    norm_forced[si] = di

    norm_forbidden = set()
    for pair in (forbidden_links or ()):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        si = _resolve_fragment_index(pair[0], id_to_idx, total_raw)
        if si is None:
            continue
        if pair[1] == '*' or pair[1] is None:
            norm_forbidden.add((si, '*'))
        else:
            di = _resolve_fragment_index(pair[1], id_to_idx, total_raw)
            if di is not None:
                norm_forbidden.add((si, di))

    forced_rev = {di: si for si, di in norm_forced.items()}
    forced_nodes = set(norm_forced.keys()) | set(norm_forced.values())
    constrained_sources = set(norm_forced.keys()) | {si for si, _ in norm_forbidden}

    candidates = list(range(total_raw))
    min_start_all = min((raw_tracklets[i]['start'] for i in candidates), default=0.0)
    if len(candidates) > MAX_TRACKLETS:
        auto_anchors = sorted(
            (
                i
                for i in candidates
                if raw_tracklets[i]['start'] <= min_start_all + 23.0
                and (raw_tracklets[i]['end'] - raw_tracklets[i]['start']) >= 0.25
            ),
            key=lambda i: (
                1 if raw_tracklets[i]['start'] <= min_start_all + 2.5 else 0,
                (raw_tracklets[i]['end'] - raw_tracklets[i]['start']) * raw_tracklets[i].get('weight', 1.0),
            ),
            reverse=True,
        )[: slots * 4]
        auto_anchor_set = set(auto_anchors)
        candidates.sort(
            key=lambda i: (
                2
                if (i in forced_nodes or _dominant_team(raw_tracklets[i]) is not None)
                else (1 if i in auto_anchor_set else 0),
                (raw_tracklets[i]['end'] - raw_tracklets[i]['start']) * raw_tracklets[i].get('weight', 1.0),
            ),
            reverse=True,
        )
        candidates = candidates[:MAX_TRACKLETS]
    order = sorted(candidates, key=lambda i: (raw_tracklets[i]['start'], i))
    items = [raw_tracklets[i] for i in order]
    doms = [_dominant_team(it) for it in items]
    eff_votes = [_effective_votes(it) for it in items]
    n = len(items)
    min_start = min((it['start'] for it in items), default=0.0)
    stale_cost = new_chain + 0.1

    early_anchor_indices = set(
        sorted(
            (
                idx
                for idx, it in enumerate(items)
                if doms[idx] is None and it['start'] <= min_start + 2.5 and it['end'] <= min_start + 6.0
            ),
            key=lambda idx: (items[idx]['end'] - items[idx]['start']) * items[idx].get('weight', 1.0),
            reverse=True,
        )[:slots]
    )

    def is_early_start_anchor(last_idx, slot_team):
        return last_idx is not None and last_idx in early_anchor_indices and slot_team is None

    def is_stale(last_idx, slot_team, cur_start):
        if last_idx is None or last_idx < 0 or order[last_idx] in constrained_sources:
            return False
        horizon = EARLY_ANCHOR_STALE_S if is_early_start_anchor(last_idx, slot_team) else STALE_AFTER_S
        return items[last_idx]['end'] < cur_start - horizon

    def canon(state, k):
        if k < n:
            start = items[k]['start']
            state = (
                (
                    -1 if is_stale(last, team, start) else last,
                    team,
                )
                for last, team in state
            )
        return tuple(sorted(state, key=lambda st: (-2 if st[0] is None else st[0], st[1] or '')))

    @lru_cache(maxsize=None)
    def best(k, state):
        if k == n:
            return 0.0, ('skip', None)
        item = items[k]
        dom = doms[k]
        orig_k = order[k]
        forced_src = forced_rev.get(orig_k)
        has_forced_src_slot = (
            forced_src is not None
            and any(
                last is not None and last >= 0 and order[last] == forced_src and (team is None or dom is None or team == dom)
                for last, team in state
            )
        )
        skip = skip_per_second * max(0.1, item['end'] - item['start']) * item.get('weight', 1.0)
        if orig_k in norm_forced or has_forced_src_slot:
            best_val, best_choice = float('inf'), ('skip', None)
        else:
            best_val, best_choice = skip + best(k + 1, canon(state, k + 1))[0], ('skip', None)
        prev = object()
        for s in range(slots):
            slot = state[s]
            if slot == prev:
                continue
            prev = slot
            last, team = slot
            if team is not None and dom is not None and team != dom:
                continue
            if has_forced_src_slot and not (last is not None and last >= 0 and order[last] == forced_src):
                continue
            new_team = dom or team
            if last is None:
                cost = new_chain
            elif last == -1:
                if team is not None and team == dom:
                    cost = new_chain * 0.5
                elif team is not None and dom is None and eff_votes[k].get(team, 0) > 0:
                    cost = new_chain * 0.75
                else:
                    cost = stale_cost
            else:
                orig_last = order[last]
                if (orig_last, orig_k) in norm_forbidden or (orig_last, '*') in norm_forbidden:
                    continue
                if orig_last in norm_forced:
                    if norm_forced[orig_last] != orig_k:
                        continue
                    cost = 0.0
                elif item['start'] - items[last]['end'] > STALE_AFTER_S and is_early_start_anchor(last, team):
                    gap = item['start'] - items[last]['end']
                    dist = math.hypot(item['first'][0] - items[last]['last'][0], item['first'][1] - items[last]['last'][1])
                    if gap <= EARLY_ANCHOR_STALE_S and dist <= VMAX * 1.25 * gap + 0.9:
                        cost = new_chain * 0.85 + 1e-4 * dist
                    else:
                        cost = stale_cost
                else:
                    cost = _link_cost(items[last], item, slot_team=team)
                    if cost is None:
                        continue
            nxt = state[:s] + ((k, new_team),) + state[s + 1:]
            val = cost + best(k + 1, canon(nxt, k + 1))[0]
            if val < best_val:
                best_val, best_choice = val, ('slot', slot)
        if best_val == float('inf'):
            best_val, best_choice = skip + best(k + 1, canon(state, k + 1))[0], ('skip', None)
        return best_val, best_choice

    slot_states = [(None, None)] * slots
    actual_last = [None] * slots
    chains = [[] for _ in range(slots)]
    for k in range(n):
        start = items[k]['start']
        for i, (last, team) in enumerate(slot_states):
            if is_stale(last, team, start):
                slot_states[i] = (-1, team)
        c_state = canon(slot_states, k)
        _, (action, target_slot) = best(k, c_state)
        if action == 'slot':
            matching = [i for i, st in enumerate(slot_states) if st == target_slot]
            if len(matching) > 1 and target_slot[0] == -1:
                s = min(matching, key=lambda i: math.hypot(items[k]['first'][0] - items[actual_last[i]]['last'][0], items[k]['first'][1] - items[actual_last[i]]['last'][1]))
            else:
                s = matching[0]
            chains[s].append(order[k])
            actual_last[s] = k
            slot_states[s] = (k, doms[k] or slot_states[s][1])
    return chains


def apply_validated_reconnects(tracklets, events, validations, slots=3):
    """Apply accepted `Validation` decisions to `solve_alliance` and return `(chains, forced_links, forbidden_links)`."""
    forced_links = {}
    forbidden_links = set()
    events_list = list(events or [])
    if isinstance(validations, dict):
        val_by_id = {str(k): v for k, v in validations.items() if isinstance(v, dict)}
        val_list = []
    else:
        val_list = list(validations or [])
        val_by_id = {
            str(v.get('event_id')): v
            for v in val_list
            if isinstance(v, dict) and v.get('event_id') is not None
        }

    for idx, ev in enumerate(events_list):
        if not isinstance(ev, dict):
            continue
        ev_id = str(ev.get('id')) if ev.get('id') is not None else None
        val = val_by_id.get(ev_id) if (ev_id is not None and ev_id in val_by_id) else None
        if val is None and idx < len(val_list) and isinstance(val_list[idx], dict):
            if not val_list[idx].get('event_id') or val_list[idx].get('event_id') == ev.get('id'):
                val = val_list[idx]
        if not isinstance(val, dict) or not bool(val.get('accepted')):
            continue
        ended = ev.get('ended')
        ended_id = ended.get('fragment_id') if isinstance(ended, dict) else (ended if ended is not None else ev.get('ended_id'))
        if ended_id is None:
            continue
        cand_ids = [
            c.get('fragment_id') if isinstance(c, dict) else c
            for c in (ev.get('candidates') or [])
        ]
        cand_ids = [c for c in cand_ids if c is not None]
        final = val.get('final')
        choice = val.get('choice')
        if final is not None and str(final).strip() not in ('', 'none', 'unsure'):
            forced_links[ended_id] = final
            for cid in cand_ids:
                if str(cid) != str(final):
                    forbidden_links.add((ended_id, cid))
        elif str(choice).strip() == 'none':
            forced_links.pop(ended_id, None)
            if cand_ids:
                for cid in cand_ids:
                    forbidden_links.add((ended_id, cid))
            else:
                forbidden_links.add((ended_id, '*'))

    chains = solve_alliance(tracklets, slots=slots, forced_links=forced_links, forbidden_links=forbidden_links)
    return chains, forced_links, forbidden_links


SOURCE_WEIGHTS = {
    'closeup_crop': 2.5,
    'zoom_crop': 2.2,
    'prior_run': 2.0,
    'reads': 1.0,
    'appearance': 0.85,
    'auto_fingerprint': 0.80,
    'wide_frame': 0.45,
}
TRACKLET_VOTE_SOURCES = frozenset({'closeup_crop', 'zoom_crop', 'reads', 'wide_frame'})
DIRECT_READ_SOURCES = frozenset({'closeup_crop', 'zoom_crop', 'prior_run', 'reads', 'wide_frame'})
HIGH_TRUST_DIRECT_SOURCES = frozenset({'closeup_crop', 'zoom_crop', 'prior_run'})
INDIRECT_PRIOR_SOURCES = frozenset({'appearance', 'auto_fingerprint'})
WIDE_FRAME_SATURATION_CAP = 4.0
PRIOR_CONFIDENCE_CAP = 0.55
INFERRED_CONFIDENCE_CAP = 0.45   # a name that is not backed by a decisive read never looks better than this


def _canon_team_map(roster=None, raw_teams=()):
    canon = {}
    roster_strings = [str(t).strip() for t in (roster or ()) if t is not None and str(t).strip()]
    for s in roster_strings:
        canon[s] = s
        canon[s[3:] if s.startswith('frc') else f'frc{s}'] = s
    strings = [str(t).strip() for t in (raw_teams or ()) if t is not None and str(t).strip()]
    prefer_frc = any(s.startswith('frc') for s in roster_strings) or any(s.startswith('frc') for s in strings)
    for s in strings:
        if s in canon:
            continue
        bare = s[3:] if s.startswith('frc') else s
        if bare in canon:
            canon[s] = canon[bare]
            continue
        target = f'frc{bare}' if (prefer_frc and bare.isdigit()) else s
        canon[s] = target
        if bare.isdigit():
            canon[bare] = target
            canon[f'frc{bare}'] = target
    return canon


def _collect_raw_teams(tracklets=(), priors_by_chain=None):
    raw_teams = []
    for tr in tracklets or ():
        if isinstance(tr, dict):
            raw_teams.extend((tr.get('votes') or {}).keys())
            raw_teams.extend(it.get('team') for it in (tr.get('evidence') or ()) if isinstance(it, dict))
        else:
            raw_teams.extend((getattr(tr, 'votes', None) or {}).keys())
            raw_teams.extend(it.get('team') for it in (getattr(tr, 'evidence', None) or ()) if isinstance(it, dict))
    for p_list in (priors_by_chain or {}).values():
        raw_teams.extend(it.get('team') for it in p_list if isinstance(it, dict))
    return raw_teams


def _pre_dedup_vote_totals(raw_items, canon_map):
    pre_dedup_vote_raw = {}
    for item in raw_items or ():
        if not isinstance(item, dict):
            continue
        raw_team = item.get('team')
        if raw_team is None or isinstance(raw_team, bool) or not str(raw_team).strip():
            continue
        conf = item.get('confidence')
        if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not math.isfinite(float(conf)) or float(conf) <= 0.0:
            continue
        source = str(item.get('source') or 'reads').strip() or 'reads'
        if source in TRACKLET_VOTE_SOURCES:
            s_team = str(raw_team).strip()
            team = canon_map.get(s_team, s_team)
            pre_dedup_vote_raw[team] = pre_dedup_vote_raw.get(team, 0.0) + float(conf)
    return pre_dedup_vote_raw


def _valid_legacy_votes(legacy_votes, canon_map):
    valid_legacy = {}
    for k, v in (legacy_votes or {}).items():
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)) and float(v) > 0.0:
            s_k = str(k).strip()
            if s_k:
                team = canon_map.get(s_k, s_k)
                valid_legacy[team] = valid_legacy.get(team, 0.0) + float(v)
    return valid_legacy


def _dedup_evidence_items(raw_items, canon_map):
    cleaned = []
    for item in raw_items or ():
        if not isinstance(item, dict):
            continue
        raw_team = item.get('team')
        if raw_team is None or isinstance(raw_team, bool) or not str(raw_team).strip():
            continue
        conf = item.get('confidence')
        if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not math.isfinite(float(conf)) or float(conf) <= 0.0:
            continue
        s_team = str(raw_team).strip()
        team = canon_map.get(s_team, s_team)
        source = str(item.get('source') or 'reads').strip() or 'reads'
        fi = item.get('frame_index')
        frame_index = fi if (isinstance(fi, int) and not isinstance(fi, bool) and fi >= 0) else None
        sha = item.get('crop_sha256')
        crop_sha256 = sha.strip() if isinstance(sha, str) and sha.strip() else None
        cleaned.append({
            'source': source,
            'team': team,
            'confidence': float(conf),
            'frame_index': frame_index,
            'crop_sha256': crop_sha256,
        })

    # 1. Deduplicate (source, frame_index) and cancel conflicting reads on the same frame
    after_frame = []
    by_frame = {}
    for it in cleaned:
        if it['frame_index'] is None:
            after_frame.append(it)
        else:
            by_frame.setdefault((it['source'], it['frame_index']), []).append(it)
    for grp in by_frame.values():
        if len({it['team'] for it in grp}) == 1:
            after_frame.append(max(grp, key=lambda it: it['confidence']))

    # 2. Deduplicate (source, crop_sha256) and cancel conflicting reads on the same crop digest
    surviving = []
    by_sha = {}
    for it in after_frame:
        if it['crop_sha256'] is None:
            surviving.append(it)
        else:
            by_sha.setdefault((it['source'], it['crop_sha256']), []).append(it)
    for grp in by_sha.values():
        if len({it['team'] for it in grp}) == 1:
            surviving.append(max(grp, key=lambda it: it['confidence']))

    return surviving


def _reconciled_tracklet_items(tracklet, canon_map, chain_has_structured_evidence=False):
    tr_ev = (tracklet.get('evidence') if isinstance(tracklet, dict) else getattr(tracklet, 'evidence', None)) or []
    tr_votes = (tracklet.get('votes') if isinstance(tracklet, dict) else getattr(tracklet, 'votes', None)) or {}
    valid_legacy = _valid_legacy_votes(tr_votes, canon_map)
    if tr_ev:
        pre_dedup_vote_raw = _pre_dedup_vote_totals(tr_ev, canon_map)
        items = _dedup_evidence_items(tr_ev, canon_map)
        if valid_legacy:
            legacy_source = 'wide_frame' if (len(valid_legacy) > 1 or sum(valid_legacy.values()) > WIDE_FRAME_SATURATION_CAP) else 'reads'
            for team, leg_w in valid_legacy.items():
                residual = leg_w - pre_dedup_vote_raw.get(team, 0.0)
                if residual > 1e-6:
                    items.append({
                        'source': legacy_source,
                        'team': team,
                        'confidence': residual,
                        'frame_index': None,
                        'crop_sha256': None,
                    })
        return items, True

    if not valid_legacy:
        return [], False
    legacy_source = (
        'wide_frame'
        if (chain_has_structured_evidence and (len(valid_legacy) > 1 or sum(valid_legacy.values()) > WIDE_FRAME_SATURATION_CAP))
        else 'reads'
    )
    items = [
        {
            'source': legacy_source,
            'team': team,
            'confidence': leg_w,
            'frame_index': None,
            'crop_sha256': None,
        }
        for team, leg_w in valid_legacy.items()
    ]
    return items, False


def _tally_from_evidence_and_votes(evidence_items, legacy_votes, canon_map, has_structured_evidence=False):
    if not has_structured_evidence:
        totals, direct_totals, prior_totals, by_team = {}, {}, {}, {}
        for raw_team, weight in (legacy_votes or {}).items():
            if isinstance(weight, (int, float)) and not isinstance(weight, bool) and math.isfinite(float(weight)) and float(weight) > 0.0:
                s_team = str(raw_team).strip()
                team = canon_map.get(s_team, s_team)
                w = float(weight)
                totals[team] = totals.get(team, 0.0) + w
                direct_totals[team] = direct_totals.get(team, 0.0) + w
        for team, w in totals.items():
            by_team[team] = {
                'total': round(w, 4),
                'direct': round(w, 4),
                'prior': 0.0,
                'sources': {'reads': round(w, 4)},
                'evidence_count': 1,
            }
        return {'totals': totals, 'direct_totals': direct_totals, 'prior_totals': prior_totals, 'by_team': by_team}

    pre_dedup_vote_raw = _pre_dedup_vote_totals(evidence_items, canon_map)
    deduped = _dedup_evidence_items(evidence_items, canon_map)
    # Include any residual legacy votes not already represented in tracklet vote sources
    if legacy_votes:
        valid_legacy = _valid_legacy_votes(legacy_votes, canon_map)
        legacy_source = 'wide_frame' if (len(valid_legacy) > 1 or sum(valid_legacy.values()) > WIDE_FRAME_SATURATION_CAP) else 'reads'
        for team, leg_w in valid_legacy.items():
            residual = leg_w - pre_dedup_vote_raw.get(team, 0.0)
            if residual > 1e-6:
                deduped.append({
                    'source': legacy_source,
                    'team': team,
                    'confidence': residual,
                    'frame_index': None,
                    'crop_sha256': None,
                })

    raw_by_team_source = {}
    counts_by_team = {}
    for it in deduped:
        team, src, conf = it['team'], it['source'], it['confidence']
        raw_by_team_source.setdefault(team, {})[src] = raw_by_team_source.get(team, {}).get(src, 0.0) + conf
        counts_by_team[team] = counts_by_team.get(team, 0) + 1

    totals, direct_totals, prior_totals, by_team = {}, {}, {}, {}
    for team, src_map in raw_by_team_source.items():
        weighted_sources = {}
        t_total = 0.0
        t_direct = 0.0
        t_prior = 0.0
        for src, raw_w in src_map.items():
            if src == 'wide_frame':
                c = WIDE_FRAME_SATURATION_CAP
                w_eff = c * math.log(1.0 + raw_w / c) * SOURCE_WEIGHTS['wide_frame']
            else:
                w_eff = raw_w * SOURCE_WEIGHTS.get(src, 1.0)
            weighted_sources[src] = round(w_eff, 4)
            t_total += w_eff
            if src in INDIRECT_PRIOR_SOURCES:
                t_prior += w_eff
            else:
                t_direct += w_eff
        totals[team] = t_total
        if t_direct > 0.0:
            direct_totals[team] = t_direct
        if t_prior > 0.0:
            prior_totals[team] = t_prior
        by_team[team] = {
            'total': round(t_total, 4),
            'direct': round(t_direct, 4),
            'prior': round(t_prior, 4),
            'sources': weighted_sources,
            'evidence_count': counts_by_team.get(team, 0),
        }

    return {'totals': totals, 'direct_totals': direct_totals, 'prior_totals': prior_totals, 'by_team': by_team}


def aggregate_tracklet_votes(tracklet, roster=None):
    """Aggregate weighted evidence and legacy votes on a single tracklet."""
    if not isinstance(tracklet, dict):
        tracklet = {'votes': getattr(tracklet, 'votes', None) or {}, 'evidence': getattr(tracklet, 'evidence', None) or []}
    ev_list = tracklet.get('evidence') or []
    legacy_votes = tracklet.get('votes') or {}
    raw_teams = list(legacy_votes.keys()) + [it.get('team') for it in ev_list if isinstance(it, dict)]
    canon_map = _canon_team_map(roster=roster, raw_teams=raw_teams)
    items, has_ev = _reconciled_tracklet_items(tracklet, canon_map, chain_has_structured_evidence=bool(ev_list))
    if not has_ev:
        return _tally_from_evidence_and_votes([], legacy_votes, canon_map, has_structured_evidence=False)
    return _tally_from_evidence_and_votes(items, None, canon_map, has_structured_evidence=True)


def _normalize_chain_priors(chains, chain_priors):
    by_chain = {i: [] for i in range(len(chains))}
    if not chain_priors:
        return by_chain
    if isinstance(chain_priors, dict):
        for k, val in chain_priors.items():
            try:
                idx = int(k)
            except (TypeError, ValueError):
                continue
            if idx in by_chain:
                if isinstance(val, dict):
                    by_chain[idx].append(val)
                elif isinstance(val, (list, tuple)):
                    by_chain[idx].extend(it for it in val if isinstance(it, dict))
        return by_chain
    if isinstance(chain_priors, (list, tuple)):
        if any(isinstance(it, dict) and 'chain_index' in it for it in chain_priors):
            for it in chain_priors:
                if isinstance(it, dict) and isinstance(it.get('chain_index'), int) and it['chain_index'] in by_chain:
                    by_chain[it['chain_index']].append(it)
            return by_chain
        for idx, entry in enumerate(chain_priors[:len(chains)]):
            if isinstance(entry, dict):
                by_chain[idx].append(entry)
            elif isinstance(entry, (list, tuple)):
                by_chain[idx].extend(it for it in entry if isinstance(it, dict))
    return by_chain


def _aggregate_chain_tallies(chains, tracklets, canon_map, priors_by_chain=None):
    chain_aggs = []
    for c_idx, chain in enumerate(chains):
        prior_items = _dedup_evidence_items((priors_by_chain or {}).get(c_idx, ()), canon_map)
        has_tr_ev = any(
            bool((tracklets[index].get('evidence') if isinstance(tracklets[index], dict) else getattr(tracklets[index], 'evidence', None)))
            for index in chain
        )
        chain_has_ev = bool(prior_items) or has_tr_ev
        if not chain_has_ev:
            c_votes = {}
            for index in chain:
                tr = tracklets[index]
                tr_votes = (tr.get('votes') if isinstance(tr, dict) else getattr(tr, 'votes', None)) or {}
                for tm, w in _valid_legacy_votes(tr_votes, canon_map).items():
                    c_votes[tm] = c_votes.get(tm, 0.0) + w
            chain_aggs.append(_tally_from_evidence_and_votes([], c_votes, canon_map, has_structured_evidence=False))
        else:
            reconciled = list(prior_items)
            for index in chain:
                tr_items, _ = _reconciled_tracklet_items(tracklets[index], canon_map, chain_has_structured_evidence=True)
                reconciled.extend(tr_items)
            chain_aggs.append(_tally_from_evidence_and_votes(reconciled, None, canon_map, has_structured_evidence=True))
    return chain_aggs


def _dominant_source_for_team(team_info, allowed_sources=None):
    if not team_info or not team_info.get('sources'):
        return None
    candidates = [
        (src, w)
        for src, w in team_info['sources'].items()
        if w > 0.0 and (allowed_sources is None or src in allowed_sources)
    ]
    if not candidates:
        return None
    # Prefer specific direct sources over generic wide_frame when tied
    return max(candidates, key=lambda kv: (kv[1], kv[0] in HIGH_TRUST_DIRECT_SOURCES, kv[0] != 'wide_frame'))[0]


def _build_chain_summary(agg, winning_team=None, decisive=None, primary_source=None):
    totals = agg['totals']
    by_team_raw = agg['by_team']
    total_weight = sum(totals.values())
    if winning_team is None and totals:
        winning_team = max(totals, key=totals.get)
    winning_weight = totals.get(winning_team, 0.0) if winning_team else 0.0
    share = (winning_weight / total_weight) if (winning_team and total_weight > 0.0) else 0.0

    other_teams = [(t, w) for t, w in totals.items() if t != winning_team and w > 0.0]
    if other_teams:
        runner_up_team, runner_up_weight = max(other_teams, key=lambda kv: kv[1])
    else:
        runner_up_team, runner_up_weight = None, 0.0

    margin = ((winning_weight - runner_up_weight) / total_weight) if (winning_team and total_weight > 0.0) else 0.0
    by_team = {}
    for team, info in by_team_raw.items():
        t_share = (totals.get(team, 0.0) / total_weight) if total_weight > 0.0 else 0.0
        by_team[team] = {
            'total': round(info['total'], 4),
            'direct': round(info['direct'], 4),
            'prior': round(info['prior'], 4),
            'share': round(t_share, 4),
            'sources': dict(info['sources']),
            'evidence_count': int(info['evidence_count']),
        }

    if primary_source is None and winning_team and winning_team in by_team:
        if agg['direct_totals'].get(winning_team, 0.0) > 0.0:
            primary_source = _dominant_source_for_team(by_team[winning_team], DIRECT_READ_SOURCES)
        else:
            primary_source = _dominant_source_for_team(by_team[winning_team], INDIRECT_PRIOR_SOURCES)

    if decisive is None:
        decisive = bool(
            winning_team
            and agg['direct_totals'].get(winning_team, 0.0) > 0.0
            and share >= 0.5
            and winning_weight > runner_up_weight
        )

    return {
        'winning_team': winning_team,
        'winning_weight': round(winning_weight, 4),
        'total_weight': round(total_weight, 4),
        'share': round(share, 4),
        'margin': round(margin, 4),
        'runner_up_team': runner_up_team,
        'runner_up_weight': round(runner_up_weight, 4),
        'decisive': bool(decisive),
        'primary_source': primary_source,
        'by_team': by_team,
        '_agg': agg,
    }


def _strip_private_summary(summary):
    return {k: v for k, v in summary.items() if not k.startswith('_')}


def combine_chain_evidence(chains, tracklets, roster=None, chain_priors=None):
    """Aggregate weighted evidence across all tracklets and priors for each chain."""
    priors_by_chain = _normalize_chain_priors(chains, chain_priors)
    raw_teams = _collect_raw_teams(tracklets, priors_by_chain)
    canon_map = _canon_team_map(roster=roster, raw_teams=raw_teams)
    chain_aggs = _aggregate_chain_tallies(chains, tracklets, canon_map, priors_by_chain=priors_by_chain)
    return [_strip_private_summary(_build_chain_summary(agg)) for agg in chain_aggs]


def _assign_from_tallies(chains, chain_aggs, roster):
    teams = list(roster)
    votes = [agg['totals'] for agg in chain_aggs]
    totals = [sum(v.values()) for v in votes]

    def has_direct_evidence(i, team):
        return chain_aggs[i]['direct_totals'].get(str(team), 0.0) > 0.0

    def has_high_trust(i, team):
        info = chain_aggs[i]['by_team'].get(str(team))
        if not info:
            return False
        ht_w = sum(w for src, w in info['sources'].items() if src in HIGH_TRUST_DIRECT_SOURCES)
        return ht_w > 0.0

    def valid_weight(i, team):
        if team is None or totals[i] <= 0 or not has_direct_evidence(i, team):
            return 0.0
        w = votes[i].get(str(team), 0.0)
        if w <= 0:
            return 0.0
        share = w / totals[i]
        if share >= 0.7:
            return w
        if has_high_trust(i, team) and share >= 0.55 and all(w > v for k, v in votes[i].items() if k != str(team)):
            return w
        return 0.0

    def secondary_weight(i, team, strong_teams):
        if team is None or totals[i] <= 0 or str(team) in strong_teams or not has_direct_evidence(i, team):
            return 0.0
        w = votes[i].get(str(team), 0.0)
        if w <= 0 or not all(w > v for k, v in votes[i].items() if k != str(team)):
            return 0.0
        share = w / totals[i]
        if share >= 0.5:
            return w
        rem = sum(v for k, v in votes[i].items() if k not in strong_teams)
        return w if share >= 0.4 and rem > 0 and w / rem >= 0.7 else 0.0

    best_score, best_assignment = (-1.0, -1.0), None
    padded = teams + [None] * max(0, len(chains) - len(teams))
    for perm in itertools.permutations(padded, len(chains)):
        strong = {str(team) for i, team in enumerate(perm) if valid_weight(i, team) > 0}
        t1 = sum(valid_weight(i, team) for i, team in enumerate(perm))
        t2 = sum(secondary_weight(i, team, strong) for i, team in enumerate(perm) if valid_weight(i, team) <= 0)
        if (t1, t2) > best_score:
            best_score, best_assignment = (t1, t2), perm
    out = []
    strong_final = {str(team) for i, team in enumerate(best_assignment or ()) if valid_weight(i, team) > 0}
    for i, team in enumerate(best_assignment or ([None] * len(chains))):
        total = totals[i]
        w = votes[i].get(str(team), 0.0) if team is not None else 0.0
        share = w / total if (team is not None and total > 0) else 0.0
        ok = total > 0 and w > 0 and (valid_weight(i, team) > 0 or secondary_weight(i, team, strong_final) > 0)
        out.append((team if ok else None, round(share, 2) if ok else 0.0))
    return out


def assign_teams(chains, tracklets, roster):
    """Choose the roster permutation that maximises summed vote weight. Returns
    ([(team or None, confidence)] per chain). Confidence is the winning team's share of the
    chain's votes (0 when it had none): a chain without supporting reads remains unidentified. The leftover roster
    number is never evidence."""
    raw_teams = _collect_raw_teams(tracklets)
    canon_map = _canon_team_map(roster=roster, raw_teams=raw_teams)
    chain_aggs = _aggregate_chain_tallies(chains, tracklets, canon_map)
    return _assign_from_tallies(chains, chain_aggs, roster)


def _complete_from_tallies(chains, chain_aggs, roster, assigned):
    teams = [str(t) for t in roster]
    canon = _canon_team_map(roster=roster)
    out = [(team, share, 'reads' if team else None) for team, share in assigned]
    used = {canon.get(str(team), str(team)) for team, _, _ in out if team}
    left = [t for t in teams if t not in used]
    open_chains = [i for i, (team, _, _) in enumerate(out) if not team and chains[i]]
    if not left or not open_chains:
        return out
    tallies = []
    for i in open_chains:
        tally = {k: v for k, v in chain_aggs[i]['totals'].items() if k in left and v > 0.0}
        tallies.append(tally)
    best, best_perm = None, None
    padded = left + [None] * max(0, len(open_chains) - len(left))
    for perm in itertools.permutations(padded, len(open_chains)):
        score = (sum(tallies[k].get(team, 0.0) for k, team in enumerate(perm) if team), sum(1 for team in perm if team))
        if best is None or score > best:
            best, best_perm = score, perm
    for k, team in enumerate(best_perm):
        if not team:
            continue
        i = open_chains[k]
        total = sum(tallies[k].values())
        weight = tallies[k].get(team, 0.0)
        if weight > 0:
            s_team = str(team)
            direct_w = chain_aggs[i]['direct_totals'].get(s_team, 0.0)
            prior_w = chain_aggs[i]['prior_totals'].get(s_team, 0.0)
            if direct_w == 0.0 and prior_w > 0.0:
                t_info = chain_aggs[i]['by_team'].get(s_team, {})
                dom_prior = _dominant_source_for_team(t_info, INDIRECT_PRIOR_SOURCES) or 'appearance'
                out[i] = (team, round(min(PRIOR_CONFIDENCE_CAP, weight / total), 2), dom_prior)
            else:
                out[i] = (team, round(min(INFERRED_CONFIDENCE_CAP, weight / total), 2), 'weak_reads')
        else:
            out[i] = (team, round(0.5 * INFERRED_CONFIDENCE_CAP if len(left) == 1 else 0.1, 2), 'elimination')
    return out


def complete_teams(chains, tracklets, roster, assigned):
    """Give every chain a team: `assigned` (assign_teams' output) keeps its names, and the unnamed chains take the
    alliance's remaining roster teams, by the most vote weight (even when the reads were not decisive), then by
    elimination. Returns [(team or None, confidence, source)] with source 'reads', 'weak_reads', 'elimination' or None.

    Every robot should carry a team number, but a filled-in name is a guess: its confidence is capped
    (INFERRED_CONFIDENCE_CAP) and the caller marks it for review. A chain stays unnamed only when the alliance has more
    chains than roster teams left."""
    raw_teams = _collect_raw_teams(tracklets)
    canon_map = _canon_team_map(roster=roster, raw_teams=raw_teams)
    chain_aggs = _aggregate_chain_tallies(chains, tracklets, canon_map)
    return _complete_from_tallies(chains, chain_aggs, roster, assigned)


def solve_chain_identities(chains, tracklets, roster, chain_priors=None):
    """Solve chain team assignments with weighted multi-source evidence and honesty caps.

    Returns (assigned, completed, evidence_summaries) where:
      - assigned: [(team or None, confidence)] per chain
      - completed: [(team or None, confidence, source)] per chain
      - evidence_summaries: [ChainIdentityEvidence] per chain
    """
    priors_by_chain = _normalize_chain_priors(chains, chain_priors)
    raw_teams = _collect_raw_teams(tracklets, priors_by_chain)
    canon_map = _canon_team_map(roster=roster, raw_teams=raw_teams)
    chain_aggs = _aggregate_chain_tallies(chains, tracklets, canon_map, priors_by_chain=priors_by_chain)

    raw_assigned = _assign_from_tallies(chains, chain_aggs, roster)
    raw_completed = _complete_from_tallies(chains, chain_aggs, roster, raw_assigned)

    assigned = []
    completed = []
    summaries = []
    for i, ((a_team, a_share), (c_team, c_conf, c_src), agg) in enumerate(zip(raw_assigned, raw_completed, chain_aggs)):
        if a_team is not None:
            s_team = str(a_team)
            direct_w = agg['direct_totals'].get(s_team, 0.0)
            t_info = agg['by_team'].get(s_team, {})
            if direct_w > 0.0:
                dom_direct = _dominant_source_for_team(t_info, DIRECT_READ_SOURCES) or 'reads'
                comp_src = 'reads' if dom_direct == 'wide_frame' else dom_direct
                conf = round(a_share, 2)
                assigned.append((a_team, conf))
                completed.append((a_team, conf, comp_src))
                summaries.append(_strip_private_summary(_build_chain_summary(
                    agg, winning_team=a_team, decisive=True, primary_source=dom_direct
                )))
            else:
                dom_prior = _dominant_source_for_team(t_info, INDIRECT_PRIOR_SOURCES) or 'appearance'
                conf = round(min(PRIOR_CONFIDENCE_CAP, a_share), 2)
                assigned.append((None, 0.0))
                completed.append((a_team, conf, dom_prior))
                summaries.append(_strip_private_summary(_build_chain_summary(
                    agg, winning_team=a_team, decisive=False, primary_source=dom_prior
                )))
        else:
            assigned.append((None, 0.0))
            if c_team is not None:
                s_c_team = str(c_team)
                direct_w = agg['direct_totals'].get(s_c_team, 0.0)
                prior_w = agg['prior_totals'].get(s_c_team, 0.0)
                t_info = agg['by_team'].get(s_c_team, {})
                if direct_w == 0.0 and prior_w > 0.0:
                    dom_prior = _dominant_source_for_team(t_info, INDIRECT_PRIOR_SOURCES) or 'appearance'
                    c_src = dom_prior
                    total_w = sum(agg['totals'].values())
                    c_conf = round(min(PRIOR_CONFIDENCE_CAP, (agg['totals'].get(s_c_team, 0.0) / total_w) if total_w > 0 else PRIOR_CONFIDENCE_CAP), 2)
                completed.append((c_team, c_conf, c_src))
                summaries.append(_strip_private_summary(_build_chain_summary(
                    agg, winning_team=c_team, decisive=False, primary_source=c_src
                )))
            else:
                completed.append((None, 0.0, None))
                summaries.append(_strip_private_summary(_build_chain_summary(
                    agg, winning_team=None, decisive=False, primary_source=None
                )))

    return assigned, completed, summaries
