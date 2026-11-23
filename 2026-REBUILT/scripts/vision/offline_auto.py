#!/usr/bin/env python3
"""Calibrate cached source detections into an offline, review-required Auto run.

Inputs contain video detections, roster-only metadata, bumper reads and manual
field geometry. No website client, network calls or score-reference inputs.
"""
import argparse
import hashlib
import json
import math
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

import numpy as np
from project import ground_point,in_field,infer_start_pose,project_contact
from smoothing import smooth_track
from common import FIELD


def inside_hub_core(x,y):
    half=FIELD['hub']['halfSize']-.15
    return any(abs(x-cx)<half and abs(y-cy)<half for cx,cy in [(FIELD['hub']['x'],FIELD['hub']['y']),(FIELD['length']-FIELD['hub']['x'],FIELD['width']-FIELD['hub']['y'])])


def select_priority_evidence(points,radius_ms=150):
    """Use a primary source's cadence; fallback points fill only missing spans.

    Weaving two independently detected box series at alternate timestamps adds
    apparent motion from their differing box extents. Keep primary observations
    where available and preserve observed fallback evidence through its gaps.
    No times or positions are invented.
    """
    times=defaultdict(list)
    for p in points:times[p.get('source_priority',0)].append(p['t'])
    for rank in times:times[rank].sort()
    out=[]
    for p in points:
        preferred=False
        for rank,ts in times.items():
            if rank<=p.get('source_priority',0):continue
            i=bisect_left(ts,p['t'])
            if any(abs(t-p['t'])<=radius_ms for t in ts[max(0,i-1):i+1]):preferred=True;break
        if not preferred:out.append(p)
    return out


def quarantine_identity_collisions(known,unknown):
    """A team cannot occupy two distant positions simultaneously.

    Keep a clearly stronger continuous source; quarantine the weaker fragment.
    With comparable evidence both identities become unknown. Never assign the
    leftover roster number to the quarantined robot.
    """
    count=0
    strength=lambda row:max(.1,(row['ended_ms']-row['started_ms'])/1000)*row['tracking_confidence']*row['identity_confidence']
    for key,rows in known.items():
        rows.sort(key=strength,reverse=True);rejected=set()
        for i,a in enumerate(rows):
            for b in rows[i+1:]:
                if id(a) in rejected or id(b) in rejected or not a['trajectory'] or not b['trajectory']:continue
                distances=[];same_time=[]
                for p in a['trajectory']:
                    q=min(b['trajectory'],key=lambda q:abs(q['t']-p['t']))
                    dt=abs(q['t']-p['t'])
                    if dt<=200:
                        d=math.hypot(q['x']-p['x'],q['y']-p['y'])
                        distances.append(d)
                        if dt<=50:same_time.append(d)
                if not ((len(distances)>=3 and float(np.median(distances))>1) or (same_time and max(same_time)>2.2)):continue
                losers=[b] if strength(a)>=2*strength(b) else [a,b]
                for row in losers:
                    rejected.add(id(row));count+=1
                    unknown.append({**row,'team_key':None,'identity_confidence':0,'metrics':{**row['metrics'],'identity_collision':True,'identity_collision_note':'Competing same-team evidence occupies a different position at the same time; identity remains unknown.'}})
        known[key]=[row for row in rows if id(row) not in rejected]
    return count


def smooth_observed_points(points,max_gap_ms=400):
    """Denoise observed positions only; never bridge a gap or add predictions.

    Raw locations remain attached for source auditing. Calibration fit error
    alone does not account for changing axis-aligned detector boxes.
    """
    chunks=[]
    for point in sorted(points,key=lambda p:p['t']):
        if not chunks or point['t']-chunks[-1][-1]['t']>max_gap_ms:
            chunks.append([])
        chunks[-1].append(point)
    out=[]
    for chunk in chunks:
        if len(chunk)<3:
            out.extend({**p,'raw_x':p.get('raw_x',p['x']),'raw_y':p.get('raw_y',p['y'])} for p in chunk);continue
        result=smooth_track([p['t']/1000 for p in chunk],[p['x'] for p in chunk],[p['y'] for p in chunk],[max(.15,p['sigma']) for p in chunk])
        for i,p in enumerate(chunk):
            if result['kept'][i]:
                out.append({**p,'raw_x':p.get('raw_x',p['x']),'raw_y':p.get('raw_y',p['y']),'x':round(float(result['x'][i]),3),'y':round(float(result['y'][i]),3),'sigma':round(max(p['sigma'],float(result['sigma'][i])),3)})
    return out


def identity_decision(reads,roster):
    valid=[];frame_teams=defaultdict(set);digest_teams=defaultdict(set)
    for row in reads:
        team=row.get('team_key');confidence=row.get('confidence');frame=row.get('frame_index');digest=row.get('crop_sha256')
        if team not in roster or not isinstance(confidence,(int,float)) or isinstance(confidence,bool) or not math.isfinite(confidence) or not .65<=confidence<=1:continue
        if not isinstance(frame,int) or isinstance(frame,bool) or frame<0 or not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):continue
        valid.append((team,float(confidence),frame,digest))
        frame_teams[frame].add(team);digest_teams[digest].add(team)
    seen_frames=set();seen_images=set();votes=defaultdict(list)
    for team,confidence,frame,digest in valid:
        if len(frame_teams[frame])>1 or len(digest_teams[digest])>1:continue
        if frame in seen_frames or digest in seen_images:continue
        seen_frames.add(frame);seen_images.add(digest);votes[team].append(confidence)
    total=sum(sum(v) for v in votes.values())
    if not total:return None,0
    team=max(votes,key=lambda t:(sum(votes[t]),len(votes[t])));share=sum(votes[team])/total
    other_max=max((len(v) for t,v in votes.items() if t!=team),default=0)
    # A credible competing read can be a real track switch. Majority voting
    # must not label both robots as the winner across the entire fragment.
    if len(votes[team])<3 or share<.7:return None,0
    if len(votes)>1 and (len(votes[team])<6 or share<.85 or len(votes[team])<5*other_max):return None,0
    return team,share*sum(votes[team])/len(votes[team])


def remap_cached_reads(source,cached_source,cached_identities,max_delta_ms=50,min_iou=.85,max_read_ms=20000):
    """Reuse source-image bumper evidence only at a unique matching pixel box.

    Track IDs are not stable across detector passes. Never copy an identity by
    track number or by proximity alone, or use a full-match track-wide label.
    Each source crop must match its original observation and exactly one new
    observation at the same instant. Original frame/digest provenance stays.
    """
    if not 20000<=max_read_ms<=30000:raise ValueError('cached identity window must be bounded')
    if source.get('key')!=cached_source.get('key'):return []
    original={tr['track_key']:tr for tr in cached_source['tracks']}
    reads=[]
    for read in cached_identities.get('reads',[]):
        t=read.get('timestamp_ms');old=original.get(read.get('track_key'))
        if not isinstance(t,(int,float)) or isinstance(t,bool) or not 0<=t<=max_read_ms or not old:continue
        points=[p for p in old.get('trajectory',[]) if p.get('box') and abs(p['t']-t)<=1]
        if len(points)!=1:continue
        a=points[0]['box'];area_a=max(0,a[2]-a[0])*max(0,a[3]-a[1]);matches=[]
        for track in source['tracks']:
            if track.get('camera_epoch',0)!=old.get('camera_epoch',0):continue
            nearby=[p for p in track.get('trajectory',[]) if p.get('box') and abs(p['t']-t)<=max_delta_ms]
            if not nearby:continue
            point=min(nearby,key=lambda p:abs(p['t']-t));b=point['box']
            intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
            union=area_a+max(0,b[2]-b[0])*max(0,b[3]-b[1])-intersection
            if union>0 and intersection/union>=min_iou:matches.append(track['track_key'])
        if len(matches)==1:
            reads.append({**read,'track_key':matches[0],'cached_source_track_key':read['track_key'],'identity_source':'source-crop-geometric-remap'})
    return reads


def lookahead_read_is_continuous(track,read,matrix,auto_end_ms,inset_m=0.0):
    t=read['timestamp_ms'];raw=sorted(track.get('trajectory',[]),key=lambda p:p['t'])
    before=[p for p in raw if 0<=p['t']<=auto_end_ms and p.get('box')]
    if not before:return False
    span=[p for p in raw if before[-1]['t']<=p['t']<=t+50 and p.get('box')]
    if not span or abs(span[-1]['t']-t)>50 or any(b['t']-a['t']>400 for a,b in zip(span,span[1:])):return False
    projected=[ground_point(p['box'],matrix,inset_m=inset_m) for p in span]
    if any(not all(math.isfinite(v) for v in p) or not in_field(p[0],p[1],margin=.2) or inside_hub_core(p[0],p[1]) for p in projected):return False
    max_speed=FIELD['robot']['maxPlausibleSpeedMps']*1.5
    return all(math.hypot(b[0]-a[0],b[1]-a[1])<=max_speed*(span[i+1]['t']-span[i]['t'])/1000+.4 for i,(a,b) in enumerate(zip(projected,projected[1:])))


def build_offline_auto(source,identities,calibration,width=1920,height=1080,auto_end_ms=20000,identity_end_ms=None,*,footprint_inset_m=0.0):
    identity_end_ms=auto_end_ms if identity_end_ms is None else identity_end_ms
    if not auto_end_ms<=identity_end_ms<=auto_end_ms+10000:raise ValueError('identity lookahead must be bounded to 10 seconds')
    matrix=np.asarray(calibration['homography_norm'],dtype=float) @ np.diag([1/width,1/height,1])
    if matrix.shape!=(3,3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix))<1e-12:raise ValueError('invalid field calibration')
    if not isinstance(calibration.get('rms_m'),(int,float)) or isinstance(calibration.get('rms_m'),bool) or not 0<=calibration['rms_m']<=.3:raise ValueError('calibration fit must be reviewed and within 0.3m')
    inset_m=float(footprint_inset_m if footprint_inset_m>0 else calibration.get('footprint_inset_m',0.0) or 0.0)
    roi=calibration['roi'];groups=defaultdict(list)
    for read in identities.get('reads',[]):groups[read['track_key']].append(read)
    roster_map={k:v for k,v in source.get('team_roster',{}).items() if isinstance(v,list)}
    all_roster=sum(roster_map.values(),[])
    known=defaultdict(list);unknown=[];rejected=0;obstacle_rejected=0
    for track in source['tracks']:
        key=track['track_key'];alliance=track.get('alliance')
        auto_reads=[r for r in groups[key] if isinstance(r.get('timestamp_ms'),(int,float)) and not isinstance(r.get('timestamp_ms'),bool) and 0<=r['timestamp_ms']<=identity_end_ms and (r['timestamp_ms']<=auto_end_ms or lookahead_read_is_continuous(track,r,matrix,auto_end_ms,inset_m=inset_m))]
        team,confidence=identity_decision(auto_reads,roster_map.get(alliance,all_roster))
        if team:alliance=next(a for a,roster in roster_map.items() if team in roster)
        points=[];pre_roll=[]
        for p in sorted(track.get('trajectory',[]),key=lambda p:p['t']):
            if not -1000<=p['t']<=auto_end_ms or not p.get('box'):continue
            x1,y1,x2,y2=p['box'];cx=(x1+x2)/2
            if not roi[0]<=cx/width<=roi[2] or not roi[1]<=y2/height<=roi[3]:continue
            contact=p.get('ground_contact_px') or [(x1+x2)/2,y2]
            if p.get('ground_contact_visible') is False:continue
            x,y,sigma=project_contact(contact,matrix) if (p.get('ground_contact_px') or inset_m<=0) else ground_point([x1,y1,x2,y2],matrix,inset_m=inset_m)
            if not all(math.isfinite(v) for v in [x,y,sigma]) or not in_field(x,y,margin=.2):
                if p['t']>=0:rejected+=1
                continue
            if inside_hub_core(x,y):
                if p['t']>=0:obstacle_rejected+=1
                continue
            item={'t':p['t'],'x':round(x,3),'y':round(y,3),'heading':None,'calibrated':True,'sigma':round(max(sigma,calibration['rms_m']),3),'confidence':track.get('tracking_confidence',.5),'box':[round(v,4) for v in [x1/width,y1/height,x2/width,y2/height]],'source_track_key':key,'source_priority':track.get('source_priority',0),'camera_epoch':track.get('camera_epoch',0),'position_reference':'floor-contact','ground_contact_source':'supplied' if p.get('ground_contact_px') else 'box-bottom-estimate'}
            if p['t']<0:pre_roll.append(item)
            else:points.append(item)
        if pre_roll and not any(0<=p['t']<=250 for p in points):
            points.insert(0,{**pre_roll[-1],'t':0,'pre_roll_anchored':True})
        points=smooth_observed_points(points)
        if len(points)<10 or points[-1]['t']-points[0]['t']<500:continue
        start_pose=infer_start_pose(points,alliance) if alliance in ('red','blue') else None
        start_zone=start_pose['position'] if start_pose and start_pose.get('inAllianceZone') else None
        row={'team_key':team,'alliance':alliance,'auto_start_zone':start_zone,'identity_confidence':confidence,'tracking_confidence':track.get('tracking_confidence',.5),'trajectory':points,'started_ms':points[0]['t'],'ended_ms':points[-1]['t'],'needs_review':True,'metrics':{'source_track_keys':[key],'source_camera_epoch':track.get('camera_epoch',0),'offline_auto':True,'calibration_fit_rms_m':calibration['rms_m'],'autoStartZone':start_zone}}
        if team:known[(team,track.get('camera_epoch',0))].append(row)
        else:unknown.append(row)
    collisions=quarantine_identity_collisions(known,unknown)
    tracks=[]
    for (team,epoch),rows in known.items():
        if not rows:continue
        at_time=defaultdict(list)
        for row in rows:
            for point in row['trajectory']:at_time[point['t']].append(point)
        points=[];conflicts=0
        for t,candidates in sorted(at_time.items()):
            if any(math.hypot(a['x']-b['x'],a['y']-b['y'])>.8 for a in candidates for b in candidates):conflicts+=1;continue
            points.append(max(candidates,key=lambda p:(p.get('source_priority',0),p['confidence'])))
        points=select_priority_evidence(points)
        if len(rows)>1:points=smooth_observed_points(points)
        if len(points)<10:continue
        merged_alliance=rows[0].get('alliance')
        merged_pose=infer_start_pose(points,merged_alliance) if merged_alliance in ('red','blue') else None
        merged_zone=merged_pose['position'] if merged_pose and merged_pose.get('inAllianceZone') else rows[0].get('auto_start_zone')
        tracks.append({**rows[0],'auto_start_zone':merged_zone,'trajectory':points,'started_ms':points[0]['t'],'ended_ms':points[-1]['t'],'identity_confidence':min(r['identity_confidence'] for r in rows),'metrics':{'source_track_keys':[k for r in rows for k in r['metrics']['source_track_keys']],'source_camera_epoch':epoch,'offline_auto':True,'identity_overlap_conflicts':conflicts,'calibration_fit_rms_m':calibration['rms_m'],'complete_auto_verified':False,'autoStartZone':merged_zone}})
    tracks+=unknown
    # Only source-linked attempts; these are not scores and remain unreviewed.
    owners={k:r for r in tracks for k in r['metrics']['source_track_keys']}
    observations=[]
    for event in source.get('observations',[]):
        if event.get('observation_type')!='fuel_attempt' or not 0<=event['started_ms']<=auto_end_ms:continue
        owner=owners.get(event.get('track_key'));observations.append({**event,'team_key':owner.get('team_key') if owner else None,'review_status':'unreviewed'})
    return {'match':source['key'],'config':{'kind':'auto','engine':'offline-source-replay','auto_end_ms':auto_end_ms,'identity_evidence_end_ms':identity_end_ms,'auto_only':True,'frame':'field-meters','offline_only':True,'identity_mode':'bumper-reader','position_reference':'floor-contact','automatic_website_updates':False,'references_available_to_inference':False},'tracks':tracks,'observations':observations,'summary':{'known_robot_paths':sum(bool(t.get('team_key')) for t in tracks),'unknown_track_fragments':len(unknown),'projected_points':sum(len(t['trajectory']) for t in tracks),'start_positions_at_t0':sum(1 for t in tracks if t['trajectory'] and t['trajectory'][0]['t']<=1000),'identity_collision_fragments':collisions,'out_of_field_points_rejected':rejected,'static_hub_points_rejected':obstacle_rejected,'calibration_fit_rms_m':calibration['rms_m'],'accuracy_verified':False}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['source','identities','calibration','out']:parser.add_argument('--'+name,required=True)
    args=parser.parse_args();paths={n:Path(getattr(args,n)) for n in ['source','identities','calibration']}
    data={n:json.loads(p.read_text()) for n,p in paths.items()}
    result=build_offline_auto(data['source'],data['identities'],data['calibration'])
    result['config']['input_sha256']={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in paths.items()}
    Path(args.out).write_text(json.dumps(result));print(json.dumps(result['summary']))

if __name__=='__main__':main()
