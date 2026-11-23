"""Approximate raised-bump projection from image-observed vertical field landmarks.

The floor homography remains unchanged. A separate height column is fitted to
known structural heights; corrections are bounded and carry uncertainty.
"""
import numpy as np
from camera3d import camera_from_pixel_homography
from common import FIELD

BUMP_HEIGHT=.1654  # FIRST 2026 manual section 5.5 (6.513 inches).


def bump_height(x,y):
 for centre in (FIELD['hub']['x'],FIELD['length']-FIELD['hub']['x']):
  if abs(x-centre)<=.564 and .597<=abs(y-FIELD['hub']['y'])<=2.452:
   return BUMP_HEIGHT*max(0,1-abs(x-centre)/.564)
 return 0.


def fit_height_column(h_pixel_to_field,landmarks):
 if len(landmarks)<4:raise ValueError('At least four independent elevated landmarks required')
 g=np.linalg.inv(np.asarray(h_pixel_to_field,float));g/=g[2,2];matrix=[];rhs=[]
 for p in landmarks:
  x,y,z,u,v=[float(p[k]) for k in ['x','y','z','u','v']]
  if not np.isfinite([x,y,z,u,v]).all() or z<=.1:raise ValueError('Invalid elevated landmark')
  q=g@np.array([x,y,1.]);matrix.extend([[z,0,-u*z],[0,z,-v*z]]);rhs.extend([u*q[2]-q[0],v*q[2]-q[1]])
 column=np.linalg.lstsq(matrix,rhs,rcond=None)[0];errors=[]
 for p in landmarks:
  q=g@np.array([p['x'],p['y'],1.])+p['z']*column
  errors.append(float(np.linalg.norm(q[:2]/q[2]-[p['u'],p['v']])))
 rms=float(np.sqrt(np.mean(np.square(errors))))
 if rms>10 or max(errors)>15:raise ValueError('Elevated landmark model is not consistent enough for bounded correction')
 return {'height_column':column.tolist(),'floor_to_image':g.tolist(),'rms_px':rms,'max_px':max(errors),'approximate':True,'training_height_m':float(np.median([p['z'] for p in landmarks]))}


def _resolve_camera_center(camera,h_pixel_to_field,width=1280,height=720):
 if camera is None:
  try:cam=camera_from_pixel_homography(h_pixel_to_field,width,height)
  except Exception:cam=None
  if cam is not None and np.isfinite(cam.center).all() and 2.0<float(cam.center[2])<40.0:
   return np.asarray(cam.center,dtype=float)
  return None
 if hasattr(camera,'center'):
  center=np.asarray(camera.center,dtype=float)
 elif isinstance(camera,dict):
  raw=camera.get('camera_center',camera.get('center'))
  if raw is None:return None
  center=np.asarray(raw,dtype=float)
 else:
  center=np.asarray(camera,dtype=float)
 if center.shape==(3,) and np.isfinite(center).all() and float(center[2])>2.0:
  return center
 return None


def ray_terrain_contact(pixel,h_pixel_to_field,camera=None,*,width=1280,height=720,contact_z_m=0.0,apply_bump=True):
 """Project an image pixel onto the 2026 REBUILT terrain or bumper clearance height using a 3D camera ray.

 Given floor intersection base = H(pixel) at z = 0 and camera center C = (Cx, Cy, Cz) with Cz > 2,
 the ray through base intersects horizontal plane z >= 0 at:
   p(z) = base + (z / Cz) * (C_xy - base).
 """
 h=np.asarray(h_pixel_to_field,float);base=h@np.r_[pixel,1.];base=base[:2]/base[2]
 if not np.isfinite(base).all():return float(base[0]),float(base[1]),0.
 center=_resolve_camera_center(camera,h,width=width,height=height)
 if center is None or float(center[2])<=2.0:
  return float(base[0]),float(base[1]),0.
 extra_z=contact_z_m
 if isinstance(camera,dict) and extra_z==0.0 and 'contact_z_m' in camera:
  extra_z=camera['contact_z_m']
 z_contact=max(0.0,float(extra_z or 0.0))
 direction=(center[:2]-base)/float(center[2])
 z_bump=0.0
 if apply_bump:
  z_bump=float(bump_height(base[0],base[1]))
  for _ in range(3):
   pt=base+(z_bump+z_contact)*direction
   z_bump=float(bump_height(pt[0],pt[1]))
  p0=base+z_contact*direction
  p1=base+(BUMP_HEIGHT+z_contact)*direction
  x_min,x_max=min(float(p0[0]),float(p1[0])),max(float(p0[0]),float(p1[0]))
  hub_x=float(FIELD['hub']['x']);red_hub_x=float(FIELD['length'])-hub_x
  if (x_min<=hub_x+.564 and x_max>=hub_x-.564) or (x_min<=red_hub_x+.564 and x_max>=red_hub_x-.564):
   fn=lambda z:z-bump_height(*(base+(z+z_contact)*direction))
   grid=np.linspace(0.0,BUMP_HEIGHT,65)
   roots=[z_bump] if abs(fn(z_bump))<1e-5 else ([0.0] if abs(fn(0.0))<1e-9 else [])
   for a,b in zip(grid,grid[1:]):
    fa,fb=fn(a),fn(b)
    if fa*fb>0:continue
    lo,hi=a,b
    for _ in range(25):
     mid=.5*(lo+hi)
     if fa*fn(mid)<=0:hi=mid
     else:lo=mid;fa=fn(mid)
    roots.append(.5*(lo+hi))
   if roots:z_bump=float(max(roots))
  if z_bump<1e-5:z_bump=0.0
 z_total=z_bump+z_contact
 point=base+z_total*direction
 if not np.isfinite(point).all():return float(base[0]),float(base[1]),0.
 delta=point-base;dist=float(np.linalg.norm(delta))
 if dist>.6:point=base+delta*(.6/dist)
 return float(point[0]),float(point[1]),float(z_bump)


def terrain_contact(pixel,h_pixel_to_field,model):
 h=np.asarray(h_pixel_to_field,float);base=h@np.r_[pixel,1.];base=base[:2]/base[2]
 if not model:return float(base[0]),float(base[1]),0.
 if hasattr(model,'center') or (isinstance(model,dict) and ('camera_center' in model or 'center' in model) and 'height_column' not in model):
  return ray_terrain_contact(pixel,h_pixel_to_field,camera=model)
 g=np.asarray(model['floor_to_image']);column=np.asarray(model['height_column']);u,v=pixel
 matrix=np.array([[g[0,0]-u*g[2,0],g[0,1]-u*g[2,1]],[g[1,0]-v*g[2,0],g[1,1]-v*g[2,1]]])
 try:direction=np.linalg.solve(matrix,[u*column[2]-column[0],v*column[2]-column[1]])
 except np.linalg.LinAlgError:return float(base[0]),float(base[1]),0.
 fn=lambda z:z-bump_height(*(base+z*direction))
 grid=np.linspace(0,BUMP_HEIGHT,65);roots=[0.] if abs(fn(0))<1e-9 else []
 for a,b in zip(grid,grid[1:]):
  if fn(a)*fn(b)>0:continue
  for _ in range(25):
   mid=(a+b)/2
   if fn(a)*fn(mid)<=0:b=mid
   else:a=mid
  roots.append((a+b)/2)
 z=max(roots) if roots else 0.
 if z<1e-5:z=0.
 point=base+z*direction
 if not np.isfinite(point).all() or np.linalg.norm(point-base)>.6:return float(base[0]),float(base[1]),0.
 return float(point[0]),float(point[1]),float(z)
