# cube_gesture.py
from collections import deque
import time, math
import numpy as np
import cv2 as cv

import mediapipe as mp


class OneEuro:
    def __init__(self, min_cutoff=1.2, beta=0.3, d_cutoff=1.0):
        self.min_cut = float(min_cutoff); self.beta = float(beta); self.d_cut = float(d_cutoff)
        self.x_prev=None; self.dx_prev=0.0; self.y_prev=None
    def _alpha(self, cutoff, dt):
        r = 2.0*math.pi*cutoff*dt
        return r/(r+1.0) if r>0 else 1.0
    def filter(self, x, dt):
        if self.y_prev is None: self.y_prev=x; self.x_prev=x; return x
        dx=(x-self.x_prev)/max(dt,1e-6); self.x_prev=x
        a_d=self._alpha(self.d_cut,dt); dxh=a_d*dx+(1-a_d)*self.dx_prev; self.dx_prev=dxh
        a=self._alpha(self.min_cut+self.beta*abs(dxh),dt)
        y=a*x+(1-a)*self.y_prev; self.y_prev=y; return y

class OneEuro2D:
    def __init__(self, min_cutoff=1.2, beta=0.3, d_cutoff=1.0):
        self.fx=OneEuro(min_cutoff,beta,d_cutoff); self.fy=OneEuro(min_cutoff,beta,d_cutoff)
    def filter(self, xy, dt):
        x,y=xy; return np.array([self.fx.filter(x,dt), self.fy.filter(y,dt)], float)

MODEL_COMPLEXITY=0; MIN_DET_CONF=0.7; MIN_TRK_CONF=0.7

S_HISTORY=12; S_MIN=20.0; S_MAX=600.0
OPEN_ON=2.3


POINT_STRAIGHT_DEG=30.0
POINT_STREAK_ON=2

# Two-hand zoom (distance-based)
ZOOM_ENGAGE_FR=3
ZOOM_LP_ALPHA=0.35
ZOOM_GAIN=1.8          # positive=zoom in when hands move apart
ZOOM_DEAD=0.02

# Flicks (wrist/anchor velocity)
FLICK_TH=2.6
FLICK_MIN_GAP_S=0.25
VEL_WINDOW=6
FLICK_LATCH_S=0.25     # block pointer for a moment after a flick

# Discretes
CIRCLE_COOLDOWN_S=0.60; CIRCLE_AREA_TH=0.15
SWIPE_COOLDOWN_S=0.50; SWIPE_VX_THRESH=2.5; SWIPE_VY_MAX=1.0
DISC_LOCK_S=0.20
ACTION_LATCH_S=0.12

def _dist(a,b): return math.hypot(a[0]-b[0], a[1]-b[1])
def _angle(a,b,c):
    v1=a-b; v2=c-b; n1=np.linalg.norm(v1); n2=np.linalg.norm(v2)
    if n1<1e-6 or n2<1e-6: return 180.0
    cos=np.clip(np.dot(v1,v2)/(n1*n2), -1, 1)
    return math.degrees(math.acos(cos))

def _quantize8(dx, dy):
    # snap to 8 directions
    ang = math.atan2(dy, dx) 
    sector = round(ang / (math.pi/4))  # nearest of 8
    q = sector * (math.pi/4)
    return math.cos(q), math.sin(q)

class _Lock:
    def __init__(self,s): self.until=0.0; self.s=s
    def arm(self): self.until=time.time()+self.s
    def active(self): return time.time()<self.until
    def clear(self): self.until=0.0

class GestureInput:
    """
    Returns (single arbitrated action):
      {
        "fps": float,
        "auto_spin": bool,              # OFF when exactly 1 open hand
        "zoom_d": float,                # two-hand distance change based
        "point_active": bool,
        "point_px": (x,y),
        "open_count": int,
        "hand_vel": (vx,vy,speed),      # dominant wrist/center
        "flick": bool,
        "flick_dir": (dx,dy),           # 8-dir quantized
        "action": ("swipe"| "circle"| "zoom"| "point"| "flick", value) | None
      }
    """
    def __init__(self, device=0, width=960, height=540):
        self.cap=cv.VideoCapture(device)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH,width); self.cap.set(cv.CAP_PROP_FRAME_HEIGHT,height)
        self.hands=mp.solutions.hands.Hands(False,2,MODEL_COMPLEXITY,MIN_DET_CONF,MIN_TRK_CONF)
        self.width=width; self.height=height

        self.anchor_f=OneEuro2D(1.2,0.3,1.0)   
        self.tip_f   =OneEuro2D(1.2,0.3,1.0)
        self.dist_f  =OneEuro(min_cutoff=1.0, beta=0.2, d_cutoff=1.0) 

        self.prev_t=time.time(); self.fps_s=0.0
        self.s_hist=deque(maxlen=S_HISTORY)

        self.dom_hist=deque(maxlen=24)
        self.path_hist=deque(maxlen=24)

        
        self._prev_two_d = None
        self._zoom_lp    = 0.0
        self._two_streak = 0

        
        self.disc_lock=_Lock(DISC_LOCK_S)
        self.action_latch=_Lock(ACTION_LATCH_S)
        self.flick_block=_Lock(FLICK_LATCH_S)  # blocks pointer after flick
        self.current_action=None

        self._point_streak=0
        self._last_flick_t=0.0

    def close(self):
        self.hands.close(); self.cap.release()

    def _hand_pts_px(self, lm):
        return np.array([(p.x*self.width, p.y*self.height) for p in lm.landmark], float)
    def _palm_width(self, pts): return _dist(pts[5], pts[17]) + 1e-6
    def _hand_center(self, pts): return np.mean(pts[[0,5,9,13,17]], axis=0)

    def _open_and_point(self, pts, S):
        WR=0
        # openness (average tip-to-wrist)
        o = sum(_dist(pts[t], pts[WR]) for t in (8,12,16,20))/(4*S)

        # pointer detection (either straight+high OR index clearly farthest)
        idx_tip, idx_dip, idx_pip, idx_mcp = pts[8], pts[7], pts[6], pts[5]
        a1=_angle(idx_tip,idx_dip,idx_pip); a2=_angle(idx_dip,idx_pip,idx_mcp)
        straight = (a1<POINT_STRAIGHT_DEG and a2<POINT_STRAIGHT_DEG) and (idx_tip[1] < idx_mcp[1]-6)
        d_idx=_dist(pts[8], pts[WR]); d_mid=_dist(pts[12],pts[WR]); d_ring=_dist(pts[16],pts[WR]); d_pnk=_dist(pts[20],pts[WR])
        far_idx = d_idx > max(d_mid,d_ring,d_pnk) * 1.07
        return (o > OPEN_ON), (straight or far_idx), idx_tip

    def _vel(self, hist):
        n=len(hist)
        if n<4: return 0.0,0.0,0.0
        data=list(hist)
        seg=data[-VEL_WINDOW:] if n>=VEL_WINDOW else data
        dt=max(1e-6, seg[-1][0]-seg[0][0])
        vx=(seg[-1][1]-seg[0][1])/dt; vy=(seg[-1][2]-seg[0][2])/dt
        return vx,vy,math.hypot(vx,vy)

    def _swipe(self, now, vx, vy):
        if (now - getattr(self,'last_swipe_t',0.0)) < SWIPE_COOLDOWN_S: return None
        if abs(vx)>SWIPE_VX_THRESH and abs(vy)<SWIPE_VY_MAX:
            self.last_swipe_t=now
            return ("swipe", -1 if vx<0 else +1)
        return None

    def _circle(self, now):
        if (now - getattr(self,'last_circle_t',0.0)) < CIRCLE_COOLDOWN_S: return None
        if len(self.path_hist)<self.path_hist.maxlen//2: return None
        xs=np.array([x for _,x,_ in self.path_hist]); ys=np.array([y for _,_,y in self.path_hist])
        area=0.5*np.sum(xs*np.roll(ys,-1)-ys*np.roll(xs,-1))
        if abs(area)>CIRCLE_AREA_TH:
            self.last_circle_t=now
            return ("circle", area<0)
        return None

    def poll(self):
        t=time.time(); dt=max(1e-6, t-self.prev_t); self.prev_t=t
        self.fps_s=0.9*self.fps_s+0.1*(1.0/dt)

        ok, frame=self.cap.read()
        if not ok:
            return {"fps":self.fps_s,"auto_spin":True,"zoom_d":0.0,"point_active":False,"point_px":(0,0),
                    "open_count":0,"hand_vel":(0,0,0),"flick":False,"flick_dir":(0,0),"action":None}
        frame=cv.flip(frame,1)
        rgb=cv.cvtColor(frame, cv.COLOR_BGR2RGB); rgb.flags.writeable=False
        res=self.hands.process(rgb); rgb.flags.writeable=True

        zoom_d=0.0; point_active=False; point_px=(0.0,0.0)
        vx=vy=speed=0.0; flick=False; flick_dir=(0.0,0.0)
        action=None; open_hands=0
        self.action_latch.clear()

        hands=[]
        if res.multi_hand_landmarks:
            for lm in res.multi_hand_landmarks:
                pts=self._hand_pts_px(lm); S=self._palm_width(pts)
                self.s_hist.append(S)
                Smed=float(np.median(self.s_hist)) if self.s_hist else S
                Smed=max(S_MIN,min(S_MAX,Smed))
                is_open, is_point, tip = self._open_and_point(pts,Smed)
                center=self._hand_center(pts)
                wrist = pts[0]
                anchor = (center + wrist)/2.0
                hands.append(dict(S=Smed, pts=pts, open=is_open, point=is_point, tip=tip, center=center, anchor=anchor))

        open_hands = sum(1 for h in hands if h["open"])

       
        dom=None
        for h in hands:
            if h["open"]: dom=h; break
        if dom is None and hands:
            dom=max(hands, key=lambda r: r["S"])

        # Two-hand zoom (distance-based)
        if open_hands >= 2:
            # pick two largest S hands
            hs=sorted(hands, key=lambda r: -r["S"])[:2]
            p1=np.array(hs[0]["center"]); p2=np.array(hs[1]["center"])
            d=_dist(p1,p2)
            if self._prev_two_d is None:
                self._prev_two_d = d
                self._two_streak = 1
            else:
                self._two_streak = min(99, self._two_streak+1)
                dd = (d - self._prev_two_d) / max(dt,1e-6)  
                self._prev_two_d = d
                
                Smean=(hs[0]["S"]+hs[1]["S"])/2.0
                rate = dd / max(1.0, Smean)
                # Low-pass and scale
                z_raw = rate * ZOOM_GAIN
                self._zoom_lp = (1-ZOOM_LP_ALPHA)*getattr(self,'_zoom_lp',0.0) + ZOOM_LP_ALPHA*z_raw
                zoom_d = self._zoom_lp if self._two_streak >= ZOOM_ENGAGE_FR and abs(self._zoom_lp) > ZOOM_DEAD else 0.0
                if abs(zoom_d) > 0.0:
                    self.current_action="zoom"; self.action_latch.arm()
        else:
            self._prev_two_d = None
            self._two_streak = 0
            self._zoom_lp    = 0.0

        if dom:
            
            an = self.anchor_f.filter(dom["anchor"]/dom["S"], dt)
            self.dom_hist.append((t, an[0], an[1]))
            self.path_hist.append((t, an[0], an[1]))
            vx,vy,speed = self._vel(self.dom_hist)

            
            if not self.disc_lock.active() and self.current_action is None:
                sw=self._swipe(t, vx, vy)
                if sw:
                    action=sw; self.disc_lock.arm()
                else:
                    cir=self._circle(t)
                    if cir:
                        action=cir; self.disc_lock.arm()

            
            if action is None and self.current_action is None and open_hands==1:
                if (t-self._last_flick_t)>FLICK_MIN_GAP_S and speed>FLICK_TH:
                    self._last_flick_t=t
                    qx,qy=_quantize8(vx,vy)
                    flick=True; flick_dir=(qx,qy)
                    action=("flick", flick_dir)
                    # briefly block pointer so drag doesn't steal
                    self.flick_block.arm()

            
            if action is None and self.current_action is None and not self.flick_block.active() and open_hands<=1:
                if dom["point"]: self._point_streak=min(99, self._point_streak+1)
                else: self._point_streak=0
                if self._point_streak>=POINT_STREAK_ON:
                    tipf=self.tip_f.filter(dom["tip"]/dom["S"], dt)
                    px,py=(tipf[0]*dom["S"], tipf[1]*dom["S"])
                    point_active=True; point_px=(float(px), float(py))
                    action=("point", point_px)
                    self.current_action="point"; self.action_latch.arm()

        
        if action is None and abs(zoom_d)>0.0:
            action=("zoom", float(zoom_d))
            self.current_action="zoom"; self.action_latch.arm()

       
        if not self.action_latch.active():
            self.current_action=None

        auto_spin = not (open_hands == 1)

        return {
            "fps": self.fps_s,
            "auto_spin": bool(auto_spin),
            "zoom_d": float(zoom_d),
            "point_active": bool(point_active),
            "point_px": (float(point_px[0]), float(point_px[1])),
            "open_count": int(open_hands),
            "hand_vel": (float(vx), float(vy), float(speed)),
            "flick": bool(flick),
            "flick_dir": (float(flick_dir[0]), float(flick_dir[1])),
            "action": action
        }
