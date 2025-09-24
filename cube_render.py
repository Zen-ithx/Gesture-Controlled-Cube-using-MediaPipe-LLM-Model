# cube_render.py
import math
import numpy as np
import pygame as pg

BLACK=(0,0,0); RED=(255,0,0); MAGENTA=(255,0,255); WHITE=(255,255,255)

class CubeRenderer:
    def __init__(self, width=800, height=600, scale=100):
        # viewport & camera
        self.width, self.height = width, height
        self.center = np.array([width/2, height/2], dtype=float)
        self.scale  = float(scale)
        self.zoom   = 1.0

       
        self.angle_x = 0.0  # pitch
        self.angle_y = 0.0  # yaw
        self.angle_z = 0.0  # roll

        
        self.omega_x = 0.0
        self.omega_y = 0.0
        self.omega_z = 0.0

        # auto-rotation 
        self.auto_spin   = True
        self.auto_speed_x = math.radians(12)
        self.auto_speed_y = math.radians(22)
        self.auto_speed_z = math.radians(9)

        # dynamics
        self.damping     = 2.0                 # per-second decay for omega_*
        self.throw_gain  = math.radians(220)   # impulse magnitude from gestures
        self.stationary_thresh = 0.15

        # flick continuation (keeps spinning after a flick)
        self.flick_hold_s   = 0.9
        self._flick_t_left  = 0.0
        self._flick_omega_x = 0.0
        self._flick_omega_y = 0.0
        self._flick_omega_z = 0.0

        
        self.points = [np.array(p, float) for p in
            [(-1,-1, 1),( 1,-1, 1),( 1, 1, 1),(-1, 1, 1),
             (-1,-1,-1),( 1,-1,-1),( 1, 1,-1),(-1, 1,-1)]]
        self.edges = [(0,1),(1,2),(2,3),(3,0),
                      (4,5),(5,6),(6,7),(7,4),
                      (0,4),(1,5),(2,6),(3,7)]

       
        self.wave_hist_len = 160                 # ring buffer length (frames)
        self.wave_hist     = np.zeros((8, self.wave_hist_len), float)
        self.wave_write    = 0                   # write cursor

    
    def reset_orientation(self):
        self.angle_x = self.angle_y = self.angle_z = 0.0
        self.omega_x = self.omega_y = self.omega_z = 0.0
        self.zoom    = 1.0
        self._flick_t_left = 0.0

    def set_center_px(self, x, y):
        self.center[:] = (float(x), float(y))

    def snap_turn(self, direction):
        """±1 -> 90° yaw snap"""
        self.angle_y += direction * (math.pi/2)
        self.omega_x = self.omega_y = self.omega_z = 0.0
        self._flick_t_left = 0.0

    def roll_snap(self, cw=True):
        self.angle_z += math.radians(15) * (1 if cw else -1)

    def impulse_from_hand(self, dir_x, dir_y, flick=False):
        """
        Apply an angular-velocity impulse from a 2D direction.
        +x (right) -> yaw+, +y (down) -> pitch- (invert y).
        """
        speed = (dir_x**2 + dir_y**2) ** 0.5
        if speed < self.stationary_thresh:
            return
        gain = self.throw_gain * (1.5 if flick else 1.0)
        self.omega_y += gain * dir_x   
        self.omega_x += -gain * dir_y  

        if flick:
           
            self._flick_omega_x = self.omega_x
            self._flick_omega_y = self.omega_y
            self._flick_omega_z = 0.0
            self._flick_t_left  = self.flick_hold_s

    def update(self, zoom_d=0.0, auto_spin=True, dt=0.0):
        self.auto_spin = bool(auto_spin)

        
        self.angle_x += self.omega_x * dt
        self.angle_y += self.omega_y * dt
        self.angle_z += self.omega_z * dt

        # auto spin or flick hold
        if self._flick_t_left > 0.0:
            self._flick_t_left = max(0.0, self._flick_t_left - dt)
            self.angle_x += self._flick_omega_x * dt
            self.angle_y += self._flick_omega_y * dt
            self.angle_z += self._flick_omega_z * dt
        elif self.auto_spin:
            self.angle_x += self.auto_speed_x * dt
            self.angle_y += self.auto_speed_y * dt
            self.angle_z += self.auto_speed_z * dt

        # damping
        if dt > 0:
            damp = math.exp(-self.damping * dt)
            self.omega_x *= damp
            self.omega_y *= damp
            self.omega_z *= damp

        # zoom
        self.zoom = max(0.4, min(3.0, self.zoom + float(zoom_d) * dt))

   
    def _rot(self):
        cx,sx = math.cos(self.angle_x), math.sin(self.angle_x)
        cy,sy = math.cos(self.angle_y), math.sin(self.angle_y)
        cz,sz = math.cos(self.angle_z), math.sin(self.angle_z)
        Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]], float)
        Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]], float)
        Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]], float)
        return Rz @ Ry @ Rx

    def _draw_wave_line(self, screen, a, b, samples, amp_px, color=MAGENTA, width=2):
        """
        Draw a scrolling waveform (samples in [-1..+1]) along the line a->b
        using perpendicular offsets scaled by amp_px.
        """
        ax, ay = a; bx, by = b
        dx = bx - ax; dy = by - ay
        L = max(1e-6, (dx*dx + dy*dy)**0.5)
        ux, uy = dx / L, dy / L                   
        px, py = -uy, ux                           
        n = len(samples)
        if n < 2:
            pg.draw.line(screen, color, a, b, width); return
        pts = []
        for i, s in enumerate(samples):
            t = i / (n - 1)
            off = amp_px * float(s)
            x = ax + ux * (t * L) + px * off
            y = ay + uy * (t * L) + py * off
            pts.append((int(x), int(y)))
        pg.draw.lines(screen, color, False, pts, width)

    
    def render(self, screen, hud_lines=(), audio=None, t_sec=0.0):
        """
        Draw the cube.
        - audio: None or dict with keys 'level' (0..1) and 'bands' (len>=8, 0..1)
        - t_sec: seconds since start (not required for this renderer, but kept for API)
        """
        screen.fill(BLACK)
        R = self._rot()

        # project vertices
        projected=[]
        for p in self.points:
            pr = R @ p
            x = int(pr[0]*self.scale*self.zoom + self.center[0])
            y = int(pr[1]*self.scale*self.zoom + self.center[1])
            projected.append((x,y))
            pg.draw.circle(screen, RED, (x,y), 5)

        # audio snapshot 
        bands = None
        level = 0.0
        if isinstance(audio, dict):
            b = audio.get("bands", None)
            if b is not None:
                b = np.asarray(b, float).reshape(-1)
                if b.size >= 8:
                    bands = b[:8]
            level = float(audio.get("level", 0.0))

        if bands is not None:
            
            centered = np.clip((bands - 0.5) * 2.0, -1.0, 1.0)
            self.wave_hist[:, self.wave_write] = centered
            self.wave_write = (self.wave_write + 1) % self.wave_hist_len

        # draw edges
        for ei, (a_idx, b_idx) in enumerate(self.edges):
            a = projected[a_idx]; b = projected[b_idx]
            if bands is None:
                pg.draw.line(screen, MAGENTA, a, b, 2)
                continue

           
            bi = ei % 8

            
            segs = 22
            stride = max(1, self.wave_hist_len // segs)
            idxs = (self.wave_write - np.arange(segs) * stride) % self.wave_hist_len
            samples = self.wave_hist[bi, idxs][::-1]

            # amplitude scaled by band & overall level
            amp_px = self.zoom * (4.0 + 18.0 * float(np.clip(bands[bi], 0, 1))) * (0.6 + 0.8*level)

            self._draw_wave_line(screen, a, b, samples, amp_px, color=MAGENTA, width=2)

        # HUD
        if hud_lines:
            font = pg.font.SysFont("Arial", 18)
            y=8
            for line in hud_lines:
                screen.blit(font.render(line, True, WHITE), (10,y))
                y += 20
