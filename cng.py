# main.py
import os, time, collections
import pygame as pg
from cube_render import CubeRenderer
from cube_gesture import GestureInput
from audio_input import AudioMeter   

TRAIL_FADE_S = 1.5


def main():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    pg.init()
    W,H = 800,600
    screen = pg.display.set_mode((W,H))
    pg.display.set_caption("Cubey • audio-reactive edges")
    clock = pg.time.Clock()

    cube = CubeRenderer(W,H,scale=100)
    g    = GestureInput(device=0, width=960, height=540)
    meter= AudioMeter()  

    trail = collections.deque()
    t0 = time.time()

    running=True
    try:
        while running:
            now = time.time()
            t_sec = now - t0
            dt = clock.tick(60)/1000.0

            for e in pg.event.get():
                if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE):
                    running = False

            sig = g.poll()
            act = sig["action"]

            # Single action
            if act is not None:
                kind, val = act
                if kind == "swipe":
                    cube.snap_turn(val)
                elif kind == "circle":
                    cube.roll_snap(cw=val)
                elif kind == "point":
                    px,py = val
                    px = max(0, min(W, px)); py = max(0, min(H, py))
                    cube.set_center_px(px, py)
                elif kind == "flick":
                    dx,dy = val
                    cube.impulse_from_hand(dx, dy, flick=True)

            # Pointer trail recording
            if sig["point_active"]:
                px,py = sig["point_px"]
                px = max(0, min(W, px)); py = max(0, min(H, py))
                trail.append((now, px, py))
            # expire old points
            while trail and (now - trail[0][0]) > TRAIL_FADE_S:
                trail.popleft()

            # Update cube (zoom & auto/manual)
            cube.update(
                zoom_d=sig["zoom_d"],
                auto_spin=sig["auto_spin"],
                
                
                dt=dt,
            )

            
            audio_data = {"level": meter.level, "bands": meter.bands} if meter and meter._ok else None

            # Render cube with audio-reactive squiggles (isn't working right now )
            cube.render(
                screen,
                hud_lines=[
                    f"Render {clock.get_fps():4.1f}  Gesture {sig['fps']:4.1f}",
                    f"OpenHands: {sig['open_count']}  AutoSpin: {sig['auto_spin']}",
                    f"Action: {sig['action'][0] if sig['action'] else '—'}  Zoom_d: {sig['zoom_d']:+.2f}",
                    f"Audio lvl: {(meter.level if meter and meter._ok else 0):.2f}",
                    f"Audio lvl: {(meter.level if meter and meter._ok else 0):.2f}  Source: {(meter.source if meter else 'none')}",
                    f"Audio lvl: {(meter.level if meter and meter._ok else 0):.2f}  Src: {getattr(meter,'source','none')}  Dev: {getattr(meter,'device','n/a')}"
                    f"Audio lvl: {meter.level:.2f}  Src: {meter.source}  Dev: {meter.device}"
,


                ],
                audio=audio_data,
                t_sec=t_sec,
            )

            # Draw pointer trail 
            if trail:
                for t0p, x, y in trail:
                    a = max(0.0, 1.0 - (now - t0p)/TRAIL_FADE_S)
                    radius = max(2, int(2 + 6 * a))
                    color = (0, int(255*a), int(255*a))
                    pg.draw.circle(screen, color, (int(x), int(y)), radius)

            pg.display.flip()

    finally:
        g.close()
        if meter: meter.close()
        pg.quit()

if __name__ == "__main__":
    main()
