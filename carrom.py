"""
=============================================================
  LIVE CARROM — Finger Camera Control  (MediaPipe Tasks v0.10+)
=============================================================
  pip install opencv-python mediapipe pygame numpy --break-system-packages
  python carrom.py [--players 2|3|4] [--camera 0]

  Controls (camera):
    ✋  Move index finger L/R  →  slide striker on baseline
    ↕   Tilt finger up/down   →  aim direction (arrow follows)
    ⬆   Raise whole hand      →  more power
    🤏  PINCH thumb+index     →  SHOOT  (aim locks at pinch moment)
=============================================================
"""

import cv2, mediapipe as mp, pygame, sys, math, argparse, os
import urllib.request, threading, time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# ── Model download ────────────────────────────────────────────
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "hand_landmarker.task")

def ensure_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 100_000:
        return
    print("[INFO] Downloading hand landmark model (~1 MB) ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"[INFO] Saved → {MODEL_PATH}")
    except Exception as e:
        print(f"[ERROR] Download failed: {e}\n  URL: {MODEL_URL}")
        sys.exit(1)

# ── Camera auto-detect ────────────────────────────────────────
def find_camera(preferred=0):
    for idx in [preferred] + [i for i in range(10) if i != preferred]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                # Set lower resolution for speed
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                print(f"[INFO] Camera index {idx}")
                return cap, idx
            cap.release()
    return None, -1

# ── Constants ─────────────────────────────────────────────────
BOARD_SIZE  = 700
WIN_W       = BOARD_SIZE + 340
WIN_H       = BOARD_SIZE
FPS         = 60
BORDER      = 60
INNER       = BOARD_SIZE - 2 * BORDER

POCKET_R    = 28
COIN_R      = 18
STRIKER_R   = 24
FRICTION    = 0.978          # slightly more friction = stops naturally
MIN_SPEED   = 0.25
SUBSTEPS    = 4              # physics sub-steps per frame (prevents tunnelling)
MAX_SPEED   = 40.0           # cap speed to prevent tunnelling

C_BOARD      = (34,  139,  34)
C_BORDER     = (139,  90,  43)
C_LINE       = (210, 180, 140)
C_WHITE_COIN = (245, 245, 220)
C_BLACK_COIN = ( 40,  40,  40)
C_RED_QUEEN  = (200,  30,  30)
C_STRIKER    = (180, 180, 220)
C_POCKET     = ( 20,  20,  20)
C_POWER_BG   = ( 60,  60,  60)
C_HUD_BG     = ( 20,  20,  30)
C_HUD_TEXT   = (220, 220, 200)
C_ARROW      = (255, 220,  50)
C_AIM_LINE   = (255, 80,   80)   # red aim line when locked

PLAYER_COLORS = [(66,135,245),(245,66,66),(66,245,132),(245,200,66)]
PLAYER_NAMES  = ["Player 1","Player 2","Player 3","Player 4"]

# MediaPipe landmark indices
LM_WRIST     = 0
LM_THUMB_TIP = 4
LM_INDEX_MCP = 5
LM_INDEX_TIP = 8
LM_MIDDLE_MCP= 9

# ── Data classes ──────────────────────────────────────────────
@dataclass
class Coin:
    x: float; y: float
    vx: float = 0.0; vy: float = 0.0
    radius: int = COIN_R
    color: Tuple = C_WHITE_COIN
    kind: str = "white"
    pocketed: bool = False

@dataclass
class Striker:
    x: float; y: float
    vx: float = 0.0; vy: float = 0.0
    radius: int = STRIKER_R
    color: Tuple = C_STRIKER
    active: bool = False
    angle: float = -math.pi / 2

@dataclass
class Player:
    name: str; color: Tuple
    score: int = 0
    pocketed: List = field(default_factory=list)

# ── Threaded finger detector ──────────────────────────────────
class FingerDetector:
    """
    Runs MediaPipe HandLandmarker in a background thread.
    The game loop reads latest values without ever blocking.

    Aiming logic:
      • Striker X  ← index fingertip X (mirrored)
      • Shot angle ← wrist → index_tip vector  (long, very stable)
      • Power      ← wrist Y (how high whole hand is held)
      • Shoot      ← pinch (thumb tip ↔ index tip distance)
    """
    PINCH_THRESH    = 0.06
    PINCH_CONFIRM   = 3
    COOLDOWN_SEC    = 1.0     # seconds between shots

    HAND_CONN = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17),
    ]

    def __init__(self, cap):
        from mediapipe.tasks.python import vision as _vis
        from mediapipe.tasks.python import BaseOptions as _BO
        opts = _vis.HandLandmarkerOptions(
            base_options=_BO(model_asset_path=MODEL_PATH),
            running_mode=_vis.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.55,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.50,
        )
        self._det  = _vis.HandLandmarker.create_from_options(opts)
        self._cap  = cap

        # Outputs (read by game loop)
        self.tip_x      = None   # normalised [0,1]
        self.tip_y      = None
        self.angle      = None   # shot angle in radians
        self.power      = 0.4
        self.shoot      = False
        self.hand_vis   = False
        self._lock      = threading.Lock()

        # Internal smoothing
        self._sx        = 0.5
        self._sy        = 0.5
        self._sang      = None
        self._pinch_n   = 0
        self._last_shot = 0.0

        # Start background thread
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            self._process(frame)

    def _process(self, frame):
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_im = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res   = self._det.detect(mp_im)

        now   = time.time()

        if not res.hand_landmarks:
            with self._lock:
                self.hand_vis = False
            cv2.putText(frame,"Show index finger!",(10,40),
                        cv2.FONT_HERSHEY_SIMPLEX,0.85,(0,200,255),2)
            cv2.putText(frame,"Point it at camera",(10,70),
                        cv2.FONT_HERSHEY_SIMPLEX,0.60,(160,160,255),1)
            cv2.imshow("Finger Cam",frame)
            cv2.waitKey(1)
            return

        lm     = res.hand_landmarks[0]
        wrist  = lm[LM_WRIST]
        thumb  = lm[LM_THUMB_TIP]
        imcp   = lm[LM_INDEX_MCP]
        itip   = lm[LM_INDEX_TIP]

        # ── Smooth tip position ──────────────────────────────
        a = 0.30
        self._sx += a * (itip.x - self._sx)
        self._sy += a * (itip.y - self._sy)

        # ── Shot angle: WRIST → INDEX_TIP  (long, stable vector) ──
        dx = itip.x - wrist.x
        dy = itip.y - wrist.y
        raw_ang = math.atan2(dy, dx)
        if self._sang is None:
            self._sang = raw_ang
        else:
            diff = raw_ang - self._sang
            while diff >  math.pi: diff -= 2*math.pi
            while diff < -math.pi: diff += 2*math.pi
            self._sang += diff * 0.25   # faster tracking

        # ── Power: wrist Y  (higher hand = lower y = more power) ──
        power = max(0.05, min(1.0, 1.0 - wrist.y))

        # ── Pinch detection ──────────────────────────────────
        pdist  = math.hypot(thumb.x - itip.x, thumb.y - itip.y)
        pinching = pdist < self.PINCH_THRESH
        if pinching:
            self._pinch_n += 1
        else:
            self._pinch_n = 0

        shoot = False
        cooldown_ok = (now - self._last_shot) > self.COOLDOWN_SEC
        if self._pinch_n >= self.PINCH_CONFIRM and cooldown_ok:
            shoot = True
            self._last_shot = now
            self._pinch_n   = 0

        # ── Write outputs ────────────────────────────────────
        with self._lock:
            self.tip_x    = self._sx
            self.tip_y    = self._sy
            self.angle    = self._sang
            self.power    = power
            self.shoot    = shoot
            self.hand_vis = True

        # ── Annotate camera preview ──────────────────────────
        pts = [(int(p.x*w), int(p.y*h)) for p in lm]
        for a2,b2 in self.HAND_CONN:
            cv2.line(frame,pts[a2],pts[b2],(60,200,60),2)
        for px,py in pts:
            cv2.circle(frame,(px,py),4,(255,255,255),-1)

        fx,fy = pts[LM_INDEX_TIP]
        wx,wy = pts[LM_WRIST]
        # Green dot on fingertip
        cv2.circle(frame,(fx,fy),14,(0,255,120),-1)
        cv2.circle(frame,(fx,fy),14,(255,255,255), 2)
        # Direction arrow: wrist → fingertip (extended)
        ext = 1.5
        ax = int(wx + (fx-wx)*ext); ay = int(wy + (fy-wy)*ext)
        cv2.arrowedLine(frame,(wx,wy),(ax,ay),(255,220,0),3,tipLength=0.20)

        # Pinch indicator
        tx,ty = pts[LM_THUMB_TIP]
        cv2.circle(frame,(tx,ty),10,(200,80,255),-1)
        col_p = (0,60,255) if pinching else (100,200,100)
        cv2.line(frame,(fx,fy),(tx,ty),col_p,2)
        if shoot:
            cv2.putText(frame,"SHOOT!",(max(0,fx-45),max(30,fy-20)),
                        cv2.FONT_HERSHEY_SIMPLEX,1.4,(0,30,255),3)

        # Power bar
        bh=int(power*80)
        cv2.rectangle(frame,(w-28,h-100),(w-8,h-10),(50,50,50),-1)
        cv2.rectangle(frame,(w-28,h-10-bh),(w-8,h-10),
                      (int(50+205*power),int(220-160*power),50),-1)
        cv2.putText(frame,"PWR",(w-38,h-106),
                    cv2.FONT_HERSHEY_SIMPLEX,0.38,(200,200,200),1)
        cv2.putText(frame,f"P:{pdist:.2f}",(8,32),
                    cv2.FONT_HERSHEY_SIMPLEX,0.60,col_p,2)
        cv2.putText(frame,"PINCH=SHOOT",(8,h-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.50,(160,160,255),1)
        cv2.imshow("Finger Cam", frame)
        cv2.waitKey(1)

    def read(self):
        """Return latest (tip_x, tip_y, angle, shoot, power, hand_visible)."""
        with self._lock:
            s = self.shoot
            self.shoot = False   # consume shoot signal
            return (self.tip_x, self.tip_y, self.angle,
                    s, self.power, self.hand_vis)

    def stop(self):
        self._running = False

# ── Physics ───────────────────────────────────────────────────
def circle_collide(a, b):
    dx,dy = b.x-a.x, b.y-a.y
    dist  = math.hypot(dx,dy)
    md    = a.radius + b.radius
    if dist >= md or dist == 0: return
    nx,ny = dx/dist, dy/dist
    # Push apart
    ov    = (md - dist) + 0.5   # small extra separation
    a.x  -= nx*ov/2;  b.x += nx*ov/2
    a.y  -= ny*ov/2;  b.y += ny*ov/2
    # Exchange velocity along normal
    rel   = (b.vx-a.vx)*nx + (b.vy-a.vy)*ny
    if rel >= 0: return
    # Slightly inelastic (0.92) for more realistic feel
    imp   = rel * 0.92
    a.vx += imp*nx;  a.vy += imp*ny
    b.vx -= imp*nx;  b.vy -= imp*ny

def wall_bounce(obj, x0,x1,y0,y1):
    r = obj.radius
    if obj.x-r < x0: obj.x=x0+r; obj.vx= abs(obj.vx)*0.80
    if obj.x+r > x1: obj.x=x1-r; obj.vx=-abs(obj.vx)*0.80
    if obj.y-r < y0: obj.y=y0+r; obj.vy= abs(obj.vy)*0.80
    if obj.y+r > y1: obj.y=y1-r; obj.vy=-abs(obj.vy)*0.80

def clamp_speed(obj):
    spd = math.hypot(obj.vx, obj.vy)
    if spd > MAX_SPEED:
        obj.vx = obj.vx/spd*MAX_SPEED
        obj.vy = obj.vy/spd*MAX_SPEED

def in_pocket(obj, pockets):
    return any(math.hypot(obj.x-px,obj.y-py) < POCKET_R+2 for px,py in pockets)

# ── Board setup ───────────────────────────────────────────────
def make_coins():
    cx=cy=BOARD_SIZE//2; gap=COIN_R*2+3
    coins=[Coin(cx,cy,kind="queen",color=C_RED_QUEEN)]
    alt6 =[C_WHITE_COIN,C_BLACK_COIN]*3; knd6 =["white","black"]*3
    alt12=[C_WHITE_COIN,C_BLACK_COIN]*6; knd12=["white","black"]*6
    for i,a in enumerate(i*math.pi/3   for i in range(6)):
        coins.append(Coin(cx+math.cos(a)*gap,   cy+math.sin(a)*gap,
                          kind=knd6[i],  color=alt6[i]))
    for i,a in enumerate(i*math.pi/6   for i in range(12)):
        coins.append(Coin(cx+math.cos(a)*gap*2, cy+math.sin(a)*gap*2,
                          kind=knd12[i], color=alt12[i]))
    return coins

# Striker baselines for each player
_STRIKER_DEFS = [
    # (x, y, angle,  axis_is_x)
    (BOARD_SIZE//2, BOARD_SIZE-BORDER-30, -math.pi/2, True ),  # P1 bottom
    (BOARD_SIZE//2, BORDER+30,             math.pi/2, True ),  # P2 top
    (BORDER+30,     BOARD_SIZE//2,         0.0,       False),  # P3 left
    (BOARD_SIZE-BORDER-30, BOARD_SIZE//2,  math.pi,   False),  # P4 right
]

def make_striker(pidx):
    x,y,ang,_ = _STRIKER_DEFS[pidx]
    return Striker(x, y, angle=ang)

def make_pockets():
    b,s=BORDER,BOARD_SIZE
    return[(b,b),(s-b,b),(b,s-b),(s-b,s-b)]

# ── Drawing ───────────────────────────────────────────────────
def draw_board(surf):
    surf.fill(C_BORDER)
    pygame.draw.rect(surf,C_BOARD,(BORDER,BORDER,INNER,INNER))
    cx=cy=BOARD_SIZE//2; b=BORDER
    for d in(20,25):
        pygame.draw.rect(surf,C_LINE,(b-d,b-d,INNER+2*d,INNER+2*d),2)
    pygame.draw.circle(surf,C_LINE,(cx,cy),40,2)
    pygame.draw.circle(surf,C_LINE,(cx,cy),80,1)
    for(x1,y1),(x2,y2) in[
        ((b,b),(b+60,b+60)),
        ((BOARD_SIZE-b,b),(BOARD_SIZE-b-60,b+60)),
        ((b,BOARD_SIZE-b),(b+60,BOARD_SIZE-b-60)),
        ((BOARD_SIZE-b,BOARD_SIZE-b),(BOARD_SIZE-b-60,BOARD_SIZE-b-60))]:
        pygame.draw.line(surf,C_LINE,(x1,y1),(x2,y2),2)
    off=80
    pygame.draw.line(surf,C_LINE,(cx-off,BOARD_SIZE-b-30),(cx+off,BOARD_SIZE-b-30),2)
    pygame.draw.line(surf,C_LINE,(cx-off,b+30),(cx+off,b+30),2)
    pygame.draw.line(surf,C_LINE,(b+30,cy-off),(b+30,cy+off),2)
    pygame.draw.line(surf,C_LINE,(BOARD_SIZE-b-30,cy-off),(BOARD_SIZE-b-30,cy+off),2)

def draw_pockets(surf,pockets):
    for px,py in pockets:
        pygame.draw.circle(surf,C_POCKET,(int(px),int(py)),POCKET_R)
        pygame.draw.circle(surf,(70,70,70),(int(px),int(py)),POCKET_R,2)

def draw_coins(surf,coins):
    for c in coins:
        if c.pocketed: continue
        x,y=int(c.x),int(c.y)
        pygame.draw.circle(surf,(0,0,0),(x+2,y+2),c.radius)
        pygame.draw.circle(surf,c.color,(x,y),c.radius)
        pygame.draw.circle(surf,(200,200,200),(x,y),c.radius,1)
        if c.kind=="queen":
            pygame.draw.circle(surf,(255,200,50),(x,y),c.radius//2,2)

def draw_striker(surf, striker, power, pcol, aim_locked):
    if striker is None: return
    x,y=int(striker.x),int(striker.y)
    if not striker.active:
        alen = 60 + int(power*100)
        # Draw dotted aim line
        n_dots = 14
        for i in range(1, n_dots+1):
            t   = i/n_dots
            px2 = int(x + math.cos(striker.angle)*alen*t)
            py2 = int(y + math.sin(striker.angle)*alen*t)
            col = C_AIM_LINE if aim_locked else C_ARROW
            fade= tuple(int(c*(1.0-t*0.6)) for c in col)
            pygame.draw.circle(surf,fade,(px2,py2),max(2,4-i//4))
        ex=int(x+math.cos(striker.angle)*alen)
        ey=int(y+math.sin(striker.angle)*alen)
        col = C_AIM_LINE if aim_locked else C_ARROW
        pygame.draw.circle(surf,col,(ex,ey),7)
        # Ring around striker when locked
        rc = C_AIM_LINE if aim_locked else (255,220,50)
        pygame.draw.circle(surf,rc,(x,y),STRIKER_R+9,2)
    pygame.draw.circle(surf,(0,0,0),(x+2,y+2),striker.radius)
    pygame.draw.circle(surf,pcol,(x,y),striker.radius)
    pygame.draw.circle(surf,C_STRIKER,(x,y),striker.radius,3)
    pygame.draw.circle(surf,(240,240,255),(x-6,y-6),5)

def draw_hud(surf,players,cp,N,power,msg,font,small,hand_vis):
    hx=BOARD_SIZE
    pygame.draw.rect(surf,C_HUD_BG,(hx,0,340,WIN_H))
    t=font.render("CARROM",True,(220,180,80))
    surf.blit(t,(hx+170-t.get_width()//2,14))
    for i,p in enumerate(players[:N]):
        yo=65+i*108; bg=(42,42,65) if i==cp else(24,24,38)
        pygame.draw.rect(surf,bg,(hx+10,yo,320,96),border_radius=10)
        pygame.draw.rect(surf,p.color,(hx+10,yo,320,96),2,border_radius=10)
        surf.blit(font.render(p.name,True,p.color),(hx+20,yo+8))
        surf.blit(font.render(f"Score: {p.score}",True,C_HUD_TEXT),(hx+20,yo+36))
        for ci,ck in enumerate(p.pocketed[:15]):
            cc=C_RED_QUEEN if ck=="queen" else(C_WHITE_COIN if ck=="white" else C_BLACK_COIN)
            pygame.draw.circle(surf,cc,(hx+20+ci*16,yo+74),6)
        if i==cp: surf.blit(small.render("◀ YOUR TURN",True,(255,220,50)),(hx+175,yo+8))
    # Power bar
    pby=WIN_H-185
    surf.blit(small.render("POWER",True,C_HUD_TEXT),(hx+20,pby-22))
    pygame.draw.rect(surf,C_POWER_BG,(hx+20,pby,300,20),border_radius=5)
    pw_col=(int(50+200*power),int(220-160*power),50)
    pygame.draw.rect(surf,pw_col,(hx+20,pby,int(300*power),20),border_radius=5)
    pygame.draw.rect(surf,C_HUD_TEXT,(hx+20,pby,300,20),1,border_radius=5)
    surf.blit(small.render(f"{int(power*100)}%",True,C_HUD_TEXT),(hx+230,pby-20))
    # Camera status dot
    dot_col=(0,220,80) if hand_vis else (220,60,60)
    pygame.draw.circle(surf,dot_col,(hx+320,pby-10),7)
    # Status
    surf.blit(small.render(msg[:40],True,(255,200,80)),(hx+10,WIN_H-152))
    # Controls hint
    for i,h in enumerate(["☝  Finger L/R  → slide striker",
                           "↕  Tilt finger → aim direction",
                           "⬆  Raise hand  → more power",
                           "🤏  PINCH       → SHOOT!"]):
        surf.blit(small.render(h,True,(120,120,165)),(hx+14,WIN_H-118+i*24))

# ── Main ──────────────────────────────────────────────────────
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--players",type=int,default=2,choices=[2,3,4])
    parser.add_argument("--camera", type=int,default=0)
    args=parser.parse_args()

    ensure_model()
    cap, cam_idx = find_camera(args.camera)
    if cap is None:
        print("ERROR: No webcam found (0-9). Plug in webcam and retry.")
        print("  Check:  ls /dev/video*")
        sys.exit(1)

    pygame.init()
    screen=pygame.display.set_mode((WIN_W,WIN_H))
    pygame.display.set_caption(f"Carrom — Finger Control (cam {cam_idx})")
    clock=pygame.time.Clock()
    font =pygame.font.SysFont("Arial",22,bold=True)
    small=pygame.font.SysFont("Arial",15)

    N=args.players
    players=[Player(PLAYER_NAMES[i],PLAYER_COLORS[i]) for i in range(N)]
    coins=make_coins(); pockets=make_pockets()
    cp=0; striker=make_striker(cp)
    det=FingerDetector(cap)

    power=0.4; state_msg="Point finger to aim, PINCH to shoot!"
    phase="aim"; turn_timer=0.0; winner_msg=""

    # Per-player baseline constraints
    Xmin=BORDER+STRIKER_R+5; Xmax=BOARD_SIZE-BORDER-STRIKER_R-5
    Ymin=BORDER+STRIKER_R+5; Ymax=BOARD_SIZE-BORDER-STRIKER_R-5

    aim_locked = False      # True once pinch starts (angle frozen)
    locked_angle = None

    running=True
    while running:
        dt=clock.tick(FPS)/1000.0

        for event in pygame.event.get():
            if event.type==pygame.QUIT: running=False
            if event.type==pygame.KEYDOWN:
                if event.key==pygame.K_q: running=False
                if event.key==pygame.K_r and phase=="gameover":
                    coins=make_coins()
                    players=[Player(PLAYER_NAMES[i],PLAYER_COLORS[i]) for i in range(N)]
                    cp=0; striker=make_striker(cp); power=0.4
                    phase="aim"; aim_locked=False; locked_angle=None
                    state_msg="New game! Aim with finger."

        tip_x,tip_y,angle,shoot,cam_power,hand_vis=det.read()

        # ── AIM PHASE ────────────────────────────────────────
        if phase=="aim" and striker is not None:
            _,_,_,ax_is_x = _STRIKER_DEFS[cp]

            if tip_x is not None:
                # Move striker along its baseline
                if ax_is_x:
                    target = BORDER+45 + tip_x*(INNER-90)
                    striker.x += (target-striker.x)*0.35   # snappier tracking
                    striker.x  = max(Xmin,min(Xmax,striker.x))
                else:
                    target = BORDER+45 + tip_x*(INNER-90)
                    striker.y += (target-striker.y)*0.35
                    striker.y  = max(Ymin,min(Ymax,striker.y))

                # Update aim angle (only if not locked)
                if angle is not None and not aim_locked:
                    striker.angle = angle

                # Power from wrist height
                power += (cam_power - power)*0.20

            state_msg=(f"{players[cp].name}  |  "
                       f"{'AIM LOCKED — release & pinch' if aim_locked else 'PINCH to shoot'}  |  "
                       f"{int(power*100)}%")

            if shoot:
                phase="rolling"
                # Use locked angle if we had one, else current angle
                fire_angle = locked_angle if locked_angle else striker.angle
                striker.angle = fire_angle
                spd = 12 + power*35    # faster shots
                striker.vx = math.cos(fire_angle)*spd
                striker.vy = math.sin(fire_angle)*spd
                clamp_speed(striker)
                striker.active=True; aim_locked=False; locked_angle=None
                state_msg="Rolling..."

        # ── ROLLING PHASE ────────────────────────────────────
        elif phase=="rolling":
            all_stopped=True

            # Run SUBSTEPS physics steps per frame
            for _ in range(SUBSTEPS):
                fric_step = FRICTION ** (1/SUBSTEPS)

                if striker and striker.active:
                    striker.x+=striker.vx/SUBSTEPS
                    striker.y+=striker.vy/SUBSTEPS
                    striker.vx*=fric_step
                    striker.vy*=fric_step
                    wall_bounce(striker,BORDER,BOARD_SIZE-BORDER,BORDER,BOARD_SIZE-BORDER)
                    clamp_speed(striker)
                    spd=math.hypot(striker.vx,striker.vy)
                    if spd < MIN_SPEED:
                        striker.vx=striker.vy=0; striker.active=False
                    else:
                        all_stopped=False
                    for coin in coins:
                        if not coin.pocketed: circle_collide(striker,coin)

                for coin in coins:
                    if coin.pocketed: continue
                    coin.x+=coin.vx/SUBSTEPS
                    coin.y+=coin.vy/SUBSTEPS
                    coin.vx*=fric_step
                    coin.vy*=fric_step
                    wall_bounce(coin,BORDER,BOARD_SIZE-BORDER,BORDER,BOARD_SIZE-BORDER)
                    clamp_speed(coin)
                    spd=math.hypot(coin.vx,coin.vy)
                    if spd < MIN_SPEED: coin.vx=coin.vy=0
                    else: all_stopped=False
                    for other in coins:
                        if other is not coin and not other.pocketed:
                            circle_collide(coin,other)

            # Check pockets (outside substep loop)
            for coin in coins:
                if coin.pocketed: continue
                if in_pocket(coin,pockets):
                    coin.pocketed=True; coin.vx=coin.vy=0
                    pts2=5 if coin.kind=="queen" else 1
                    players[cp].score+=pts2
                    players[cp].pocketed.append(coin.kind)
                    state_msg=f"+{pts2} pt!  ({coin.kind})"

            if striker and in_pocket(striker,pockets):
                striker.vx=striker.vy=0; striker.active=False
                players[cp].score=max(0,players[cp].score-1)
                state_msg="Striker pocketed!  -1 pt"
                striker=make_striker(cp); all_stopped=True

            if all_stopped:
                turn_timer+=dt
                if turn_timer>1.0:
                    turn_timer=0
                    remaining=[c for c in coins if not c.pocketed]
                    if not remaining:
                        best=max(range(N),key=lambda i:players[i].score)
                        winner_msg=f"{players[best].name} WINS!  ({players[best].score} pts)"
                        phase="gameover"
                    else:
                        cp=(cp+1)%N; striker=make_striker(cp)
                        power=0.4; phase="aim"
                        aim_locked=False; locked_angle=None
                        state_msg=f"{players[cp].name}'s turn — aim with finger!"

        elif phase=="gameover":
            state_msg=winner_msg

        # ── DRAW ─────────────────────────────────────────────
        draw_board(screen)
        draw_pockets(screen,pockets)
        draw_coins(screen,coins)
        if striker:
            draw_striker(screen,striker,power,players[cp].color,aim_locked)
        draw_hud(screen,players,cp,N,power,state_msg,font,small,hand_vis)

        if phase=="gameover":
            ov=pygame.Surface((BOARD_SIZE,WIN_H),pygame.SRCALPHA)
            ov.fill((0,0,0,160)); screen.blit(ov,(0,0))
            gf=pygame.font.SysFont("Arial",50,bold=True)
            screen.blit(gf.render("GAME OVER",True,(255,220,50)),
                (BOARD_SIZE//2-gf.size("GAME OVER")[0]//2,WIN_H//2-90))
            screen.blit(font.render(winner_msg,True,(255,255,255)),
                (BOARD_SIZE//2-font.size(winner_msg)[0]//2,WIN_H//2-10))
            screen.blit(small.render("R to restart  |  Q to quit",True,(180,180,180)),
                (BOARD_SIZE//2-120,WIN_H//2+55))

        pygame.display.flip()

    det.stop()
    cap.release(); cv2.destroyAllWindows(); pygame.quit(); sys.exit()

if __name__=="__main__":
    main()
