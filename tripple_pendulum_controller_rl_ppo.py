"""
Triple Inverted Pendulum on a Cart
====================================
Driven by a trained PPO Reinforcement Learning Policy.

Controls:
  SPACE    Start / Pause          R       Reset (hanging down)
  1        Reset (near upright)   U       Toggle controller ON/OFF
  UP/DOWN  Simulation speed       Mouse   Drag bobs (when paused)
"""

import math
import sys
import pickle
import numpy as np
import pygame

# =====================================================================
# Physical Parameters (MATCH TRAINING ENV)
# =====================================================================
from config import * 

# =====================================================================
# RL Model Paths (SWING-UP + BALANCE, switched by angle threshold)
# =====================================================================
SWINGUP_MODEL_PATH = "ppo_triple_pendulum_swingup.zip"
SWINGUP_VNORM_PATH = "ppo_triple_pendulum_swingup.pkl"

BALANCE_MODEL_PATH = "ppo_triple_pendulum_balance.zip"
BALANCE_VNORM_PATH = "ppo_triple_pendulum_balance.pkl"

# Switch to the balance controller once every angle is within this many
# radians of upright (0 rad convention, matching the "upright" reset state)
BALANCE_ANGLE_THRESHOLD = 0.25

# =====================================================================
# Simulation / Display
# =====================================================================
TARGET_FPS = 60
SUBSTEPS   = 2
WIDTH, HEIGHT = 2500, 1100
PPM_X     = 200            # Horizontal scale (keeps 10m track visible)
PPM_Y     = 200            # Vertical scale (zooms in on the pendulum)
CART_W, CART_H = 150, 10
BOB_R      = 6
GRAB_R     = 25
GROUND_Y   = 480

# Colours
BG      = ( 18,  20,  26)
PANEL   = ( 28,  31,  38)
GND     = ( 50,  52,  60)
C_COL   = ( 90, 170, 255)
C_BORD  = (140, 200, 255)
ROD     = (200, 200, 210)
B1_COL  = (231,  76,  60)
B2_COL  = ( 52, 152, 219)
B3_COL  = (155,  89, 182)  
TXT     = (230, 230, 235)
DIM     = (120, 120, 130)
GREEN   = ( 46, 204, 113)
YELLOW  = (241, 196,  15)
RED     = (231,  76,  60)

def wrap_to_pi(theta):
    """Wrap angle to (-pi, pi] so 'near upright' works regardless of winding."""
    return (theta + math.pi) % (2 * math.pi) - math.pi

# =====================================================================
# DYNAMICS  (4x4 Matrix for Triple Pendulum - Updated to match config.py)
# =====================================================================
def dynamics(state, u):
    _, _, t1, w1, t2, w2, t3, w3 = state
    s1, c1   = math.sin(t1), math.cos(t1)
    s2, c2   = math.sin(t2), math.cos(t2)
    s3, c3   = math.sin(t3), math.cos(t3)
    s12, c12 = math.sin(t1 - t2), math.cos(t1 - t2)
    s13, c13 = math.sin(t1 - t3), math.cos(t1 - t3)
    s23, c23 = math.sin(t2 - t3), math.cos(t2 - t3)

    M_mat = np.array([
        [CART_MASS + M1 + M2 + M3,(M1 + M2 + M3)*L1*c1,(M2 + M3)*L2*c2,M3*L3*c3],
        [(M1 + M2 + M3)*L1*c1,(M1 + M2 + M3)*L1**2,  (M2 + M3)*L1*L2*c12, M3*L1*L3*c13    ],
        [(M2 + M3)*L2*c2,(M2 + M3)*L1*L2*c12,   (M2 + M3)*L2**2,M3*L2*L3*c23],
        [M3*L3*c3, M3*L1*L3*c13, M3*L2*L3*c23,M3*L3**2]])

    f_vec = np.array([
        u - DAMPING*state[1] + (M1 + M2 + M3)*L1*w1**2*s1 + (M2 + M3)*L2*w2**2*s2 + M3*L3*w3**2*s3,
        (M1 + M2 + M3)*GRAVITY*L1*s1 - (M2 + M3)*L1*L2*w2**2*s12 - M3*L1*L3*w3**2*s13,
        (M2 + M3)*GRAVITY*L2*s2 + (M2 + M3)*L1*L2*w1**2*s12 - M3*L2*L3*w3**2*s23,
        M3*GRAVITY*L3*s3 + M3*L1*L3*w1**2*s13 + M3*L2*L3*w2**2*s23])

    try:
        acc = np.linalg.solve(M_mat, f_vec)
    except np.linalg.LinAlgError:
        acc = np.zeros(4)

    return np.array([state[1], acc[0], w1, acc[1], w2, acc[2], w3, acc[3]])


def rk4(state, u, dt):
    k1 = dynamics(state, u)
    k2 = dynamics(state + 0.5*dt*k1, u)
    k3 = dynamics(state + 0.5*dt*k2, u)
    k4 = dynamics(state + dt*k3, u)
    return state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)


# =====================================================================
#  Energy
# =====================================================================
def total_energy(s):
    x, v, t1, w1, t2, w2, t3, w3 = s
    c1, s1 = math.cos(t1), math.sin(t1)
    c2, s2 = math.cos(t2), math.sin(t2)
    c3, s3 = math.cos(t3), math.sin(t3)
    vx1 = v + L1*w1*c1;               vy1 = -L1*w1*s1
    vx2 = vx1 + L2*w2*c2;             vy2 = vy1 - L2*w2*s2
    vx3 = vx2 + L3*w3*c3;             vy3 = vy2 - L3*w3*s3
    KE = (0.5*CART_MASS*v**2 + 0.5*M1*(vx1**2 + vy1**2) + 0.5*M2*(vx2**2 + vy2**2) + 0.5*M3*(vx3**2 + vy3**2))
    PE = M1*GRAVITY*L1*c1 + M2*GRAVITY*(L1*c1 + L2*c2) + M3*GRAVITY*(L1*c1 + L2*c2 + L3*c3)
    return KE + PE


# =====================================================================
#  Simulator  
# =====================================================================
class Simulator:
    def __init__(self):
        from stable_baselines3 import PPO

        print(f"[GUI] Loading Swing-Up Model from: {SWINGUP_MODEL_PATH}")
        self.swingup_model = PPO.load(SWINGUP_MODEL_PATH)
        with open(SWINGUP_VNORM_PATH, "rb") as f:
            swingup_vec = pickle.load(f)
        self.swingup_mean = swingup_vec.obs_rms.mean
        self.swingup_var = swingup_vec.obs_rms.var

        print(f"[GUI] Loading Balance Model from: {BALANCE_MODEL_PATH}")
        self.balance_model = PPO.load(BALANCE_MODEL_PATH)
        with open(BALANCE_VNORM_PATH, "rb") as f:
            balance_vec = pickle.load(f)
        self.balance_mean = balance_vec.obs_rms.mean
        self.balance_var = balance_vec.obs_rms.var

        self.active_controller = "swingup"  # for HUD display

        self.reset("hanging")

    def _get_obs(self, state, obs_mean, obs_var):
            """Convert 8-element GUI state to 11-element Gym observation and normalize."""
            x, v, t1, w1, t2, w2, t3, w3 = state
            raw_obs = np.array([
                math.sin(t1), math.cos(t1),
                math.sin(t2), math.cos(t2),
                math.sin(t3), math.cos(t3),
                w1, w2, w3, x, v
            ], dtype=np.float32)
    
            normalized_obs = (raw_obs - obs_mean) / np.sqrt(obs_var + 1e-8)
            normalized_obs = np.clip(normalized_obs, -10.0, 10.0)
            return normalized_obs

    def reset(self, preset="hanging"):
        if preset == "hanging":
            self.state = np.array([0., 0., math.pi, 0., math.pi, 0., math.pi, 0.])
        else:
            self.state = np.array([0., 0., 0.15, 0., 0.10, 0., 0.10, 0.])
        self.time = 0.0
        self.u    = 0.0
        self.mode = "idle"
        self.ctrl_on = True
        self.E0 = total_energy(self.state)

    def toggle_ctrl(self):
        self.ctrl_on = not self.ctrl_on
        if not self.ctrl_on:
            self.u = 0.0
            self.mode = "off"
        else:
            self.mode = "rl"
            
    def step(self, dt):
        s = self.state
        if self.mode == "off":
            self.u = 0.0
        elif self.mode == "idle":
            self.u = 0.0
        elif self.mode == "rl":
            t1, t2, t3 = s[2], s[4], s[6]
            e1, e2, e3 = wrap_to_pi(t1), wrap_to_pi(t2), wrap_to_pi(t3)
            near_upright = (abs(e1) < BALANCE_ANGLE_THRESHOLD and
                            abs(e2) < BALANCE_ANGLE_THRESHOLD and
                            abs(e3) < BALANCE_ANGLE_THRESHOLD)

            if near_upright:
                model, mean, var = self.balance_model, self.balance_mean, self.balance_var
                self.active_controller = "balance"
            else:
                model, mean, var = self.swingup_model, self.swingup_mean, self.swingup_var
                self.active_controller = "swingup"

            obs = self._get_obs(s, mean, var)
            action, _ = model.predict(obs, deterministic=True)
            self.u = float(action[0]) * FORCE_LIMIT
            
      # Apply wall contact force BEFORE integration
        u_effective = self.u
    
        # Right wall (x > 10)
        if self.state[0] > X_LIMIT:
            penetration = self.state[0] - X_LIMIT
            k_contact = 1000.0   # Wall stiffness
            d_contact = 50.0     # Wall damping
            contact_force = -k_contact * penetration - d_contact * self.state[1]
            u_effective = u_effective + contact_force
            
        # Left wall (x < -10)
        elif self.state[0] < -X_LIMIT:
            penetration = self.state[0] + X_LIMIT  
            k_contact = 1000.0
            d_contact = 50.0
            contact_force = -k_contact * penetration - d_contact * self.state[1]
            u_effective = u_effective + contact_force    

        # Integrate with wall force included
        self.state = rk4(self.state, u_effective, dt)
        self.time += dt
    
    
    def positions(self):
        cx  = WIDTH // 2 + self.state[0] * PPM_X
        piv = (cx, GROUND_Y - (CART_H//2))
        
        b1x = piv[0] + L1*PPM_X*math.sin(self.state[2])
        b1y = piv[1] - L1*PPM_Y*math.cos(self.state[2])
        
        b2x = b1x    + L2*PPM_X*math.sin(self.state[4])
        b2y = b1y    - L2*PPM_Y*math.cos(self.state[4])
        
        b3x = b2x    + L3*PPM_X*math.sin(self.state[6])
        b3y = b2y    - L3*PPM_Y*math.cos(self.state[6])
        
        rect = pygame.Rect(cx - CART_W//2, GROUND_Y - CART_H, CART_W, CART_H)
        return rect, piv, (b1x, b1y), (b2x, b2y), (b3x, b3y)


# =====================================================================
#  GUI
# =====================================================================
class App:
    def __init__(self):
        pygame.init()
        self.scr = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Triple Inverted Pendulum - PPO Neural Net Policy")
        self.clk = pygame.time.Clock()
        self.fn  = pygame.font.SysFont("consolas", 17)
        self.fns = pygame.font.SysFont("consolas", 14)
        self.fnl = pygame.font.SysFont("consolas", 22, bold=True)
        self.sim = Simulator()
        self.running  = False
        self.speed    = 1.0
        self.drag     = None
        self.log_t    = 0.0
        self.log_hdr  = False

    def events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            elif e.type == pygame.KEYDOWN:
                self._key(e.key)
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                self._mdown(e.pos)
            elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
                self.drag = None
            elif e.type == pygame.MOUSEMOTION:
                self._mmove(e.pos)

    def _key(self, k):
        s = self.sim
        if k == pygame.K_ESCAPE:
            pygame.quit(); sys.exit()
        elif k == pygame.K_SPACE:
            self.running = not self.running
            if self.running and s.mode == "idle" and s.ctrl_on:
                s.mode = "rl"
                print("[ctrl] RL policy started")
        elif k == pygame.K_r:
            s.reset("hanging"); 
            self.running = False; self.log_hdr = False
        elif k == pygame.K_1:
            s.reset("upright"); 
            self.running = False; self.log_hdr = False
            if s.ctrl_on: s.mode = "rl"
            print("[ctrl] near-upright reset -> RL policy")
        elif k == pygame.K_u:
            s.toggle_ctrl()
            print(f"[ctrl] {'ON' if s.ctrl_on else 'OFF (u=0)'}")
        elif k == pygame.K_UP:
            self.speed = min(3.0, self.speed + 0.25)
        elif k == pygame.K_DOWN:
            self.speed = max(0.25, self.speed - 0.25)

    def _mdown(self, pos):
        if self.running: return
        _, _, b1, b2, b3 = self.sim.positions()
        if math.hypot(pos[0]-b3[0], pos[1]-b3[1]) <= GRAB_R:
            self.drag = "b3"
        elif math.hypot(pos[0]-b2[0], pos[1]-b2[1]) <= GRAB_R:
            self.drag = "b2"
        elif math.hypot(pos[0]-b1[0], pos[1]-b1[1]) <= GRAB_R:
            self.drag = "b1"

    def _mmove(self, pos):
        if self.running or self.drag is None: return
        s = self.sim.state
        _, piv, b1, b2, _ = self.sim.positions()
        if self.drag == "b1":
            dx = pos[0] - piv[0]; dy = -(pos[1] - piv[1])
            if abs(dx) + abs(dy) > 2: s[2] = math.atan2(dx, dy)
            s[3] = s[5] = s[7] = s[1] = 0.0
        elif self.drag == "b2":
            dx = pos[0] - b1[0]; dy = -(pos[1] - b1[1])
            if abs(dx) + abs(dy) > 2: s[4] = math.atan2(dx, dy)
            s[3] = s[5] = s[7] = s[1] = 0.0
        elif self.drag == "b3":
            dx = pos[0] - b2[0]; dy = -(pos[1] - b2[1])
            if abs(dx) + abs(dy) > 2: s[6] = math.atan2(dx, dy)
            s[3] = s[5] = s[7] = s[1] = 0.0
        self.sim.E0 = total_energy(s)

    def update(self):
        if not self.running: return
        frame_dt = (1.0 / TARGET_FPS) * self.speed 
        dt = frame_dt / SUBSTEPS 
        
        for _ in range(SUBSTEPS):
            self.sim.step(dt)
            
        self.log_t += frame_dt
        if self.log_t >= 0.25:
            self.log_t = 0.0
            self._log()

    def _log(self):
        s = self.sim.state
        E  = total_energy(s); dE = E - self.sim.E0
        if not self.log_hdr:
            print("\n" + "="*120)
            print(f"{'t':>7} | {'KE':>10} {'PE':>10} {'E_tot':>10} |"
                  f" {'dE':>12} | {'θ1(deg)':>9} {'θ2(deg)':>9} {'θ3(deg)':>9} |"
                  f" {'ω1':>7} {'ω2':>7} {'ω3':>7} | {'u':>8} | {self.sim.mode:>10}")
            print("-"*120)
            self.log_hdr = True
        x, v, t1, w1, t2, w2, t3, w3 = s
        c1, s1 = math.cos(t1), math.sin(t1)
        c2, s2 = math.cos(t2), math.sin(t2)
        c3, s3 = math.cos(t3), math.sin(t3)
        vx1 = v+L1*w1*c1;  vy1 = -L1*w1*s1
        vx2 = vx1+L2*w2*c2; vy2 = vy1-L2*w2*s2
        vx3 = vx2+L3*w3*c3; vy3 = vy2-L3*w3*s3
        KE = 0.5*CART_MASS*v**2 + 0.5*M1*(vx1**2+vy1**2) + 0.5*M2*(vx2**2+vy2**2) + 0.5*M3*(vx3**2+vy3**2)
        PE = M1*GRAVITY*L1*c1 + M2*GRAVITY*(L1*c1+L2*c2) + M3*GRAVITY*(L1*c1+L2*c2+L3*c3)
        t1d = math.degrees(t1) % 360; t2d = math.degrees(t2) % 360; t3d = math.degrees(t3) % 360
        print(f"{self.sim.time:7.2f} | {KE:10.4f} {PE:10.4f} {E:10.4f} |"
              f" {dE:12.2e} | {t1d:9.1f} {t2d:9.1f} {t3d:9.1f} |"
              f" {w1:7.2f} {w2:7.2f} {w3:7.2f} | {self.sim.u:8.2f} |"
              f" {self.sim.mode:>10}")

    def draw(self):
        self.scr.fill(BG)
        self._draw_ground()
        self._draw_pend()
        self._draw_arrow()
        self._draw_panel()
        pygame.display.flip()

    def _draw_ground(self):
        pygame.draw.line(self.scr, GND, (0, GROUND_Y), (WIDTH, GROUND_Y), 3)
        for i in range(-10, 11):
            mx = WIDTH // 2 + i * PPM_X  # <--- CHANGE PPM TO PPM_X HERE
            if 0 <= mx <= WIDTH:
                h = 10 if i == 0 else 6 if i % 5 == 0 else 3
                pygame.draw.line(self.scr, GND, (mx, GROUND_Y), (mx, GROUND_Y + h), 1)
                if i % 5 == 0:
                    self.scr.blit(self.fns.render(f"{i}m", True, DIM), (mx - 8, GROUND_Y + 12))

    def _draw_pend(self):
        cr, piv, b1, b2, b3 = self.sim.positions()
        ip  = (int(piv[0]), int(piv[1]))
        ib1 = (int(b1[0]),  int(b1[1]))
        ib2 = (int(b2[0]),  int(b2[1]))
        ib3 = (int(b3[0]),  int(b3[1]))
        
        pygame.draw.line(self.scr, ROD, ip,  ib1, 3)
        pygame.draw.line(self.scr, ROD, ib1, ib2, 3)
        pygame.draw.line(self.scr, ROD, ib2, ib3, 3)
        
        pygame.draw.rect(self.scr, C_COL, cr, border_radius=5)
        pygame.draw.rect(self.scr, C_BORD, cr, 2, border_radius=5)
        for dx in (-CART_W // 3, CART_W // 3):
            wx = cr.centerx + dx
            pygame.draw.circle(self.scr, (70, 70, 80),  (wx, GROUND_Y), 7)
            pygame.draw.circle(self.scr, (110, 110, 120), (wx, GROUND_Y), 7, 1)
        pygame.draw.circle(self.scr, (255, 255, 255), ip, 4)
        
        pygame.draw.circle(self.scr, B1_COL, ib1, BOB_R)
        pygame.draw.circle(self.scr, (255, 130, 110), ib1, BOB_R, 2)
        pygame.draw.circle(self.scr, B2_COL, ib2, BOB_R)
        pygame.draw.circle(self.scr, (100, 180, 240), ib2, BOB_R, 2)
        pygame.draw.circle(self.scr, B3_COL, ib3, BOB_R)
        pygame.draw.circle(self.scr, (190, 130, 220), ib3, BOB_R, 2)
        
        if not self.running:
            for b in (ib1, ib2, ib3):
                pygame.draw.circle(self.scr, (70, 70, 90), b, GRAB_R, 1)

    def _draw_arrow(self):
        u = self.sim.u
        if abs(u) < 0.3: return
        cr, _, _, _, _ = self.sim.positions()
        cx, cy = cr.centerx, cr.centery
        length = u / FORCE_LIMIT * 65
        ex = cx + length
        col = GREEN if self.sim.ctrl_on else RED
        pygame.draw.line(self.scr, col, (cx, cy), (int(ex), cy), 3)
        sgn = 1 if length > 0 else -1
        pygame.draw.polygon(self.scr, col, [
            (int(ex), cy), (int(ex - sgn*9), cy - 6), (int(ex - sgn*9), cy + 6)])

    def _draw_panel(self):
        ph = 165  
        pygame.draw.rect(self.scr, PANEL, (0, HEIGHT - ph, WIDTH, ph))
        pygame.draw.line(self.scr, (50, 53, 62), (0, HEIGHT - ph), (WIDTH, HEIGHT - ph), 1)
        s = self.sim.state
        y0 = HEIGHT - ph + 10
        
        self._t(f"x  = {s[0]:+7.3f} m      v  = {s[1]:+7.3f} m/s", 20, y0, TXT)
        
        t1d = math.degrees(s[2]) % 360
        t2d = math.degrees(s[4]) % 360
        t3d = math.degrees(s[6]) % 360
        self._t(f"θ1 = {t1d:7.1f} deg   θ2 = {t2d:7.1f} deg   θ3 = {t3d:7.1f} deg", 20, y0 + 22, TXT)
        
        self._t(f"ω1 = {s[3]:+7.2f} rad/s  ω2 = {s[5]:+7.2f} rad/s  ω3 = {s[7]:+7.2f} rad/s", 20, y0 + 44, TXT)
        
        E = total_energy(s); dE = E - self.sim.E0
        self._t(f"E  = {E:8.3f} J      dE = {dE:+10.2e} J      u  = {self.sim.u:+8.2f} N      t  = {self.sim.time:7.2f} s     x{self.speed:.2f}", 20, y0 + 66, TXT)
        
        # ---- Middle Column ----
        mx = 1100
        cc = GREEN if self.sim.ctrl_on else RED
        ct = "ON" if self.sim.ctrl_on else "OFF (u=0)"
        self._t(f"Controller: {ct}", mx, y0, cc, self.fnl)
        
        mc = {"idle": DIM, "rl": GREEN, "off": RED}
        self._t(f"Mode: {self.sim.mode.upper()}", mx, y0 + 30, mc.get(self.sim.mode, TXT), self.fnl)
       
        if self.sim.mode == "rl":
            label = "BALANCE_MODEL" if self.sim.active_controller == "balance" else "SWINGUP_MODEL"
            self._t(f"Active: {label}", mx, y0 + 60, YELLOW, self.fnl)

        # ---- Right Column ----
        rx = 1900
        for i, line in enumerate([
            "SPACE   Start / Pause",
            "R       Reset (hanging)",
            "1       Reset (near upright)",
            "U       Toggle controller",
            "UP/DOWN  Speed"]):
            self._t(line, rx, y0 + i * 22, TXT, self.fn)
            
        if self.running:
            self._t("RUNNING", WIDTH - 140, y0, GREEN, self.fns)
        else:
            self._t("PAUSED", WIDTH - 140, y0, YELLOW, self.fns)

        # ---- Bottom Section: Active Control Law ----
        law_y = y0 + 112
        
        mode = self.sim.mode
        
        if mode == "rl":
            law_str = f"u = PPO_{self.sim.active_controller.upper()}(state) = {self.sim.u:+.1f} N"
            law_color = GREEN
        elif mode == "idle":
            law_str = "u = 0.0 N  (Press SPACE to start)"
            law_color = DIM
        else:
            law_str = "u = 0.0 N  (Press U to enable)"
            law_color = RED
            
        self._t(law_str, 20, law_y, law_color, self.fn)

    def _t(self, txt, x, y, c, f=None):
        self.scr.blit((f or self.fn).render(txt, True, c), (x, y))

    def loop(self):
        print("+" + "="*98 + "+")
        print("|  Triple Inverted Pendulum on Cart - PPO Best Policy                   |")
        print("+" + "="*98 + "+")
        print("|  SPACE  start/pause    R  reset (hang)   1  reset (up)                 |")
        print("|  U  toggle ctrl                                                        |")
        print("|  UP/DOWN  speed         Mouse drag bobs (paused)                       |")
        print("+" + "="*98 + "+\n")
        while True:
            self.events()
            self.update()
            self.draw()
            self.clk.tick(TARGET_FPS)


if __name__ == "__main__":
    App().loop()