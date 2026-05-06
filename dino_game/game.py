"""
IMU Dino Jump — Pygame Game
----------------------------
Two screens in one window:

  GAME SCREEN (default)
  ─────────────────────
  Chrome-dino-style endless runner controlled by your IMU.
  • Dino jumps when the jump detector fires (single jump = one dino jump).
  • HUD shows current activity label (updated every ~0.4 s) + last jump height.
  • Press TAB to switch to the Analyze screen.

  ANALYZE SCREEN (press TAB)
  ──────────────────────────
  Records a 5-second window of IMU data, then classifies it and displays
  a colour-coded bar chart of the activity distribution.
  • Countdown 3 → 2 → 1 → RECORDING → ANALYSING → RESULT
  • Press TAB again (or SPACE) to go back to the game.

Modes:
  --demo     Keyboard-only.  Space = jump.  No IMU / no model needed.
  --port COM6  Live IMU mode (requires trained model).

Run:
  python main.py play --demo
  python main.py play --port COM6
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from collections import deque
from typing import Dict, List, Optional

try:
    import pygame
except ImportError:
    print("ERROR: pygame not installed.  Run:  pip install pygame")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Layout & palette
# ---------------------------------------------------------------------------
FPS        = 60
W, H       = 900, 440
GROUND_Y   = 360

SKY_COLOR      = (247, 247, 247)
GROUND_COLOR   = (83,  83,  83)
DINO_COLOR     = (83,  83,  83)
CACTUS_COLOR   = (83,  83,  83)
CLOUD_COLOR    = (210, 210, 210)
TEXT_DARK      = (50,  50,  50)
TEXT_GRAY      = (120, 120, 120)
WHITE          = (255, 255, 255)
BLACK          = (0,   0,   0)
ACCENT_GREEN   = (22,  163,  74)
ACCENT_ORANGE  = (234,  88,  12)
ACCENT_BLUE    = (37,   99, 235)
ACCENT_RED     = (220,  38,  38)
ACCENT_PURPLE  = (147,  51, 234)

ACTIVITY_COLORS: Dict[str, tuple] = {
    "still":    TEXT_GRAY,
    "walk":     ACCENT_BLUE,
    "run":      ACCENT_ORANGE,
    "jump":     ACCENT_GREEN,
    "freefall": ACCENT_GREEN,
    "demo":     TEXT_DARK,
}
# Bar colours for the Analyze chart (same palette, fixed order)
CHART_COLORS = [ACCENT_ORANGE, ACCENT_BLUE, ACCENT_GREEN, ACCENT_PURPLE,
                ACCENT_RED, TEXT_GRAY]

DINO_W, DINO_H = 44, 50
DINO_X         = 110

GAME_GRAVITY   = 1800   # px / s²
PX_PER_CM      = 3.2
MAX_JUMP_PX    = 200
MIN_JUMP_PX    = 80
DEMO_JUMP_CM   = 40.0

BASE_SPEED     = 280
SPEED_INC      = 12
MAX_SPEED      = 650

OBS_MIN_GAP    = 1.4    # s
OBS_MAX_GAP    = 2.8    # s

ANALYZE_RECORD_S  = 5.0   # seconds to record in Analyze mode
ANALYZE_COUNTDOWN = 3     # countdown seconds before recording


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _rect(surf, color, x, y, w, h):
    pygame.draw.rect(surf, color, (int(x), int(y), int(w), int(h)))


def draw_dino(surf, cx, bottom_y, color=DINO_COLOR, frame=0, dead=False):
    _rect(surf, color, cx-34, bottom_y-30, 14,  8)   # tail far
    _rect(surf, color, cx-44, bottom_y-22, 12,  6)   # tail tip
    _rect(surf, color, cx-18, bottom_y-44, 32, 30)   # body
    _rect(surf, color, cx+4,  bottom_y-52, 16, 12)   # neck
    _rect(surf, color, cx,    bottom_y-52, 26, 18)   # head
    _rect(surf, color, cx,    bottom_y-36, 26,  5)   # jaw
    _rect(surf, color, cx-4,  bottom_y-28, 14,  5)   # arm
    _rect(surf, WHITE, cx+12, bottom_y-50,  8,  8)   # eye white
    _rect(surf, BLACK, cx+14, bottom_y-48,  4,  5)   # pupil
    if dead:
        _rect(surf, color, cx-16, bottom_y-12, 12, 12)
        _rect(surf, color, cx+4,  bottom_y-12, 12, 12)
    elif frame == 0:
        _rect(surf, color, cx-14, bottom_y-16, 11, 16)
        _rect(surf, color, cx+4,  bottom_y-10, 11, 10)
    else:
        _rect(surf, color, cx-14, bottom_y-10, 11, 10)
        _rect(surf, color, cx+4,  bottom_y-16, 11, 16)


def dino_hitbox(cx, bottom_y):
    s = 6
    return pygame.Rect(cx-18+s, bottom_y-44+s, 32-2*s, 44-s)


def draw_cactus(surf, x, ground_y, variant=0):
    c = CACTUS_COLOR
    if variant == 0:
        _rect(surf, c, x+6,  ground_y-55, 16, 55)
        _rect(surf, c, x-10, ground_y-40, 18, 10)
        _rect(surf, c, x-10, ground_y-55, 10, 18)
        _rect(surf, c, x+22, ground_y-38, 18, 10)
        _rect(surf, c, x+28, ground_y-55, 10, 20)
    elif variant == 1:
        _rect(surf, c, x+4,  ground_y-45, 14, 45)
        _rect(surf, c, x-8,  ground_y-34, 14,  8)
        _rect(surf, c, x-8,  ground_y-44, 10, 14)
        _rect(surf, c, x+22, ground_y-65, 14, 65)
        _rect(surf, c, x+36, ground_y-50, 14,  8)
        _rect(surf, c, x+36, ground_y-60, 10, 14)
    else:
        for ox, oh in [(-2, 50), (16, 70), (34, 55)]:
            _rect(surf, c, x+ox,    ground_y-oh, 12, oh)
            _rect(surf, c, x+ox-12, ground_y-oh+15, 14, 8)
            _rect(surf, c, x+ox+12, ground_y-oh+15, 14, 8)


def cactus_hitbox(x, ground_y, variant):
    widths  = [28, 48, 58]
    heights = [55, 65, 70]
    return pygame.Rect(x, ground_y - heights[variant], widths[variant], heights[variant])


def draw_cloud(surf, x, y):
    pygame.draw.ellipse(surf, CLOUD_COLOR, (x,    y+8,  50, 22))
    pygame.draw.ellipse(surf, CLOUD_COLOR, (x+10, y,    40, 28))
    pygame.draw.ellipse(surf, CLOUD_COLOR, (x+30, y+5,  40, 24))


# ---------------------------------------------------------------------------
# Game entities
# ---------------------------------------------------------------------------

class Dino:
    def __init__(self):
        self.cx       = DINO_X
        self.bottom_y = float(GROUND_Y)
        self.vy       = 0.0
        self.on_ground = True
        self.dead     = False
        self._frame   = 0
        self._ftimer  = 0.0

    @property
    def hitbox(self):
        return dino_hitbox(self.cx, int(self.bottom_y))

    def jump(self, height_cm: float):
        if not self.on_ground or self.dead:
            return
        px = max(MIN_JUMP_PX, min(height_cm * PX_PER_CM, MAX_JUMP_PX))
        self.vy = -math.sqrt(2.0 * GAME_GRAVITY * px)
        self.on_ground = False

    def update(self, dt):
        if self.dead:
            return
        if not self.on_ground:
            self.vy       += GAME_GRAVITY * dt
            self.bottom_y += self.vy * dt
            if self.bottom_y >= GROUND_Y:
                self.bottom_y  = GROUND_Y
                self.vy        = 0.0
                self.on_ground = True
        if self.on_ground:
            self._ftimer += dt
            if self._ftimer >= 0.12:
                self._ftimer = 0.0
                self._frame  = 1 - self._frame

    def draw(self, surf):
        frame = 0 if not self.on_ground else self._frame
        draw_dino(surf, self.cx, int(self.bottom_y),
                  color=ACCENT_RED if self.dead else DINO_COLOR,
                  frame=frame, dead=self.dead)


class Obstacle:
    def __init__(self, x, variant):
        self.x, self.variant = x, variant

    @property
    def hitbox(self):
        return cactus_hitbox(self.x, GROUND_Y, self.variant)

    def update(self, speed, dt):
        self.x -= speed * dt

    def draw(self, surf):
        draw_cactus(surf, self.x, GROUND_Y, self.variant)

    @property
    def off_screen(self):
        return self.x < -120


class Cloud:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def update(self, dt):
        self.x -= 40 * dt

    def draw(self, surf):
        draw_cloud(surf, int(self.x), int(self.y))

    @property
    def off_screen(self):
        return self.x < -100


# ---------------------------------------------------------------------------
# Analyze screen state machine
# ---------------------------------------------------------------------------

class AnalyzeScreen:
    """
    States: "countdown" → "recording" → "analysing" → "result"
    """

    RECORD_S    = ANALYZE_RECORD_S
    COUNTDOWN_S = ANALYZE_COUNTDOWN

    def __init__(self, classifier, demo_mode: bool):
        self.clf       = classifier
        self.demo_mode = demo_mode
        self._state    = "countdown"
        self._timer    = float(self.COUNTDOWN_S)
        self._recorded: List[float] = []
        self._results:  Dict[str, float] = {}
        self._result_label = ""

    # ---- update -----------------------------------------------------------

    def update(self, dt: float):
        if self._state == "countdown":
            self._timer -= dt
            if self._timer <= 0:
                self._state  = "recording"
                self._timer  = self.RECORD_S
                self._recorded.clear()

        elif self._state == "recording":
            self._timer -= dt
            if self._timer <= 0:
                self._state = "analysing"

        elif self._state == "analysing":
            self._run_analysis()
            self._state = "result"

    def _run_analysis(self):
        if self.demo_mode or self.clf is None:
            self._results = {"walk": 0.55, "run": 0.30, "still": 0.10, "jump": 0.05}
            self._result_label = "walk"
            return

        try:
            from src.realtime_classifier import classify_buffer
            ax_buf, ay_buf, az_buf = self.clf.snapshot_axes()
            if len(ax_buf) < 50:
                self._results = {}
                self._result_label = "no data"
                return
            self._results = classify_buffer(
                ax_buf, ay_buf, az_buf,
                self.clf.clf, self.clf.le,
            )
            self._result_label = next(iter(self._results), "no data")
        except Exception as exc:
            print(f"[analyze] Classification error: {exc}")
            self._results = {}
            self._result_label = "error"

    # ---- draw -------------------------------------------------------------

    def draw(self, surf, fonts):
        font_lg, font_md, font_sm, font_xs = fonts

        surf.fill(SKY_COLOR)

        # Panel background
        panel = pygame.Rect(W//2 - 340, 40, 680, H - 80)
        pygame.draw.rect(surf, WHITE, panel, border_radius=14)
        pygame.draw.rect(surf, CLOUD_COLOR, panel, width=2, border_radius=14)

        # Title
        title = font_md.render("ACTIVITY ANALYSER", True, TEXT_DARK)
        surf.blit(title, (W//2 - title.get_width()//2, 60))

        cx = W // 2

        if self._state == "countdown":
            n = max(1, int(math.ceil(self._timer)))
            big = font_lg.render(str(n), True, ACCENT_ORANGE)
            surf.blit(big, (cx - big.get_width()//2, H//2 - 60))
            sub = font_sm.render("Get ready...", True, TEXT_GRAY)
            surf.blit(sub, (cx - sub.get_width()//2, H//2 + 20))

        elif self._state == "recording":
            rec = font_md.render("● RECORDING", True, ACCENT_RED)
            surf.blit(rec, (cx - rec.get_width()//2, 130))
            elapsed = self.RECORD_S - self._timer
            pct     = elapsed / self.RECORD_S
            bar_w   = 480
            bar_x   = cx - bar_w // 2
            bar_y   = 190
            pygame.draw.rect(surf, CLOUD_COLOR, (bar_x, bar_y, bar_w, 22), border_radius=6)
            pygame.draw.rect(surf, ACCENT_RED,
                             (bar_x, bar_y, int(bar_w * pct), 22), border_radius=6)
            t_surf = font_sm.render(
                f"{elapsed:.1f}s / {self.RECORD_S:.0f}s", True, TEXT_GRAY
            )
            surf.blit(t_surf, (cx - t_surf.get_width()//2, bar_y + 30))

            hint = font_xs.render(
                "Perform the activity you want to classify", True, TEXT_GRAY
            )
            surf.blit(hint, (cx - hint.get_width()//2, bar_y + 60))

        elif self._state == "analysing":
            a = font_md.render("Analysing…", True, ACCENT_BLUE)
            surf.blit(a, (cx - a.get_width()//2, H//2 - 20))

        elif self._state == "result":
            self._draw_result(surf, fonts)

    def _draw_result(self, surf, fonts):
        font_lg, font_md, font_sm, font_xs = fonts
        cx  = W // 2
        bar_w = 460

        # Dominant label
        lbl_color = ACTIVITY_COLORS.get(self._result_label, TEXT_DARK)
        lbl_surf  = font_md.render(
            f"Result: {self._result_label.upper()}", True, lbl_color
        )
        surf.blit(lbl_surf, (cx - lbl_surf.get_width()//2, 110))

        # Bar chart
        y = 165
        for i, (label, frac) in enumerate(self._results.items()):
            color = CHART_COLORS[i % len(CHART_COLORS)]

            # Label name
            name_surf = font_sm.render(label.capitalize(), True, TEXT_DARK)
            surf.blit(name_surf, (cx - bar_w//2, y + 4))

            # Bar background
            bx = cx - bar_w//2 + 90
            pygame.draw.rect(surf, CLOUD_COLOR, (bx, y, bar_w - 90, 26), border_radius=5)

            # Filled bar
            filled_w = int((bar_w - 90) * frac)
            if filled_w > 0:
                pygame.draw.rect(surf, color, (bx, y, filled_w, 26), border_radius=5)

            # Percentage text
            pct_surf = font_sm.render(f"{frac*100:.0f}%", True, TEXT_DARK)
            surf.blit(pct_surf, (bx + bar_w - 90 + 8, y + 4))

            y += 42

        # Back hint
        back = font_xs.render("Press  TAB  or  SPACE  to return to game", True, TEXT_GRAY)
        surf.blit(back, (cx - back.get_width()//2, H - 80))


# ---------------------------------------------------------------------------
# Main Game class
# ---------------------------------------------------------------------------

class DinoGame:
    """
    screen_mode: "game" | "analyze"
    game state:  "title" | "running" | "dead"
    """

    def __init__(self, classifier=None, demo_mode: bool = False):
        pygame.init()
        pygame.display.set_caption("IMU Dino Jump")
        self.screen     = pygame.display.set_mode((W, H))
        self.clock      = pygame.time.Clock()
        self.classifier = classifier
        self.demo_mode  = demo_mode

        self.font_lg = pygame.font.SysFont("Arial", 44, bold=True)
        self.font_md = pygame.font.SysFont("Arial", 26, bold=True)
        self.font_sm = pygame.font.SysFont("Arial", 20)
        self.font_xs = pygame.font.SysFont("Arial", 16)
        self._fonts  = (self.font_lg, self.font_md, self.font_sm, self.font_xs)

        self.screen_mode   = "game"
        self.analyze: Optional[AnalyzeScreen] = None

        self._reset_game()

    # ------------------------------------------------------------------
    # Game state
    # ------------------------------------------------------------------

    def _reset_game(self):
        self.game_state      = "title"
        self.dino            = Dino()
        self.obstacles:  List[Obstacle] = []
        self.clouds:     List[Cloud]    = []
        self.score           = 0.0
        self.speed           = BASE_SPEED
        self.elapsed         = 0.0
        self._next_obstacle  = random.uniform(OBS_MIN_GAP, OBS_MAX_GAP)
        self._ground_offset  = 0.0
        self.current_activity = "demo" if self.demo_mode else "still"
        self.last_jump_cm:   Optional[float] = None
        for _ in range(3):
            self.clouds.append(Cloud(
                random.randint(100, W - 50),
                random.randint(40, 130),
            ))

    def _start_game(self):
        self.game_state     = "running"
        self.score          = 0.0
        self.elapsed        = 0.0
        self.speed          = BASE_SPEED
        self._next_obstacle = random.uniform(OBS_MIN_GAP, OBS_MAX_GAP)
        self.obstacles.clear()
        self.dino = Dino()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        running = True
        while running:
            dt = min(self.clock.tick(FPS) / 1000.0, 0.05)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    self._handle_key(event.key)

            self._poll_classifier()

            if self.screen_mode == "game":
                if self.game_state == "running":
                    self._update_game(dt)
                self._draw_game()
            else:
                self.analyze.update(dt)
                self.analyze.draw(self.screen, self._fonts)

            pygame.display.flip()

        pygame.quit()

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    def _handle_key(self, key):
        if key == pygame.K_ESCAPE:
            pygame.event.post(pygame.event.Event(pygame.QUIT))

        # TAB switches between game and analyze screens
        if key == pygame.K_TAB:
            if self.screen_mode == "game":
                self.screen_mode = "analyze"
                self.analyze = AnalyzeScreen(self.classifier, self.demo_mode)
            else:
                self.screen_mode = "game"
            return

        # Analyze screen: SPACE goes back to game
        if self.screen_mode == "analyze":
            if key in (pygame.K_SPACE, pygame.K_RETURN) and \
                    self.analyze._state == "result":
                self.screen_mode = "game"
            return

        # Game screen keys
        if self.game_state == "title":
            if key in (pygame.K_SPACE, pygame.K_UP, pygame.K_RETURN):
                self._start_game()

        elif self.game_state == "running":
            if self.demo_mode and key in (pygame.K_SPACE, pygame.K_UP):
                self.dino.jump(DEMO_JUMP_CM)
                self.last_jump_cm = DEMO_JUMP_CM

        elif self.game_state == "dead":
            if key in (pygame.K_r, pygame.K_SPACE, pygame.K_RETURN):
                self._reset_game()
                self._start_game()

    # ------------------------------------------------------------------
    # IMU events
    # ------------------------------------------------------------------

    def _poll_classifier(self):
        if self.classifier is None:
            return
        while True:
            event = self.classifier.get_event()
            if event is None:
                break
            etype = event.get("type")

            if etype == "activity":
                self.current_activity = event["label"]

            elif etype == "jump":
                h = event["height_cm"]
                self.last_jump_cm = h
                if self.screen_mode == "game":
                    if self.game_state == "title":
                        self._start_game()
                    elif self.game_state == "running":
                        self.dino.jump(h)
                    elif self.game_state == "dead":
                        self._reset_game()
                        self._start_game()

    # ------------------------------------------------------------------
    # Game update
    # ------------------------------------------------------------------

    def _update_game(self, dt):
        self.elapsed += dt
        self.score   += dt * (self.speed / BASE_SPEED) * 10
        self.speed    = min(BASE_SPEED + self.elapsed * SPEED_INC, MAX_SPEED)

        self.dino.update(dt)
        self._ground_offset = (self._ground_offset + self.speed * dt) % 30

        self._next_obstacle -= dt
        if self._next_obstacle <= 0:
            variant = random.choices([0, 1, 2], weights=[5, 3, 1])[0]
            self.obstacles.append(Obstacle(W + 20, variant))
            self._next_obstacle = random.uniform(
                OBS_MIN_GAP * (BASE_SPEED / self.speed),
                OBS_MAX_GAP * (BASE_SPEED / self.speed),
            )

        for obs in self.obstacles:
            obs.update(self.speed, dt)
        self.obstacles = [o for o in self.obstacles if not o.off_screen]

        dino_hb = self.dino.hitbox
        for obs in self.obstacles:
            if dino_hb.colliderect(obs.hitbox):
                self.dino.dead  = True
                self.game_state = "dead"
                break

        for cloud in self.clouds:
            cloud.update(dt)
        self.clouds = [c for c in self.clouds if not c.off_screen]
        if len(self.clouds) < 4 and random.random() < 0.008:
            self.clouds.append(Cloud(W + 20, random.randint(40, 130)))

    # ------------------------------------------------------------------
    # Drawing — game screen
    # ------------------------------------------------------------------

    def _draw_game(self):
        self.screen.fill(SKY_COLOR)

        for cloud in self.clouds:
            cloud.draw(self.screen)

        pygame.draw.rect(self.screen, GROUND_COLOR, (0, GROUND_Y, W, 3))
        dash_w = 18
        for i in range(0, W // 30 + 2):
            x = int(i * 30 - self._ground_offset)
            pygame.draw.rect(self.screen, CLOUD_COLOR, (x, GROUND_Y + 6, dash_w, 2))

        for obs in self.obstacles:
            obs.draw(self.screen)

        self.dino.draw(self.screen)
        self._draw_hud()

        if self.game_state == "title":
            self._draw_title()
        elif self.game_state == "dead":
            self._draw_game_over()

    def _draw_hud(self):
        act       = self.current_activity.upper()
        act_color = ACTIVITY_COLORS.get(self.current_activity, TEXT_DARK)
        lbl       = self.font_md.render(f"Activity: {act}", True, act_color)
        self.screen.blit(lbl, (16, 14))

        if self.last_jump_cm is not None:
            jh = self.font_sm.render(
                f"Last jump: {self.last_jump_cm:.1f} cm", True, ACCENT_GREEN
            )
            self.screen.blit(jh, (16, 46))

        score_s = self.font_md.render(f"{int(self.score):05d}", True, TEXT_GRAY)
        self.screen.blit(score_s, (W - score_s.get_width() - 16, 14))

        # TAB hint (small, top centre)
        tab_hint = self.font_xs.render("TAB → Analyse", True, CLOUD_COLOR)
        self.screen.blit(tab_hint, (W//2 - tab_hint.get_width()//2, 8))

        if self.demo_mode:
            dm = self.font_xs.render("DEMO — Space to jump", True, TEXT_GRAY)
            self.screen.blit(dm, (W//2 - dm.get_width()//2, 26))

    def _draw_title(self):
        ov = pygame.Surface((W, H), pygame.SRCALPHA)
        ov.fill((247, 247, 247, 180))
        self.screen.blit(ov, (0, 0))

        t = self.font_lg.render("IMU DINO JUMP", True, TEXT_DARK)
        self.screen.blit(t, (W//2 - t.get_width()//2, H//2 - 90))
        draw_dino(self.screen, W//2, H//2 - 10, frame=0)

        if self.demo_mode:
            sub = self.font_sm.render("Press  SPACE  to start", True, TEXT_GRAY)
        else:
            sub = self.font_sm.render(
                "Press  SPACE  or  JUMP  to start", True, TEXT_GRAY
            )
        self.screen.blit(sub, (W//2 - sub.get_width()//2, H//2 + 60))

        hint = self.font_xs.render(
            "TAB = Analyse activity  |  ESC = quit", True, CLOUD_COLOR
        )
        self.screen.blit(hint, (W//2 - hint.get_width()//2, H//2 + 92))

    def _draw_game_over(self):
        ov = pygame.Surface((W, H), pygame.SRCALPHA)
        ov.fill((247, 247, 247, 180))
        self.screen.blit(ov, (0, 0))

        go = self.font_lg.render("GAME OVER", True, ACCENT_RED)
        self.screen.blit(go, (W//2 - go.get_width()//2, H//2 - 70))

        sc = self.font_md.render(f"Score: {int(self.score)}", True, TEXT_DARK)
        self.screen.blit(sc, (W//2 - sc.get_width()//2, H//2 - 14))

        if self.last_jump_cm is not None:
            jh = self.font_sm.render(
                f"Best jump: {self.last_jump_cm:.1f} cm", True, ACCENT_GREEN
            )
            self.screen.blit(jh, (W//2 - jh.get_width()//2, H//2 + 26))

        rs = self.font_sm.render(
            "R / SPACE to restart  |  TAB to analyse", True, TEXT_GRAY
        )
        self.screen.blit(rs, (W//2 - rs.get_width()//2, H//2 + 60))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="IMU Dino Jump Game")
    parser.add_argument("--demo",  action="store_true",
                        help="Keyboard demo mode — no IMU needed")
    parser.add_argument("--port",  type=str, default=None,
                        help="Serial port for live IMU (e.g. COM6)")
    parser.add_argument("--baud",  type=int, default=115200)
    parser.add_argument("--model", type=str, default="data/model.joblib")
    args = parser.parse_args()

    classifier = None
    if not args.demo:
        if args.port is None:
            print("ERROR: Provide --port COMX or use --demo.")
            sys.exit(1)
        if not os.path.exists(args.model):
            print(f"ERROR: Model not found at '{args.model}'.")
            print("  Train: python main.py train --still ... --walk ... --run ... --jump ...")
            print("  Demo:  python main.py play --demo")
            sys.exit(1)
        from src.realtime_classifier import RealtimeClassifier
        classifier = RealtimeClassifier(
            port=args.port, baud=args.baud, model_path=args.model
        )
        classifier.start()

    game = DinoGame(
        classifier=classifier,
        demo_mode=args.demo or (args.port is None),
    )
    try:
        game.run()
    finally:
        if classifier:
            classifier.stop()


if __name__ == "__main__":
    main()
