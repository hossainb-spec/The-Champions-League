
import math
import random
import sys
import asyncio
from collections import deque

import pygame

pygame.init()

# Rehan - Section 3: builds the maze with DFS backtracking and sets the spawn probabilities.
CELL = 32
COLS = 21
ROWS = 17
WIDTH = COLS * CELL
HEIGHT = ROWS * CELL
HUD = 85

PROB_ENERGY = 0.05
PROB_DEFENDER = 0.05
PROB_REFEREE = 0.04
MAX_ENERGY = 3
MAX_DEFENDERS = 3
MAX_REFEREES = 2
BALLS_NEEDED = 5
EXTRA_OPENINGS = 0.40

SKY = (110, 205, 255)
CLOUD = (240, 250, 255)
GRASS_A = (72, 190, 80)
GRASS_B = (58, 168, 68)
WALL = (60, 90, 200)
WALL_TOP = (110, 145, 250)
NAVY = (25, 40, 95)
BLACK = (25, 25, 35)
WHITE = (250, 250, 250)
YELLOW = (255, 220, 40)
ORANGE = (255, 150, 30)
RED = (235, 60, 60)
BLUE = (40, 120, 255)
CYAN = (60, 225, 255)
PURPLE = (170, 80, 255)


class MazeGenerator:
    def __init__(self):
        self.grid = [[1] * COLS for _ in range(ROWS)]

    def build(self):
        grid = self.grid
        stack = [(1, 1)]
        visited = {(1, 1)}
        grid[1][1] = 0

        while stack:
            x, y = stack[-1]
            choices = []

            for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2)):
                nx = x + dx
                ny = y + dy

                if 1 <= nx < COLS - 1 and 1 <= ny < ROWS - 1 and (nx, ny) not in visited:
                    choices.append((nx, ny))

            if choices:
                nx, ny = random.choice(choices)
                grid[(y + ny) // 2][(x + nx) // 2] = 0
                grid[ny][nx] = 0
                visited.add((nx, ny))
                stack.append((nx, ny))
            else:
                stack.pop()

        walls = []

        for y in range(1, ROWS - 1):
            for x in range(1, COLS - 1):
                if grid[y][x] == 1:
                    sideways = grid[y][x - 1] == 0 and grid[y][x + 1] == 0
                    upwards = grid[y - 1][x] == 0 and grid[y + 1][x] == 0

                    if sideways or upwards:
                        walls.append((x, y))

        random.shuffle(walls)

        for x, y in walls[:int(len(walls) * EXTRA_OPENINGS)]:
            grid[y][x] = 0

        for step in range(3):
            grid[ROWS - 2][COLS - 2 - step] = 0

        return grid


def scatter(maze, goal):
    spots = []

    for y in range(ROWS):
        for x in range(COLS):
            if maze[y][x] == 0:
                if abs(x - 1) + abs(y - 1) > 4 and abs(x - goal[0]) + abs(y - goal[1]) > 3:
                    spots.append((x, y))

    random.shuffle(spots)

    balls = spots[:BALLS_NEEDED]
    energy = []
    defenders = []
    referees = []

    for spot in spots[BALLS_NEEDED:]:
        roll = random.random()

        if roll < PROB_ENERGY and len(energy) < MAX_ENERGY:
            energy.append(spot)
        elif roll < PROB_ENERGY + PROB_DEFENDER and len(defenders) < MAX_DEFENDERS:
            defenders.append(spot)
        elif roll < PROB_ENERGY + PROB_DEFENDER + PROB_REFEREE and len(referees) < MAX_REFEREES:
            referees.append(spot)

    return {"balls": balls, "energy": energy, "defenders": defenders, "referees": referees}


# Sha - Section 4: the player controls and the defender's three-state FSM with BFS pathfinding.
DIRS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}


def neighbors(cell):
    x, y = cell
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def is_open(maze, cell):
    x, y = cell
    return 0 <= x < COLS and 0 <= y < ROWS and maze[y][x] == 0


def direction_between(a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]

    if dx > 0:
        return "RIGHT"
    if dx < 0:
        return "LEFT"
    if dy > 0:
        return "DOWN"
    if dy < 0:
        return "UP"
    return None


def find_path(maze, start, target):
    queue = deque([start])
    came_from = {start: None}

    while queue:
        current = queue.popleft()

        if current == target:
            break

        for step in neighbors(current):
            if is_open(maze, step) and step not in came_from:
                came_from[step] = current
                queue.append(step)

    if target not in came_from:
        return []

    path = []
    current = target

    while current != start:
        path.append(current)
        current = came_from[current]

    return path[::-1]


class Mover:
    def __init__(self, cell, speed):
        self.col, self.row = cell
        self.x = self.col * CELL + CELL / 2
        self.y = self.row * CELL + CELL / 2
        self.speed = speed
        self.dir = None
        self.want = None
        self.target = None

    def can_move(self, maze, name):
        dx, dy = DIRS[name]
        return is_open(maze, (self.col + dx, self.row + dy))

    def cell(self):
        return (int(self.x // CELL), int(self.y // CELL))

    def stop(self):
        self.want = None
        self.dir = None

    def decide(self, maze):
        pass

    def update(self, maze, dt):
        distance = self.speed * dt

        while distance > 0:
            if self.target is None:
                self.decide(maze)

                if self.want and self.can_move(maze, self.want):
                    self.dir = self.want

                if not self.dir or not self.can_move(maze, self.dir):
                    break

                dx, dy = DIRS[self.dir]
                self.target = (self.col + dx, self.row + dy)

            tx = self.target[0] * CELL + CELL / 2
            ty = self.target[1] * CELL + CELL / 2
            gap = math.hypot(tx - self.x, ty - self.y)

            if gap <= distance:
                self.x = tx
                self.y = ty
                self.col, self.row = self.target
                self.target = None
                distance -= gap
            else:
                self.x += (tx - self.x) / gap * distance
                self.y += (ty - self.y) / gap * distance
                distance = 0


class Player(Mover):
    def __init__(self, cell):
        super().__init__(cell, 155)

    def read_keys(self, keys):
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            self.want = "UP"
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            self.want = "DOWN"
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
            self.want = "LEFT"
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            self.want = "RIGHT"
        else:
            self.want = None

    def update(self, maze, dt):
        if self.want is None:
            self.dir = None
            return

        if self.target is None and not self.can_move(maze, self.want):
            self.dir = None

        super().update(maze, dt)


class Defender(Mover):
    def __init__(self, cell, player):
        super().__init__(cell, 65)
        self.player = player
        self.state = "Patrol"
        self.last_seen = cell
        self.previous = None
        self.path = []

    def decide(self, maze):
        here = (self.col, self.row)
        prey = self.player.cell()
        gap = abs(prey[0] - here[0]) + abs(prey[1] - here[1])

        if self.state == "Patrol" and gap <= 6:
            self.state = "Chase"
        elif self.state == "Chase" and gap > 6:
            self.state = "Search"
        elif self.state == "Search":
            if gap <= 6:
                self.state = "Chase"
            elif here == self.last_seen:
                self.state = "Patrol"

        if self.state == "Chase":
            self.last_seen = prey
            self.path = find_path(maze, here, prey)
        elif self.state == "Search":
            self.path = find_path(maze, here, self.last_seen)
        else:
            options = [p for p in neighbors(here) if is_open(maze, p)]
            forward = [p for p in options if p != self.previous]
            pick = random.choice(forward or options)
            self.path = [pick]

        if self.path:
            self.previous = here
            self.want = direction_between(here, self.path[0])
        else:
            self.stop()


# Barack - Section 5: the Markov predictor and the goalkeeper's ambush scoring.
WEIGHT_A = 10
WEIGHT_B = 1


class MarkovPredictor:
    def __init__(self):
        self.names = list(DIRS.keys())
        self.counts = {a: {b: 0 for b in self.names} for a in self.names}
        self.last = None

    def record(self, direction):
        if self.last is not None and direction is not None:
            self.counts[self.last][direction] += 1

        if direction is not None:
            self.last = direction

    def predict(self):
        if self.last is None:
            return "UP", 0.25

        row = self.counts[self.last]
        total = sum(row.values())

        if total == 0:
            return self.last, 0.25

        best = max(row, key=row.get)
        return best, row[best] / total


class Goalkeeper(Mover):
    def __init__(self, cell, player, predictor, goal):
        super().__init__(cell, 55)
        self.player = player
        self.predictor = predictor
        self.goal = goal
        self.path = []
        self.predicted_cell = cell
        self.predicted_dir = "UP"
        self.chance = 0.25

    def decide(self, maze):
        here = (self.col, self.row)
        prey = self.player.cell()

        if abs(prey[0] - self.goal[0]) + abs(prey[1] - self.goal[1]) > 10:
            self.path = []
            self.stop()
            return

        self.predicted_dir, self.chance = self.predictor.predict()
        dx, dy = DIRS[self.predicted_dir]
        candidates = []

        for step in range(1, 4):
            spot = (prey[0] + dx * step, prey[1] + dy * step)

            if is_open(maze, spot):
                candidates.append(spot)

        if not candidates:
            candidates = [p for p in neighbors(prey) if is_open(maze, p)]

        best_score = -999
        best_path = []

        for spot in candidates:
            path = find_path(maze, here, spot)

            if spot != here and not path:
                continue

            score = WEIGHT_A * self.chance - WEIGHT_B * len(path)

            if score > best_score:
                best_score = score
                best_path = path
                self.predicted_cell = spot

        self.path = best_path

        if self.path:
            self.want = direction_between(here, self.path[0])
        else:
            self.stop()


# Aiden - Section 6: the HUD, win and lose rules, and the main game loop.
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT + HUD))
pygame.display.set_caption("THE CHAMPIONS LEAGUE")

FONT = pygame.font.SysFont("Arial", 20, bold=True)
SMALL = pygame.font.SysFont("Arial", 14, bold=True)
MID = pygame.font.SysFont("Arial", 30, bold=True)
BIG = pygame.font.SysFont("Arial", 58, bold=True)


def draw_text(text, font, color, x, y, center=True):
    surface = font.render(text, True, color)

    if center:
        SCREEN.blit(surface, surface.get_rect(center=(x, y)))
    else:
        SCREEN.blit(surface, (x, y))


def draw_outlined(text, font, color, x, y, thickness=3):
    for ox in (-thickness, 0, thickness):
        for oy in (-thickness, 0, thickness):
            if ox or oy:
                draw_text(text, font, BLACK, x + ox, y + oy)

    draw_text(text, font, color, x, y)


def draw_panel(rect, fill, radius=14):
    pygame.draw.rect(SCREEN, BLACK, rect.inflate(6, 6), border_radius=radius + 3)
    pygame.draw.rect(SCREEN, fill, rect, border_radius=radius)


def draw_kicker(x, y, color, look=(0, 0), gloves=None):
    x = int(x)
    y = int(y)
    pygame.draw.ellipse(SCREEN, (40, 120, 50), (x - 12, y + 9, 24, 10))

    if gloves:
        for side in (-1, 1):
            pygame.draw.circle(SCREEN, BLACK, (x + side * 15, y + 3), 7)
            pygame.draw.circle(SCREEN, gloves, (x + side * 15, y + 3), 5)

    pygame.draw.circle(SCREEN, BLACK, (x, y), 14)
    pygame.draw.circle(SCREEN, color, (x, y), 11)

    for ex in (-4, 4):
        pygame.draw.circle(SCREEN, WHITE, (int(x + ex + look[0] * 1.5), int(y - 3 + look[1] * 1.5)), 4)
        pygame.draw.circle(SCREEN, BLACK, (int(x + ex + look[0] * 2.5), int(y - 3 + look[1] * 2.5)), 2)


def draw_referee(x, y):
    draw_kicker(x, y, (45, 45, 60))
    card = pygame.Rect(int(x) + 7, int(y) - 20, 9, 12)
    pygame.draw.rect(SCREEN, BLACK, card.inflate(4, 4), border_radius=2)
    pygame.draw.rect(SCREEN, YELLOW, card, border_radius=2)


def draw_ball(x, y, radius=10):
    x = int(x)
    y = int(y)
    pygame.draw.circle(SCREEN, BLACK, (x, y), radius + 2)
    pygame.draw.circle(SCREEN, WHITE, (x, y), radius)
    pygame.draw.circle(SCREEN, BLACK, (x, y), radius // 2)

    for angle in (0, 2.1, 4.2):
        px = x + math.cos(angle) * (radius - 2)
        py = y + math.sin(angle) * (radius - 2)
        pygame.draw.circle(SCREEN, BLACK, (int(px), int(py)), 2)


def draw_can(x, y):
    x = int(x)
    y = int(y)
    rect = pygame.Rect(x - 7, y - 11, 14, 22)
    pygame.draw.rect(SCREEN, BLACK, rect.inflate(4, 4), border_radius=5)
    pygame.draw.rect(SCREEN, CYAN, rect, border_radius=4)
    bolt = [(x + 1, y - 8), (x - 4, y + 1), (x, y + 1), (x - 1, y + 8), (x + 4, y - 2), (x, y - 2)]
    pygame.draw.polygon(SCREEN, YELLOW, bolt)


def draw_sky(now):
    SCREEN.fill(SKY)

    for i in range(4):
        cx = (i * 210 + now // 40) % (WIDTH + 200) - 100
        cy = 60 + (i % 2) * 70
        pygame.draw.ellipse(SCREEN, CLOUD, (cx, cy, 120, 40))
        pygame.draw.ellipse(SCREEN, CLOUD, (cx + 25, cy - 18, 70, 40))

    pygame.draw.rect(SCREEN, GRASS_A, (0, HEIGHT + HUD - 90, WIDTH, 90))
    pygame.draw.rect(SCREEN, BLACK, (0, HEIGHT + HUD - 90, WIDTH, 4))


class Game:
    def __init__(self):
        self.state = "menu"
        self.reset()

    def reset(self):
        self.maze = MazeGenerator().build()
        self.goal = (COLS - 2, ROWS - 2)
        items = scatter(self.maze, self.goal)

        self.player = Player((1, 1))
        self.predictor = MarkovPredictor()
        self.keeper = Goalkeeper((COLS - 3, ROWS - 2), self.player, self.predictor, self.goal)
        self.defenders = [Defender(c, self.player) for c in items["defenders"]]
        self.balls = items["balls"]
        self.energy = items["energy"]
        self.referees = items["referees"]

        self.score = 0
        self.boost_until = 0
        self.start_time = 0
        self.last_cell = (1, 1)

    def begin(self):
        self.reset()
        self.state = "playing"
        self.start_time = pygame.time.get_ticks()

    def update(self, now, dt):
        self.player.read_keys(pygame.key.get_pressed())
        self.player.speed = 215 if now < self.boost_until else 155
        self.player.update(self.maze, dt)

        cell = self.player.cell()

        if cell != self.last_cell:
            self.predictor.record(direction_between(self.last_cell, cell))
            self.last_cell = cell

        for defender in self.defenders:
            defender.update(self.maze, dt)

        self.keeper.update(self.maze, dt)
        self.check_rules(now)

        if self.state == "playing" and now - self.start_time >= 180000:
            self.state = "over"

    def check_rules(self, now):
        cell = self.player.cell()

        if cell in self.balls:
            self.balls.remove(cell)
            self.score += 1

        if cell in self.energy:
            self.energy.remove(cell)
            self.boost_until = now + 5000

        for defender in self.defenders:
            if math.hypot(self.player.x - defender.x, self.player.y - defender.y) < 22:
                self.state = "over"

        for spot in self.referees:
            cx = spot[0] * CELL + CELL / 2
            cy = spot[1] * CELL + CELL / 2

            if math.hypot(self.player.x - cx, self.player.y - cy) < 21:
                self.state = "over"

        if math.hypot(self.player.x - self.keeper.x, self.player.y - self.keeper.y) < 23:
            self.state = "over"

        if self.state == "playing" and cell == self.goal and self.score == BALLS_NEEDED:
            self.state = "win"

    def draw_pitch(self):
        for y in range(ROWS):
            for x in range(COLS):
                rect = pygame.Rect(x * CELL, y * CELL, CELL, CELL)
                pygame.draw.rect(SCREEN, GRASS_A if x % 2 == 0 else GRASS_B, rect)

        pygame.draw.rect(SCREEN, WHITE, (CELL, CELL, WIDTH - 2 * CELL, HEIGHT - 2 * CELL), 3)
        pygame.draw.line(SCREEN, WHITE, (WIDTH // 2, CELL), (WIDTH // 2, HEIGHT - CELL), 3)
        pygame.draw.circle(SCREEN, WHITE, (WIDTH // 2, HEIGHT // 2), 50, 3)

        for y in range(ROWS):
            for x in range(COLS):
                if self.maze[y][x] == 1:
                    rect = pygame.Rect(x * CELL + 1, y * CELL + 1, CELL - 2, CELL - 2)
                    pygame.draw.rect(SCREEN, WALL, rect, border_radius=7)
                    pygame.draw.rect(SCREEN, WALL_TOP, (rect.x + 4, rect.y + 3, rect.w - 8, 6), border_radius=3)
                    pygame.draw.rect(SCREEN, BLACK, rect, 3, border_radius=7)

    def draw_goal(self, now):
        x, y = self.goal
        rect = pygame.Rect(x * CELL + 2, y * CELL + 2, CELL - 4, CELL - 4)

        if self.score == BALLS_NEEDED:
            glow = 14 + int(math.sin(now / 150) * 4)
            pygame.draw.circle(SCREEN, YELLOW, rect.center, glow)

        pygame.draw.rect(SCREEN, WHITE, rect, border_radius=4)
        pygame.draw.rect(SCREEN, BLACK, rect, 3, border_radius=4)

        for i in range(1, 4):
            pygame.draw.line(SCREEN, (170, 170, 180), (rect.x + i * 7, rect.y), (rect.x + i * 7, rect.bottom), 1)
            pygame.draw.line(SCREEN, (170, 170, 180), (rect.x, rect.y + i * 7), (rect.right, rect.y + i * 7), 1)

        draw_outlined("GOAL", SMALL, YELLOW, rect.centerx, rect.top - 12, 2)

    def draw_world(self, now):
        self.draw_pitch()
        self.draw_goal(now)

        for bx, by in self.balls:
            bob = math.sin(now / 220 + bx) * 2
            draw_ball(bx * CELL + CELL / 2, by * CELL + CELL / 2 + bob)

        for ex, ey in self.energy:
            draw_can(ex * CELL + CELL / 2, ey * CELL + CELL / 2)

        for rx, ry in self.referees:
            px = rx * CELL + CELL / 2
            py = ry * CELL + CELL / 2
            draw_referee(px, py)
            draw_outlined("REF", SMALL, WHITE, px, py - 24, 2)

        for defender in self.defenders:
            draw_kicker(defender.x, defender.y, RED, DIRS.get(defender.dir, (0, 0)))
            draw_outlined("DEF", SMALL, WHITE, defender.x, defender.y - 24, 2)

        keeper = self.keeper
        draw_kicker(keeper.x, keeper.y, YELLOW, DIRS.get(keeper.dir, (0, 0)), ORANGE)
        draw_outlined("GK", SMALL, WHITE, keeper.x, keeper.y - 24, 2)

        player = self.player
        draw_kicker(player.x, player.y, BLUE, DIRS.get(player.dir, (0, 0)))
        draw_outlined("YOU", SMALL, WHITE, player.x, player.y - 24, 2)

    def draw_hud(self, now):
        pygame.draw.rect(SCREEN, NAVY, (0, HEIGHT, WIDTH, HUD))
        pygame.draw.rect(SCREEN, YELLOW, (0, HEIGHT, WIDTH, 5))

        for i in range(BALLS_NEEDED):
            cx = 30 + i * 32
            cy = HEIGHT + 28

            if i < self.score:
                draw_ball(cx, cy, 11)
            else:
                pygame.draw.circle(SCREEN, (70, 85, 140), (cx, cy), 11)
                pygame.draw.circle(SCREEN, BLACK, (cx, cy), 11, 2)

        if now < self.boost_until:
            ratio = (self.boost_until - now) / 5000
            bar = pygame.Rect(215, HEIGHT + 18, 170, 20)
            draw_panel(bar, NAVY, 8)
            pygame.draw.rect(SCREEN, CYAN, (bar.x, bar.y, int(bar.w * ratio), bar.h), border_radius=8)
            draw_text("BOOST", SMALL, WHITE, bar.centerx, bar.centery)

        remaining = max(0, 180 - (now - self.start_time) // 1000)
        pill = pygame.Rect(WIDTH - 160, HEIGHT + 12, 145, 34)
        draw_panel(pill, YELLOW, 14)
        draw_text("TIME " + str(remaining // 60) + ":" + str(remaining % 60).zfill(2), FONT, BLACK, pill.centerx, pill.centery)

        draw_text("WASD / ARROWS: Move     R: Restart     ESC: Quit", SMALL, WHITE, WIDTH // 2, HEIGHT + 66)

    def draw(self, now):
        self.draw_world(now)
        self.draw_hud(now)

    def draw_menu(self, now):
        draw_sky(now)

        bounce = math.sin(now / 300) * 4
        draw_outlined("THE CHAMPIONS", BIG, YELLOW, WIDTH // 2, 75 + bounce, 4)
        draw_outlined("LEAGUE", BIG, WHITE, WIDTH // 2, 135 + bounce, 4)

        panel = pygame.Rect(60, 190, WIDTH - 120, 120)
        draw_panel(panel, WHITE)
        draw_text("Collect all 5 soccer balls and reach the goal.", FONT, BLACK, WIDTH // 2, 220)
        draw_text("Avoid referees, defenders and the goalkeeper.", FONT, BLACK, WIDTH // 2, 255)
        draw_text("Energy drinks give you a 5-second speed boost.", FONT, BLACK, WIDTH // 2, 290)

        draw_outlined("WASD / ARROWS - Move", SMALL, WHITE, WIDTH // 2, 345, 2)
        draw_outlined("R - Restart     ESC - Quit", SMALL, WHITE, WIDTH // 2, 370, 2)

        draw_kicker(110, 440, BLUE, (1, 0))
        draw_kicker(260, 440, RED, (1, 0))
        draw_referee(410, 440)
        draw_kicker(560, 440, YELLOW, (-1, 0), ORANGE)

        draw_outlined("YOU", SMALL, WHITE, 110, 475, 2)
        draw_outlined("DEFENDER", SMALL, WHITE, 260, 475, 2)
        draw_outlined("REFEREE", SMALL, WHITE, 410, 475, 2)
        draw_outlined("GOALKEEPER", SMALL, WHITE, 560, 475, 2)

        button = pygame.Rect(0, 0, 300, 64)
        button.center = (WIDTH // 2, 555 + int(math.sin(now / 200) * 3))
        draw_panel(button, ORANGE, 24)
        draw_text("PRESS ENTER TO PLAY", FONT, WHITE, button.centerx, button.centery)

    def draw_end(self, now):
        draw_sky(now)

        if self.state == "win":
            draw_outlined("GOAL!", BIG, YELLOW, WIDTH // 2, 190, 4)
            draw_outlined("YOU WIN!", MID, WHITE, WIDTH // 2, 255, 3)
        else:
            shade = pygame.Surface((WIDTH, HEIGHT + HUD))
            shade.set_alpha(110)
            shade.fill(NAVY)
            SCREEN.blit(shade, (0, 0))
            draw_outlined("GAME OVER", BIG, RED, WIDTH // 2, 190, 4)

        draw_ball(WIDTH // 2, 320, 24)
        draw_outlined(str(self.score) + " / " + str(BALLS_NEEDED) + " BALLS", FONT, WHITE, WIDTH // 2, 370, 2)

        button = pygame.Rect(0, 0, 420, 56)
        button.center = (WIDTH // 2, 490)
        draw_panel(button, ORANGE, 22)
        draw_text("Press R to play again or ESC to quit", FONT, WHITE, button.centerx, button.centery)


async def main():
    clock = pygame.time.Clock()
    game = Game()
    running = True

    while running:
        dt = min(clock.tick(60) / 1000, 0.03)
        now = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False

                if event.key == pygame.K_RETURN and game.state == "menu":
                    game.begin()

                if event.key == pygame.K_r and game.state != "menu":
                    game.begin()

        if game.state == "menu":
            game.draw_menu(now)
        elif game.state == "playing":
            game.update(now, dt)
            game.draw(now)
        else:
            game.draw_end(now)

        pygame.display.flip()
        await asyncio.sleep(0)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    asyncio.run(main())
