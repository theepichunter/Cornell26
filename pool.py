import pymunk
import pygame
import pymunk.pygame_util
import math
from pygame.locals import QUIT, KEYDOWN,  K_q, K_ESCAPE
import numpy as np
import matplotlib.pyplot as plt
import itertools
import random
import sys

import tkinter as tk
from tkinter import ttk

BALL_ELASTICITY = 0
FRICTION_COEFF = 0

if len(sys.argv) >= 3:
    BALL_ELASTICITY = float(sys.argv[1])
    if not (BALL_ELASTICITY >= 0 and BALL_ELASTICITY <= 1):
        raise ValueError("Elasticity must be between 0 and 1 (inclusive)")
    FRICTION_COEFF = float(sys.argv[2])
    if not (FRICTION_COEFF >= 0 and FRICTION_COEFF <= 1):
        raise ValueError("Coefficient of Friction must be between 0 and 1 (inclusive)")
    FRICTION_ACCEL = FRICTION_COEFF * 2
else:
    BALL_ELASTICITY = 1
    FRICTION_ACCEL = 1

BALL_MASS = 1
BALL_RADIUS = 25
BALL_FRICTION = 0

WORLD_DIMS = (1200, 600)
GATE_ELASTICITY = 1
GATE_FRICTION = 0


class Ball:
    def __init__(self, position, design, color, number, collision_type=1):
        ball_body = pymunk.Body(mass=BALL_MASS, moment=math.inf)
        # ball_body = pymunk.Body(mass=BALL_MASS, moment=pymunk.moment_for_circle(BALL_MASS, 0, BALL_RADIUS))
        ball_body.position = position
        ball_shape = pymunk.Circle(ball_body, radius=BALL_RADIUS)
        ball_shape.friction = BALL_FRICTION
        ball_shape.elasticity = BALL_ELASTICITY
        ball_shape.collision_type = collision_type
        self.velocity = pymunk.Vec2d(0, 0)
        self.shape = ball_shape
        self.design = design  # either solid (0), striped (1), or eight ball (8), or cue ball (-1)
        if self.design == 0:
            self.shape.color = (169, 0, 0, 255)
        if self.design == 1:
            self.shape.color = (0, 0, 169, 255)
        if self.design == 8:
            self.shape.color = (0, 0, 0, 255)
        if self.design == -1:
            self.shape.color = (255, 255, 255, 255)
        self.color = color
        self.number = number


    def set_position(self, position):
        self.shape.body.position = self.set_position(position)

class CueBall(Ball):
    def __init__(self, position, design=-1, collision_type=2):
        super().__init__(position, design=design, color=(255, 255, 255, 255), number=-1, collision_type=collision_type)

def initialize_border(space):
    t = pymunk.Body(body_type=pymunk.Body.STATIC)
    t.elasticity = GATE_ELASTICITY
    t.friction = GATE_FRICTION
    t.position = (WORLD_DIMS[0]/2, WORLD_DIMS[1])
    ts = pymunk.Poly.create_box(body=t, size=(WORLD_DIMS[0], BALL_RADIUS))
    ts.collision_type = 3
    ts.color = (75, 55, 28, 255)

    b = pymunk.Body(body_type=pymunk.Body.STATIC)
    b.elasticity = GATE_ELASTICITY
    b.friction = GATE_FRICTION
    b.position = (WORLD_DIMS[0]/2, 0)
    bs = pymunk.Poly.create_box(body=b, size=(WORLD_DIMS[0], BALL_RADIUS))
    bs.collision_type = 3
    bs.color = (75, 55, 28, 255)

    l = pymunk.Body(body_type=pymunk.Body.STATIC)
    l.elasticity = GATE_ELASTICITY
    l.friction = GATE_FRICTION
    l.position = (0, WORLD_DIMS[1]/2)
    ls = pymunk.Poly.create_box(body=l, size=(BALL_RADIUS, WORLD_DIMS[1]))
    ls.collision_type = 4
    ls.color = (75, 55, 28, 255)

    r = pymunk.Body(body_type=pymunk.Body.STATIC)
    r.elasticity = GATE_ELASTICITY
    r.friction = GATE_FRICTION
    r.position = (WORLD_DIMS[0], WORLD_DIMS[1]/2)
    rs = pymunk.Poly.create_box(body=r, size=(BALL_RADIUS, WORLD_DIMS[1]))
    rs.collision_type = 4
    rs.color = (75, 55, 28, 255)

    space.add(t, b, l, r)
    space.add(ts, bs, ls, rs)



def initialize_holes(space):
    for i in range(6):
        hole = bot_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        hole.position = (WORLD_DIMS[0] * ((i % 3)/2), WORLD_DIMS[1] * (i//3))
        hole_shape = pymunk.Circle(bot_body, radius=BALL_RADIUS * 1.5)
        hole_shape.collision_type = 5
        hole_shape.color = (0, 0, 0, 255)
        space.add(hole, hole_shape)


def detect_collisions(b1, b2):
    return np.linalg.norm(b1.shape.body.position - b2.shape.body.position) <= 2 * BALL_RADIUS and (np.dot(b2.shape.body.position - b1.shape.body.position, (b2.velocity[0] - b1.velocity[0], b2.velocity[1] - b1.velocity[1])) < 0)

def collision(b1, b2):
    m1 = b1.shape.body.mass
    m2 = b2.shape.body.mass
    v1x, v1y = b1.velocity
    v2x, v2y = b2.velocity

    c1 = (((1 + BALL_ELASTICITY) * m2) / (m1 + m2)) * np.dot((b1.velocity[0] - b2.velocity[0], b1.velocity[1] - b2.velocity[1]), (b1.shape.body.position - b2.shape.body.position)) / (np.linalg.norm(b1.shape.body.position - b2.shape.body.position) ** 2)
    v1f = (v1x - c1 * (b1.shape.body.position[0] - b2.shape.body.position[0]), v1y - c1 * (b1.shape.body.position[1] - b2.shape.body.position[1]))

    c2 = (((1 + BALL_ELASTICITY) * m1) / (m1 + m2)) * np.dot((b2.velocity[0] - b1.velocity[0], b2.velocity[1] - b1.velocity[1]), (b2.shape.body.position - b1.shape.body.position)) / (np.linalg.norm(b2.shape.body.position - b1.shape.body.position) ** 2)
    v2f = (v2x - c2 * (b2.shape.body.position[0] - b1.shape.body.position[0]), v2y - c2 * (b2.shape.body.position[1] - b1.shape.body.position[1]))

    b1.velocity = v1f
    b2.velocity = v2f
    return True

def friction(b, friction_accel_magnitude, dt):
    v = math.sqrt(b.velocity[0]**2 + b.velocity[1]**2)
    if v != 0:
        b.velocity = (b.velocity[0] - friction_accel_magnitude * dt * b.velocity[0] / v, b.velocity[1] - friction_accel_magnitude * dt * b.velocity[1] / v)
    if abs(v) <= 0.01:
        b.velocity = (0, 0)

class player:
    def __init__(self, ix):
        self.ix = ix
        self.turn = False
        self.ball_type = None
        self.prev_len = 7
        self.text = ""


def main():
    # root window
    root = tk.Tk()
    root.geometry('300x200')
    root.resizable(False, False)
    root.title('Set up world')

    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=3)

    current_value = tk.DoubleVar()

    def get1():
        return '{: .2f}'.format(current_value.get())

    var1 = tk.DoubleVar()
    label1 = tk.Label(root, text="Elasticity [0, 1]:").pack()
    slider1 = ttk.Scale(root, from_=0, to=1, orient='horizontal', variable=var1)
    slider1.pack()
    current1 = tk.Label(root, textvariable=var1).pack()

    var2 = tk.DoubleVar()
    label2 = tk.Label(root, text="Friction [0, 1]:").pack()
    slider2 = ttk.Scale(root, from_=0, to=1, orient='horizontal', variable=var2)
    slider2.pack()
    current2 = tk.Label(root, textvariable=var2).pack()

    def get_value():
        global BALL_ELASTICITY
        BALL_ELASTICITY = slider1.get()
        global FRICTION_COEFF
        FRICTION_COEFF = slider2.get()
        global FRICTION_ACCEL
        FRICTION_ACCEL = 2 * FRICTION_COEFF
        root.destroy()

    button = tk.Button(root, text="Submit", command=get_value).pack()

    root.mainloop()

    dt = 0.01
    pygame.init()
    pygame.font.init()
    my_font = pygame.font.SysFont('Comic Sans MS', 30)
    small_font = pygame.font.SysFont('Comic Sans MS', 20)
    num_font = pygame.font.SysFont('Comic Sans MS', 12)
    screen = pygame.display.set_mode((WORLD_DIMS[0], WORLD_DIMS[1] + 150))
    screen.fill((0, 102, 0))
    clock = pygame.time.Clock()
    running = True
    draw_options = pymunk.pygame_util.DrawOptions(screen)
    pymunk.pygame_util.positive_y_is_up = False

    space = pymunk.Space(WORLD_DIMS)
    space.gravity = 0, 0
    initialize_border(space)
    initialize_holes(space)

    balls = []

    cueball = CueBall(position=(1000, 300))
    space.add(cueball.shape, cueball.shape.body)
    balls.append(cueball)

    colorssolid = [(255, 255, 51, 255), (0, 0, 255, 255), (255, 0, 0, 255), (76, 0, 153, 255), (255, 128, 0, 255), (51, 255, 51, 255), (102, 0, 0)]
    colorsstripe = [(255, 255, 51, 255), (0, 0, 255, 255), (255, 0, 0, 255), (76, 0, 153, 255), (255, 128, 0, 255), (51, 255, 51, 255), (102, 0, 0)]
    numsolid = [1, 2, 3, 4, 5, 6, 7]
    numstripe = [9, 10, 11, 12, 13, 14, 15]

    ix = 0
    ixs = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    solids = np.random.choice(ixs, 7, replace=False)
    for i in range(1, 6, 1):
        for j in range(1, i + 1, 1):
            if ix == 4:
                design = 8
                color = (0, 0, 0, 255)
                number = 8
            elif ix in solids:
                design = 0
                color = colorssolid.pop(0)
                number = numsolid.pop(0)
            else:
                design = 1
                color = colorsstripe.pop(0)
                number = numstripe.pop(0)
            ix = ix + 1
            x = (3 - i) * BALL_RADIUS * 1.05 * math.sqrt(3) + 300
            y = (j - 3) * BALL_RADIUS * 2 * 1.05 + 300 + ((5 - i) * BALL_RADIUS * 1.05)
            ball = Ball(position=(x, y), design=design, color=color, number=number)
            space.add(ball.shape, ball.shape.body)
            balls.append(ball)

    ke = 0
    new_iter = True
    for ball in balls:
        ke = ke + .5 * ball.shape.body.mass * (np.linalg.norm(ball.velocity) ** 2)

    solids = list(filter(None, [b if b.design == 0 else None for b in balls]))
    stripes = list(filter(None, [b if b.design == 1 else None for b in balls]))
    eightball = list(filter(None, [b if b.design == 8 else None for b in balls]))

    prev_solids_len = len(solids)
    prev_stripes_len = len(stripes)

    p1 = player(1)
    p2 = player(2)
    p1.turn = True

    trigger = False
    winner = 0
    reset_cueball = False

    xs = []
    ys = []
    kes = []
    cs = []
    c = 0
    while running:
        for event in pygame.event.get():
            if len(balls) == 1:
                running = False
            elif ke == 0:
                new_iter = True
            elif event.type == QUIT or (event.type == KEYDOWN and event.key in (K_q, K_ESCAPE)):
                running = False
            if new_iter:
                new_pos = pymunk.pygame_util.get_mouse_pos(screen)
                diff = new_pos - cueball.shape.body.position
                dx, dy = diff[0], diff[1]
                if event.type == pygame.MOUSEBUTTONDOWN:
                    prev_solids_len = len(solids)
                    prev_stripes_len = len(stripes)

                    if type(p1.ball_type) == list:
                        p1.prev_len = len(p1.ball_type)
                    if type(p2.ball_type) == list:
                        p2.prev_len = len(p2.ball_type)

                    trigger = True

                    if dx**2 + dy**2 >= (100 ** 2):
                        h = ((dx ** 2) + (dy ** 2)) ** 0.5
                        if not h == 0:
                            dx = 100 * dx / h
                            dy = 100 * dy / h
                            cueball.velocity = dx, dy
                            new_iter = False
                    else:
                        cueball.velocity = dx, dy
                        new_iter = False
                    c = 0

        contact_pairs = []
        for pair in itertools.combinations(balls, 2):
            if detect_collisions(pair[0], pair[1]):
                contact_pairs.append(pair)

        for pair in contact_pairs:
            collision(pair[0], pair[1])

        for ball in balls:
            x = ball.shape.body.position[0]
            y = ball.shape.body.position[1]
            if ((x)**2)+((y)**2) <= (2.5*BALL_RADIUS)**2:
                if type(ball) == CueBall:
                    reset_cueball = True
                space.remove(ball.shape.body, ball.shape)
                balls.remove(ball)
                if ball in solids:
                    solids.remove(ball)
                if ball in stripes:
                    stripes.remove(ball)
                if ball in eightball:
                    eightball.remove(ball)
            elif ((x-600)**2)+((y)**2) <= (2.5*BALL_RADIUS)**2:
                if type(ball) == CueBall:
                    reset_cueball = True
                space.remove(ball.shape.body, ball.shape)
                balls.remove(ball)
                if ball in solids:
                    solids.remove(ball)
                if ball in stripes:
                    stripes.remove(ball)
                if ball in eightball:
                    eightball.remove(ball)
            elif ((x-1200)**2)+((y)**2) <= (2.5*BALL_RADIUS)**2:
                if type(ball) == CueBall:
                    reset_cueball = True
                space.remove(ball.shape.body, ball.shape)
                balls.remove(ball)
                if ball in solids:
                    solids.remove(ball)
                if ball in stripes:
                    stripes.remove(ball)
                if ball in eightball:
                    eightball.remove(ball)
            elif ((x)**2)+((y-600)**2) <= (2.5*BALL_RADIUS)**2:
                if type(ball) == CueBall:
                    reset_cueball = True
                space.remove(ball.shape.body, ball.shape)
                balls.remove(ball)
                if ball in solids:
                    solids.remove(ball)
                if ball in stripes:
                    stripes.remove(ball)
                if ball in eightball:
                    eightball.remove(ball)
            elif ((x-600)**2)+((y-600)**2) <= (2.5*BALL_RADIUS)**2:
                if type(ball) == CueBall:
                    reset_cueball = True
                space.remove(ball.shape.body, ball.shape)
                balls.remove(ball)
                if ball in solids:
                    solids.remove(ball)
                if ball in stripes:
                    stripes.remove(ball)
                if ball in eightball:
                    eightball.remove(ball)
            elif ((x-1200)**2)+((y-600)**2) <= (2.5*BALL_RADIUS)**2:
                if type(ball) == CueBall:
                    reset_cueball = True
                space.remove(ball.shape.body, ball.shape)
                balls.remove(ball)
                if ball in solids:
                    solids.remove(ball)
                if ball in stripes:
                    stripes.remove(ball)
                if ball in eightball:
                    eightball.remove(ball)
            else:
                if x <= 1.5 * BALL_RADIUS:
                    ball.velocity = (-1 * ball.velocity[0], ball.velocity[1])
                if x >= 1200 - 1.5 * BALL_RADIUS:
                    ball.velocity = (-1 * ball.velocity[0], ball.velocity[1])
                if y <= 600 - 1.5 * BALL_RADIUS:
                    ball.velocity = (ball.velocity[0], -1 * ball.velocity[1])
                if y >= 1.5 * BALL_RADIUS:
                    ball.velocity = (ball.velocity[0], -1 * ball.velocity[1])
                friction(ball, FRICTION_ACCEL / ball.shape.body.mass, dt)

        ke = 0
        for ball in balls:
            ke = ke + .5 * ball.shape.body.mass * (np.linalg.norm(ball.velocity) ** 2)
        kes.append(ke)

        if ke == 0 and trigger:
            c = 0
            if len(eightball) == 0:
                if p1.turn:
                    if p1.ball_type == None:
                        winner = 2
                    elif len(p1.ball_type) > 0:
                        winner = 2
                    else:
                        winner = 1
                else:
                    if p2.ball_type == None:
                        winner = 1
                    elif len(p2.ball_type) > 0:
                        winner = 1
                    else:
                        winner = 2
            else:
                if reset_cueball == True:
                    good = True
                    for ball in balls:
                        if (ball.shape.body.position[0] - 1000) ** 2 + (ball.shape.body.position[1] - 300) < BALL_RADIUS ** 2:
                            good = False
                    if good == True:
                        cueball = CueBall(position=(1000, 300))
                        space.add(cueball.shape, cueball.shape.body)
                        balls.append(cueball)
                        p1.turn = not p1.turn
                        p2.turn = not p2.turn
                        trigger = False
                        reset_cueball = False
                    else:
                        found_location = False
                        while not found_location:
                            x = random.randint(100, 1100)
                            y = random.randint(100, 500)
                            good = True
                            for ball in balls:
                                if (ball.shape.body.position[0] - x) ** 2 + (ball.shape.body.position[1] - y) < BALL_RADIUS ** 2:
                                    good = False
                            if good == True:
                                cueball = CueBall(position=(x, y))
                                space.add(cueball.shape, cueball.shape.body)
                                balls.append(cueball)
                                p1.turn = not p1.turn
                                p2.turn = not p2.turn
                                trigger = False
                                reset_cueball = False
                                found_location = True
                elif len(balls) == 16:
                    p1.turn = not p1.turn
                    p2.turn = not p2.turn
                    trigger = False
                else:
                    if prev_solids_len == 7 and len(solids) < 7:
                        if p1.turn:
                            p1.ball_type = solids
                            p2.ball_type = stripes
                            p1.text = "SOLIDS"
                            p2.text = "STRIPES"
                        else:
                            p1.ball_type = stripes
                            p2.ball_type = solids
                            p1.text = "STRIPES"
                            p2.text = "SOLIDS"
                    if prev_stripes_len == 7 and len(stripes) < 7:
                        if p1.turn:
                            p1.ball_type = stripes
                            p2.ball_type = solids
                            p1.text = "STRIPES"
                            p2.text = "SOLIDS"
                        else:
                            p1.ball_type = solids
                            p2.ball_type = stripes
                            p1.text = "SOLIDS"
                            p2.text = "STRIPES"

                    if p1.turn:
                        if p1.prev_len - len(p1.ball_type) > 0:
                            trigger = False
                        else:
                            p1.turn = False
                            p2.turn = True
                            trigger = False
                    else:
                        if p2.prev_len - len(p2.ball_type) > 0:
                            trigger = False
                        else:
                            p1.turn = True
                            p2.turn = False
                            trigger = False

        screen.fill((0, 102, 0))

        pygame.draw.rect(screen, (75, 55, 28, 255), pygame.Rect(0,  -0.5 * BALL_RADIUS, WORLD_DIMS[0], BALL_RADIUS))
        pygame.draw.rect(screen, (75, 55, 28, 255), pygame.Rect(0, WORLD_DIMS[1] - (0.5 * BALL_RADIUS), WORLD_DIMS[0], BALL_RADIUS))
        pygame.draw.rect(screen, (75, 55, 28, 255), pygame.Rect(-0.5 * BALL_RADIUS, 0, BALL_RADIUS, WORLD_DIMS[1]))
        pygame.draw.rect(screen, (75, 55, 28, 255), pygame.Rect(WORLD_DIMS[0] - (0.5 * BALL_RADIUS), 0, BALL_RADIUS, WORLD_DIMS[1]))

        for i in range(6):
            pygame.draw.circle(screen, (0, 0, 0, 255), (WORLD_DIMS[0] * ((i % 3) / 2), WORLD_DIMS[1] * (i // 3)), BALL_RADIUS * 1.5)

        for ball in balls:
            ball.shape.body.position = (ball.shape.body.position[0] + ball.velocity[0] * dt, ball.shape.body.position[1] + ball.velocity[1] * dt)
            if ball.design == 0 or ball.design == 8 or ball.design == -1:
                pygame.draw.circle(screen, ball.color, ball.shape.body.position, BALL_RADIUS)
                pygame.draw.circle(screen, (255, 255, 255, 255), ball.shape.body.position, BALL_RADIUS / 3)
                if ball.number > 0:
                    numtext = num_font.render(str(ball.number), False, (0, 0, 0))
                    screen.blit(numtext, (ball.shape.body.position[0] - 2.5, ball.shape.body.position[1] - 7.5))
            if ball.design == 1:
                pygame.draw.circle(screen, (255, 255, 255, 255), ball.shape.body.position, BALL_RADIUS)
                pygame.draw.rect(screen, ball.color, pygame.Rect((ball.shape.body.position[0] - BALL_RADIUS / 2, ball.shape.body.position[1] - BALL_RADIUS), (BALL_RADIUS, 2 * BALL_RADIUS)))
                pygame.draw.circle(screen, (255, 255, 255, 255), ball.shape.body.position, BALL_RADIUS / 3)
                if ball.number > 0:
                    numtext = num_font.render(str(ball.number), False, (0, 0, 0))
                    screen.blit(numtext, (ball.shape.body.position[0] - 3.5, ball.shape.body.position[1] - 8))


        if winner == 0:
            p1text = my_font.render('PLAYER ONE', False, (0, 0, 0))
            screen.blit(p1text, (2 * BALL_RADIUS, WORLD_DIMS[1] + BALL_RADIUS / 2))

            p2text = my_font.render('PLAYER TWO', False, (0, 0, 0))
            screen.blit(p2text, (WORLD_DIMS[0] - 2 * BALL_RADIUS - 200, WORLD_DIMS[1] + BALL_RADIUS / 2))

            p1designtext = my_font.render(p1.text, False, (0, 0, 0))
            screen.blit(p1designtext, (2 * BALL_RADIUS, WORLD_DIMS[1] + BALL_RADIUS / 2 + 30))

            p2designtext = my_font.render(p2.text, False, (0, 0, 0))
            screen.blit(p2designtext, (WORLD_DIMS[0] - 2 * BALL_RADIUS - 200, WORLD_DIMS[1] + BALL_RADIUS / 2 + 30))

            if p1.turn:
                turntext = my_font.render("<-- PLAYER ONE MOVE", False, (0, 0, 0))
                screen.blit(turntext, (WORLD_DIMS[0] / 2 - 200, WORLD_DIMS[1] + BALL_RADIUS / 2 + 25))
            else:
                turntext = my_font.render("PLAYER TWO MOVE -->", False, (0, 0, 0))
                screen.blit(turntext, (WORLD_DIMS[0] / 2 - 160, WORLD_DIMS[1] + BALL_RADIUS / 2 + 25))

            instrtext = "CLICK ANYWHERE ON TABLE TO SET DIRECTION AND MAGNITUDE"
            instrtext = small_font.render(instrtext, False, (0, 0, 0))
            screen.blit(instrtext, (2 * BALL_RADIUS + 205, WORLD_DIMS[1] + BALL_RADIUS / 2 + 15 + 50))
            instrtext = "(INCREASES WITH DISTANCE FROM CUEBALL) OF CUEBALL VELOCITY"
            instrtext = small_font.render(instrtext, False, (0, 0, 0))
            screen.blit(instrtext, (2 * BALL_RADIUS + 195, WORLD_DIMS[1] + BALL_RADIUS / 2 + 15 + 75))
        else:
            if winner == 1:
                winnertext = my_font.render("PLAYER ONE WINS!!!", False, (0, 0, 0))
                screen.blit(winnertext, (WORLD_DIMS[0] / 2 - 200, WORLD_DIMS[1] + BALL_RADIUS / 2 + 45))
                instrtext = small_font.render("PRESS (q) to QUIT", False, (0, 0, 0))
                screen.blit(instrtext, (WORLD_DIMS[0] / 2 - 100, WORLD_DIMS[1] + BALL_RADIUS / 2 + 85))
            if winner == 2:
                winnertext = my_font.render("PLAYER TWO WINS!!!", False, (0, 0, 0))
                screen.blit(winnertext, (WORLD_DIMS[0] / 2 - 200, WORLD_DIMS[1] + BALL_RADIUS / 2 + 45))
                instrtext = small_font.render("PRESS (q) to QUIT", False, (0, 0, 0))
                screen.blit(instrtext, (WORLD_DIMS[0] / 2 - 100, WORLD_DIMS[1] + BALL_RADIUS / 2 + 85))

        pygame.display.update()
        space.step(dt)

        x = 0
        y = 0
        n = len(balls)
        for ball in balls:
            x = x + ball.shape.body.position[0]
            y = y + ball.shape.body.position[1]
        c = c + 1
        cs.append(c)
        xs.append(x/n)
        ys.append(y/n)
        kes.append(ke)
    plt.plot([t * dt for t in range(len(kes))], kes)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Kinetic Energy of Entire System (joules)")
    plt.title("Total Kinetic Energy vs. Time")
    plt.show()

    cs = [-1 * c for c in cs]

    plt.scatter(xs, ys, c=cs, cmap='Greys')
    plt.xlabel("X-Coordinate")
    plt.ylabel("Y-Coordinate")
    plt.title("Center of Mass trajectory")
    plt.figtext(0.5, 0, "Note: Within every episode, lighter lines indicate progressing time", ha="center", fontsize=8)
    plt.show()

if __name__ == '__main__':
    main()
