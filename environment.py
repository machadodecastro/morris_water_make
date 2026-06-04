"""
environment.py

Morris Water Maze environment.

The environment contains:
    - the circular tank;
    - the hidden platform;
    - visual cues around the border.

The environment does not provide rewards.
"""

import math
import random
from dataclasses import dataclass

import numpy as np
import pygame


SCREEN_WIDTH = 1050
SCREEN_HEIGHT = 760
FPS = 60

TANK_CENTER = np.array([370.0, 380.0], dtype=np.float32)
TANK_RADIUS = 300

MOUSE_RADIUS = 13
MOUSE_SPEED = 2.9

PLATFORM_RADIUS = 26


def to_screen_point(point):
    """Convert a NumPy point to a PyGame-safe integer tuple."""
    return (int(point[0]), int(point[1]))


def vector_from_angle(angle):
    """Return a 2D unit vector from an angle."""
    return np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)


def normalize(vector):
    """Normalize a 2D vector."""
    length = np.linalg.norm(vector)
    if length < 1e-6:
        return np.array([1.0, 0.0], dtype=np.float32)
    return vector / length


def distance(point_a, point_b):
    """Euclidean distance."""
    return float(np.linalg.norm(point_a - point_b))


def random_point_inside_tank():
    """Return a random point inside the tank."""
    angle = random.uniform(0, 2 * math.pi)
    radius = TANK_RADIUS * math.sqrt(random.uniform(0.05, 0.90))
    return TANK_CENTER + vector_from_angle(angle) * radius


@dataclass
class VisualCue:
    """A fixed visual cue outside the tank border."""
    name: str
    angle: float
    color: tuple

    @property
    def position(self):
        return TANK_CENTER + vector_from_angle(self.angle) * (TANK_RADIUS + 42)


class MorrisWaterMaze:
    """Circular tank with one hidden platform and fixed border cues."""

    def __init__(self):
        self.platform_position = TANK_CENTER + np.array([110.0, -90.0], dtype=np.float32)

        self.visual_cues = [
            VisualCue("star", -math.pi / 2, (255, 230, 40)),
            VisualCue("triangle", 0.0, (255, 80, 80)),
            VisualCue("circle", math.pi / 2, (80, 180, 255)),
            VisualCue("cross", math.pi, (90, 255, 130)),
            VisualCue("bars", -2.35, (230, 120, 255)),
        ]

    def is_inside_tank(self, position):
        """Return True if a position is inside the tank."""
        return distance(position, TANK_CENTER) <= TANK_RADIUS - MOUSE_RADIUS

    def reached_platform(self, position):
        """Return True if the mouse reached the platform."""
        return distance(position, self.platform_position) <= PLATFORM_RADIUS

    def platform_quadrant(self):
        """Return the quadrant containing the platform."""
        dx = self.platform_position[0] - TANK_CENTER[0]
        dy = self.platform_position[1] - TANK_CENTER[1]

        if dx >= 0 and dy < 0:
            return 0
        if dx < 0 and dy < 0:
            return 1
        if dx < 0 and dy >= 0:
            return 2
        return 3

    def position_quadrant(self, position):
        """Return the quadrant of a position."""
        dx = position[0] - TANK_CENTER[0]
        dy = position[1] - TANK_CENTER[1]

        if dx >= 0 and dy < 0:
            return 0
        if dx < 0 and dy < 0:
            return 1
        if dx < 0 and dy >= 0:
            return 2
        return 3

    def visible_cues_from(self, position):
        """Return cues visible from the current mouse position."""
        radius = distance(position, TANK_CENTER)

        if radius < TANK_RADIUS * 0.72:
            return []

        mouse_angle = math.atan2(position[1] - TANK_CENTER[1], position[0] - TANK_CENTER[0])
        visible_cues = []

        for cue in self.visual_cues:
            angle_difference = abs((cue.angle - mouse_angle + math.pi) % (2 * math.pi) - math.pi)
            if angle_difference < 0.35:
                visible_cues.append(cue)

        return visible_cues

    def observation_from(self, mouse_position, mouse_heading):
        """
        Return a small numeric observation vector.

        This observation is used by the self-supervised SSM.
        """
        relative_position = (mouse_position - TANK_CENTER) / TANK_RADIUS
        radial_distance = distance(mouse_position, TANK_CENTER) / TANK_RADIUS

        observation = [
            float(relative_position[0]),
            float(relative_position[1]),
            float(math.cos(mouse_heading)),
            float(math.sin(mouse_heading)),
            float(radial_distance),
        ]

        mouse_angle = math.atan2(mouse_position[1] - TANK_CENTER[1], mouse_position[0] - TANK_CENTER[0])

        for cue in self.visual_cues:
            angle_difference = (cue.angle - mouse_angle + math.pi) % (2 * math.pi) - math.pi
            observation.append(float(math.cos(angle_difference)))
            observation.append(float(math.sin(angle_difference)))

        return np.array(observation, dtype=np.float32)

    def draw(self, surface):
        """Draw the environment."""
        center = to_screen_point(TANK_CENTER)

        pygame.draw.circle(surface, (35, 120, 170), center, TANK_RADIUS)
        pygame.draw.circle(surface, (230, 245, 255), center, TANK_RADIUS, 4)

        for radius in range(50, TANK_RADIUS, 55):
            pygame.draw.circle(surface, (60, 150, 195), center, radius, 1)

        pygame.draw.line(
            surface,
            (80, 160, 200),
            (int(TANK_CENTER[0] - TANK_RADIUS), int(TANK_CENTER[1])),
            (int(TANK_CENTER[0] + TANK_RADIUS), int(TANK_CENTER[1])),
            1,
        )

        pygame.draw.line(
            surface,
            (80, 160, 200),
            (int(TANK_CENTER[0]), int(TANK_CENTER[1] - TANK_RADIUS)),
            (int(TANK_CENTER[0]), int(TANK_CENTER[1] + TANK_RADIUS)),
            1,
        )

        platform_x, platform_y = to_screen_point(self.platform_position)
        platform_surface = pygame.Surface((62, 62), pygame.SRCALPHA)
        pygame.draw.rect(platform_surface, (255, 255, 255, 95), (5, 5, 52, 52), 3)
        surface.blit(platform_surface, (platform_x - 31, platform_y - 31))

        for cue in self.visual_cues:
            self.draw_visual_cue(surface, cue)

    def draw_visual_cue(self, surface, cue):
        """Draw a visual cue."""
        x, y = to_screen_point(cue.position)

        if cue.name == "star":
            points = []
            for index in range(10):
                angle = -math.pi / 2 + index * math.pi / 5
                radius = 23 if index % 2 == 0 else 10
                points.append((int(x + math.cos(angle) * radius), int(y + math.sin(angle) * radius)))
            pygame.draw.polygon(surface, cue.color, points)

        elif cue.name == "triangle":
            pygame.draw.polygon(surface, cue.color, [(x, y - 24), (x - 24, y + 22), (x + 24, y + 22)])

        elif cue.name == "circle":
            pygame.draw.circle(surface, cue.color, (x, y), 23, 5)

        elif cue.name == "cross":
            pygame.draw.line(surface, cue.color, (x - 24, y), (x + 24, y), 6)
            pygame.draw.line(surface, cue.color, (x, y - 24), (x, y + 24), 6)

        elif cue.name == "bars":
            for index in range(4):
                pygame.draw.rect(surface, cue.color, (x - 26 + index * 15, y - 26, 8, 52))
