"""
agent.py

Mouse-like agent using:
    - natural border swimming;
    - persistent spatial memory;
    - self-supervised SSM hidden state.

Still not reinforcement learning.
"""

import math

import numpy as np
import pygame

from environment import (
    MOUSE_RADIUS,
    MOUSE_SPEED,
    TANK_CENTER,
    TANK_RADIUS,
    distance,
    normalize,
    random_point_inside_tank,
    to_screen_point,
    vector_from_angle,
)


class MouseAgent:
    """Simple mouse-like agent."""

    def __init__(self, spatial_memory, ssm_trainer):
        self.spatial_memory = spatial_memory
        self.ssm_trainer = ssm_trainer
        self.start_new_trial()

    def start_new_trial(self):
        """Start a new trial without clearing long-term memory."""
        self.position = random_point_inside_tank()
        self.heading = np.random.uniform(0, 2 * math.pi)

        self.steps = 0
        self.path_length = 0.0
        self.phase = "border_swimming"

        self.path = [self.position.copy()]
        self.platform_crossings = 0
        self.target_quadrant_time = 0
        self.was_in_platform_zone = False

        self.last_action = np.array([0.0, 0.0], dtype=np.float32)
        self.ssm_trainer.reset_hidden_state()

    def choose_direction(self):
        """Choose movement direction from memory and self-supervised state."""
        self.steps += 1

        remembered_platform = self.spatial_memory.get_remembered_platform()
        ssm_bias = self.ssm_trainer.hidden_direction_bias()
        novelty_bias = self.spatial_memory.get_visit_gradient_direction(self.position)

        if remembered_platform is not None and self.steps > 55:
            self.phase = "memory_guided_search"

            memory_direction = normalize(remembered_platform - self.position)

            # The SSM adds a weak temporal bias. It does not replace the map.
            direction = normalize(
                0.86 * memory_direction +
                0.08 * ssm_bias +
                0.06 * novelty_bias
            )

            small_noise = np.random.normal(0, 0.025, size=2).astype(np.float32)
            return normalize(direction + small_noise)

        radial_vector = self.position - TANK_CENTER
        current_angle = math.atan2(radial_vector[1], radial_vector[0])

        if self.steps < 260:
            self.phase = "border_swimming"

            tangent_direction = np.array(
                [-math.sin(current_angle), math.cos(current_angle)],
                dtype=np.float32,
            )

            desired_border_position = TANK_CENTER + normalize(radial_vector) * (TANK_RADIUS * 0.86)
            border_correction = normalize(desired_border_position - self.position)

            direction = normalize(
                0.82 * tangent_direction +
                0.14 * border_correction +
                0.04 * ssm_bias
            )

            small_noise = np.random.normal(0, 0.05, size=2).astype(np.float32)
            return normalize(direction + small_noise)

        self.phase = "inner_exploration"

        target = self.spatial_memory.get_least_visited_inner_target()
        map_direction = normalize(target - self.position)

        direction = normalize(
            0.78 * map_direction +
            0.12 * novelty_bias +
            0.10 * ssm_bias
        )

        small_noise = np.random.normal(0, 0.075, size=2).astype(np.float32)
        return normalize(direction + small_noise)

    def update_trial_metrics(self, environment):
        """Update Morris Water Maze biological metrics."""
        if environment.position_quadrant(self.position) == environment.platform_quadrant():
            self.target_quadrant_time += 1

        platform_zone_radius = 2.0 * MOUSE_RADIUS
        in_platform_zone = distance(self.position, environment.platform_position) <= platform_zone_radius

        if in_platform_zone and not self.was_in_platform_zone:
            self.platform_crossings += 1

        self.was_in_platform_zone = in_platform_zone

    def move(self, environment):
        """Move the agent and return the action vector."""
        direction = self.choose_direction()
        previous_position = self.position.copy()

        proposed_position = self.position + direction * MOUSE_SPEED

        if environment.is_inside_tank(proposed_position):
            self.position = proposed_position
        else:
            self.position = self.position + normalize(TANK_CENTER - self.position) * MOUSE_SPEED

        self.heading = math.atan2(direction[1], direction[0])
        self.path_length += distance(previous_position, self.position)
        self.path.append(self.position.copy())

        self.update_trial_metrics(environment)

        self.last_action = direction.astype(np.float32)
        return self.last_action

    def current_trial_metrics(self):
        """Return metrics collected during the current trial."""
        return {
            "latency_steps": self.steps,
            "path_length": self.path_length,
            "target_quadrant_time": self.target_quadrant_time,
            "platform_crossings": self.platform_crossings,
        }

    def draw(self, surface):
        """Draw the mouse and its recent trajectory."""
        if len(self.path) > 1:
            points = [to_screen_point(point) for point in self.path[-500:]]
            if len(points) >= 2:
                pygame.draw.lines(surface, (230, 245, 255), False, points, 2)

        x, y = to_screen_point(self.position)

        pygame.draw.ellipse(surface, (185, 170, 155), (x - 16, y - 10, 32, 20))

        nose_direction = vector_from_angle(self.heading)
        head_position = self.position + nose_direction * 16
        pygame.draw.circle(surface, (205, 190, 175), to_screen_point(head_position), 9)

        left_ear = head_position + vector_from_angle(self.heading + 2.2) * 8
        right_ear = head_position + vector_from_angle(self.heading - 2.2) * 8

        pygame.draw.circle(surface, (220, 170, 170), to_screen_point(left_ear), 4)
        pygame.draw.circle(surface, (220, 170, 170), to_screen_point(right_ear), 4)

        nose_position = self.position + nose_direction * 27
        pygame.draw.circle(surface, (60, 40, 40), to_screen_point(nose_position), 3)

        tail_position = self.position - nose_direction * 22
        pygame.draw.line(
            surface,
            (210, 150, 150),
            to_screen_point(self.position),
            to_screen_point(tail_position),
            3,
        )
