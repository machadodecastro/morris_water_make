"""
drawing.py

Drawing helper functions.
"""

import pygame

from memory import GRID_SIZE


CANVAS_X = 740
CANVAS_Y = 90
CANVAS_SIZE = 235


def draw_memory_canvas(surface, spatial_memory, show_canvas=True):
    """Draw the persistent mental map canvas."""
    if not show_canvas:
        return

    pygame.draw.rect(surface, (20, 25, 32), (CANVAS_X - 15, CANVAS_Y - 50, 285, 585), border_radius=12)
    pygame.draw.rect(surface, (230, 240, 245), (CANVAS_X, CANVAS_Y, CANVAS_SIZE, CANVAS_SIZE), 2)

    cell_size = CANVAS_SIZE / GRID_SIZE
    max_visits = max(1.0, float(spatial_memory.visits.max()))

    for grid_y in range(GRID_SIZE):
        for grid_x in range(GRID_SIZE):
            value = spatial_memory.visits[grid_y, grid_x]

            if value <= 0:
                continue

            alpha = min(180, int(35 + 145 * value / max_visits))
            cell_surface = pygame.Surface((int(cell_size) + 1, int(cell_size) + 1), pygame.SRCALPHA)
            cell_surface.fill((60, 160, 230, alpha))

            surface.blit(
                cell_surface,
                (
                    int(CANVAS_X + grid_x * cell_size),
                    int(CANVAS_Y + grid_y * cell_size),
                ),
            )

    if spatial_memory.remembered_platform_position is not None:
        grid_x, grid_y = spatial_memory.position_to_cell(spatial_memory.remembered_platform_position)

        pygame.draw.rect(
            surface,
            (255, 240, 80),
            (
                int(CANVAS_X + grid_x * cell_size - 3),
                int(CANVAS_Y + grid_y * cell_size - 3),
                11,
                11,
            ),
        )

    color_by_cue = {
        "star": (255, 230, 40),
        "triangle": (255, 80, 80),
        "circle": (80, 180, 255),
        "cross": (90, 255, 130),
        "bars": (230, 120, 255),
    }

    for item in spatial_memory.seen_cues[-100:]:
        grid_x, grid_y = spatial_memory.position_to_cell(item["mouse_position"])

        pygame.draw.circle(
            surface,
            color_by_cue.get(item["name"], (255, 255, 255)),
            (
                int(CANVAS_X + grid_x * cell_size),
                int(CANVAS_Y + grid_y * cell_size),
            ),
            3,
        )

    if spatial_memory.best_cue_position is not None and spatial_memory.remembered_platform_position is not None:
        cue_x, cue_y = spatial_memory.position_to_cell(spatial_memory.best_cue_position)
        platform_x, platform_y = spatial_memory.position_to_cell(spatial_memory.remembered_platform_position)

        start = (
            int(CANVAS_X + cue_x * cell_size),
            int(CANVAS_Y + cue_y * cell_size),
        )

        end = (
            int(CANVAS_X + platform_x * cell_size),
            int(CANVAS_Y + platform_y * cell_size),
        )

        pygame.draw.line(surface, (255, 255, 255), start, end, 3)


def draw_text_panel(surface, font, title_font, spatial_memory, mouse_agent, ssm_loss):
    """Draw the text panel."""
    x = CANVAS_X - 15
    y = CANVAS_Y + CANVAS_SIZE + 20

    last_trial = spatial_memory.trial_metrics[-1] if spatial_memory.trial_metrics else None

    lines = [
        "SSM spatial memory",
        f"Trial: {spatial_memory.trial_number}",
        f"Agent phase: {mouse_agent.phase}",
        f"Current latency: {mouse_agent.steps}",
        f"Current path length: {mouse_agent.path_length:.1f}",
        f"Target quadrant time: {mouse_agent.target_quadrant_time}",
        f"Platform crossings: {mouse_agent.platform_crossings}",
        f"Last latency: {last_trial['latency_steps'] if last_trial else '-'}",
        f"Best cue: {spatial_memory.best_cue_name if spatial_memory.best_cue_name else '-'}",
        f"SSM prediction loss: {ssm_loss:.5f}" if ssm_loss is not None else "SSM prediction loss: collecting data",
        f"Stored cue observations: {len(spatial_memory.seen_cues)}",
        "",
        "M: show/hide memory",
        "N: new trial, keep memory",
        "C: clear memory + SSM",
        "ESC: quit",
    ]

    for index, line in enumerate(lines):
        if index == 0:
            image = title_font.render(line, True, (255, 255, 255))
        else:
            image = font.render(line, True, (225, 235, 245))

        surface.blit(image, (x, y + index * 24))
