"""
visualization.py

Desenho do canvas do mapa mental e painel informativo.

Correção PyGame:
    Todas as posições foram convertidas para tuplas de inteiros.
"""

import pygame

from memory import MEMORY_GRID


CANVAS_X, CANVAS_Y = 740, 90
CANVAS_SIZE = 235


def draw_mental_map_canvas(surface, mental_map, show_canvas=True):
    """Desenha o canvas de memória espacial."""
    if not show_canvas:
        return

    pygame.draw.rect(surface, (20, 25, 32), (CANVAS_X - 15, CANVAS_Y - 50, 285, 560), border_radius=12)
    pygame.draw.rect(surface, (230, 240, 245), (CANVAS_X, CANVAS_Y, CANVAS_SIZE, CANVAS_SIZE), 2)

    cell = CANVAS_SIZE / MEMORY_GRID
    max_visit = max(1.0, float(mental_map.visits.max()))

    for gy in range(MEMORY_GRID):
        for gx in range(MEMORY_GRID):
            value = mental_map.visits[gy, gx]
            if value <= 0:
                continue

            intensity = min(180, int(35 + 145 * value / max_visit))
            color = (60, 160, 230, intensity)

            rect = pygame.Surface((int(cell) + 1, int(cell) + 1), pygame.SRCALPHA)
            rect.fill(color)

            surface.blit(
                rect,
                (
                    int(CANVAS_X + gx * cell),
                    int(CANVAS_Y + gy * cell),
                ),
            )

    # Plataforma lembrada
    if mental_map.platform_pos is not None:
        gx, gy = mental_map.xy_to_cell(mental_map.platform_pos)

        pygame.draw.rect(
            surface,
            (255, 240, 80),
            (
                int(CANVAS_X + gx * cell - 3),
                int(CANVAS_Y + gy * cell - 3),
                11,
                11,
            ),
        )

    color_by_cue = {
        "estrela": (255, 230, 40),
        "triangulo": (255, 80, 80),
        "circulo": (80, 180, 255),
        "cruz": (90, 255, 130),
        "barras": (230, 120, 255),
    }

    # Símbolos vistos
    for item in mental_map.seen_cues[-100:]:
        gx, gy = mental_map.xy_to_cell(item["rat_pos"])

        pygame.draw.circle(
            surface,
            color_by_cue.get(item["name"], (255, 255, 255)),
            (
                int(CANVAS_X + gx * cell),
                int(CANVAS_Y + gy * cell),
            ),
            3,
        )

    # Linha entre melhor símbolo e plataforma
    if mental_map.best_cue_pos is not None and mental_map.platform_pos is not None:
        cgx, cgy = mental_map.xy_to_cell(mental_map.best_cue_pos)
        pgx, pgy = mental_map.xy_to_cell(mental_map.platform_pos)

        start = (
            int(CANVAS_X + cgx * cell),
            int(CANVAS_Y + cgy * cell),
        )

        end = (
            int(CANVAS_X + pgx * cell),
            int(CANVAS_Y + pgy * cell),
        )

        pygame.draw.line(surface, (255, 255, 255), start, end, 3)


def draw_panel(surface, font, big_font, mental_map, rat):
    """Desenha painel textual."""
    x = CANVAS_X - 15
    y = CANVAS_Y + CANVAS_SIZE + 20

    last = mental_map.trials_metrics[-1] if mental_map.trials_metrics else None

    lines = [
        "Mapa mental",
        f"Rodada: {mental_map.trial}",
        f"Fase: {rat.phase}",
        f"Latência atual: {rat.steps}",
        f"Distância atual: {rat.path_length:.1f}",
        f"Tempo quadrante-alvo: {rat.target_quadrant_time}",
        f"Cruzamentos plataforma: {rat.platform_crossings}",
        f"Última latência: {last['latency_steps'] if last else '-'}",
        f"Última distância: {last['path_length']:.1f}" if last else "Última distância: -",
        f"Melhor símbolo: {mental_map.best_cue_name if mental_map.best_cue_name else '-'}",
        f"Dist. símbolo-plataforma: {mental_map.best_cue_platform_distance:.1f}" if mental_map.best_cue_platform_distance else "Dist. símbolo-plataforma: -",
        f"Símbolos memorizados: {len(mental_map.seen_cues)}",
        "",
        "M: mostrar/ocultar mapa",
        "R: apagar memória",
        "ESC: sair",
    ]

    for i, line in enumerate(lines):
        if i == 0:
            img = big_font.render(line, True, (255, 255, 255))
        else:
            img = font.render(line, True, (225, 235, 245))

        surface.blit(img, (x, y + i * 24))
