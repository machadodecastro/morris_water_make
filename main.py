"""
main.py

Main program.

The agent uses:
    - persistent spatial memory from outputs/mental_map.json;
    - self-supervised SSM from outputs/ssm_model.pt.

New trials keep memory.
Program restarts reload memory and the SSM.
"""

import pygame

from agent import MouseAgent
from drawing import draw_memory_canvas, draw_text_panel
from environment import FPS, SCREEN_HEIGHT, SCREEN_WIDTH, MorrisWaterMaze
from memory import SpatialMemory
from ssm_model import SelfSupervisedSSMTrainer


def main():
    pygame.init()
    pygame.display.set_caption("Morris Water Maze - Self-Supervised SSM Spatial Memory Agent")

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("Arial", 18)
    title_font = pygame.font.SysFont("Arial", 24, bold=True)

    environment = MorrisWaterMaze()

    dummy_position = environment.platform_position.copy()
    dummy_heading = 0.0
    observation_size = len(environment.observation_from(dummy_position, dummy_heading))

    spatial_memory = SpatialMemory(output_dir="outputs")
    ssm_trainer = SelfSupervisedSSMTrainer(
        observation_size=observation_size,
        output_dir="outputs",
    )

    mouse_agent = MouseAgent(spatial_memory, ssm_trainer)

    show_memory_canvas = True
    running = True
    last_ssm_loss = None

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_m:
                    show_memory_canvas = not show_memory_canvas

                elif event.key == pygame.K_n:
                    mouse_agent.start_new_trial()

                elif event.key == pygame.K_c:
                    spatial_memory.reset_all_memory()
                    ssm_trainer.reset_model()
                    mouse_agent.start_new_trial()

        previous_observation = environment.observation_from(
            mouse_agent.position,
            mouse_agent.heading,
        )

        for cue in environment.visible_cues_from(mouse_agent.position):
            spatial_memory.record_seen_cue(cue, mouse_agent.position)

        action = mouse_agent.move(environment)

        next_observation = environment.observation_from(
            mouse_agent.position,
            mouse_agent.heading,
        )

        spatial_memory.record_visit(mouse_agent.position)

        ssm_trainer.observe_transition(
            previous_observation,
            action,
            next_observation,
        )
        ssm_trainer.update_hidden_state(previous_observation, action)

        loss = ssm_trainer.train_step(batch_size=64)
        if loss is not None:
            last_ssm_loss = loss

        if environment.reached_platform(mouse_agent.position):
            spatial_memory.record_platform_found(
                platform_position=environment.platform_position,
                metrics=mouse_agent.current_trial_metrics(),
            )

            ssm_trainer.save()
            mouse_agent.start_new_trial()

        screen.fill((15, 20, 28))

        environment.draw(screen)
        mouse_agent.draw(screen)

        draw_memory_canvas(screen, spatial_memory, show_memory_canvas)
        draw_text_panel(screen, font, title_font, spatial_memory, mouse_agent, last_ssm_loss)

        pygame.display.flip()
        clock.tick(FPS)

    spatial_memory.save()
    ssm_trainer.save()
    pygame.quit()


if __name__ == "__main__":
    main()
