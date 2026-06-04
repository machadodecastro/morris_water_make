"""
memory.py

Persistent spatial memory.

This is not reinforcement learning.
The map is a persistent record of experience.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from environment import TANK_CENTER, TANK_RADIUS, distance


GRID_SIZE = 50


class SpatialMemory:
    """Persistent spatial memory used by the agent."""

    def __init__(self, output_dir="outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.memory_file = self.output_dir / "mental_map.json"

        self.visits = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        self.platform_memory = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

        self.seen_cues = []

        self.remembered_platform_position = None
        self.best_cue_name = None
        self.best_cue_position = None
        self.best_cue_to_platform_distance = None

        self.trial_number = 1
        self.trial_metrics = []

        self.load()

    def reset_all_memory(self):
        """Clear all memory and saved files."""
        self.visits[:] = 0.0
        self.platform_memory[:] = 0.0
        self.seen_cues = []

        self.remembered_platform_position = None
        self.best_cue_name = None
        self.best_cue_position = None
        self.best_cue_to_platform_distance = None

        self.trial_number = 1
        self.trial_metrics = []

        for path in [
            self.memory_file,
            self.output_dir / "mental_map.csv",
            self.output_dir / "mental_map.png",
            self.output_dir / "trial_metrics.csv",
            self.output_dir / "biological_interpretation.txt",
        ]:
            if path.exists():
                path.unlink()

    def position_to_cell(self, position):
        """Convert tank position to map grid cell."""
        left = TANK_CENTER[0] - TANK_RADIUS
        top = TANK_CENTER[1] - TANK_RADIUS

        grid_x = int(np.clip((position[0] - left) / (2 * TANK_RADIUS) * GRID_SIZE, 0, GRID_SIZE - 1))
        grid_y = int(np.clip((position[1] - top) / (2 * TANK_RADIUS) * GRID_SIZE, 0, GRID_SIZE - 1))

        return grid_x, grid_y

    def cell_to_position(self, grid_x, grid_y):
        """Convert map grid cell to tank position."""
        left = TANK_CENTER[0] - TANK_RADIUS
        top = TANK_CENTER[1] - TANK_RADIUS

        x = left + (grid_x + 0.5) / GRID_SIZE * (2 * TANK_RADIUS)
        y = top + (grid_y + 0.5) / GRID_SIZE * (2 * TANK_RADIUS)

        return np.array([x, y], dtype=np.float32)

    def record_visit(self, position):
        """Record a visited position."""
        grid_x, grid_y = self.position_to_cell(position)
        self.visits[grid_y, grid_x] += 1.0

    def record_seen_cue(self, cue, mouse_position):
        """Store a visual cue observation."""
        for recent_item in reversed(self.seen_cues[-20:]):
            if recent_item["name"] == cue.name:
                recent_position = np.array(recent_item["mouse_position"], dtype=np.float32)
                if distance(recent_position, mouse_position) < 24:
                    return
                break

        self.seen_cues.append({
            "trial": self.trial_number,
            "name": cue.name,
            "cue_position": cue.position.tolist(),
            "mouse_position": mouse_position.tolist(),
        })

    def record_platform_found(self, platform_position, metrics):
        """Update memory when the platform is found."""
        self.remembered_platform_position = platform_position.copy()

        grid_x, grid_y = self.position_to_cell(platform_position)
        self.platform_memory[grid_y, grid_x] += 1.0

        self.update_best_cue(platform_position)

        row = dict(metrics)
        row["trial"] = self.trial_number
        row["best_cue_name"] = self.best_cue_name
        row["best_cue_to_platform_distance"] = self.best_cue_to_platform_distance

        self.trial_metrics.append(row)
        self.trial_number += 1

        self.save()

    def update_best_cue(self, platform_position):
        """Find the cue closest to the platform."""
        if not self.seen_cues:
            return

        best_distance = None
        best_item = None

        for item in self.seen_cues:
            cue_position = np.array(item["cue_position"], dtype=np.float32)
            current_distance = distance(cue_position, platform_position)

            if best_distance is None or current_distance < best_distance:
                best_distance = current_distance
                best_item = item

        if best_item is not None:
            self.best_cue_name = best_item["name"]
            self.best_cue_position = np.array(best_item["cue_position"], dtype=np.float32)
            self.best_cue_to_platform_distance = float(best_distance)

    def get_remembered_platform(self):
        """Return remembered platform position, if available."""
        if self.remembered_platform_position is None:
            return None
        return self.remembered_platform_position.copy()

    def get_least_visited_inner_target(self):
        """Return a weakly explored inner location."""
        best_score = None
        best_position = None

        for grid_y in range(GRID_SIZE):
            for grid_x in range(GRID_SIZE):
                position = self.cell_to_position(grid_x, grid_y)
                radius = distance(position, TANK_CENTER)

                if radius > TANK_RADIUS * 0.80:
                    continue

                score = self.visits[grid_y, grid_x]

                if best_score is None or score < best_score:
                    best_score = score
                    best_position = position

        return TANK_CENTER.copy() if best_position is None else best_position

    def get_visit_gradient_direction(self, position):
        """
        Return a simple direction toward less visited nearby cells.

        This is not a reward. It is spatial novelty from the mental map.
        """
        grid_x, grid_y = self.position_to_cell(position)

        best_score = None
        best_cell = None

        for dy in [-2, -1, 0, 1, 2]:
            for dx in [-2, -1, 0, 1, 2]:
                nx = int(np.clip(grid_x + dx, 0, GRID_SIZE - 1))
                ny = int(np.clip(grid_y + dy, 0, GRID_SIZE - 1))
                score = self.visits[ny, nx]

                if best_score is None or score < best_score:
                    best_score = score
                    best_cell = (nx, ny)

        if best_cell is None:
            return np.array([0.0, 0.0], dtype=np.float32)

        target = self.cell_to_position(best_cell[0], best_cell[1])
        direction = target - position
        norm = np.linalg.norm(direction)

        if norm < 1e-6:
            return np.array([0.0, 0.0], dtype=np.float32)

        return direction / norm

    def save(self):
        """Save memory to disk."""
        data = {
            "visits": self.visits.tolist(),
            "platform_memory": self.platform_memory.tolist(),
            "seen_cues": self.seen_cues,
            "remembered_platform_position": (
                None if self.remembered_platform_position is None else self.remembered_platform_position.tolist()
            ),
            "best_cue_name": self.best_cue_name,
            "best_cue_position": (
                None if self.best_cue_position is None else self.best_cue_position.tolist()
            ),
            "best_cue_to_platform_distance": self.best_cue_to_platform_distance,
            "trial_number": self.trial_number,
            "trial_metrics": self.trial_metrics,
        }

        with open(self.memory_file, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)

        self.save_csv()
        self.save_plot()
        self.save_biological_interpretation()

    def load(self):
        """Load memory from disk."""
        if not self.memory_file.exists():
            return

        with open(self.memory_file, "r", encoding="utf-8") as file:
            data = json.load(file)

        self.visits = np.array(data.get("visits", self.visits.tolist()), dtype=np.float32)
        self.platform_memory = np.array(data.get("platform_memory", self.platform_memory.tolist()), dtype=np.float32)
        self.seen_cues = data.get("seen_cues", [])

        platform_position = data.get("remembered_platform_position")
        self.remembered_platform_position = (
            None if platform_position is None else np.array(platform_position, dtype=np.float32)
        )

        self.best_cue_name = data.get("best_cue_name")

        cue_position = data.get("best_cue_position")
        self.best_cue_position = None if cue_position is None else np.array(cue_position, dtype=np.float32)

        self.best_cue_to_platform_distance = data.get("best_cue_to_platform_distance")
        self.trial_number = int(data.get("trial_number", 1))
        self.trial_metrics = data.get("trial_metrics", [])

    def save_csv(self):
        """Save map and metrics as CSV."""
        rows = []

        for grid_y in range(GRID_SIZE):
            for grid_x in range(GRID_SIZE):
                rows.append({
                    "grid_x": grid_x,
                    "grid_y": grid_y,
                    "visits": float(self.visits[grid_y, grid_x]),
                    "platform_memory": float(self.platform_memory[grid_y, grid_x]),
                })

        pd.DataFrame(rows).to_csv(self.output_dir / "mental_map.csv", index=False)

        if self.trial_metrics:
            pd.DataFrame(self.trial_metrics).to_csv(self.output_dir / "trial_metrics.csv", index=False)

    def save_plot(self):
        """Save mental map as PNG."""
        figure, axis = plt.subplots(figsize=(6, 6))
        axis.imshow(self.visits, cmap="Blues", origin="upper")

        if self.platform_memory.max() > 0:
            ys, xs = np.where(self.platform_memory > 0)
            axis.scatter(xs, ys, marker="s", s=90, label="remembered platform")

        axis.set_title("Persistent spatial mental map")
        axis.set_xlabel("grid x")
        axis.set_ylabel("grid y")
        axis.legend(loc="upper right")
        figure.tight_layout()
        figure.savefig(self.output_dir / "mental_map.png", dpi=160)
        plt.close(figure)

    def biological_interpretation(self):
        """Return a biological interpretation of performance."""
        if not self.trial_metrics:
            return "No completed trials yet."

        dataframe = pd.DataFrame(self.trial_metrics)

        first = dataframe.iloc[0]
        last = dataframe.iloc[-1]

        latency_improved = last["latency_steps"] < first["latency_steps"]
        path_improved = last["path_length"] < first["path_length"]
        target_time_improved = last["target_quadrant_time"] >= first["target_quadrant_time"]
        crossings_improved = last["platform_crossings"] >= first["platform_crossings"]

        all_improved = latency_improved and path_improved and target_time_improved and crossings_improved

        lines = [
            "Biological interpretation of spatial memory",
            "=" * 52,
            "",
            "Better performance is indicated by:",
            "- lower latency;",
            "- shorter path length;",
            "- more time in the target quadrant;",
            "- more crossings over the platform location.",
            "",
            f"Completed trials: {len(dataframe)}",
            f"Initial latency: {first['latency_steps']} steps",
            f"Current latency: {last['latency_steps']} steps",
            f"Initial path length: {first['path_length']:.2f}",
            f"Current path length: {last['path_length']:.2f}",
            f"Initial target quadrant time: {first['target_quadrant_time']} steps",
            f"Current target quadrant time: {last['target_quadrant_time']} steps",
            f"Initial platform crossings: {first['platform_crossings']}",
            f"Current platform crossings: {last['platform_crossings']}",
            "",
        ]

        if all_improved:
            lines.append("Conclusion: performance improved in a way compatible with stronger spatial memory.")
        else:
            lines.append("Conclusion: spatial memory is still forming or metrics have not improved consistently yet.")

        return "\n".join(lines)

    def save_biological_interpretation(self):
        """Save biological interpretation text."""
        with open(self.output_dir / "biological_interpretation.txt", "w", encoding="utf-8") as file:
            file.write(self.biological_interpretation())
