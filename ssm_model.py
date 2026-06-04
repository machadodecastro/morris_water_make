"""
ssm_model.py

A small self-supervised State Space Model.

The model learns from the agent's own movement sequence:

    observation_t + action_t + hidden_state_t -> hidden_state_{t+1}
    hidden_state_{t+1} -> predicted_observation_{t+1}

Training signal:
    prediction error between predicted_observation_{t+1}
    and actual observation_{t+1}.

This is self-supervised learning.
It is not reinforcement learning.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class SimpleSSM(nn.Module):
    """Small recurrent state-space model."""

    def __init__(self, observation_size, action_size=2, hidden_size=64):
        super().__init__()

        self.hidden_size = hidden_size

        self.state_update = nn.Sequential(
            nn.Linear(hidden_size + observation_size + action_size, 128),
            nn.Tanh(),
            nn.Linear(128, hidden_size),
            nn.Tanh(),
        )

        self.observation_predictor = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, observation_size),
        )

    def forward_step(self, observation, action, hidden_state):
        """Run one SSM step."""
        model_input = torch.cat([hidden_state, observation, action], dim=-1)
        next_hidden_state = self.state_update(model_input)
        predicted_next_observation = self.observation_predictor(next_hidden_state)
        return predicted_next_observation, next_hidden_state


class SelfSupervisedSSMTrainer:
    """Trainer and persistence helper for the SSM."""

    def __init__(self, observation_size, output_dir="outputs", device=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = SimpleSSM(
            observation_size=observation_size,
            action_size=2,
            hidden_size=64,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.loss_function = nn.MSELoss()

        self.hidden_state = torch.zeros(1, 64, device=self.device)
        self.training_buffer = []
        self.training_log = []

        self.model_file = self.output_dir / "ssm_model.pt"
        self.log_file = self.output_dir / "ssm_training_log.csv"

        self.load()

    def reset_hidden_state(self):
        """Reset only the short-term hidden state, not the learned weights."""
        self.hidden_state = torch.zeros(1, 64, device=self.device)

    def reset_model(self):
        """Reset SSM weights and delete saved model."""
        if self.model_file.exists():
            self.model_file.unlink()

        if self.log_file.exists():
            self.log_file.unlink()

        observation_size = self.model.observation_predictor[-1].out_features

        self.model = SimpleSSM(
            observation_size=observation_size,
            action_size=2,
            hidden_size=64,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.hidden_state = torch.zeros(1, 64, device=self.device)
        self.training_buffer = []
        self.training_log = []

    def observe_transition(self, observation, action, next_observation):
        """Store one self-supervised transition."""
        self.training_buffer.append((
            observation.astype(np.float32),
            action.astype(np.float32),
            next_observation.astype(np.float32),
        ))

        if len(self.training_buffer) > 6000:
            self.training_buffer.pop(0)

    def update_hidden_state(self, observation, action):
        """
        Update the SSM hidden state online.

        This hidden state is used as a compact temporal context.
        """
        with torch.no_grad():
            observation_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            action_tensor = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
            _, self.hidden_state = self.model.forward_step(
                observation_tensor,
                action_tensor,
                self.hidden_state,
            )

    def train_step(self, batch_size=64):
        """Run one self-supervised training step."""
        if len(self.training_buffer) < batch_size:
            return None

        indices = np.random.choice(len(self.training_buffer), size=batch_size, replace=False)

        observations = []
        actions = []
        next_observations = []

        for index in indices:
            observation, action, next_observation = self.training_buffer[index]
            observations.append(observation)
            actions.append(action)
            next_observations.append(next_observation)

        observations = torch.tensor(np.array(observations), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32, device=self.device)
        next_observations = torch.tensor(np.array(next_observations), dtype=torch.float32, device=self.device)

        hidden_state = torch.zeros(batch_size, 64, device=self.device)

        predicted_next_observations, _ = self.model.forward_step(
            observations,
            actions,
            hidden_state,
        )

        loss = self.loss_function(predicted_next_observations, next_observations)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_value = float(loss.item())
        self.training_log.append({
            "step": len(self.training_log) + 1,
            "prediction_loss": loss_value,
        })

        if len(self.training_log) % 50 == 0:
            self.save()

        return loss_value

    def hidden_direction_bias(self):
        """
        Convert hidden state into a small movement bias.

        This is intentionally weak. The mental map remains the main memory.
        """
        hidden = self.hidden_state.detach().cpu().numpy()[0]
        x = float(np.tanh(hidden[0]))
        y = float(np.tanh(hidden[1]))
        vector = np.array([x, y], dtype=np.float32)

        norm = np.linalg.norm(vector)
        if norm < 1e-6:
            return np.array([0.0, 0.0], dtype=np.float32)

        return vector / norm

    def save(self):
        """Save model and training log."""
        torch.save(self.model.state_dict(), self.model_file)

        if self.training_log:
            pd.DataFrame(self.training_log).to_csv(self.log_file, index=False)

    def load(self):
        """Load model if it exists."""
        if self.model_file.exists():
            state = torch.load(self.model_file, map_location=self.device)
            self.model.load_state_dict(state)

        if self.log_file.exists():
            self.training_log = pd.read_csv(self.log_file).to_dict("records")
