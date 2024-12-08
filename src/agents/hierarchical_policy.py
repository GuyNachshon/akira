from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class HierarchicalPolicy(nn.Module):
    def __init__(self, input_dim: int = 15, hidden_dim: int = 128, num_actions: int = 5):
        super().__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions

        # Strategic policy (high-level decisions)
        self.strategic_policy = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4),  # 4 high-level strategies
            nn.Softmax(dim=-1)
        )

        # Tactical policy (low-level actions)
        self.tactical_policy = nn.Sequential(
            nn.Linear(input_dim + 4, hidden_dim),  # Input + strategy
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with dimension handling."""
        # Ensure state is a tensor
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)

        # Add batch dimension if necessary
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Generate strategic decision
        strategy = self.strategic_policy(state)

        # Combine state and strategy for tactical decision
        combined = torch.cat([state, strategy], dim=-1)
        action_probs = self.tactical_policy(combined)

        return strategy, action_probs

    def _validate_input(self, state: torch.Tensor) -> None:
        """Validate input tensor dimensions and values."""
        if state.dim() not in [1, 2]:
            raise ValueError(f"Expected state dimension 1 or 2, got {state.dim()}")

        input_size = state.size(-1)
        if input_size != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {input_size}")

        if torch.isnan(state).any() or torch.isinf(state).any():
            raise ValueError("Input contains NaN or Inf values")
