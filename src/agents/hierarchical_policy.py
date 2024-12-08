import numpy as np
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class HierarchicalPolicy(nn.Module):
    def __init__(self, input_dim: int = 15, hidden_dim: int = 128, num_actions: int = 5):
        super().__init__()
        self.input_dim = input_dim
        logger.debug(f"Initializing HierarchicalPolicy with input_dim={input_dim}")

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

    def forward(self, state):
        # Shape validation and conversion
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)

        # Add batch dimension if necessary
        if state.dim() == 1:
            state = state.unsqueeze(0)

        logger.debug(f"State shape: {state.shape}")

        # Validate input dimension
        if state.shape[1] != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {state.shape[1]}")

        # Generate strategic decision
        strategy = self.strategic_policy(state)
        logger.debug(f"Strategy shape: {strategy.shape}")

        # Combine state and strategy for tactical decision
        combined = torch.cat([state, strategy], dim=-1)
        logger.debug(f"Combined shape: {combined.shape}")

        action_probs = self.tactical_policy(combined)
        logger.debug(f"Action probs shape: {action_probs.shape}")

        return strategy, action_probs
