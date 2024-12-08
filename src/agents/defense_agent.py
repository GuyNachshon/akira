from typing import Dict, List, Tuple

import numpy as np
import torch
from .hierarchical_policy import HierarchicalPolicy
import logging

logger = logging.getLogger(__name__)


class DefenseAgent:
    def __init__(
            self,
            node_id: str,
            observation_dim: int = 15,
            hidden_dim: int = 128,
            num_actions: int = 5,
            learning_rate: float = 0.001
    ):
        self.node_id = node_id
        self.observation_dim = observation_dim
        logger.debug(f"Initializing DefenseAgent {node_id} with observation_dim={observation_dim}")

        self.policy = HierarchicalPolicy(
            input_dim=observation_dim,
            hidden_dim=hidden_dim,
            num_actions=num_actions
        )
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=learning_rate
        )

    def act(self, observation: np.ndarray) -> Tuple[int, Dict[str, torch.Tensor]]:
        """Select an action based on current observation."""
        logger.debug(f"Agent {self.node_id} received observation shape: {observation.shape}")

        if observation.shape[0] != self.observation_dim:
            raise ValueError(
                f"Observation dimension mismatch. Expected {self.observation_dim}, "
                f"got {observation.shape[0]}"
            )

        with torch.no_grad():
            observation_tensor = torch.FloatTensor(observation)
            try:
                strategy, action_probs = self.policy(observation_tensor)
            except Exception as e:
                logger.error(f"Error in policy forward pass: {str(e)}")
                logger.error(f"Observation tensor shape: {observation_tensor.shape}")
                raise

            action = torch.multinomial(action_probs[0], 1).item()

        return action, {
            'strategy': strategy,
            'action_probs': action_probs
        }

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update policy using collected experience."""
        # Unpack batch
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']

        # Calculate advantages (simple version)
        advantages = rewards - rewards.mean()

        # Get policy outputs
        strategies, action_probs = self.policy(states)

        # Calculate losses
        action_log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)))

        # Policy gradient loss
        pg_loss = -(advantages * action_log_probs).mean()

        # Strategy diversity loss (optional)
        entropy_loss = -0.01 * (strategies * torch.log(strategies + 1e-10)).sum(dim=-1).mean()

        # Total loss
        total_loss = pg_loss + entropy_loss

        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            'pg_loss': pg_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item()
        }
