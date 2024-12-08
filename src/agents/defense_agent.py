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
        with torch.no_grad():
            observation_tensor = torch.FloatTensor(observation)
            strategy, action_probs = self.policy(observation_tensor)

            # Sample action from probability distribution
            action = torch.multinomial(action_probs[0], 1).item()

        return action, {
            'strategy': strategy,
            'action_probs': action_probs
        }

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update policy using collected experience."""
        # Unpack batch
        states = batch['states']
        actions = batch['actions'].view(-1, 1)  # Reshape to [batch_size, 1]
        rewards = batch['rewards'].view(-1, 1)  # Reshape to [batch_size, 1]

        # Calculate advantages (simple version)
        advantages = rewards - rewards.mean()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Get policy outputs
        strategies, action_probs = self.policy(states)

        # Calculate action log probabilities
        action_log_probs = torch.log(
            action_probs.gather(1, actions)  # Now dimensions match
        )

        # Policy gradient loss
        pg_loss = -(advantages * action_log_probs).mean()

        # Strategy diversity loss (optional)
        entropy_loss = -0.01 * (
                strategies * torch.log(strategies + 1e-10)
        ).sum(dim=-1).mean()

        # Total loss
        total_loss = pg_loss + entropy_loss

        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.optimizer.step()

        return {
            'pg_loss': pg_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item(),
            'mean_reward': rewards.mean().item(),
            'advantage_std': advantages.std().item()
        }

    def _validate_batch(self, batch: Dict[str, torch.Tensor]) -> None:
        """Validate batch dimensions and types."""
        expected_keys = ['states', 'actions', 'rewards', 'next_states']
        for key in expected_keys:
            if key not in batch:
                raise ValueError(f"Batch missing key: {key}")

            if not isinstance(batch[key], torch.Tensor):
                raise TypeError(f"Batch {key} must be a torch.Tensor")

        batch_size = batch['states'].size(0)
        if batch['actions'].size(0) != batch_size:
            raise ValueError(f"Actions batch size mismatch: {batch['actions'].size(0)} vs {batch_size}")
        if batch['rewards'].size(0) != batch_size:
            raise ValueError(f"Rewards batch size mismatch: {batch['rewards'].size(0)} vs {batch_size}")
