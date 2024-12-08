import random
from typing import Dict, List
import torch


class ExperienceBuffer:
    def __init__(self, capacity: int = 10000):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.capacity = capacity

    def add(self, state, action, reward, next_state):
        """Add experience to buffer."""
        if len(self.states) >= self.capacity:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of experiences."""
        indices = random.sample(range(len(self.states)), min(batch_size, len(self.states)))

        return {
            'states': torch.stack([self.states[i] for i in indices]),
            'actions': torch.tensor([self.actions[i] for i in indices]),
            'rewards': torch.tensor([self.rewards[i] for i in indices]),
            'next_states': torch.stack([self.next_states[i] for i in indices])
        }
