import torch
import numpy as np
from typing import Dict, List
import random


class ExperienceBuffer:
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []

    def add(self, state, action, reward, next_state):
        """Add experience to buffer with tensor conversion."""
        # Convert numpy arrays to tensors if needed
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        if isinstance(next_state, np.ndarray):
            next_state = torch.FloatTensor(next_state)

        # Ensure reward is a tensor
        if not isinstance(reward, torch.Tensor):
            reward = torch.FloatTensor([reward])

        # Convert action to tensor if needed
        if not isinstance(action, torch.Tensor):
            action = torch.LongTensor([action])

        # Remove oldest experience if buffer is full
        if len(self.states) >= self.capacity:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)

        # Add new experience
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of experiences."""
        batch_size = min(batch_size, len(self.states))
        indices = random.sample(range(len(self.states)), batch_size)

        return {
            'states': torch.stack([self.states[i] for i in indices]),
            'actions': torch.stack([self.actions[i] for i in indices]),
            'rewards': torch.stack([self.rewards[i] for i in indices]),
            'next_states': torch.stack([self.next_states[i] for i in indices])
        }

    def clear(self):
        """Clear the buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []

    def __len__(self):
        return len(self.states)
