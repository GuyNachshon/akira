from typing import Dict
from dataclasses import dataclass

from src.environment.network_env import NetworkEnvironment
from src.environment.state import NodeState


@dataclass
class RewardConfig:
    detection_success: float = 1.0
    false_positive: float = -0.5
    resource_usage: float = -0.1
    network_health: float = 2.0
    collaboration_bonus: float = 0.3


class RewardCalculator:
    def __init__(self, config: RewardConfig = None):
        self.config = config or RewardConfig()
        self.previous_states = {}

    def calculate_rewards(
            self,
            env: NetworkEnvironment,
            actions: Dict[str, str],
            agent_observations: Dict[str, Dict]
    ) -> Dict[str, float]:
        """Calculate rewards for all agents based on their actions and environment state."""
        rewards = {}

        # Calculate global network health
        network_health = self._calculate_network_health(env)

        for agent_id, action in actions.items():
            reward = 0.0

            # Get agent's local state and action results
            local_reward = self._calculate_local_reward(
                env, agent_id, action, agent_observations[agent_id]
            )
            reward += local_reward

            # Add network health component
            reward += self.config.network_health * network_health

            # Add collaboration bonus if applicable
            if self._check_collaboration(env, agent_id, action):
                reward += self.config.collaboration_bonus

            rewards[agent_id] = reward

        return rewards

    def _calculate_network_health(self, env: NetworkEnvironment) -> float:
        """Calculate overall network health score."""
        total_nodes = len(env.graph.nodes())
        compromised = sum(
            1 for node in env.graph.nodes()
            if env.graph.nodes[node]['data'].state == NodeState.COMPROMISED
        )
        isolated = sum(
            1 for node in env.graph.nodes()
            if env.graph.nodes[node]['data'].state == NodeState.ISOLATED
        )

        health_score = 1.0 - (compromised + isolated * 0.5) / total_nodes
        return max(0.0, health_score)

    def _calculate_local_reward(
            self,
            env: NetworkEnvironment,
            agent_id: str,
            action: str,
            observation: Dict
    ) -> float:
        """Calculate reward based on local action and observation."""
        reward = 0.0
        node = env.graph.nodes[agent_id]['data']

        # Reward for successful detection
        if action == "scan":
            neighbors = list(env.graph.neighbors(agent_id))
            compromised_neighbors = sum(
                1 for n in neighbors
                if env.graph.nodes[n]['data'].state == NodeState.COMPROMISED
            )
            if compromised_neighbors > 0:
                reward += self.config.detection_success * compromised_neighbors
            else:
                reward += self.config.resource_usage  # Small penalty for unnecessary scan

        # Penalty for unnecessary isolation
        elif action == "isolate":
            if node.state != NodeState.COMPROMISED:
                reward += self.config.false_positive

        # Resource usage penalty
        reward += self.config.resource_usage

        return reward

    def _check_collaboration(
            self,
            env: NetworkEnvironment,
            agent_id: str,
            action: str
    ) -> bool:
        """Check if agent's action helped neighboring nodes."""
        if action not in ["scan", "isolate"]:
            return False

        neighbors = list(env.graph.neighbors(agent_id))
        neighbor_states = [
            env.graph.nodes[n]['data'].state for n in neighbors
        ]

        # Check if action prevented spread to neighbors
        return any(
            state == NodeState.SAFE for state in neighbor_states
        ) and any(
            state == NodeState.COMPROMISED for state in neighbor_states
        )

    def reset(self):
        """Reset reward calculator state."""
        self.previous_states = {}