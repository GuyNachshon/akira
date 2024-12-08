import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

from src.agents.defense_agent import DefenseAgent
from src.defense.actions import NetworkAction
from src.environment.network_env import NetworkEnvironment

logger = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    """Container for validation metrics."""
    mean_reward: float
    success_rate: float
    response_time: float
    false_positive_rate: float
    network_health: float
    compromised_ratio: float
    model_consistency: float

    def to_dict(self) -> Dict:
        return {
            'mean_reward': self.mean_reward,
            'success_rate': self.success_rate,
            'response_time': self.response_time,
            'false_positive_rate': self.false_positive_rate,
            'network_health': self.network_health,
            'compromised_ratio': self.compromised_ratio,
            'model_consistency': self.model_consistency
        }


class ModelValidator:
    def __init__(
            self,
            env: NetworkEnvironment,
            validation_episodes: int = 100,
            performance_threshold: float = 0.6,
            consistency_threshold: float = 0.8
    ):
        self.env = env
        self.validation_episodes = validation_episodes
        self.performance_threshold = performance_threshold
        self.consistency_threshold = consistency_threshold

    def validate_model(self, agents: Dict[str, DefenseAgent]) -> Tuple[bool, ValidationMetrics]:
        """Perform comprehensive model validation."""
        # Check model structure
        if not self._validate_model_structure(agents):
            return False, None

        # Validate model weights
        if not self._validate_model_weights(agents):
            return False, None

        # Perform validation episodes
        metrics = self._run_validation_episodes(agents)

        # Check if metrics meet thresholds
        passed = self._check_validation_thresholds(metrics)

        return passed, metrics

    def _validate_model_structure(self, agents: Dict[str, DefenseAgent]) -> bool:
        """Validate the structure of agent models."""
        try:
            for agent_id, agent in agents.items():
                # Check policy components exist
                if not hasattr(agent, 'policy') or not agent.policy:
                    logger.error(f"Agent {agent_id} missing policy")
                    return False

                # Validate input/output dimensions
                input_dim = agent.policy.input_dim
                if input_dim != self.env.obs_dim:
                    logger.error(
                        f"Agent {agent_id} input dimension mismatch: "
                        f"expected {self.env.obs_dim}, got {input_dim}"
                    )
                    return False

                # Validate action space
                num_actions = agent.policy.num_actions
                if num_actions != NetworkAction.get_action_space_size():
                    logger.error(
                        f"Agent {agent_id} action space mismatch: "
                        f"expected {NetworkAction.get_action_space_size()}, got {num_actions}"
                    )
                    return False

            return True

        except Exception as e:
            logger.error(f"Model structure validation failed: {str(e)}")
            return False

    def _validate_model_weights(self, agents: Dict[str, DefenseAgent]) -> bool:
        """Validate model weights for numerical stability."""
        try:
            for agent_id, agent in agents.items():
                # Check for NaN or infinite values
                for name, param in agent.policy.named_parameters():
                    if torch.isnan(param.data).any():
                        logger.error(f"Agent {agent_id}: NaN found in {name}")
                        return False
                    if torch.isinf(param.data).any():
                        logger.error(f"Agent {agent_id}: Inf found in {name}")
                        return False

                    # Check for unreasonably large values
                    if param.data.abs().max() > 1e6:
                        logger.warning(
                            f"Agent {agent_id}: Large values found in {name}: "
                            f"max abs value = {param.data.abs().max()}"
                        )

            return True

        except Exception as e:
            logger.error(f"Model weights validation failed: {str(e)}")
            return False

    def _run_validation_episodes(self, agents: Dict[str, DefenseAgent]) -> ValidationMetrics:
        """Run validation episodes and collect metrics."""
        total_rewards = []
        success_counts = 0
        response_times = []
        false_positives = 0
        health_scores = []
        compromised_ratios = []
        action_consistency = []

        for episode in range(self.validation_episodes):
            episode_metrics = self._run_single_validation_episode(agents)

            total_rewards.append(episode_metrics['total_reward'])
            success_counts += episode_metrics['successes']
            response_times.extend(episode_metrics['response_times'])
            false_positives += episode_metrics['false_positives']
            health_scores.append(episode_metrics['health_score'])
            compromised_ratios.append(episode_metrics['compromised_ratio'])
            action_consistency.append(episode_metrics['action_consistency'])

        return ValidationMetrics(
            mean_reward=np.mean(total_rewards),
            success_rate=success_counts / (self.validation_episodes * len(agents)),
            response_time=np.mean(response_times) if response_times else float('inf'),
            false_positive_rate=false_positives / (self.validation_episodes * len(agents)),
            network_health=np.mean(health_scores),
            compromised_ratio=np.mean(compromised_ratios),
            model_consistency=np.mean(action_consistency)
        )

    def _run_single_validation_episode(self, agents: Dict[str, DefenseAgent]) -> Dict:
        """Run a single validation episode and collect metrics."""
        obs = self.env.reset()
        done = False
        total_reward = 0
        successes = 0
        response_times = []
        false_positives = 0
        previous_actions = {agent_id: None for agent_id in agents}
        action_consistency = []

        while not done:
            actions = {}
            for agent_id, agent in agents.items():
                action, info = agent.act(obs[agent_id])
                actions[agent_id] = action

                # Check action consistency
                if previous_actions[agent_id] is not None:
                    action_consistency.append(
                        1.0 if action == previous_actions[agent_id] else 0.0
                    )
                previous_actions[agent_id] = action

            next_obs, rewards, dones, infos = self.env.step(actions)

            # Update metrics
            total_reward += sum(rewards.values())

            # Track successful defenses and response times
            for agent_id, info in infos.items():
                if info.get('attack_prevented', False):
                    successes += 1
                    response_times.append(info.get('response_time', 0))
                if info.get('false_positive', False):
                    false_positives += 1

            obs = next_obs
            done = dones["__all__"]

        return {
            'total_reward': total_reward,
            'successes': successes,
            'response_times': response_times,
            'false_positives': false_positives,
            'health_score': self.env._calculate_network_health(),
            'compromised_ratio': self.env._count_compromised_nodes() / self.env.num_nodes,
            'action_consistency': np.mean(action_consistency) if action_consistency else 1.0
        }

    def _check_validation_thresholds(self, metrics: ValidationMetrics) -> bool:
        """Check if validation metrics meet required thresholds."""
        checks = [
            (metrics.success_rate >= self.performance_threshold,
             f"Success rate {metrics.success_rate:.2f} below threshold {self.performance_threshold}"),

            (metrics.false_positive_rate <= (1 - self.performance_threshold),
             f"False positive rate {metrics.false_positive_rate:.2f} above threshold"),

            (metrics.network_health >= self.performance_threshold,
             f"Network health {metrics.network_health:.2f} below threshold"),

            (metrics.model_consistency >= self.consistency_threshold,
             f"Model consistency {metrics.model_consistency:.2f} below threshold")
        ]

        passed = True
        for check, message in checks:
            if not check:
                logger.warning(f"Validation check failed: {message}")
                passed = False

        return passed