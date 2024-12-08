import os
import torch
import json
from typing import Dict, Optional
from datetime import datetime
import logging

from src.agents.defense_agent import DefenseAgent

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self, base_dir: str = "saved_models"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def save_model(
            self,
            agents: Dict[str, DefenseAgent],
            config: dict,
            metrics: dict,
            phase: str = None
    ) -> str:
        """Save trained agents and training information."""
        # Create timestamp-based directory for this save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(self.base_dir, f"model_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)

        # Save each agent's policy
        for agent_id, agent in agents.items():
            agent_dir = os.path.join(save_dir, f"agent_{agent_id}")
            os.makedirs(agent_dir, exist_ok=True)

            # Save policy state
            model_path = os.path.join(agent_dir, "policy.pth")
            torch.save(agent.policy.state_dict(), model_path)

            # Save optimizer state
            optimizer_path = os.path.join(agent_dir, "optimizer.pth")
            torch.save(agent.optimizer.state_dict(), optimizer_path)

        # Save configuration
        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        # Save metrics
        metrics_path = os.path.join(save_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        # Save phase information if provided
        if phase:
            phase_path = os.path.join(save_dir, "phase_info.txt")
            with open(phase_path, 'w') as f:
                f.write(phase)

        logger.info(f"Model saved to {save_dir}")
        return save_dir

    def load_model(
            self,
            model_dir: str,
            observation_dim: int = 15
    ) -> Dict[str, DefenseAgent]:
        """Load saved agents and their policies."""
        # Load configuration
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)

        agents = {}
        # Load each agent
        for agent_dir in os.listdir(model_dir):
            if agent_dir.startswith("agent_"):
                agent_id = agent_dir.split("_")[1]
                agent_path = os.path.join(model_dir, agent_dir)

                # Initialize new agent
                agent = DefenseAgent(
                    node_id=agent_id,
                    observation_dim=observation_dim,
                    hidden_dim=config['agent']['hidden_dim'],
                    learning_rate=config['agent']['learning_rate']
                )

                # Load policy state
                policy_path = os.path.join(agent_path, "policy.pth")
                agent.policy.load_state_dict(
                    torch.load(policy_path)
                )

                # Load optimizer state
                optimizer_path = os.path.join(agent_path, "optimizer.pth")
                agent.optimizer.load_state_dict(
                    torch.load(optimizer_path)
                )

                agents[agent_id] = agent

        logger.info(f"Loaded {len(agents)} agents from {model_dir}")
        return agents

    def get_latest_model(self) -> Optional[str]:
        """Get the path to the most recently saved model."""
        if not os.path.exists(self.base_dir):
            return None

        model_dirs = [
            d for d in os.listdir(self.base_dir)
            if d.startswith("model_")
        ]

        if not model_dirs:
            return None

        latest_dir = max(model_dirs)
        return os.path.join(self.base_dir, latest_dir)
