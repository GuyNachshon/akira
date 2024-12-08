import os
import torch
import json
from typing import Dict, Optional
from datetime import datetime
import logging

from src.agents.defense_agent import DefenseAgent
from src.environment.network_env import NetworkEnvironment
from src.training.model_validator import ModelValidator

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self, base_dir: str = "saved_models"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self.validator = None  # Initialize later when we have an environment

    def save_model(
            self,
            agents: Dict[str, DefenseAgent],
            config: dict,
            metrics: dict,
            env: NetworkEnvironment,
            phase: str = None,
            validate: bool = True
    ) -> Optional[str]:
        """Save model with optional validation."""
        if validate:
            if self.validator is None:
                self.validator = ModelValidator(env)

            # Validate model before saving
            passed, val_metrics = self.validator.validate_model(agents)
            if not passed:
                logger.warning("Model validation failed, model will not be saved")
                return None

            # Add validation metrics
            metrics['validation'] = val_metrics.to_dict()

        # Create save directory and save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(self.base_dir, f"model_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)

        # Save validation status
        if validate:
            validation_path = os.path.join(save_dir, "validation_results.json")
            with open(validation_path, 'w') as f:
                json.dump(metrics['validation'], f, indent=2)

        # Save model components
        for agent_id, agent in agents.items():
            agent_dir = os.path.join(save_dir, f"agent_{agent_id}")
            os.makedirs(agent_dir, exist_ok=True)

            torch.save(agent.policy.state_dict(), os.path.join(agent_dir, "policy.pth"))
            torch.save(agent.optimizer.state_dict(), os.path.join(agent_dir, "optimizer.pth"))

        # Save other components
        with open(os.path.join(save_dir, "config.json"), 'w') as f:
            json.dump(config, f, indent=2)

        with open(os.path.join(save_dir, "metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=2)

        if phase:
            with open(os.path.join(save_dir, "phase_info.txt"), 'w') as f:
                f.write(phase)

        logger.info(f"Model saved to {save_dir}")
        return save_dir

    def load_model(
            self,
            model_dir: str,
            env: NetworkEnvironment,
            validate: bool = True
    ) -> Optional[Dict[str, DefenseAgent]]:
        """Load model with optional validation."""
        try:
            # Load configuration and initialize agents
            with open(os.path.join(model_dir, "config.json"), 'r') as f:
                config = json.load(f)

            agents = {}
            for agent_dir in os.listdir(model_dir):
                if agent_dir.startswith("agent_"):
                    agent_id = agent_dir.split("_")[1]
                    agent_path = os.path.join(model_dir, agent_dir)

                    agent = DefenseAgent(
                        node_id=agent_id,
                        observation_dim=env.obs_dim,
                        hidden_dim=config['agent']['hidden_dim'],
                        learning_rate=config['agent']['learning_rate']
                    )

                    # Load states
                    agent.policy.load_state_dict(
                        torch.load(os.path.join(agent_path, "policy.pth"))
                    )
                    agent.optimizer.load_state_dict(
                        torch.load(os.path.join(agent_path, "optimizer.pth"))
                    )

                    agents[agent_id] = agent

            # Validate loaded model if requested
            if validate:
                if self.validator is None:
                    self.validator = ModelValidator(env)

                passed, val_metrics = self.validator.validate_model(agents)
                if not passed:
                    logger.warning("Loaded model failed validation")
                    return None

                logger.info("Loaded model passed validation")
                logger.info(f"Validation metrics: {val_metrics.to_dict()}")

            return agents

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None
