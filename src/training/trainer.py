from typing import Dict, List, Optional
import logging

from tqdm import tqdm

from src.agents.defense_agent import DefenseAgent
from src.agents.experience_buffer import ExperienceBuffer
from src.environment.network_env import NetworkEnvironment
import torch

from src.visualization.training_visualizer import TrainingVisualizer

logger = logging.getLogger(__name__)


class TrainingManager:
    def __init__(
            self,
            env: NetworkEnvironment,
            config: Dict = None,
            visualizer: Optional['TrainingVisualizer'] = None,
            num_agents: Optional[int] = None,
            batch_size: Optional[int] = None,
            training_phases: Optional[List[Dict]] = None
    ):
        self.env = env
        self.config = config or self._default_config()
        self.visualizer = visualizer

        # Get number of agents from env if not specified
        self.num_agents = num_agents or len(env.graph.nodes())

        # Get batch size from config if not specified
        self.batch_size = batch_size or self.config['agent']['batch_size']

        # Get training phases from config if not specified
        self.training_phases = training_phases or self.config['phases']

        # Initialize agents
        self.agents = {
            str(i): DefenseAgent(
                node_id=str(i),
                observation_dim=env.observation_space.shape[0],
                hidden_dim=self.config['agent']['hidden_dim'],
                learning_rate=self.config['agent']['learning_rate']
            )
            for i in range(self.num_agents)
        }

        # Initialize experience buffers
        self.experience_buffers = {
            agent_id: ExperienceBuffer(capacity=self.config['agent']['buffer_size'])
            for agent_id in self.agents.keys()
        }

    def _default_config(self) -> Dict:
        """Provide default configuration if none is provided."""
        return {
            'agent': {
                'learning_rate': 0.001,
                'batch_size': 32,
                'hidden_dim': 128,
                'buffer_size': 10000,
                'update_interval': 10
            },
            'phases': [
                {
                    'name': "Static Attacks",
                    'episodes': 1000,
                    'attack_types': ['ransomware', 'apt', 'ddos', 'zeroday'],
                    'llm_attack_probability': 0.0,
                    'enable_collaboration': False
                },
                {
                    'name': "LLM Attacks",
                    'episodes': 1000,
                    'attack_types': ['ransomware', 'apt', 'ddos', 'zeroday', 'llm_driven'],
                    'llm_attack_probability': 0.2,
                    'enable_collaboration': False
                },
                {
                    'name': "Collaborative Defense",
                    'episodes': 1000,
                    'attack_types': ['ransomware', 'apt', 'ddos', 'zeroday', 'llm_driven'],
                    'llm_attack_probability': 0.3,
                    'enable_collaboration': True
                }
            ]
        }

    def train(self):
        """Execute full training pipeline."""
        for phase in tqdm(self.training_phases, desc='Training Phases', unit='phases', position=0, leave=True, total=len(self.training_phases)):
            logger.info(f"Starting training phase: {phase['name']}")
            logger.info(f"Attack Phase: {phase}")

            # Configure environment for this phase
            self.env.config['attack_probabilities']['llm_driven'] = phase['llm_attack_prob']

            for episode in tqdm(range(phase['episodes']), desc=phase['name'], unit='episodes', position=1, leave=True, total=phase['episodes']):
                self._train_episode(
                    enable_collaboration=phase.get('enable_collaboration', False)
                )

                if episode % 100 == 0:
                    self._evaluate()

    def _train_episode(self, enable_collaboration: bool = False):
        """Train for one episode."""
        observations = self.env.reset()
        episode_rewards = {agent_id: 0 for agent_id in self.agents.keys()}
        done = False

        while not done:
            # Collect actions from all agents
            actions = {}
            for agent_id, agent in self.agents.items():
                obs = torch.tensor(observations[agent_id], dtype=torch.float32)
                action, _ = agent.act(obs)
                actions[agent_id] = action

            # Execute actions in environment
            next_observations, rewards, dones, _ = self.env.step(actions)

            # Store experiences
            for agent_id in self.agents.keys():
                self.experience_buffers[agent_id].add(
                    torch.tensor(observations[agent_id], dtype=torch.float32),
                    actions[agent_id],
                    rewards[agent_id],
                    torch.tensor(next_observations[agent_id], dtype=torch.float32)
                )
                episode_rewards[agent_id] += rewards[agent_id]

            # Update observations
            observations = next_observations
            done = dones["__all__"]

            # Training step
            if len(self.experience_buffers[agent_id].states) >= self.batch_size:
                self._update_agents(enable_collaboration)

    def _update_agents(self, enable_collaboration: bool):
        """Update all agents using collected experience."""
        for agent_id, agent in self.agents.items():
            # Sample experience
            batch = self.experience_buffers[agent_id].sample(self.batch_size)

            # Update agent
            loss_info = agent.update(batch)

            if enable_collaboration:
                # Share experience with neighbors
                neighbors = list(self.env.graph.neighbors(agent_id))
                for neighbor_id in neighbors:
                    neighbor_batch = self.experience_buffers[neighbor_id].sample(
                        self.batch_size // 2
                    )
                    agent.update(neighbor_batch)

    def _evaluate(self):
        """Evaluate current policy performance."""
        eval_episodes = 10
        total_rewards = {agent_id: 0 for agent_id in self.agents.keys()}
        total_compromised = 0

        for _ in range(eval_episodes):
            observations = self.env.reset()
            done = False

            while not done:
                actions = {}
                for agent_id, agent in self.agents.items():
                    obs = torch.tensor(observations[agent_id], dtype=torch.float32)
                    action, _ = agent.act(obs)
                    actions[agent_id] = action

                observations, rewards, dones, infos = self.env.step(actions)

                for agent_id in self.agents.keys():
                    total_rewards[agent_id] += rewards[agent_id]

                total_compromised += infos[list(infos.keys())[0]]['compromised_nodes']
                done = dones["__all__"]

        # Log evaluation results
        avg_reward = sum(total_rewards.values()) / (len(self.agents) * eval_episodes)
        avg_compromised = total_compromised / eval_episodes

        logger.info(f"Evaluation - Avg Reward: {avg_reward:.2f}, "
                    f"Avg Compromised Nodes: {avg_compromised:.2f}")