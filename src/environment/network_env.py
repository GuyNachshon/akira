import random
from typing import Dict, Tuple, List
import networkx as nx
import numpy as np
from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from src.attacks.attack_types import AttackType
from src.attacks.llm_generator import LLMAttackGenerator
from src.defense.actions import NetworkAction
from src.environment.node import Node
from src.environment.state import NodeState
import logging

logger = logging.getLogger(__name__)


class NetworkEnvironment(MultiAgentEnv):
    def __init__(
            self,
            num_nodes: int,
            false_positive_rate: float = 0.05,
            compromise_detection_rate: float = 0.8,
            config: dict = None
    ):
        super().__init__()
        self.graph = nx.Graph()
        self.num_nodes = num_nodes
        self.false_positive_rate = false_positive_rate
        self.compromise_detection_rate = compromise_detection_rate
        self.config = config or self._default_config()
        self.current_step = 0
        self.max_steps = 1000
        self.llm_attack_generator = LLMAttackGenerator()

        # Define observation space components:
        # [
        #   node_state (5 values: safe, compromised, overloaded, isolated, scanning),
        #   vulnerability score (1 value),
        #   node value (1 value),
        #   neighbor states summary (5 values),
        #   network health indicators (3 values)
        # ]
        self.obs_dim = 15

        # Define action and observation spaces
        self.action_space = spaces.Discrete(NetworkAction.get_action_space_size())
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.obs_dim,),
            dtype=np.float32
        )

        logger.info(f"Initialized environment with observation dimension: {self.obs_dim}")
        self._initialize_network()

    def _initialize_network(self):
        """Initialize the network with nodes and edges."""
        # Clear existing graph
        self.graph.clear()

        # Create nodes with random vulnerabilities
        for i in range(self.num_nodes):
            node = Node(
                id=str(i),
                vulnerability=round(random.uniform(1, 10), 1),
                state=NodeState.SAFE,
                value=round(random.uniform(1, 10), 1),
                last_scan_time=0,
                isolation_time=0
            )
            self.graph.add_node(str(i), data=node)

        # Create random edges (ensuring connectivity)
        # First create a minimum spanning tree to ensure connectivity
        nodes = list(self.graph.nodes())
        for i in range(1, len(nodes)):
            j = random.randint(0, i - 1)
            self.graph.add_edge(nodes[i], nodes[j])

        # Add some additional random edges for higher connectivity
        num_extra_edges = self.num_nodes // 2
        for _ in range(num_extra_edges):
            i, j = random.sample(nodes, 2)
            if not self.graph.has_edge(i, j):
                self.graph.add_edge(i, j)

    def _default_config(self) -> dict:
        return {
            'reward_weights': {
                'detection': 1.0,
                'false_positive': -0.5,
                'resource_usage': -0.1,
                'network_health': 2.0
            },
            'attack_probabilities': {
                'ransomware': 0.2,
                'apt': 0.3,
                'ddos': 0.2,
                'zeroday': 0.1,
                'llm_driven': 0.2,
            }
        }

    def reset(self):
        """Reset the environment to initial state."""
        self.current_step = 0
        self._initialize_network()
        return self._get_observations()

    def step(self, action_dict: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, Dict]]:
        """Execute one step in the environment."""
        self.current_step += 1

        # Convert numeric actions to NetworkAction enums and execute
        for agent_id, action in action_dict.items():
            try:
                network_action = NetworkAction.from_index(action)
                self._execute_action(agent_id, network_action)
            except ValueError as e:
                logger.error(f"Invalid action {action} from agent {agent_id}: {e}")
                # Use MONITOR as default action in case of error
                self._execute_action(agent_id, NetworkAction.MONITOR)

        # Generate and execute attacks
        self._execute_attacks()

        # Get new observations
        observations = self._get_observations()

        # Calculate rewards
        rewards = self._calculate_rewards(action_dict)

        # Check if episode is done
        dones = {
            agent_id: self.current_step >= self.max_steps
            for agent_id in action_dict.keys()
        }
        dones["__all__"] = self.current_step >= self.max_steps

        # Additional info
        infos = {
            agent_id: {
                'network_health': self.calculate_network_health(),
                'compromised_nodes': self._count_compromised_nodes(),
                'action_taken': NetworkAction.from_index(action).name
            }
            for agent_id, action in action_dict.items()
        }

        return observations, rewards, dones, infos

    def _execute_action(self, agent_id: str, action: NetworkAction) -> None:
        """Execute a single agent's action."""
        node = self.graph.nodes[agent_id]['data']

        if action == NetworkAction.SCAN:
            node.state = NodeState.SCANNING
            node.last_scan_time = self.current_step
            logger.debug(f"Agent {agent_id} performed SCAN")

        elif action == NetworkAction.ISOLATE:
            if node.state != NodeState.ISOLATED:
                node.state = NodeState.ISOLATED
                node.isolation_time = self.current_step
                logger.debug(f"Agent {agent_id} performed ISOLATE")

        elif action == NetworkAction.RESTORE:
            if node.state in [NodeState.COMPROMISED, NodeState.OVERLOADED, NodeState.ISOLATED]:
                node.state = NodeState.SAFE
                logger.debug(f"Agent {agent_id} performed RESTORE")

        elif action == NetworkAction.PATCH:
            node.vulnerability = max(1, node.vulnerability - 1)
            logger.debug(f"Agent {agent_id} performed PATCH")

        elif action == NetworkAction.MONITOR:
            # Default monitoring action
            logger.debug(f"Agent {agent_id} performed MONITOR")

    def _execute_attacks(self):
        """Execute random attacks based on configuration."""
        for attack_type, prob in self.config['attack_probabilities'].items():
            if random.random() < prob:
                if attack_type == 'llm_driven':
                    self._execute_llm_attack()
                else:
                    self._execute_static_attack(AttackType(attack_type))

    def _execute_llm_attack(self):
        """Execute an LLM-generated attack sequence."""
        network_state = {
            node: self.graph.nodes[node]['data']
            for node in self.graph.nodes()
        }

        attack_sequence = self.llm_attack_generator.generate_attack_sequence(network_state)

        for step in attack_sequence:
            target_nodes = self._select_attack_targets(step['target_type'])
            for node_id in target_nodes:
                node = self.graph.nodes[node_id]['data']
                if random.random() < step['probability']:
                    node.state = NodeState.COMPROMISED

    def _execute_static_attack(self, attack_type: AttackType):
        """Execute a predefined static attack."""
        if attack_type == AttackType.RANSOMWARE:
            target_nodes = random.sample(
                list(self.graph.nodes()),
                k=int(self.num_nodes * 0.8)
            )
            for node_id in target_nodes:
                node = self.graph.nodes[node_id]['data']
                if random.random() < (node.vulnerability / 10):
                    node.state = NodeState.COMPROMISED

        elif attack_type == AttackType.APT:
            for node_id in self.graph.nodes():
                node = self.graph.nodes[node_id]['data']
                if node.is_high_value():
                    if random.random() < (node.vulnerability / 8):
                        node.state = NodeState.COMPROMISED

        elif attack_type == AttackType.DDOS:
            target_nodes = random.sample(
                list(self.graph.nodes()),
                k=self.num_nodes // 3
            )
            for node_id in target_nodes:
                node = self.graph.nodes[node_id]['data']
                node.state = NodeState.OVERLOADED

        elif attack_type == AttackType.ZERODAY:
            target = random.choice(list(self.graph.nodes()))
            node = self.graph.nodes[target]['data']
            node.state = NodeState.COMPROMISED

    def _calculate_rewards(self, action_dict: Dict) -> Dict[str, float]:
        """Calculate rewards for each agent."""
        rewards = {}

        for agent_id, action in action_dict.items():
            node = self.graph.nodes[agent_id]['data']
            reward = 0

            # Reward for detecting actual threats
            if action == NetworkAction.SCAN.value:
                neighbors = list(self.graph.neighbors(agent_id))
                compromised_neighbors = sum(
                    1 for n in neighbors
                    if self.graph.nodes[n]['data'].state == NodeState.COMPROMISED
                )
                reward += compromised_neighbors * self.config['reward_weights']['detection']

            # Penalty for unnecessary isolation
            if action == NetworkAction.ISOLATE.value:
                if node.state != NodeState.COMPROMISED:
                    reward += self.config['reward_weights']['false_positive']

            # Global reward component based on network health
            reward += (
                    self.calculate_network_health() *
                    self.config['reward_weights']['network_health']
            )

            rewards[agent_id] = reward

        return rewards

    def calculate_network_health(self) -> float:
        """Calculate overall network health score."""
        total_nodes = self.num_nodes
        compromised = self._count_compromised_nodes()
        isolated = sum(
            1 for node in self.graph.nodes()
            if self.graph.nodes[node]['data'].state == NodeState.ISOLATED
        )

        health_score = 1.0 - (compromised + isolated * 0.5) / total_nodes
        return max(0.0, health_score)

    def _count_compromised_nodes(self) -> int:
        """Count number of compromised nodes."""
        return sum(
            1 for node in self.graph.nodes()
            if self.graph.nodes[node]['data'].state == NodeState.COMPROMISED
        )

    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Generate observations for all agents."""
        observations = {}
        for node_id in self.graph.nodes():
            observations[node_id] = self._get_node_observation(node_id)
        return observations

    def _get_node_observation(self, node_id: str) -> np.ndarray:
        """Generate observation for a single node."""
        node = self.graph.nodes[node_id]['data']
        obs = np.zeros(self.obs_dim, dtype=np.float32)

        # 1. Node state one-hot encoding (5 values)
        state_idx = list(NodeState).index(node.state)
        obs[0:5] = 0
        obs[state_idx] = 1

        # 2. Vulnerability score (normalized)
        obs[5] = node.vulnerability / 10.0

        # 3. Node value (normalized)
        obs[6] = node.value / 10.0

        # 4. Neighbor states summary (5 values)
        neighbors = list(self.graph.neighbors(node_id))
        neighbor_states = [self.graph.nodes[n]['data'].state for n in neighbors]
        for state in NodeState:
            state_idx = list(NodeState).index(state) + 7
            obs[state_idx] = sum(1 for s in neighbor_states if s == state) / max(len(neighbors), 1)

        # 5. Network health indicators (3 values)
        obs[12] = self.calculate_network_health()
        obs[13] = self._count_compromised_nodes() / self.num_nodes
        obs[14] = sum(1 for n in self.graph.nodes() if
                      self.graph.nodes[n]['data'].state == NodeState.ISOLATED) / self.num_nodes

        logger.debug(f"Generated observation shape for node {node_id}: {obs.shape}")
        return obs

    def _select_attack_targets(self, target_type: str) -> List[str]:
        """Select nodes to target based on attack type."""
        if target_type == 'vulnerable':
            return [
                node_id for node_id in self.graph.nodes()
                if self.graph.nodes[node_id]['data'].vulnerability > 7
            ]
        elif target_type == 'high_value':
            return [
                node_id for node_id in self.graph.nodes()
                if self.graph.nodes[node_id]['data'].value > 7
            ]
        elif target_type == 'random':
            num_targets = max(1, self.num_nodes // 4)
            return random.sample(list(self.graph.nodes()), num_targets)
        return []
