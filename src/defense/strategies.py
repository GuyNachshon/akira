from collections import defaultdict
from typing import List, Tuple

from src.environment.network_env import NetworkEnvironment
from src.environment.state import NodeState


class AdvancedDefenseStrategies:
    def __init__(self, env: NetworkEnvironment):
        self.env = env
        self.strategy_costs = {
            'isolate': 2.0,
            'patch': 1.5,
            'scan': 1.0,
            'restore': 3.0
        }
        self.resource_pool = 100.0
        self.strategy_cooldowns = defaultdict(float)

    def get_available_strategies(
            self,
            node_id: str,
            current_time: float
    ) -> List[str]:
        """Get list of available strategies considering cooldowns and resources."""
        available = []
        for strategy, cost in self.strategy_costs.items():
            if (
                    current_time > self.strategy_cooldowns[f"{node_id}_{strategy}"] and
                    cost <= self.resource_pool
            ):
                available.append(strategy)
        return available

    def execute_strategy(
            self,
            node_id: str,
            strategy: str,
            current_time: float
    ) -> Tuple[bool, float]:
        """Execute a defense strategy and return success and cost."""
        if strategy not in self.get_available_strategies(node_id, current_time):
            return False, 0.0

        cost = self.strategy_costs[strategy]
        self.resource_pool -= cost
        self.strategy_cooldowns[f"{node_id}_{strategy}"] = (
                current_time + self._get_cooldown(strategy)
        )

        success = self._execute_specific_strategy(node_id, strategy)
        return success, cost

    def _get_cooldown(self, strategy: str) -> float:
        """Get cooldown time for a strategy."""
        cooldowns = {
            'isolate': 10.0,
            'patch': 5.0,
            'scan': 2.0,
            'restore': 15.0
        }
        return cooldowns.get(strategy, 5.0)

    def _execute_specific_strategy(self, node_id: str, strategy: str) -> bool:
        """Execute a specific defense strategy."""
        node = self.env.graph.nodes[node_id]['data']

        if strategy == 'isolate':
            if node.state != NodeState.ISOLATED:
                node.state = NodeState.ISOLATED
                return True

        elif strategy == 'patch':
            if node.vulnerability > 1:
                node.vulnerability = max(1, node.vulnerability - 2)
                return True

        elif strategy == 'scan':
            neighbors = list(self.env.graph.neighbors(node_id))
            found_threat = False
            for neighbor in neighbors:
                neighbor_node = self.env.graph.nodes[neighbor]['data']
                if neighbor_node.state == NodeState.COMPROMISED:
                    found_threat = True
            return found_threat

        elif strategy == 'restore':
            if node.state in [NodeState.COMPROMISED, NodeState.OVERLOADED]:
                node.state = NodeState.SAFE
                return True

        return False
