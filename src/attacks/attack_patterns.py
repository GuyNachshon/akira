import random
from typing import Set

from src.environment.network_env import NetworkEnvironment


class AdvancedAttackPatterns:
    """Implements sophisticated attack patterns using common APT techniques."""

    def __init__(self):
        self.attack_chains = {
            'data_theft': [
                ('reconnaissance', 0.9),
                ('initial_access', 0.7),
                ('privilege_escalation', 0.6),
                ('lateral_movement', 0.8),
                ('data_exfiltration', 0.7)
            ],
            'infrastructure_damage': [
                ('vulnerability_scan', 0.8),
                ('exploit_execution', 0.7),
                ('backdoor_installation', 0.6),
                ('system_sabotage', 0.9)
            ],
            'ransomware_advanced': [
                ('phishing_campaign', 0.8),
                ('dropper_execution', 0.7),
                ('credential_theft', 0.6),
                ('domain_admin_compromise', 0.5),
                ('encryption_deployment', 0.9)
            ]
        }

    def execute_attack_chain(self, env: NetworkEnvironment, chain_type: str) -> Set[str]:
        """Execute a complete attack chain and return compromised nodes."""
        compromised_nodes = set()
        chain = self.attack_chains.get(chain_type, [])

        for step, success_prob in chain:
            new_compromised = self._execute_attack_step(
                env, step, success_prob, compromised_nodes
            )
            compromised_nodes.update(new_compromised)

        return compromised_nodes

    def _execute_attack_step(
            self,
            env: NetworkEnvironment,
            step: str,
            base_prob: float,
            existing_compromised: Set[str]
    ) -> Set[str]:
        """Execute a single step in the attack chain."""
        newly_compromised = set()

        if step == 'reconnaissance':
            # Identify high-value targets
            for node_id in env.graph.nodes():
                node = env.graph.nodes[node_id]['data']
                if node.vulnerability > 8:
                    if random.random() < base_prob:
                        newly_compromised.add(node_id)

        elif step == 'lateral_movement':
            # Spread from compromised nodes to neighbors
            for node_id in existing_compromised:
                neighbors = list(env.graph.neighbors(node_id))
                for neighbor in neighbors:
                    if random.random() < base_prob:
                        newly_compromised.add(neighbor)

        elif step == 'privilege_escalation':
            # Increase impact on already compromised nodes
            for node_id in existing_compromised:
                node = env.graph.nodes[node_id]['data']
                if random.random() < base_prob:
                    node.vulnerability = min(10, node.vulnerability + 2)

        # Add more sophisticated patterns for other steps...

        return newly_compromised
