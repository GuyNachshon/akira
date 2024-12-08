import logging

from src.attacks.attack_patterns import AdvancedAttackPatterns
from src.environment.network_env import NetworkEnvironment

logging.basicConfig(level=logging.INFO)


def main():
    # Initialize environment
    env = NetworkEnvironment(num_nodes=20)

    # Create custom attack pattern
    class CustomAttackPattern(AdvancedAttackPatterns):
        def __init__(self):
            super().__init__()
            self.attack_chains['custom_chain'] = [
                ('initial_breach', 0.9),
                ('system_scan', 0.8),
                ('privilege_escalation', 0.7),
                ('data_theft', 0.6)
            ]

    # Execute custom attack
    attack_pattern = CustomAttackPattern()
    compromised_nodes = attack_pattern.execute_attack_chain(env, 'custom_chain')

    logging.info(f"Compromised nodes: {compromised_nodes}")
    logging.info(f"Network health: {env.calculate_network_health()}")


if __name__ == "__main__":
    main()