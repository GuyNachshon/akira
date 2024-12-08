from collections import defaultdict
from typing import Dict, List, Any
import numpy as np

from src.environment.state import NodeState


def calculate_network_metrics(state: Dict) -> Dict[str, float]:
    """Calculate various network metrics."""
    total_nodes = len(state)
    compromised = sum(1 for node in state.values() if node.state == NodeState.COMPROMISED)
    isolated = sum(1 for node in state.values() if node.state == NodeState.ISOLATED)

    metrics = {
        'network_health': 1.0 - (compromised + isolated * 0.5) / total_nodes,
        'compromised_ratio': compromised / total_nodes,
        'isolated_ratio': isolated / total_nodes,
        'average_vulnerability': sum(node.vulnerability for node in state.values()) / total_nodes,
        'high_value_nodes_compromised': sum(
            1 for node in state.values()
            if node.is_high_value() and node.state == NodeState.COMPROMISED
        ) / max(sum(1 for node in state.values() if node.is_high_value()), 1)
    }

    return metrics


def analyze_agent_performance(history: List[Dict]) -> Dict[str, Any]:
    """Analyze agent performance metrics."""
    performance = {
        'average_reward': np.mean([h['reward'] for h in history]),
        'reward_trend': [h['reward'] for h in history],
        'action_distribution': defaultdict(int),
        'successful_defenses': 0,
        'false_positives': 0
    }

    for entry in history:
        performance['action_distribution'][entry['action']] += 1
        if entry.get('success', False):
            performance['successful_defenses'] += 1
        if entry.get('false_positive', False):
            performance['false_positives'] += 1

    # Calculate success rate
    performance['success_rate'] = (
        performance['successful_defenses'] / len(history)
        if history else 0
    )

    # Calculate efficiency
    performance['efficiency'] = (
        (performance['successful_defenses'] - performance['false_positives']) /
        len(history) if history else 0
    )

    return performance
