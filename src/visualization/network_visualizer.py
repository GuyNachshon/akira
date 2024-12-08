from collections import defaultdict
from typing import Dict, List

import numpy as np
import plotly.graph_objects as go
import networkx as nx
from matplotlib import pyplot as plt

from ..environment.network_env import NetworkEnvironment
from ..environment.state import NodeState


class NetworkVisualizer:
    """Provides visualization tools for network state and training progress."""

    def __init__(self, env: NetworkEnvironment):
        self.env = env
        self.history = {
            'network_health': [],
            'compromised_nodes': [],
            'agent_rewards': defaultdict(list),
            'attack_success_rate': []
        }

    def update_history(
            self,
            network_health: float,
            compromised_nodes: int,
            agent_rewards: Dict[str, float],
            attack_success: float
    ):
        """Update historical data."""
        self.history['network_health'].append(network_health)
        self.history['compromised_nodes'].append(compromised_nodes)
        for agent_id, reward in agent_rewards.items():
            self.history['agent_rewards'][agent_id].append(reward)
        self.history['attack_success_rate'].append(attack_success)

    def plot_training_progress(self):
        """Plot training progress metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Network Health
        axes[0, 0].plot(self.history['network_health'])
        axes[0, 0].set_title('Network Health Over Time')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Health Score')

        # Compromised Nodes
        axes[0, 1].plot(self.history['compromised_nodes'], color='red')
        axes[0, 1].set_title('Compromised Nodes Over Time')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Number of Nodes')

        # Agent Rewards
        mean_rewards = [
            np.mean([rewards[i] for rewards in self.history['agent_rewards'].values()])
            for i in range(len(next(iter(self.history['agent_rewards'].values()))))
        ]
        axes[1, 0].plot(mean_rewards, color='green')
        axes[1, 0].set_title('Average Agent Reward Over Time')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Reward')

        # Attack Success Rate
        axes[1, 1].plot(self.history['attack_success_rate'], color='purple')
        axes[1, 1].set_title('Attack Success Rate Over Time')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Success Rate')

        plt.tight_layout()
        return fig

    def visualize_network_state(self, highlight_compromised: bool = False, highlight_defense: bool = False) -> go.Figure:
        """Create interactive network visualization."""
        G = self.env.graph
        pos = nx.spring_layout(G)

        # Create edges trace
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        # Create nodes trace
        node_x, node_y = [], []
        node_colors = []
        node_text = []

        for node_id in G.nodes():
            x, y = pos[node_id]
            node_x.append(x)
            node_y.append(y)

            node = G.nodes[node_id]['data']
            color = self._get_node_color(
                node.state,
                highlight_compromised=highlight_compromised,
                highlight_defense=highlight_defense
            )
            node_colors.append(color)

            node_text.append(self._get_node_text(node_id))

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                size=10,
                color=node_colors,
                line_width=2
            )
        )

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40)
            )
        )

        return fig

    def show_node_details(self, node_id: str) -> None:
        """Display detailed information about a specific node."""
        node = self.env.graph.nodes[node_id]['data']
        neighbors = list(self.env.graph.neighbors(node_id))

        print(f"\nNode {node_id} Details:")
        print(f"State: {node.state.value}")
        print(f"Vulnerability: {node.vulnerability}")
        print(f"Value: {node.value}")
        print(f"Last Scan: {node.last_scan_time}")
        print(f"Connected to nodes: {', '.join(neighbors)}")

    def show_time_series(self, metrics: List[str], last_n_steps: int = 50) -> go.Figure:
        """Show time series of specified metrics."""
        fig = go.Figure()

        for metric in metrics:
            if metric in self.history:
                data = self.history[metric][-last_n_steps:]
                fig.add_trace(go.Scatter(
                    y=data,
                    name=metric,
                    mode='lines+markers'
                ))

        fig.update_layout(
            title='Network Metrics Over Time',
            xaxis_title='Time Step',
            yaxis_title='Value'
        )

        return fig

    def _get_node_color(self, state: NodeState, highlight_compromised: bool = False, highlight_defense: bool = False) -> str:
        """Get color for node based on its state and highlighting options."""
        colors = {
            NodeState.SAFE: '#00ff00',
            NodeState.COMPROMISED: '#ff0000',
            NodeState.OVERLOADED: '#ffa500',
            NodeState.ISOLATED: '#808080',
            NodeState.SCANNING: '#0000ff'
        }

        base_color = colors.get(state, '#000000')

        if highlight_compromised and state == NodeState.COMPROMISED:
            return '#ff0000'  # Bright red
        if highlight_defense and state in [NodeState.ISOLATED, NodeState.SCANNING]:
            return '#00ff00'  # Bright green

        return base_color

    def _get_node_text(self, node_id: str) -> str:
        """Generate hover text for node."""
        node = self.env.graph.nodes[node_id]['data']
        return (f"Node {node_id}<br>"
                f"State: {node.state.value}<br>"
                f"Vulnerability: {node.vulnerability}<br>"
                f"Value: {node.value}")

    @staticmethod
    def _get_node_color(state: NodeState) -> str:
        """Get color for node based on its state."""
        colors = {
            NodeState.SAFE: '#00ff00',  # Green
            NodeState.COMPROMISED: '#ff0000',  # Red
            NodeState.OVERLOADED: '#ffa500',  # Orange
            NodeState.ISOLATED: '#808080',  # Gray
            NodeState.SCANNING: '#0000ff'  # Blue
        }
        return colors.get(state, '#000000')
