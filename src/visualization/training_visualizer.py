import numpy as np
from typing import Dict, List, Optional
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from collections import defaultdict

from matplotlib import pyplot as plt


class TrainingVisualizer:
    def __init__(self, save_dir: Optional[str] = None):
        self.save_dir = Path(save_dir) if save_dir else Path("training_plots")
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.history = defaultdict(list)
        self.metrics = {
            'network_health': [],
            'compromised_nodes': [],
            'agent_rewards': defaultdict(list),
            'attack_success_rate': []
        }

    def update_metrics(
            self,
            network_health: float,
            compromised_nodes: int,
            agent_rewards: Dict[str, float],
            attack_success: float
    ):
        """Update training metrics."""
        self.metrics['network_health'].append(network_health)
        self.metrics['compromised_nodes'].append(compromised_nodes)
        for agent_id, reward in agent_rewards.items():
            self.metrics['agent_rewards'][agent_id].append(reward)
        self.metrics['attack_success_rate'].append(attack_success)

    def plot_training_progress(self, save: bool = True) -> plt.Figure:
        """Create comprehensive training progress visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress Overview', size=16)

        # Network Health
        self._plot_network_health(axes[0, 0])

        # Compromised Nodes
        self._plot_compromised_nodes(axes[0, 1])

        # Agent Rewards
        self._plot_agent_rewards(axes[1, 0])

        # Attack Success Rate
        self._plot_attack_success(axes[1, 1])

        plt.tight_layout()

        if save:
            fig.savefig(self.save_dir / 'training_progress.png')

        return fig

    def create_interactive_dashboard(self) -> go.Figure:
        """Create interactive Plotly dashboard."""
        fig = go.Figure()

        # Add network health trace
        fig.add_trace(go.Scatter(
            y=self.metrics['network_health'],
            name='Network Health',
            mode='lines',
            line=dict(color='green')
        ))

        # Add compromised nodes trace
        fig.add_trace(go.Scatter(
            y=self.metrics['compromised_nodes'],
            name='Compromised Nodes',
            mode='lines',
            line=dict(color='red')
        ))

        # Add average agent reward trace
        avg_rewards = np.mean(
            [rewards for rewards in self.metrics['agent_rewards'].values()],
            axis=0
        )
        fig.add_trace(go.Scatter(
            y=avg_rewards,
            name='Average Agent Reward',
            mode='lines',
            line=dict(color='blue')
        ))

        fig.update_layout(
            title='Training Metrics Dashboard',
            xaxis_title='Episode',
            yaxis_title='Value',
            hovermode='x unified'
        )

        return fig

    def _plot_network_health(self, ax: plt.Axes):
        """Plot network health trend."""
        ax.plot(self.metrics['network_health'], color='green')
        ax.set_title('Network Health Over Time')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Health Score')
        ax.grid(True)

    def _plot_compromised_nodes(self, ax: plt.Axes):
        """Plot compromised nodes trend."""
        ax.plot(self.metrics['compromised_nodes'], color='red')
        ax.set_title('Compromised Nodes Over Time')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Number of Nodes')
        ax.grid(True)

    def _plot_agent_rewards(self, ax: plt.Axes):
        """Plot agent rewards."""
        mean_rewards = np.mean(
            [rewards for rewards in self.metrics['agent_rewards'].values()],
            axis=0
        )
        std_rewards = np.std(
            [rewards for rewards in self.metrics['agent_rewards'].values()],
            axis=0
        )

        ax.plot(mean_rewards, color='blue', label='Mean Reward')
        ax.fill_between(
            range(len(mean_rewards)),
            mean_rewards - std_rewards,
            mean_rewards + std_rewards,
            alpha=0.2,
            color='blue'
        )
        ax.set_title('Agent Rewards Over Time')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.grid(True)

    def _plot_attack_success(self, ax: plt.Axes):
        """Plot attack success rate."""
        ax.plot(self.metrics['attack_success_rate'], color='purple')
        ax.set_title('Attack Success Rate Over Time')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Success Rate')
        ax.grid(True)

    def save_metrics(self):
        """Save metrics to CSV files."""
        # Save network metrics
        pd.DataFrame({
            'network_health': self.metrics['network_health'],
            'compromised_nodes': self.metrics['compromised_nodes'],
            'attack_success_rate': self.metrics['attack_success_rate']
        }).to_csv(self.save_dir / 'network_metrics.csv')
        # Save agent rewards
        pd.DataFrame(self.metrics['agent_rewards']).to_csv(self.save_dir / 'agent_rewards.csv')
