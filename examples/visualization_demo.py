import random
import time

from src.environment.network_env import NetworkEnvironment
from src.environment.state import NodeState
from src.visualization.network_visualizer import NetworkVisualizer
from src.visualization.training_visualizer import TrainingVisualizer


def main():
    # Initialize environment and visualizers
    env = NetworkEnvironment(num_nodes=20)
    network_vis = NetworkVisualizer(env)
    training_vis = TrainingVisualizer(save_dir="visualization_demo")

    # Simulate some episodes for demonstration
    for episode in range(10):
        network_health = random.uniform(0.5, 1.0)
        compromised_nodes = random.randint(0, 5)
        agent_rewards = {
            str(i): random.uniform(-1, 1)
            for i in range(env.num_nodes)
        }
        attack_success = random.uniform(0, 0.5)

        # Update metrics
        training_vis.update_metrics(
            network_health=network_health,
            compromised_nodes=compromised_nodes,
            agent_rewards=agent_rewards,
            attack_success=attack_success
        )

        # Simulate some network changes
        for _ in range(3):
            # Randomly compromise or restore nodes
            node_id = random.randint(0, env.num_nodes - 1)
            node = env.graph.nodes[str(node_id)]['data']
            node.state = random.choice(list(NodeState))

            # Create real-time visualization
            network_vis.visualize_network_state()
            time.sleep(1)  # Pause to show changes

    # Generate final visualizations
    print("Generating training progress plot...")
    training_vis.plot_training_progress(save=True)

    print("Creating interactive dashboard...")
    dashboard = training_vis.create_interactive_dashboard()
    dashboard.write_html("visualization_demo/interactive_dashboard.html")

    print("Saving metrics...")
    training_vis.save_metrics()

    print("Visualization demo completed. Check the visualization_demo directory for outputs.")


def show_visualization_features():
    """Demonstrate different visualization features"""
    env = NetworkEnvironment(num_nodes=10)
    network_vis = NetworkVisualizer(env)

    print("\nVisualization Features Demo:")

    # 1. Basic Network State
    print("\n1. Basic Network State")
    network_vis.visualize_network_state()

    # 2. Attack Visualization
    print("\n2. Simulating and Visualizing Attack")
    # Simulate attack
    random_node = random.randint(0, env.num_nodes - 1)
    env.graph.nodes[str(random_node)]['data'].state = NodeState.COMPROMISED
    network_vis.visualize_network_state(highlight_compromised=True)

    # 3. Defense Action Visualization
    print("\n3. Simulating and Visualizing Defense")
    another_node = random.randint(0, env.num_nodes - 1)
    env.graph.nodes[str(another_node)]['data'].state = NodeState.ISOLATED
    network_vis.visualize_network_state(highlight_defense=True)

    # 4. Node Details View
    print("\n4. Showing Node Details")
    network_vis.show_node_details(str(random_node))

    # 5. Time-series View
    print("\n5. Generating Time-series Analysis")
    network_vis.show_time_series(
        metrics=['network_health', 'compromised_nodes'],
        last_n_steps=50
    )


if __name__ == "__main__":
    main()
    print("\nRunning additional visualization features demo...")
    show_visualization_features()