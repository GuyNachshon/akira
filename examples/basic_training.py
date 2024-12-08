import yaml
import logging

from src.environment.network_env import NetworkEnvironment
from src.training.trainer import TrainingManager
from src.visualization.training_visualizer import TrainingVisualizer

logging.basicConfig(level=logging.INFO)


def main():
    # Load configuration
    with open("../config/training_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    logging.info(f"Loaded configuration: {config}")

    # Initialize components
    env = NetworkEnvironment(
        num_nodes=config['environment']['num_nodes'],
        false_positive_rate=config['environment']['false_positive_rate'],
        compromise_detection_rate=config['environment']['compromise_detection_rate']
    )

    logging.info("Initialized environment")

    visualizer = TrainingVisualizer(save_dir="training_output")
    logging.info("Initialized visualizer")

    trainer = TrainingManager(
        env=env,
        config=config,
        visualizer=visualizer
    )

    logging.info("Initialized training manager")

    # Run training
    trainer.train()

    # Generate and save visualizations
    visualizer.plot_training_progress()
    visualizer.save_metrics()


if __name__ == "__main__":
    main()
