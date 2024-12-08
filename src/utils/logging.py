import logging
from typing import Any, Dict


def setup_logging():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def log_training_metrics(metrics: Dict[str, Any]):
    """Log training metrics."""
    logger = logging.getLogger(__name__)
    logger.info(f"Training metrics: {metrics}")


def log(level: int, message: str):
    """Log a message with the specified level."""
    logger = logging.getLogger(__name__)
    if level == logging.DEBUG:
        logger.debug(message)
    elif level == logging.INFO:
        logger.info(message)
    elif level == logging.WARNING:
        logger.warning(message)
    elif level == logging.ERROR:
        logger.error(message)
    elif level == logging.CRITICAL:
        logger.critical(message)
    else:
        raise ValueError(f"Invalid log level: {level}")