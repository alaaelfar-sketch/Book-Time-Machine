"""Central logging utility for consistent console output across the project."""

import logging
import sys


def get_logger(name: str = "book_time_machine", level: int = logging.INFO) -> logging.Logger:
    """
    Create and configure a logger instance.

    Parameters
    ----------
    name : str
        Logger name (usually module name).
    level : int
        Logging level (INFO, DEBUG, etc.)

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Prevent duplicate handlers when re-importing modules
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)

        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s │ %(name)s │ %(message)s",
            datefmt="%H:%M:%S",
        )

        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(level)
    return logger