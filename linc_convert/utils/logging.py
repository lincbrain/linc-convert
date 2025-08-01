"""Logging utility for the linc_convert application."""

import logging
from os import PathLike

logger = logging.getLogger()


def setup_logging() -> None:
    """Set up the logging configuration for the application."""
    logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.captureWarnings(True)


def add_file_handler(log_file_path: str | PathLike[str] = None) -> None:
    """Add a file handler to the logger to log messages to a file."""
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)


# TODO: add file handler for each platform. prob ref dandi-ci. prevent logging error
#  in unit test
setup_logging()
# add_file_handler("/scratch/converter.log")
