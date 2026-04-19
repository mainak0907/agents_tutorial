"""
utils/logger.py
────────────────
Consistent logging setup for the whole project.
"""

import logging
import sys
from config import settings


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, settings.LOG_LEVEL, logging.INFO))
    return logger
