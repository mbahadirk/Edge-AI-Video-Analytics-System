import logging
import json
import sys


def setup_logger(name="perf_monitor"):
    """Configures a logger to output raw JSON lines to stdout."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if re-imported
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        # We use a custom formatter to ensure pure JSON output
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def log_json(logger, data: dict):
    """Helper to dump a dictionary as a JSON string."""
    try:
        json_str = json.dumps(data)
        logger.info(json_str)
    except TypeError:
        logger.error(f"Failed to serialize log data: {data}")