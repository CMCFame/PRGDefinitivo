import logging
import sys

def get_logger(name="progol", level=logging.INFO):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Reutiliza si ya existe

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(stream)
    return logger
