import logging
import platform
import sys

LOGGER_NAME = 'crowd_detector'
WINDOWS, LINUX, MAC = [
    platform.system() == s for s in ['Windows', 'Linux', 'Darwin']
    ]


def setup_logger(name: str, verbose=True) -> logging.Logger:
    """
    Set up logging with custom name and verbosity.

    This function configures logging by setting the  logging level, formatter ands verbosity.
    Args:
        name (str): Name of the logger.
        verbose (bool): Flag to set logging level to INFO or ERROR.

    Returns:
        (logging.Logger): Configured logger object.

    Examples:
        >>> LOGGER = setup_logger(name=__name__, verbose=True)
        >>> LOGGER.info("text")
    """
    level = logging.INFO if verbose else logging.ERROR
    formatter = logging.Formatter(
        fmt='[%(asctime)s: %(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


LOGGER = setup_logger(LOGGER_NAME, verbose=True)
