import logging

from config import LOG_FILE

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

file_handler = logging.FileHandler(
    LOG_FILE,
    mode="a"
)

file_handler.setFormatter(
    formatter
)

console_handler = logging.StreamHandler()

console_handler.setFormatter(
    formatter
)

logger.addHandler(file_handler)
logger.addHandler(console_handler)