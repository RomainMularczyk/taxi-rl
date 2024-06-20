import logging
from functools import lru_cache


LOG_FORMAT = '%(asctime)s [%(levelname)s] %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
LOG_LEVEL = 'INFO'


@lru_cache
def get_logger():
    logging.basicConfig(
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        # filename=LOG_FILE_NAME,
        # filemode=LOG_FILE_MODE
    )
    logger = logging.getLogger()
    logger.setLevel(LOG_LEVEL or logging.INFO)
    logger.info(f"Logger initialized with level {logging.getLevelName(logger.level)}.")
    return logger
