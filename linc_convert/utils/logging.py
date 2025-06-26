import logging

logger = logging.getLogger()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.captureWarnings(True)


def add_file_handler(log_file_path=None):
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)


# TODO: add file handler for each platform. prob ref dandi-ci. prevent logging error in unit test
setup_logging()
# add_file_handler("/scratch/converter.log")
