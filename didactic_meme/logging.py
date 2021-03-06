import os
import logging


def setup_loggers(config, level=logging.INFO):
    if logging.root:
        del logging.root.handlers[:]

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter('%(message)s'))
    root_logger.addHandler(console)

    file_handler = logging.FileHandler(os.path.join(config.model_dir, 'train.log'), mode='w', encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-7s: %(message)-60s -- %(filename)s:%(lineno)d (%(funcName)s)', datefmt='%Y-%m-%d %H:%M'))
    root_logger.addHandler(file_handler)
