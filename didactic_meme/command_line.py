import argparse
import shutil
from typing import Callable
from .config import Config
import logging
logger = logging.getLogger(__name__)


def create(ModelConfig: Config) -> None:
    parser = argparse.ArgumentParser(description='Create model config')
    parser.add_argument('model_dir', default='model')
    args = parser.parse_args()
    config = ModelConfig()
    config.save(args.model_dir)

def train(ModelConfig: Config, train_func: Callable[[Config], None]) -> None:
    parser = argparse.ArgumentParser(description='Create model config')
    parser.add_argument('model_dir', default='model')
    args = parser.parse_args()
    config = ModelConfig(args.model_dir)
    shutil.rmtree(config.model_dir)
    config.save()

    train_func(config)
