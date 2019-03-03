import argparse
import shutil
from typing import Callable
from .config import Config
import logging
logger = logging.getLogger(__name__)


def create(ModelConfig: Config) -> None:
    parser = argparse.ArgumentParser(description='Create model config')
    parser.add_argument('model_dir')
    args = parser.parse_args()
    config = ModelConfig()
    config.save(args.model_dir)


def train(ModelConfig: Config, train_func: Callable[[Config], None]) -> None:
    parser = argparse.ArgumentParser(description='Train using model config')
    parser.add_argument('model_dir')
    args = parser.parse_args()
    config = ModelConfig(args.model_dir)
    shutil.rmtree(config.model_dir)
    config.save()

    train_func(config)

def visualize(ModelConfig: Config, visualize_func: Callable[[Config], None]) -> None:
    parser = argparse.ArgumentParser(description='Visualize model')
    parser.add_argument('model_dir')
    parser.add_argument('epoch', type=int, nargs='?')
    args = parser.parse_args()
    config = ModelConfig(args.model_dir)

    epoch = args.epoch
    if epoch is None:
        epoch = config.n_epochs

    model = ProblemModel(config)
    checkpoint = torch.load(config.get_checkpoint_path(epoch))
    model.load_state_dict(checkpoint['model'])

    visualize_func(model)
