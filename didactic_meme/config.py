import os
import sys
import json
import random
from .adict import adict


def make_config_class(**default_values):
    class ModelConfig(Config):
        def __init__(self, model_dir=None):
            super().__init__(model_dir, default_values)

    return ModelConfig


class Config(adict):
    def __init__(self, model_dir=None, default_values={}):
        super().__init__(**default_values)
        if 'seed' not in self:
            self['seed'] = random.randrange(sys.maxsize)

        self.model_dir = model_dir
        if model_dir is not None:
            self.load(model_dir)

    def config_path(self):
        return os.path.join(self.model_dir, 'config.json')

    def save(self, model_dir=None):
        if model_dir is not None:
            self.model_dir = model_dir

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        with open(self.config_path(), 'w') as file:
            json.dump(self, file, indent=4)

    def load(self, model_dir):
        self.model_dir = model_dir

        with open(self.config_path(), 'r') as file:
            new_config = json.load(file)
            self.deep_update(new_config)

    def get_checkpoint_dir(self):
        return os.path.join(self.model_dir, 'checkpoints')

    def get_checkpoint_path(self, epoch):
        n_epochs_chars = len(str(self.n_epochs))
        checkpoint_filename = f'epoch{epoch:0{n_epochs_chars}d}.pth'
        return os.path.join(self.get_checkpoint_dir(), checkpoint_filename)
