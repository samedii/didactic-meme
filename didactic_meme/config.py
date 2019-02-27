import os
import json
from .adict import adict


def make_config_class(**default_values):
    class ModelConfig(Config):
        def default_values(self):
            return default_values

    return ModelConfig


class Config(adict):
    def __init__(self, model_dir=None, **kwargs):
        super().__init__(**self.default_values())
        self.model_dir = model_dir
        if model_dir is not None:
            self.load(model_dir)

    def default_values(self):
        raise NotImplementedError()

    def config_path(self):
        return os.path.join(self.model_dir, 'config.json')

    def save(self, model_dir=None):
        if model_dir is not None:
            self.model_dir = model_dir

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        with open(self.config_path(), 'w') as file:
            json.dump(self.to_dict(), file, indent=4)

    def load(self, model_dir):
        self.model_dir = model_dir

        with open(self.config_path(), 'r') as file:
            new_config = json.load(file)
            self.deep_update(new_config)
