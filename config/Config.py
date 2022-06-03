import json
from abc import abstractmethod


class Config(object):
    config = None

    @classmethod
    def read_file(cls, config_path):
        with open(config_path, 'r') as f:
            cls.config = json.load(f)
            return cls.config
