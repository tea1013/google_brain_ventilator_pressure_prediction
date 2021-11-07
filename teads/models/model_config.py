import os
from abc import ABC, abstractmethod
from typing import Dict


class ModelConfig(ABC):
    def __init__(self, save_dir: str, save_file_name: str, model_file_type: str) -> None:
        self.save_dir = save_dir
        self.save_file_name = save_file_name
        self.model_file_type = model_file_type

    @abstractmethod
    def model_params(self) -> Dict:
        pass

    @property
    def save_model_path(self) -> str:
        os.makedirs(f"./models/{self.save_dir}", exist_ok=True)
        return f"./models/{self.save_dir}/{self.save_file_name}.{self.model_file_type}"

    @property
    def save_params_path(self) -> str:
        os.makedirs(f"./params/{self.save_dir}", exist_ok=True)
        return f"./params/{self.save_dir}/{self.save_file_name}.pickle"
