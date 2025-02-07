import os
import json
from .experiment_loader import ExperimentLoader


class ImageDatabase:
    """Manages datasets and dynamically loads them from a JSON config."""

    def __init__(self, root_path, config_path):
        self.root_path = root_path
        self.config_path = config_path
        self.dataset_config = self._load_config()

    def _load_config(self):
        """Loads dataset structure from JSON."""
        script_dir = os.path.dirname(os.path.abspath(__file__))  # Current script's directory
        config_path = os.path.abspath(os.path.join(script_dir, "..", "data", "dataset_config.json"))
        with open(config_path, 'r') as file:
            return json.load(file)

    def get_experiment_loader(self, label, experiment_name):
        """Returns an ExperimentLoader for a given label and experiment."""
        if label not in self.dataset_config:
            raise ValueError(f"Invalid label '{label}'. Available labels: {list(self.dataset_config.keys())}")

        class_info = self.dataset_config[label]
        experiment_dir = class_info["experiments"].get(experiment_name)

        if experiment_dir is None:
            raise FileNotFoundError(f"Experiment '{experiment_name}' not found under '{label}' dataset.")

        experiment_path = os.path.join(self.root_path, class_info["path"], experiment_dir)

        if not os.path.exists(experiment_path):
            raise FileNotFoundError(f"Directory '{experiment_path}' does not exist.")

        return ExperimentLoader(experiment_path)

    def all_images_generator(self, label):
        """Generator that yields all images from all experiments of a given label."""
        if label not in self.dataset_config:
            raise ValueError(f"Invalid label '{label}'. Available labels: {list(self.dataset_config.keys())}")

        class_info = self.dataset_config[label]

        def generator():
            for experiment_name, experiment_dir in class_info["experiments"].items():
                experiment_path = os.path.join(self.root_path, class_info["path"], experiment_dir)
                if os.path.isdir(experiment_path):
                    exp_loader = ExperimentLoader(experiment_path)
                    yield from exp_loader.image_generator()()

        return generator
