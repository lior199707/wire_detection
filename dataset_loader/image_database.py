import os
import json
from .experiment_loader import ExperimentLoader


class ImageDatabase:
    """Manages datasets and dynamically loads them from a JSON config."""
    EXPERIMENTS = "experiments"
    PATH = "path"


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


    def get_experiment_generator(self, label, experiment_name):
        """Returns an ExperimentLoader for a given label and experiment."""
        experiment_path = self._validate_data(label, experiment_name)

        if not os.path.exists(experiment_path):
            raise FileNotFoundError(f"Directory '{experiment_path}' does not exist.")

        return ExperimentLoader(experiment_path).image_generator()

    def get_experiment_image_paths(self, label, experiment_name):
        experiment_path = self._validate_data(label, experiment_name)

        if not os.path.exists(experiment_path):
            raise FileNotFoundError(f"Directory '{experiment_path}' does not exist.")

        return ExperimentLoader(experiment_path).get_image_paths()


    def _is_valid_label(self, label):
        return label in self.dataset_config


    def _is_valid_experiment(self, label_data, experiment_name):
        # return label_data["experiments"].get(experiment_name)
        return experiment_name in label_data[ImageDatabase.EXPERIMENTS]


    def _validate_data(self, label, experiment_name):
        if not self._is_valid_label(label):
            raise ValueError(f"Invalid label '{label}'. Available labels: {list(self.dataset_config.keys())}")
        label_data = self.dataset_config[label]

        if not self._is_valid_experiment(label_data, experiment_name):
            raise FileNotFoundError(f"Experiment '{experiment_name}' not found under '{label}' dataset. Available experiments {list(label_data[ImageDatabase.EXPERIMENTS])}")
        experiment_dir = label_data[ImageDatabase.EXPERIMENTS][experiment_name]

        # Returns the path to the experiment directory
        return os.path.join(self.root_path, label_data[ImageDatabase.PATH], experiment_dir)


    def all_images_generator(self, label):
        """Generator that yields all images from all experiments of a given label."""
        if self._is_valid_label(label):
            raise ValueError(f"Invalid label '{label}'. Available labels: {list(self.dataset_config.keys())}")

        label_data = self.dataset_config[label]

        def generator():
            for experiment_dir in label_data[ImageDatabase.EXPERIMENTS].values():
                experiment_path = os.path.join(self.root_path, label_data[ImageDatabase.PATH], experiment_dir)
                if os.path.isdir(experiment_path):
                    exp_loader = ExperimentLoader(experiment_path)
                    yield from exp_loader.image_generator()()

        return generator

    def all_images_paths(self, label):
        if not self._is_valid_label(label):
            raise ValueError(f"Invalid label '{label}'. Available labels: {list(self.dataset_config.keys())}")

        label_data = self.dataset_config[label]
        res = []
        for experiment_dir in label_data[ImageDatabase.EXPERIMENTS].values():
            experiment_path = os.path.join(self.root_path, label_data[ImageDatabase.PATH], experiment_dir)
            if os.path.isdir(experiment_path):
                res.append(ExperimentLoader(experiment_path).get_image_paths())
        return res

