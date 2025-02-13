import torch
from torchgen.executorch.api.et_cpp import returntype_type
from torchvision import transforms
from PIL import Image
from dataset_loader.image_database import ImageDatabase



# Dataset Loader
class UltrasoundDataset(torch.utils.data.Dataset):
    # Image transformations (grayscale, resize, normalization)
    TRANSFORM = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1,1]
    ])
    def __init__(self, root_dir,config_file,label,experiment_name=None, transform=None):
        self.root_dir = root_dir
        self.config_file = config_file
        self.label = label
        self.experiment_name = experiment_name
        self.db = ImageDatabase(self.root_dir, self.config_file)
        self.transform = UltrasoundDataset.TRANSFORM if not transform else transform
        self.image_paths = self._load_images_paths()

    def __len__(self):
        return len(self.get_image_paths())

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("L")  # Convert to grayscale
        if self.transform:
            image = self.transform(image)
        return image

    def _load_images_paths(self):
        if not self.experiment_name:
            return self.db.all_images_paths(self.label)
        return self.db.get_experiment_image_paths(self.label, self.experiment_name)

    def get_image_paths(self):
        return self.image_paths
    
    def get_db_generator_tf(self, batch_size=8):
        if not self.experiment_name:
            return self.db.all_images_generator_tf(self.label, batch_size)
        return self.db.get_experiment_generator_tf(self.label, self.experiment_name, batch_size)
    
    def get_db_generator(self):
        if not self.experiment_name:
            return self.db.all_images_generator(self.label)()
        return self.db.get_experiment_generator(self.label, self.experiment_name)()
    