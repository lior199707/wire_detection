import os
from dataset_loader.image_database import ImageDatabase

if __name__ == '__main__':
    # Define paths
    root_folder = os.path.join("data")
    config_file = "dataset_config.json"

    # Initialize database with the JSON config
    image_db = ImageDatabase(root_folder, config_file)

    # Load a specific experiment
    exp_gen = image_db.get_experiment_generator("wire", "experiment15")

    for img, name in exp_gen():
        print(f"Processing {name}, Shape: {img.shape}")

    # Get all images of 'not_wire'
    wire_gen = image_db.all_images_generator("wire")
    for img, name in wire_gen():
        print(f"Processing {name} from not_wire dataset")
