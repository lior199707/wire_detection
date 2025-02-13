import os
import cv2
from mpl_toolkits.axes_grid1.axes_rgb import RGBAxes
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np


class ExperimentLoader:
    """Loads images from a specific experiment folder."""

    def __init__(self, experiment_path):
        self.experiment_path = experiment_path
        self.image_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

    def image_generator(self):
        """Generator that yields images and their filenames."""

        def generator():
            for file_name in os.listdir(self.experiment_path):
                file_path = os.path.join(self.experiment_path, file_name)
                if self._is_image(file_path):
                    # TODO: Switch between grayscale and RGB
                    # mage = cv2.imread(file_path)
                    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    if image is not None:
                        yield image, file_name

        return generator  # Returns generator function
    
    # def image_generator_tf(self, batch_size=8):
    #     """Generator that yields images and their filenames using tensorflow."""
    #     def generator():
    #         i = 0
    #         batch_images = []
    #         for file_name in os.listdir(self.experiment_path):
    #                 file_path = os.path.join(self.experiment_path, file_name)
    #                 if self._is_image(file_path):
    #                     img = load_img(file_path, color_mode="grayscale", target_size=(512, 1024))
    #                     img_array = img_to_array(img) / 255.0  # Normalize
    #                     batch_images.append(img_array)
    #                     if len(batch_images) == batch_size:
    #                         yield np.array(batch_images)
    #                         batch_images = []
    #                         i += 1
    #                     if i == 2:
    #                         break
    #     return generator
    
    def image_generator_tf(self, batch_size=8):
        """Generator that yields images and their filenames using tensorflow."""
        def generator():
            for file_name in os.listdir(self.experiment_path):
                    file_path = os.path.join(self.experiment_path, file_name)
                    if self._is_image(file_path):
                        img = load_img(file_path, color_mode="grayscale", target_size=(512, 1024))
                        img_array = img_to_array(img) / 255.0  # Normalize
                        yield np.array(img_array)

        return generator
                    

    def get_image_paths(self):
        return [os.path.join(self.experiment_path, file_name) for file_name in os.listdir(self.experiment_path) if self._is_image(file_name)]

    def _is_image(self, file_path):
        return file_path.lower().endswith(self.image_formats)
