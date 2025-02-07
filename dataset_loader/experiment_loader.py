import os
import cv2
from mpl_toolkits.axes_grid1.axes_rgb import RGBAxes


class ExperimentLoader:
    """Loads images from a specific experiment folder."""

    def __init__(self, experiment_path):
        self.experiment_path = experiment_path

    def image_generator(self):
        """Generator that yields images and their filenames."""

        def generator():
            for file_name in os.listdir(self.experiment_path):
                file_path = os.path.join(self.experiment_path, file_name)
                if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    # TODO: Switch between grayscale and RGB
                    # mage = cv2.imread(file_path)
                    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    if image is not None:
                        yield image, file_name

        return generator  # Returns generator function
