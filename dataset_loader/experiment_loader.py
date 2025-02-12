import os
import cv2
from mpl_toolkits.axes_grid1.axes_rgb import RGBAxes


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

    def get_image_paths(self):
        return [os.path.join(self.experiment_path, file_name) for file_name in os.listdir(self.experiment_path) if self._is_image(file_name)]

    def _is_image(self, file_path):
        return file_path.lower().endswith(self.image_formats)
