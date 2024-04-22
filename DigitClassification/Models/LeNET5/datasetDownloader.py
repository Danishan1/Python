import numpy as np  # linear algebra
import struct
from array import array
import time


#
# MNIST Data Loader Class
#
class MnistDataloader(object):

    def __init__(
        self,
        training_images_filepath,
        training_labels_filepath,
        test_images_filepath,
        test_labels_filepath,
    ):
        self.__training_images_filepath = training_images_filepath
        self.__training_labels_filepath = training_labels_filepath
        self.__test_images_filepath = test_images_filepath
        self.__test_labels_filepath = test_labels_filepath

    def __read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, "rb") as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(
                    "Magic number mismatch, expected 2049, got {}".format(magic)
                )
            labels = array("B", file.read())

        with open(images_filepath, "rb") as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(
                    "Magic number mismatch, expected 2051, got {}".format(magic)
                )
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols : (i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img
        return images, labels

    def load_data(self):
        """
        Load training and test data from image and label files.

        Reads image and label data from specified file paths for training and test datasets.

        Returns:
        - Tuple containing:
            - Tuple of training images (NumPy array of shape (num_samples, height, width)) and
                training labels (NumPy array of shape (num_samples,)) with data type np.uint8.
            - Tuple of test images (NumPy array of shape (num_samples, height, width)) and
                test labels (NumPy array of shape (num_samples,)) with data type np.uint8.

        Example:
        >>> loader = DataLoader()
        >>> (x_train, y_train), (x_test, y_test) = loader.load_data()

        """
        startTime = time.time()
        x_train, y_train = self.__read_images_labels(
            self.__training_images_filepath, self.__training_labels_filepath
        )
        x_test, y_test = self.__read_images_labels(
            self.__test_images_filepath, self.__test_labels_filepath
        )

        from helperFunction import printTime

        printTime(startTime, "Data Loaded from binary to List")
        return (x_train, y_train), (x_test, y_test)
