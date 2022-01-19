import numpy as np
from tensorflow.keras.datasets import mnist
import tools
import os


class Preprocessing:
    def __init__(self, case, specific_classes=None, retrieve_from_file=False):

        self.features_train = None
        self.features_test = None
        self.features_train_flat = None
        self.features_test_flat = None
        self.labels_train = None
        self.labels_test = None
        self.case = case
        self.specific_classes = specific_classes

        self.retrieve_from_file = retrieve_from_file
        self.dataset_directory = f"dataset_{self.case}"
        if self.dataset_directory not in os.listdir():
            os.mkdir(self.dataset_directory)

        print(f"-- Preprocess --")

    def import_dataset(self):

        filepath = self.generate_filepath()
        convert = True

        if self.retrieve_from_file:
            self.features_train = tools.load_data(filepath[0], convert)
            self.labels_train = tools.load_data(filepath[1], convert)
            self.features_test = tools.load_data(filepath[2], convert)
            self.labels_test = tools.load_data(filepath[3], convert)

        else:
            (self.features_train, self.labels_train), (self.features_test, self.labels_test) = mnist.load_data()
            if self.case == f"fisher":
                # Saving now
                self.save_data(filepath)

    def additional_processing(self):

        # Selecting the data
        print(f"Selecting data for the classes: {self.specific_classes}")
        (self.features_train, self.labels_train), (self.features_test, self.labels_test) = mnist.load_data()
        self.features_train = self.features_train[np.isin(self.labels_train, [self.specific_classes[0], self.specific_classes[1]]), :, :]
        self.labels_train = 1 * (self.labels_train[np.isin(self.labels_train, [self.specific_classes[0], self.specific_classes[1]])] > self.specific_classes[0])
        self.features_test = self.features_test[np.isin(self.labels_test, [self.specific_classes[0], self.specific_classes[1]]), :, :]
        self.labels_test = 1 * (self.labels_test[np.isin(self.labels_test, [self.specific_classes[0], self.specific_classes[1]])] > self.specific_classes[0])

        # Reshaping
        self.features_train_flat = self.features_train.reshape(self.features_train.shape[0], -1)
        self.features_test_flat = self.features_test.reshape(self.features_test.shape[0], -1)

        # Standardize the data
        self.features_train_flat = self.features_train_flat / 255
        self.features_test_flat = self.features_test_flat / 255

        # Saving now
        filepath = self.generate_filepath()
        self.save_data(filepath)

    def obtaining_data(self):
        
        print(f"Case: {self.case}")
        print(f"features_train: {self.features_train.shape}")
        print(f"labels_train: {self.labels_train.shape}")
        print(f"features_test: {self.features_test.shape}")
        print(f"labels_test: {self.labels_test.shape}")

        if self.features_train_flat is not None:
            print(f"Train: {str(self.features_train_flat.shape[0])} images and {str(self.features_train_flat.shape[1])}"
                  f" neurons")
            print(f"Test: {str(self.features_test_flat.shape[0])} images and {str(self.features_test_flat.shape[1])}"
                  f" neurons")

        return self.features_train, self.features_train_flat, self.features_test, self.features_test_flat, \
            self.labels_train, self.labels_test

    def save_data(self, filepath, convert=True):
        tools.save_data(self.features_train, filepath[0], convert)
        tools.save_data(self.labels_train, filepath[1], convert)
        tools.save_data(self.features_test, filepath[2], convert)
        tools.save_data(self.labels_test, filepath[3], convert)

    def generate_filepath(self):

        filenames = [f"features_train", "labels_train", "features_test", "labels_test"]

        if self.specific_classes is not None:
            add = [str(c) for c in self.specific_classes]
            add = f"_".join(add)
            filenames = [f + f"_{add}" for f in filenames]

        filepath = [os.path.join(self.dataset_directory, filenames[f]) for f in range(len(filenames))]

        return filepath
