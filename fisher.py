import numpy as np
from skimage import measure
import os
import tools
from scipy.stats import norm

import matplotlib.pyplot as plt


class FisherDiscriminant:

    def __init__(self):
        self.properties = ['area', 'perimeter', 'eccentricity', 'minor_axis_length', 'euler_number',
                           'extent', 'orientation', 'solidity', 'convex_area']
        self.classes = [number for number in np.arange(10)]
        self.dataset_directory = f'dataset_fisher'

    @staticmethod
    def average_image(vector_image):
        return np.mean(vector_image, axis=0)

    @staticmethod
    def original_region_proprieties(image_threshold):

        image_threshold_proprieties = measure.regionprops(image_threshold)
        number_threshold = len(image_threshold_proprieties)
        print(f"{number_threshold} region(s) found")

        area = image_threshold_proprieties[0].area
        print(f"Area: {area} pixels")

        perimeter = image_threshold_proprieties[0].perimeter
        print(f"Perimeter: {perimeter} pixels")

        centroid = image_threshold_proprieties[0].centroid
        print(f"centroid: {centroid}")

        eccentricity = image_threshold_proprieties[0].eccentricity
        print(f"Eccentricity: {eccentricity}")

        minor_axis = image_threshold_proprieties[0].minor_axis_length
        print(f"minor_axis: {minor_axis}")

    def obtain_proprieties_all_number(self, images, labels, threshold=60, retrieving_file=None):

        if retrieving_file is not None:
            filename = f'{retrieving_file}'
            filepath_dataset_property = os.path.join(self.dataset_directory, filename)

            dataset = tools.load_data(filepath_dataset_property, convert=False)

        else:
            dataset = []
            for propriety in self.properties:

                filename = f'dataset_property_{propriety}'
                filepath_dataset_property = os.path.join(self.dataset_directory, filename)

                data_by_property_all_number_all_image = []
                for number in self.classes:

                    images_class = images[labels == number, :, :]
                    images_threshold = 1 * (images_class > threshold)

                    data_by_property_by_number_all_image = []

                    for image_threshold in images_threshold:

                        image_threshold_proprieties = measure.regionprops(image_threshold)
                        propriety_value = getattr(image_threshold_proprieties[0], propriety)

                        try:
                            data_by_property_by_number_all_image.append(float(propriety_value))
                        except TypeError:
                            print(f"conversion")
                            data_by_property_by_number_all_image.append(propriety_value)

                    data_by_property_all_number_all_image.append(data_by_property_by_number_all_image)

                dataset.append(data_by_property_all_number_all_image)
                # Saving data
                tools.save_data(data_by_property_all_number_all_image, filepath_dataset_property, convert=False)

        return dataset

    def obtain_properties_by_class(self, images, labels, class_name, threshold=60, retrieving_file=None):

        dataset = []

        if retrieving_file is not None:
            filename = f'{retrieving_file}'
            filepath_dataset_property = os.path.join(self.dataset_directory, filename)
            dataset = tools.load_data(filepath_dataset_property, convert=False)

        else:

            filename = f'dataset_properties_class{class_name}'
            filepath_dataset_property = os.path.join(self.dataset_directory, filename)

            for propriety in self.properties:

                images_class = images[labels == class_name, :, :]
                images_threshold = 1 * (images_class > threshold)

                data_by_property_by_number_all_image = []
                for image_threshold in images_threshold:

                    image_threshold_proprieties = measure.regionprops(image_threshold)
                    propriety_value = getattr(image_threshold_proprieties[0], propriety)

                    try:
                        data_by_property_by_number_all_image.append(float(propriety_value))
                    except TypeError:
                        data_by_property_by_number_all_image.append(propriety_value)

                dataset.append(data_by_property_by_number_all_image)

            # Saving data
            tools.save_data(dataset, filepath_dataset_property, convert=False)

        return np.array(dataset)

    @staticmethod
    def computing_omegas(features_1, features_2):

        # Regions first and second class
        mean_features_1, mean_features_2 = np.mean(features_1, axis=1), np.mean(features_2, axis=1)

        cov_features_1, cov_features_2 = np.cov(features_1), np.cov(features_2)
        total_covariance = cov_features_1 + cov_features_2

        if len(mean_features_1) > 1:
            omega_1_2 = np.inner(np.linalg.inv(total_covariance), (mean_features_1 - mean_features_2))
        else:
            omega_1_2 = np.inner(1 / total_covariance, (mean_features_1 - mean_features_2))

        return omega_1_2

    def computing_discriminants(self, datasets):

        dataset_properties_class_1 = datasets[0]
        dataset_properties_class_2 = datasets[1]
        dataset_properties_class_3 = datasets[2]
        dataset_properties_class_4 = datasets[3]

        # Omegas
        omega_1_2 = self.computing_omegas(dataset_properties_class_1, dataset_properties_class_2)
        omega_3_4 = self.computing_omegas(dataset_properties_class_3, dataset_properties_class_4)

        # Discriminants
        discriminant_1 = [float(np.dot(omega_1_2, feat)) for feat in dataset_properties_class_1.T]
        discriminant_2 = [float(np.dot(omega_1_2, feat)) for feat in dataset_properties_class_2.T]
        discriminant_3 = [float(np.dot(omega_3_4, feat)) for feat in dataset_properties_class_3.T]
        discriminant_4 = [float(np.dot(omega_3_4, feat)) for feat in dataset_properties_class_4.T]

        discriminants = [discriminant_1, discriminant_2, discriminant_3, discriminant_4]

        omegas = [omega_1_2, omega_3_4]

        return discriminants, omegas

    @staticmethod
    def computing_thresholds(ax, gaussian_curves):

        # Computing thresholds

        x_min_1, x_max_1 = ax[0].get_xlim()
        x_min_2, x_max_2 = ax[1].get_xlim()

        x_axis_1 = np.linspace(x_min_1, x_max_1, 1000)
        x_axis_2 = np.linspace(x_min_2, x_max_2, 1000)

        threshold_1 = x_axis_1[np.argwhere(np.diff(np.sign(gaussian_curves[0] - gaussian_curves[1]))).flatten()]
        threshold_2 = x_axis_2[np.argwhere(np.diff(np.sign(gaussian_curves[2] - gaussian_curves[3]))).flatten()]

        return threshold_1, threshold_2

    @staticmethod
    def confusion_matrix(discriminants, misclassified):
        
        # Confusion matrix
        confusion_matrix_0_1 = np.zeros((2, 2))
        confusion_matrix_2_3 = np.zeros((2, 2))

        confusion_matrix_0_1[0, 0] = (len(discriminants[0])) - len(misclassified[0])
        confusion_matrix_0_1[1, 1] = (len(discriminants[1])) - len(misclassified[1])
        confusion_matrix_0_1[0, 1] = len(misclassified[0])
        confusion_matrix_0_1[1, 0] = len(misclassified[1])

        confusion_matrix_2_3[0, 0] = (len(discriminants[2])) - len(misclassified[2])
        confusion_matrix_2_3[1, 1] = (len(discriminants[3])) - len(misclassified[3])
        confusion_matrix_2_3[0, 1] = len(misclassified[2])
        confusion_matrix_2_3[1, 0] = len(misclassified[3])

        return confusion_matrix_0_1, confusion_matrix_2_3

    @staticmethod
    def accuracies(confusion_matrix):
        sensitivity = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[1, 0])
        specificity = confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[0, 1])
        return 0.5 * (specificity + sensitivity)
