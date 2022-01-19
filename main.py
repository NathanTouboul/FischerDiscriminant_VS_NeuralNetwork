import preprocess
import fisher
import neural

import plot_fish
import plot_neural


def preprocessing(case, classes, retrieve_from_file=False):

    # Preprocessing data
    preprocessing_class = preprocess.Preprocessing(case, classes, retrieve_from_file=retrieve_from_file)

    # Importation mnist dataset_fisher
    preprocessing_class.import_dataset()

    if preprocessing_class.case == f"fisher":

        # Obtaining datasets
        feats_train, _, features_test, _, label_train, labels_test = preprocessing_class.obtaining_data()

        return feats_train, features_test, label_train, labels_test

    elif preprocessing_class.case == f"neural_network":
        # Selecting, reshaping, standardizing dataset
        preprocessing_class.additional_processing()
        # Obtaining datasets
        feats_train, features_train_flat, features_test, features_test_flat, label_train, labels_test = \
            preprocessing_class.obtaining_data()

        return feats_train, features_train_flat, features_test, features_test_flat, label_train, labels_test


def fisher_main_intro(classes, pre=False):

    # PART A - Fisher Discriminant
    fisher_inst = fisher.FisherDiscriminant()
    case = f"fisher"

    # Collecting feature pattern for two specific class

    # Dataset for each class
    feats_train, _, label_train, _ = preprocessing(case, None, retrieve_from_file=pre)
    features_train_class_1 = feats_train[label_train == classes[0], :, :]
    features_train_class_2 = feats_train[label_train == classes[1], :, :]
    features_train_class_3 = feats_train[label_train == classes[2], :, :]
    features_train_class_4 = feats_train[label_train == classes[3], :, :]

    # Obtaining the average image for each class
    average_image_1 = fisher_inst.average_image(features_train_class_1)
    average_image_2 = fisher_inst.average_image(features_train_class_2)
    average_image_3 = fisher_inst.average_image(features_train_class_3)
    average_image_4 = fisher_inst.average_image(features_train_class_4)

    # Obtaining the threshold image of the average image
    thresh = 60
    threshold_image_1 = 1 * (average_image_1 > thresh)
    threshold_image_2 = 1 * (average_image_2 > thresh)
    threshold_image_3 = 1 * (average_image_3 > thresh)
    threshold_image_4 = 1 * (average_image_4 > thresh)

    # Plotting Average and threshold
    images = [average_image_1, threshold_image_1, average_image_2, threshold_image_2]
    plot_fish.plotting_image(images, title=f"Average and Threshold images for classes {classes[:2]}")

    images = [average_image_3, threshold_image_3, average_image_4, threshold_image_4]
    plot_fish.plotting_image(images, title=f"Average and Threshold images for classes {classes[2:]}")

    # Geometric properties of our images for the classes
    print("---------------------------------------------------")
    print(f"Properties for the handwritten number {classes[0]}")
    fisher_inst.original_region_proprieties(threshold_image_1)
    print(f"Properties for the handwritten number {classes[1]}")
    fisher_inst.original_region_proprieties(threshold_image_2)
    print(f"Properties for the handwritten number {classes[2]}")
    fisher_inst.original_region_proprieties(threshold_image_3)
    print(f"Properties for the handwritten number {classes[3]}")
    fisher_inst.original_region_proprieties(threshold_image_4)

    return fisher_inst, feats_train, label_train, thresh


def fisher_main_hist_props(fisher_inst, classes, feats_train, label_train, thresh):

    # Searching for relevant properties over all numbers
    dataset_properties = fisher_inst.obtain_proprieties_all_number(feats_train, label_train, thresh)

    for p, property_name in enumerate(fisher_inst.properties):
        plot_fish.plotting_histograms_properties(dataset_properties[p], classes, title=property_name)


def fisher_main_discriminant(fisher_inst, classes, features, labels):

    # Computing the Fisher Linear Discriminant for the following classes

    class_0, class_1 = classes[0], classes[1]
    class_2, class_3 = classes[2], classes[3]

    dataset_properties_class_0 = fisher_inst.obtain_properties_by_class(features, labels, class_0)
    dataset_properties_class_1 = fisher_inst.obtain_properties_by_class(features, labels, class_1)
    dataset_properties_class_5 = fisher_inst.obtain_properties_by_class(features, labels, class_2)
    dataset_properties_class_6 = fisher_inst.obtain_properties_by_class(features, labels, class_3)

    dataset_discriminants = [dataset_properties_class_0, dataset_properties_class_1, dataset_properties_class_5,
                             dataset_properties_class_6]

    discriminants, omegas = fisher_inst.computing_discriminants(dataset_discriminants)

    gaussian_curves, thresholds = plot_fish.plot_discriminants_gaussian(discriminants, specific_classes, fisher_inst,
                                                                        title=f"Histograms Discriminants")

    threshold_0_1, threshold_2_3 = thresholds
    print(f"thresholds_{class_0}_{class_1}: {threshold_0_1}")
    print(f"thresholds_{class_2}_{class_3}: {threshold_2_3}")

    # Mis classification Percentages
    misclassified_0 = [i for i in range(len(discriminants[0])) if discriminants[0][i] < threshold_0_1]
    misclassified_1 = [i for i in range(len(discriminants[1])) if discriminants[1][i] > threshold_0_1]
    misclassified_2 = [i for i in range(len(discriminants[2])) if discriminants[2][i] < threshold_2_3]
    misclassified_3 = [i for i in range(len(discriminants[3])) if discriminants[3][i] > threshold_2_3]

    misclassified = [misclassified_0, misclassified_1, misclassified_2, misclassified_3]

    print(f"Percentage class {class_0} misclassified: {100 * len(misclassified_0) / len(discriminants[0])}")
    print(f"Percentage class {class_1}  misclassified: {100 * len(misclassified_1) / len(discriminants[1])}")
    print(f"Percentage class {class_2}  misclassified: {100 * len(misclassified_2) / len(discriminants[2])}")
    print(f"Percentage class {class_3}  misclassified: {100 * len(misclassified_3) / len(discriminants[3])}")

    # Confusion Matrix and precision

    confusion_matrix_0_1, confusion_matrix_2_3 = fisher_inst.confusion_matrix(discriminants, misclassified)

    accuracy_0_1 = fisher_inst.accuracies(confusion_matrix_0_1)
    accuracy_2_3 = fisher_inst.accuracies(confusion_matrix_2_3)

    print(f"Accuracies {class_0}, {class_1}: {accuracy_0_1 * 100} %")
    print(f"Accuracies {class_2}, {class_3}: {accuracy_2_3 * 100} %")

    return accuracy_0_1, accuracy_2_3


def main_neural_network(classes=None):

    # PART A - Neural Network
    if classes is None:
        classes = [0, 1]

    neural_network = neural.NeuralNetwork()
    case = f"neural_network"

    # Loading mnist dataset: if set to True, loading from mnist
    loading_from_file = False

    # Complete Dataset for 0 and 1 labels
    _, features_train_flat, _, features_test_flat, label_train, label_test = \
        preprocessing(case, classes, retrieve_from_file=loading_from_file)

    # Training the model
    # Initialize parameters with zeros
    weights, bias = neural_network.initialize_weights(features_train_flat.shape[1])

    print(f"features_train_flat.shape: {features_train_flat.shape}")
    print(f"labels_train.shape: {label_train.shape}")

    parameters, gradients, costs = neural_network.gradient_descent(weights, bias, features_train_flat,
                                                                   label_train)
    print(f"Costs: {costs}")

    weights = parameters["weights"]
    bias = parameters["bias"]

    predictions_training = neural_network.predict(weights, bias, features_train_flat)
    predictions_testing = neural_network.predict(weights, bias, features_test_flat)

    print(f"Precision of the training: {neural_network.compute_precision(predictions_training, label_train)}")
    print(f"Precision of the testing: {neural_network.compute_precision(predictions_testing, label_test)}")

    plot_neural.plot_cost(costs, weights, title=f"Training for the classes {classes}")


if __name__ == '__main__':

    # Task B1 - Neural Network class 0 - 1
    print(f"TASK B1")
    specific_classes = [0, 1]
    main_neural_network(specific_classes)

    # Task B2 - Neural Network class 5 - 6
    print(f"TASK B2")
    specific_classes = [5, 6]
    main_neural_network(specific_classes)

    pass

    # Task A1 - Average and Threshold Images class 0 - 1 and class 5 - 6
    print(f"Preprocessing and Overview")
    specific_classes = [0, 1, 5, 6]
    load_from_file = True
    fisher_class, features_train, labels_train, threshold = fisher_main_intro(specific_classes, load_from_file)

    # Task A2 - All relevant properties 0 - 1 and class 5 - 6
    print(f"Studying relevant properties")
    fisher_main_hist_props(fisher_class, specific_classes, features_train, labels_train, threshold)

    # Task A3 - Computing Discriminants
    print(f"Computing Discriminants")
    fisher_main_discriminant(fisher_class, specific_classes, features_train, labels_train)
