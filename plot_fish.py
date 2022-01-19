import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import norm

BLOCK = False
FIGURES_DIRECTORY = f"figures_fisher"

if FIGURES_DIRECTORY not in os.listdir():
    os.mkdir(FIGURES_DIRECTORY)


def plotting_image(data_image, title=f"title", n_rows=1, n_cols=None):

    if n_cols is None:
        n_cols = len(data_image)

    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols)
    fig.suptitle(title)
    for i, image in enumerate(data_image):
        ax[i].imshow(image)

    plt.show(block=BLOCK)

    filepath_figure = os.path.join(FIGURES_DIRECTORY, title)
    plt.savefig(filepath_figure)
    plt.close("all")

    return fig, ax


def plotting_histograms_properties(dataset, specific_classes, title=f"Histograms properties"):

    fig, ax = plt.subplots(1, 2)
    fig.suptitle(title)

    for n, value_propriety_all_image_by_number in enumerate(dataset):

        if n in specific_classes[:2]:
            ax[0].hist(value_propriety_all_image_by_number, bins=35, density=True, alpha=0.5)
        elif n in specific_classes[2:]:
            ax[1].hist(value_propriety_all_image_by_number, bins=35, density=True, alpha=0.5)

    legend_handles_0 = [f"Class {n}" for n in specific_classes[:2]]
    legend_handles_1 = [f"Class {n}" for n in specific_classes[2:]]

    ax[0].legend(legend_handles_0)
    ax[1].legend(legend_handles_1)

    plt.show(block=BLOCK)
    filepath_figure = os.path.join(FIGURES_DIRECTORY, title)
    plt.savefig(filepath_figure)
    plt.close("all")

    return fig, ax


def plot_discriminants_gaussian(discriminants, classes, fisher_inst, title=f"Histograms Discriminants"):
    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(18, 10)
    fig.suptitle(title)

    ax[0].hist(discriminants[0], bins=100, alpha=0.4, density=True)
    ax[0].hist(discriminants[1], bins=100, alpha=0.4, density=True)
    ax[0].legend([str(c) for c in classes[:2]])

    ax[1].hist(discriminants[2], bins=100, alpha=0.4, density=True)
    ax[1].hist(discriminants[3], bins=100, alpha=0.4, density=True)
    ax[1].legend([str(c) for c in classes[2:]])

    # plotting gaussian fits
    x_min_1, x_max_1 = ax[0].get_xlim()
    x_min_2, x_max_2 = ax[1].get_xlim()

    x_axis_1 = np.linspace(x_min_1, x_max_1, 1000)
    x_axis_2 = np.linspace(x_min_2, x_max_2, 1000)

    mean_1, std_1 = np.mean(discriminants[0]), np.std(discriminants[0])
    mean_2, std_2 = np.mean(discriminants[1]), np.std(discriminants[1])
    mean_3, std_3 = np.mean(discriminants[2]), np.std(discriminants[2])
    mean_4, std_4 = np.mean(discriminants[3]), np.std(discriminants[3])

    discriminant_normal_fit_1 = norm.pdf(x_axis_1, mean_1, std_1)
    discriminant_normal_fit_2 = norm.pdf(x_axis_1, mean_2, std_2)
    discriminant_normal_fit_3 = norm.pdf(x_axis_2, mean_3, std_3)
    discriminant_normal_fit_4 = norm.pdf(x_axis_2, mean_4, std_4)

    gaussian_curves = [discriminant_normal_fit_1, discriminant_normal_fit_2, discriminant_normal_fit_3,
                       discriminant_normal_fit_4]

    # Computing the thresholds
    thresholds = fisher_inst.computing_thresholds(ax, gaussian_curves)

    ax[0].axvline(x=thresholds[0], color='r', linestyle='--')
    ax[1].axvline(x=thresholds[1], color='r', linestyle='--')

    ax[0].plot(x_axis_1, gaussian_curves[0])
    ax[0].plot(x_axis_1, gaussian_curves[1])
    ax[1].plot(x_axis_2, gaussian_curves[2])
    ax[1].plot(x_axis_2, gaussian_curves[3])

    plt.show(block=BLOCK)
    filepath_figure = os.path.join(FIGURES_DIRECTORY, title)
    plt.savefig(filepath_figure)
    plt.close("all")

    return gaussian_curves, thresholds

