import matplotlib.pyplot as plt
import os

BLOCK = False
FIGURES_DIRECTORY = f"figures_neural"

if FIGURES_DIRECTORY not in os.listdir():
    os.mkdir(FIGURES_DIRECTORY)


def plot_cost(costs, weights, title=f"Evolution of the cost and final weights"):

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(13, 5))
    fig.suptitle(title)

    ax[0].plot(range(0, len(costs)), costs)
    ax[0].set_xlabel('Iterations')
    ax[0].set_ylabel('Cost')
    ax[0].set_xticks(range(0, len(costs), 100))
    ax[0].set_title('Evolution of the cost')

    ax[1].imshow(weights.reshape(28, 28))
    ax[1].set_title('Template')

    plt.show(block=BLOCK)

    filepath_figure = os.path.join(FIGURES_DIRECTORY, title)
    plt.savefig(filepath_figure)
    plt.close()

    return fig, ax
