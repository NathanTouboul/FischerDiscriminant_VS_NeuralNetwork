import json
import numpy as np


def save_data(data, filename, convert=False):

    if convert:
        if type(data) == np.ndarray:
            data = data.tolist()

    with open(filename, 'w') as f:
        json.dump(data, f)

    print(f"Saving the data as {filename}")


def load_data(filename, convert=False):

    with open(filename, 'r') as f:
        data = json.load(f)

    if convert:
        if type(data) == list:
            data = np.array(data)

    print(f"Loading the data {filename}")
    return data

