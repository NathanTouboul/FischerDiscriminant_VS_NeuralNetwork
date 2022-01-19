import numpy as np


class NeuralNetwork:

    def __init__(self):
        self.learning_rate = 1
        self.number_iterations = 100
        self.decision_threshold = 0.5

    @staticmethod
    def sigmoid(x: float):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def initialize_weights(dimension):

        """
        This function creates a vector of zeros of shape (dim, 1) for weights and initializes biais to 0.

        Argument:
        dimension -- size of the weights vector we want (or number of parameters in this case)

        Returns:
        weights -- initialized vector of shape (dim, 1)
        bias -- initialized scalar (corresponds to the bias)
        """

        weights = np.zeros((dimension, 1))
        bias = 0.
        assert(weights.shape == (dimension, 1))
        assert(isinstance(bias, float) or isinstance(bias, int))

        return weights, bias

    def propagate(self, weights, bias, features, labels):
        """
        Implement the cost function and its gradient for the propagation explained in the assignment

        Arguments:
        weights -- weights, a numpy array of size (num_px * num_px, 1)
        bias -- bias, a scalar
        features -- data of size (number of examples, num_px * num_px)
        labels -- true "label" vector of size (1, number of examples)

        Return:
        cost -- negative log-likelihood cost for logistic regression
        dj_dw -- gradient of the loss with respect to weights, thus same shape as weights
        dj_db -- gradient of the loss with respect to biais, thus same shape as biais

        """
        number_images = features.shape[0]
        cost = 0.0
        dj_db = 0.
        dj_dw = np.zeros(weights.shape)

        # FORWARD PROPAGATION (FROM features TO COST)
        for i, image_vector in enumerate(features):

            z_i = np.inner(weights.T, image_vector) + bias

            y_hat_i = self.sigmoid(z_i)
            y_i = labels[i]

            # Loss Function
            loss_y_hat_i_y_i = - y_i * np.log(y_hat_i) - (1 - y_i) * np.log(1 - y_hat_i)

            # Cost computation
            cost += loss_y_hat_i_y_i

            # BACKWARD PROPAGATION (TO FIND GRAD)

            difference = y_hat_i - y_i
            dj_db += difference

            increment = image_vector * difference
            increment = np.expand_dims(increment, axis=1)

            dj_dw += increment

        dj_db /= number_images
        dj_dw /= number_images
        cost /= number_images

        assert(dj_dw.shape == weights.shape)

        cost = np.squeeze(cost)
        assert (cost.shape == ())
        grads = {"dj_dw": dj_dw,
                 "dj_db": dj_db}

        return grads, cost

    def gradient_descent(self, weights, bias, features, labels):
        """
        This function optimizes weights and bias by running a gradient descent algorithm

        Arguments:
        weights -- weights, a numpy array of size (num_px * num_px, 1)
        bias -- bias, a scalar
        features -- data of shape (num_px * num_px, number of examples)
        labels -- true "label" vector of shape (1, number of examples)
        num_iterations -- number of iteration of the optimization loop
        learning_rate -- learning rate of the gradient descent update rule

        Returns:
        params -- dictionary containing the weights weights and bias biais
        grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
        costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

        Tips:
        You basically need to write down two steps and iterate through them:
            1) Calculate the cost and the gradient for the current parameters. Use propagate().
            2) Update the parameters using gradient descent rule for weights and biais.
        """

        costs = []
        dj_dw = None
        dj_db = None

        for iteration in range(self.number_iterations):

            # Cost and gradient calculation

            grads, cost = self.propagate(weights, bias, features, labels)

            # Retrieve derivatives from grads
            dj_dw = grads["dj_dw"]
            dj_db = grads["dj_db"]

            # update rule
            weights = weights - self.learning_rate * dj_dw
            bias = bias - self.learning_rate * dj_db

            # Record the costs
            if iteration % 10 == 0:
                costs.append(cost)
                # Print the cost every 100 training examples
                print(f"Cost after iteration {iteration}: {cost}")

        params = {"weights": weights,
                  "bias": bias}

        grads = {"dj_dw": dj_dw,
                 "dj_db": dj_db}

        return params, grads, costs

    def predict(self, weights, bias, features):

        """
        Predict whether the label is 0 or 1 using learned logistic regression parameters (weights, bias)

        Arguments:
        weights -- weights, a numpy array of size (num_px * num_px, 1)
        bias -- bias, a scalar
        features -- data of size (num_px * num_px, number of examples)

        Returns:
        Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in features
        """

        number_examples = features.shape[0]
        weights = weights.reshape(features.shape[1], 1)

        # Compute vector "A" predicting the probabilities of the picture containing a 1

        probabilities = self.sigmoid(np.dot(weights.T, features.T) + bias)

        # Convert probabilities A[0,i] to actual predictions p[0,i]
        predictions = np.array([1 if probability > self.decision_threshold else 0 for probability in probabilities.T])

        predictions = np.expand_dims(predictions, axis=0)

        assert (predictions.shape == (1, number_examples))

        return predictions

    @staticmethod
    def compute_precision(predictions, labels):
        return 100 - np.mean(np.abs(predictions - labels)) * 100
