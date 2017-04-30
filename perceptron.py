import matplotlib.pyplot as plt
import numpy as np

# TODO: visualization part


class LinearEquation:
    # structure that keeps parameters of linear equation

    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.x0 = - b / a


def generate_sample(lin_eq, range_start=-1000, range_end=1000):
    # generates sample point with x, y coordinates in specified range
    # additionally returns proper classification for a point: 1 if it is
    # above line specified by lin_eq parameter and -1 if it lies below
    cord_x = np.random.uniform(range_start, range_end)
    cord_y = np.random.uniform(range_start, range_end)
    result = 1 if cord_y >= lin_eq.a * cord_x + lin_eq.b else -1
    return np.array([cord_x, cord_y, result])


def draw_plot(orig_a, orig_b, pred_a, pred_b):
    # for future visualization purposes
    plt.plot([-10, 10], [-10 * orig_a + orig_b, 10 * orig_a + orig_b], c='red')
    plt.plot([-10, 10], [-10 * pred_a + pred_b, 10 * pred_a + pred_b], c='blue')
    plt.title('Visualization test')
    plt.show()


class SimplePerceptron:
    # class that implements perceptron model
    # attributes:
    # eta - learning rate, should be real in range <0;1>
    # weights - weights values for each perceptron inputs
    # bias - 'weight 0', -threshold

    def __init__(self, n_inputs=2, eta=0.01):
        self._weights = np.zeros(n_inputs).reshape((1, n_inputs))
        self._bias = 0
        self._eta = eta

    def predict(self, inputs):
        # predict value for provided inputs (that should be passed as np array)
        if self._weights.dot(inputs) + self._bias >= 0:
            return 1
        else:
            return -1

    def fit(self, X_train, y_train):
        # trains perceptron from provided training data, X_train is a matrix of observations,
        # while y_train is vector of correct result for each observation
        for idx in range(X_train.shape[0]):
            y_pred = self.predict(X_train[idx, :])
            self._weights += self._eta * X_train * (y_train[idx] - y_pred)
            self._bias += self._eta * (y_train[idx] - y_pred)


def benchmark(lin_eq, predictor, n_tests=1000):
    # tests trained predictor, lin_eq that predictor was trained against needs to
    # be provided, tests quantity specified by n_tests
    errors = 0
    for i in range(n_tests):
        sample = generate_sample(lin_eq)
        if sample[2] != predictor.predict(sample[:2]):
            errors += 1

    return errors, errors / 1000


def main():
    # program solves easy classification problem: if point is above line defined by provided
    # linear equation or below?

    # actual linear equation coefficients
    lin_eq = LinearEquation(a=2.3, b=-0.4)

    # number of training iterations
    iterations = 10 ** 5

    sp = SimplePerceptron(eta=0.01)

    for i in range(iterations):
        sample = generate_sample(lin_eq)
        X_train = np.array([sample[0:2]]).reshape((1, 2))
        y_train = np.array([sample[2]]).reshape((1, 1))
        sp.fit(X_train, y_train)
        # if i % 10000 == 0:
        # draw_plot(a, b, sn._weights[0], sn._bias)

    errors, error_rate = benchmark(lin_eq, sp)
    print('Total errors after training:', errors)
    print('Error rate after training:', error_rate)
    print('Accuracy: {}%'.format((1 - error_rate) * 100))

main()
