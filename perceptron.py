import matplotlib.pyplot as plt
import numpy as np


class LinearEquation:
    # structure that keeps parameters of linear equation

    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.x0 = - b / a

    def count(self, x):
        # return value of function for provided x
        return self.a * x + self.b


def generate_sample(lin_eq, range_start=-50, range_end=50):
    # generates sample point with x, y coordinates in specified range
    # additionally returns proper classification for a point: 1 if it is
    # above line specified by lin_eq parameter and -1 if it lies below
    cord_x = np.random.uniform(range_start, range_end)
    cord_y = np.random.uniform(range_start, range_end)
    result = 1 if cord_y >= lin_eq.a * cord_x + lin_eq.b else -1
    return np.array([cord_x, cord_y, result])


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

    def predict(self, X_test):
        # predict value for provided inputs (that should be passed as np array)
        predictions = []
        for idx in range(X_test.shape[0]):
            if self._weights.dot(X_test[idx, :]) + self._bias >= 0:
                predictions.append(1)
            else:
                predictions.append(-1)

        return np.array(predictions).reshape((-1, 1))

    def fit(self, X_train, y_train):
        # trains perceptron from provided training data, X_train is a matrix of observations,
        # while y_train is vector of correct result for each observation
        for idx in range(X_train.shape[0]):
            y_pred = self.predict(X_train[idx, :].reshape((1, -1)))
            self._weights += self._eta * X_train[idx, :] * (y_train[idx] - y_pred)
            self._bias += self._eta * (y_train[idx] - y_pred)


def benchmark(lin_eq, predictor, n_tests=1000):
    # tests trained predictor, lin_eq that predictor was trained against needs to
    # be provided, tests quantity specified by n_tests

    correctly_classified = []
    wrong_classified = []

    for i in range(n_tests):
        sample = generate_sample(lin_eq)
        if sample[-1] != predictor.predict(sample[:-1].reshape((1, -1))):
            wrong_classified.append(sample[:-1])
        else:
            correctly_classified.append(sample[:-1])

    return np.array(correctly_classified), np.array(wrong_classified)


class TestStats:
    # structure to keep test results and some statistics about them

    def __init__(self, errors, total):
        self.errors = errors
        self.total = total
        self.error_rate = errors / total
        self.accuracy = (1 - self.error_rate) * 100


def count_statistics(correctly_classified, wrongly_classified):
    # return TestStats object from data about correctly and wrongly classified points
    return TestStats(wrongly_classified.shape[0], correctly_classified.shape[0] + wrongly_classified.shape[0])


def visualize_training(lin_eq, predictor, n_iterations, samples_per_iteration):
    # On each iteration it visualizes results of prediction for a few samples, at the end of iteration
    # it predictor uses this data again to learn from it
    # We move on to next iteration by simple mouse button click

    for i in range(n_iterations):

        # generating set of samples
        for j in range(samples_per_iteration):
            if j == 0:
                samples = generate_sample(lin_eq)
                continue
            new_sample = generate_sample(lin_eq)
            samples = np.vstack([samples, new_sample])

        # predict and count divide good/bad predictions
        predictions = predictor.predict(samples[:, :-1])
        correctly_classified = []
        wrongly_classified = []

        for j in range(predictions.shape[0]):
            if predictions[j, -1] != samples[j, -1]:
                wrongly_classified.append(samples[j, :-1])
            else:
                correctly_classified.append(samples[j, :-1])

        correctly_classified = np.array(correctly_classified)
        wrongly_classified = np.array(wrongly_classified)

        # visualization itself
        if correctly_classified.shape[0] != 0:
            plt.scatter(correctly_classified[:, 0], correctly_classified[:, 1], c='green')
        if wrongly_classified.shape[0] != 0:
            plt.scatter(wrongly_classified[:, 0], wrongly_classified[:, 1], c='red')
        plt.plot([-50, 50], [lin_eq.count(-50), lin_eq.count(50)])
        plt.title('Iteration nr: {}, Sample size: {}'.format(i + 1, samples_per_iteration))
        plt.xlabel('X')
        # plt.xlim(-55, 55)
        plt.ylabel('Y')
        plt.pause(0.05)
        plt.waitforbuttonpress()
        plt.clf()

        # predictor learns from data it predicted earlier
        predictor.fit(samples[:, :-1], samples[:, -1])


def main():
    # program solves easy classification problem: if point is above line defined by provided
    # linear equation or below?

    # actual linear equation coefficients
    lin_eq = LinearEquation(a=2.3, b=-0.4)

    # number of training iterations
    iterations = 20

    sp = SimplePerceptron(eta=0.01)

    visualize_training(lin_eq, sp, iterations, samples_per_iteration=100)

    # Benchmark after training
    correctly_classified, wrongly_classified = benchmark(lin_eq, sp)
    test_stats = count_statistics(correctly_classified, wrongly_classified)

    print('Total errors after training:', test_stats.errors)
    print('Error rate after training:', test_stats.error_rate)
    print('Accuracy: {}%'.format(test_stats.accuracy))

main()
