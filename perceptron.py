import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2

# actual linear equation coefficients
a = 2.3
b = -0.4
iterations = 10 ** 5#int(10e5)

# eta as a learning rate should be provided
# when to stop learning? - let's say after 10k of iterations
# as it is for presentation the idea is to present it dynamically using matplot (maybe with help in handling
# keyboard using opcv.


points_below_line = []
points_above_line = []

#def test_linear_function(x):
#    return x *


def generate_sample(a, b):
    cord_x = np.random.uniform(-1000, 1000)
    cord_y = np.random.uniform(-1000, 1000)
    result = 1 if cord_y > a * cord_x + b else -1
    return np.array([cord_x, cord_y, result])

def generate_sample2(a, b):
    cord_x = np.random.uniform(-100000, 100000)
    cord_y = np.random.uniform(-100000, 100000)
    result = 1 if cord_y > a * cord_x + b else -1
    return np.array([cord_x, cord_y, result])


def draw_plot(orig_a, orig_b, pred_a, pred_b):
    plt.plot([-10, 10], [-10 * orig_a + orig_b, 10 * orig_a + orig_b], c='red')
    plt.plot([-10, 10], [-10 * pred_a + pred_b, 10 * pred_a + pred_b], c='blue')
    plt.title('Visualization test')
    plt.show()


class SimplePerceptron:

    def __init__(self, n_inputs=2, eta=0.01):
        self._weights = np.zeros(n_inputs).reshape((1, n_inputs))
        self._bias = 0
        self._eta = eta

    def predict(self, inputs):
        #if len(inputs) != len(self._weights):
        #    raise Exception('Wrong size of input vector')

        #if (self._weights * inputs).sum() + self._bias >= 0:
        if (self._weights.dot(inputs)).sum() + self._bias >= 0:
            return 1
        else:
            return -1

    def fit(self, X_train, y_train): #X_train need to be matrix
        for i in range(X_train.shape[0]):
            y_pred = self.predict(X_train[i, :])
            self._weights += self._eta * X_train * (y_train[i] - y_pred)
            self._bias += self._eta * (y_train[i] - y_pred)

# after training should predict 1 if point is above line and 0 if it is below

sn = SimplePerceptron(eta=0.01)

# do it one by one instead of training all in one time for clean visualization sake
for i in range(iterations):
    sample = generate_sample(a, b)
    X_train = np.array([sample[0:2]]).reshape((1, 2))
    y_train = np.array([sample[2]]).reshape((1, 1))
    sn.fit(X_train, y_train)
    #if i % 10000 == 0:
        #draw_plot(a, b, sn._weights[0], sn._bias)

print('Params after training:')
print(sn._weights, sn._bias)


def benchmark():
    errors = 0
    for i in range(1000):
        sample = generate_sample2(a, b) #some tests

        if sample[2] != sn.predict(sample[:2]):
            errors += 1
    print(errors)

benchmark()
