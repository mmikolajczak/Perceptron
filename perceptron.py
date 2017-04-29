import matplotlib.pyplot as plt
import numpy as np
import cv2

# actual linear equation coefficients
a = 0.6
b = 1.2

# eta as a learning rate should be provided
# when to stop learning? - let's say after 10k of iterations
# as it is for presentation the idea is to present it dynamically using matplot (maybe with help in handling
# keyboard using opcv.


points_below_line = []
points_above_line = []


class SimplePerceptron:

    def __init__(self, n_inputs=2):
        self._weights = np.ones(n_inputs)
        self._bias = 0

    def predict(self, inputs):
        if len(inputs) != len(self._weights):
            raise Exception('Wrong size of input vector')

        if (self._weights * inputs).sum() + self._bias > 0:
            return 1
        else:
            return 0

    def fit(self, X_train, y_train):
        pass


sn = SimplePerceptron()
y_pred = sn.predict([1, 2])
print(y_pred)