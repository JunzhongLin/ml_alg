# Linear regression module

import numpy as np


class LinearRegression:
    def __init__(self, output_std: bool):
        ''' Linear regression model using maximum likely-hood method
         y = X @ w, where X is the design matrix, w is the parameter vector
         t ~ N (t|X @ w, var), where t is the label, var is the variance from noise
        '''
        self.w = None
        self.var = None
        self.output_std = output_std

    def fit(self, train_X: np.ndarray, train_y: np.ndarray):
        """
        least square fitting against training data
        ------ Parameters:
        :param train_x: np.ndarray, shape ~ (n, m) n: number of data points; m: number of feature dimensions
        :param train_y: np.ndarray, shape ~ (n, 1) currently only support 1D output
        :return: None
        """

        self.w = np.linalg.pinv(train_X) @ train_y
        self.var = np.mean(np.square(train_X @ self.w - train_y))

    def predict(self, test_X):
        """
        Calculate the prediction given the input
        ------ Parameters:
        :param test_X: np.ndarray, shape ~ (n, m)
                       input for the new data point for the prediction
        :return: y: np.ndarray, shape ~ (n, 1)
                    predictions for each input data point
        """

        y = test_X @ self.w
        if self.output_std:
            y_std = np.sqrt(self.var) + np.zeros_like(y)
            return y, y_std

        return y

