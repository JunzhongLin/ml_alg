# Bayesian regression model with known precision

import numpy as np


class BayesianRegression:
    """
    Bayesian Regression model
    p(w|alpha) = N(w|0, alpha^(-1))  : Prior
    p(t|X, w, beta) = N(t| X@w, beta^(-1)): Likelihood
    p(w|t,X,w,beta,alpha) = N(w|m, S) : Posterior
        m = beta @ S @ X.T @ t
        S = alpha @ I + beta @ X.T @ X

    y = X@w : Linear Model

    Predictive distribution
    p(t|x,X,alpha,beta) = N(t|w.T@x, 1/beta + x.T@S@x)
    """
    def __init__(self, alpha: float = 1., beta: float = 1.):
        """
        Initialize the bayesian regression model

        ------Parameters
        :param alpha: float, optional, precision of prior for w (model parameter)
        :param beta: float, optional, precision of likelihood, from data noise
        self.m: expectation of posterior for w
        self.S: Variance of posterior for w
        """
        self.alpha = alpha
        self.beta = beta
        self.w_mean = None
        self.w_precision = None

    def _has_prior(self):
        return self.w_mean is not None and self.w_precision is not None

    def _get_prior(self, ndim: int) -> tuple:
        if self._has_prior():
            return self.w_mean, self.w_precision
        return np.zeros(ndim), self.alpha * np.eye(ndim)

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        """
        Update the mean and precision of posterior of w given new training data
        ----Parameters:
        :param train_x: np.ndarray, shape ~ (n, m)
        :param train_y: np.ndarray, shape ~ (n, 1)
        :return: None
        """
        mean_prev, precision_prev = self._get_prior(np.size(train_x, 1))

        w_precision = precision_prev + self.beta * train_x.T @ train_x
        w_mean = np.linalg.solve(
            w_precision,
            precision_prev @ mean_prev + self.beta * train_x.T @ train_y,
        )
        self.w_mean = w_mean
        self.w_precision = w_precision
        self.w_cov = np.linalg.inv(self.w_precision)

        return None

    def predict(self,
                x: np.ndarray,
                return_std: bool = True,
                sample_size: int = None
                ):
        """
        Do prediction based on input of new data pints
            Predictive distribution
            p(t|x,X,alpha,beta) = N(t|w.T@x, 1/beta + x.T@S@x)

        Parameters:
        -----------
        :param x: np.ndarray, shape ~ (N, n_sample)
        :param return_std: bool, optional, flag to return stand deviation of predictions
            default is True
        :param sample_size: int, optional, number of samples to draw from the predictive distribution
            (the default is None, no sampling from the distribution)

        Returns:
        -----------
        y : np.ndarray
            mean of the predictive distribution (N,)
        y_std : np.ndarray
            standard deviation of the predictive distribution (N,)
        y_sample : np.ndarray
            samples from the predictive distribution (N, sample_size)
        """

        y = x @ self.w_mean
        if sample_size is not None:
            w_sample = np.random.multivariate_normal(
                self.w_mean, self.w_cov, size=sample_size
            )
            y_sample = x @ w_sample.T

        if return_std:
            y_var = 1 / self.beta + np.sum(x @ self.w_cov * x, axis=1)
            y_std = np.sqrt(y_var)
            return y, y_std
        return y