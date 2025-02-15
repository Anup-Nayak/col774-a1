# Imports -do not import any other libraries other than these
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

# You may add any other functions to make your code more modular. However,
# do not change the function signatures (name and arguments) of the given functions,
# as these functions will be called by the autograder.


class GaussianDiscriminantAnalysis:
    # Assume Binary Classification
    def __init__(self):
        self.mu_0 = None
        self.mu_1 = None
        self.sigma = None
        self.sigma_0 = None
        self.sigma_1 = None
        self.phi = None  # Class prior
        
    def normalize(self, X, use_stored_params=False):
        """
        Normalize the dataset X to have zero mean and unit variance.
        If use_stored_params is True, use the stored training mean and std.
        """
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    
    def fit(self, X, y, assume_same_covariance=False):
        """
        Fit the Gaussian Discriminant Analysis model to the data.
        Remember to normalize the input data X before fitting the model.
        
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input data.
            
        y : numpy array of shape (n_samples,)
            The target labels - 0 or 1.
        
        learning_rate : float
            The learning rate to use in the update rule.
        
        Returns
        -------
        Parameters: 
            If assume_same_covariance = True - 3-tuple of numpy arrays mu_0, mu_1, sigma 
            If assume_same_covariance = False - 4-tuple of numpy arrays mu_0, mu_1, sigma_0, sigma_1
            The parameters learned by the model.
        """
        
        # Normalize input data
        X = self.normalize(X)
        
        # Compute class priors
        self.phi = np.mean(y)
        
        # Compute class-wise means
        self.mu_0 = np.mean(X[y == 0], axis=0)
        self.mu_1 = np.mean(X[y == 1], axis=0)
        
        # Compute class-wise covariance matrices
        if assume_same_covariance:
            n_0 = np.sum(y == 0)
            n_1 = np.sum(y == 1)
            sigma_0 = np.cov(X[y == 0].T, bias=True)
            sigma_1 = np.cov(X[y == 1].T, bias=True)
            self.sigma = (n_0 * sigma_0 + n_1 * sigma_1) / (n_0 + n_1)
            return self.mu_0, self.mu_1, self.sigma
        else:
            self.sigma_0 = np.cov(X[y == 0].T, bias=True)
            self.sigma_1 = np.cov(X[y == 1].T, bias=True)
            return self.mu_0, self.mu_1, self.sigma_0, self.sigma_1
    
    def predict(self, X):
        """
        Predict the target values for the input data.
        
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input data.
            
        Returns
        -------
        y_pred : numpy array of shape (n_samples,)
            The predicted target label.
        """
        X = self.normalize(X)  # Normalize input data
        
        if self.sigma is not None:
            # If shared covariance is used
            inv_sigma = np.linalg.inv(self.sigma)
            term_0 = -0.5 * np.sum((X - self.mu_0) @ inv_sigma * (X - self.mu_0), axis=1)
            term_1 = -0.5 * np.sum((X - self.mu_1) @ inv_sigma * (X - self.mu_1), axis=1)
        else:
            # If separate covariance matrices are used
            term_0 = multivariate_normal.logpdf(X, mean=self.mu_0, cov=self.sigma_0)
            term_1 = multivariate_normal.logpdf(X, mean=self.mu_1, cov=self.sigma_1)
        # Compute log posterior probability
        log_posterior_0 = term_0 + np.log(1 - self.phi)
        log_posterior_1 = term_1 + np.log(self.phi)
        
        # Predict class based on higher posterior probability
        return (log_posterior_1 > log_posterior_0).astype(int)
