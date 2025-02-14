import numpy as np

def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

class LinearRegressor:
    def __init__(self):
        self.theta = None
        self.iterations = 1000
        self.epsilon = 1e-15
    
    def fit(self, X, y, learning_rate=0.01):
        """
        Fit the linear regression model to the data using Gradient Descent.
        
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input data.
            
        y : numpy array of shape (n_samples,)
            The target values.

        learning_rate : float
            The learning rate to use in the update rule.
            
        Returns
        -------
        List of Parameters: numpy array of shape (n_iter, n_features+1,)
            The list of parameters obtained after each iteration of Gradient Descent.
        """
        m, n = X.shape
        X = np.c_[np.ones(m), X] 
        self.theta = np.zeros(n + 1)
        theta_history = []
        theta_history.append(self.theta.copy())
        J_prev = float('inf')
        
        for _ in range(self.iterations):
            predictions = X.dot(self.theta)
            errors = predictions - y
            gradient = (1 / m) * X.T.dot(errors)
            self.theta -= learning_rate * gradient
            J_curr = compute_cost(X, y, self.theta)
            theta_history.append(self.theta.copy())
            
            if abs(J_prev - J_curr) < self.epsilon:
                break
            J_prev = J_curr
        
        return np.array(theta_history)
    
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
            The predicted target values.
        """
        m = X.shape[0]
        X = np.c_[np.ones(m), X]
        return X.dot(self.theta)