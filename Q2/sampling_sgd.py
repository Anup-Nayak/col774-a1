# Imports - you can add any other permitted libraries
import numpy as np

# You may add any other functions to make your code more modular. However,
# do not change the function signatures (name and arguments) of the given functions,
# as these functions will be called by the autograder.

def generate(N, theta, input_mean, input_sigma, noise_sigma):
    """
    Generate normally distributed input data and target values
    Note that we have 2 input features
    Parameters
    ----------
    N : int
        The number of samples to generate.
        
    theta : numpy array of shape (3,)
        The true parameters of the linear regression model.
        
    input_mean : numpy array of shape (2,)
        The mean of the input data.
        
    input_sigma : numpy array of shape (2,)
        The standard deviation of the input data.
        
    noise_sigma : float
        The standard deviation of the Gaussian noise.
        
    Returns
    -------
    X : numpy array of shape (N, 2)
        The input data.
        
    y : numpy array of shape (N,)
        The target values.
    """
    # Generate x1 and x2 independently
    X = np.random.normal(loc=input_mean, scale=input_sigma, size=(N, 2))
    
    # Add intercept term (x0 = 1)
    X_intercept = np.hstack((np.ones((N, 1)), X))  # Shape (N, 3)
    
    # Generate noise
    noise = np.random.normal(loc=0, scale=noise_sigma, size=N)
    
    # Compute target variable y
    y = X_intercept @ theta + noise  # Matrix multiplication
    
    return X, y

class StochasticLinearRegressor:
    def __init__(self):
        self.theta = None
        self.epochs = 1000
        self.batch_size = 16
        self.epsilon = 1e-6
    
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
        N, d = X.shape
        X_intercept = np.hstack((np.ones((N, 1)), X))
        self.theta = np.zeros(d + 1) 
        
        history = [] 
        prev_loss = float('inf')
        
        for epoch in range(self.epochs):
            indices = np.random.permutation(N) 
            X_shuffled = X_intercept[indices]
            y_shuffled = y[indices]
            
            loss_sum = 0
            for i in range(0, N, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]
                
                # Compute gradient
                gradient = -2 / self.batch_size * X_batch.T @ (y_batch - X_batch @ self.theta)
                
                # Update parameters
                self.theta -= learning_rate * gradient
                
                # Compute batch loss
                loss_sum += np.mean((y_batch - X_batch @ self.theta) ** 2)
            
            avg_loss = loss_sum / (N // self.batch_size)
            history.append(self.theta.copy())
            
            # Check convergence criteria
            if abs(avg_loss - prev_loss) <= self.epsilon:
                break
            prev_loss = avg_loss
        
        return np.array(history)
    
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
        N = X.shape[0]
        X_intercept = np.hstack((np.ones((N, 1)), X)) 
        return X_intercept @ self.theta
