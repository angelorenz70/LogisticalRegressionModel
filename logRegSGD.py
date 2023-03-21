import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import pickle
from scipy.special import expit

class LogisticRegressionSGD:
    def __init__(self, learning_rate=0.01, n_epochs=1000, batch_size=1):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size

    def fit(self, X, y):
        # add bias term to X
        X = np.insert(X, 0, 1, axis=1)

        # initialize weights
        self.weights = np.zeros(X.shape[1])

        # loop over epochs
        for epoch in range(self.n_epochs):
            # shuffle data
            idx = np.random.permutation(X.shape[0])
            X = X[idx]
            y = y[idx]

            # loop over batches
            for i in range(0, X.shape[0], self.batch_size):
                # get batch
                X_batch = X[i:i+self.batch_size]
                y_batch = y[i:i+self.batch_size]

                # compute gradient
                y_pred = expit(X_batch.dot(self.weights))
                gradient = X_batch.T.dot(y_pred - y_batch) / self.batch_size

                # update weights
                self.weights -= self.learning_rate * gradient

    def predict(self, X):
        # add bias term to X
        X = np.insert(X, 0, 1, axis=1)

        # compute predictions
        y_pred = expit(X.dot(self.weights))

        # convert probabilities to binary predictions
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0

        return y_pred.astype(int)