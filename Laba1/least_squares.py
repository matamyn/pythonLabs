# let it be library for solving machine learning problems

import numpy as np
import matplotlib.pyplot as plt
import csv

class LeastSquares:
    def __init__(self, constant=False, pow_2_factors = []):
        self.constant = constant
        self.pow_2_factors = pow_2_factors
        pass
    
    def fit(self, X, y):
        '''
        Fit Least Squares regression model 
        
        using formula B = (X^T * X)^(-1) * X^T * Y

        Parameters
        ----------
        X - matrix of features of size (m x n), [[x_11, x_21, ... , x_m1]
                                                 [x_12, x_22, ... , x_m2]
                                                              ...
                                                 [x_1n, x_2n, ... , x_mn]]
        y - array of true outputs
        
        Returns
        -------
        self : returns an instance of self.
        B: returns weights for model
        '''
        self.X=X
        self.y=y
        # implement kernel tricks
        for factor in self.pow_2_factors:
            self.X = self.add_squared(self.X, factor)
        if self.constant:
            self.X = self.add_constant(self.X)
        
        C = np.dot(np.transpose(self.X), self.X)
        G = np.linalg.inv(C)

        self.B = np.dot(np.dot(G, np.transpose(self.X)),y)
        return self.B

    def predict(self, X):
        '''
        Predicts outputs based on X and model
        '''
        # implement kernel tricks
        for factor in self.pow_2_factors:
            X = self.add_squared(X, factor)
        if self.constant:
            X = self.add_constant(X)
        # Actual prediction
        Y = np.dot(X, self.B)
        return Y

    def solve_(self, X, y):
        C = np.dot(np.transpose(X),X)
        G = np.linalg.inv(C)
        B = np.dot(np.dot(G, np.transpose(X)),y)
        return B

    def add_constant(self, X):
        '''
        Adds row of ones to matrix X
        '''
        if isinstance(X[0], list):
            return [[1.] + x for x in X]
        else:
            return [[1, x] for x in X]

    def add_squared(self, X, factor):
        '''
        Adds square of factor to matrix X
        '''
        if isinstance(X[0], list):
            return [x + [x[factor]**2] for x in X]
        else:
            return [[x, x**2] for x in X]
        
def plot_errors(Y_true, Y_pred):
    Y_true = np.array(Y_true)
    Y_pred = np.array(Y_pred)
    E = Y_true-Y_pred
    plt.figure()
    plt.scatter(x = Y_true, y=E)
    plt.show()
    
def solve(X, y):
    C = np.dot(np.transpose(X),X)
    G = np.linalg.inv(C)
    B = np.dot(np.dot(G, np.transpose(X)),y)
    return B

def generate_data(X, Y, ratio = 0.9):
    '''
    Splits data in two according to ratio into train data and test data
    '''
    from random import shuffle
    corpus = [[X[i]]+[Y[i]] for i, _ in enumerate(Y)]
    shuffle(corpus)
    train_amount = int(len(corpus)*ratio)
    train_target = [corpus[i][1] for i in range(train_amount)]
    train_data = [corpus[i][0] for i in range(train_amount)]
    test_target = [corpus[i][1] for i in range(train_amount, len(corpus))]
    test_data = [corpus[i][0] for i in range(train_amount, len(corpus))]
    return train_data, train_target, test_data, test_target

def normalize(X):
    # count max
    multiplier = [max(factor) for factor in X]
    new_X = []
    for i, factor in enumerate(X):
        # divide every entry by it
        new_X.append([x_i/multiplier[i] for x_i in factor])
    new_X = np.transpose(new_X)
    return new_X, multiplier

def open_csv(filename, delimiter = ","):
    with open(filename) as fn:
        data = csv.reader(fn, delimiter=delimiter)
        titles = next(data)
        features = []
        for line in data:
            features.append(line)
    return features, titles

if __name__ == "__main__":
    # one-dimensional
    features, titles = open_csv("car_price/data.csv", delimiter=';')
    X = [float(x[0]) for x in features]
    Y = [float(x[-1]) for x in features]
    del features, titles
    
    train_data, train_target, test_data, test_target = generate_data(X, Y)

    clf = LeastSquares(constant=True, pow_2_factors=[])
    clf.fit(train_data, train_target)
    
    plot_errors(test_target, clf.predict(test_data))
