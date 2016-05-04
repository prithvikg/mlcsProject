
import logging

import sys
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import Ridge
import time
from scipy.optimize import minimize
import pickle
import util
import collections as col
from itertools import izip, tee, islice
import pandas as pd
import numpy as np
import math
import clean
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn import linear_model


def feature_normalization(train, test):
    """Rescale the data so that each feature in the training set is in
    the interval [0,1], and apply the same transformations to the test
    set, using the statistics computed on the training set.

    Args:
        train - training set, a 2D numpy array of size (num_instances, num_features)
        test  - test set, a 2D numpy array of size (num_instances, num_features)
    Returns:
        train_normalized - training set after normalization
        test_normalized  - test set after normalization

    """
    #print "inside feature_normalization"
    #print train
    #print test
    nptrain = np.array(train, dtype=np.float64)
    nptest = np.array(test, dtype=np.float64)
    #print nptrain
    r,c = nptrain.shape
    #print r
    #print c
    maxvalues = []
    minvalues = []
    for i in range(0,c,1):
        #print "%d" %(i)
        vslice = nptrain[:,i:i+1]
        #print vslice.shape
        vslicemax = vslice.max()
        vslicemin = vslice.min()
        #print "vslicemax is %d" %(vslicemax)
        #print "vslicemin is %d" %(vslicemin)
        maxvalues.append(vslicemax-vslicemin)
        minvalues.append(vslicemin)

    #print "Outside for loop"
    npmax = np.array(maxvalues)
    npmin = np.array(minvalues)


    rows= len(npmax)
    # print "inside normalisation"
    # print rows
    for i in range(0,rows):
        if (npmax[i] - npmin[i] == 0):
            npmax[i] = npmin[i] + 1


    train_normalized =  np.array((nptrain - npmin) / (npmax - npmin), dtype=np.float64)
    #print train_normalized
    test_normalized =  np.array((nptest - npmin) / (npmax - npmin), dtype=np.float64)
    #print test_normalized
    return train_normalized,test_normalized


def compute_square_loss(y1, y2, threshold):
    r,c = y1.shape
    J = y1 - y2
    for i in range(0,r):
        if J[i] < threshold:
            J[i] = 0
        else:
            J[i] = J[i]/threshold
    norm = np.linalg.norm(J)
    loss = np.square(norm) / (2*r)
    return loss

def main():
    inputFrame = pd.read_csv("DesignMatrix")
    # print inputFrame.tail()
    input = inputFrame.as_matrix()
    X = input[:,1:-1]
    Y = input[:,-1:]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size =500, random_state=5)

    Xnan = np.isnan(X)
    np.isnan(Y)

    r,c = X.shape
    for i in range(0,r):
        for j in range(0,c):
            if Xnan[i,j] == True:
                print i,j

    X_train, X_test =  feature_normalization(X_train, X_test)

    Xnan = np.isnan(X_train)
    r,c = X_train.shape
    for i in range(0,r):
        for j in range(0,c):
            if Xnan[i,j] == True:
                print i,j

    Xnan = np.isnan(X_test)
    r,c = X_test.shape
    for i in range(0,r):
        for j in range(0,c):
            if Xnan[i,j] == True:
                print i,j


    #Ridge Regression
    print "Starting ridge Regression"
    for i in range(-10,10):
        threshold = 10000
        alpha = 10**i;
        clf = Ridge(alpha=alpha)
        clf.fit(X_train, y_train)
        ytestpredict = clf.predict(X_test)
        ytrainpredict = clf.predict(X_train)
        # print y_test.shape
        # print ytestpredict.shape
        # print y_train.shape
        # print ytrainpredict.shape
        print i, compute_square_loss(ytestpredict,y_test,threshold), compute_square_loss(ytrainpredict,y_train,threshold)

    #Lasso Regression
    print "Starting Lasso Regression"
    for i in range(-10,10):
        threshold = 10000
        alpha = 10**i;
        clf = linear_model.Lasso(alpha=alpha)
        clf.fit(X_train,y_train)
        ytestpredict = clf.predict(X_test)
        ytrainpredict = clf.predict(X_train)
        ytestpredict = np.reshape(ytestpredict, (len(ytestpredict), 1))
        ytrainpredict = np.reshape(ytrainpredict, (len(ytrainpredict), 1))
        # print y_test.shape
        # print ytestpredict.shape
        # print y_train.shape
        # print ytrainpredict.shape
        print i, compute_square_loss(y_test,ytestpredict,threshold), compute_square_loss(y_train,ytrainpredict,threshold)





if __name__ == "__main__":
    main()

