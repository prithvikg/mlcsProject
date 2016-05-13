from sklearn import preprocessing, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV

import sys
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import Ridge
# import mord.multiclass
import pandas as pd
import numpy as np
import math
from datetime import datetime
from ord.mord import multiclass,regression_based,threshold_based


class Custom_Ridge(Ridge):
    def __init__(self, alpha=1.0, fit_intercept=True, normalize=False,
                 copy_X=True, max_iter=None, tol=1e-3, solver="auto",
                 random_state=None):
        super(Custom_Ridge, self).__init__(alpha=alpha, fit_intercept=fit_intercept,
                                    normalize=normalize, copy_X=copy_X,
                                    max_iter=max_iter, tol=tol, solver=solver,
                                    random_state=random_state)

    def score(self, X, y, sample_weight=None):
        y_t = y
        y_p = self.predict(X)
        r,c = y_t.shape
        J = y_p - y_t
        loss = np.divide(J,y_t)
        meanLoss = np.sum(np.square(loss))/r
        return meanLoss


def testscore(y_t, y_p, sample_weight=None):
    r,c = y_t.shape
    J = y_p - y_t
    loss = np.divide(J,y_t)
    meanLoss = np.sum(np.square(loss))/r
    return meanLoss


def main():
    dm = pd.read_csv("DesignMatrix2", sep=',')
    dm = dm.drop(dm.columns[[0]], axis=1)
    # print dm.tail()
    salary_bins = dm['Salary']
    # bins = pd.qcut(salary_bins, [0, .1, .4, .7, .85, 1.],labels=[1,2,3,4,5])

    bins = pd.qcut(salary_bins, [0, .20, .4, .60, .8, 1.],labels=[1,2,3,4,5])
    # bins = pd.qcut(salary_bins, [0, .33, .66, 1.],labels=[1,2,3])
    dm['Salary'] = bins

    dm.to_csv("DesignMatrixClassification")
    inputFrame = pd.read_csv("DesignMatrixClassification")
    inputFrame = inputFrame.drop(inputFrame.columns[[0]], axis=1)
    # print inputFrame.tail()
    inputmat = inputFrame.as_matrix()
    X = inputmat[:,0:-1]
    Y = inputmat[:,-1:]
    # print X.shape

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size =500, random_state=5)

    # X_train_scaled = scale(X_train)
    scaler_train = StandardScaler().fit(X)
    X_trans_train = scaler_train.transform(X_train)
    X_trans_test = scaler_train.transform(X_test)

    y_train = np.ndarray.flatten(y_train)
    y_train = y_train.astype(int)
    y_test = np.ndarray.flatten(y_test)
    y_test = y_test.astype(int)

    X_scaled = scale(X)
    scaler = StandardScaler().fit(X)
    X_trans = scaler.transform(X)


    # for i in range(-10,10):
    #     alpha = 10**i;
    #     clf2 = regression_based.OrdinalRidge(alpha=alpha)
    #     clf2.fit(X_trans_train, y_train)
    #     error = metrics.mean_absolute_error(clf2.predict(X_trans_train), y_train)
    #     error2 = metrics.mean_absolute_error(clf2.predict(X_trans_test), y_test)
    #     print('Mean Absolute Error of LAD %s %s %s' %(alpha, error, error2))

    clf2 = regression_based.OrdinalRidge(alpha=100)
    clf2.fit(X_trans_train, y_train)
    clf2.predict(X_trans_test)
    coef = clf2.coef_
    printWstats(clf2, inputFrame, threshold=0.1, threshold2=0.01)

    print "Done"

    clf2 = threshold_based.LogisticIT(alpha=1000)
    clf2.fit(X_trans_train, y_train)
    clf2.predict(X_trans_test)
    coef = clf2.coef_
    # print coef
    # printWstats(clf2, inputFrame, threshold=0.01, threshold2=0.001)


    print "Done"

    # for i in range(-10,10):
    #     alpha = 10**i;
    #     clf2 = threshold_based.LogisticAT(alpha=alpha)
    #     clf2.fit(X_trans_train, y_train)
    #     error = metrics.mean_absolute_error(clf2.predict(X_trans_train), y_train)
    #     error2 = metrics.mean_absolute_error(clf2.predict(X_trans_test), y_test)
    #     print('Mean Absolute Error of LogisticAT %s %s %s' %(alpha, error, error2))
    #
    # print "Done"
    #
    # for i in range(-10,10):
    #     alpha = 10**i;
    #     clf2 = threshold_based.LogisticIT(alpha=alpha)
    #     clf2.fit(X_trans_train, y_train)
    #     error = metrics.mean_absolute_error(clf2.predict(X_trans_train), y_train)
    #     error2 = metrics.mean_absolute_error(clf2.predict(X_trans_test), y_test)
    #     print('Mean Absolute Error of LogisticIT %s %s %s' %(alpha, error, error2))
    #
    # print "Done"
    #
    # for i in range(-10,10):
    #     alpha = 10**i;
    #     clf2 = threshold_based.LogisticSE(alpha=alpha)
    #     clf2.fit(X_trans_train, y_train)
    #     error = metrics.mean_absolute_error(clf2.predict(X_trans_train), y_train)
    #     error2 = metrics.mean_absolute_error(clf2.predict(X_trans_test), y_test)
    #     print('Mean Absolute Error of LogisticSE %s %s %s' %(alpha, error, error2))





    # clf2 = threshold_based.LogisticAT(alpha=1.)
    # tempy = np.ndarray.flatten(Y)
    # tempy = tempy.astype(int)
    #
    # clf2.fit(X, tempy)
    # print clf2.predict(X)
    # print('Mean Absolute Error of LogisticAT %s' %
    # metrics.mean_absolute_error(clf2.predict(X), tempy))

    # clf3 = threshold_based.LogisticIT(alpha=1.)
    # clf3.fit(X, tempy)
    # print('Mean Absolute Error of LogisticIT %s' %
    # metrics.mean_absolute_error(clf3.predict(X), tempy))
    #
    # clf4 = threshold_based.LogisticSE(alpha=1.)
    # clf4.fit(X, tempy)
    # print('Mean Absolute Error of LogisticSE %s' %
    # metrics.mean_absolute_error(clf4.predict(X), tempy))







    # grid_search = GridSearchCV(pipe, parameters, n_jobs=-1, verbose=1)

    # gr = grid_search.fit(X, Y)

    # print gr.best_params_

    # print gr.best_score_

    # pipe.set_params(preprocessing__with_mean=True,preprocessing__with_std=True,custom_ridge__alpha=0.0000001).fit(X,Y)
    #
    # prediction = pipe.predict(X)
    #
    # pipe.score(X,Y)
    #
    # pipe = Pipeline([
    #     ('preprocessing', StandardScaler()),
    #     ('ridge', Ridge()),
    # ])
    #
    # pipe.set_params(preprocessing__with_mean=True,preprocessing__with_std=True,ridge__alpha=10).fit(X,Y)



    # prediction = pipe.predict(X)
    # pipe.score(X,Y)


def printWstats(clf, dm, threshold=0.1, threshold2=0.01):
    # threshold =0.1
    # threshold2 =0.01
    coef1 = np.copy(clf.coef_)
    for i in range(0,len(coef1)):
        if abs(coef1[i]) < threshold:
            coef1[i] = 0

    cols = list(dm.columns.values)

    print "Tier 1"
    for i in range(0,len(coef1)):
        if coef1[i] > 0:
            print "Pos", i+1, cols[i], coef1[i]
    for i in range(0,len(coef1)):
        if coef1[i] < 0:
            print "Neg", i+1, cols[i], coef1[i]

    print "Tier 2"
    coef2 = np.copy(clf.coef_)
    for i in range(0,len(coef2)):
        if not (abs(coef2[i]) <= threshold and abs(coef2[i] >= threshold2)):
            coef2[i] = 0
    for i in range(0,len(coef2)):
        if coef2[i] > 0:
            print "Pos", i+1, cols[i], coef2[i]
    for i in range(0,len(coef2)):
        if coef2[i] < 0:
            print "Neg", i+1, cols[i], coef2[i]


if __name__ == "__main__":
    main()
