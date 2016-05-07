
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
from datetime import datetime

def main2():
    time = "07/09/2011 07:33"
    time2 = "07/09/2011 07:35"
    s1 = '10:33:26'
    s2 = '11:15:49' # for example
    FMT = '%m/%d/%Y %H:%M'
    tdelta = datetime.strptime(time2, FMT)  - datetime.strptime(time, FMT)
    print tdelta.seconds


def main():
    T = 32
    df = pd.read_csv("train.csv", sep=',')
    # city = df['JobCity']
    # distinctCities = np.sort(city.unique())
    # print df['Gender'].unique()
    # print df['CollegeTier'].unique()
    print df['Degree'].unique()
    # print df['CollegeState'].unique()
    # print df['CollegeCityTier'].unique()
    # df = df.tail()

    dumCompanyName = pd.get_dummies(df['Gender'])
    result = dumCompanyName

    df['10board'] = df['10board'].apply(filterBoard)
    df['12board'] = df['12board'].apply(filterBoard)
    dataframe = df['10board']
    dummyFrame = pd.get_dummies(dataframe)
    result = pd.concat([result, dummyFrame], axis=1)

    dataframe = df['12board']
    dummyFrame = pd.get_dummies(dataframe)
    result = pd.concat([result, dummyFrame], axis=1)

    dataframe = df['Degree']
    dummyFrame = pd.get_dummies(dataframe)
    result = pd.concat([result, dummyFrame], axis=1)

    searchwords = addCustomSpecialisationColumns(df)
    for word in searchwords:
        if word!="eng":
            dummyFrame = df[word]
            result = pd.concat([result, dummyFrame], axis=1)

    dataframe =df['CollegeState']
    dummyFrame = pd.get_dummies(dataframe)
    result = pd.concat([result, dummyFrame], axis=1)

    yearConverter = lambda x: x.year
    df['myDOB'] = pd.to_datetime(df['DOB'])
    df['myDOB'] = df['myDOB'].apply(yearConverter)
    df['12age'] = df['12graduation'] - df['myDOB']
    df['gradage'] = df['GraduationYear'] - df['myDOB']

    result = pd.concat([result, df['myDOB']], axis=1)
    result = pd.concat([result, df['12age']], axis=1)
    result = pd.concat([result, df['gradage']], axis=1)
    result = pd.concat([result, df['GraduationYear']], axis=1)
    result = pd.concat([result, df['12graduation']], axis=1)

    result = pd.concat([result, df['10percentage']], axis=1)
    result = pd.concat([result, df['12percentage']], axis=1)
    result = pd.concat([result, df['CollegeTier']], axis=1)
    result = pd.concat([result, df['collegeGPA']], axis=1)
    result = pd.concat([result, df['CollegeCityTier']], axis=1)
    result = pd.concat([result, df['English']], axis=1)
    result = pd.concat([result, df['Logical']], axis=1)
    result = pd.concat([result, df['Quant']], axis=1)

    columns = modifyScoreColumns(df)
    for column in columns:
        result = pd.concat([result, df[column]], axis=1)

    result = pd.concat([result, df['conscientiousness']], axis=1)
    result = pd.concat([result, df['agreeableness']], axis=1)
    result = pd.concat([result, df['extraversion']], axis=1)
    result = pd.concat([result, df['nueroticism']], axis=1)
    result = pd.concat([result, df['openess_to_experience']], axis=1)

    result = pd.concat([result, df['Salary']], axis=1)
    result.to_csv("DesignMatrix")


def filterBoard(board):
    if  "cbse" in board.lower():
        return "cbse"
    elif "central board" in board.lower():
        return "cbse"
    elif "icse" in board.lower():
        return "icse"
    elif board == "0":
        return "na"
    else:
        return "state"

def getSpecialisationStats(series):
    dict = {}
    for index, value in series.iteritems():
        list = value.split(" ")
        for str in list:
            if str in dict:
                dict[str] += 1
            else:
                dict[str] = 1
    return dict

def addCustomSpecialisationColumns(dataframe):
    series = dataframe['Specialization']
    searchwords = ["computer","mechanical","biotechnology","science","applied","instrumentation","electronics","electrical","application","production","information","eng","telecommunications","technology","civil","industrial","control","engineering","other"]
    customColumns = {"computer":[],"mechanical":[],"biotechnology":[],"science":[],"applied":[],"instrumentation":[],"electronics":[],"electrical":[],"application":[],"production":[],"information":[],"eng":[],"telecommunications":[],"technology":[],"civil":[],"industrial":[],"control":[],"engineering":[], "other":[]}
    # For each row in the column,
    for row in dataframe['Specialization']:
        keywords = row.lower().split(" ")
        tempmap = {"computer":False,"mechanical":False,"biotechnology":False,"science":False,"applied":False,"instrumentation":False,"electronics":False,"electrical":False,"application":False,"production":False,"information":False,"eng":False,"telecommunications":False,"technology":False,"civil":False,"industrial":False,"control":False,"engineering":False,"other":False}
        someKeyWasFound = False
        for word in keywords:
            if word in searchwords:
                if word == "eng":
                    customColumns["engineering"].append(1)
                    tempmap["engineering"] = True
                    someKeyWasFound = True
                else:
                    customColumns[word].append(1)
                    tempmap[word] = True
                    someKeyWasFound = True

        if someKeyWasFound == False:
            customColumns["other"].append(1)
            tempmap["other"]=True

        # print len(tempmap)
        for key,value in tempmap.iteritems():
            if value is False:
                customColumns[key].append(0)


    # this should all be of same length
    for key, list in customColumns.iteritems():
        # print len(list)
        if key!="eng":
            dataframe[key] = list

    return searchwords


def modifyScoreColumns(dataframe):
    columns = ["Domain",	"ComputerProgramming",	"ElectronicsAndSemicon"	,"ComputerScience", 	"MechanicalEngg"	,"ElectricalEngg",	"TelecomEngg",	"CivilEngg"]
    for i, row in dataframe.iterrows():
        for column in columns:
            value = row[column]
            if value == -1:
                dataframe.set_value(i,column,0)
    return columns


if __name__ == "__main__":
    main()
