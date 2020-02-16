"""
Deep Patient implementation
Nathanael Romano
"""

import bcolz
import numpy as np
import os
import pandas as pd
import csv
from sklearn.base import BaseEstimator, TransformerMixin


# Varaibles in uppercase can be designated by user.
DATA_PATH = "jupyter/datasets/HEBEI_datase.csv"
PREDICT_COL = "Age"
DROP_COL = []
TEST_RATIO = 0.2
BCOLZ_PATH = "jupyter/datasets/bcolz"

class Dataset(object):
    '''Dataset class to handle loading and parsing training data.'''

    def __init__(self, use_notes=True, **kwargs):
        # Load the data
        self.dataset = self.load_data()
        # One hot encoding
        # Kate self.catonu()
        self.catonu2()
        np.savetxt('jupyter/datasets/hebei_transformed.csv', self.dataset, delimiter = ',')
        # Split dataset into train_set and val_set and test_set
        self.train_set, self.test_set =  self.split_train_test(test_ratio = TEST_RATIO)

        # Split x and y in both train_set and test_set
        xt = self.train_set.drop("age", axis = 1).values
        self.xtrain = xt[:, 1:]
        self.ytrain = self.train_set["age"].copy().values
        np.savetxt('jupyter/datasets/train_y.csv', self.ytrain, delimiter = ',')
        
        xtest = self.test_set.drop("age", axis = 1).values
        self.xtest = xtest[:, 1:]
        self.ytest = self.test_set["age"].copy().values
        np.savetxt('jupyter/datasets/test_y.csv', self.ytest, delimiter = ',')

        # Get the dimension of x
        self.dimension = self.xtrain.shape[1]
        self._index_in_epochs = 0
        self._epochs_completed = 0


    def load_data(self, data_path = DATA_PATH):
        return pd.read_csv(data_path)

    def catonu(self):
        with open('jupyter/datasets/HEBEI_datase.csv', 'r') as f:
            d_reader = csv.DictReader(f)
            #get fieldnames from DictReader object and store in list
            headers = d_reader.fieldnames

        cont_headers = []
        dum_headers = []
        for header in headers[1:-1]:
            if self.dataset[header].nunique() > 10:
                cont_headers.append(header)
            else:
                dum_headers.append(header)

        for header in dum_headers:
            self.dataset[header].apply(str)
            self.dataset=pd.concat([self.dataset, pd.get_dummies(self.dataset[header], prefix = header, columns = header)], axis=1)
            self.dataset=self.dataset.drop(header, axis=1)

    def catonu2(self):
        with open('jupyter/datasets/HEBEI_datase.csv', 'r') as f:
            d_reader = csv.DictReader(f)
            #get fieldnames from DictReader object and store in list
            headers = d_reader.fieldnames

        cont_headers = []
        dum_headers = []
        for header in headers[1:-1]:
            if self.dataset[header].nunique() > 10:
                cont_headers.append(header)
            else:
                dum_headers.append(header)


        # transform continous variables into categorical variables
        for header in cont_headers:
            labelList = [1,2,3,4,5,6,7]
            var_name = header + ' bin'
            self.dataset[var_name] = pd.cut(self.dataset[header],7)
            self.dataset=pd.concat([self.dataset, pd.get_dummies(self.dataset[var_name], prefix = header, columns = header)], axis=1)
            self.dataset=self.dataset.drop(var_name, axis=1)            
            self.dataset = self.dataset.drop(header, axis = 1)


        # one-hot encoding all the variables
        for header in dum_headers:
            self.dataset[header].apply(str)
            self.dataset=pd.concat([self.dataset, pd.get_dummies(self.dataset[header], prefix = header, columns = header)], axis=1)
            self.dataset=self.dataset.drop(header, axis=1)


    def split_train_test(self, test_ratio):
        shuffled_indices = np.random.permutation(len(self.dataset))
        test_set_size = int(len(self.dataset)*test_ratio)
        test_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_set_size:]
        return self.dataset.iloc[train_indices], self.dataset.iloc[test_indices]

    def load_set(self, set_name):
        """
        Loads a train/val/test set and its labels.
        Uses bcolz if the dataset is in /bcolz/
        """
        if set_name == 'test':
            return self.xtest

    def next_batch(self, batchSize, use_labels=False):
        """
        Gets the next data batch, and shuffles if end of epoch.
        """
        start = self._index_in_epochs
        self._index_in_epochs += batchSize

        if self._index_in_epochs >= self.xtrain.shape[0]:
            self._epochs_completed += 1
            perm = np.arange(self.xtrain.shape[0])
            np.random.shuffle(perm)
            self.xtrain = self.xtrain[perm, :]
            self.ytrain = self.ytrain[perm]
            start = 0
            self._index_in_epochs = batchSize

        end = self._index_in_epochs
        if use_labels:
            return self.xtrain[start:end, :], self.ytrain[start:end]
        else:
            return self.xtrain[start:end, :]

def save_data(data, name, path=None):
    '''Saves a dataset as bcolz archive.'''
    if path is None:
        path = BCOLZ_PATH
    car = bcolz.carray(data, rootdir=path+name)
    car.flush()
