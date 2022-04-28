from itertools import count
from re import sub
import numpy as np
# import cv2
import os
import random
# from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

import tensorflow as tf
import matplotlib.pyplot as plt
from operator import add

class Model:

    def __init__(self, shape,dataset_path, labels_file_path, learning_rate, reg, epochs,classes=None):
        self.shape = shape
        self.weights = np.zeros((shape, ))
        self.classes = classes
        # self.data_range = data_range
        self.dataset_file_path = dataset_path
        self.labels_file_path = labels_file_path
        self.lr = learning_rate
        self.reg = reg
        self.epochs = epochs
        # self.load_data()

    def make_model(self,shape, classes):
        model = np.zeros((shape,))
        return model

    def set_data_range(self, data_range):
        self.data_range = data_range
    # def 
    def load_data(self):
        feature_data = list(np.load(self.dataset_file_path))
        label_data = list(np.load(self.labels_file_path))
        data = list(zip(feature_data, label_data))
        client_data = data[self.data_range[0]:self.data_range[1]]
        self.data = client_data
        self.n_samples = len(self.data)


    def loss_prime(self,y,y_hat):

        return y_hat - y

    def loss_dprime(self,y,y_hat):

        return np.ones_like(y_hat)

    def regularizer_prime(self,x):

        return x

    def regularizer_dprime(self,x):

        return np.ones_like(x)


    def train(self, num_epochs):

        n,d = len(self.data), len(self.data[0][0])
        # print(n,d)
        alphas = np.zeros((n,d))
        wts_copy = self.weights.copy()
        # print(wts_copy.shape)
        for epoch in range(num_epochs):
            for i in range(n):

                data_i = np.array(self.data[i][0])
                # print(data_i.shape)
                label_i = int(self.data[i][1])

                if(i==n):

                    alphas = alphas - np.mean(alphas, axis=0, keepdims=True)  # update all alphas

                else:
                    dot_i = data_i @ wts_copy

                    dprime = self.loss_dprime(label_i, dot_i)

                    diff = alphas[i,:] - (self.loss_prime(label_i, dot_i)*data_i) - self.reg*self.regularizer_prime(wts_copy)

                    # print(diff.shape)
                    inv = 1. / (1. + self.reg * self.regularizer_dprime(wts_copy))

                    # print(inv.shape)
                    scaled_data = inv * data_i
                    # print(scaled_data.shape)

                    # 10th line 1st term

                    cte = dprime * (scaled_data @ diff) / (1 + dprime * (data_i @ scaled_data))

                    # print(cte.shape)
                    # inv*diff = 10th line 2nd term and update is gamma*dk
                    update = self.lr * (inv * diff - cte * scaled_data)

                    # print(update.shape)
                    alphas[i, :] -= update  # update i-th alpha
                    wts_copy += update  # update wts
                #print(i)
        self.weights = wts_copy
        #print(wts_copy)
        return wts_copy
    
    # def test_model(self):

    #     cross_entropy = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    #     logits = model.predict(X_test)
    #     print(logits)
    #     print(Y_test)
    #     loss = cross_entropy(Y_test, logits)
    #     acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(Y_test, axis=1))