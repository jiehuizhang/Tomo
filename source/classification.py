'''Conduct Classification'''

import numpy as np
from sklearn import svm

class classifier:

    data_train = None
    data_test = None

    lab_train = None
    lab_test = None

    opt = None
    prams = None
    predicts = None
    accuracy = None

    def __init__ (self,data_in=None,lab_in=None):

        self.data_train = None
        self.data_test = None

        self.lab_train = None
        self.lab_test = None

        self.opt = None
        self.prams = None
        self.predicts = None
        self.accuracy = None

        if data_in != None:
            self.split_data(data_in,lab_in)
        
    def train(self, opt, data = None, label = None):

        if data != None:
            self.data_train = data
            self.lab_train = label

        if opt =='SVM':
            self.opt = 'SVM'
            '''self.prams = svm.SVC(C=0.7, kernel='rbf', degree=2, gamma=0.2,
                             coef0=0.0, shrinking=True, probability=False, tol=0.001,
                             cache_size=200, class_weight=None, verbose=False,
                             max_iter=-1, random_state=None)'''
            self.prams = svm.SVR(kernel='rbf', degree=3, gamma=0.0, coef0=0.0, tol=0.001,
                                 C=1.0, epsilon=0.1, shrinking=True, probability=False,
                                 cache_size=200, verbose=False, max_iter=-1, random_state=None)
            self.prams.fit(self.data_train,self.lab_train)

    def classify(self, data = None, label = None):

        if data != None:
            self.data_test = data
            self.lab_test = label
        if self.opt =='SVM':
            self.predicts = self.prams.predict(self.data_test)

    def split_data(self,data_projected,label):

        lab_cla_1 = np.where(label == 0)[0]
        num_cla_1 = lab_cla_1.shape[0] 
        train_ids_1 = lab_cla_1[range(0,num_cla_1,2)]
        test_ids_1 = lab_cla_1[range(1,num_cla_1,2)]
        train_data_1 = data_projected[train_ids_1,:]
        train_label_1 = label[train_ids_1]
        test_data_1 = data_projected[test_ids_1,:]
        test_label_1 = label[test_ids_1]

        lab_cla_2 = np.where(label == 1)[0]
        num_cla_2 = lab_cla_2.shape[0]
        train_ids_2 = lab_cla_2[range(0,num_cla_2,2)]
        test_ids_2 = lab_cla_2[range(1,num_cla_2,2)]
        train_data_2 = data_projected[train_ids_2,:]
        train_label_2 = label[train_ids_2]
        test_data_2 = data_projected[test_ids_2,:]
        test_label_2 = label[test_ids_2]

        self.data_train = np.vstack((train_data_1,train_data_2))
        self.lab_train = np.hstack((train_label_1,train_label_2))

        self.data_test = np.vstack((test_data_1,test_data_2))
        self.lab_test = np.hstack((test_label_1,test_label_2))
        

        
    
