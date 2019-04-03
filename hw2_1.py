import numpy as np
import pandas as pd
import math
from sklearn.model_selection import KFold

class logistic_regression():
    def __init__(self):
        self.iter_numb =100
        self.learning_rate = 0.001

    def sigmoid(self,a):
        return 1.0/(1.0+math.exp(-a))

    def initial(self,w_numb):
        self.w = np.zeros((1,w_numb))
        self.w0 = 0

    def stochastic_gradient_descent(self,train_data):
        dw = 0
        dw0 = 0
        for i in range(self.iter_numb):
            np.random.shuffle(train_data)
            for data in train_data:
                data = data.reshape((1,11))
                x,y = self.seperate_data(data)
                fx  = self.sigmoid((np.dot(self.w,x.T)+self.w0))
                dw += np.dot((fx-y),x)
                dw0 += (fx-y)
            self.w = self.w - self.learning_rate * dw
            self.w0 = self.w0 -self.learning_rate * dw0

    def minibatch_gradient_descent(self,train_data):
        dw = 0
        dw0 = 0
        for i in range(self.iter_numb):
            np.random.shuffle(train_data)
            idx = np.random.randint(train_data.shape[0], size=20)
            for data in train_data[idx,:]:
                data = data.reshape((1,11))
                x,y = self.seperate_data(data)
                fx  = self.sigmoid((np.dot(self.w,x.T)+self.w0))
                dw += np.dot((fx-y),x)
                dw0 += (fx-y)
            self.w = self.w - self.learning_rate * dw
            self.w0 = self.w0 -self.learning_rate * dw0

    def predict(self,test_data):
        y_predict = []
        for data in test_data:
            data = data.reshape((1, 11))
            x,y = self.seperate_data(data)
            if self.sigmoid(np.dot(self.w,x.T)+self.w0)>0.5:
                y_predict.append(1)
            else:
                y_predict.append(0)
        return y_predict

    def seperate_data(self,data):
        x = data[:,1:10]
        y = 0
        if data[:,10] == 4:
            y = 1
        return x,y

def readfile(filename):
    data = pd.read_csv(filename)
    data = np.array(data).astype(np.float64)
    return data

def result(y_target,y_predict):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i in range(len(y_target)):
        if y_target[i] == 1:
            if y_predict[i] == 1:
                tp += 1
            else:
                fn += 1
        else:
            if y_predict[i] == 1:
                fp += 1
            else:
                tn += 1

    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    return recall,precision,accuracy

def get_y(data):
    y = data[:,10]
    y_lable = []
    for i in range (len(y)):
        if y[i] == 4:
            y_lable.append(1)
        else:
            y_lable.append(0)
    return y_lable

def sgd_prediction(train_data,test_data):
    #data = readfile("D:\graduate\courses\cs-559\hw2_1data.csv")
    #train_data = data[0:500, :]
    #test_data = data[500:550, :]
    y_target = get_y(test_data)
    logistic_Regression = logistic_regression()
    logistic_Regression.initial(9)
    logistic_Regression.stochastic_gradient_descent(train_data)
    y_predict = logistic_Regression.predict(test_data)
    recall, precision, accuracy = result(y_target, y_predict)
    return recall,precision,accuracy
def mini_batch_prediction(train_data,test_data):
    y_target = get_y(test_data)
    logistic_Regression = logistic_regression()
    logistic_Regression.initial(9)
    logistic_Regression.minibatch_gradient_descent(train_data)
    y_predict = logistic_Regression.predict(test_data)
    recall, precision, accuracy = result(y_target, y_predict)
    return recall,precision,accuracy

def cross_val_result_sgd():
    data = readfile("D:\graduate\courses\cs-559\hw2_1data.csv")
    kf = KFold(n_splits=5)
    i=1;
    for train_index,test_index in kf.split(data):
        train_data = data[train_index,:]
        test_data = data[test_index,:]
        recall,precision,accuracy = sgd_prediction(train_data, test_data)
        print('Result of {} dataset: recall: {}%; precision: {}%; accuracy: {}%'.format(i,100*recall,100*precision,100*accuracy))
        i = i+1
def cross_val_result_minibatch():
    data = readfile("D:\graduate\courses\cs-559\hw2_1data.csv")
    kf = KFold(n_splits=5)
    i=1;
    for train_index,test_index in kf.split(data):
        train_data = data[train_index,:]
        test_data = data[test_index,:]
        recall, precision, accuracy = mini_batch_prediction(train_data, test_data)
        print('Result of {} dataset: recall: {}%; precision: {}%; accuracy: {}%'.format(i,100*recall,100*precision,100*accuracy))
        i = i+1
print('results using stochastic gradient descent:')
cross_val_result_sgd()
print('results using mini_batch gradient descent')
cross_val_result_minibatch()













