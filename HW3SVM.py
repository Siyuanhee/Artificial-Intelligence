# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 22:15:51 2018

@author: Siyuan
"""

import matplotlib.pylab as plt
from numpy import *
import numpy as np
import math
from scipy import linalg as LA
import scipy.io as scio
 
dataFile = 'mnist_01.mat'
data = scio.loadmat(dataFile)
X_train = data["X_train"]
X_test = data["X_test"]
label_test_copy = data["label_test"]
label_train_copy = data["label_train"]
label_test= []
label_train= []
for i in range(len(label_test_copy)):
    if (label_test_copy[i] == 0):
        label_test.append(-1)
    else:
        label_test.append(1)
label_test = np.array(label_test).reshape(1000,1)
#print (label_test)
for i in range(len(label_train_copy)):
    if (label_train_copy[i] == 0):
        label_train.append(-1)
    else:
        label_train.append(1)
label_train = np.array(label_train).reshape(10000,1)
#print(label_train)
from matplotlib import pyplot as plt
print('Data Preparation :')
ran_train = np.random.randint(0,9999,(1,10))
#print(ran_train)
count = 1
for i in ran_train[0]:
    pic = X_train[i]
    pic = pic.reshape((28, 28))
    plt.subplot(2,5,count)
    plt.title('train image' + str(count))
    plt.imshow(pic, cmap=plt.cm.binary)
    count += 1
plt.show()
print('label train:')
for i in ran_train[0]:
    print(label_train[i])
count = 1
ran_test = np.random.randint(0,999,(1,10))
for i in ran_test[0]:
    pic = X_test[i]
    pic = pic.reshape((28, 28))
    plt.subplot(2,5,count)
    plt.title('test image' + str(count))
    plt.imshow(pic, cmap=plt.cm.binary)
    count += 1
plt.show()
print('label test:')
for i in ran_test[0]:
    print(label_test[i])

def stochastic_gradient_descent(Lambda, iteration,inX,inY):

    X = mat(inX)
    n = shape(X)[0]
    
    y = mat(inY)
    d = shape(X)[1]
    #print(n)
    wt = np.zeros(d)
    wt = np.array(wt).reshape(d,1)
    Fw = 0
    total = np.zeros(d)
    total = np.array(total).reshape(d,1)

    #rate = 0.01
    result = []
    temp = 0
    for n in range(iteration):
        #rate = 1/(n+1)**2
        rate = 1/(n+1)
        i = np.random.randint(0,shape(X)[0]-1)
        if 1 - np.dot(y[i], np.dot(X[i,:] , wt)) > 0 :
            gra = -np.dot(y[i],X[i,:]).reshape(d,1) + np.dot(Lambda , wt)
            total += (1 - np.dot(y[i], np.dot(X[i,:] , wt)) + (Lambda / 2) * (LA.norm((wt),2))**2)
            wt = wt - np.dot(rate , gra)
            Fw = total/(n + 1)
            
            result.append(Fw.item(0))
        elif((1 - np.dot(y[i], np.dot(X[i,:] , wt))) <= 0):
            #print((1 - np.dot(y[i], np.dot(X[i,:] , wt))))
            gra = np.dot(Lambda , wt)
            total += (Lambda / 2) * (LA.norm((wt),2))**2
            wt = wt - np.dot(rate , gra)
            Fw = total/(n + 1)
            
            result.append(Fw.item(0))
    return result,wt

result,w = stochastic_gradient_descent(1,20000,X_train,label_train)
print('2.3 Convergence of SGD :')
t = np.arange(1, 20001, 1)
plt.title('t and Fw with rate 1/t')
plt.plot(t, result)
plt.xlabel('t')
plt.ylabel('Fw')
plt.show()
plt.title('1/t and Fw with rate 1/t')
plt.plot(1/t, result)
plt.xlabel('1/t')
plt.ylabel('Fw')
plt.show()

def stochastic_gradient_descent_staticRate(Lambda, iteration,inX,inY):

    X = mat(inX)
    n = shape(X)[0]
    
    y = mat(inY)
    d = shape(X)[1]
    #print(n)
    wt = np.zeros(d)
    wt = np.array(wt).reshape(d,1)
    Fw = 0
    total = np.zeros(d)
    total = np.array(total).reshape(d,1)

    rate = 0.01
    result = []
    temp = 0
    for n in range(iteration):
        #rate = 1/(n+1)**2
        #rate = 2/(Lambda*(n+1))
        i = np.random.randint(0,shape(X)[0]-1)
        if((1 - np.dot(y[i], np.dot(X[i,:] , wt))) > 0):
            gra = -np.dot(y[i],X[i,:]).reshape(d,1) + np.dot(Lambda , wt)
            temp = (np.dot(y[i], np.dot(X[i,:] , wt)))
            total += 1 - temp + (Lambda / 2) * (LA.norm((wt),2))**2
            Fw = total/(n + 1)
            wt = wt - np.dot(rate , gra)
            result.append(Fw.item(0))
        elif((1 - np.dot(y[i], np.dot(X[i,:] , wt))) <= 0):
            gra = np.dot(Lambda , wt)
            total += (Lambda / 2) * (LA.norm((wt),2))**2
            Fw = total/(n + 1)
            wt = wt - np.dot(rate , gra)
            result.append(Fw.item(0))

    return result,wt

result,w = stochastic_gradient_descent_staticRate(1,20000,X_train,label_train)
#print(result)
t = np.arange(1, 20001, 1)
plt.title('t and Fw with rate 0.01')
plt.plot(t, result)
plt.xlabel('t')
plt.ylabel('Fw')
plt.show()
plt.title('1/t and Fw with rate 0.01')
plt.plot(1/t, result)
plt.xlabel('1/t')
plt.ylabel('Fw')
plt.show()


def stochastic_gradient_descent_rateT(Lambda, iteration,inX,inY):

    X = mat(inX)
    n = shape(X)[0]
    
    y = mat(inY)
    d = shape(X)[1]
    #print(n)
    wt = np.zeros(d)
    wt = np.array(wt).reshape(d,1)
    Fw = 0
    total = np.zeros(d)
    total = np.array(total).reshape(d,1)

    rate = 0.01
    result = []
    temp = 0
    for n in range(iteration):
        rate = 1/(n+1)**2
        #rate = 2/(Lambda*(n+1))
        i = np.random.randint(0,shape(X)[0]-1)
        if((1 - np.dot(y[i], np.dot(X[i,:] , wt))) > 0):
            gra = -np.dot(y[i],X[i,:]).reshape(d,1) + np.dot(Lambda , wt)
            temp = (np.dot(y[i], np.dot(X[i,:] , wt)))
            total += 1 - temp + (Lambda / 2) * (LA.norm((wt),2))**2
            Fw = total/(n + 1)
            wt = wt - np.dot(rate , gra)
            result.append(Fw.item(0))
        elif((1 - np.dot(y[i], np.dot(X[i,:] , wt))) <= 0):
            gra = np.dot(Lambda , wt)
            total += (Lambda / 2) * (LA.norm((wt),2))**2
            Fw = total/(n + 1)
            wt = wt - np.dot(rate , gra)
            result.append(Fw.item(0))

    return result,wt

result,w = stochastic_gradient_descent_rateT(1,20000,X_train,label_train)
#print(result)
t = np.arange(1, 20001, 1)
plt.title('t and Fw with rate 1/t^2')
plt.plot(t, result)
plt.xlabel('t')
plt.ylabel('Fw')
plt.show()
plt.title('1/t and Fw with rate 1/t^2')
plt.plot(1/t, result)
plt.xlabel('1/t')
plt.ylabel('Fw')
plt.show()


def evaluate_accuracy(x,y,w):
    
    sign = np.dot(x,w)
    count = 0
    #print(sign)
    for i in range(len(sign)):

        if(sign[i] > 0 ):
            sign[i] = 1
        elif(sign[i] < 0):
            sign[i] = -1
        elif(sign[i] == 0):
            sign[i] = 0
    #print(shape(sign))
    #print(sign)
    for i in range(len(y)):
        if(y[i] == sign[i]):
            count += 1

    return count/len(y)

print('2.4 Hyper-Parameter :')

Lambda = [0.000001, 0.001, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 500, 1000]
w10000 = []
wnorm2 = []
training_accuracy = []
test_accuracy = []
for lam in Lambda:
    result,w = stochastic_gradient_descent(lam,10000,X_train,label_train)
    w10000.append(w)
    wnorm2.append(LA.norm((w),2)/784)
    test_accuracy.append(evaluate_accuracy(X_test,label_test,w))
    training_accuracy.append(evaluate_accuracy(X_train,label_train,w))
    

    
t = np.arange(1, 14, 1)
plt.title('lambda and training accuracy')
plt.plot(t, training_accuracy)
plt.xlabel('lambda')
plt.ylabel('accuracy')
plt.show()

t = np.arange(1, 14, 1)
plt.title('lambda and test accuracy')
plt.plot(t, test_accuracy)
plt.xlabel('lambda')
plt.ylabel('accuracy')
plt.show()
print(wnorm2)

plt.show()


best_w = w10000[3].copy()
best_w = np.array(np.absolute(best_w)).reshape(1,784)
#print(best_w[0])
idx_10 = np.argpartition(best_w[0],-10)[-10:]
idx_20 = np.argpartition(best_w[0],-20)[-20:]
idx_50 = np.argpartition(best_w[0],-50)[-50:]
idx_100 = np.argpartition(best_w[0],-100)[-100:]
idx_200 = np.argpartition(best_w[0],-200)[-200:]
idx_400 = np.argpartition(best_w[0],-400)[-400:]
#print(idx)

w_10 = w10000[3].copy()
test_pic1 = X_train[0].copy()
test_pic2 = X_train[9999].copy()
for i in range(len(w_10)):
    if(i not in idx_10):
        w_10[i] = 0
        test_pic1[i] = 0
        test_pic2[i] = 0
test_pic1 = test_pic1.reshape((28, 28))
plt.subplot(1,2,1)
plt.title('w10 test image 1')
plt.imshow(test_pic1, cmap=plt.cm.binary)

test_pic2 = test_pic2.reshape((28, 28))
plt.subplot(1,2,2)
plt.title('w10 test image 2')
plt.imshow(test_pic2, cmap=plt.cm.binary)
plt.show()
print('accuracy of w10')
print(evaluate_accuracy(X_test,label_test,w_10))


w_20 = w10000[3].copy()
test_pic1 = X_train[0].copy()
test_pic2 = X_train[9999].copy()
for i in range(len(w_20)):
    if(i not in idx_20):
        w_20[i] = 0
        test_pic1[i] = 0
        test_pic2[i] = 0
test_pic1 = test_pic1.reshape((28, 28))
plt.subplot(1,2,1)
plt.title('w20 test image 1')
plt.imshow(test_pic1, cmap=plt.cm.binary)

test_pic2 = test_pic2.reshape((28, 28))
plt.subplot(1,2,2)
plt.title('w20 test image 2')
plt.imshow(test_pic2, cmap=plt.cm.binary)
plt.show()
print('accuracy of w20')
print(evaluate_accuracy(X_test,label_test,w_20))

w_50 = w10000[3].copy()
test_pic1 = X_train[0].copy()
test_pic2 = X_train[9999].copy()
for i in range(len(w_50)):
    if(i not in idx_50):
        w_50[i] = 0
        test_pic1[i] = 0
        test_pic2[i] = 0
test_pic1 = test_pic1.reshape((28, 28))
plt.subplot(1,2,1)
plt.title('w50 test image 1')
plt.imshow(test_pic1, cmap=plt.cm.binary)

test_pic2 = test_pic2.reshape((28, 28))
plt.subplot(1,2,2)
plt.title('w50 test image 2')
plt.imshow(test_pic2, cmap=plt.cm.binary)
plt.show()
print('accuracy of w50')
print(evaluate_accuracy(X_test,label_test,w_50))

w_100 = w10000[3].copy()
test_pic1 = X_train[0].copy()
test_pic2 = X_train[9999].copy()
for i in range(len(w_100)):
    if(i not in idx_100):
        w_100[i] = 0
        test_pic1[i] = 0
        test_pic2[i] = 0
test_pic1 = test_pic1.reshape((28, 28))

plt.subplot(1,2,1)
plt.title('w100 test image 1')
plt.imshow(test_pic1, cmap=plt.cm.binary)

test_pic2 = test_pic2.reshape((28, 28))
plt.subplot(1,2,2)
plt.title('w100 test image 2')
plt.imshow(test_pic2, cmap=plt.cm.binary)
plt.show()
print('accuracy of w100')
print(evaluate_accuracy(X_test,label_test,w_100))

w_200 = w10000[3].copy()
test_pic1 = X_train[0].copy()
test_pic2 = X_train[9999].copy()
for i in range(len(w_200)):
    if(i not in idx_200):
        w_200[i] = 0
        test_pic1[i] = 0
        test_pic2[i] = 0
test_pic1 = test_pic1.reshape((28, 28))
plt.subplot(1,2,1)
plt.title('w200 test image 1')
plt.imshow(test_pic1, cmap=plt.cm.binary)

test_pic2 = test_pic2.reshape((28, 28))
plt.subplot(1,2,2)
plt.title('w200 test image 2')
plt.imshow(test_pic2, cmap=plt.cm.binary)
plt.show()
print('accuracy of w200')
print(evaluate_accuracy(X_test,label_test,w_100))

w_400 = w10000[3].copy()
test_pic1 = X_train[0].copy()
test_pic2 = X_train[9999].copy()
for i in range(len(w_400)):
    if(i not in idx_400):
        w_400[i] = 0
        test_pic1[i] = 0
        test_pic2[i] = 0
        
test_pic1 = test_pic1.reshape((28, 28))
plt.subplot(1,2,1)
plt.title('w400 test image 1')
plt.imshow(test_pic1, cmap=plt.cm.binary)

test_pic2 = test_pic2.reshape((28, 28))
plt.subplot(1,2,2)
plt.title('w400 test image 2')
plt.imshow(test_pic2, cmap=plt.cm.binary)
plt.show()
print('accuracy of w400')
#print(w_400)
print(evaluate_accuracy(X_test,label_test,w_400))


print('2.5 Noisy Labels :')


label_train_noisy = label_train.copy()

level = [0,100,1000,2000,3000,5000,7000]
lambda_noisy = [0.000001, 0.001, 0.1, 0.5,1, 5, 10, 20, 50]
for i in level:
    print('number of noisy signal ' + str(i))
    ran_noisy = np.random.randint(0,9999,(1,i))
    for i in ran_noisy:
        #print(label_train_noisy[i])
        if(label_train_noisy[i].any() == -1):
            label_train_noisy[i] = 1
        elif(label_train_noisy[i].any() == 1):
            label_train_noisy[i] = -1
    #print(label_train_noisy)

    for i in lambda_noisy:
        result,w = stochastic_gradient_descent(i,5000,X_train,label_train_noisy)
        print('train accuracy for labbda = '+str(i) )
        print(evaluate_accuracy(X_train,label_train_noisy,w))
        print('test accuracy for lamdda = ' +str(i))
        print(evaluate_accuracy(X_test,label_test,w))
        print('l2-norm of w')
        print(LA.norm((w),2)/784)
        print()
    label_train_noisy = label_train.copy()



