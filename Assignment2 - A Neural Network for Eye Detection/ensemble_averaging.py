from __future__ import division
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from tryEyeDetector import tryEyeDetector
from trainNN import trainNN
from testNN import testNN
from drawRocCurve import drawRocCurve

""" ====================================
       Neural Network Training
==================================== """
# load training dataset
trainSet = loadmat('trainSet.mat')
eyeIm = trainSet['eyeIm']
nonIm = trainSet['nonIm']
X = np.hstack((eyeIm, nonIm)).T #input image data represented as matrix of size #training images x #dimensions (=#pixels)
Y = np.vstack((np.ones((eyeIm.shape[1], 1)), np.zeros((nonIm.shape[1], 1)))) # binary labels (eye/non-eye) represented as a vector (#train images]x1)

""" Want to see the images? Uncomment the following lines """
# plt.imshow(np.reshape(X[0, :].T, (25, 20),order='F'),cmap='Greys_r')      #'positive' example
# plt.show()
# plt.imshow(np.reshape(X[2999, :].T, (25, 20),order='F'),cmap='Greys_r')   #'negative' example
# plt.show()

"""
normalize each image according to the mean and standard deviation of its pixel intensities
this makes our classifier more robust to brightness/constrast changes
"""
for i in range(X.shape[0]):
    X[i, :] -= np.mean(X[i, :])
    X[i, :] /= (np.std(X[i, :]) + 1e-6)

""" perform neural network training - you need to complete the trainNN.py script! """
threshold = 0.95
models = []
num_models = 3
for i in range(num_models):
    print('training model: ' + str(i) )
    models.append(trainNN(X, Y, threshold, verbose = False))



""" ====================================
       Neural Network Testing
==================================== """
# load the test dataset
testSet = loadmat('testSet.mat')
testEyeIm = testSet['testEyeIm']
testNonIm = testSet['testNonIm']
Xtest = np.hstack((testEyeIm, testNonIm)).T # [#test images] x [#dimensions]
Ytest = np.vstack((np.ones((testEyeIm.shape[1], 1)), np.zeros((testNonIm.shape[1], 1)))) # [#test images] x [1]

#normalize each image as above
for i in range(Xtest.shape[0]):
    Xtest[i, :] -= np.mean(Xtest[i, :])
    Xtest[i, :] /= (np.std(Xtest[i, :]) + 1e-6)

""" evaluate your network - you need to complete the testNN.m script! """
Ypreds = []
i = 0
for model in models:
    print('testing model: ' + str(i))
    Ypreds.append(testNN(model, Xtest, threshold=threshold, probabilities = True).T[0].tolist())
    i += 1
#average these bad boys
Ypreds = np.matrix(Ypreds)
Ypred = np.sum(Ypreds,axis = 0) / num_models
Ypred = Ypred.T
assert  Ypred.shape == Ytest.shape

Ypred[Ypred >= threshold] = 1
Ypred[Ypred < threshold] = 0

err = np.mean(Ypred!=Ytest)
print('Test error is %.2f%%\n'%(100 * err))
