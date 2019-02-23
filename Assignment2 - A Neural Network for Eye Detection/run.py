from __future__ import division
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from tryEyeDetector import tryEyeDetector
from trainNN import trainNN
from trainNN_batched import trainNN_batched
from testNN import testNN
from drawRocCurve import drawRocCurve
import pickle

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
model = trainNN(X, Y, threshold)
#pickle.dump(model,open('trainNNmodel.pkl','wb'))

#model = trainNN_batched(X, Y, threshold)
#pickle.dump(model,open('trainNNmodel_batched.pkl','wb'))

#model = pickle.load(open('trainNNmodel.pkl','rb'))
#model = pickle.load(open('trainNNmodel_batched.pkl','rb'))
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

Ypred = testNN(model, Xtest, threshold=threshold)
err = np.mean(Ypred!=Ytest)
print('Test error is %.2f%%\n'%(100 * err))


#""" Draw ROC Curve """
drawRocCurve(Ytest, testNN(model, Xtest, True, threshold))



img = 'star_trek1.pgm'
corners_mat_filename = 'star_trek1_corners.mat'
sizeIm = [25, 20]
threshold = 0.9995
tryEyeDetector(model, img, sizeIm, corners_mat_filename, threshold)

