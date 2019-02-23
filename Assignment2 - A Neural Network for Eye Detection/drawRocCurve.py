from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

def drawRocCurve(Ytrue, Ypred):

    tpr = []
    fpr = []
    Ytrue_indices = (Ytrue==1).astype(int)
    for i in np.linspace(0,1,11):
        Y = Ypred > i
        Ypred_indices = (Y==1).astype(int)
        total = Ytrue_indices + Ypred_indices
        diffe = Ytrue_indices - Ypred_indices
        tp = np.sum(total==2)
        tn = np.sum(total==0)
        fp = np.sum(diffe==-1)
        fn = np.sum(diffe==1)
        tpr.append(tp/(tp+fn))
        fpr.append(fp/(fp+tn))


    plt.plot(fpr,tpr)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    return


