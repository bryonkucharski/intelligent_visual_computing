from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import mode
from testNN import testNN
import matplotlib.patches as patches

def tryEyeDetector(model, image_filename, window_size, corners_mat_filename, threshold = 0.95):
    """
    code for detecting eyes in an image with a ML classifier

    Input:
    'model' is storing the neural net parameters (see trainNN.py)
    'image_filename' is the file name of the input image
    'window_size' should be the training image size: [25 20] 
    'threshold': your model outputs a continuous value 0...1 through the topmost sigmoid
    function. You can threshold it to decide whether you have eye or not
    (it does not need to be 0.5, but something larger depending on the desired
    true positive rate - see ROC curves)
    'corners_mat_filename' is an optional mat file storing the positions of
    corners (from a Harris corner detector) in the input image - use this
    argument if you don't have the image processing toolbox

    feel free to modify the input and output arguments if necessary
    """

    whole_image = plt.imread(image_filename)

    # load the corners from the given mat file.
    corners = loadmat(corners_mat_filename)['corners']
    
    save_x = []
    save_y = []
    # if you want to see the corners in the image:
    # plt.imshow(whole_image)
    # plt.show()
    # hold on
    # plot(corners(:,1), corners(:,2), 'r*')
    # hold off
    # pause

    # for each corner
    for c in range(corners.shape[0]):
        # search along perturbed windows (+/-5 pixels horizontally/vertically) around the corner 
        for px in range(-5,6):
            for py in range(-5,6):
                min_x = corners[c, 1] - int(np.ceil(window_size[0]/2)) + px
                min_y = corners[c, 0] - int(np.ceil(window_size[1]/2)) + py   
                if (min_x + window_size[0]> whole_image.shape[0]) or \
                ( min_y + window_size[1] > whole_image.shape[1] ) or \
                ( min_x < 0) or \
                ( min_y < 0):
                    continue
                
                # take 20x25 pixels image, size [25 20]
                subim = whole_image[min_x:min_x + window_size[0], min_y : min_y + window_size[1]].astype(np.float32)
                
                # normalize it as in training & testing
                Xtest  = subim.flatten(order='F').T
                Xtest -= np.mean(Xtest, axis = 0)
                Xtest /= (np.std(Xtest, axis = 0) + 1e-6)
                
                # call testNN.py - you need to complete this function
                Xtest = Xtest.reshape(1, -1)
                Y = testNN(model, Xtest, threshold=threshold)

                # store window coordinates
                if Y == 1:
                    save_x += [min_x]
                    save_y += [min_y]

    # show image + detected eyes as green rectangles
    # print len(save_x)
    fig,ax = plt.subplots(1)
    ax.imshow(whole_image, cmap = 'Greys_r')

    for n in range(len(save_x)):
        rect = patches.Rectangle((save_y[n],save_x[n]),window_size[1],window_size[0],linewidth=2,edgecolor='g',facecolor='none')
        ax.add_patch(rect)

    plt.show()