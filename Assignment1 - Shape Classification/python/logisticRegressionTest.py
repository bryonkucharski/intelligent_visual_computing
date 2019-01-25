from __future__ import division
from Mesh import *
import _pickle as p
import numpy as np
from computeShapeHistogram import computeShapeHistogram
from utils import save_data, load_data

def logisticRegressionTest(test_dir, w, min_y, max_y, loadData=False):
    """
    complete this function to test a logistic regression
    classifier on a specified dataset

    input:
    test_dir is the path to a directory containing meshes
    in OBJ format. The directory must also contain a 
    ground_truth_labels.txt file that contains labels 
    (category id) for each mesh (necessary here to compute 
    test error)
    w are the classifier parameters learned by logisticRegressionTrain
    min_y, max_y are used to compute the range of the histogram
    for the shape descriptor (produced by logisticRegressionTrain)

    output:
    t: a row vector storing the probability of table (category '1') per test mesh
    test_err: test error (fraction of shapes whose label was mispredicted)
    
    
    if you want to avoid reloading meshes each time you run the code,
    change loadData to True in argument. The code automatically saves the data it loads in the first iteration
    """

    number_of_bins = w.shape[1]-1

    if loadData:
        meshes, N, shape_labels = load_data("tmp_test.p")

    else:
        #OPEN ground_truth_labels.txt
        shape_filenames = []
        shape_labels    = []

        try:
            with open(os.path.join(test_dir,'ground_truth_labels.txt'),'rU') as ground_truth_labels_file:
                for line in ground_truth_labels_file:
                    name, label = line.split()
                    shape_filenames.append(name)
                    shape_labels.append(int(label))
        
        except IOError as e:
            print("Couldn't open file (%s)." % e)
            return

        """
        read the training meshes, compute 'lowest' and 'highest' surface
        point across all meshes, move meshes such that their centroid is
        at (0, 0, 0), scale meshes such that average vertical distance to
        mesh centroid is one.
        """
        meshes = [] #A cell array storing all meshes
        N      = len(shape_filenames) #number of training meshes

        for n in range(N):
            meshes.append(Mesh(test_dir, shape_filenames[n], number_of_bins))
            number_of_mesh_vertices = meshes[n].V.shape[0] #number of mesh vertices
            mesh_centroid           = np.mean(meshes[n].V, axis=0, keepdims = True)
            meshes[n].V             = meshes[n].V -  mesh_centroid #center mesh at origin
            average_distance_to_centroid_along_y = np.mean(np.abs(meshes[n].V[:,1])) #average vertical distance to centroid
            meshes[n].V             = meshes[n].V/average_distance_to_centroid_along_y #scale meshes
            print(shape_filenames[n] + " Processed")

        save_data(meshes, 0, 0, N, shape_labels, 'tmp_test.p')


    """
    this loop calls your histogram computation code!
    all shape descriptors are organized into a NxD matrix
    N training meshes, D dimensions (number of bins in your histogram)
    """

    X = np.zeros((N, number_of_bins))
    for n in range(N):
        X[n,:] = computeShapeHistogram( meshes[n], min_y, max_y, number_of_bins )


    """""""""""""""""""""""""""""""""""""""
    ADD CODE HERE TO TEST CLASSIFIER
    """""""""""""""""""""""""""""""""""""""

    t = .5  # your predictions (change this!), report t values in your report!!!
    test_err = 1  # recompute this based on your predictions, report it in your report!!!

    print('Test classification error : %.2f%%'%(test_err * 100))

    return t, test_err


