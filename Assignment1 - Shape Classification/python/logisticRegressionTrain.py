from __future__ import division
from Mesh import *
import _pickle as p
import numpy as np
from computeShapeHistogram import computeShapeHistogram
from utils import save_data, load_data

def logisticRegressionTrain(train_dir, number_of_bins, loadData=False):
    """
    complete this function to train a logistic regression classifier

    input:
    train_dir is the path to a directory containing meshes
    in OBJ format used for training. The directory must
    also contain a ground_truth_labels.txt file that
    contains the training labels (category id) for each mesh
    number_of_bins specifies the number of bins to use
    for your histogram-based mesh descriptor

    output:
    a row vector storing the learned classifier parameters (you must compute it)
    histogram range (this is computed for you)

    if you want to avoid reloading meshes each time you run the code,
    change loadData to True in argument. The code automatically saves the data it loads in the first iteration
    """

    if loadData:
        meshes, min_y, max_y, N, shape_labels = load_data('tmp_train.p')
    
    else:
        #OPEN ground_truth_labels.txt
        shape_filenames = []
        shape_labels = []
        
        try:
            with open(os.path.join(train_dir,'ground_truth_labels.txt'),'rU') as ground_truth_labels_file:
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
        min_y  = np.float('inf') #smallest y-axis position in dataset
        max_y  = np.float('-inf') #largest y-axis position in dataset

        for n in range(N):
            meshes.append(Mesh(train_dir, shape_filenames[n], number_of_bins))
            number_of_mesh_vertices = meshes[n].V.shape[0] #number of mesh vertices
            mesh_centroid           = np.mean(meshes[n].V, axis=0, keepdims = True)
            meshes[n].V             = meshes[n].V -  mesh_centroid #center mesh at origin
            average_distance_to_centroid_along_y = np.mean(np.abs(meshes[n].V[:,1])) #average vertical distance to centroid
            meshes[n].V             = meshes[n].V/average_distance_to_centroid_along_y #scale meshes
            min_y                   = min(min_y, min(meshes[n].V[:,1]))
            max_y                   = max(max_y, max(meshes[n].V[:,1]))
            print(shape_filenames[n] + " Processed")

        save_data(meshes, min_y, max_y, N, shape_labels, 'tmp_train.p')

    """
    this loop calls your histogram computation code!
    all shape descriptors are organized into a NxD matrix
    N training meshes, D dimensions (number of bins in your histogram)
    """

    X = np.zeros((N, number_of_bins))
    for n in range(N):
        X[n,:] = computeShapeHistogram( meshes[n], min_y, max_y, number_of_bins )
    
    w = .1 * np.random.randn( 1, number_of_bins+1 ) # +1 for bias, random initialization
    
    """""""""""""""""""""""""""""""""""""""
     ADD CODE HERE TO LEARN PARAMETERS w
    """""""""""""""""""""""""""""""""""""""
 
    return w, min_y, max_y


