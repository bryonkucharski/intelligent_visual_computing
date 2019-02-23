from __future__ import division
import numpy as np
from forwardPropagate import forwardPropagate

def testNN(model, X, probabilities=False, threshold=0.95):
    """
    code for applying a neural network on new data
    you need to complete this script!

    Input:
    'model' stores the net parameters (including the weights you need here) - see trainNN.py

    'X' is your test data represented as a N x D matrix, 
    where N is the number of test instances and D is
    the number of dimensions (number of pixels)

    'threshold': your model outputs a continuous value 0...1 through the topmost sigmoid
    function. You can threshold it to decide whether you have eye or not
    (it does not need to be 0.5, but something larger depending on the desired
    true positive rate - see ROC curves)

    Output:
    Ypred is your prediction (1 - eye, 0 - non-eye) per training instance
    it should be a N x 1 vector

    feel free to modify the input and output arguments if necessary
    """

    X = X.copy()
    N = X.shape[0]

    # normalize (standardize) input given training mean/std dev.
    X = X - model.mean
    X = X /(model.std + 1e-6) #1e-6 helps avoid any divisions by 0
    
    """ ===============================================================
            YOUR CODE goes here - change the following lines 
    call forwardPropagate appropriately
    use the output probabilities to determine the predictions (eye/not eye)
    =============================================================== """

    model.outputs[0] = X  # the input layer provides the input data to the net
    for layer_id in range(1, model.num_layers):
        #print(model.outputs[layer_id - 1].shape, model.param[layer_id].shape, model.b[layer_id].shape )
        is_output_layer = (layer_id == model.num_layers - 1)
        model.outputs[layer_id] = forwardPropagate(model.outputs[layer_id - 1], model.param[layer_id], model.b[layer_id],model.activation,is_output_layer)

    Ypred = model.outputs[model.num_layers - 1]

    if probabilities:
        return Ypred
    Ypred[Ypred >= threshold] = 1
    Ypred[Ypred < threshold] = 0
    
    return Ypred
