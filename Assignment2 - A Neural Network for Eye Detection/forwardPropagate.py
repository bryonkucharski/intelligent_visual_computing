from __future__ import division
import numpy as np
import utils

def forwardPropagate(layer_input, layer_param, b,activation, output_layer):
    """
    code for forward propagation through a sigmoid layer
    you need to complete this script!

    Input:
    'layer_input' is the input data to the layer (N x Di matrix, where
    N is the number of input samples, Di is the number of dimensions 
    of your input data). 
    'layer_param' represents the parameters of the layer
    used to transform the input data (see trainNN.py)

    Output:
    'layer_output' represents the output data produced by the fully connected, 
    sigmoid layer (N x Do matrix, where Do is the number of output nodes of this layer)

    """

    N = layer_input.shape[0]
    Do = layer_param.shape[1]
    layer_output = np.zeros(shape=(N, Do))
    Z = np.add(np.dot(layer_input,layer_param), b)
    #print(layer_input.shape, layer_param.shape, b.shape,Z.shape)
    #Z = np.dot(layer_input,layer_param)
    if activation == 'relu':
        if output_layer:
            layer_output = utils.sigmoid(Z)
        else:
            layer_output = utils.relu(Z)
    else:
        layer_output = utils.sigmoid(Z)



    assert layer_output.shape == (N,Do)

    return layer_output