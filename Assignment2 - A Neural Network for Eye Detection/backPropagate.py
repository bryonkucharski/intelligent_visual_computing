from __future__ import division
import numpy as np
import utils


def backPropagate(received_msg, layer_param, layer_output, layer_input,layer_b, weight_decay, activation, is_output_layer):
    """
    code for back propagation through a sigmoid layer
    you need to complete this script!

    Input:
    'received_msg': are the messages sent to this layer from another layer in the net
    'layer_param' represents the parameters of the layer used to transform the input 
    data (see trainNN.py, forwardPropagate.py)
    'layer_output' represents the output data produced by this layer during
    forward propagation (N x Do matrix, where Do is the number of output nodes of this layer)
    'layer_input' represents the input data to this layer during during 
    forward propagation  (N x Di matrix, where N is the number of input samples, 
    Di is the number of dimensions)
    'weight_decay' is the L2 regularization parameter
    'is_output_layer' is a binary variable which should be true if this layer is the
    output classification layer, and false otherwise

    Output:
    'derivatives' should store the derivatives of the parameters of this layer
    'sent_msg' should store the messages emitted by this layer 

    feel free to modify the input and output arguments if necessary

    % YOUR CODE goes here - modify the following lines!
    """

    N, Di = layer_input.shape

    Do = layer_output.shape[1]

    sent_msg = np.zeros(shape=(N, Di))

    derivatives = np.zeros(shape=(Di, Do))
    Z = np.add(np.dot(layer_input, layer_param), layer_b)

    if is_output_layer:
        dY_hat = utils.der_cost(received_msg, layer_output)
        dZ = dY_hat * utils.sigmoid_derivative( Z )

    else:
        dA_prev = received_msg
        if activation == 'relu':
            dZ = dA_prev * utils.relu_derivative(Z)  # element wise multiplication
        else:
            dZ = dA_prev * utils.sigmoid_derivative( Z )  # element wise multiplication
    A_prev = layer_input
    dA_prev, dW, db = utils.calculate_backprop(derZ=dZ, A_previous=A_prev, W=layer_param, m=N)

    assert db.shape == (1,layer_param.shape[1])
    return dA_prev, dW, db
