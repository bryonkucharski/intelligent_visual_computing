from __future__ import division
import numpy as np
from forwardPropagate import forwardPropagate
from backPropagate import backPropagate
from model import Model
import matplotlib.pyplot as plt
import utils
import math


def returnBatch(X, Y, bs):
    if bs == X.shape[0]: return X, Y
    idx = np.random.randint(0, X.shape[0], bs)
    return X[idx], Y[idx]


def trainNN_batched(X, Y, threshold=0.95, verbose=True, batch_size=32):
    """
    code for learning a neural network
    you need to complete this script!

    Input:
    'X' is your input training data represented as a N x D matrix,
    where N is the number of training examples and D is
    the number of dimensions (number of pixels)

    'Y' represent the training labels (0 or 1 / non-eye or eye respectively)
    and is a N x 1 vector

    'model' is structure defining:
    'model.num_hidden_nodes_per_layer' stores the desired number of nodes per hidden layer
    'model.iterations' stores the number of maximum training iterations to use
    'model.learning_rate' stores the desired step size for gradient descent
    'model.momentum' stores the desired momentum parameter for accelerating gradient descent
    'model.weight_decay' stores the desired L2-norm regularization parameter
     You can change these parameters in the model.py file!

    Output:
    an object of model class storing:
    'model.num_nodes' stores number of nodes per layer (already computed for you, see below)
    'model.num_layers' stores number of nodes per layer (already computed for you, see below)
    'model.param' stores the weights per layer that your code needs to learn
    'model.outputs' stores the output per layer that your code needs to compute for the training data

    feel free to modify the input and output arguments if necessary
    """
    # initialize model

    X = X.copy()
    Y = Y.copy()

    model = Model(num_hidden_nodes_per_layer=[16, 8, 4],
                  iterations=1000,
                  learning_rate=0.0009,
                  momentum=0.99,
                  weight_decay=0.001,
                  verbose=verbose,
                  activation='relu',
                  update_method='GDM')

    N = X.shape[0]  # number of training sample images
    D = X.shape[1]  # number of input features (input nodes) (in this assignment: #pixels)
    M = Y.shape[1]  # number of outputs - for this assignment, we just have a single classification output: prob(eye)

    num_batches = math.ceil(N / batch_size)
    print(N / batch_size, num_batches)
    X_batches = np.array_split(X, num_batches)
    Y_batches = np.array_split(Y, num_batches)

    model.num_nodes = [D] + model.num_hidden_nodes_per_layer + [M]  # including input & output nodes
    model.num_layers = len(model.num_nodes)  # number of layers, including input and output classification layer
    if model.verbose:
        print('Number of layers (including input and output layer): %d' % (model.num_layers))
        print('Number of nodes in each layer will be: %s' % (' '.join(map(str, model.num_nodes))))
        print('Will run for %d iterations' % (model.iterations))
        print('Learning rate: %f, Momentum: %f , Weight decay: %f\n' % (
        model.learning_rate, model.momentum, model.weight_decay))

    # initialize model parameters
    # input layer
    # model.param[0] = .1 * np.random.randn(D, model.num_nodes[0])
    # model.b[0] = np.zeros(model.num_nodes[0])

    # output layer
    # model.param[model.num_layers]= .1 * np.random.randn(model.num_nodes[model.num_layers-1], 1)
    # model.b[model.num_layers] = np.zeros(1)

    # hidden layers
    for layer_id in range(1, model.num_layers):
        model.param[layer_id] = .1 * np.random.randn(model.num_nodes[layer_id - 1],
                                                     model.num_nodes[layer_id])  # plus one unit (+1) for bias
        model.b[layer_id] = np.zeros((1, model.num_nodes[layer_id]))
        model.velocity[layer_id] = np.zeros_like(model.param[layer_id])
        # print(layer_id, model.param[layer_id].shape,model.b[layer_id].shape)
        # model.param[layer_id] = .1 * np.random.randn(model.num_nodes[layer_id-1], model.num_nodes[layer_id])
        # model.b[layer_id] = np.zeros( model.num_nodes[layer_id])
        model.prev_grad[layer_id] = 0

    # normalize (standardize) input, store mean/std in the input layer parameters
    model.mean = np.mean(X, axis=0)
    model.std = np.std(X, axis=0)
    X -= model.mean
    X /= (model.std + 1e-6)  # 1e-6 helps avoid any divisions by 0

    """ ===============================================================
             YOUR CODE goes here - change the following lines
    =============================================================== """

    epoch = 0
    num_epochs = 50
    costs = []
    for i in range(0, num_epochs):
        for j, (x_B, y_b) in enumerate(zip(X_batches, Y_batches)):
            iter = 0
            # complete this loop for learning
            # call forwardPropagate.py, backPropagate.py appropriately, update net parameters

            model.outputs[0] = x_B  # the input layer provides the input data to the net
            Y = y_b

            for layer_id in range(1, model.num_layers):
                is_output_layer = (layer_id == model.num_layers - 1)
                model.outputs[layer_id] = forwardPropagate(model.outputs[layer_id - 1], model.param[layer_id],
                                                           model.b[layer_id], model.activation, is_output_layer)

            Y_hat = model.outputs[model.num_layers - 1]
            assert Y_hat.shape == Y.shape
            cost_function = utils.calculate_cost(Y, Y_hat)

            for layer_id in reversed(range(1, model.num_layers)):
                cost_function += (model.weight_decay * np.sum(
                    model.param[layer_id] * model.param[layer_id]))  # add l2 penalty to cost
                if layer_id == model.num_layers - 1:
                    is_output_layer = True
                    message = Y
                else:
                    is_output_layer = False
                    message = dA_prev

                dA_prev, model.prev_grad[layer_id], model.db[layer_id] = backPropagate(message, model.param[layer_id],
                                                                                       model.outputs[layer_id],
                                                                                       model.outputs[layer_id - 1],
                                                                                       model.b[layer_id],
                                                                                       model.weight_decay,
                                                                                       model.activation,
                                                                                       is_output_layer)
                model.prev_grad[layer_id] = model.prev_grad[layer_id] + (
                            model.weight_decay * model.param[layer_id] * 2)  # Add l2 reg

            for layer_id in range(1, model.num_layers):
                if model.update_method == 'GD':
                    model.param[layer_id] = model.param[layer_id] - model.learning_rate * model.prev_grad[layer_id]
                elif model.update_method == 'GDM':
                    model.velocity[layer_id] = model.momentum * model.velocity[layer_id] - model.learning_rate * \
                                               model.prev_grad[layer_id]  # integrate velocity
                    model.param[layer_id] = model.param[layer_id] + model.velocity[layer_id]  # integrate position

                model.b[layer_id] = model.b[layer_id] - model.learning_rate * model.db[layer_id]



            if model.verbose:
                print('Iteration %d/%d, Cost function: %f' % (
                j, num_batches, cost_function))

        if model.verbose:
            Yp = Y_hat
            Yp[Yp >= threshold] = 1
            Yp[Yp < threshold] = 0
            classification_error = np.mean(Y != Yp)
            print('Epoch %d, Cost function: %f, classification error: %f %%' % (
            i, cost_function, classification_error * 100))
            costs.append(cost_function)



            # epoch += 1
            # if iter > model.iterations:
            # break
    plt.plot(costs)
    plt.show()
    print("Model Trained")
    return model

