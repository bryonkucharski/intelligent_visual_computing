from collections import defaultdict

class Model(object):
    def __init__(self, num_hidden_nodes_per_layer = [16, 8, 4], iterations = 1000, learning_rate = 0.5, momentum = 0.9, weight_decay = 0.00,\
                activation = "sigmoid", verbose = True,update_method = 'GD'):
        self.num_hidden_nodes_per_layer = num_hidden_nodes_per_layer
        self.iterations    = iterations
        self.learning_rate = learning_rate
        self.momentum      = momentum
        self.weight_decay  = weight_decay
        self.param         = defaultdict(float)
        self.outputs       = defaultdict(float)
        self.prev_grad     = defaultdict(float)
        self.b             = defaultdict(float)
        self.db            = defaultdict(float)
        self.velocity      = defaultdict(float)
        self.mean          = 0
        self.std           = 1
        self.activation    = activation
        self.verbose       = verbose
        self.update_method = update_method