import numpy as np

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def init_to_onehot(y, num_classes):
    arr_y = np.zeros((y.shape[0], num_classes))

    for i, val in enumerate(y):
        arr_y[i, val] = 1

    return arr_y


class NeuralNetMLP:

    # init constructor instantiates the weight matrices and bias vectors for the hidden and the output layer
    def __init__(self, num_features, num_hidden, num_classes, random_seed=123):
        super().__init__()

        self.num_classes = num_classes

        # hidden layer
        rng = np.random.RandomState(random_seed)
        print(rng.normal(loc=0.0, scale=0.1, size=(num_hidden, num_features)).shape)
        self.weight_h = rng.normal(loc=0.0, scale=0.1, size=(num_hidden, num_features))
        self.bias_h = np.zeros(num_hidden)

        # output layer
        self.weight_out = rng.normal(loc=0.0, scale=0.1, size=(num_classes, num_hidden))
        self.bias_out = np.zeros(num_classes)


    def forward(self, x):
        # hidden layer

        net_input_hidden_layer = np.dot(x, self.weight_h.T) + self.bias_h
        activation_hidden_layer = sigmoid(net_input_hidden_layer)

        # output layer
        net_input_output_hidden_layer = np.dot(activation_hidden_layer, self.weight_out.T) + self.bias_out
        activation_output_layer = sigmoid(net_input_output_hidden_layer)

        return activation_hidden_layer, activation_output_layer

    def backward(self, x, activation_hidden_layer, activation_output_layer, y):
        # output layer weights

        # one-hot encoding
        y_one_hot = init_to_onehot(y, self.num_classes)

        # Part 1: dLoss/dOutWeights = dLoss/dOutActiv * dLoss/dOutNet * dOutNet/dOutWeight
        d_loss__d_a_out = 2. * (activation_output_layer - y_one_hot) / y.shape[0]
        d_a_out__d_z_out = activation_output_layer * (1. - activation_output_layer) # sigmoid derivate
        delta_out = d_loss__d_a_out * d_a_out__d_z_out

        # gradient for output weights
        d_z_out__dw_out = activation_hidden_layer


        d_loss__dw_out = np.dot(delta_out.T, d_z_out__dw_out)
        d_loss__db_out = np.sum(delta_out, axis=0)

        #################################
        # Part 2: dLoss/dHiddenWeights = DeltaOut * dOutNet/dHiddenAct * dHiddenAct/dHiddenNet * dHiddenNet/dWeight
        d_z_out_a_h = self.weight_out

        d_loss__a_h = np.dot(delta_out, d_z_out_a_h)

        d_a_h__d_z_h = activation_hidden_layer * (1. - activation_hidden_layer) # sigmoid derivative

        d_z_h__d_w_h = x


        d_loss__d_w_h = np.dot((d_loss__a_h * d_a_h__d_z_h).T, d_z_h__d_w_h)
        d_loss_d_b_h = np.sum(d_loss__a_h * d_a_h__d_z_h, axis=0)

        return d_loss__dw_out, d_loss__db_out, d_loss__d_w_h, d_loss_d_b_h