import random
import math
from math import exp

from .. Connection import Connection

from .. Neuron import Neuron


class NN:
    def __init__(self, input_values, expected_output_values, hidden_layers_amount, hidden_nodes_amount, output_nodes_amount, learnrate=0.01, threshold=1, momentum=0.5, maximum=0, minimum=1000):
        self.input_values = input_values  # values from training data
        self.test_values = []
        self.train_values = []
        self.cross_valid_inputs = []
        self.rand_selections = []
        self.hidden_layers_amount = hidden_layers_amount  # number of hidden layers
        self.hidden_nodes_amount = hidden_nodes_amount  # number of nodes in hidden layer
        self.expected_output_values = expected_output_values  # expected output for error checking
        self.output_nodes_amount = output_nodes_amount  # number of output nodes
        self.learnRate = learnrate  # learnrate duh
        self.threshold = 0.01 * threshold  # ?
        self.momentum = momentum  # momentum function
        self.maximum = maximum  # ?
        self.minimum = minimum  # ?
        self.inputNodes = []  # list of nodes in the input layer
        self.hiddenLayers = []  # list of hidden Layers (because this can be 0 to 2), each of these lists contains a list of hiddenNodes (see build_network)
        self.hiddenLayers2 = []
        self.outputNodes = []  # list of output nodes
        self.network = []  # list of ALL the layers in the network
        self.connections = []
        self.converged = True  # ?

    def set_rand_validate(self):
    #     # self.test_values = []
    #     # self.train_values = []
    #     # i = i % len(self.input_values)
    #     # i += 10
    #     # print("enter")
    #     # for i in range(int(len(self.input_values))):
    #     #     print(i)
    #     #     i = i % len(self.input_values)
    #     #     if (i < int(len(self.input_values) * .1)):
    #     #         self.test_values.append(self.input_values[i])
    #     #     else:
    #     #         self.train_values.append(self.input_values[i])
    #     #
    #     # return i
    #
        #current_train_values = self.input_values
        # print(int(len(self.input_values)))
        # print(len(self.input_values))
        self.cross_valid_inputs[:] = []
        self.rand_selections[:] = []
        self.train_values[:] = []
        self.test_values[:] = []

        for n in self.input_values:
            self.cross_valid_inputs.append(n)
        #self.cross_valid_inputs = self.input_values
        for i in range(int(len(self.input_values) * .1)):
            rand = int((random.random() * 90))
            self.test_values.append(self.cross_valid_inputs[rand])
            self.cross_valid_inputs.pop(rand)
            self.rand_selections.append(rand)
        self.train_values = self.cross_valid_inputs


        # print(len(self.cross_valid_inputs))
        # print(self.test_values)
        # print(len(self.train_values))


    # Build network structure (layers and nodes)
    def build_network(self):
        for input_neuron in range(len(self.input_values[0])):               #added [0] because self.input_values = 100, not 2
            n = Neuron()
            self.inputNodes.append(n)

        for hidden_neuron in range(self.hidden_layers_amount):
            hidden_nodes = []
            hidden_nodes2 = []
            for x in range(self.hidden_nodes_amount):                       #currently only 1 hidden layer
                n = Neuron()
                hidden_nodes.append(n)
                self.hiddenLayers.append(n)                                 #TODO only appending 1 node at a time, may need to be changed
            self.network.append(hidden_nodes)                               #confirmed correct number of nodes in network
            for x in range(self.hidden_nodes_amount):
                n = Neuron()
                hidden_nodes2.append(n)
                self.hiddenLayers2.append(n)
            self.network.append(hidden_nodes2)

        for output_neuron in range(self.output_nodes_amount):
                n = Neuron()
                self.outputNodes.append(n)
            #self.network.append(self.outputNodes)

    # Connect the above structure by linking the nodes in the layers correctly
    def connect_network(self):
        # for x in self.inputNodes:
        #     x.setOutputNodes(self.hiddenNodes)  #prob come back and init more stuff
        if(self.hidden_layers_amount == 0):
            for neuron in self.outputNodes:
                #outputConnections = []
                for n in self.inputNodes:
                    c = Connection()
                    c.setFromNeuron(n)
                    c.setToNeuron(neuron)
                    self.connections.append(c)

        if(self.hidden_layers_amount == 1):
            for neuron in self.hiddenLayers:                             #Maybe would work with self.network[0]
                for n in self.inputNodes:
                    c = Connection()
                    c.setFromNeuron(n)
                    c.setToNeuron(neuron)                                #TODO very much only for 1 hidden layer!  Needs to be reworked in a big way!!!!
                    self.connections.append(c)

            for neuron in self.outputNodes:
                for n in self.hiddenLayers:
                    c = Connection()
                    c.setFromNeuron(n)
                    c.setToNeuron(neuron)
                    self.connections.append(c)

        if (self.hidden_layers_amount == 2):
            for neuron in self.hiddenLayers:                             #Maybe would work with self.network[0]
                for n in self.inputNodes:
                    c = Connection()
                    c.setFromNeuron(n)
                    c.setToNeuron(neuron)                                #TODO very much only for 1 hidden layer!  Needs to be reworked in a big way!!!!
                    self.connections.append(c)

            for neuron in self.hiddenLayers2:                             #Maybe would work with self.network[0]
                for n in self.hiddenLayers:
                    c = Connection()
                    c.setFromNeuron(n)
                    c.setToNeuron(neuron)                                #TODO very much only for 1 hidden layer!  Needs to be reworked in a big way!!!!
                    self.connections.append(c)

            for neuron in self.outputNodes:
                for n in self.hiddenLayers2:
                    c = Connection()
                    c.setFromNeuron(n)
                    c.setToNeuron(neuron)
                    self.connections.append(c)

    # Initialize weights for all layers except input layer (because it doesn't connect back to anything, and thus has no weight)
    def initialize_weights(self):
        for c in self.connections:
            rand = random.random()
            c.setWeight(rand)

    # Sets value of each node to weighted sum of connections to neurons in the layer above, processed with sigmoid funciton
    def feedforward(self, row, indicator):
        k = 0
        if (indicator == 0):
            for neuron in self.inputNodes:          #set values from input nodes
                neuron.setValue(float(self.train_values[row][k]))
                k += 1
        else:
            for neuron in self.inputNodes:          #set values from input nodes
                neuron.setValue(float(self.test_values[row][k]))
                k += 1

        if (self.hidden_layers_amount == 0):
            for neuron in self.outputNodes:
                weights2 = []
                values2 = []
                for connection in self.connections:
                    if connection.getToNeuron() == neuron:
                        weights2.append(connection.getWeight())
                        values2.append(connection.getFromNeuron().getValue())
                activation = neuron.activate(weights2, values2)
                neuron.setValue(neuron.linear(activation))

        if (self.hidden_layers_amount == 1):
            for neuron in self.hiddenLayers:
                weights = []
                values = []
                for connection in self.connections:                                 #iterate to get wieghts and values=
                    if connection.getToNeuron() == neuron:
                        weights.append(connection.getWeight())
                        values.append(connection.getFromNeuron().getValue())
                activation = neuron.activate(weights, values)
                neuron.setValue(neuron.sigmoid(activation))  # perform sigmoid on activation summation

            for neuron in self.outputNodes:
                weights2 = []
                values2 = []
                for connection in self.connections:
                    if connection.getToNeuron() == neuron:
                        weights2.append(connection.getWeight())
                        values2.append(connection.getFromNeuron().getValue())
                activation = neuron.activate(weights2, values2)
                neuron.setValue(neuron.linear(activation))                                 #TODO all sigmoid baby, update when ready

        if (self.hidden_layers_amount == 2):
            for neuron in self.hiddenLayers:
                weights = []
                values = []
                for connection in self.connections:                                 #iterate to get wieghts and values=
                    if connection.getToNeuron() == neuron:
                        weights.append(connection.getWeight())
                        values.append(connection.getFromNeuron().getValue())
                activation = neuron.activate(weights, values)
                neuron.setValue(neuron.sigmoid(activation))  # perform sigmoid on activation summation

            for neuron in self.hiddenLayers2:
                weights = []
                values = []
                for connection in self.connections:                                 #iterate to get wieghts and values=
                    if connection.getToNeuron() == neuron:
                        weights.append(connection.getWeight())
                        values.append(connection.getFromNeuron().getValue())
                activation = neuron.activate(weights, values)
                neuron.setValue(neuron.sigmoid(activation))  # perform sigmoid on activation summation

            for neuron in self.outputNodes:
                weights2 = []
                values2 = []
                for connection in self.connections:
                    if connection.getToNeuron() == neuron:
                        weights2.append(connection.getWeight())
                        values2.append(connection.getFromNeuron().getValue())
                activation = neuron.activate(weights2, values2)
                neuron.setValue(neuron.linear(activation))

        return self.outputNodes[0].getValue()

    # Determines errors of neurons, then update weights based on these errors
    def backprop(self, out_row):
        self.update_error_output(out_row)
        self.update_weights_output()

        if (self.hidden_layers_amount > 0):
            self.update_error_hidden()
            self.update_weights_hidden()

    def update_error_output(self, out_row):
        for neuron in self.outputNodes:  # for every neuron in the outputNodes
            unprocessed_error = int(self.expected_output_values[out_row][0]) - neuron.getValue()  # get error w/o derivative
            error_w_pd = unprocessed_error * neuron.linear_derivative(neuron.getValue())  # get error w derivative
            neuron.setError(error_w_pd)  # set as error

    def update_weights_output(self):
        for neuron in self.outputNodes:  # for all neurons in outputNodes
            for connection in self.connections:  # for all connections to that neuron
                if connection.getToNeuron() == neuron:
                    new_weight = connection.getWeight() + (self.learnRate * connection.getToNeuron().getError() * connection.getFromNeuron().getValue())  # set weight like the function we talked about
                    connection.setWeight(new_weight)

    def update_error_hidden(self):
        # hidden layer error = (weight_k * error_j) * transfer_derivative(output)

        if (self.hidden_layers_amount == 2):
            for neuron in self.hiddenLayers2:  # for every neuron in hidden nodes
                error_w_pd = 0
                weight = 0.0
                for connect in self.connections:  # for all the connections to that neuron
                    if connect.getFromNeuron() == neuron:
                        weight = connect.getWeight()
                        error = connect.getToNeuron().getError()
                        unprocessed_error = error * weight  # start at first hidden node, iterate over connections, sum with error and connection weights
                        error_w_pd += unprocessed_error * neuron.transfer_derivative(neuron.getValue())
                neuron.setError(error_w_pd)
                self.update_weights_hidden2()

        for neuron in self.hiddenLayers:  # for every neuron in hidden nodes
            error_w_pd = 0
            #update errors for each connection
            weight = 0.0
            for connect in self.connections:  # for all the connections to that neuron
                if connect.getFromNeuron() == neuron:
                    weight = connect.getWeight()
                    error = connect.getToNeuron().getError()
                    unprocessed_error = error * weight  # start at first hidden node, iterate over connections, sum with error and connection weights
                    error_w_pd += unprocessed_error * neuron.transfer_derivative(neuron.getValue())
            neuron.setError(error_w_pd)

    def update_weights_hidden(self):
        k = 0
        for neuron in self.hiddenLayers:
            for connect in self.connections:  # for all connections to that neuron
                if connect.getToNeuron() == neuron:
                    new_weight = connect.getWeight() + (self.learnRate * connect.getToNeuron().getError() * neuron.getValue())  # set weight like the function we talked about
                    connect.setWeight(new_weight)
            k += 1

    def update_weights_hidden2(self):
        k = 0
        for neuron in self.hiddenLayers2:
            for connect in self.connections:
                if connect.getToNeuron() == neuron:
                    new_weight = connect.getWeight() + (self.learnRate * connect.getToNeuron().getError() * neuron.getValue())  # set weight like the function we talked about
                    connect.setWeight(new_weight)
            k += 1

    def initialize(self):
        self.build_network()
        self.connect_network()
        self.initialize_weights()

    def train(self, epochs):
        print("New Test")
        print("hidden layers = %d, hidden nodes = %d, " % self.hidden_layers_amount, self.hidden_nodes_amount)
        # i = 0
        while (self.converged):
            self.set_rand_validate()
            sum_error = 0
            sum_error2 = 0
            for i, row in enumerate(self.train_values):
                output_values = self.feedforward(i, 0)
                sum_error += (int(self.expected_output_values[i][0]) - output_values)
                self.backprop(i)
            for i, row in enumerate(self.test_values):
                output_values = self.feedforward(i, 1)
                sum_error2 += (int(self.expected_output_values[self.rand_selections[i]][0]) - output_values)
            print('>epoch=%d, lrate=%.2f, training error=%.3f, testing error=%.3f' % (epochs, self.learnRate, sum_error, sum_error2))
            epochs -= 1
            if (sum_error < self.threshold):
                self.converged = False
    # def train(self, epochs):
    #     for epoch in range(epochs):
    #         sum_error = 0
    #         for i, row in enumerate(self.input_values):
    #             output_values = self.feedforward(row)
    #             sum_error += (int(self.expected_output_values[i][0])-output_values)
    #             self.backprop(i)
    #         print('>epoch=%d, lrate=%.2f, error=%.3f' % (epoch, self.learnRate, sum_error))


