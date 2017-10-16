import random

import math

from Project2.NeuralNetwork.RBF.K_Means import K_Means

from Project2.NeuralNetwork.RBF.CentroidCreator import Centroids

from Project2.NeuralNetwork.Neuron import Neuron

from Project2.NeuralNetwork.Connection import Connection


class NN:

    def __init__(self, input_values, expected_output_values, input_nodes_amount,hidden_layer_amount, gaussian_amount, output_nodes_amount, learnrate = 0.0001, momentum = 0.5):
        self.input_values = input_values
        self.expected_output_values = expected_output_values
        self.hidden_layers_amount = 1
        self.input_nodes_amount = input_nodes_amount
        self.output_nodes_amount = output_nodes_amount
        self.learnRate = learnrate
        self.momentum = momentum
        self.inputNodes = []
        self.hiddenNodes = []
        self.outputNodes = []
        self.centroids = []
        self.beta = 0.05
        self.dim = len(input_values[0])
        self.gaussian_amount = 0
        self.centroids = self.calculate_centroids(self.dim)
        self.converged = False
        self.connections = []
        self.inputConnections = []
        self.network = []

    # Initialize a network
    def initialize(self):
        self.build_network()
        self.connect_network()
        self.initialize_weights()


    # Build network structure (layers and nodes)
    def build_network(self):
        for input_neuron in range(self.input_nodes_amount):
            n = Neuron()
            self.inputNodes.append(n)

        for hidden_neuron in range(self.hidden_layers_amount):
            #hidden_nodes = []
            for x in range(self.gaussian_amount):  # currently only 1 hidden layer
                n = Neuron()
                n.centroid = self.centroids[x]
                #n.beta = self.betas[x]
                #hidden_nodes.append(n)
                self.hiddenNodes.append(n)
            self.network.append(self.hiddenNodes)  # confirmed correct number of nodes in network

        for output_neuron in range(self.output_nodes_amount):
            n = Neuron()
            self.outputNodes.append(n)
            # self.network.append(self.outputNodes)

    # Connect the above structure by linking the nodes in the layers correctly
    def connect_network(self):
        for neuron in self.hiddenNodes:
            for n in self.inputNodes:
                c = Connection()
                c.setFromNeuron(n)
                c.setToNeuron(neuron)
                self.inputConnections.append(c)

        for neuron in self.outputNodes:   #for nodes in the output nodes
            for n in self.hiddenNodes:
                c = Connection()
                c.setFromNeuron(n)
                c.setToNeuron(neuron)
                self.connections.append(c)

    #initialize weights for only the connections between the hidden layer and the output layer
    def initialize_weights(self):
        for i in self.inputConnections:   # no weights between input and hidden layer in rbf, so initialize to 1
            i.setWeight(1.0)
        for c in self.connections:
            rand = random.uniform(0, 0.5)  #float between 0.0 and 0.5
            c.setWeight(rand)

    #calculate centroids evenly spced between -3 and 3.  You can adjust the amount of points you want in each dimension
    def calculate_centroids(self,dim):
        self.gaussian_amount = Centroids(dim).k
        print(self.gaussian_amount)
        return Centroids(dim).get_centroids()

    #activate hidden layer and produce weighted sum as output
    def forward_prop(self, row):
        value = []
        for neuron in self.inputNodes:  # set values from input nodes
            for i in row:
                value.append(int(i))
            neuron.setValue(value)

        for neuron in self.hiddenNodes:  #hidden layer activation
            value = self.inputNodes[0].getValue()
            phi = neuron.gauusian(value, neuron.centroid, self.beta) # activation with gaussian function
            neuron.setValue(phi)

        for neuron in self.outputNodes:  #calculate weighted sum
            values = []
            weights = []
            for connection in self.connections:
                if connection.getToNeuron() == neuron:
                    values.append(connection.getFromNeuron().getValue())
                    weights.append(connection.getWeight())
            activation = neuron.outputActivate(weights,values)
            neuron.setValue(neuron.linear(activation))

        return self.outputNodes[0].getValue()  #weighted sum (linear output)

    #calcualte error before weights are updated
    def update_error_output(self, out_row):
        for neuron in self.outputNodes:  # for every neuron in the outputNodes
            unprocessed_error = int(self.expected_output_values[out_row][0]) - neuron.getValue()  # get error w/o derivative
            error_w_pd = unprocessed_error  # get error w derivative (no derivative since it's a linear output)
            neuron.setError(error_w_pd)  # set as error

    #update weights according to error, momentum, and learning rate
    def update_weights_output(self):
        for neuron in self.outputNodes:  # for all neurons in outputNodes
            for connection in self.connections:  # for all connections to that neuron
                new_weight = 0
                if connection.getToNeuron() == neuron:
                    new_weight = connection.getWeight() + ((1 - self.momentum) * self.learnRate * connection.getToNeuron().getError() * connection.getFromNeuron().getValue())  # set weight like the function we talked about
                    connection.setWeight(new_weight)

    #update errors and weights
    def update(self, out_row):
        self.update_error_output(out_row)
        self.update_weights_output()

    # train a neural network for a certain number of epochs
    def train(self, epochs):
        outfile = open("out_3_5_smaller_dim.txt", "a")
        for epoch in range(epochs):
            sum_error = 0
            for i, row in enumerate(self.input_values):
                output_values = self.forward_prop(row)
                #old_error = sum_error
                sum_error += (int(self.expected_output_values[i][0]) - output_values)
                #if math.fabs(old_error - sum_error) < 100:  #the shift in errors is pretty minimal (in relation to the size of our very large errors)
                 #   return #terminate
                self.update(i)
            print(epoch)
            outfile.write('>epoch=%d, lrate=%.2f, error=%.3f' % (epoch, self.learnRate, sum_error))
            outfile.write('\n')
            self.learnRate = self.learnRate + 0.001  #learning rate annealing


    #test the network with a set of inputs after training is complete
    def test(self,input_values):
        outfile = open("out_4_6_dim.txt", "w")
        sum_error = 0
        for i, row in enumerate(self.input_values):
            output_values = self.forward_prop(row)
            sum_error += (int(self.expected_output_values[i][0]) - output_values)
        outfile.write('>lrate=%.3f, error=%.3f' % (self.learnRate, sum_error))
        outfile.write('\n')
        return

