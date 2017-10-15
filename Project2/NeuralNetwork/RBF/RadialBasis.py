import random

import math

from Project2.NeuralNetwork.RBF.K_Means import K_Means

from Project2.NeuralNetwork.Neuron import Neuron

from Project2.NeuralNetwork.Connection import Connection


class NN:

    def __init__(self, input_values, expected_output_values, input_nodes_amount,hidden_layer_amount, gaussian_amount, output_nodes_amount, learnrate = .2, threshold = 1, momentum = 0.5, maximum = 0, minimum = 1000):
        self.input_values = input_values
        self.expected_output_values = expected_output_values
        self.training, self.testing = self.create_io_pairs(self.input_values, self.expected_output_values)
        self.hidden_layers_amount = 1
        self.input_nodes_amount = input_nodes_amount
        self.gaussian_amount = gaussian_amount
        self.output_nodes_amount = output_nodes_amount
        #self.answerSet = answers
        self.learnRate = learnrate
        self.threshold = 0.1 * threshold
        self.momentum = momentum
        self.maximum = maximum
        self.minimum = minimum
        self.inputNodes = []
        self.hiddenNodes = []
        self.outputNodes = []
        self.centroids = []
        self.betas = []
        self.centroids, self.betas = self.get_centroids_and_betas(input_values, expected_output_values, self.gaussian_amount)
        self.converged = False
        #self.bias = 1.0
        self.connections = []
        self.inputConnections = []
        self.network = []
        #self.network = self.initialize()
        #self.weighted_sum = 0
        #self.error = 0.0
        #self.delta = 0

    def create_io_pairs(self,input,output):
        self.training = []
        self.testing = []
        partition = len(input) * 0.8
        for i in range(len(input)):
            if i < partition:
                ex = Example(input[i],output[i][0])
                self.training.append(ex)
            elif i >= partition:
                ex = Example(input[i], output[i][0])
                self.testing.append(ex)
        return self.training, self.testing

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
                n.beta = self.betas[x]
                #hidden_nodes.append(n)
                self.hiddenNodes.append(n)
            self.network.append(self.hiddenNodes)  # confirmed correct number of nodes in network

        for output_neuron in range(self.output_nodes_amount):
            n = Neuron()
            self.outputNodes.append(n)
            # self.network.append(self.outputNodes)

    # Connect the above structure by linking the nodes in the layers correctly
    def connect_network(self):
        #for x in self.inputNodes:
             #x.setOutputNodes(self.hiddenNodes)  #prob come back and init more stuff
        for neuron in self.hiddenNodes:  # Maybe would work with self.network[0]
            for n in self.inputNodes:
                c = Connection()
                c.setFromNeuron(n)
                c.setToNeuron(neuron)
                self.inputConnections.append(c)
                #neuron.setFromConnections(connections)

                #neuron.setOutputNodes(self.outputNodes)
                #neuron.setInputNodes(self.inputNodes)
                #neuron.setNodeWeightsLength(len(neuron.getInputNodes))

        for neuron in self.outputNodes:
            # outputConnections = []
            for n in self.hiddenNodes:
                c = Connection()
                c.setFromNeuron(n)
                c.setToNeuron(neuron)
                self.connections.append(c)
                # neuron.setFromConnections(outputConnections)
                # neuron.setInputNodes(self.hiddenNodes)
                # neuron.setNodeWeightsLength(len(neuron.getInputNodes))

    #initialize weights for only the connections between the hidden layer and the output layer
    def initialize_weights(self):
        for i in self.inputConnections:   # no weights between input and hidden layer in rbf
            i.setWeight(1.0)
        for c in self.connections:
            rand = random.uniform(0, 0.5)
            c.setWeight(rand)

    # runs k-means clustering algorithm and returns k number of clusters and their corresponding centroids and betas
    def get_centroids_and_betas(self,input_values,output_values, k):
        temp_input = []
        for input in range(len(input_values)):
            if input < int(len(input_values) * 0.8):
                temp_input.append(input_values[input])
        clusters = K_Means(temp_input,k).get_clusters()
        for i in clusters:
            centroid = i.get_centroid()
            beta = i.get_beta()
            self.centroids.append(centroid)
            self.betas.append(beta)
        return self.centroids, self.betas

    # euclidean distance between a given x and a given centroid
    def calculate_distance(self, x, mu):  # euclidean distance between two n-dimensional points
        difference = 0.0
        for i in range(mu.n):
            squareDifference = pow(((x.input[i]) - mu.coords[i]), 2)
            difference += squareDifference
        distance = math.sqrt(difference)
        return distance

    def forward_prop(self, row):
        value = []
        for neuron in self.inputNodes:  # set values from input nodes
            for i in row:
                value.append(int(i))
            neuron.setValue(value)
            #print(neuron.getValue())

        #for neuron in self.hiddenNodes:
          #  weights = []
          #  values = []
          #  for connection in self.connections:  # iterate to get wieghts and values=
           #     if connection.getToNeuron() == neuron:
            #        weights.append(connection.getWeight())
             #       # print(connection.getWeight())
              #      values.append(connection.getFromNeuron().getValue())
                    # print(connection.getFromNeuron().getValue())

        for neuron in self.hiddenNodes:
            value = self.inputNodes[0].getValue()
            #print(value)
            phi = neuron.gauusian(value, neuron.centroid, neuron.beta) # activation with gaussian function
            neuron.setValue(phi)
            #print("hidden node value", neuron.getValue())

        for neuron in self.outputNodes:
            values = []
            weights = []
            for connection in self.connections:
                if connection.getToNeuron() == neuron:
                    values.append(connection.getFromNeuron().getValue())
                    #print("hidden node value", connection.getFromNeuron().getValue())
                    weights.append(connection.getWeight())
                    #print("output connection weight", connection.getWeight())
            activation = neuron.outputActivate(weights,values)
            #print("output activation", activation)
            neuron.setValue(neuron.linear(activation))
            #print("output node value", neuron.getValue())

        return self.outputNodes[0].getValue()

    def update_error_output(self, out_row):
        for neuron in self.outputNodes:  # for every neuron in the outputNodes
            unprocessed_error = int(self.expected_output_values[out_row][0]) - neuron.getValue()  # get error w/o derivative
            #print("output unprocessed output error = %f" % unprocessed_error)
            #print("output node value: ", neuron.getValue())
            error_w_pd = unprocessed_error #* neuron.linear_derivative()  # get error w derivative
            #print("transfer derivative: ", neuron.linear_derivative())
            #print("output error = %f" % error_w_pd)
            neuron.setError(error_w_pd)  # set as error

    def update_weights_output(self):
        for neuron in self.outputNodes:  # for all neurons in outputNodes
            for connection in self.connections:  # for all connections to that neuron
                new_weight = 0
                if connection.getToNeuron() == neuron:
                    #print("update output:")
                    # print(connection.getWeight())
                    # print('learn_rate=%f' % (self.learnRate))
                    # print('error=%f' % (connection.getToNeuron().getError()))
                    #print('phi value=%f' % (connection.getFromNeuron().getValue()))
                    new_weight = connection.getWeight() + (self.learnRate * connection.getToNeuron().getError() * connection.getFromNeuron().getValue())  # set weight like the function we talked about
                    # print(new_weight)
                    connection.setWeight(new_weight)
                    # print(connection.getWeight())

    def update(self, out_row):
        self.update_error_output(out_row)
        self.update_weights_output()

    # train a neural network for a certain number of epochs
    def train(self, epochs):
        for epoch in range(epochs):
            # while not(self.converged):
            sum_error = 0
            for i, row in enumerate(self.input_values):
                #print("row:", i)
                output_values = self.forward_prop(row)
                sum_error += (int(self.expected_output_values[i][0]) - output_values)
                self.update(i)
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, self.learnRate, sum_error))


    def test(self):
        return


# bind the inputs with corresponding output from the rosenbrock function
class Example():
    def __init__(self, input, output):
        self.input = self.create_int_array(input)
        self.output = int(output)
        self.length = len(input)

    # convert string to list of integers
    def create_int_array(self, input):
        coordinate_list = []
        for i in range(len(input)):
            coordinate = int(input[i])
            coordinate_list.append(coordinate)
        return coordinate_list

