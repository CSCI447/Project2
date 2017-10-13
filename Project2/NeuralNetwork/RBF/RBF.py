import random

import math

from Project2.NeuralNetwork.RBF.K_Means import K_Means

from Project2.NeuralNetwork.Neuron import Neuron

from Project2.NeuralNetwork.Connection import Connection


class NN:

    def __init__(self, input_values, expected_output_values, gaussian_amount, output_nodes_amount, learnrate = 0.3, threshold = 1, momentum = 0.5, maximum = 0, minimum = 1000):
        self.input_values = input_values
        self.expected_output_values = expected_output_values
        self.training, self.testing = self.create_io_pairs(self.input_values, self.expected_output_values)
        self.hidden_layers_amount = 1
        self.gaussian_amount = gaussian_amount
        self.output_nodes_amount = output_nodes_amount
        #self.answerSet = answers
        self.learnRate = learnrate
        self.threshold = 0.01 * threshold
        self.momentum = momentum
        self.maximum = maximum
        self.minimum = minimum
        self.inputNodes = []
        self.hiddenNodes = []
        self.outputNodes = []
        self.centroids = self.get_centroids(input_values, self.gaussian_amount)
        self.converged = False
        self.beta = 1
        self.bias = 1.0
        self.connections = []
        self.network = []
        self.network = self.initialize()
        self.weighted_sum = 0
        self.squared_error = 0

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
        return self.network

    # Build network structure (layers and nodes)
    def build_network(self):
        for x in self.training:
            n = Neuron()
            self.inputNodes.append(n)

        for x in range(self.gaussian_amount):
            n = Neuron()
            self.hiddenNodes.append(n)
        self.network.append(self.hiddenNodes)

        for x in range(self.output_nodes_amount):
            n = Neuron()
            self.outputNodes.append(n)
        self.network.append(self.outputNodes)

    # Connect the above structure by linking the nodes in the layers correctly
    def connect_network(self):

        for neuron in self.hiddenNodes:
            connections = []
            for n in self.inputNodes:
                c = Connection()
                c.setFromNeuron(n)
                c.setToNeuron(neuron)
                connections.append(c)

        for neuron in self.outputNodes:
            connections = []
            for n in self.hiddenNodes:
                c = Connection()
                c.setFromNeuron(n)
                c.setToNeuron(neuron)
                #neuron.setConnections(c)
                n.setConnections(c)
                self.connections.append(c)

    #initialize weights for only the connections between the hidden layer and the output layer
    def initialize_weights(self):
        for c in self.connections:
            c.setWeight(random.random())

    # runs k-means clustering algorithm and returns k number of clusters and their corresponding centroids
    def get_centroids(self,input_values,k):
        self.centroids = K_Means(input_values,k).get_centroids()
        return self.centroids

    #step forward through the network and activate hidden layer, calculate weighted sum, calculate error, and update the weights until error is within threshold
    def forward_prop(self,epochs):
        for i in self.training:
            for n in range(self.gaussian_amount):
                value = self.apply_gaussian(i,self.centroids[n]) #activation with gaussian function
                self.hiddenNodes[n].setValue(value)
            self.calculate_weighted_sum()
            #while self.calculate_error(i) > self.threshold:
            self.update_weights()
            print('Predicted = ' + str(self.weighted_sum))
            print('Actual = ' + str(i.output))

    #euclidean distance between a given x and a given centroid
    def calculate_distance(self, x, mu):                                                      #euclidean distance between two n-dimensional points
        difference = 0.0
        for i in range(mu.n):
            squareDifference = pow(((x.input[i]) - mu.coords[i]), 2)
            difference += squareDifference
        distance = math.sqrt(difference)
        return distance

    #activate each hidden node with the gaussian function
    def apply_gaussian(self,x,mu):
        phi = math.exp(-self.beta * math.pow(self.calculate_distance(x,mu),2))
        return phi

    #sum the weights on each connection between the hidden layer and the output layer
    def calculate_weighted_sum(self):
        weighted_sum = 0
        for i in self.hiddenNodes:
            value = i.value
            weights = i.getWeights()
            for j in range(len(weights)):
                weighted_sum += (value * weights[j])
        self.weighted_sum = weighted_sum
        return self.weighted_sum

    #calculate squared error
    def calculate_error(self,input):
        self.squared_error = input.output - self.weighted_sum
        print(self.squared_error)
        return self.squared_error

    #update weights based on error and learning rate
    def update_weights(self):
        for neuron in self.outputNodes:
            for c in neuron.getConnections():
                old_weight = c.getWeight()
                value = c.getFromNeuron().value
                weight = self.learnRate * self.squared_error * value * old_weight
                c.setWeight(weight)
        self.weighted_sum = self.calculate_weighted_sum()
        return

    #train the model
    def train(self,epochs):
        for epoch in range(epochs):
            self.forward_prop(epochs)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, self.learnRate, self.squared_error))

    #test the network
    def test(NN,testing_set):
        for example in testing_set:
            for mu in range(NN.gaussian_amount):
                value = NN.apply_gaussian(example, NN.centroids[mu])
                NN.hiddenNodes[mu].setValue(value)
            NN.calculate_weighted_sum()
            NN.calculate_error(example)
            print('Predicted = ' + str(NN.weighted_sum))
            print('Actual = ' + str(example.output))
            print("Error = " + str(NN.squared_error))
        return

    def main(self):
        self.train(10)
        self.test(self.testing)

    if __name__ == '__main__':
          main()

#bind the inputs with corresponding output from the rosenbrock function
class Example():
    def __init__(self, input, output):
        self.input = self.create_int_array(input)
        self.output = int(output)

    # convert string to list of integers
    def create_int_array(self,input):
        coordinate_list =[]
        for i in range(len(input)):
            coordinate = int(input[i])
            coordinate_list.append(coordinate)
        return coordinate_list
