import random

import math

from Project2.NeuralNetwork.RBF.K_Means import K_Means

from Project2.NeuralNetwork.Neuron import Neuron

from Project2.NeuralNetwork.Connection import Connection

#IGNORE THIS FILE, this was a first attempt at the rbf
class rbf:

    def __init__(self, input_values, expected_output_values, hidden_layer_amount, gaussian_amount, output_nodes_amount, learnrate = 0.0001, threshold = 1, momentum = 0.5, maximum = 0, minimum = 1000):
        self.input_values = input_values
        self.expected_output_values = expected_output_values
        self.training, self.testing = self.create_io_pairs(self.input_values, self.expected_output_values)
        self.hidden_layers_amount = 1
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
                self.connections.append(c)

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
            c.setPrevWeight(0)

    # runs k-means clustering algorithm and returns k number of clusters and their corresponding centroids and betas
    def get_centroids_and_betas(self,input_values,output_values, k):
        temp_input = []
        for input in range(len(input_values)):
            if input < (len(input_values) * 0.8):
                temp_input.append(input_values[input])
        clusters = K_Means(temp_input,k).get_clusters()
        for i in clusters:
            centroid = i.get_centroid()
            beta = i.get_beta()
            self.centroids.append(centroid)
            self.betas.append(beta)
        return self.centroids, self.betas

    #step forward through the network and activate hidden layer, calculate weighted sum, calculate error, and update the weights until error is within threshold
    def forward_prop(self,outfile,epochs):
        self.error = 0
        self.delta = 0
        self.weighted_sum = 0
        for i in self.training:
            for n in range(self.gaussian_amount):
                value = self.apply_gaussian(i, self.centroids[n],self.betas[n])  # activation with gaussian function
                self.hiddenNodes[n].setValue(value)
            self.weighted_sum += self.calculate_weighted_sum()     #output of the network
            self.error += self.calculate_error(i.output)
        self.error = self.error  *  1/ len(self.training)
        self.error = self.error * (self.weighted_sum * (1.0 - self.weighted_sum))  # transfer derivative
        self.weighted_sum = self.update_weights()
        #self.error = self.calculate_error(i.output)
        #print("\n")
        #print('Predicted = ' + str(self.weighted_sum))
        #print('Actual = ' + str(i.output))
        #self.squared_error = self.squared_error / epochs
        outfile.write("Error = " + str(self.error))
        outfile.write("\n")
        print("Error = " + str(self.error))
        self.delta = self.delta / epochs
        print("DELTA:" + str(self.delta))
        outfile.write("DELTA:" + str(self.delta))
        outfile.write("\n")
        return self.error

    #euclidean distance between a given x and a given centroid
    def calculate_distance(self, x, mu):                                                      #euclidean distance between two n-dimensional points
        difference = 0.0
        for i in range(mu.n):
            squareDifference = pow(((x.input[i]) - mu.coords[i]), 2)
            difference += squareDifference
        distance = math.sqrt(difference)
        return distance

    #activate each hidden node with the gaussian function
    def apply_gaussian(self,x,mu,beta):
        phi = math.exp(-beta * math.pow(self.calculate_distance(x,mu),2))
        return phi

    #sum the weights on each connection between the hidden layer and the output layer
    def calculate_weighted_sum(self):
        weighted_sum = 0
        for i in self.hiddenNodes:
            value = i.getValue()
            weights = i.getWeights()
            for j in range(len(weights)):
                weighted_sum += (value * weights[j])
        self.weighted_sum = weighted_sum + self.bias
        return self.weighted_sum

    #calculate squared error
    def calculate_error(self, input):
        self.error = input - self.weighted_sum

        return self.error

    #update weights based on error, learning rate, and momentum
    def update_weights(self):
        for neuron in self.hiddenNodes:
            temp = neuron.getWeights()[0]
            weight = neuron.getWeights()[0]
            prev_weight = neuron.getPrevWeight()[0]
            value = neuron.value
            weight = -self.learnRate     #weight + (self.learnRate * self.error * self.weighted_sum) #+ (weight - prev_weight)
            self.delta += math.fabs(weight - prev_weight)
            neuron.setWeight(weight)
            neuron.setPrevWeight(temp)
        self.weighted_sum = self.calculate_weighted_sum()
        return self.weighted_sum

    #train the model
    def train(self,outfile,epochs):
        for j in range(epochs):
            self.forward_prop(outfile,epochs)
        print('error=%.3f' % (self.error))

    #test the network
    def test(NN,testing_set):
        for example in testing_set:
            for mu in range(NN.gaussian_amount):
                value = NN.apply_gaussian(example, NN.centroids[mu], NN.betas[mu])
                NN.hiddenNodes[mu].setValue(value)
            NN.calculate_weighted_sum()
            NN.calculate_error(example)
            print('Predicted = ' + str(NN.weighted_sum))
            print('Actual = ' + str(example.output))
            print("Error = " + str(NN.error))
        return

 #  def main(self):
  #      outfile = open("out.txt", 'w')
   #     self.train(outfile,10)
        #self.test(self.testing)

 #   if __name__ == '__main__':
  #        main()

#bind the inputs with corresponding output from the rosenbrock function
class Example():
    def __init__(self, input, output):
        self.input = self.create_int_array(input)
        self.output = int(output)
        self.length = len(input)

    # convert string to list of integers
    def create_int_array(self,input):
        coordinate_list =[]
        for i in range(len(input)):
            coordinate = int(input[i])
            coordinate_list.append(coordinate)
        return coordinate_list
