
from NeuralNetwork import *
from Project2.NeuralNetwork.RBF.RadialBasis import NN
import csv
inputArray = []
expectedOutputArray = []

with open('NeuralNetwork/Data/2_dim.csv', 'r', encoding='utf-8') as inputcsvfile:
    csv_input = csv.reader(inputcsvfile, delimiter=",")
    for row in csv_input:
        #row.insert(0,1)
        inputArray.append(row)

with open('NeuralNetwork/Data/2_dim_out.csv', 'r', encoding='utf-8') as outputcsvfile:
    csv_output = csv.reader(outputcsvfile, delimiter=",")
    for row in csv_output:
        expectedOutputArray.append(row)

input_nodes_amount = 1
hidden_layer_amount = 1
hidden_nodes_amount = 15
output_nodes_amount = 1

rbf = NN(inputArray,expectedOutputArray,input_nodes_amount,hidden_layer_amount, hidden_nodes_amount, output_nodes_amount)

rbf.initialize()

rbf.train(50)

