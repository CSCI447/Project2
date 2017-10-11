from Project2.NeuralNetwork import *
from Project2.NeuralNetwork.FeedForward.FF import NN
import csv

inputArray = []
expectedOutputArray = []
with open('NeuralNetwork/Data/2_dim.csv', 'r', encoding='utf-8') as inputcsvfile:
    csv_input = csv.reader(inputcsvfile, delimiter=",")
    for row in csv_input:
        inputArray.append(row)

with open('NeuralNetwork/Data/2_dim_out.csv', 'r', encoding='utf-8') as outputcsvfile:
    csv_output = csv.reader(outputcsvfile, delimiter=",")
    for row in csv_output:
        expectedOutputArray.append(row)

feedforward = NN(inputArray, expectedOutputArray, 1, 2, 1)

feedforward.initialize()
feedforward.train(5)