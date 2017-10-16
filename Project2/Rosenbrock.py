# This file contains methods to generate input and expected output data for the Rosenbrock function

import random
import math
import sys
import csv

#generates the input data, based on the amount of dimensions in the Rosenbrock function, 
# minimum and maximum values and the amount of rows of data you want.
# The created array list gets written to a csv file

def generate_input_list(dimensions, min_value, max_value, amount):
    value_list = []
    rand = 0
    for x in range(amount):
        value_list.append([])
        for y in range(dimensions):
            rand = random.randint(min_value,max_value)
            value_list[x].append(rand)
    dim_str = str(dimensions)
    with open("NeuralNetwork/Data/"+dim_str+'_dim.csv', 'w', newline='', encoding='utf-8') as csvfile:
        csvWriter = csv.writer(csvfile, delimiter=',')
        for row in value_list:
            csvWriter.writerow(row)
    return value_list

#The actual rosenbrock function, used to calculate the output
def rosenbrock(vector):
    rosenbrock = 0
    for i in range(len(vector) - 1):
        rosenbrock += (((1 - vector[i]) ** 2) + 100 * ((vector[i + 1] - (vector[i] ** 2)) ** 2))
<<<<<<< HEAD
    outfile.write(str(rosenbrock))
    outfile.write('\n')

def main():
    vector = []
    global outfile
    outfile = open("NeuralNetwork/Data/five_out.csv", 'w')
    with open("NeuralNetwork/Data/five.csv", 'r') as file:
        for line in file:
            currentline = line.split(",")
            vector = []
            for dim in currentline:
                vector.append(int(dim))
            evaluate(vector)
    file.close()
=======
    return rosenbrock
>>>>>>> d3097c63dcd6c5f4fe9def27306646912c232da1

#Generates the output based on the input arraylist, and writes the results to a csv file
def generate_output_list(input_list):
    dimensions = len(input_list[0])
    output = []
    for x in input_list:
        print(x)
        print(rosenbrock(x))
        output.append(rosenbrock(x))
    dim_str = str(dimensions)
    with open("NeuralNetwork/Data/"+dim_str+'_dim_out.csv', 'w', newline='', encoding='utf-8') as csvfile:
        csvWriter = csv.writer(csvfile, delimiter=',')
        for val in output:
            csvWriter.writerow([val])
    return output
