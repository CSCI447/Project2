# USAGE: python rosenbrock.py <input>.txt <output>.txt runs total_sets
# where total_sets is the range of numbers to choose from for the values
# passed into rosenbrock, runs is the total times rosenbrock is computed
# per dimension, must be less than 1/dimensions(total_sets)

import random
import math
import sys
import csv

def generate_input_list(dimensions, min_value, max_value, amount):
    value_list = []
    rand = 0
    for x in range(amount):
        value_list.append([])
        for y in range(dimensions):
            rand = random.randint(min_value,max_value)
            value_list[x].append(rand)
    dim_str = str(dimensions)
    with open("../"+dim_str+'_dim_input.csv', 'w', newline='', encoding='utf-8') as csvfile:
        csvWriter = csv.writer(csvfile, delimiter=',')
        for row in value_list:
            csvWriter.writerow(row)
    return value_list

def generate_output_list(input_list):
    dimensions = len(input_list[0])
    rosenbrock_output = []
    rosenbrock_values = []
    output = []
    for i,x in enumerate(input_list):
        rosenbrock = 0
        rosenbrock_output.append([])
        for dim in range(len(x)-1):
            rosenbrock = (((1 - x[dim]) ** 2) + 100 * ((x[dim+1] - (x[dim] ** 2)) ** 2))
            rosenbrock_output[i].append(rosenbrock)
    for x in rosenbrock_output:
        sum_values = sum(x)
        output.append(sum_values)
    dim_str = str(dimensions)
    with open("../"+dim_str+'_dim_output.csv', 'w', newline='', encoding='utf-8') as csvfile:
        csvWriter = csv.writer(csvfile, delimiter=',')
        for val in output:
            csvWriter.writerow([val])
    return output