#!/usr/bin/python
from random import random
from math import exp
import pickle
# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
INPUT_NEURONS = 5
HIDDEN_NEURONS = 3
OUTPUT_NEURONS = 2
DATA_FILE = '/Users/g6714/Data/amazonaws/mnist.pkl'

# ------------------------------------------------------------------------------
# Neuron
# ------------------------------------------------------------------------------


def sigmoid(x):
    return 1./(1 + exp(-x))
# ------------------------------------------------------------------------------
# Neural Network
# ------------------------------------------------------------------------------


class Network:
    def __init__(self):
        # Neurons: id -> output
        self.neurons = {}
        # Forward connections: id -> {id -> weight}
        self.forward_connections = {}
        # Backward connections: id -> {id -> delta_weight_sum}
        self.backward_connections = {}

    def add_neuron(self):
        neuron_count = len(self.neurons)
        # Add the new neuron to the neuron list
        self.neurons[neuron_count] = 0
        # Add to the feed-forward connections
        self.forward_connections[neuron_count] = {}
        # Add to the backpropagation connections
        self.backward_connections[neuron_count] = {}
        # Return a handle if needed
        return neuron_count

    def connect(self, source, target, weight=0):
        # Set up the forward and backward connections between neurons
        self.forward_connections[source][target] = weight
        self.backward_connections[target][source] = 0

    def sort(self):
        # Store the resulting sorted list of neurons here
        sorted_list = []
        # Find the set of vertices without incoming edges (input neurons)
        queue = {
            n_id for n_id, incoming_connections
            in self.backward_connections.items()
            if len(incoming_connections)==0
        }
        # We need to update a list of the in-degrees of all neurons
        in_degree = {n_id: len(connections) for n_id, connections in self.backward_connections.items()}
        while queue:
            # Take an element from the queue and add it to the sorted list
            n_id = queue.pop()
            sorted_list.append(n_id)
            # Since we removed the neuron from the queue, update the incoming connections of its
            # neighbors to be one less:
            for neighbor_id in self.forward_connections.get(n_id, {}).keys():
                in_degree[neighbor_id] -= 1
                # If the neighbor has reached in-degree 0, add it to the queue
                if in_degree[neighbor_id] == 0:
                    queue.add(neighbor_id)

        # We need to check for cycles (Kahn's only makes sense for a DAG)
        if len(sorted_list) != len(in_degree):
            return None
        return sorted_list

#    def evaluate(self, input_):
#        neuron_count = len(self.connections.keys())
#        sorted_neurons = self.sort()
#        # Store the outputs
#        outputs = [0] * neuron_count
#        # First we need to multiply the input with the input neurons. These are not sigmoid neurons, so we have a simple
#        # multiplication to produce their output. We assume that the input is always equal to the number of input
#        # neurons.
#        for n_id in sorted_neurons[:len(input_)]:
#            for neighbor_id, weight in self.connections[n_id].items():
#                outputs[neighbor_id] += input_[n_id] * weight
#
#        # Calculate the output of the rest of the sigmoid neurons
#        for n_id in sorted_neurons[n_id+1:]:
#            outputs[n_id] = sigmoid(outputs[n_id])
#            # push the result to the neighbor neurons
#            for neighbor_id, weight in self.connections[n_id].items():
#                outputs[neighbor_id] += outputs[n_id] * weight
#
#        print(outputs)
#
# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------


if __name__ == '__main__':
    # This is our main object
    net = Network()
    # Keep track of the neurons
    input_neurons = []
    hidden_neurons = []
    output_neurons = []
    # Add the neuron layers
    for n in range(INPUT_NEURONS):
        input_neurons.append(net.add_neuron())
    for n in range(HIDDEN_NEURONS):
        hidden_neurons.append(net.add_neuron())
    for n in range(OUTPUT_NEURONS):
        output_neurons.append(net.add_neuron())
    # Connect the neuron layers
    for input_id in input_neurons:
        for hidden_id in hidden_neurons:
            net.connect(input_id, hidden_id, random())

    for hidden_id in hidden_neurons:
        for output_id in output_neurons:
            net.connect(hidden_id, output_id, random())

    print(net.sort())
##    result = net.evaluate([0.35, 0.31, 0.23, 0.21, 0.02])
##    print(result)
#
#
#
#
#
#
#
#    # Load the data set
##    with open(DATA_FILE, 'rb') as fp:
##        data = pickle.load(fp, encoding='latin1')
##        training_images = data[0][0]
##        training_labels = data[0][1]
##        test_images = data[1][0]
##        test_labels = data[1][1]
#
