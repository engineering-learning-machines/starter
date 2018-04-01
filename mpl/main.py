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
        # Sorted neurons
        self.sorted_order = []

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

    def evaluate(self, input_):
        # Clean up the previous values, so we don't have to worry about this
        for n_id in self.neurons:
            self.neurons[n_id] = 0

        # Order the neurons in the direction of forward propagation. We can reuse this list
        if len(self.sorted_order) == 0:
              self.sorted_order = self.sort()

        # First we need to multiply the input with the input neurons. These are not sigmoid neurons, so we have a simple
        # multiplication to produce their output. We assume that the input is always equal to the number of input
        # neurons.
        input_length = len(input_)
        for n_id in self.sorted_order[:input_length]:
            # Push the weight x input product to the next layer
            for neighbor_id, weight in self.forward_connections[n_id].items():
                # outputs[neighbor_id] += input_[n_id] * weight
                self.neurons[neighbor_id] += input_[n_id] * weight

        # Calculate the output of the rest of the sigmoid neurons
        for n_id in self.sorted_order[input_length+1:]:
            # Apply the activation function to the accumulated weighted sum produced
            # by the previous operations
            self.neurons[n_id] = sigmoid(self.neurons[n_id])
            # push the result to the neighbor neurons
            for neighbor_id, weight in self.forward_connections[n_id].items():
                self.neurons[neighbor_id] += self.neurons[n_id] * weight

    def backpropagate(self, label):
        """
        Note that the weight deltas should be cleaned up during the weight update!!! We rely on that here.
        """
        # The number of outputs *must match* the one-hot encoded label vector dimension
        output_length = len(label)
        reverse_sorted_order = list(reversed(self.sorted_order))
        # We need to handle outputs separately. More code for labels with dim > 1?
        # Here we use the index i just to fetch elements of label. The fixed order list guarantees that
        # the output neuron values are always matched with their labels.
        for i, n_id in enumerate(reverse_sorted_order[:output_length]):
            output = self.neurons[n_id]
            error = (label[i] - output) * output * (1. - output)
            for incoming_neighbor_id in self.backward_connections[n_id].keys():
                self.backward_connections[n_id][incoming_neighbor_id] = error * self.neurons[incoming_neighbor_id]

        # Backpropagation rule for the rest of the neurons
        for n_id in reverse_sorted_order[output_length+1:]:
            output = self.neurons[n_id]
            # Find the multiplicative derivative terms from previous steps
            error_sum = sum([
                self.backward_connections[outgoing_neighbor_id][n_id]
                for outgoing_neighbor_id
                in self.forward_connections[n_id].keys()
            ])
            error = output * (1 - output) * error_sum
            # Multiply with the previous inputs in the chain to finalize the derivatives w.r.t. each incoming connection
            for incoming_neighbor_id in self.backward_connections[n_id].keys():
                self.backward_connections[n_id][incoming_neighbor_id] = error * self.neurons[incoming_neighbor_id]

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

    x = [
        [0.35, 0.31, 0.23, 0.21, 0.02],
        [0.25, -0.1, 0.20, 0.81, -1.02],
        [0.83, 0.21, 0.8, -1.21, 3.02]
    ]
    l = [
        [0, 0],
        [0, 1],
        [1, 0]
    ]
    # net.evaluate(x)
    # print(net.neurons)
    # Labels should be one-hot encoded. For only one output the value is either 0 or 1
    # label = [1]
    # net.backpropagate(label)

    for i in range(len(x)):
        net.evaluate(x[i])
        net.backpropagate(l[i])



#    # Load the data set
##    with open(DATA_FILE, 'rb') as fp:
##        data = pickle.load(fp, encoding='latin1')
##        training_images = data[0][0]
##        training_labels = data[0][1]
##        test_images = data[1][0]
##        test_labels = data[1][1]
#
