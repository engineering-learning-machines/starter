#!/usr/bin/env python3
from random import random
from random import shuffle
from math import exp
import pickle
import time
import logging
import sys
# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
INPUT_NEURONS = 768
HIDDEN_NEURONS = 50
OUTPUT_NEURONS = 10
DATA_FILE = '/Users/g6714/Data/amazonaws/mnist.pkl'
# BATCH_SIZE = 64
BATCH_SIZE = 16
LEARN_RATE = 0.1
# Just to control the overall length of the training cycle during development
MAX_BATCH_COUNT = 500
BATCH_TEST_INTERVAL = 10

log = logging.getLogger()
log.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
log.addHandler(handler)
# ------------------------------------------------------------------------------
# Neuron
# ------------------------------------------------------------------------------


def sigmoid(x):
    return 1./(1 + exp(-x))

def encode_one_hot(x, dim=10):
    vector = [0]*dim
    vector[x] = 1
    return vector
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
        # Architecture
        self.layers = {}
        # self.input_neurons = []
        # self.hidden_neurons = []
        # self.output_neurons = []

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

    def __add_layer(self, name, neuron_count):
        self.layers[name] = [net.add_neuron() for _ in range(neuron_count)]

    def __connect(self, source, target, weight=0):
        # Set up the forward and backward connections between neurons
        self.forward_connections[source][target] = weight
        self.backward_connections[target][source] = 0

    def add_input_layer(self, input_neuron_count):
        self.__add_layer('input', input_neuron_count)

    def add_hidden_layer(self, hidden_neuron_count):
        self.__add_layer('hidden', hidden_neuron_count)
        # Connect to input layer with random weights
        for input_id in self.layers['input']:
            for hidden_id in self.layers['hidden']:
                self.__connect(input_id, hidden_id, random())

    def add_output_layer(self, output_neuron_count):
        self.__add_layer('output', output_neuron_count)
        # Connect to hidden layer with random weights
        for hidden_id in self.layers['hidden']:
            for output_id in self.layers['output']:
                self.__connect(hidden_id, output_id, random())

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

    def update_weights(self, batch_size, learn_rate):
        reverse_sorted_order = list(reversed(self.sorted_order))
        for n_id in reverse_sorted_order:
            for incoming_neighbor_id in self.backward_connections[n_id].keys():
                # Divide the accumulated gradient sum by the batch size to get
                # the single weight delta
                weight_delta = self.backward_connections[n_id][incoming_neighbor_id] / batch_size
                self.forward_connections[incoming_neighbor_id][n_id] += learn_rate * weight_delta

    def get_output_squared_error(self, label):
        squared_error = 0
        # We need this for consistence with back prop
        output_length = len(label)
        reverse_sorted_order = list(reversed(self.sorted_order))
        # Subtract the label from the result, square, and accumulate the sum
        for i, n_id in enumerate(reverse_sorted_order[:output_length]):
            squared_error += (self.neurons[n_id] - label[i]) ** 2
        return squared_error

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------


if __name__ == '__main__':
    # Start
    log.debug('Constructing network...')
    # This is our main object
    net = Network()
    # Keep track of the neurons
    # input_neurons = [net.add_neuron() for n in range(INPUT_NEURONS)]
    # hidden_neurons = [net.add_neuron() for n in range(HIDDEN_NEURONS)]
    # output_neurons = [net.add_neuron() for n in range(OUTPUT_NEURONS)]
    net.add_input_layer(INPUT_NEURONS)
    net.add_hidden_layer(HIDDEN_NEURONS)
    # Add the neuron layers
    # Connect the neuron layers
#    for input_id in input_neurons:
#        for hidden_id in hidden_neurons:
#            net.connect(input_id, hidden_id, random())

#    for hidden_id in hidden_neurons:
#        for output_id in output_neurons:
#            net.connect(hidden_id, output_id, random())

    # Load the data set
    log.info('Loading data...')
    start = time.time()
    with open(DATA_FILE, 'rb') as fp:
        data = pickle.load(fp, encoding='latin1')
        training_images = data[0][0]
        training_labels = data[0][1]
        test_images = data[1][0]
        test_labels = data[1][1]
    log.info('Training data set: {} images'.format(training_labels.shape[0]))
    log.info('Test data set: {} images'.format(test_labels.shape[0]))
    log.info('Loaded in {:.1f} seconds'.format(time.time() - start))

    # Convert the data into the proper format to speed up calculations
    start = time.time()
    log.info('Converting data...')
    # Convert the images to image vectors
    image_vector_length = training_images.shape[1] * training_images.shape[2]
    training_images.shape = (training_images.shape[0], image_vector_length)
    test_images.shape = (test_images.shape[0], image_vector_length)
    # One-hot encode the labels
    training_labels = [encode_one_hot(label) for label in training_labels]
    test_labels = [encode_one_hot(label) for label in test_labels]
    log.info('Done in {:.1f} s'.format(time.time() - start))

    # Prepare for testing
    test_index = list(range(test_images.shape[0]))
    # Train the system
    start_total_training_time = time.time()
    for batch_index, batch_partition in enumerate(range(test_images.shape[0])[::BATCH_SIZE]):
        if batch_index >= MAX_BATCH_COUNT:
            break
        # Process one batch
        start_batch = time.time()
        batch_data = test_images[batch_partition:batch_partition+BATCH_SIZE]
        batch_labels = test_labels[batch_partition:batch_partition+BATCH_SIZE]
        for image, label in zip(batch_data, batch_labels):
            net.evaluate(image)
            net.backpropagate(label)
        # Update weights when finished with batch
        net.update_weights(batch_data.shape[0], LEARN_RATE)
        log.debug('Processed batch: {:0>6} [{:.1f}] s'.format(batch_index+1, time.time() - start_batch))

        # Test every BATCH_TEST_INTERVAL batches:
        # (might compare overfitting vs unseen test data)
        if (batch_index + 1) % BATCH_TEST_INTERVAL == 0:
            total_loss = 0
            shuffle(test_index)
            for index in test_index[:BATCH_SIZE]:
                image = test_images[index]
                label = test_labels[index]
                net.evaluate(image)
                total_loss += net.get_output_squared_error(label)
            total_loss = total_loss / BATCH_SIZE
            log.info('Test Loss: {:.6f}'.format(total_loss))

    log.info('Total training time: {:.1f} seconds'.format(time.time() - start_total_training_time))
