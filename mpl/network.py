from random import random
from math import exp

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
WEIGHT_INIT_SCALING_FACTOR = 0.01

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
        # Biases
        self.biases = {}
        # Bias gradients
        self.bias_gradients = {}
        # Forward connections: id -> {id -> weight}
        self.forward_connections = {}
        # Backward connections: id -> {id -> delta_weight_sum}
        self.backward_connections = {}
        # Sorted neurons
        self.sorted_order = []
        # Architecture
        self.layers = {}

    def add_neuron(self):
        neuron_count = len(self.neurons)
        # Add the new neuron to the neuron list
        self.neurons[neuron_count] = 0
        # Add the neuron's bias
        self.biases[neuron_count] = 0
        # Bias gradients
        self.bias_gradients[neuron_count] = 0
        # Add to the feed-forward connections
        self.forward_connections[neuron_count] = {}
        # Add to the backpropagation connections
        self.backward_connections[neuron_count] = {}
        # Return a handle if needed
        return neuron_count

    def __add_layer(self, name, neuron_count):
        self.layers[name] = [self.add_neuron() for _ in range(neuron_count)]

    def __connect(self, source, target, weight=0):
        # Set up the forward and backward connections between neurons
        self.forward_connections[source][target] = weight
        self.backward_connections[target][source] = WEIGHT_INIT_SCALING_FACTOR * random()

    def add_input_layer(self, input_neuron_count):
        self.__add_layer('input', input_neuron_count)

    def add_hidden_layer(self, hidden_neuron_count):
        self.__add_layer('hidden', hidden_neuron_count)
        # Connect to input layer with random weights
        for input_id in self.layers['input']:
            for hidden_id in self.layers['hidden']:
                self.__connect(input_id, hidden_id, WEIGHT_INIT_SCALING_FACTOR * random())

    def add_output_layer(self, output_neuron_count):
        self.__add_layer('output', output_neuron_count)
        # Connect to hidden layer with random weights
        for hidden_id in self.layers['hidden']:
            for output_id in self.layers['output']:
                self.__connect(hidden_id, output_id, WEIGHT_INIT_SCALING_FACTOR * random())

    def sort(self):
        # Store the resulting sorted list of neurons here
        sorted_list = []
        # Find the set of vertices without incoming edges (input neurons)
        queue = {
            n_id for n_id, incoming_connections
            in self.backward_connections.items()
            if len(incoming_connections) == 0
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

        # We need to check for cycles (Kahn's algorithm only makes sense for a DAG)
        if len(sorted_list) != len(in_degree):
            return None
        return sorted_list

    def evaluate(self, input_):
        """
        Calculate the forward pass of the neural network. This is basically the neural network function
        R^n -> R^m
        where n is the input vector dimension and m is the output vector dimension.
        :param input_: Image vector
        :return: Vector of probabilities for individual classes
        """
        # Clean up the previous values, so we don't have to worry about this
        for n_id in self.neurons:
            self.neurons[n_id] = 0

        # Order the neurons in the direction of forward propagation. We can reuse this list
        if len(self.sorted_order) == 0:
            self.sorted_order = self.sort()

        # First we need to multiply the input with the input neurons. These are not sigmoid neurons, so we have a simple
        # multiplication to produce their output. We assume that the input is always equal to the number of input
        # neurons.
        for n_id in self.layers['input']:
            for neighbor_id, weight in self.forward_connections[n_id].items():
                # Since we added the input neuron first, their indices should be 0..m where
                # m is the dimension of the input vector
                self.neurons[neighbor_id] += input_[n_id] * weight

        # Calculate the output of the rest of the sigmoid neurons
        input_length = len(self.layers['input'])
        for n_id in self.sorted_order[input_length+1:]:
            # Apply the activation function to the accumulated weighted sum produced
            # by the previous operations + the neuron's bias
            # self.neurons[n_id] = sigmoid(self.neurons[n_id])
            self.neurons[n_id] = sigmoid(self.neurons[n_id] + self.biases[n_id])
            # push the result to the neighbor neurons
            for neighbor_id, weight in self.forward_connections[n_id].items():
                self.neurons[neighbor_id] += self.neurons[n_id] * weight

        return [self.neurons[n_id] for n_id in self.layers['output']]

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
            # Calculate bias gradients
            self.bias_gradients[n_id] = error

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
            # Calculate bias gradients
            self.bias_gradients[n_id] = error

    def update_weights(self, batch_size, learn_rate):
        reverse_sorted_order = list(reversed(self.sorted_order))
        for n_id in reverse_sorted_order:
            for incoming_neighbor_id in self.backward_connections[n_id].keys():
                # Divide the accumulated gradient sum by the batch size to get
                # the single weight delta
                weight_delta = self.backward_connections[n_id][incoming_neighbor_id] / batch_size
                self.forward_connections[incoming_neighbor_id][n_id] += learn_rate * weight_delta
                # self.forward_connections[incoming_neighbor_id][n_id] += learn_rate * random()

    def update_biases(self, batch_size, learn_rate):
        for n_id in self.biases.keys():
            self.biases[n_id] += learn_rate * self.bias_gradients[n_id]

    def get_output_squared_error(self, label):
        squared_error = 0
        # We need this for consistence with back prop
        output_length = len(label)
        reverse_sorted_order = list(reversed(self.sorted_order))
        # Subtract the label from the result, square, and accumulate the sum
        for i, n_id in enumerate(reverse_sorted_order[:output_length]):
            squared_error += (self.neurons[n_id] - label[i]) ** 2
        return squared_error

    def get_output(self):
        # We need this for consistence with back prop
        output_length = len(self.layers['output'])
        reverse_sorted_order = list(reversed(self.sorted_order))
        # Subtract the label from the result, square, and accumulate the sum
        return [self.neurons[n_id] for n_id in reverse_sorted_order[:output_length]]
