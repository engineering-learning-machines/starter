#!/usr/bin/env python3
from random import shuffle
import time
import logging
import pickle
import numpy as np
# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
INPUT_NEURONS = 768
HIDDEN_NEURONS = 50
OUTPUT_NEURONS = 10
DATA_FILE = '/Users/g6714/Data/amazonaws/mnist.pkl'
BATCH_SIZE = 64
LEARN_RATE = 0.01
# Just to control the overall length of the training cycle during development
MAX_BATCH_COUNT = 60000
BATCH_TEST_INTERVAL = 10
EPOCH_COUNT = 4
NORMALIZATION_FACTOR = 1./255
# WEIGHT_INIT_SCALING_FACTOR = 0.01
LOG_FORMAT = '%(asctime)-15s %(levelname)-8s %(message)s'

log = logging.getLogger()
formatter = logging.Formatter(LOG_FORMAT)
log.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
handler.setFormatter(formatter)
log.addHandler(handler)
file_handler = logging.FileHandler('mn.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
log.addHandler(file_handler)
#
INPUT_SIZE = 784
HIDDEN_SIZE = 100
OUTPUT_SIZE = 10
# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------

def save_array(x, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(x, fp)

def sigmoid(x):
    return 1./(1 + np.exp(-x))


def sigmoid_derivative(x):
    y = sigmoid(x)
    return y*(1.-y)


def encode_one_hot(x, dim=10):
    vector = [0]*dim
    vector[x] = 1
    return vector


class MatrixNet:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.randn(hidden_size, input_size)
        self.biases_hidden = np.random.rand(hidden_size)
        self.hidden_out = None
        self.hidden_activations = None
        self.weights_hidden_output = np.random.randn(output_size, hidden_size)
        self.biases_output = np.random.randn(output_size)
        self.output_out = None
        self.output_activations = None

    def forward(self, x):
        self.hidden_out = np.dot(self.weights_input_hidden, x) + self.biases_hidden
        self.hidden_activations = sigmoid(self.hidden_out)
        self.output_out = np.dot(self.weights_hidden_output, self.hidden_activations) + self.biases_output
        self.output_activations = sigmoid(self.output_out)
        return self.output_activations

    def backpropagate(self, x, label):
        error = (label - self.output_activations)
        output_derivative = self.output_activations * (1. - self.output_activations)
        error_output_derivative = error * output_derivative
        grad_weights_out = np.outer(error_output_derivative, self.hidden_activations)
        grad_biases_out = error_output_derivative

        hidden_derivative = self.hidden_activations * (1. - self.hidden_activations)
        a1 = np.dot(self.weights_hidden_output.T, error_output_derivative)
        a2 = np.dot(a1, hidden_derivative)
        grad_weights_hidden = np.outer(a2, x)
        grad_biases_hidden = a2

        return np.dot(error, error), grad_weights_out, grad_biases_out, grad_weights_hidden, grad_biases_hidden

    def update_params(self, batch_size, learn_rate, grad_output_weights, grad_output_biases, grad_hidden_weights, grad_hidden_biases):
        inv_batch_size = 1./batch_size
        self.weights_hidden_output += learn_rate * inv_batch_size * grad_output_weights
        self.biases_output += learn_rate * inv_batch_size * grad_biases_out
        self.weights_input_hidden += learn_rate * inv_batch_size * grad_weights_hidden
        self.biases_output += learn_rate * inv_batch_size * grad_biases_hidden

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------


if __name__ == '__main__':
    # Start
    log.debug('Constructing network...')
    # This is our main object
    net = MatrixNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

    # Load the data set
    log.info('Loading data...')
    start = time.time()
    with open(DATA_FILE, 'rb') as fp:
        data = pickle.load(fp, encoding='latin1')
        training_images = NORMALIZATION_FACTOR * data[0][0]
        training_labels = data[0][1]
        test_images = NORMALIZATION_FACTOR * data[1][0]
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

    # Create an index for the training data
    training_index = list(range(training_images.shape[0]))
    # Create an index for the testing data
    test_index = list(range(test_images.shape[0]))

    for epoch_index in range(EPOCH_COUNT):
        log.debug('--------------- EPOCH [{:0>3}] ---------------'.format(EPOCH_COUNT))
        # Randomize the training data
        shuffle(training_index)
        # Train the system
        start_total_training_time = time.time()
        # for batch_index, batch_partition in enumerate(range(training_images.shape[0])[::BATCH_SIZE]):
        for batch_index, batch_partition in list(enumerate(range(training_images.shape[0])[::BATCH_SIZE])):
            log.debug('=== batch [{:0>3}] ==='.format(batch_index))
            if batch_index >= MAX_BATCH_COUNT:
                break
            # Process one batch
            start_batch = time.time()
            batch_item_indices = training_index[batch_partition:batch_partition+BATCH_SIZE]
            #
            total_loss = 0
            output_gradients = np.zeros((OUTPUT_SIZE, HIDDEN_SIZE))
            output_biases = np.zeros(OUTPUT_SIZE)
            hidden_gradients = np.zeros((HIDDEN_SIZE, INPUT_SIZE))
            hidden_biases = np.zeros(HIDDEN_SIZE)
            #
            for j in batch_item_indices:
                network_input = training_images[j]
                label = training_labels[j]
                network_output = net.forward(network_input)
                single_loss, grad_weights_out, grad_biases_out, grad_weights_hidden, grad_biases_hidden = net.backpropagate(network_input, label)
                #
                total_loss += single_loss
                output_gradients += grad_weights_out
                output_biases += grad_biases_out
                hidden_gradients += grad_weights_hidden
                hidden_biases += grad_biases_hidden

            # Update weights when finished with batch
            batch_size = len(batch_item_indices)
            net.update_params(batch_size, LEARN_RATE, output_gradients, output_biases, hidden_gradients, hidden_biases)
            total_loss /= float(batch_size)

            log.debug('Processed batch: {:0>6} [{:.1f}] s'.format(batch_index+1, time.time() - start_batch))

            # Test every BATCH_TEST_INTERVAL batches:
            # (might compare overfitting vs unseen test data)
            if (batch_index + 1) % BATCH_TEST_INTERVAL == 0:
                total_loss = 0
                shuffle(test_index)
                for index in test_index[:BATCH_SIZE]:
                    image = test_images[index]
                    label = test_labels[index]
                    output = net.forward(image)
                    error = label - output
                    total_loss += np.dot(error, error)
                total_loss = total_loss / BATCH_SIZE
                log.info('Test Loss: {:.6f}'.format(total_loss))

            # Save model at this checkpoint
#            with open('model.pkl', 'wb') as fp:
#                pickle.dump(net, fp)
#
        log.info('Total training time: {:.1f} seconds'.format(time.time() - start_total_training_time))

        save_array(net.weights_input_hidden, 'weights_hidden.pkl')
        save_array(net.biases_hidden, 'biases_hidden.pkl')
        save_array(net.weights_hidden_output, 'weights_output.pkl')
        save_array(net.biases_output, 'biases_output.pkl')

#        # Save model when training is done
#        with open('model.pkl', 'wb') as fp:
#            pickle.dump(net, fp)
#