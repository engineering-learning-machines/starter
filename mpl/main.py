#!/usr/bin/env python3
from random import shuffle
import time
import logging
import pickle
from network import Network
# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
INPUT_NEURONS = 768
HIDDEN_NEURONS = 50
OUTPUT_NEURONS = 10
DATA_FILE = '/Users/g6714/Data/amazonaws/mnist.pkl'
BATCH_SIZE = 32
LEARN_RATE = 0.05
# Just to control the overall length of the training cycle during development
MAX_BATCH_COUNT = 60000
BATCH_TEST_INTERVAL = 10
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
file_handler = logging.FileHandler('nn.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
log.addHandler(file_handler)
# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------


def encode_one_hot(x, dim=10):
    vector = [0]*dim
    vector[x] = 1
    return vector

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------


if __name__ == '__main__':
    # Start
    log.debug('Constructing network...')
    # This is our main object
    net = Network()
    # Construct the neural network
    net.add_input_layer(INPUT_NEURONS)
    net.add_hidden_layer(HIDDEN_NEURONS)
    net.add_output_layer(OUTPUT_NEURONS)

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

    # Prepare for testing
    test_index = list(range(test_images.shape[0]))
    # Train the system
    start_total_training_time = time.time()
    for batch_index, batch_partition in enumerate(range(training_images.shape[0])[::BATCH_SIZE]):
        if batch_index >= MAX_BATCH_COUNT:
            break
        # Process one batch
        start_batch = time.time()
        batch_data = training_images[batch_partition:batch_partition+BATCH_SIZE]
        batch_labels = training_labels[batch_partition:batch_partition+BATCH_SIZE]
        for image, label in zip(batch_data, batch_labels):
            output = net.evaluate(image)
            net.backpropagate(label)
        # Update weights when finished with batch
        net.update_weights(batch_data.shape[0], LEARN_RATE)
        net.update_biases(batch_data.shape[0], LEARN_RATE)
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

        # Save model at this checkpoint
        with open('model.pkl', 'wb') as fp:
            pickle.dump(net, fp)

    log.info('Total training time: {:.1f} seconds'.format(time.time() - start_total_training_time))

    # Save model when training is done
    with open('model.pkl', 'wb') as fp:
        pickle.dump(net, fp)
