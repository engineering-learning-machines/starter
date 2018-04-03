#!/usr/bin/env python3
from random import shuffle
import time
import logging
import pickle
from network import Network
# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
DATA_FILE = '/Users/g6714/Data/amazonaws/mnist.pkl'
NORMALIZATION_FACTOR = 1./255
MODEL_FILE = 'model.pkl'
LOG_FORMAT = '%(asctime)-15s %(levelname)-8s %(message)s'

log = logging.getLogger()
formatter = logging.Formatter(LOG_FORMAT)
log.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
handler.setFormatter(formatter)
log.addHandler(handler)
file_handler = logging.FileHandler('test-nn.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
log.addHandler(file_handler)
# ------------------------------------------------------------------------------
# Neuron
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
    log.debug('Loading network...')
    # This is our main object
    with open('model.pkl', 'rb') as fp:
        net = pickle.load(fp)

    # Load the data set
    log.info('Loading data...')
    start = time.time()
    with open(DATA_FILE, 'rb') as fp:
        data = pickle.load(fp, encoding='latin1')
        test_images = NORMALIZATION_FACTOR * data[1][0]
        test_labels = data[1][1]
    log.info('Test data set: {} images'.format(test_labels.shape[0]))
    log.info('Loaded in {:.1f} seconds'.format(time.time() - start))

    # Convert the data into the proper format to speed up calculations
    start = time.time()
    log.info('Converting data...')
    # Convert the images to image vectors
    image_vector_length = test_images.shape[1] * test_images.shape[2]
    test_images.shape = (test_images.shape[0], image_vector_length)
    # One-hot encode the labels
    # test_labels = [encode_one_hot(label) for label in test_labels]
    labels = [encode_one_hot(label) for label in test_labels]
    log.info('Done in {:.1f} s'.format(time.time() - start))

    # Prepare for testing
    log.info('Starting to evaluate the test data...')
    start_total_testing_time = time.time()
    test_index = list(range(test_images.shape[0]))
    shuffle(test_index)
    total_loss = 0
    # Sum of all correctly classified digits for each class
    correct_count = [0]*len(labels[0])
    # Sum of all incorrectly classified digits for each class
    incorrect_count = [0]*len(correct_count)
    # Evaluate on test data
    for index in test_index[:100]:
        image = test_images[index]
        label = labels[index]
        digit_label = test_labels[index]
        net.evaluate(image)
        total_loss += net.get_output_squared_error(label)
        # Determine which digit this was and whether it was correctly classified
        output = net.get_output()
        print(output)
#        if max(output) != output[digit_label]:
#            incorrect_count[digit_label] += 1
#        else:
#            correct_count[digit_label] += 1

    total_loss = total_loss / 100
    log.info('Test Loss: {:.6f}'.format(total_loss))
    log.info('Total testing time: {:.1f} seconds'.format(time.time() - start_total_testing_time))

#    mean_accuracy = 0
#    for i in range(len(correct_count)):
#        total_count = correct_count[i] + incorrect_count[i]
#        accuracy = correct_count[i] / total_count
#        log.info('Class {} accuracy: {:.2f}'.format(i, accuracy))
#        mean_accuracy += accuracy
#    mean_accuracy /= len(correct_count)
#    log.info('Mean accuracy: {:.2f}'.format(mean_accuracy))
#    log.info('Classification rate: {:.1f} %'.format(mean_accuracy*100))
