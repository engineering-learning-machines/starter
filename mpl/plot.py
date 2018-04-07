#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

DATA_FILE = 'loss.dat'
BATCH_SAMPLE_INTERVAL = 10

with open(DATA_FILE, 'r') as fp:
    loss = np.array(list(map(lambda s: float(s.strip('\n')), fp.readlines())))

batch_number = np.arange(loss.shape[0]) * BATCH_SAMPLE_INTERVAL
plt.plot(batch_number, loss)
plt.grid()
plt.show()
