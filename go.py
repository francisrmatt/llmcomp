"""Main experiment runner"""

import sys
import logging.config
import matplotlib.pyplot as plt
import numpy as np

import config
import get_data

def go_go():

    n_chunks = 1
    cw = 128
    filename = 0
    compressor = 'gzip'

    data = get_data.fetch(n_chunks, cw, filename)
    for datum in data:
        plt.plot(np.frombuffer(datum, dtype = np.uint8))
        plt.savefig('foo.png')

if __name__ == '__main__':
    logging.config.dictConfig(config.LOGGING_CONFIG)
    logger = logging.getLogger(__name__)
    logger.info("Starting new run")
    go_go()