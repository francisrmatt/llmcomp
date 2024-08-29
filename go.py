"""Main experiment runner"""

import sys
import logging.config
import matplotlib.pyplot as plt
import numpy as np

import config
import get_data
import compressor

def go_go():

    logging.config.dictConfig(config.LOGGING_CONFIG)
    logger = logging.getLogger()
    logger.info("Starting new run")

    n_chunks = 10
    cw = 128
    filename = 0
    compressor_name = 'btransformer'

    # Fetch data
    data = get_data.fetch(n_chunks, cw, filename)

    # Compress data 
    rate, time = compressor.evaluate_compressor(compressor_name, data, None, n_chunks, cw)
    logger.info(f'Compressor ran with {rate=} and {time=}')




if __name__ == '__main__':
    go_go()