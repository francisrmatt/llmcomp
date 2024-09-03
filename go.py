"""Main experiment runner"""

import sys
import logging.config
import matplotlib.pyplot as plt
import numpy as np

import constants
import get_data
import compressor

from btransformer.transformer import TransformerConfig
import btransformer.train as train

def go():

    logging.config.dictConfig(constants.LOGGING_CONFIG)
    logger = logging.getLogger()
    logger.info("Starting new run")

    # Fetch data
    n_chunks = 1
    cw = 32
    filename = 0
    compressor_name = 'llama'
    data = get_data.fetch(n_chunks, cw, filename)

    # Training
#    params, last_loss = train.train_transformer_decoder(data = data,
          #tconfig = constants.TRANSFORMER_CONFIG,
          #training_steps = 20000,
          #log_every = 1000,
          #batch_size = 1,
          #learning_rate = 1e-5,
          #use_tqdm = True,
          #)

    #logger.info(f'{last_loss=}')
    #np.savez('btransformer/params.npz', **params)
    #logging.info('Parameters saved in file btransformer/params.npz')

    # Compress data 
    test_data = []
    idx = 0
    for datum in data:
        test_data.append(datum)
        idx += 1
        if idx == 1:
            break

    plt.figure()
    plt.plot(np.frombuffer(test_data[0], dtype = np.uint8))
    plt.savefig('foobar.png')

    rate, time = compressor.evaluate_compressor(compressor_name, test_data, None, len(test_data), cw)
    logger.info(f'Compressor ran with {rate=} and {time=}')


if __name__ == '__main__':
    go()