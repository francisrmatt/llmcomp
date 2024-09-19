"""Main data fetcher"""

from typing import Generator, Callable
import numpy as np
import audioop
import logging.config
import os
import sys
from numpy.lib.stride_tricks import sliding_window_view
import constants

import matplotlib.pyplot as plt


def fetch(stream_mode : bool,
          amt : int, # number of chunks for non stream and length of stream for stream
          context_window: int, # needed for non stream
          filename: int, 
          scale: float = 1,
          offset: int = 0,
          map_fn: Callable = None,
          rchunks : bool = False,
          ) -> Generator[any, any, any]:
    """Returns chunks of float data which are fetched from wifi data

    Args:
        n_chunks: Number of chunks to fetch. If -1 works in stream mode and assumes test data
        context_window: How large each context window is.
        filename: Integer describing which file to fetch from, if -1, will ignore and start from 0 and go until n_chunks limit is reached.
        normalisation: Default True, will transform the data so the maximum value is 255 and the minimum value is 0.
    """

    # Logging config
    logging.config.dictConfig(constants.LOGGING_CONFIG)
    logger = logging.getLogger(__name__) 

    # All file names should be in the format data/wX.data where X is an integer.
    if stream_mode:
        logger.debug(f'Fetching data in stream mode with stream length {amt}')
        context_window = amt
        n_chunks = 1
        
    else:
        # TODO this is wrong if we do all
        logger.debug(f'Fetching data in chunk mode with {"all" if filename == -1 else amt} chunks each size {context_window}')
        n_chunks = amt

    # Load in data 
    if filename == -1: # TODO as we need to consider pre-processed data
        n_chunks = 2e11
        logger.debug('Loading all data')
        n_files = len(next(os.walk('data/raw_data'))[2])
        print(f'{n_files=}')
        data = np.fromfile('data/raw_data/w0.data', np.int16)
        for i in range(1, n_files - 1):
            data = np.append(data, np.fromfile('data/raw_data/w{}.data'.format(i), np.int16))
    elif filename == -2: # Test file is -2
        logger.debug(f'Loading test file with offset {offset}')
        data = np.fromfile('data/test/w_test.data', np.int16)
        logger.debug(f'Ofsetting by {offset}')
        data = data[offset:]
    else:
        logger.debug(f'Loading from file {filename} with offset {offset}')
        data = np.fromfile('data/raw_data/w{}.data'.format(filename), np.int16)
        data = data[offset:]


    data_iq = data[0::2] + 1j*data[1::2]
    if scale != 1.0:
        logger.debug(f'Scaling by factor {scale}')
    signal_real = (scale*np.real(data_iq).copy()).astype(np.int16) # Add a scaling factor?
    signal_real = signal_real.tobytes()

    new_signal = audioop.lin2lin(signal_real, 2, 1)
    new_signal = audioop.bias(new_signal, 1, 2**7)

    def _extract_rf_slide(sample: bytes):
        x = np.frombuffer(sample, dtype = np.uint8)
        if rchunks:
            patches = np.array_split(x,
                range(
                    context_window,
                    len(sample),
                    context_window,))

        else:
            patches = sliding_window_view(x, context_window)

        if len(patches[-1]) != context_window:
            patches.pop()

        if map_fn is not None:
            return map(map_fn, patches)

        return map(lambda patch: patch.tobytes(), patches)

    idx = 0
    for patch in _extract_rf_slide(new_signal):
        if idx == n_chunks:
            return
        yield patch
        idx += 1
