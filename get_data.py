"""Main data fetcher"""

from typing import Generator
import numpy as np
import audioop
import logging.config
import os
import sys
from numpy.lib.stride_tricks import sliding_window_view
import constants

import matplotlib.pyplot as plt


def fetch(n_chunks: int, 
          context_window: int, 
          filename: int, 
          normalisation: bool = True,
          ) -> Generator[any, any, any]:
    """Returns chunks of float data which are fetched from wifi data

    Args:
        n_chunks: Number of chunks to fetch.
        context_window: How large each context window is.
        filename: Integer describing which file to fetch from, if -1, will ignore and start from 0 and go until n_chunks limit is reached.
        normalisation: Default True, will transform the data so the maximum value is 255 and the minimum value is 0.
    """

    # Logging config
    logging.config.dictConfig(constants.LOGGING_CONFIG)
    logger = logging.getLogger(__name__) 

    # All file names should be in the format data/wX.data where X is an integer.
    file_str = 'all' if filename == -1 else str(filename)
    logger.info(f'Fetching {n_chunks} of size {context_window} from file {file_str} with {normalisation=}')

    # Load in data 
    if filename == -1:
        logger.info('Loading all data')

        n_files = len(next(os.walk('data/'))[2])
        print("DEBUG")
        print(n_files)

        data = np.fromfile('data/w0.data', np.int16)
        for i in range(1, n_files - 1):
            data = np.append(data, np.fromfile('data/w{}.data'.format(i), np.int16))

    else:
        logger.info('Loading from file {}'.format(filename))
        data = np.fromfile('data/w{}.data'.format(filename), np.int16)

    data_iq = data[0::2] + 1j*data[1::2]
    signal_real = np.real(data_iq).copy().astype(np.int16)
    signal_real = signal_real.tobytes()

    new_signal = audioop.lin2lin(signal_real, 2, 1)
    new_signal = audioop.bias(new_signal, 1, 2**7)

    def _extract_rf_slide(sample: bytes):
        x = np.frombuffer(sample, dtype = np.uint8)
        patches = sliding_window_view(x, context_window)
        if len(patches[-1]) != context_window:
            patches.pop()

        def our_norm(patch):
            maxx = max(patch)
            minx = min(patch)
            old_range = maxx - minx
            new_range = 255

            patch = [(i - minx) * (new_range // old_range) for i in patch]
            return bytes(patch)

        if normalisation:
            return map(our_norm, patches)

        return map(lambda patch: patch.tobytes(), patches)

    idx = 0
    for patch in _extract_rf_slide(new_signal):
        if idx == n_chunks:
            return
        yield patch
        idx += 1
