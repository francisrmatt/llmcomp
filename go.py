"""Main experiment runner"""

import sys
import matplotlib.pyplot as plt
import numpy as np
import yaml

import time
import constants
import get_data
import compressor

from btransformer.transformer import TransformerConfig
import btransformer.train
import language_model
import utils
import os
import functools

# Arg-parser
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--action', dest = 'action', required=True)
parser.add_argument('--compressor', dest = 'compressor')
parser.add_argument('--amt', dest = 'amt')
parser.add_argument('--cw', dest = 'cw')
parser.add_argument('--which', dest = 'which')
parser.add_argument('--stream', dest = 'stream', action=argparse.BooleanOptionalAction)
parser.add_argument('--shh', dest = 'shh', action=argparse.BooleanOptionalAction)
parser.add_argument('--rchunks', dest = 'rchunks', action=argparse.BooleanOptionalAction)
parser.add_argument('--file', dest = 'file')
parser.add_argument('--chunks', dest = 'chunks')
parser.add_argument('--offset', dest = 'offset')
parser.add_argument('--scale', dest = 'scale')
args = parser.parse_args()

# Logger
import logging.config
logging.config.dictConfig(constants.LOGGING_CONFIG)
logger = logging.getLogger()

def eval():
    
    logger.info('--- EVALUATING COMPRESSOR ---') 

    which = args.which
    if not os.path.isdir(f'params/{which}'):
        logger.error(f'{which} is not a valid parameter set, quitting.')
        sys.exit(-1)

def train():

    # Check which is a valid folder
    logger.info('---- BEGINNING TRAINING ----')
    which = args.which
    if not os.path.isdir(f'params/{which}'):
        logger.error(f'{which} is not a valid parameter set, quitting.')
        sys.exit(-1)
    
    # If it is a valid folder there are two outcomes, it is new training
    # Or there are parameters already there and we continue training
    with open(f'params/{which}/info.yml', 'r') as f:
        info = yaml.safe_load(f)

    new_train = not os.path.isfile(f'params/{which}/params.npz')

    if new_train:
        logger.info(f'New parameters')
    else:
        logger.info(f'Old parameters with {info["training"]} runs')

    logger.info(f'Training with an extra {args.amt} steps with batch size {info["bs"]}')

    config = TransformerConfig(
        vocab_size = info['vocab_size'],
        embedding_dim = info['embedding_dim'],
        num_heads = info['num_heads'],
        num_layers = info['num_layers'],
        emb_init_scale = info['emb_init_scale'],
        widening_factor = info['widening_factor'],
    )

    logger.info(f'Parameters for transformer are {config=}')

    t0 = time.perf_counter()
    params, loss = btransformer.train.train_transformer_decoder(
        new_train = new_train,
        which = which,
        config = config,
        training_steps = int(args.amt),
        cw = info['cw'],
        log_every = int(args.amt)//10,
        batch_size = info['bs'],
        use_tqdm = not args.shh,
    )
    t1 = time.perf_counter()
    running_time = t1 - t0
    logger.info(f'{args.amt} training run complete (total {info["training"] + int(args.amt)}) in {running_time} seconds with loss {loss}')

    info['training'] += int(args.amt)
    np.savez(f'params/{which}/params.npz', **params)
    logger.info(f'Saved params in params/{which}/params.npz file')

    # Rewrite yaml
    with open(f'params/{which}/info.yml', 'w') as f:
        yaml.dump(info, f, default_flow_style=False)


# TODO this is unsophisticated
def decompress():

    logger.info('Decompressing')
    with open('compressed_bytes.data', 'rb') as f:
        d = f.read()

    logger.info(f'Read in {len(d)} bytes from compressed_bytes.data')

    cw = int(args.cw)
    decompressed_data = language_model.decompress(d, 5, cw)
    
    p = np.frombuffer(decompressed_data, dtype = np.uint8)
    plt.figure()
    plt.plot(p)
    plt.savefig('figs/tmp/decompressed.png')
    plt.close()

def compress():
    pass

    # Options for compressing
    # mRF
    # non-mRF
        # Sequential, cheating
        # Non-sequential, cheating
        # Sequential, non-cheating
        # Non-sequential, non-cheating

    # We want to change the compressor so that it works on a long stream of bytes
    # rather than chunks. If we do end up doing chunking then we can specify that within 
    # the compression function

    # What we actually want to do is pass in 'c512_001' and load in the info file
    logger.info('---- BEGINNING COMPRESSION ----')
    compressor_name = args.compressor
    if compressor_name == 'gzip' or compressor_name == 'llama':
        config = None
        cw = int(args.cw)
        info = {}
        info['vocab_size'] = 256
    else:
        with open(f'params/{args.which}/info.yml', 'r') as f:
            info = yaml.safe_load(f)
        config = TransformerConfig(
            vocab_size = info['vocab_size'],
            embedding_dim = info['embedding_dim'],
            num_heads = info['num_heads'],
            num_layers = info['num_layers'],
            emb_init_scale = info['emb_init_scale'],
            widening_factor = info['widening_factor'],
        )
        cw = info['cw'] 

    # Stream mode
    stream_mode = 0
    if args.stream:
        logging.info('Compressing in stream mode')
        compressor_name += '_smooth'
        stream_mode = 1
        cw = 0

    offset = 0 if not args.offset else int(args.offset)
    scale = 0 if not args.scale else float(args.scale)
    rchunks = 0 if not args.rchunks else 1
        
    if not args.scale:
        scale = 1.0
    else:
        scale = float(args.scale)

    logging.info(f'Considering model {args.which} using compressor {compressor_name}')
    logging.info(f'Model information: {config}')
    logging.info(f'Scale = {scale}, Offset = {offset}')


    data = get_data.fetch(stream_mode = stream_mode,
                               amt = int(args.amt),
                               context_window = cw, 
                               filename = int(args.file), 
                               scale = scale,
                               offset = offset,
                               map_fn = None,
                               rchunks = rchunks,
    )

    mask_fn = None
    if info['vocab_size'] == 128:
        mask_fn = utils.right_shift_bytes_by_one

    rate, time = compressor.evaluate_compressor(compress_fn_name = compressor_name, 
                                                params = args.which,
                                                config = config,
                                                data = data,
                                                mask_fn = mask_fn,
                                                )

    logger.info(f'Compressor ran with {rate=} and {time=}') 


def go():

    logger.info("Starting new run")

    # Fetch data
    n_chunks = 1
    cw = 512
    filename = 0
    compressor_name = 'btransformer'
    data = get_data.fetch(n_chunks, cw, filename)

    # Training
#    params, last_loss = btransformer.train.train_transformer_decoder(data = data,
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

    if args.action == 'train':
        train()
    elif args.action == 'compress':
        compress()
    elif args.action == 'decompress':
        decompress()
    else:
        logger.error('Not a valid action')

    logger.debug('Programme exiting')