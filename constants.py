"""Project wide default values"""
from btransformer.transformer import TransformerConfig
#from llama.llama import LlamaConfig
# Do logging levels
import os
LOG_LEVEL = os.environ.get('LLMCOMP_LOG_LEVEL', 'INFO')

LOGGING_CONFIG = {
    'version' : 1,
    'formatters' : 
        {'standard' : {
            'format' : '%(asctime)s %(name)s [%(levelname)s] %(message)s'},
        },
    'handlers' : 
        {'console' : {
            'class' : 'logging.StreamHandler',
            'formatter' : 'standard',
            'level' : LOG_LEVEL,
        },
         'file' : {
             'class' : 'logging.FileHandler',
             'formatter' : 'standard',
             'filename' : 'debug.log',
             'level' : LOG_LEVEL,
         }},
    'loggers' : {
        '' : {
            'handlers' : ['console', 'file'],
            'level' : LOG_LEVEL,
            'propagate' : False,
        },
        'compressor' : {
            'handlers' : ['console', 'file'],
            'level' : LOG_LEVEL,
            'propagate' : False,
        },
        'get_data' : {
            'handlers' : ['console', 'file'],
            'level' : LOG_LEVEL,
            'propagate' : False,
        },
        'language_model' : {
            'handlers' : ['console', 'file'],
            'level' : LOG_LEVEL,
            'propagate' : False,
        },
        # Add as needed
        'btransformer.train' : {
            'handlers' : ['console', 'file'],
            'level' : LOG_LEVEL,
            'propagate' : False,
        },
        'llama.compress' : {
            'handlers' : ['console', 'file'],
            'level' : LOG_LEVEL,
            'propagate' : False,
        },
    }
}
TRANSFORMER_CONFIG = TransformerConfig(
        vocab_size = 128,
        embedding_dim = 128,
        num_heads = 8,
        num_layers = 16,
        emb_init_scale = 0.02,
        widening_factor = 4,
)

#LLAMA_CONFIG = LlamaConfig()

CODER_PRECISION = 32

SNR_SD = [8879.514423764482,
 4993.317908106538,
 2807.949009541332,
 887.9514423764482,
 280.79490095413325,
 88.79514423764483,
 28.079490095413323]

SNR_SET = [
    0,
    5,
    10,
    20,
    30,
    40,
    50,
]

NUM_TEST_FILES = 10