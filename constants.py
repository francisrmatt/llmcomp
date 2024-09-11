"""Project wide default values"""
from btransformer.transformer import TransformerConfig
from llama.llama import LlamaConfig

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
            'level' : 'INFO',
        },
         'file' : {
             'class' : 'logging.FileHandler',
             'formatter' : 'standard',
             'filename' : 'debug.log',
             'level' : 'INFO',
         }},
    'loggers' : {
        '' : {
            'handlers' : ['console', 'file'],
            'level' : 'INFO',
            'propagate' : False,
        },
        'compressor' : {
            'handlers' : ['console', 'file'],
            'level' : 'INFO',
            'propagate' : False,
        },
        'get_data' : {
            'handlers' : ['console', 'file'],
            'level' : 'INFO',
            'propagate' : False,
        },
        'language_model' : {
            'handlers' : ['console', 'file'],
            'level' : 'INFO',
            'propagate' : False,
        },
        # Add as needed
        'btransformer.train' : {
            'handlers' : ['console', 'file'],
            'level' : 'INFO',
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

LLAMA_CONFIG = LlamaConfig()

CODER_PRECISION = 32