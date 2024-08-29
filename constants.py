"""Project wide default values"""


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
    }
}

COMPRESSOR = 'gzip'
ALPHABET_SIZE = 256