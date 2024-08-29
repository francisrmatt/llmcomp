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
            #'stream' : 'ext://sys.stdout',
        },
         'file' : {
             'class' : 'logging.FileHandler',
             'formatter' : 'standard',
             'filename' : 'debug.log',
         }},
    'loggers' : {
        '' : {
            'handlers' : ['console', 'file'],
            'level' : 'INFO',
            'propagate' : False,
        },
    }
}

COMPRESSOR = 'gzip'