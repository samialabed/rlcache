import logging

# TODO maybe make this a yml?


LOG_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': logging.INFO,
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',  # Default is stderr
        },
        'errors_to_file': {
            'level': logging.WARN,
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': 'errors.log'
        }
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['default', 'errors_to_file'],
            'level': logging.INFO,
            'propagate': True
        },
        'tensorflow': {
            'handlers': ['default'],
            'level': logging.WARN,
            'propagate': False
        }
    }
}


def create_file_logger(name: str, result_dir: str):
    fmt = logging.Formatter('%(asctime)s,%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler = logging.FileHandler(f'{result_dir}/{name}.log')
    handler.setFormatter(fmt)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger
