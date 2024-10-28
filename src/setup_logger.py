import logging

def logger(name:str)->logging.Logger:
    # Set up logging for the model building process
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Create a console handler to output logs to the console
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)

    # Define the log message format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
