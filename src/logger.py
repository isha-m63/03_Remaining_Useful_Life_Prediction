import logging
import os
from datetime import datetime

FORMAT = "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

try:
    logs_path = os.path.join(os.getcwd(), 'logs', LOG_FILE)
    os.makedirs(logs_path, exist_ok=True)
    LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)
    logging.basicConfig(filename=LOG_FILE_PATH, format=FORMAT, level=logging.INFO)
except OSError:
    #Read-only file system (e.g. some hosted environments): log to stderr instead
    logging.basicConfig(format=FORMAT, level=logging.INFO)

if __name__ == "__main__":
    logging.info('Logging has started')
