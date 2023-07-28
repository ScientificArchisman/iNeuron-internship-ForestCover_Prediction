import logging
import os
import datetime


LOG_FILE = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}log.log"
LOG_PATH = os.path.join(os.getcwd(), "logs", LOG_FILE)

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO, 
    format="%(asctime)s:%(levelname)s:%(message)s"
)

if __name__ == "__main__":
    logging.info("This is an info message")
    logging.warning("This is a warning message")
    logging.error("This is an error message")
    logging.critical("This is a critical message")