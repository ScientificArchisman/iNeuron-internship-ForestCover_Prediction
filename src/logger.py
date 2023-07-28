import logging
import os
import datetime


LOG_FILE = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.log"
LOG_PATH = os.path.join(os.getcwd(), "hello")

os.makedirs(os.path.dirname(LOG_PATH), exist_ok = True)
print(LOG_PATH)

# LOGS_FILE_PATH = os.path.join(LOG_PATH, LOG_FILE)

# logging.basicConfig(
#     filename=LOGS_FILE_PATH,
#     level=logging.INFO, 
#     format="%(asctime)s:%(levelname)s:%(message)s"
# )
