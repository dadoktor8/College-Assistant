import os
import logging
from datetime import datetime

class Logger:
    def __init__(self,log_dir="logs",log_level=logging.INFO):
        os.makedirs(log_dir,exist_ok=True)
        log_filename = os.path.join(log_dir,f"log_{datetime.now().strftime('%Y-%m-%d')}.txt")

        logging.basicConfig(
            level=log_level,
            format="%(asctime)s [%(levelname)s]: %(message)s",
            handlers=[
                logging.FileHandler(log_filename,encoding="utf8"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def info(self, message):
        self.logger.info(message)
    
    def error(self, message):
        self.logger.error(message)
    
    def warning(self, message):
        self.logger.warning(message)

    
if __name__ == "__main__":
    log = Logger
    log.info("Log initialized successfully")
    log.error("This is error")
    log.warning("This is a warning")