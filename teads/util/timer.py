import time

from teads.util.logger import StdoutLogger


class Timer:
    def __init__(self) -> None:
        self.start_time = None
        self.end_time = None
        self.logger = StdoutLogger()

    def start(self):
        self.start_time = time.time()

    def end(self):
        if self.start_time is None:
            self.logger.error("Timer.start has not been done.")
            return

        self.end_time = time.time()

    @property
    def result(self):
        if self.end_time is None:
            self.logger.error("Timer.end() has not been done.")
            return None

        return self.end_time - self.start_time
