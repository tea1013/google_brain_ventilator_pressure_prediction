import logging
import os
import warnings
from abc import ABC, abstractmethod
from enum import Enum
from logging import getLogger
from typing import List


class LogLevel(Enum):
    INFO = logging.INFO
    DEBUG = logging.DEBUG
    WARNING = logging.WARNING
    ERROR = logging.ERROR


class Logger(ABC):
    @abstractmethod
    def default(self, messages: List[str]):
        pass

    @abstractmethod
    def info(self, message: str):
        pass

    @abstractmethod
    def warn(self, message: str):
        pass

    @abstractmethod
    def error(self, message: str):
        pass


class FileLogger(Logger):
    def __init__(self, file_name: str, level: LogLevel = LogLevel.DEBUG):
        os.makedirs("./log", exist_ok=True)
        if os.path.exists(f"./log/{file_name}.log"):
            os.remove(f"./log/{file_name}.log")

        logger = getLogger()
        logger.addHandler(logging.FileHandler(f"./log/{file_name}.log", "a"))
        logger.setLevel(level.value)
        self.logger = logger

    def default(self, messages: List[str]):
        for message in messages:
            self.logger.info(message)

    def info(self, message: str):
        message = f"[INFO] {message}"
        self.logger.info(message)

    def warn(self, message: str):
        message = f"[WARNING] {message}"
        self.logger.warning(message)

    def error(self, message: str):
        message = f"[ERROR] {message}"
        self.logger.error(message)


class StdoutLogger(Logger):
    def __init__(self) -> None:
        pass

    def default(self, messages: List[str]):
        for message in messages:
            print(message)

    def info(self, message: str):
        message = f"[INFO] {message}"
        print(message)

    def warn(self, message: str):
        message = f"[WARNING] {message}"
        warnings.warn(message)

    def error(self, message: str):
        message = f"[ERROR] {message}"
        print(message)
