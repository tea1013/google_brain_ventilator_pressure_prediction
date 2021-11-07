from abc import ABC, abstractclassmethod
from enum import Enum

import slackweb


class NotificationPlatform(Enum):
    Slack = 0


class Notification(ABC):
    def __init__(self) -> None:
        pass

    @abstractclassmethod
    def notify(self, body: str) -> None:
        pass


class Slack(Notification):
    def __init__(self, url: str) -> None:
        self.slack = slackweb.Slack(url=url)

    def notify(self, body: str) -> None:
        self.slack.notify(text=body)
