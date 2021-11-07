from teads.notebook.version import Version
from teads.util.logger import FileLogger, StdoutLogger
from teads.util.notification import Notification


class NotebookConfig:
    def __init__(
        self,
        version: Version,
        logger: StdoutLogger,
        file_logger: FileLogger,
        notification: Notification,
        seed: int,
        is_local: bool,
    ) -> None:
        self.version = version
        self.logger = logger
        self.file_logger = file_logger
        self.notification = notification
        self.seed = seed
        self.is_local = is_local

        self.file_logger.default(
            [
                "================Experiment==================",
                f"Version: {version.n}",
                version.description,
                "============================================",
                "",
            ]
        )

        self.notification.notify(f"Experiment [V{version.n}] Start.")
