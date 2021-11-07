class ValueNoneError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message

    def __str__(self) -> str:
        return f"[{__class__}] {self.message}"


class NotOptimizedError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message

    def __str__(self) -> str:
        return f"[{__class__}] {self.message}"
