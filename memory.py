from collections import deque

from typing import Deque


class BufferWindowMemory:
    """A simple in-memory short-term memory store for interactive chat sessions."""

    def __init__(self, max_messages: int):
        assert max_messages > 0, "Maximum messages must be positive."

        self.messages: Deque[dict] = deque()

        self.max_messages: int = max_messages

    def add_message(self, message: dict) -> None:
        """Add a message to the chat history."""

        self.messages.append(message)

        while len(self.messages) > self.max_messages:
            _ = self.messages.popleft()

    def get_history(self) -> list[dict]:
        """Return the most recent chat history."""

        return list(self.messages)
