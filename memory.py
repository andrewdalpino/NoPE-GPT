from collections import deque

from itertools import chain

from typing import Iterator, Deque


class ChatMemory:
    """A simple in-memory short-term memory store for interactive chat sessions."""

    def __init__(self, max_length: int):
        self.messages: Deque[list[int]] = deque()
        self.total_length = 0

        self.max_length = max_length

    def add_message(self, message: list[int]):
        """Add a message to the chat history."""

        self.messages.append(message)

        self.total_length += len(message)

        while self.total_length >= self.max_length:
            old_message = self.messages.popleft()

            self.total_length -= len(old_message)

    def get_history(self) -> Iterator[int]:
        """Return the most recent chat history."""

        return chain.from_iterable(self.messages)
