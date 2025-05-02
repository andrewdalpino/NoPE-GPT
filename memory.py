from collections import deque

from itertools import chain

from typing import Iterator, Deque


class ShortTermMemory:
    """A simple in-memory short-term memory store for interactive chat sessions."""

    def __init__(self, max_tokens: int):
        self.messages: Deque[list[int]] = deque()

        self.max_tokens: int = max_tokens
        self.total_tokens: int = 0

    @property
    def utilization(self) -> float:
        """Calculate the current memory utilization.

        Returns:
            float: The percentage of memory used.
        """

        return self.total_tokens / self.max_tokens

    def add_message(self, message: list[int]):
        """Add a message to the chat history.

        Args:
            message (list[int]): The token-encoded message to add.
        """

        self.messages.append(message)

        self.total_tokens += len(message)

        while self.total_tokens > self.max_tokens:
            old_message = self.messages.popleft()

            self.total_tokens -= len(old_message)

    def get_history(self) -> Iterator[int]:
        """Return the most recent chat history.

        Returns:
            Iterator[int]: An iterator over the chat history.
        """

        return chain.from_iterable(self.messages)
