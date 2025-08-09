from collections import deque

from itertools import chain

from typing import Iterator, Deque


class BufferWindowMemory:
    """A simple in-memory short-term memory store for interactive chat sessions."""

    def __init__(self, max_messages: int):
        """Initialize the memory buffer with a maximum number of messages.

        Args:
            max_messages (int): The maximum number of messages to store in memory.
        """

        assert max_messages > 0, "Maximum messages must be positive."

        self.messages: Deque[str] = deque()

        self.max_messages: int = max_messages

    def add_message(self, message: str) -> None:
        """Add a message to the chat history.

        Args:
            message (str): The message to add to the chat history.
        """

        self.messages.append(message)

        while len(self.messages) > self.max_messages:
            _ = self.messages.popleft()

    def get_history(self) -> Iterator[str]:
        """Return the most recent chat history.

        Returns:
            Iterator[str]: An iterator over the chat history.
        """

        return chain.from_iterable(self.messages)
