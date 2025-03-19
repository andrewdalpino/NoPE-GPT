class ChatMemory:
    """A simple short-term memory store for a chat session."""

    def __init__(self, max_length: int):
        self.max_length = max_length
        self.messages = []
        self.total_length = 0

    def add_message(self, message: list[int]):
        """Add a message to the chat history."""

        self.messages.append(message)

        self.total_length += len(message)

        while self.total_length >= self.max_length:
            old_message = self.messages.pop(0)

            self.total_length -= len(old_message)

    def get_history(self) -> list[int]:
        """Return the most recent chat history."""

        history = []

        for message in self.messages:
            history.extend(message)

        return history
