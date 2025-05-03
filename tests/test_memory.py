import unittest

from collections import deque
from memory import ShortTermMemory


class TestShortTermMemory(unittest.TestCase):
    """Test cases for the ShortTermMemory class."""

    def setUp(self):
        """Set up a memory instance for testing."""
        self.max_tokens = 100
        self.memory = ShortTermMemory(max_tokens=self.max_tokens)

    def test_initialization(self):
        """Test that the memory initializes correctly."""
        self.assertEqual(self.memory.max_tokens, 100)
        self.assertEqual(self.memory.total_tokens, 0)
        self.assertIsInstance(self.memory.messages, deque)
        self.assertEqual(len(self.memory.messages), 0)

    def test_add_message(self):
        """Test adding a single message."""
        message = [1, 2, 3, 4, 5]
        self.memory.add_message(message)

        self.assertEqual(len(self.memory.messages), 1)
        self.assertEqual(self.memory.total_tokens, 5)
        self.assertEqual(list(self.memory.messages)[0], message)

    def test_add_multiple_messages(self):
        """Test adding multiple messages."""
        message1 = [1, 2, 3]
        message2 = [4, 5, 6]

        self.memory.add_message(message1)
        self.memory.add_message(message2)

        self.assertEqual(len(self.memory.messages), 2)
        self.assertEqual(self.memory.total_tokens, 6)
        self.assertEqual(list(self.memory.messages)[0], message1)
        self.assertEqual(list(self.memory.messages)[1], message2)

    def test_utilization_property(self):
        """Test the utilization property."""
        # Empty memory
        self.assertEqual(self.memory.utilization, 0.0)

        # Add half-capacity
        message = [i for i in range(50)]  # 50 tokens
        self.memory.add_message(message)
        self.assertEqual(self.memory.utilization, 0.5)

        # Add to full capacity
        self.memory.add_message(message)  # Another 50 tokens
        self.assertEqual(self.memory.utilization, 1.0)

        # Add beyond capacity (should maintain at or below max)
        self.memory.add_message([101])
        self.assertLessEqual(self.memory.utilization, 1.0)

    def test_automatic_memory_cleanup(self):
        """Test that old messages are removed when memory is full."""
        # Add messages until we reach capacity
        message1 = [1, 2, 3, 4, 5]  # 5 tokens
        message2 = [6, 7, 8, 9, 10]  # 5 tokens
        large_message = [i for i in range(95)]  # 95 tokens

        # Fill memory to 10 tokens
        self.memory.add_message(message1)
        self.memory.add_message(message2)

        # Add a large message that will push out older ones
        self.memory.add_message(large_message)

        # Check that we've removed the oldest message
        self.assertEqual(len(self.memory.messages), 2)
        self.assertEqual(self.memory.total_tokens, 100)  # Still at capacity
        self.assertEqual(
            list(self.memory.messages)[0], message2
        )  # message1 was removed
        self.assertEqual(list(self.memory.messages)[1], large_message)

    def test_cleanup_multiple_messages(self):
        """Test that multiple old messages are removed if necessary."""
        small_messages = [[i] for i in range(20)]  # 20 messages of 1 token each
        large_message = [i for i in range(90)]  # 90 tokens

        # Add all small messages
        for msg in small_messages:
            self.memory.add_message(msg)

        self.assertEqual(self.memory.total_tokens, 20)
        self.assertEqual(len(self.memory.messages), 20)

        # Add large message - should remove most small messages
        self.memory.add_message(large_message)

        self.assertEqual(self.memory.total_tokens, 100)  # At capacity
        self.assertEqual(len(self.memory.messages), 11)  # 10 small msgs + large msg
        self.assertEqual(list(self.memory.messages)[-1], large_message)

    def test_get_history(self):
        """Test retrieving the chat history."""
        message1 = [1, 2, 3]
        message2 = [4, 5, 6]

        self.memory.add_message(message1)
        self.memory.add_message(message2)

        expected_history = [1, 2, 3, 4, 5, 6]
        actual_history = list(self.memory.get_history())

        self.assertEqual(actual_history, expected_history)

    def test_get_history_empty(self):
        """Test retrieving history when memory is empty."""
        history = list(self.memory.get_history())
        self.assertEqual(history, [])

    def test_max_tokens_edge_case(self):
        """Test behavior when a single message exceeds max_tokens."""
        # Create a message larger than max_tokens
        large_message = [i for i in range(150)]  # 150 tokens

        # This should still add the message but immediately trigger cleanup
        self.memory.add_message(large_message)

        # The message should be added but then removed since it exceeds capacity
        self.assertEqual(len(self.memory.messages), 0)
        self.assertEqual(self.memory.total_tokens, 0)

        # Now add a small message first, then the large one
        small_message = [1, 2, 3]  # 3 tokens
        self.memory.add_message(small_message)
        self.memory.add_message(large_message)

        # Small message should be removed, large message should be too big to keep
        self.assertEqual(len(self.memory.messages), 0)
        self.assertEqual(self.memory.total_tokens, 0)

    def test_zero_max_tokens(self):
        """Test with max_tokens set to zero."""
        memory = ShortTermMemory(max_tokens=0)

        # Adding a message should immediately remove it
        message = [1, 2, 3]
        memory.add_message(message)

        self.assertEqual(len(memory.messages), 0)
        self.assertEqual(memory.total_tokens, 0)

    def test_empty_message(self):
        """Test adding an empty message."""
        empty_message = []
        self.memory.add_message(empty_message)

        self.assertEqual(len(self.memory.messages), 1)
        self.assertEqual(self.memory.total_tokens, 0)

        # Check that we can retrieve the empty message
        history = list(self.memory.get_history())
        self.assertEqual(history, [])


if __name__ == "__main__":
    unittest.main()
