import unittest

from collections import deque

from src.nope_gpt.memory import BufferWindowMemory


class TestBufferWindowMemory(unittest.TestCase):
    def test_init(self):
        # Test with valid max_messages
        memory = BufferWindowMemory(max_messages=5)
        self.assertEqual(memory.max_messages, 5)
        self.assertIsInstance(memory.messages, deque)
        self.assertEqual(len(memory.messages), 0)

        # Test with invalid max_messages
        with self.assertRaises(AssertionError):
            BufferWindowMemory(max_messages=0)

        with self.assertRaises(AssertionError):
            BufferWindowMemory(max_messages=-1)

    def test_add_message(self):
        memory = BufferWindowMemory(max_messages=3)

        # Add a single message
        message1 = {"role": "user", "content": "Hello"}
        memory.add_message(message1)

        self.assertEqual(len(memory.messages), 1)
        self.assertEqual(memory.messages[0], message1)

        # Add multiple messages
        message2 = {"role": "assistant", "content": "Hi there"}
        message3 = {"role": "user", "content": "How are you?"}

        memory.add_message(message2)
        memory.add_message(message3)

        self.assertEqual(len(memory.messages), 3)
        self.assertEqual(list(memory.messages), [message1, message2, message3])

    def test_add_message_with_overflow(self):
        memory = BufferWindowMemory(max_messages=2)

        message1 = {"role": "user", "content": "Message 1"}
        message2 = {"role": "assistant", "content": "Message 2"}
        message3 = {"role": "user", "content": "Message 3"}

        # Add messages that will exceed the limit
        memory.add_message(message1)
        memory.add_message(message2)
        memory.add_message(message3)

        # Check that only the most recent messages are kept (up to max_messages)
        self.assertEqual(len(memory.messages), 2)
        self.assertEqual(list(memory.messages), [message2, message3])

        # Add one more message to ensure oldest messages continue to be removed
        message4 = {"role": "assistant", "content": "Message 4"}
        memory.add_message(message4)

        self.assertEqual(len(memory.messages), 2)
        self.assertEqual(list(memory.messages), [message3, message4])

    def test_get_history(self):
        memory = BufferWindowMemory(max_messages=3)

        # Test with empty history
        self.assertEqual(memory.get_history(), [])

        # Add some messages
        message1 = {"role": "user", "content": "Hello"}
        message2 = {"role": "assistant", "content": "Hi there"}

        memory.add_message(message1)
        memory.add_message(message2)

        # Test with some history
        history = memory.get_history()

        self.assertEqual(len(history), 2)
        self.assertEqual(history, [message1, message2])

        # Verify that get_history returns a list (not a deque)
        self.assertIsInstance(history, list)

        # Verify that the returned history is a copy, not the internal reference
        history.append({"role": "test", "content": "This shouldn't affect memory"})
        self.assertEqual(len(memory.messages), 2)  # Should still be 2

    def test_add_complex_messages(self):
        memory = BufferWindowMemory(max_messages=5)

        # Test with messages containing nested structures
        complex_message = {
            "role": "assistant",
            "content": "Here's the information",
            "metadata": {"confidence": 0.95, "sources": ["source1", "source2"]},
            "tool_calls": [{"name": "calculator", "input": {"a": 1, "b": 2}}],
        }

        memory.add_message(complex_message)

        history = memory.get_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0], complex_message)

        # Make sure nested structures are preserved
        self.assertEqual(history[0]["metadata"]["confidence"], 0.95)
        self.assertEqual(history[0]["tool_calls"][0]["name"], "calculator")


if __name__ == "__main__":
    unittest.main()
