from collections import deque

class StreamingState:

    def __init__(self, max_size=168):

        self.buffer = deque(
            maxlen=max_size
        )

    def update(self, event):

        self.buffer.append(event)

    def get_state(self):

        return list(self.buffer)

    def size(self):

        return len(self.buffer)
