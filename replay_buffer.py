import random
from collections import deque, namedtuple
from config import MEM_LENGTH, BATCH_SIZE

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:
    def __init__(self, 
                 capacity: int = MEM_LENGTH):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Almacena una experiencia en el buffer de experiencias.
        """
        self.memory.append(Transition(state, action, reward, next_state, done))

    def sample(self, 
               batch_size: int = BATCH_SIZE):
        """Devuelve un batch aleatorio de experiencias."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Devuelve longitud del buffer"""
        return len(self.memory)
