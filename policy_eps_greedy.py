import torch
import random
from config import DEVICE, EPSILON_START, EPSILON_END, EPS_DECAY_STEPS

class EpsilonGreedyPolicy:
    def __init__(self,
                 action_dim: int,
                 q_model: torch.nn.Module,
                 start: float = EPSILON_START, 
                 end: float = EPSILON_END, 
                 decay_steps: int = EPS_DECAY_STEPS
                 ):
        """
        Inicializa la política epsilon-greedy.
        """
        self.start = start              
        self.end = end                  
        self.decay_steps = decay_steps  
        self.steps_done = 0  
        self.q_model = q_model            
        self.action_dim = action_dim    

    def get_epsilon(self) -> float:
        # Decaimiento lineal:
        frac = max(0, (self.decay_steps - self.steps_done)) / self.decay_steps
        return self.end + (self.start - self.end) * frac

    def select_action(self, state: torch.Tensor) -> int:
        
        eps_threshold = self.get_epsilon()
        self.steps_done += 1

        if random.random() < eps_threshold:
            # Explorar
            action = random.randrange(self.action_dim)
        else:
            # Explotar
            with torch.no_grad():
                state = state.unsqueeze(0).to(DEVICE)
                q_values = self.q_model(state)
                action = q_values.max(1)[1].item() 
        return action
