# train_agent.py

import os
import torch
import torch.nn.functional as F
from dqn import DQN
from replay_buffer import ReplayBuffer
from policy_eps_greedy import EpsGreedyPolicy

from config import EPSILON_START, EPSILON_END, EPS_DECAY_STEPS, LEARNING_RATE

def build_agent(state_dim: int,
                action_dim: int,
                device: torch.device,
                config: dict) -> dict:
    """
    Construye y devuelve un dict con:
      - 'q_online'     : red Q principal
      - 'q_target'  : red target (inicialmente idéntica a q_online)
      - 'optimizer' : optimizador de q_online (usando config['LR'])
      - 'buffer'    : ReplayBuffer(capacity=config['BUFFER_CAPACITY'], batch_size=config['BATCH_SIZE'])
      - 'policy'    : EpsGreedyPolicy(q_online, action_dim, eps_start=config['EPS_START'], eps_end=config['EPS_END'], eps_decay=config['EPS_DECAY'])
      - 'total_steps': 0
    """
    q_online = DQN(input_dim=state_dim, output_dim=action_dim).to(device)
    q_target = DQN(input_dim=state_dim, output_dim=action_dim).to(device)
    q_target.load_state_dict(q_online.state_dict())

    optimizer = torch.optim.Adam(q_online.parameters(), lr=LEARNING_RATE)
    buffer = ReplayBuffer()
    policy = EpsGreedyPolicy(q_model = q_online,
                             action_dim = action_dim,
                             eps_start = EPSILON_START,
                             eps_end = EPSILON_END,
                             eps_decay_steps = EPS_DECAY_STEPS)
    return {
        "q_online": q_online,
        "q_target": q_target,
        "optimizer": optimizer,
        "buffer": buffer,
        "policy": policy,
        "total_steps": 0
    }

