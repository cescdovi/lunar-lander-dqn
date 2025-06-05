# 4. Función de optimización para Double DQN
import torch
from torch import nn
import torch.nn.functional as F

def optimize_model(memory, Q_MODEL, T_MODEL, optimizer,
                   batch_size, device, gamma):
    if len(memory) < batch_size:
        return

    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    next_state_batch = torch.cat(batch.next_state)
    done_batch = torch.cat(batch.done)

    # 4.1. Predicciones actuales Q(s_t, a_t; theta)
    state_action_values = Q_MODEL(state_batch).gather(1, action_batch)

    # 4.2. Double DQN: Selección con Q_MODEL, evaluación con T_MODEL
    with torch.no_grad():
        # Seleccionamos la acción que maximiza Q con la red online
        next_state_best_actions = Q_MODEL(next_state_batch).max(1)[1].unsqueeze(1)  # (batch_size x 1)
        # Evaluamos esas acciones con la red objetivo
        next_state_values = T_MODEL(next_state_batch).gather(1, next_state_best_actions)
        # Si es terminal, anulamos valor futuro
        next_state_values = next_state_values * (1 - done_batch)
        # Construimos el target
        target_values = reward_batch + (gamma * next_state_values)

    # 4.3. Cálculo de pérdida MSE y optimización
    loss = F.mse_loss(state_action_values, target_values)

    optimizer.zero_grad()
    loss.backward()
    # Clipping de gradientes (opcional, pero generalmente útil)
    for param in Q_MODEL.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return loss
