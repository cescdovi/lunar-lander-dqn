
# eval_agent.py

import torch
import numpy as np
import gymnasium as gym
from dqn import DQN
from config import DEVICE, MAX_STEPS_PER_EPISODE

def evaluate(env: gym.Env,
             state_dim: int,
             action_dim: int,
             model_path: str = "results/dqn_best.pt",
             num_episodes: int = 5,
             render: bool = False) -> float:
    """
    Evalúa un agente DQN ya entrenado en el entorno `env`.

    Parámetros:
    -----------
    - env        : instancia de gym.Env (ej. gym.make("LunarLander-v2"))
    - state_dim  : dimensión del espacio de observación (ej. env.observation_space.shape[0])
    - action_dim : número de acciones discretas (ej. env.action_space.n)
    - model_path : ruta al archivo .pt con los pesos guardados del DQN
    - num_episodes: cuántos episodios ejecutar para promediar la recompensa
    - render     : si True, llamará a env.render() en cada paso para mostrar la simulación

    Retorna:
    --------
    - reward_avg : recompensa media obtenida sobre los `num_episodes` ejecutados.
    """

    # 1) Construimos la red con la misma arquitectutra que en entrenamiento
    Q_MODEL = DQN(input_dim=state_dim, output_dim=action_dim)

    # 2) cargar pesps
    Q_MODEL.load_state_dict(torch.load(model_path, map_location = DEVICE))
    Q_MODEL.to(DEVICE)
    Q_MODEL.eval()  #modo evaluación (desactiva dropout, batchnorm...)

    rewards_por_episodio = []

    for ep in range(1, num_episodes + 1):
        state = env.reset()
        total_reward = 0.0 #reward de episodio actual

        for time_step in range(1, MAX_STEPS_PER_EPISODE):
            # 3) Seleccionamos acción greedy: argmax Q(s, a)
            state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                q_values = Q_MODEL(state_tensor)
            action = torch.argmax(q_values, dim=1).item()

            # 4) Interactuamos con el entorno
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            if render:
                env.render()

            state = next_state

        rewards_por_episodio.append(total_reward)
        print(f"[Eval] Episodio {ep}/{num_episodes} → Recompensa: {total_reward:.2f}")

    env.close()

    reward_avg = np.mean(rewards_por_episodio)
    print(f"[Eval] Recompensa media en {num_episodes} episodios: {reward_avg:.2f}\n")

    return reward_avg
