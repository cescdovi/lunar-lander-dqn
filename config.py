import os

# Entorno
ENV_NAME     = "LunarLander-v2"
SEED         = 42

# Hiperparámetros DQN (valores por defecto; pueden ser reemplazados por Optuna)
GAMMA                   = 0.99
LEARNING_RATE           = 1e-3
BATCH_SIZE              = 64
MEM_LENGTH              = 100000
TARGET_UPDATE           = 1000       # cada cuántos pasos sincronizar target_net
EPSILON_START           = 1.0
EPSILON_END             = 0.01
EPS_DECAY_STEPS         = 300000        # pasos para llegar a EPS_END

# Entrenamiento
MAX_EPISODES            = 500
MAX_STEPS_PER_EPISODE   = 1000
EVAL_INTERVAL           = 50         # cada cuántos episodios hago evaluación

# Rutas
RESULTS_DIR       = "results"
CHECKPOINT_DIR    = os.path.join(RESULTS_DIR, "checkpoints")
MLFLOW_TRACKING_URI = None   # Si se quiere apuntar a servidor remoto, p.ej. "http://localhost:5000"
MLFLOW_EXPERIMENT  = "LunarLander_DQN"

# Streamlit
STREAMLIT_PORT   = 8501

# Semilla global
def set_global_seed(seed: int):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
