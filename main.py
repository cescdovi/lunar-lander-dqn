import gymnasium as gym
import torch
import argparse

from train_agent import train
from eval_agent import eval

from config import ENV_NAME, SEED

def parse_args():
    parser = argparse.ArgumentParser(
        description="Realizar entrenamiento con ajuste optimo de hiperparametros (train), evaluar agente (eval)"
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["train", "eval"],
        help=(
            "train → realizar entrenamiento con ajuste óptimo de hiperparámetros\n"
            "eval  → evaluar agente"
        )
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pass

    #crear entorno y fijar semilla
    env = gym.make(ENV_NAME)
    env.seed(SEED)

    #leer dimensiones automáticas
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    if args.mode == "train":
        train(env, state_dim, action_dim)
    else:  # args.mode == "eval"
        evaluate(env, state_dim, action_dim)


