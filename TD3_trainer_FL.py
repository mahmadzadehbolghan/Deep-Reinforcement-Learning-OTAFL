import numpy as np
from stable_baselines3 import TD3
import os
from MyLetterOTAFL_ddpg_3 import SnekEnv
import time
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from typing import Callable


models_dir = f"models/TD3_1/ris=6r1ho0.76nlr0.001/{int(time.time())}/"
logdir = f"logs/TD3_1/ris=6rho0.76lr0.001/{int(time.time())}/"
# logdir = "logs"
if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

env = SnekEnv()

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.001 * np.ones(n_actions))



env.reset()

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:11
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

model = TD3('MlpPolicy', env, action_noise=action_noise, learning_rate=linear_schedule(0.001),verbose=1, tensorboard_log=logdir)

TIMESTEPS = 1000
iters = 0
while True:
	iters += 1
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"DDPG")
	model.save(f"{models_dir}/{TIMESTEPS*iters}")