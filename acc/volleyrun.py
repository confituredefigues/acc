#!/usr/bin/env python3

# Trains a convnet PPO agent to play SlimeVolley from pixels (SlimeVolleyNoFrameskip-v0)
# requires stable_baselines (I used 2.10)

# run with
# mpirun -np 96 python train_ppo_pixel.py (replace 96 with number of CPU cores you have.)

import os
import gym
import slimevolleygym

from mpi4py import MPI
from stable_baselines.common import set_global_seeds
from stable_baselines.common.atari_wrappers import ClipRewardEnv, NoopResetEnv, MaxAndSkipEnv, WarpFrame
from stable_baselines.common.policies import CnnPolicy
from stable_baselines import logger, PPO2_ACC
from stable_baselines.common.monitor import Monitor
from stable_baselines.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines.common.tf_util import linear_schedule

from slimevolleygym import FrameStack # doesn't use Lazy Frames, easier to debug


NUM_TIMESTEPS = int(2e8)
SEED = 831
EVAL_FREQ = 200000
EVAL_EPISODES = 1000
LOGDIR = "logs/SlimeVolley-v0/single/ppo2_acc_mlp_T1000_true2otherkl_s%s" % SEED # moved to zoo afterwards.


def make_env(seed):
  env = gym.make("SlimeVolley-v0")
  env.seed(seed)
  return env


def train():
  rank = MPI.COMM_WORLD.Get_rank()

  if rank == 0:
      logger.configure(folder=LOGDIR)

  else:
      logger.configure(format_strs=[])
  workerseed = SEED + 10000 * MPI.COMM_WORLD.Get_rank()
  set_global_seeds(workerseed)
  env = make_env(workerseed)

  env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
  env.seed(workerseed)

  model = PPO2_ACC('MlpPolicy_ACC', env, n_steps=4096, cliprange=0.2, ent_coef=0.0, noptepochs=10, max_steps=1000,
                   learning_rate=linear_schedule(3e-4), nminibatches=64, gamma=0.99, lam=0.95, verbose=1)

  eval_callback = EvalCallback(env, best_model_save_path=LOGDIR, log_path=LOGDIR, eval_freq=EVAL_FREQ,
                               n_eval_episodes=EVAL_EPISODES)

  model.learn(total_timesteps=NUM_TIMESTEPS, callback=eval_callback)

  env.close()
  del env
  if rank == 0:
      model.save(os.path.join(LOGDIR, "final_model"))  # probably never get to this point.


if __name__ == '__main__':
    train()
