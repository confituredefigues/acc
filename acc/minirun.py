# MiniGrid installed via https://github.com/maximecb/gym-minigrid

import os

from stable_baselines.common.tf_util import linear_schedule
from stable_baselines import PPO2
from stable_baselines import PPO2_ACC
from stable_baselines.common import make_vec_env
from gym_minigrid.wrappers import *

for env_id in ["MiniGrid-MultiRoom-N7-S4-v0"]:
    for seed in [457]:
    # for seed in [123,456,457]:
    # for seed in [101,729,728]:
        ########################### PPO ACC ############################
        log_dir = "./logs/%s/final/PPO_seed%s_kernel32_transformer_rewschedule004_loss01_coefw1oversqrtT100_approxkl_lr0001_stackframes_nsteps2048_bs256_noavec" % (
        env_id, seed)
        os.makedirs(log_dir, exist_ok=True)
        env = make_vec_env(env_id, 1, seed, monitor_dir=log_dir, wrapper_class=ImgObsWrapper)
        model = PPO2_ACC('CnnPolicy_ACC', env, verbose=1, seed=seed, vf_coef=0.5, tensorboard_log=log_dir,
                         n_steps=2048, nminibatches=8, acc_rew_coef=linear_schedule(0.04), acc_loss_coef=0.1,
                         use_avec=False, learning_rate=0.001,
                         max_steps=20 * 5)  # 7 corresponds to number of rooms (see gym-minigrid/gym_minigrid/envs/multiroom.py:38)
        model.learn(total_timesteps=10000000, tb_log_name="tb/PPO")

        ########################### PPO ############################
        # log_dir = "./logs/%s/final/PPO_seed%s_lr0001_stackframes_nsteps2048_bs256_noavec" % (env_id, seed)
        # os.makedirs(log_dir, exist_ok=True)
        # env = make_vec_env(env_id, 1, seed, monitor_dir=log_dir, wrapper_class=ImgObsWrapper)
        # model = PPO2('CnnPolicy', env, verbose=1, seed=seed, vf_coef=0.5, tensorboard_log=log_dir,
        #              n_steps=2048, nminibatches=8, learning_rate=0.001, ent_coef=0.01,
        #              use_avec=False)
        # model.learn(total_timesteps=10000000, tb_log_name="tb/PPO")
