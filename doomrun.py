# VizDoom installed via https://github.com/shakenes/vizdoomgym

import os

from stable_baselines.common.tf_util import linear_schedule
from stable_baselines import PPO2, PPO2_ACC
from stable_baselines.common import make_doom_env

for env_id in ["VizdoomMyWayHome-v0"]:
    for seed in [123,456]:
    # for seed in [457,728]:
    # for seed in [101, 729]:
        ########################### PPO ACC ############################
        # log_dir = "./logs/%s/final/PPO_seed%s_kernel32_transformer_rewschedule004_loss01_coefw1oversqrtT100_approxkl_lr0001_stackframes_nsteps2048_bs256_noavec" % (
        # env_id, seed)
        # os.makedirs(log_dir, exist_ok=True)
        # env = make_doom_env(env_id, 1, seed, monitor_dir=log_dir)
        # model = PPO2_ACC('CnnPolicy_ACC', env, verbose=1, seed=seed, vf_coef=0.5, tensorboard_log=log_dir,
        #                  n_steps=2048, nminibatches=8, acc_rew_coef=linear_schedule(0.04), acc_loss_coef=0.1,
        #                  use_avec=False, learning_rate=0.001, ent_coef=0.01,
        #                  max_steps=100)
        # model.learn(total_timesteps=10000000, tb_log_name="tb/PPO")

        ########################### PPO ############################
        log_dir = "./logs/%s/final/PPO_seed%s_lr0001_stackframes_nsteps2048_bs256_noavec" % (env_id, seed)
        os.makedirs(log_dir, exist_ok=True)
        env = make_doom_env(env_id, 1, seed, monitor_dir=log_dir)
        model = PPO2('CnnPolicy', env, verbose=1, seed=seed, vf_coef=0.5, tensorboard_log=log_dir,
                     n_steps=2048, nminibatches=8, learning_rate=0.001, ent_coef=0.01,
                     use_avec=False)
        model.learn(total_timesteps=10000000, tb_log_name="tb/PPO")
