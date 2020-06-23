# PyBullet / MuJoCo / Continuous Control

import os

from stable_baselines.common.tf_util import linear_schedule
import gym
from stable_baselines import PPO2, PPO2_ACC
from stable_baselines.common import make_bullet_env

for env_id in ["MinitaurBulletEnv-v0"]:
    for seed in [123]:
    # for seed in [123,456]:
    # for seed in [457,728]:
    # for seed in [101,729]:
        ########################### PPO ACC ############################
        log_dir = "./logs/%s/tb/PPO_seed%s_kernel32_transformer_rew0004_loss01_coefw1overT1000_approxkl_lr00002_nsteps2048_bs64_nepoch10_entcoef0_noavec" % (
            env_id, seed)
        os.makedirs(log_dir, exist_ok=True)
        env = make_bullet_env(env_id, 1, seed, monitor_dir=log_dir)
        model = PPO2_ACC('MlpPolicy_ACC', env, verbose=1, seed=seed, vf_coef=0.5, tensorboard_log=log_dir,
                         n_steps=2048, nminibatches=32, noptepochs=10, learning_rate=0.0002,
                         ent_coef=0.0, lam=0.95,
                         use_avec=False, max_steps=gym.make(env_id)._max_episode_steps,
                         acc_loss_coef=0.1, acc_rew_coef=0.004)
        model.learn(total_timesteps=2000000, tb_log_name="tb/PPO")

        ########################### PPO ############################
        # log_dir = "./logs/%s/tb/PPO_seed%s_lr00002_nsteps2048_bs64_nepoch10_entcoef0_noavec_noclipvf" % (env_id, seed)
        # os.makedirs(log_dir, exist_ok=True)
        # env = make_bullet_env(env_id, 1, seed, monitor_dir=log_dir)
        # model = PPO2('MlpPolicy', env, verbose=1, seed=seed, vf_coef=0.5, tensorboard_log=log_dir,
        #              n_steps=2048, nminibatches=32, noptepochs=10, learning_rate=0.0002,
        #              ent_coef=0.0, lam=0.95,
        #              use_avec=False)
        # model.learn(total_timesteps=2000000, tb_log_name="tb/PPO")
