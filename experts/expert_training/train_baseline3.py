import os
import gym
import pybullet_envs
import numpy as np

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import TD3, SAC, PPO
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.monitor import Monitor

import sys

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(TensorboardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-3:])
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

                self.logger.record('reward', value)

        return True


ENVS = ['Walker2DBulletEnv-v0', 'AntBulletEnv-v0', 'HopperBulletEnv-v0', "HalfCheetahBulletEnv-v0", 
        'HumanoidBulletEnv-v0', "BipedalWalker-v3", 'LunarLanderContinuous-v2', "ThrowerBulletEnv-v0", "PusherBulletEnv-v0"]
env_name =  ENVS[2]
algo_name = 'sac' #td3
start_train = True
total_steps = 1500000

env = gym.make(env_name)
env = Monitor(env)
env = DummyVecEnv([lambda: env])
# Automatically normalize the input features and reward
env = VecNormalize(env, norm_obs=True, norm_reward=False,
                   clip_obs=1000., clip_reward=1000.)


# Don't forget to save the VecNormalize statistics when saving the agent
log_dir = "weights/"
stats_path = os.path.join(log_dir, "vec_normalize_"+env_name+".pkl")
model_save_pth = log_dir + algo_name +"_"+env_name
tensorlog_path = "tmp/"+algo_name+"_"+env_name+"/"

callback = TensorboardCallback(check_freq=1, log_dir=log_dir)


if start_train:
    # Train model
    # The noise objects for TD3
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    if algo_name=='td3':
        model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1, tensorboard_log=tensorlog_path)
    elif algo_name=='sac':
        model = SAC("MlpPolicy", env, action_noise=action_noise, verbose=1, tensorboard_log=tensorlog_path)
    elif algo_name=='ppo':
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tensorlog_path)

    model.learn(total_timesteps=total_steps)

    model.save(model_save_pth)
    env.save(stats_path)

    # To demonstrate loading
    del model, env

# Load the saved statistics
env = DummyVecEnv([lambda: gym.make(env_name)])
env = VecNormalize.load(stats_path, env)
#  do not update them at test time
env.training = False
# reward normalization is not needed at test time
env.norm_reward = False

# Load the agent
if algo_name=='td3':
    model = TD3.load(model_save_pth, env=env)
elif algo_name=='sac':
    model = SAC.load(model_save_pth, env=env)
elif algo_name=='ppo':
    model = PPO.load(model_save_pth, env=env)

env.render()
obs = env.reset()
ep_reward = 0
ep_counts = 0
ep_steps = 1000
step_count = 0
while ep_counts<ep_steps:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    print(obs, env.get_original_obs())
    ep_reward += reward
    step_count+=1
    env.render()
    if done or step_count>ep_steps:
        print(ep_reward, done)
        obs = env.reset()
        ep_reward = 0
        step_count = 0
    ep_counts+=1


# python train_baselines3.py Walker2DBulletEnv-v0