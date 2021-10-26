import os
import gym
import pybullet_envs
import numpy as np
import pickle
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
env_name =  ENVS[3]
algo_name = 'sac' #td3
start_train = False
total_steps = 4000000



# Don't forget to save the VecNormalize statistics when saving the agent
log_dir = "weights/"
stats_path = os.path.join(log_dir, "vec_normalize_"+env_name+".pkl")
model_save_pth = log_dir + algo_name +"_"+env_name
tensorlog_path = "tmp/"+algo_name+"_"+env_name+"/"

callback = TensorboardCallback(check_freq=1, log_dir=log_dir)


# Load the saved statistics
env = gym.make(env_name)
#env = Monitor(env)
env = DummyVecEnv([lambda: env])

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

ep_reward = 0
ep_counts = 0
ep_steps = 1000
episodes = 50
step_count = 0

traj_records = []

#for e in range(episodes):
while len(traj_records)<50: 
    traj = []
    env.render()
    obs = env.reset()
    ep_reward = 0
    
    for s in range(ep_steps):

        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        step_dict = {'observation':obs, 'origin_observation':env.get_original_obs(), 'reward':reward, 'done':done}
        traj.append(step_dict)
        #print("origin:", obs)
        #print("normal:", env.get_original_obs())
        ep_reward += reward
        step_count+=1

        if done:
            print(ep_reward, done)
            break

    if ep_reward > 1500:
        traj_records.append(traj)

#print(traj_records)
#print(traj_records[0])

demo_name = "demo/"+env_name+".pkl"

filehandler = open(demo_name,"wb")
pickle.dump(traj_records,filehandler)
filehandler.close()
"""
file = open(demo_name,'rb')
object_file = pickle.load(file)
print(object_file)
file.close()
"""

