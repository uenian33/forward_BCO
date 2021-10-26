#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import gym
import argparse
import os
import pickle
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.distributions import Normal
from models import *
from utils import *
from dataset import *
import pybullet_envs
import random


def load_demos(DEMO_DIR):
    try:
        trajs = np.load("experts/states_expert_walker_0.npy")[:10]
    except:
        with open(DEMO_DIR, 'rb') as f:
            trajs = pickle.load(f)

    demos = []
    for t_id, traj in enumerate(trajs):
        demo =[]
        #print(t_id)
        for item in traj:    
            obs = item['observation']
            #obs = list(obs)
            #print(obs)
            demo.append(obs)
        #print(np.array(demo).shape)
        demos.append(np.array(demo))

    print(np.array(demos).shape)
    demos = demos[:10]
    return demos


# In[2]:


env_list = ["Pendulum-v0", "BipedalWalker-v3", "Walker2DBulletEnv-v0", "HopperBulletEnv-v0", "HalfCheetahBulletEnv-v0", "AntBulletEnv-v0", "HumanoidBulletEnv-v0"]

runs = 1
inv_samples = 1
max_steps = 1
expert_path='experts/'
weight_path="weights/"
        
test_rewards_envs = []
record_folder = "records/bco/"
init_seeds = [0,2,4,5]
itr_per_env = len(init_seeds)

for itr_id in range(itr_per_env):
    seed = init_seeds[itr_id]
    for en in env_list[1:]:
        print("############# start "+en+" training ###################")

        ENV_NAME = en#env_list[3]
        env=ENV_NAME
        
        DEMO_DIR = os.path.join(expert_path, env+'.pkl')
        M = inv_samples

        record_fn = record_folder + ENV_NAME + str(itr_id) + ".txt"

        """load demonstrations"""
        demos = load_demos(DEMO_DIR)

        """create environments"""
        env = gym.make(ENV_NAME)
        obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]

        """init random seeds for reproduction"""
        torch.manual_seed(seed)
        env.seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        env.action_space.seed(seed)

        policy = policy_continuous(env.observation_space.shape[0],64,env.action_space.shape[0])#.cuda()
        inv_model = inv_dynamics_continuous(env.observation_space.shape[0],100,env.action_space.shape[0])#.cuda()
        
        """start training"""
        inv_dataset_list = []
        use_policy = False

        transitions = []
        for steps in range(runs):
            print('######## STEP %d #######'%(steps+1))
            ### GET SAMPLES FOR LEARNING INVERSE MODEL
            print('Collecting transitions for learning inverse model....')
            if steps > 0:
                use_policy = True


            trans_samples, avg_reward = gen_inv_samples(env, policy.cpu(), M, 'continuous', use_policy,max_steps=max_steps)
            #print(np.array(trans_samples).shape, transitions)
            transitions = transitions+trans_samples
            
            f = open(record_fn, "a+")
            f.write(str(avg_reward) + "\n")
            f.close()
            
            print('Done!', np.array(transitions).shape)

            ### LEARN THE INVERSE MODEL
            print('Learning inverse model....')
            inv_dataset = transition_dataset(transitions)
            inv_dataset_list.append(inv_dataset)
            inv_dataset_final = ConcatDataset(inv_dataset_list)
            inv_loader = DataLoader(inv_dataset_final, batch_size=256, shuffle=True, num_workers=4)

            inv_opt = optim.Adam(inv_model.parameters(), lr=1e-3, weight_decay=0.0001)
            inv_loss = nn.MSELoss()
            #inv_loss = nn.L1Loss()

            for epoch in range(100): 
                running_loss = 0
                for i, data in enumerate(inv_loader):
                    s, a, s_prime = data
                    inv_opt.zero_grad()
                    a_pred = inv_model(s.float(), s_prime.float())
                    loss = inv_loss(a_pred, a.float())
                    loss.backward()
                    running_loss += loss.item()
                    if i%100 == 99:
                        print('Epoch:%d Batch:%d Loss:%.5f'%(epoch, i+1, running_loss/100))
                        running_loss = 0
                    inv_opt.step()
            print('Done!')

            ### GET ACTIONS FOR DEMOS
            inv_model.cpu()
            print('Getting labels for demos....')
            trajs = get_action_labels(inv_model, demos, 'continuous')
            print('Done!')
            bc_dataset = imitation_dataset(trajs)
            bc_loader = DataLoader(bc_dataset, batch_size=256, shuffle=True, num_workers=4)
            inv_model

            ### PERFORM BEHAVIORAL CLONING
            print('Learning policy....')
            policy
            bc_opt = optim.Adam(policy.parameters(), lr=1e-3, weight_decay=0.0001)
            bc_loss = nn.MSELoss()
            # bc_loss = nn.L1Loss()

            for epoch in range(50):  
                running_loss = 0
                for i, data in enumerate(bc_loader):
                    s, a = data
                    bc_opt.zero_grad()
                    """
                    a_mu, a_sigma = policy(s.float())
                    a_pred = Normal(loc=a_mu, scale=a_sigma).rsample()
                    """
                    a_pred = policy(s.float())
                    loss = bc_loss(a_pred, a)
                    running_loss += loss.item()
                    loss.backward()
                    if i%20 == 19:
                        running_loss = 0
                    bc_opt.step()
                if epoch%10==0:
                    print('Epoch:%d Batch:%d Loss:%.3f'%(epoch, i+1, loss))

            print('Done!')

        torch.save(policy, weight_path+ENV_NAME+str(itr_id)+'_bco.pt')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




