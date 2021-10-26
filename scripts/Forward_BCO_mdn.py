
# In[2]:


import torch
import gym
import argparse
import os
import pickle
import numpy as np
import random
import torch.optim as optim
import torch_optimizer as th_optim
import pybullet_envs

from torch.utils.data import DataLoader, ConcatDataset
from torch.distributions import Normal
from models import *
from utils import *
from dataset import *


# In[3]:


import matplotlib.pyplot as plt
import numpy as np

def plot_simple_bco(records, records_f, s=100, off=0.8):
    ypoints = np.array(records)
    plt.plot(ypoints, linestyle = 'dotted')
    ypoints[1:] = ypoints[1:] #- (np.random.rand(ypoints[1:].shape[0])-0.25)*2*s + (np.random.rand(ypoints[1:].shape[0])-off)*4*s
    plt.plot(ypoints, linestyle = 'dotted')
    plt.show()


# In[4]:


def inv_model_training(transitions, inv_model, ep_num=100):
    inv_dataset = transition_dataset(transitions)
    inv_dataset_list.append(inv_dataset)
    inv_dataset_final = ConcatDataset(inv_dataset_list)
    inv_loader = DataLoader(inv_dataset_final, batch_size=1024, shuffle=True, num_workers=1)

    inv_opt = optim.Adam(inv_model.parameters(), lr=1e-3, weight_decay=0.0001)
    """
    inv_opt_yogi = th_optim.Yogi(
        inv_model.parameters(),
        lr= 1e-2,
        betas=(0.9, 0.999),
        eps=1e-3,
        initial_accumulator=1e-6,
        weight_decay=0,
    )

    inv_opt = th_optim.Lookahead(inv_opt_yogi,  alpha=0.5)#k=5
    """
    inv_loss = nn.MSELoss()
    #inv_loss = nn.L1Loss()

    for epoch in range(ep_num): 
        running_loss = 0
        for i, data in enumerate(inv_loader):
            s, a, s_prime = data
            inv_opt.zero_grad()
            """
            a_pred = inv_model(s.float(), s_prime.float())
            """
            #print(s.shape, a.shape, s_prime.shape, torch.cat((s.float(), a.float()), dim=1).shape)
            try:
                inputs = torch.cat((s.float(), a.float()), dim=1)
                pred = inv_model.reparam_forward(inputs)
            except:
                sprime_m, sprime_v = inv_model(s.float(), a.float())
                pred = Normal(sprime_m, sprime_v).rsample()
            loss = inv_loss(pred, s_prime.float())
            #loss2 = inv_model.mdn_loss(inputs, s_prime.float())
            #loss = loss1 #+ loss2
            #loss = inv_loss(a_pred, a.float())
            loss.backward()
            running_loss += loss.item()
            if i%100 == 99:
                running_loss = 0
            inv_opt.step()
        if epoch%20==0:
            print('Epoch:%d Batch:%d Loss:%.5f'%(epoch, i+1, loss))
    print('Done!')
    return inv_model

def train_bc(trajs, policy, dynamics,  ep_num=50, sample_itr=500):
    bc_dataset = imitation_dataset(trajs)
    bc_loader = DataLoader(bc_dataset, batch_size=1024, shuffle=True, num_workers=1)
    print('Learning policy....')
    #bc_opt = optim.Adam(policy.parameters(), lr=1e-3, weight_decay=0.0001)
    #"""
    bc_opt_yogi = th_optim.Yogi(
        policy.parameters(),
        lr= 1e-2,
        betas=(0.9, 0.999),
        eps=1e-3,
        initial_accumulator=1e-6,
        weight_decay=0,
    )

    bc_opt = th_optim.Lookahead(bc_opt_yogi,  alpha=0.5)#k=5,
    #"""
    bc_loss = nn.MSELoss()
    # bc_loss = nn.L1Loss()

    for epoch in range(ep_num):  
        running_loss = 0
        for i, data in enumerate(bc_loader):
            s, s_prime = data
            bc_opt.zero_grad()
            
            #"""
            #a_pred = policy.reparam_forward(s.float())
            try:
                a_mu, a_sigma = policy(s.float())
                a_pred = Normal(loc=a_mu, scale=a_sigma).rsample()
            except:
                a_pred = policy.reparam_forward(s.float())
            #"""
            #print(torch.cat((s, a_pred), dim=1).shape)
            try:
                preds = dynamics.reparam_forward(torch.cat((s, a_pred), dim=1)) 
            except:
                d_m, d_v = dynamics(s, a_pred) 
                preds = Normal(d_m, d_v).rsample()
            loss = bc_loss(preds, s_prime)
            #print(loss, loss.shape, preds.shape, new_gts.shape)
            
            #for sid in range(sample_itr):
                #"""
                #a_mu, a_sigma = policy(s.float())
                #a_pred = Normal(loc=a_mu, scale=a_sigma).rsample()
                #"""
                #s_m, s_v = dynamics(s, a_pred) 
                #print(pred.shape, s.shape, s_prime.shape)
                #print(pred[0])
                #loss = loss + bc_loss(preds[sid], s_prime)
                
            running_loss += loss.item()
            loss.backward()
            if i%20 == 19:
                running_loss = 0
            bc_opt.step()
        if epoch%10==0:
            print('Epoch:%d Batch:%d Loss:%.3f'%(epoch, i+1, loss))

    print('Done!')
    return policy


def load_demos(DEMO_DIR):
    """load demonstrations"""
    try:
        trajstrajs = np.load("experts/states_expert_walker_.npy")[:10]
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


# In[5]:


env_list = ["Pendulum-v0", "BipedalWalker-v3", "Walker2DBulletEnv-v0", "HopperBulletEnv-v0", "HalfCheetahBulletEnv-v0", "AntBulletEnv-v0", "HumanoidBulletEnv-v0"]

runs = 20
inv_samples = 1000
max_steps = 800
expert_path='experts/'
weight_path="weights/"
        
test_rewards_envs = []
record_folder = "records/forward/"
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

        """init models"""
        policy = MDN(obs_dim, out_features=act_dim, n_hidden=32,  num_gaussians=4)
        #policy = policy_continuous(env.observation_space.shape[0],64,env.action_space.shape[0], uncertain=True)#.cuda()
        #inv_model = MDN(in_features=obs_dim+act_dim, out_features=obs_dim, n_hidden=32,  num_gaussians=10)
        inv_model = forward_dynamics_continuous(env.observation_space.shape[0],100,env.action_space.shape[0], uncertain=True)#.cuda()

        inv_model_best = None
        reward_best = -1000

        inv_dataset_list = []
        use_policy = False

        transitions = []
        test_rewards = []
        for steps in range(runs):
            print('######## STEP %d #######'%(steps+1))
            ### GET SAMPLES FOR LEARNING INVERSE MODEL
            print('Collecting transitions for learning inverse model....')
            if steps > 0:
                use_policy = True


            trans_samples, avg_reward = gen_inv_samples(env, policy.cpu(), M, 'continuous', use_policy, max_steps=max_steps)
            transitions = transitions+trans_samples

            f = open(record_fn, "a+")
            f.write(str(avg_reward) + "\n")
            f.close()

            """
            if len(transitions) > 92000:
                transitions = random.sample(transitions,92000)
            """
            test_rewards.append(avg_reward)
            print('Done!', np.array(transitions).shape)

            ### LEARN THE INVERSE MODEL
            inv_model = forward_dynamics_continuous(env.observation_space.shape[0],100,env.action_space.shape[0], uncertain=True)#.cuda()
            #inv_model = MDN(in_features=obs_dim+act_dim, out_features=obs_dim, n_hidden=32,  num_gaussians=10)#forward_dynamics_continuous(env.observation_space.shape[0],100,env.action_space.shape[0], uncertain=True)#.cuda()

            print('Learning inverse model....')
            inv_model = inv_model_training(transitions, inv_model)

            ### GET ACTIONS FOR DEMOS
            inv_model.cpu()
            print('Getting labels for demos....')
            trajs = get_state_labels(demos)
            print('Done!')


            ### PERFORM BEHAVIORAL CLONING
            policy = train_bc(trajs, policy, inv_model, sample_itr=400)

        torch.save(policy, weight_path+ENV_NAME+str(itr_id)+'.pt')
        test_rewards_envs.append(test_rewards)


# In[ ]:




