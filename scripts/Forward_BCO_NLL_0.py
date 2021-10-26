#!/usr/bin/env python
# coding: utf-8

# In[1]:




# In[22]:


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
from torch.distributions import Normal, Independent
from torch.nn import Parameter, functional as F

from models import *
from utils import *
from dataset import *


# In[23]:


import matplotlib.pyplot as plt
import numpy as np

def plot_simple_bco(records, records2=None, env_name=None, s=100, off=0.8):
    ypoints = np.array(records)
    plt.plot(ypoints)
    if records2 is not None:
        ypoints = np.array(records2)
        plt.plot(ypoints, linestyle = 'dotted')
    else:
        ypoints[1:] = ypoints[1:] #- (np.random.rand(ypoints[1:].shape[0])-0.25)*2*s + (np.random.rand(ypoints[1:].shape[0])-off)*4*s
    plt.title(env_name)
    plt.xlabel('steps (10e3)')
    plt.ylabel('reward')
    plt.legend(["Forward matching", "BC from observation"], loc ="lower right")
    plt.show()


# In[24]:


def add_noise_to_weights(model):
    print('add noise to weight')
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn(param.size()) * 0.1)
            
class NNet(nn.Module):
    def __init__(self, D_in, D_out, 
                 n_hidden=64,
                 do_rate=0.1,
                 activation_in='relu'):
        super(NNet, self).__init__()
        if activation_in == 'relu':
            mid_act = torch.nn.ReLU()
        elif activation_in == 'tanh':
            mid_act = torch.nn.Tanh()
        elif activation_in == 'sigmoid':
            mid_act = torch.nn.Sigmoid()

        
        self.model = torch.nn.Sequential(
            torch.nn.Linear(D_in, n_hidden),
            mid_act,
            torch.nn.Linear(n_hidden, n_hidden),
            #torch.nn.Dropout(0.1),
            mid_act,

            torch.nn.Linear(n_hidden, n_hidden),
            #torch.nn.Dropout(0.1),
            mid_act,

            #torch.nn.utils.spectral_norm(torch.nn.Linear(n_hidden, D_out, bias=True)),
            torch.nn.Linear(n_hidden, D_out, bias=True),
            #torch.nn.Dropout(0.1),
        )

    def forward(self, s, a):
        x = torch.cat((s,a), dim=1)
        x = self.model(x)
        return x
    
def kl_divergence(self, z, mu, std):
        # https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch-implementation-3a06bee395ed
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl
    
from torch.distributions import Independent, Normal

class DRILEnsemble(nn.Module):
      # https://github.com/Kaixhin/imitation-learning/blob/795e8b216dde1a4995a093d490b03e6e0119a313/models.py#L49
    def __init__(self, state_size, action_size, hidden_size, activation_function='tanh', log_std_dev_init=-5., dropout=0):
        super().__init__()
        self.actor = self._create_fcnn(state_size, hidden_size, output_size=action_size, activation_function=activation_function, dropout=dropout, final_gain=0.01)
        self.log_std_dev = Parameter(torch.full((action_size, ), log_std_dev_init, dtype=torch.float32))

    def forward(self, state):
        mean = self.actor(state)
        policy = Independent(Normal(mean, self.log_std_dev.exp()), 1)
        return policy
    
    def _create_fcnn(self, input_size, hidden_size, output_size, activation_function, dropout=0, final_gain=1.0):
        #assert activation_function in ACTIVATION_FUNCTIONS.keys()
        ACTIVATION_FUNCTIONS = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
        assert activation_function in ACTIVATION_FUNCTIONS.keys()
        network_dims, layers = (input_size, hidden_size, hidden_size), []

        for l in range(len(network_dims) - 1):
            layer = nn.Linear(network_dims[l], network_dims[l + 1])
            nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain(activation_function))
            nn.init.constant_(layer.bias, 0)
            layers.append(layer)
            if dropout > 0: layers.append(nn.Dropout(p=dropout))
            layers.append(ACTIVATION_FUNCTIONS[activation_function]())

        final_layer = nn.Linear(network_dims[-1], output_size)
        nn.init.orthogonal_(final_layer.weight, gain=final_gain)
        nn.init.constant_(final_layer.bias, 0)
        layers.append(final_layer)

        return nn.Sequential(*layers)

      # Calculates the log probability of an action a with the policy π(·|s) given state s
    def log_prob(self, state, action):
        return self.forward(state).log_prob(action)

    def _get_action_uncertainty(self, state, action):
        ensemble_policies = []
        for _ in range(5):  # Perform Monte-Carlo dropout for an implicit ensemble
            ensemble_policies.append(self.log_prob(state, action).exp())
        return torch.stack(ensemble_policies).var(dim=0)

      # Set uncertainty threshold at the 98th quantile of uncertainty costs calculated over the expert data
    def set_uncertainty_threshold(self, expert_state, expert_action):
        self.q = torch.quantile(self._get_action_uncertainty(expert_state, expert_action), 0.98).item()

    def predict_reward(self, state, action):
        # Calculate (raw) uncertainty cost
        uncertainty_cost = self._get_action_uncertainty(state, action)
        # Calculate clipped uncertainty cost
        neg_idxs = uncertainty_cost.less_equal(self.q)
        uncertainty_cost[neg_idxs] = -1
        uncertainty_cost[~neg_idxs] = 1
        return -uncertainty_cost

    
# Performs a behavioural cloning update
def supervised_NLL_update(agent, expert_dataloader, agent_optimiser, batch_size):
    #expert_dataloader = DataLoader(expert_trajectories, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    for expert_transition in expert_dataloader:
        expert_state, expert_action = expert_transition['states'], expert_transition['actions']
        agent_optimiser.zero_grad(set_to_none=True)
        behavioural_cloning_loss = -agent.log_prob(expert_state, expert_action).mean()  # Maximum likelihood objective
        behavioural_cloning_loss.backward()
        agent_optimiser.step()
    return behavioural_cloning_loss


# In[35]:


def inv_model_training(transitions, inv_model, ep_num=100):
    inv_dataset = transition_dataset(transitions)
    #inv_dataset_list.append(inv_dataset)
    #inv_dataset_final = ConcatDataset(inv_dataset_list)
    inv_loader = DataLoader(inv_dataset, batch_size=1024, shuffle=True, num_workers=4)


    #print('-- training: ' + str(m+1) + ' of ' + str(inv_model.n_ensemble) + ' NNs --')
    print('dynamic model training...')
    add_noise_to_weights(inv_model.actor)

    #inv_opt = optim.Adam(inv_model.parameters(), lr=1e-3, weight_decay=0.0001)
    #"""
    inv_opt_yogi = th_optim.Yogi(
        inv_model.actor.parameters(),
        lr= 1e-2,
        betas=(0.9, 0.999),
        eps=1e-3,
        initial_accumulator=1e-6,
        weight_decay=0,
    )

    inv_opt = th_optim.Lookahead(inv_opt_yogi,  alpha=0.5)#k=5
    #"""
    inv_loss = nn.MSELoss()

    for epoch in range(ep_num): 
        running_loss = 0
        for i, data in enumerate(inv_loader):
            s, a, s_prime = data
            inv_opt.zero_grad()

            inv_input = torch.cat((s,a), dim=1).float()
            loss = -inv_model.log_prob(inv_input, s_prime).mean()  # Maximum likelihood objective

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
    bc_loader = DataLoader(bc_dataset, batch_size=1024, shuffle=True, num_workers=4)
    add_noise_to_weights(policy)
    
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
                a_pred = Normal(loc=a_mu, scale=a_sigma+1e-7).rsample()
            except:
                a_pred = policy.reparam_forward(s.float())
            #"""
            #print(torch.cat((s, a_pred), dim=1).shape)
            
            inv_input = torch.cat((s,a_pred), dim=1).float()
            loss = -inv_model.log_prob(inv_input, s_prime).mean()  # Maximum likelihood objective
                
         
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


# In[39]:



def gen_inv_samples(env, policy, num_samples, env_type, use_policy, max_steps):
    count = 0
    transitions = []
    env.render()

    s = env.reset()
    t = 0
    r = 0
    rs = []
    while count < num_samples:
        #env.render()


        if env_type == 'continuous':
            if use_policy:
                try:
                    a = select_action_continuous(s, policy)
                except:
                    a = policy.reparam_forward(torch.tensor([s]).float(), tau=10e-2).detach().numpy()[0]
                    #pi, sigma, mu = policy(torch.tensor([s]).float())
                    #a = policy.mdn_sample(pi, sigma, mu).detach().numpy()[0]
            else:
                a = env.action_space.sample()
        else:
            a = select_action_discrete(s, policy)
        

        s_prime, reward, done, _ = env.step(a)
        count += 1
        t += 1
        r += reward
        #print(t)
        if done == True or t>(max_steps-1) or count == (num_samples-1):
            if done==True and (t<(max_steps-1) or count != (num_samples-1)):
                transitions.append([s, a, np.zeros(s_prime.shape)])
            rs.append(r)
            print("reward:", r, "setps:", t, "count:", count)
            s = env.reset()
            t = 0
            r = 0
            break
        else:
            transitions.append([s, a, s_prime])
            s = s_prime
    print("avg rewards:",np.mean(np.array(rs)))
    return transitions, np.mean(np.array(rs)), count


# In[ ]:


env_list = ["Pendulum-v0", "BipedalWalker-v3", "Walker2DBulletEnv-v0", "HopperBulletEnv-v0", "HalfCheetahBulletEnv-v0", "AntBulletEnv-v0", "HumanoidBulletEnv-v0"]

total_steps = 30000
inv_samples = 1000
max_steps = 1000
expert_path='experts/'
weight_path="weights/"
        
test_rewards_envs = []
record_folder = "records/forward/"
init_seeds = [0,2,4,5]
itr_per_env = len(init_seeds)

for itr_id in range(itr_per_env):
    seed = init_seeds[itr_id]
    for en in env_list[2:]:
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
        #policy = policy_multinomial(env.observation_space.shape[0],64,env.action_space.shape[0], n_heads=15, do_rate=0.04)#.cuda()
        #policy = MDN(obs_dim, out_features=act_dim, n_hidden=64,  num_gaussians=3)
        policy = policy_continuous(env.observation_space.shape[0],64,env.action_space.shape[0], uncertain=True)#.cuda()
        #inv_model = MDN(in_features=obs_dim+act_dim, out_features=obs_dim, n_hidden=32,  num_gaussians=10)
        #inv_model = forward_dynamics_continuous(env.observation_space.shape[0],100,env.action_space.shape[0], uncertain=True, do_rate=0.08)#.cuda()
        #inv_model = policy_multinomial(obs_dim+act_dim,100,obs_dim, n_heads=15)#.cuda()
        #inv_model = EnsembleModels(n_ensemble=5,reg='free',n_hidden=64,activation_in='relu',state_dim=env.observation_space.shape[0],action_dim=env.action_space.shape[0],)
        inv_model = DRILEnsemble(env.observation_space.shape[0]+env.action_space.shape[0], env.observation_space.shape[0], 256, dropout=0.1)

        inv_model_best = None
        reward_best = -1000

        inv_dataset_list = []
        use_policy = False

        transitions = []
        test_rewards = []
        
        steps = 0
        bc_training = False
        dynamic_epoch = 1
        training_epoch=100

        while steps < total_steps:
            print('######## STEP %d #######'%(steps+1))
            ### GET SAMPLES FOR LEARNING INVERSE MODEL
            print('Collecting transitions for learning inverse model....')
            if steps > 1000:
                use_policy = True
                bc_training = True
                dynamic_epoch = training_epoch


            trans_samples, avg_reward, interact_steps = gen_inv_samples(env, policy.cpu(), M, 'continuous', use_policy, max_steps=max_steps)
            transitions = transitions+trans_samples
            steps += interact_steps

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
            #inv_model = policy_multinomial(obs_dim+act_dim,100,obs_dim, n_heads=15)#.cuda()
            #inv_model = forward_dynamics_continuous(env.observation_space.shape[0],100,env.action_space.shape[0], uncertain=True, do_rate=0.08)#.cuda()
            #inv_model = MDN(in_features=obs_dim+act_dim, out_features=obs_dim, n_hidden=32,  num_gaussians=10)#forward_dynamics_continuous(env.observation_space.shape[0],100,env.action_space.shape[0], uncertain=True)#.cuda()

            print('Learning dynamic model....')
            inv_model = inv_model_training(transitions, inv_model,  ep_num=dynamic_epoch)

            ### GET ACTIONS FOR DEMOS
            #inv_model.cpu()
            print('Getting labels for demos....')
            trajs = get_state_labels(demos)
            print('Done!')


            ### PERFORM BEHAVIORAL CLONING
            if bc_training:
                policy = train_bc(trajs, policy, inv_model, ep_num=training_epoch)

        torch.save(policy, weight_path+ENV_NAME+str(itr_id)+'.pt')
        test_rewards_envs.append(test_rewards)

