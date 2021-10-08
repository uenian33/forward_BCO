import gym
import torch
import numpy as np
from torch.distributions import Normal
from torch.distributions import Categorical

def select_action_continuous(state, policy):
    state = torch.from_numpy(state).float()
    try:
        if policy.uncertain:
            mean, sigma = policy(state)
            pi = Normal(loc=mean, scale=sigma)
            action = pi.sample()
        else:
            action = policy(state)  
    except:
        action = policy(state) 
    
    try:
        action.detach().numpy()
    except:
        print(action)
        print(policy.uncertain)
    print(action)
    
        
    return action.detach().numpy()

def select_action_discrete(state, policy):
    state = torch.from_numpy(state).float()
    probs = policy(state)
    pi = Categorical(probs)
    action = pi.sample()
    return action.detach().numpy()

def gen_inv_samples(env, policy, num_samples, env_type, use_policy, max_steps):
    count = 0
    transitions = []
    s = env.reset()
    t = 0
    r = 0
    rs = []
    while count < num_samples:
        
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
        transitions.append([s, a, s_prime])
        count += 1
        t += 1
        r += reward
        #print(t)
        if done == True or t>(max_steps-1) or count == (num_samples-1):
            rs.append(r)
            print("reward:", r, "setps:", t, "count:", count)
            s = env.reset()
            t = 0
            r = 0
        else:
            s = s_prime
    print("avg rewards:",np.mean(np.array(rs)))
    return transitions, np.mean(np.array(rs))

def get_action_labels(inv_model, state_trajs, env_type):
    trajs = []
    for state_traj in state_trajs:
        traj = []
        for idx in range(len(state_traj)-1):
            s = state_traj[idx]
            s_prime = state_traj[idx+1]
            #print(s, s_prime)
            a = inv_model(torch.from_numpy(s).unsqueeze(0).float(), torch.from_numpy(s_prime).unsqueeze(0).float())
            if env_type == 'continuous':
                traj.append([s, a.detach().numpy()])
            else:
                a = np.max(a.detach().numpy())
                traj.append([s, a])
        trajs.append(traj)
    return trajs

def get_state_labels(state_trajs):
    trajs = []
    for state_traj in state_trajs:
        traj = []
        for idx in range(len(state_traj)-1):
            s = state_traj[idx]
            s_prime = state_traj[idx+1]
            #print(s.shape, s_prime.shape)
            traj.append([s, s_prime])
            
        trajs.append(traj)
    return trajs

