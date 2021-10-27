import torch
import numpy as np
from torch.utils.data import Dataset
import pickle

def load_demos(DEMO_DIR, num=10):
    """load demonstrations"""
    try:
        trajstrajs = np.load("experts/states_expert_walker_.npy")[:num]
    except:
        with open(DEMO_DIR, 'rb') as f:
            trajs = pickle.load(f)
    demos = []
    for t_id, traj in enumerate(trajs):
        demo =[]
        #print(t_id)
        for item in traj:    
            obs = item['observation']
            demo.append(obs)
        #print(np.array(demo).shape)
        demos.append(np.array(demo))

    print(np.array(demos).shape)
    demos = demos[:10]
    return demos

class transition_dataset(Dataset):
    def __init__(self, transitions):
        self.data = []      
        for transition in transitions:
            s, a, s_prime = transition
            self.data.append([s, a, s_prime])   
            self.x = np.array([s, a]).flatten() 
            self.y = np.array(s_prime).flatten()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        s, a, s_prime = self.data[idx]
        return s, a, s_prime

class imitation_dataset(Dataset):
    def __init__(self, trajs, type="state", from_file=False):
        self.data = []
        for traj in trajs:
            #reward = 0
            for s_id, sample in enumerate(traj):
                if type=="state":
                    x, y = sample
                    if len(x.shape) > 1:
                        x = x[0]
                    if len(y.shape) > 1:
                        y = y[0]
                    self.data.append([x, y])
                else:
                    s = sample['observation']
                    a = sample['action']
                    self.data.append([s, a])
                    #print(sample)
                    #reward+=sample['reward']
            #print(reward)
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        s, a = self.data[idx]        
        return s, a