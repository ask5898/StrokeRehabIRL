import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import torch.optim as optim

'''Initialize the policy 𝜋_𝜃 randomly or sometimes with behavioral cloning (optional).

Roll out the policy in the environment to collect new trajectories.

Use expert data and generated data to train the discriminator to distinguish between them.

Use the discriminator's output as a surrogate cost to update the policy via reinforcement learning (e.g., policy gradients).

Repeat steps 2–4 until the policy imitates the expert well.'''

PATH = r'C:\Users\ROBOTIC5\Desktop\IRL\Expert'.replace('\\','/')
PATH_STATE  = r'C:\Users\ROBOTIC5\Desktop\IRL\Expert\states\all-states_clean.csv'.replace('\\','/')
PATH_ACTION = r'C:\Users\ROBOTIC5\Desktop\IRL\Expert\actions\all-actions_clean.csv'.replace('\\','/')

class PolicyNet(nn.Module) :
    def __init__(self, state_dims, action_dims):
        super(PolicyNet,self).__init__() 
        self.state_dims = state_dims
        self.action_dims = action_dims
        # Policy being parameterised with a neural network
        self.policy = nn.Sequential(nn.Linear(self.state_dims, 64),
                              nn.Tanh(),
                              nn.Linear(64,64),
                              nn.Tanh(),
                              nn.Linear(64,64),
                              nn.Tanh(),
                              nn.Linear(64,self.action_dims))
        
        # state independent standard deviation
        self.log_std = nn.Parameter(torch.zeros(self.action_dims))
        
        
    def forward(self, states) :
        if not isinstance(states, torch.Tensor) :
            states = torch.tensor(states)
        mean = self.policy(states)
        std = torch.exp(self.log_std)
        cov_mat = torch.eye(self.action_dims)*(std**2)
        return MultivariateNormal(mean, cov_mat)
    

class ValueNet(nn.Module) :
    def __init__(self, state_dims) :
        #maps state -> expected reward
        super(ValueNet, self).__init__()
        self.reward = nn.Sequential(nn.Linear(state_dims,64),
                                    nn.Tanh(),
                                    nn.Linear(64,64),
                                    nn.Tanh(),
                                    nn.Linear(64,64),
                                    nn.Tanh(),
                                    nn.Linear(64,1))
        
    def forward(self, states) :
        reward = self.reward(states)
        return reward
    

class Discriminator(nn.Module) :
    def __init__(self, state_dims, action_dims):
        super(Discriminator, self).__init__()
        # Maps (state,action) -> [0,1] ,i.e., how good is the selected policy
        self.discriminator = nn.Sequential(nn.Linear(state_dims+action_dims,64),
                                           nn.Tanh(),
                                           nn.Linear(64,64),
                                           nn.Tanh(),
                                           nn.Linear(64,64),
                                           nn.Tanh(),
                                           nn.Linear(64,1))
        
    def forward(self, state, action) :
        return torch.sigmoid(self.discriminator(torch.cat([state,action], dim=-1)))
    


class ExpertDataset(Dataset) :
    def __init__(self,state_file_path, expert_file_path):
        states = pd.read_csv(state_file_path)
        actions = pd.read_csv(expert_file_path)
        self.states = states.values
        self.actions = torch.from_numpy(actions.values)
        self.mean = torch.mean(self.actions,axis=0,dtype=torch.float64)
        self.std = torch.std(self.actions)
        self.cov_mat = torch.eye(self.states.shape[1],dtype=torch.float64)*(self.std**2)
        self.action_dist = MultivariateNormal(self.mean, self.cov_mat)

    def __len__(self) :
        return self.states.shape[1]
    
    def __getitem__(self, index):
        if torch.is_tensor(index) :
            index = index.tolist()

        traj = {
                'state':self.states[index].T,
                'action':self.action_dist
                }
        
        return traj
    
def KLDiv_loss(p,q) :
    p_mean = p.mean.detach().numpy()
    q_mean = q.mean.detach().numpy()
    cov_mat_p = p.covariance_matrix.detach().numpy()
    cov_mat_q = q.covariance_matrix.detach().numpy()
    sigma_p_mod = np.linalg.norm(cov_mat_p)
    sigma_q_mod = np.linalg.norm(cov_mat_q)
    inv_covmat_q = np.linalg.inv(cov_mat_q)
    loss = 0.5*(np.log(np.divide(sigma_q_mod,sigma_p_mod)) - p_mean.shape[0]
                + np.trace(np.matmul(inv_covmat_q,cov_mat_p)) + 
                np.matmul(np.matmul(np.transpose(q_mean-p_mean),inv_covmat_q),(q_mean-p_mean)))

    return torch.tensor(loss)

def loglikelihood_loss(output_dist, target_dist) :
    action = target_dist.sample()
    return -output_dist.log_prob(action).mean()

data = ExpertDataset(PATH_STATE,PATH_ACTION)

state_dim = len(data)
action_dim = state_dim
def train(network, epochs) :
    running_loss = 0.0
    optimizer = optim.Adam(network.parameters())
    for epoch in range(0,epochs) :
        for i, d in enumerate(data) :
            inp = torch.Tensor(d['state'])
            target = d['action']
            optimizer.zero_grad()
            out = network(inp)
            loss = loglikelihood_loss(out,target)
            loss = Variable(loss, requires_grad = True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
    torch.save(network.state_dict(), os.path.join(PATH,'expert_model.pth'))
    print('done')

