
from create_environment import PatientSim, ACTION, OBSERVATION_LEN
from Networks import PolicyNet,ValueNet,Discriminator
import torch
import torch.nn as nn
import glob
import os
import numpy as np
import scipy

PATH = os.getcwd()
REMOVE = glob.glob(r'C:\Users\ROBOTIC5\Desktop\IRL\Datasets\SharingData_IBEC_VALDUCE_July2021\HM*Healthy*.mot'.replace('\\','/'))
N = len(glob.glob(r'C:\Users\ROBOTIC5\Desktop\IRL\Datasets\SharingData_IBEC_VALDUCE_July2021\*Healthy*.mot')) - len(REMOVE)
MOTION_PATHS = [file for file in glob.glob(r'C:\Users\ROBOTIC5\Desktop\IRL\Datasets\SharingData_IBEC_VALDUCE_July2021\*Healthy*.mot'.replace('\\','/'))  if file not in REMOVE]
MODEL_PATH = os.path.join(PATH, 'FullBodyModel_withObject.osim')

class Gail(nn.Module) :
    def __init__(self, state_dims, action_dims, batch_size):
        super(Gail,self).__init__()
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.batch_size = batch_size
        self.pi = PolicyNet(self.state_dims, self.action_dims)
        self.value = ValueNet(self.state_dims)
        self.discriminator = Discriminator(self.state_dims, self.action_dims)

    def rolloutExpert(self, env) :
        done = False
        exp_obs = []
        exp_act = []
        observation,_ = env.reset(expert=True)
        while not done:
            action = env.sample_expert_action(env.current_step)
            observation, _, done = env.step(action)
            exp_obs.append(observation)
            exp_act.append(action)

        exp_obs = scipy.signal.savgol_filter(exp_obs, 5, 2)
        exp_obs = torch.FloatTensor(np.array(exp_obs))
        exp_act = torch.FloatTensor(np.array(exp_act))
        exp_act = torch.diff(exp_act,axis=0)
        exp_act = torch.cat((exp_act, torch.zeros(1,exp_act.shape[1])))
        exp_obs = (exp_obs - exp_obs.mean())/(exp_obs.std() + 1e-8)
        exp_act = 2*((exp_act - exp_act.min())/(exp_act.max() - exp_act.min() + 1e-8)) - 1
        return exp_obs, exp_act


    def rolloutPolicy(self, env, gamma, lambd, horizon, num_iters) :
        obs = []
        act = []
        gam = []
        lmd = []
        advs = []
        rets = []
        cost = []
        
        self.pi.eval()
        step = 0

        while step < num_iters :
            ep_obs = []
            ep_act = []
            ep_gamma = []
            ep_lambda = []
            ep_advs = []
            observation = []
            done = False
            observation = scipy.signal.savgol_filter(env.reset()[0],5,2)
            t=0
            while not done and step<num_iters:
                observation = np.float32(observation)
                observation = (observation - observation.mean())/(observation.std() + 1e-8)
                action = self.pi(observation).sample().numpy()
                action = 2*((action - action.min())/(action.max() - action.min() + 1e-8)) - 1
                ep_obs.append(observation)
                obs.append(observation)
                ep_act.append(action)
                act.append(action)
                ep_gamma.append(gamma**t)
                ep_lambda.append(lambd**t)
                observation, _, done = env.step(action)
                observation = scipy.signal.savgol_filter(observation,5,2)
                
                t = t + 1
                step = step+1
                if horizon is not None and t>=horizon:
            
                    done = True
                    break

            ep_obs = torch.FloatTensor(np.array(ep_obs))
            ep_act = torch.FloatTensor(np.array(ep_act))
            ep_gamma = torch.FloatTensor(ep_gamma)
            ep_lambda = torch.FloatTensor(ep_lambda)
            ep_costs = torch.log(torch.FloatTensor([1.0]) - self.discriminator(ep_obs,ep_act) + 1e-8).squeeze().detach() 
            cost.append(ep_costs)
            ep_disc_cost = ep_gamma*ep_costs
            ep_disc_rets = torch.FloatTensor([sum(ep_disc_cost[i:]) for i in range(t)])
            ep_rets = ep_disc_rets / ep_gamma

            rets.append(ep_rets)

            self.value.eval()
            ep_curr_vals = self.value(ep_obs).detach()
            #ep_advs = ep_disc_cost.sum() - ep_curr_vals
            ep_next_vals = torch.cat((self.value(ep_obs)[1:], torch.FloatTensor([[0.]]))).detach()
            ep_deltas = (-1)*ep_costs.unsqueeze(-1) + gamma * ep_next_vals - ep_curr_vals

            ep_advs = torch.FloatTensor([((ep_gamma * ep_lambda)[:t - j].unsqueeze(-1) * ep_deltas[j:]).sum()for j in range(t)])
            advs.append(ep_advs)
            
            gam.append(ep_gamma)
            lmd.append(ep_lambda)
        

        
        obs = torch.FloatTensor(np.array(obs))
        act = torch.FloatTensor(np.array(act))
        act =torch.diff(act, axis=0)
        act = torch.cat((act, torch.zeros(1,act.shape[1])))
        cost = torch.cat(cost)
        print('Episode Cost = {}'.format(cost.mean()))
        advs = torch.cat(advs)
        gam = torch.cat(gam)
        lmd = torch.cat(lmd)
        rets = torch.cat(rets)
 
        advs = (advs - advs.mean())/(advs.std() + 1e-8)
        

        return obs, act, advs, gam, lmd, rets

   

    def kld(self, obs, old_distb):
        distb = self.pi(obs)
        old_mean = old_distb.mean.detach()
        old_std = old_distb.variance.detach()
        mean = distb.mean
        std = distb.variance
        return (torch.log(old_std/std) + (((std)**2 + (mean-old_mean)**2))/(2*(old_std**2)) - 0.5).mean()



    def train(self, env, num_iters, clip_eps, max_kl, cg_damping, lambda_) :
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=3e-3)
        opt_v = torch.optim.Adam(self.value.parameters(), lr = 3e-3)
        opt_pi = torch.optim.Adam(self.pi.parameters(), lr = 3e-3)
        exp_obs, exp_act = self.rolloutExpert(env)
        #exp_obs, exp_act = exp_obs[:1000], exp_act[:1000]
        print(exp_obs.shape, exp_act.shape)
        for i in range(0, num_iters) :
            print('Episode {}'.format(i+1))
            obs, act, advs, gamma, lmd, rets = self.rolloutPolicy(env, 0.995, 0.999, None, env.max_steps)
            opt_d.zero_grad()
           
            exp_score = self.discriminator(exp_obs, exp_act)
            so_score = self.discriminator(obs, act)
            print('Expert Confidence = {}'.format(exp_score.mean()))
            print('Agent Confidence = {}'.format(so_score.mean()))
            loss_d_exp = torch.nn.functional.binary_cross_entropy_with_logits(exp_score, torch.ones_like(exp_score))
            loss_d_so = torch.nn.functional.binary_cross_entropy_with_logits(so_score, torch.zeros_like(so_score))
            loss_d = (loss_d_exp + loss_d_so)
            loss_d.backward() 
            opt_d.step()
      
            print('Discriminator Loss = {}'.format(loss_d.item()))
            
            self.value.train()
            opt_v.zero_grad()
            L_v = torch.autograd.Variable(((self.value(obs).squeeze() - advs) ** 2).mean(), requires_grad=True)
            L_v.backward()
            opt_v.step()
            print('Critic Loss = {}'.format(L_v.item()))
            

            self.pi.train()
            opt_pi.zero_grad()
            distb_old = self.pi(obs)
            ratio = torch.exp(self.pi(obs).log_prob(act)- distb_old.log_prob(act).detach())
            L_pi_1 = (-1)*ratio*advs
            L_pi_2 = (-1)*advs*torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
            L_pi = torch.max(L_pi_1,L_pi_2).mean()
            L_pi.backward()
            opt_pi.step()
            print('Actor Loss = {}'.format(L_pi.item()))
            print('----------------------------------------------------------')

        return None

            

if __name__ == '__main__' :
   env = PatientSim(model_path=MODEL_PATH, visualize=False, motion_paths=MOTION_PATHS)
   gail = Gail(OBSERVATION_LEN,len(ACTION),4)
   #gail.discriminator.load_state_dict(torch.load(os.path.join(PATH, 'discriminator.ckpt')))
   #gail.pi.policy.load_state_dict(torch.load(os.path.join(PATH,'policy.ckpt')))
   #gail.value.load_state_dict(torch.load(os.path.join(PATH,'value.ckpt')))
   clip_eps = 0.2
   max_kl = 0.01
   cg_damping = 0.1
   lambda_ = 1e-2
   num_iters = 500
   gail.train( env, num_iters, clip_eps, max_kl, cg_damping, lambda_)
   print('Training Complete!!')
   torch.save(gail.pi.state_dict(), os.path.join("policy.ckpt"))
   torch.save(gail.value.state_dict(), os.path.join( "value.ckpt"))
   torch.save(gail.discriminator.state_dict(), os.path.join("discriminator.ckpt"))

