from create_environment import PatientSim, ACTION, OBSERVATION_LEN
from Networks import PolicyNet,ValueNet,Discriminator
import torch
import torch.nn as nn
import glob
import os
import numpy as np

PATH = os.getcwd()
REMOVE = glob.glob(r'C:\Users\ROBOTIC5\Desktop\IRL\Datasets\SharingData_IBEC_VALDUCE_July2021\HM*Healthy*.mot'.replace('\\','/'))
N = len(glob.glob(r'C:\Users\ROBOTIC5\Desktop\IRL\Datasets\SharingData_IBEC_VALDUCE_July2021\*Healthy*.mot')) - len(REMOVE)
MOTION_PATHS = [file for file in glob.glob(r'C:\Users\ROBOTIC5\Desktop\IRL\Datasets\SharingData_IBEC_VALDUCE_July2021\*Healthy*.mot'.replace('\\','/'))  if file not in REMOVE]
MODEL_PATH = os.path.join(PATH, 'FullBodyModel_withObject.osim')

class Gail(nn.Module) :
    def __init__(self, state_dims, action_dims):
        super(Gail,self).__init__()
        self.state_dims = state_dims
        self.action_dims = action_dims
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

        exp_obs = torch.FloatTensor(np.array(exp_obs))
        exp_act = torch.FloatTensor(np.array(exp_act))

        return exp_obs, exp_act


    def rolloutPolicy(self, env, gamma, lambd, horizon, num_iters) :
        obs = []
        act = []
        gam = []
        lmd = []
        advs = []
        rets = []
        cost = []
        pi = PolicyNet(OBSERVATION_LEN,len(ACTION))
        pi.eval()
        step = 0
        while step < num_iters :
            ep_obs = []
            ep_act = []
            ep_gamma = []
            ep_lambda = []
            ep_advs = []
            done = False
            observation,_ = env.reset()
            t=0
            step

            while not done and step<num_iters:
                observation = np.float32(observation)
                action = pi(observation).sample().numpy()
                ep_obs.append(observation)
                obs.append(observation)
                ep_act.append(action)
                act.append(action)
                ep_gamma.append(gamma**t)
                ep_lambda.append(lambd**t)
                observation, _, done = env.step(action)
                
                t = t + 1
                step = step+1
                if horizon is not None and t>=horizon:
            
                    done = True
                    break

            ep_obs = torch.FloatTensor(np.array(ep_obs))
            ep_act = torch.FloatTensor(np.array(ep_act))
            ep_gamma = torch.FloatTensor(ep_gamma)
            ep_lambda = torch.FloatTensor(ep_lambda)
            ep_costs = (-1)*torch.log(self.discriminator(ep_obs,ep_act)).squeeze().detach()
            ep_disc_cost = ep_gamma*ep_costs
            cost.append(ep_disc_cost)
            ep_disc_rets = torch.FloatTensor([sum(ep_disc_cost[i:]) for i in range(t)])
            ep_rets = ep_disc_rets / ep_gamma

            rets.append(ep_rets)

            self.value.eval()
            ep_curr_val = self.value(ep_obs)
            ep_advs = ep_disc_cost.sum() - ep_curr_val
            advs.append(ep_advs)
            gam.append(ep_gamma)
            lmd.append(ep_lambda)

        print('Episode cost = ',np.mean(cost))
        obs = torch.FloatTensor(np.array(obs))
        act = torch.FloatTensor(np.array(act))
        advs = torch.cat(advs)
        gam = torch.cat(gam)
        lmd = torch.cat(lmd)
        rets = torch.cat(rets)
        advs = (advs - advs.mean())/advs.std()

        return obs, act, advs, gam, lmd, rets

    def getGrads(self, func, net) :
        return torch.cat([grad.view(-1) for grad in torch.autograd.grad(func, net.parameters(), create_graph=True)])
    
    def conjugateGradient(self, Av_func, b, old, cg_damping = None,max_iter=10, residual_tol=1e-10):
        x = torch.zeros_like(b)
        if cg_damping is not None :
            r = b - Av_func(x, old, cg_damping)
        else :
            r = b - Av_func(x,old)
        p = r
        rsold = r.norm() ** 2

        for _ in range(max_iter):
            if cg_damping is not None :
                Ap = Av_func(p,old,cg_damping)
            else :
                Ap = Av_func(p, old)

            alpha = rsold / torch.dot(p, Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = r.norm() ** 2
            if torch.sqrt(rsnew) < residual_tol:
                break
            p = r + (rsnew / rsold) * p
            rsold = rsnew

        return x
    
    def getParams(self, net):
        return torch.cat([param.view(-1) for param in net.parameters()])


    def setParams(self,net, new_flat_params):
        start_idx = 0
        for param in net.parameters():
            end_idx = start_idx + np.prod(list(param.shape))
            param.data = torch.reshape(
                new_flat_params[start_idx:end_idx], param.shape
            )

            start_idx = end_idx

    def rescale_and_linesearch(self,g, s, Hs, max_kl, L, obs, old_distb, old_params, pi, max_iter=10,success_ratio=0.1):
        self.setParams(pi, old_params)
        L_old = L().detach()

        beta = torch.sqrt((2 * max_kl) / torch.dot(s, Hs))

        for _ in range(max_iter):
            new_params = old_params + beta * s

            self.setParams(pi, new_params)
            kld_new = self.kld(obs, old_distb).detach()

            L_new = L().detach()

            actual_improv = L_new - L_old
            approx_improv = torch.dot(g, beta * s)
            ratio = actual_improv / approx_improv

            if ratio > success_ratio and actual_improv > 0 and kld_new < max_kl:
                return new_params

            beta *= 0.5

        print("The line search was failed!")
        return old_params
    
    
    def valueConstraint(self, old_v, obs) :
        return torch.mean((old_v - self.value(obs))**2)
    
    
    def kld(self, obs, old_distb):
        distb = self.pi(obs)
        old_mean = old_distb.mean.detach()
        old_cov = old_distb.covariance_matrix.sum(-1).detach()
        mean = distb.mean
        cov = distb.covariance_matrix.sum(-1)

        return (0.5) * ((old_cov / cov).sum(-1)+ (((old_mean - mean) ** 2) / cov).sum(-1)- self.action_dims+ torch.log(cov).sum(-1)
                - torch.log(old_cov).sum(-1)
            ).mean()

    def valueHessian(self,v, grad_diff):
            hessian = self.getGrads(torch.dot(grad_diff, v), self.value).detach()

            return hessian
    
    def policyHessian(self, v, grad_kld_old_param, cg_damping):
            hessian = self.getGrads(torch.dot(grad_kld_old_param, v),self.pi).detach()

            return hessian + cg_damping * v

    def train(self, env, num_iters, trpo_eps, max_kl, cg_damping,lambda_) :
        opt_d = torch.optim.Adam(self.discriminator.parameters())
        exp_obs, exp_act = self.rolloutExpert(env)
        for i in range(num_iters) :
            print('Episode {}'.format(i))
            obs, act, advs, gamma, lmd, rets = self.rolloutPolicy(env, 0.99, 0.99, None, env.max_steps)
            exp_score = self.discriminator(exp_obs, exp_act)
            so_score = self.discriminator(obs,act)
            opt_d.zero_grad()
            loss_d_exp = torch.nn.functional.binary_cross_entropy_with_logits(exp_score, torch.zeros_like(exp_score))
            loss_d_so = torch.nn.functional.binary_cross_entropy_with_logits(so_score, torch.zeros_like(so_score))
            loss = loss_d_exp + loss_d_so
            loss.backward()
      
            print('Discriminator Loss = {}'.format(loss.item()))
            print('----------------------------------------------------------')
            
            opt_d.step()
            self.value.train()
            old_params = self.getParams(self.value).detach()
            old_v = self.value(obs).detach()

            grad_diff = self.getGrads(self.valueConstraint(old_v,obs), self.value)

            
            g = self.getGrads(((-1) * (self.value(obs).squeeze() - rets) ** 2).mean(), self.value).detach()
            s = self.conjugateGradient(self.valueHessian, g, grad_diff).detach()

            Hs = self.valueHessian(s, grad_diff).detach()
            alpha = torch.sqrt(2 * trpo_eps / torch.dot(s, Hs))

            new_params = old_params + alpha * s

            self.setParams(self.value, new_params)

            self.pi.train()
            old_params = self.getParams(self.pi).detach()
            old_distb = self.pi(obs)

            def L():
                distb = self.pi(obs)

                return (advs * torch.exp(distb.log_prob(act)- old_distb.log_prob(act).detach())).mean()

            

            grad_kld_old_param = self.getGrads(self.kld(obs, old_distb), self.pi)

            

            g = self.getGrads(L(), self.pi).detach()

            s = self.conjugateGradient(self.policyHessian, g, grad_kld_old_param, cg_damping=cg_damping).detach()
            Hs = self.policyHessian(s, grad_kld_old_param, cg_damping).detach()

            new_params = self.rescale_and_linesearch(g, s, Hs, max_kl,L , obs, old_distb, old_params, self.pi)

            disc_causal_entropy = ((-1) * gamma * self.pi(obs).log_prob(act)).mean()
            grad_disc_causal_entropy = self.getGrads(disc_causal_entropy, self.pi)
            new_params += lambda_ * grad_disc_causal_entropy

            self.setParams(self.pi, new_params)

        return None

            

if __name__ == '__main__' :
   env = PatientSim(model_path=MODEL_PATH, visualize=False, motion_paths=MOTION_PATHS)
   gail = Gail(OBSERVATION_LEN,len(ACTION))
   trpo_eps = 0.01
   max_kl = 0.01
   cg_damping = 0.1
   lambda_ = 1e-3
   gail.train(env,1000, trpo_eps, max_kl, cg_damping, lambda_ )
   torch.save(gail.pi.state_dict(), os.path.join("policy.ckpt"))
   torch.save(gail.value.state_dict(), os.path.join( "value.ckpt"))
   torch.save(gail.discriminator.state_dict(), os.path.join("discriminator.ckpt"))




