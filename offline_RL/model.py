import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
import copy

class QFlowMLP(T.nn.Module):
    def __init__(self, s_dim, a_dim, is_qflow=False, q_net=None, alpha=None):
        super(QFlowMLP, self).__init__()
        device = torch.device('cuda')
        self.is_qflow = is_qflow
        self.q = q_net
        self.alpha = alpha
        self.x_model = T.nn.Sequential(
            T.nn.Linear(s_dim+a_dim+128,256),nn.LayerNorm(256),T.nn.GELU(),
            T.nn.Linear(256,256),nn.LayerNorm(256),T.nn.GELU()
        )

        self.out_model = T.nn.Sequential(
            T.nn.Linear(256,256),nn.LayerNorm(256),T.nn.GELU(),
            #T.nn.Linear(256,128),nn.LayerNorm(128),T.nn.GELU(),
            T.nn.Linear(256,a_dim)
        )

        self.means_scaling_model = T.nn.Sequential(
            T.nn.Linear(128,128),nn.LayerNorm(128), T.nn.GELU(),
            T.nn.Linear(128,128),nn.LayerNorm(128), T.nn.GELU(),
            T.nn.Linear(128,a_dim), 
        )
        
        self.harmonics = T.nn.Parameter(T.arange(1,64+1).float() * 2 * np.pi).requires_grad_(False)
        self.B = T.randn((a_dim,64)).cuda()
        self.B.requires_grad_(False)

    def forward(self, s, x, t):
        t_fourier1 = (t.unsqueeze(1) * self.harmonics).sin()
        t_fourier2 = (t.unsqueeze(1) * self.harmonics).cos()
        t_emb = T.cat([t_fourier1, t_fourier2], 1)
        if not self.is_qflow:
            x_emb = self.x_model(torch.cat([s,x,t_emb],1))
        if self.is_qflow:
            #with torch.no_grad():
            x_emb = self.x_model(torch.cat([s,x,t_emb],1))
            with T.enable_grad():
                x.requires_grad_(True)
                means_scaling = self.means_scaling_model(t_emb) * self.q.score(s, x, alpha=self.alpha)
            return self.out_model(x_emb) + means_scaling
        return self.out_model(x_emb)
    
class DiffusionModel(nn.Module):
    def __init__(self, state_dim, action_dim, diffusion_steps, schedule='linear', predict='epsilon', policy_net='mlp'):
        super(DiffusionModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.diffusion_steps = diffusion_steps
        self.schedule = schedule
        self.policy = QFlowMLP(s_dim=state_dim, a_dim=action_dim)
        self.diffusion_steps = diffusion_steps
        self.predict = predict
        if self.schedule == 'linear':
            beta1 = 0.02
            beta2 = 1e-4
            beta_t = (beta1 - beta2) * torch.arange(diffusion_steps+1, 0, step=-1, dtype=torch.float32) / (diffusion_steps) + beta2
        alpha_t = 1 - torch.flip(beta_t, dims=[0])
        log_alpha_t = torch.log(alpha_t)
        alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()
        sqrtab = torch.sqrt(alphabar_t)
        oneover_sqrta = 1 / torch.sqrt(alpha_t)
        sqrtmab = torch.sqrt(1 - alphabar_t)
        mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab
        self.register_buffer('beta_t', beta_t)
        self.register_buffer('alpha_t', torch.flip(alpha_t, dims=[0]))
        self.register_buffer('log_alpha_t', torch.flip(log_alpha_t, dims=[0]))
        self.register_buffer('alphabar_t', torch.flip(alphabar_t, dims=[0]))
        self.register_buffer('sqrtab', torch.flip(sqrtab, dims=[0]))
        self.register_buffer('oneover_sqrta', torch.flip(oneover_sqrta, dims=[0]))
        self.register_buffer('sqrtmab', torch.flip(sqrtmab, dims=[0]))
        self.register_buffer('mab_over_sqrtmab_inv', torch.flip(mab_over_sqrtmab_inv, dims=[0]))
        
    def forward(self, s, x, t):
        epsilon = self.policy(s, x, t)
        return epsilon
    
    def score(self, s, x, t):
        t_idx = (t*self.diffusion_steps).long().unsqueeze(1)
        epsilon = self(s, x, t)
        if self.predict == 'epsilon':
            score = -epsilon/self.sqrtmab[t_idx]
        elif self.predict == 'x0':
            score = (self.sqrtab[t_idx]*epsilon - x)/(1-self.alphabar_t[t_idx])
        return score
    
    def sample(self, s):
        x = torch.randn(s.shape[0], self.action_dim).to(s.device)
        t = torch.zeros((s.shape[0],), device=s.device)
        dt = 1/self.diffusion_steps
        for i in range(self.diffusion_steps):
            epsilon = self(s, x, t)
            if self.predict == 'epsilon':
                x = self.oneover_sqrta[i] * (x - self.mab_over_sqrtmab_inv[i] * epsilon) + torch.sqrt(self.beta_t[i]) * torch.randn_like(x)
            elif self.predict == 'x0':
                x = (1/torch.sqrt(self.alpha_t[i])) * ((1-(1-self.alpha_t[i])/(1-self.alphabar_t[i]))*x + ((1-self.alpha_t[i])/(1-self.alphabar_t[i]))*self.sqrtab[i]*epsilon) + torch.sqrt(self.beta_t[i]) * torch.randn_like(x)
            t += dt
        return x
    
    def compute_loss(self, s, x):
        t_idx = torch.randint(0, self.diffusion_steps, (s.shape[0], 1)).to(s.device)
        t = t_idx.float().squeeze(1)/self.diffusion_steps
        epsilon = torch.randn_like(x).to(s.device)
        x_t = self.sqrtab[t_idx] * x + self.sqrtmab[t_idx] * epsilon
        epsilon_pred = self(s, x_t, t)
        if self.predict == 'epsilon':
            w = torch.minimum(torch.tensor(5)/((self.sqrtab[t_idx] / self.sqrtmab[t_idx]) ** 2), torch.tensor(1)) # Min-SNR-gamma weights
            loss = (w * (epsilon - epsilon_pred) ** 2).mean()
        elif self.predict == 'x0':
            #epsilon_pred = torch.tanh(epsilon_pred)
            w = torch.minimum((self.sqrtab[t_idx] / self.sqrtmab[t_idx]) ** 2, torch.tensor(5))
            loss = (w * (x - epsilon_pred) ** 2).mean()
        return loss
    
    
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 1)
        self.gelu = nn.GELU()
        self.ln1 = nn.LayerNorm(256)
        self.ln2 = nn.LayerNorm(256)
        self.ln3 = nn.LayerNorm(64)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = self.gelu(self.ln1(self.fc1(x)))
        x = self.gelu(self.ln2(self.fc2(x)))
        x = self.gelu(self.ln3(self.fc3(x)))
        x = self.fc4(x)
        return x

# For DDPG
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, max_action=1.0):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.max_action = max_action
        self.gelu = nn.GELU()
        self.ln1 = nn.LayerNorm(256)
        self.ln2 = nn.LayerNorm(256)

    def forward(self, x):
        x = self.gelu(self.ln1(self.fc1(x)))
        x = self.gelu(self.ln2(self.fc2(x)))
        x = self.fc3(x)
        x = self.max_action * torch.tanh(x)
        return x
    
    
class QFlow(nn.Module):
    def __init__(self, state_dim, action_dim, diffusion_steps, schedule='linear', predict='epsilon', q_net=None, bc_net=None, alpha=1.0):
        super(QFlow, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.diffusion_steps = diffusion_steps
        self.schedule = schedule
        self.predict = predict
        self.q_net = q_net
        self.bc_net = bc_net
        self.alpha = alpha
        self.qflow = copy.deepcopy(bc_net.policy)#QFlowMLP(state_dim, action_dim)#QFlowMLP(state_dim, action_dim, is_qflow=True, q_net=q_net)
        self.qflow.is_qflow = True
        self.qflow.q = q_net
        self.qflow.alpha = alpha
        self.alpha = alpha
    
    def forward(self, s, a, t):
        q_epsilon = self.qflow(s, a, t)
        with torch.no_grad():
            bc_epsilon = self.bc_net(s, a, t).detach()
        return q_epsilon, bc_epsilon
    
    def sample(self, s, extra=False):
        bs = s.shape[0]
        dim = self.action_dim
        normal_dist = torch.distributions.Normal(T.zeros((bs,self.action_dim), device=s.device), T.ones((bs,self.action_dim), device=s.device))
        x = normal_dist.sample()
        t = T.zeros((bs,), device=s.device)
        dt = 1/self.diffusion_steps

        logpf_pi = normal_dist.log_prob(x).sum(1)
        logpf_p = normal_dist.log_prob(x).sum(1)
        
        extra_steps = 1
        if extra:
            extra_steps = 20
        for i in range(self.diffusion_steps):
            for j in range(extra_steps):
                q_epsilon, bc_epsilon = self(s, x, t)
                pflogvars = np.log(torch.sqrt(self.bc_net.beta_t[i]).cpu().numpy()) * 2
                pflogvars_sample = pflogvars
                
                epsilon = q_epsilon + bc_epsilon
                new_x = self.bc_net.oneover_sqrta[i] * (x - self.bc_net.mab_over_sqrtmab_inv[i] * epsilon.detach()) + torch.sqrt(self.bc_net.beta_t[i]) * torch.randn_like(x)

                pf_pi_dist = torch.distributions.Normal(self.bc_net.oneover_sqrta[i] * (x - self.bc_net.mab_over_sqrtmab_inv[i] * bc_epsilon), torch.sqrt(self.bc_net.beta_t[i])*torch.ones_like(x))
                logpf_pi += pf_pi_dist.log_prob(new_x).sum(1)

                pf_p_dist = torch.distributions.Normal(self.bc_net.oneover_sqrta[i] * (x - self.bc_net.mab_over_sqrtmab_inv[i] * epsilon), torch.sqrt(self.bc_net.beta_t[i])*torch.ones_like(new_x))
                logpf_p += pf_p_dist.log_prob(new_x).sum(1)
            
                x = new_x
                if i < self.diffusion_steps-1:
                    break 
            t = t + dt
            
        return x, logpf_pi, logpf_p
    
    def posterior_log_reward(self, s, a):
        q_r = self.q_net.log_reward(s, a, alpha=self.alpha).squeeze()
        return q_r
    
    def compute_loss_with_sample(self, s, x, gfn_batch_size=64):
        s_repeat = s.repeat_interleave(gfn_batch_size, 0)
        x_repeat = x.repeat_interleave(gfn_batch_size, 0)
        bs = s_repeat.shape[0]
        dim = self.action_dim
        minlogvar,maxlogvar=-4,4
        t = T.zeros((bs,), device=s.device)
        dt = 1/self.diffusion_steps

        logpf_pi = T.zeros((bs,), device=s.device)
        logpf_p = T.zeros((bs,), device=s.device)
        logr = self.posterior_log_reward(s, x)
        logr = logr.repeat_interleave(gfn_batch_size, 0)
        
        for i in range(self.diffusion_steps-1, -1, -1):
            pb_dist = torch.distributions.Normal(torch.sqrt(self.bc_net.alpha_t[i])*x_repeat, torch.sqrt(self.bc_net.beta_t[i])*torch.ones_like(x_repeat))
            new_x = pb_dist.sample()
            
            q_epsilon, bc_epsilon = self(s_repeat, new_x, t+i*dt)
            epsilon = q_epsilon + bc_epsilon
            
            pf_pi_dist = torch.distributions.Normal(self.bc_net.oneover_sqrta[i] * (new_x - self.bc_net.mab_over_sqrtmab_inv[i] * bc_epsilon), torch.sqrt(self.bc_net.beta_t[i])*torch.ones_like(new_x))
            logpf_pi += pf_pi_dist.log_prob(x_repeat).sum(1)

            pf_p_dist = torch.distributions.Normal(self.bc_net.oneover_sqrta[i] * (new_x - self.bc_net.mab_over_sqrtmab_inv[i] * epsilon), torch.sqrt(self.bc_net.beta_t[i])*torch.ones_like(new_x))
            logpf_p += pf_p_dist.log_prob(x_repeat).sum(1)

            x_repeat = new_x
        prior_dist = torch.distributions.Normal(torch.zeros_like(x_repeat), torch.ones_like(x_repeat))
        logpf_pi += prior_dist.log_prob(x_repeat).sum(1)
        logpf_p += prior_dist.log_prob(x_repeat).sum(1)
        
        logC = (logr+self.alpha*logpf_pi-self.alpha*logpf_p).view(-1, gfn_batch_size)
        logC = logC.mean(1).repeat_interleave(gfn_batch_size, 0).detach()
        
        loss = 0.5*((self.alpha*logpf_p+logC-self.alpha*logpf_pi-logr.detach())**2).mean()
        return loss, logC.mean().item()
    
    def compute_loss(self, s, gfn_batch_size=64):
        s_repeat = s.repeat_interleave(gfn_batch_size, 0)
        x, logpf_pi, logpf_p = self.sample(s_repeat)
        logr = self.posterior_log_reward(s_repeat, x)
        
        logC = (logr+self.alpha*logpf_pi-self.alpha*logpf_p).view(-1, gfn_batch_size)
        logC = logC.mean(1).repeat_interleave(gfn_batch_size, 0).detach()
        loss = 0.5*((self.alpha*logpf_p+logC-self.alpha*logpf_pi-logr.detach())**2).mean()
        return loss, logC.mean().item()