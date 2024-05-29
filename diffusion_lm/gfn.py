import torch
from sampling import get_predictor, Denoiser
from model import utils as mutils
from catsample import sample_categorical
import numpy as np


class PosteriorPriorGFN(torch.nn.Module):
    def __init__(self, prior_model, posterior_model, log_reward, graph, noise, predictor,
                 traj_len, sampling_len, eps, device):
        super(PosteriorPriorGFN, self).__init__()
        self.prior_model = prior_model
        self.posterior_model = posterior_model
        self.log_reward = log_reward
        self.graph = graph
        self.noise = noise
        self.traj_len = traj_len
        self.sampling_len = sampling_len
        self.device = device
        self.eps = eps
        self.predictor = get_predictor(predictor)(self.graph, self.noise)
        self.logZ = torch.nn.Parameter(torch.zeros(1, device=self.device))

    def replay_traj(self, batch_size, traj, detach_prob=0.8, denoise=True):
        projector = lambda x: x
        # prefix_ids = tokenizer("Here's a funny joke: ").input_ids
        # prefix_locs = list(range(len(prefix_ids)))
        # prefix_ids = torch.tensor(prefix_ids, device="cuda")[None].repeat(batch_size[0], 1)
        # input_ids = (50256 * torch.ones(1000)).long().numpy().tolist()
        # input_locs = list(range(1024 - len(input_ids), 1024))

        # input_ids = torch.tensor(input_ids, device="cuda")[None].repeat(batch_size[0], 1)
        # def projector(inp):
        #     inp = inp.clone()
        #     with torch.no_grad():
        #         inp[:, prefix_locs] = prefix_ids
        #     return inp
        total_prior_log_score = torch.zeros(batch_size[0], self.sampling_len+1, device=self.device)
        total_posterior_log_score = torch.zeros(batch_size[0], self.sampling_len+1, device=self.device)
        
        # x = self.graph.sample_limit(*batch_size).to(self.device).requires_grad_(False)
        x = traj[0]
        timesteps = torch.linspace(1, self.eps, self.sampling_len + 1, device=self.device)
        dt = (1 - self.eps) / self.sampling_len

        for i in range(self.sampling_len):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)
            x = projector(x)
            curr_sigma = self.noise(t)[0]
            next_sigma = self.noise(t - dt)[0]
            dsigma = curr_sigma - next_sigma
            # print(timesteps[i], x, curr_sigma, dsigma)

            # with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                posterior_log_scores = self.posterior_model(x, curr_sigma.reshape(-1))
                posterior_score = posterior_log_scores.exp()
                posterior_stag_score = self.graph.staggered_score(posterior_score, dsigma)
                posterior_probs = posterior_stag_score * self.graph.transp_transition(x, dsigma)
            
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    prior_log_scores = self.prior_model(x, curr_sigma.reshape(-1))
                    prior_score = prior_log_scores.exp()
                    prior_stag_score = self.graph.staggered_score(prior_score, dsigma)
                    prior_probs = prior_stag_score * self.graph.transp_transition(x, dsigma)
            x = traj[i+1]
            # x = sample_categorical(prior_probs if sample_from_prior else posterior_probs).detach()
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                prior_lp = prior_probs.log_softmax(-1).gather(2, x.unsqueeze(1)).squeeze(1).sum(1)
                posterior_lp = posterior_probs.log_softmax(-1).gather(2, x.unsqueeze(1)).squeeze(1).sum(1)

                total_prior_log_score[:, i] = prior_lp.detach() if np.random.rand() < detach_prob else prior_lp
                total_posterior_log_score[:, i] = posterior_lp.detach() if np.random.rand() < detach_prob else posterior_lp
           

        if denoise:
            # denoising step
            x = projector(x)
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device)
            sigma = self.noise(t)[0]
            # print(timesteps[-1], x, sigma)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                posterior_log_scores = self.posterior_model(x, sigma.reshape(-1))
                posterior_score = posterior_log_scores.exp()
                posterior_stag_score = self.graph.staggered_score(posterior_score, sigma)
                posterior_probs = posterior_stag_score * self.graph.transp_transition(x, sigma)
                
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    prior_log_scores = self.prior_model(x, sigma.reshape(-1))
                    prior_score = prior_log_scores.exp()
                    prior_stag_score = self.graph.staggered_score(prior_score, sigma)
                    prior_probs = prior_stag_score * self.graph.transp_transition(x, sigma)

            # truncate probabilities
            # probs = prior_probs if sample_from_prior else posterior_probs
            # probs.detach()
            # if self.graph.absorb:
            #     probs = probs[..., :-1]
            x = traj[-1]
            # x = sample_categorical(probs).detach()
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                total_prior_log_score[:, -1] = prior_probs.log_softmax(-1).gather(2, x.unsqueeze(1)).squeeze(1).sum(1)
                total_posterior_log_score[:, -1] = posterior_probs.log_softmax(-1).gather(2, x.unsqueeze(1)).squeeze(1).sum(1)

                
        return {
            'x': x,
            'posterior_log_probs': total_posterior_log_score,
            'prior_log_probs': total_prior_log_score
        }

    def sample_fwd(self, batch_size=None, sample_from_prior=False, detach_prob=0.8, denoise=True, tokenizer=None, projector=lambda x: x):
        # projector = lambda x: x
        # prefix_ids = tokenizer("David noticed he had put on a lot of weight recently. He examined his habits to try and figure out the reason. He realized he'd been eating too much fast food lately.").input_ids
        # suffix_ids = tokenizer("After a few weeks, he started to feel much better.").input_ids
        # input_ids = prefix_ids + suffix_ids
        # input_locs = list(range(len(prefix_ids))) + list(range(60-len(suffix_ids), 60))
        # input_ids = torch.tensor(input_ids, device="cuda")[None].repeat(batch_size[0], 1)
        
        # def projector(inp):
        #     inp = inp.clone()
        #     with torch.no_grad():
        #         inp[:, input_locs] = input_ids
        #     return inp
        # traj = []
        total_prior_log_score = torch.zeros(batch_size[0], self.sampling_len+1, device=self.device)
        total_posterior_log_score = torch.zeros(batch_size[0], self.sampling_len+1, device=self.device)
        
        x = self.graph.sample_limit(*batch_size).to(self.device).requires_grad_(False)
        # traj.append(x.detach())
        timesteps = torch.linspace(1, self.eps, self.sampling_len + 1, device=self.device)
        dt = (1 - self.eps) / self.sampling_len

        for i in range(self.sampling_len):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)
            x = projector(x)
            curr_sigma = self.noise(t)[0]
            next_sigma = self.noise(t - dt)[0]
            dsigma = curr_sigma - next_sigma
            # print(timesteps[i], x, curr_sigma, dsigma)

            # with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                posterior_log_scores = self.posterior_model(x, curr_sigma.reshape(-1))
                posterior_score = posterior_log_scores.exp()
                posterior_stag_score = self.graph.staggered_score(posterior_score, dsigma)
                posterior_probs = posterior_stag_score * self.graph.transp_transition(x, dsigma)
            
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    prior_log_scores = self.prior_model(x, curr_sigma.reshape(-1))
                    prior_score = prior_log_scores.exp()
                    prior_stag_score = self.graph.staggered_score(prior_score, dsigma)
                    prior_probs = prior_stag_score * self.graph.transp_transition(x, dsigma)

            # prev_x = x.clone().detach()
            x = sample_categorical(prior_probs if sample_from_prior else posterior_probs).detach()
            # import pdb; pdb.set_trace();
            # print(x != 50257 & prev_x != 50257)
            # mask = torch.ones_like(x)
            # traj.append(x.detach())
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                total_prior_log_score[:, i] = torch.distributions.Categorical(probs=prior_probs).log_prob(x).sum(-1)
                # total_prior_log_score[:, i] = prior_probs.log().gather(2, x.unsqueeze(1)).squeeze(1).sum(1)

                # posterior_lp = posterior_probs.log().gather(2, x.unsqueeze(1)).squeeze(1).sum(1)
                posterior_lp = torch.distributions.Categorical(probs=posterior_probs).log_prob(x).sum(-1)
                total_posterior_log_score[:, i] = posterior_lp.detach() if np.random.rand() < detach_prob else posterior_lp
           

        if denoise:
            # denoising step
            x = projector(x)
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device)
            sigma = self.noise(t)[0]
            # print(timesteps[-1], x, sigma)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                posterior_log_scores = self.posterior_model(x, sigma.reshape(-1))
                posterior_score = posterior_log_scores.exp()
                posterior_stag_score = self.graph.staggered_score(posterior_score, sigma)
                posterior_probs = posterior_stag_score * self.graph.transp_transition(x, sigma)
                
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    prior_log_scores = self.prior_model(x, sigma.reshape(-1))
                    prior_score = prior_log_scores.exp()
                    prior_stag_score = self.graph.staggered_score(prior_score, sigma)
                    prior_probs = prior_stag_score * self.graph.transp_transition(x, sigma)

            # truncate probabilities
            probs = prior_probs if sample_from_prior else posterior_probs
            probs.detach()
            if self.graph.absorb:
                probs = probs[..., :-1]
            
            x = sample_categorical(probs).detach()
            # traj.append(x.detach())
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # total_prior_log_score[:, -1] = prior_probs.log().gather(2, x.unsqueeze(1)).squeeze(1).sum(1)
                total_prior_log_score[:, -1] = torch.distributions.Categorical(probs=prior_probs).log_prob(x).sum(-1)
                # total_posterior_log_score[:, -1] = posterior_probs.log().gather(2, x.unsqueeze(1)).squeeze(1).sum(1)
                total_posterior_log_score[:, -1] = torch.distributions.Categorical(probs=posterior_probs).log_prob(x).sum(-1)
                
        return {
            'x': x,
            'posterior_log_probs': total_posterior_log_score,
            'prior_log_probs': total_prior_log_score,
            # 'traj': traj
        }

    def sample_back(self, x, sample_from_prior=False, detach_prob=0.8, denoise=True, tokenizer=None, projector=lambda x: x):
        # projector = lambda x: x
        # prefix_ids = tokenizer("David noticed he had put on a lot of weight recently. He examined his habits to try and figure out the reason. He realized he'd been eating too much fast food lately.").input_ids
        # suffix_ids = tokenizer("After a few weeks, he started to feel much better.").input_ids
        
        # input_ids = prefix_ids + suffix_ids
        # input_locs = list(range(len(prefix_ids))) + list(range(60-len(suffix_ids), 60))
        # input_ids = torch.tensor(input_ids, device="cuda")[None].repeat(x.shape[0], 1)
        
        # def projector(inp):
        #     inp = inp.clone()
        #     with torch.no_grad():
        #         inp[:, input_locs] = input_ids
        #     return inp

        total_prior_log_score_back = torch.zeros(x.shape[0], self.sampling_len+1, device=self.device)
        total_posterior_log_score_back = torch.zeros(x.shape[0], self.sampling_len+1, device=self.device)

        timesteps = torch.linspace(1, self.eps, self.sampling_len + 1, device=self.device)
        dt = (1 - self.eps) / self.sampling_len
        if denoise:
            x = projector(x)
            t = timesteps[self.sampling_len] * torch.ones(x.shape[0], 1, device=self.device)
            sigma = self.noise(t)[0]
            # perturbed_x = traj[-2]
            perturbed_x = self.graph.sample_transition(x, sigma)
            # print(timesteps[self.sampling_len], perturbed_x, sigma)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                posterior_log_scores = self.posterior_model(perturbed_x, sigma.reshape(-1))
                posterior_score = posterior_log_scores.exp()
                posterior_stag_score = self.graph.staggered_score(posterior_score, sigma)
                posterior_probs = posterior_stag_score * self.graph.transp_transition(perturbed_x, sigma)
                
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    prior_log_scores = self.prior_model(perturbed_x, sigma.reshape(-1))
                    prior_score = prior_log_scores.exp()
                    prior_stag_score = self.graph.staggered_score(prior_score, sigma)
                    prior_probs = prior_stag_score * self.graph.transp_transition(perturbed_x, sigma)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # total_prior_log_score_back[:, self.sampling_len] = prior_probs.log_softmax(-1).gather(2, x.unsqueeze(1)).squeeze(1).sum(1)
                total_prior_log_score_back[:, self.sampling_len] = torch.distributions.Categorical(probs=prior_probs).log_prob(x).sum(-1)
                # total_posterior_log_score_back[:, self.sampling_len] = posterior_probs.log_softmax(-1).gather(2, x.unsqueeze(1)).squeeze(1).sum(1)
                total_posterior_log_score_back[:, self.sampling_len] = torch.distributions.Categorical(probs=posterior_probs).log_prob(x).sum(-1)
            
            x = perturbed_x.detach()

        for i in range(self.sampling_len-1, -1, -1):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)
            x = projector(x)

            curr_sigma = self.noise(t)[0]
            next_sigma = self.noise(t - dt)[0]
            dsigma = curr_sigma - next_sigma
            # perturbed_x = traj[i]
            if i == 0:
                perturbed_x = torch.ones_like(x) * 50257
            else:
                perturbed_x = self.graph.sample_transition(x, dsigma)
            # print(timesteps[i], perturbed_x, curr_sigma, dsigma)
            
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                posterior_log_scores = self.posterior_model(perturbed_x, curr_sigma.reshape(-1))
                posterior_score = posterior_log_scores.exp()
                posterior_stag_score = self.graph.staggered_score(posterior_score, dsigma)
                posterior_probs = posterior_stag_score * self.graph.transp_transition(perturbed_x, dsigma)
                
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    prior_log_scores = self.prior_model(perturbed_x, curr_sigma.reshape(-1))
                    prior_score = prior_log_scores.exp()
                    prior_stag_score = self.graph.staggered_score(prior_score, dsigma)
                    prior_probs = prior_stag_score * self.graph.transp_transition(perturbed_x, dsigma)
            
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # total_prior_log_score_back[:, i] = prior_probs.log_softmax(-1).gather(2, x.unsqueeze(1)).squeeze(1).sum(1)
                total_prior_log_score_back[:, i] = torch.distributions.Categorical(probs=prior_probs).log_prob(x).sum(-1)
                # posterior_lp = posterior_probs.log_softmax(-1).gather(2, x.unsqueeze(1)).squeeze(1).sum(1)
                posterior_lp = torch.distributions.Categorical(probs=posterior_probs).log_prob(x).sum(-1)
                total_posterior_log_score_back[:, i] = posterior_lp.detach() if np.random.rand() < detach_prob else posterior_lp
            x = perturbed_x.detach()
        

        # t = rev_timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device)
        # # print(rev_timesteps[i])
        # x = projector(x)
        # curr_sigma = self.noise(t)[0]
        # prev_sigma = self.noise(t - dt)[0]
        # dsigma = curr_sigma - prev_sigma
        # # import pdb; pdb.set_trace();
        # perturbed_x = torch.ones_like(x) * 50257
        # print(rev_timesteps[-1], traj[-2 -i])
        # # perturbed_x = self.graph.sample_transition(x, dsigma)
        
        # with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        #     posterior_log_scores = self.posterior_model(perturbed_x, curr_sigma.reshape(-1))
        #     posterior_score = posterior_log_scores.exp()
        #     posterior_stag_score = self.graph.staggered_score(posterior_score, dsigma)
        #     posterior_probs = posterior_stag_score * self.graph.transp_transition(perturbed_x, dsigma)
            
        
        # with torch.no_grad():
        #     with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        #         prior_log_scores = self.prior_model(perturbed_x, curr_sigma.reshape(-1))
        #         prior_score = prior_log_scores.exp()
        #         prior_stag_score = self.graph.staggered_score(prior_score, dsigma)
        #         prior_probs = prior_stag_score * self.graph.transp_transition(perturbed_x, dsigma)
        
        # with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        #     total_prior_log_score_back[:, -1] = prior_probs.log_softmax(-1).gather(2, x.unsqueeze(1)).squeeze(1).sum(1)
        #     posterior_lp = posterior_probs.log_softmax(-1).gather(2, x.unsqueeze(1)).squeeze(1).sum(1)
        #     total_posterior_log_score_back[:, -1] = posterior_lp.detach() if np.random.rand() < detach_prob else posterior_lp
        # x = perturbed_x.detach()
        

        return {
            'posterior_log_pf': total_posterior_log_score_back,
            'prior_log_pf': total_prior_log_score_back
        }

    def sample_back_and_forth(self, x, k, detach_prob=0.8, denoise=True, sample_from_prior=False, tokenizer=None, projector=lambda x: x):
        # projector = lambda x: x
        # prefix_ids = tokenizer("David noticed he had put on a lot of weight recently. He examined his habits to try and figure out the reason. He realized he'd been eating too much fast food lately.").input_ids
        # suffix_ids = tokenizer("After a few weeks, he started to feel much better.").input_ids
        
        # input_ids = prefix_ids + suffix_ids
        # input_locs = list(range(len(prefix_ids))) + list(range(60-len(suffix_ids), 60))
        # input_ids = torch.tensor(input_ids, device="cuda")[None].repeat(x.shape[0], 1)
        
        # def projector(inp):
        #     inp = inp.clone()
        #     with torch.no_grad():
        #         inp[:, input_locs] = input_ids
        #     return inp
        
        total_prior_log_score_back = torch.zeros(x.shape[0], self.sampling_len+1, device=self.device)
        total_posterior_log_score_back = torch.zeros(x.shape[0], self.sampling_len+1, device=self.device)
        
        total_prior_log_score_forth = torch.zeros(x.shape[0], self.sampling_len+1, device=self.device)
        total_posterior_log_score_forth = torch.zeros(x.shape[0], self.sampling_len+1, device=self.device)
        

        timesteps = torch.linspace(1, self.eps, self.sampling_len + 1, device=self.device)
        rev_timesteps = torch.linspace(self.eps, 1, self.sampling_len + 1, device=self.device)
        dt = (1 - self.eps) / self.sampling_len

        if denoise:
            x = projector(x)
            t = timesteps[self.sampling_len] * torch.ones(x.shape[0], 1, device=self.device)
            sigma = self.noise(t)[0]
            # perturbed_x = traj[-2]
            perturbed_x = self.graph.sample_transition(x, sigma)
            # print(timesteps[self.sampling_len], perturbed_x, sigma)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                posterior_log_scores = self.posterior_model(perturbed_x, sigma.reshape(-1))
                posterior_score = posterior_log_scores.exp()
                posterior_stag_score = self.graph.staggered_score(posterior_score, sigma)
                posterior_probs = posterior_stag_score * self.graph.transp_transition(perturbed_x, sigma)
                
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    prior_log_scores = self.prior_model(perturbed_x, sigma.reshape(-1))
                    prior_score = prior_log_scores.exp()
                    prior_stag_score = self.graph.staggered_score(prior_score, sigma)
                    prior_probs = prior_stag_score * self.graph.transp_transition(perturbed_x, sigma)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # total_prior_log_score_back[:, self.sampling_len] = prior_probs.log_softmax(-1).gather(2, x.unsqueeze(1)).squeeze(1).sum(1)
                total_prior_log_score_back[:, self.sampling_len] = torch.distributions.Categorical(probs=prior_probs).log_prob(x).sum(-1)
                # total_posterior_log_score_back[:, self.sampling_len] = posterior_probs.log_softmax(-1).gather(2, x.unsqueeze(1)).squeeze(1).sum(1)
                total_posterior_log_score_back[:, self.sampling_len] = torch.distributions.Categorical(probs=posterior_probs).log_prob(x).sum(-1)
            
            x = perturbed_x.detach()

        for i in range(self.sampling_len-1, k-1, -1):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)
            x = projector(x)

            curr_sigma = self.noise(t)[0]
            next_sigma = self.noise(t - dt)[0]
            dsigma = curr_sigma - next_sigma
            # perturbed_x = traj[i]
            # if i == 0:
            #     perturbed_x = torch.ones_like(x) * 50257
            # else:
            perturbed_x = self.graph.sample_transition(x, dsigma)
            # print(i, timesteps[i], perturbed_x, curr_sigma, dsigma)
            
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                posterior_log_scores = self.posterior_model(perturbed_x, curr_sigma.reshape(-1))
                posterior_score = posterior_log_scores.exp()
                posterior_stag_score = self.graph.staggered_score(posterior_score, dsigma)
                posterior_probs = posterior_stag_score * self.graph.transp_transition(perturbed_x, dsigma)

            
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    prior_log_scores = self.prior_model(perturbed_x, curr_sigma.reshape(-1))
                    prior_score = prior_log_scores.exp()
                    prior_stag_score = self.graph.staggered_score(prior_score, dsigma)
                    prior_probs = prior_stag_score * self.graph.transp_transition(perturbed_x, dsigma)
            
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # total_prior_log_score_back[:, i] = prior_probs.log_softmax(-1).gather(2, x.unsqueeze(1)).squeeze(1).sum(1)
                total_prior_log_score_back[:, i] = torch.distributions.Categorical(probs=prior_probs).log_prob(x).sum(-1)
                # posterior_lp = posterior_probs.log_softmax(-1).gather(2, x.unsqueeze(1)).squeeze(1).sum(1)
                posterior_lp = torch.distributions.Categorical(probs=posterior_probs).log_prob(x).sum(-1)
                total_posterior_log_score_back[:, i] = posterior_lp.detach() if np.random.rand() < detach_prob else posterior_lp
            x = perturbed_x.detach()

        for i in range(k, self.sampling_len):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)
            x = projector(x)
            curr_sigma = self.noise(t)[0]
            next_sigma = self.noise(t - dt)[0]
            dsigma = curr_sigma - next_sigma
            # print(i, timesteps[i], x, curr_sigma, dsigma)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                posterior_log_scores = self.posterior_model(x, curr_sigma.reshape(-1))
                posterior_score = posterior_log_scores.exp()
                posterior_stag_score = self.graph.staggered_score(posterior_score, dsigma)
                posterior_probs = posterior_stag_score * self.graph.transp_transition(x, dsigma)

                
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    prior_log_scores = self.prior_model(x, curr_sigma.reshape(-1))
                    prior_score = prior_log_scores.exp()
                    prior_stag_score = self.graph.staggered_score(prior_score, dsigma)
                    prior_probs = prior_stag_score * self.graph.transp_transition(x, dsigma)


            x = sample_categorical(prior_probs if sample_from_prior else posterior_probs).detach()
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # total_prior_log_score_forth[:, i] = prior_probs.log_softmax(-1).gather(2, x.unsqueeze(1)).squeeze(1).sum(1)
                total_prior_log_score_forth[:, i] = torch.distributions.Categorical(probs=prior_probs).log_prob(x).sum(-1)
                # posterior_lp = posterior_probs.log_softmax(-1).gather(2, x.unsqueeze(1)).squeeze(1).sum(1)
                posterior_lp = torch.distributions.Categorical(probs=posterior_probs).log_prob(x).sum(-1)
                total_posterior_log_score_forth[:, i] = posterior_lp.detach() if np.random.rand() < detach_prob else posterior_lp
            

        if denoise:
            # denoising step
            x = projector(x)
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device)
            sigma = self.noise(t)[0]
            # print(timesteps[-1], x, sigma)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                posterior_log_scores = self.posterior_model(x, sigma.reshape(-1))
                posterior_score = posterior_log_scores.exp()
                posterior_stag_score = self.graph.staggered_score(posterior_score, sigma)
                posterior_probs = posterior_stag_score * self.graph.transp_transition(x, sigma)
                
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    prior_log_scores = self.prior_model(x, sigma.reshape(-1))
                    prior_score = prior_log_scores.exp()
                    prior_stag_score = self.graph.staggered_score(prior_score, sigma)
                    prior_probs = prior_stag_score * self.graph.transp_transition(x, sigma)

            # truncate probabilities
            probs = prior_probs if sample_from_prior else posterior_probs
            if self.graph.absorb:
                probs = probs[..., :-1]
            
            x = sample_categorical(probs).detach()
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # total_prior_log_score_forth[:, -1] = prior_probs.log_softmax(-1).gather(2, x.unsqueeze(1)).squeeze(1).sum(1)
                total_prior_log_score_forth[:, -1] = torch.distributions.Categorical(probs=prior_probs).log_prob(x).sum(-1)
                # posterior_lp = posterior_probs.log_softmax(-1).gather(2, x.unsqueeze(1)).squeeze(1).sum(1)
                posterior_lp = torch.distributions.Categorical(probs=posterior_probs).log_prob(x).sum(-1)
                total_posterior_log_score_forth[:, -1] = posterior_lp.detach() if np.random.rand() < detach_prob else posterior_lp
        
        return {
            'x': x,
            'posterior_log_pf_back': total_posterior_log_score_back,
            'prior_log_pf_back': total_prior_log_score_back,
            'posterior_log_pf_forth': total_posterior_log_score_forth,
            'prior_log_pf_forth': total_prior_log_score_forth
        }
