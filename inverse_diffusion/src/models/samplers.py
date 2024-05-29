import copy
import os
from typing import Optional, Union

from diffusers import DDIMPipeline, LDMPipeline, ScoreSdeVeScheduler, DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from peft import PeftConfig, PeftModel, load_peft_weights, set_peft_model_state_dict
from tqdm import tqdm
from diffusers.models.unets.unet_2d import UNet2DOutput
from huggingface_hub import create_repo, upload_folder, login

from utils.diffusers.pipelines.ddpm_dp.pipeline_ddpm_dp import DDPMDPPipeline
from utils.diffusion import linear_beta_schedule, sigmoid_beta_schedule, cosine_beta_schedule, cycle
from functools import partial
from utils.pytorch_utils import NoContext, print_gpu_memory
from torch.cuda.amp import autocast

import torch
import torch as T
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import huggingface_hub as hb
import math
import wandb

from utils.pytorch_utils import maybe_detach

logtwopi = np.log(2 * 3.14159265358979)


def identity(t, *args, **kwargs):
    return t


class GFNode(nn.Module):
    """
    Class handling gfn node specific operations
    """

    langevin = False

    def __init__(
            self,
            policy_model,
            x_dim,
            drift_model=None,
            ddim=False,
            sampling_step=1.,
            variance_type='fixed_small',
            train=False,
            clip=True,
            device='cuda',
            checkpointing=True,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.device = device

        self.x_dim = x_dim
        self.logvarrange = 2
        self.maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip else identity
        self.ddim = ddim
        self.sampling_step = sampling_step
        self.train = train
        self.variance_type = variance_type

        if not checkpointing:
            self.policy_model = nn.DataParallel(policy_model).to(device)
        else:
            self.policy_model = policy_model.to(device)
        self.drift_model = nn.DataParallel(drift_model).to(device) if drift_model is not None else None

        self.checkpointing = checkpointing
        # freeze
        if not train or drift_model is not None:
            for p in self.policy_model.parameters():
                p.requires_grad = False
            self.policy_model.eval()
        if not train and self.drift_model is not None:
            for p in self.drift_model.parameters():
                p.requires_grad = False
            self.drift_model.eval()

    def to(self, device):
        self.device = device
        return super().to(device)

    def get_parameters(self):
        return self.policy_model.parameters()

    def add_langevin(
            self,
            lgv_model=None,
            log_reward=None,
            lgv_clip=1e2,
            lgv_clipping=True,
            finetune_class=1
    ):

        self.langevin = True
        self.lgv_model = nn.DataParallel(lgv_model).to(self.device)
        self.log_reward = log_reward
        self.lgv_clip = lgv_clip
        self.lgv_clipping = lgv_clipping
        self.finetune_class = finetune_class

        if not self.train:
            for l in self.lgv_model.parameters():
                l.requires_grad = False
            self.lgv_model.eval()

    # def burn_in_drift(self, data, burn_in_iterations, args):
    #     zero_burn_in(self.drift_model.module, data, args, burn_in_iterations=burn_in_iterations)
        # diffusion_burn_in(self.drift_model.module, data, args, burn_in_iterations=burn_in_iterations)

    def run_policy(self, x, t):
        """
        in place policy run, modifies self.pfs and self.pflogvar to be used in later operations
        @param x:
        @param t:
        @return:
        """
        # m0 = T.cuda.memory_allocated()

        context = torch.no_grad() if not self.train or self.drift_model is not None else NoContext()
        with context:
            t_ = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)

            # get prob. distribution for next step
            # res = self.policy_model(x, t_)
            if self.checkpointing:
                res = torch.utils.checkpoint.checkpoint(self.policy_model, x, t_, use_reentrant=False)
            else:
                res = self.policy_model(x, t_)

            if isinstance(res, UNet2DOutput):
                res = res.sample

            if isinstance(res, tuple):
                self.pf_mean, sig = res
            else:
                self.pf_mean = res
                sig = torch.zeros_like(self.pf_mean)

        # m1 = T.cuda.memory_allocated()

        if self.drift_model is not None:
            res = self.drift_model(x, t_)
            if isinstance(res, tuple):
                mean_drift, _ = res
            else:
                mean_drift = res
            self.pf_mean += mean_drift

        self.pflogvars = sig.tanh() * self.logvarrange  # + np.log(sigma * torch.sqrt(beta)) * 2

        if self.langevin:
            scale = self.lgv_model(x, t)
            x.requires_grad_(True)
            with torch.enable_grad():
                grad_log_r = torch.autograd.grad(self.log_reward(x).log_softmax(1)[:, self.finetune_class].sum(), x)[
                    0].detach()
                grad_log_r = torch.nan_to_num(grad_log_r)
                if self.lgv_clipping:
                    grad_log_r = torch.clip(grad_log_r, -self.lgv_clip, self.lgv_clip)

            self.pf_mean += scale * grad_log_r

        return self.pf_mean, self.pflogvars

    def forward(
            self,
            x,
            t,
            posterior_log_variance_clipped,
            sqrt_one_minus_alphas_cumprod,
            sqrt_alphas_cumprod,
            posterior_mean_coef1,
            posterior_mean_coef2,
            ddim=False,
            alphas_cumprod=None,
            t_next=None,
            ddim_eta=None,
            sqrt_recip_alphas_cumprod=None,
            sqrt_recipm1_alphas_cumprod=None,
            noise=None,
            backward=False,
            last=False,
            x_n=None,
    ):

        if backward:
            assert x_n is not None, "you must provide a proper final sample 'x_n' for backword at the moment"

            if ddim:
                t_prev = t
            else:
                t_prev = t + 1

            self.noise = noise if noise is not None else torch.randn_like(x)

            # compute jump backward -- can do all at once from x_n/x_start
            x = self.maybe_clip(sqrt_alphas_cumprod[t_prev] * x_n + sqrt_one_minus_alphas_cumprod[t_prev] * self.noise)

        # --- FORWARD MOTION: compute policy then move to x_{t+1} ----

        # get prob. distribution for next step -- in place, modifies self.pf_mean and self.pflogvar
        self.run_policy(x, t)

        # compute step forward
        # https://github.com/Pablo-Lemos/ContinuousGFN/blob/e38899b7a467ff5518d18f667ee3aeefd36b9ed1/energy_sampling/models/gfn.py#L174

        x_start_pred = self.maybe_clip(sqrt_alphas_cumprod[t] * x - sqrt_one_minus_alphas_cumprod[t] * self.pf_mean)
        self.noise = noise if noise is not None else torch.randn_like(x)

        # handle last step in ddpm
        if last:
            self.posterior_std = (posterior_log_variance_clipped[t] / 2).exp()  # posterior std shouldn't be 0 (for pfs matching later), but best we can do so far is base it on t
            self.posterior_mean = x_start_pred
            return x_start_pred

        else:
            if ddim:
                noise_pred = (sqrt_recip_alphas_cumprod[t] * x - x_start_pred) / sqrt_recipm1_alphas_cumprod[t]  # epsilon (predicted)
                self.posterior_std = ddim_eta * ((1 - alphas_cumprod[t] / alphas_cumprod[t_next]) * (1 - alphas_cumprod[t_next]) / (1 - alphas_cumprod[t])).sqrt()
                self.posterior_mean = x_start_pred * alphas_cumprod[t_next].sqrt() + (1 - alphas_cumprod[t_next] - self.posterior_std ** 2).sqrt() * noise_pred # where pred_sample_direction = pred_sample_direction
            else:
                self.posterior_std = (posterior_log_variance_clipped[t] / 2).exp()
                self.posterior_mean = posterior_mean_coef1[t] * x_start_pred + posterior_mean_coef2[t] * x

        if backward:
            return x
        else:
            return self.posterior_mean + self.posterior_std * self.noise

    def get_logpf(self, x, mean=None, std=None):
        """
        @param x: prev x state
        @param new_x: new x state
        @param pf_mean: if pfs is specified, then it overrides the saved self.pfs
        @param pflogvars: if pflogvars is specified, then it overrides the saved self.pflogvars
        @return: logpf of last call to "forward method"
        """

        mean = mean if mean is not None else self.posterior_mean
        std = std if std is not None else self.posterior_std

        pf_dist = torch.distributions.Normal(mean, std)
        return pf_dist.log_prob(x).sum(tuple(range(1, len(x.shape))))

    def get_logpb(self, t, denoised_x=None, delta_x=None, denoised=False):
        # todo
        pass


class HGFNode(nn.Module):
    """
    Class handling gfn node specific operations
    """

    langevin = False

    def __init__(
            self,
            policy_model,
            x_dim,
            drift_model=None,
            ddim=False,
            sampling_step=1.,
            variance_type='fixed_small',
            train=False,
            clip=True,
            device='cuda',
            checkpointing=True,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.device = device

        self.x_dim = x_dim
        self.logvarrange = 2
        self.maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip else identity
        self.ddim = ddim
        self.sampling_step = sampling_step
        self.training = train
        self.variance_type = variance_type

        self.policy = policy_model
        if not checkpointing:
            self.policy.unet = nn.DataParallel(self.policy.unet).to(device)
        else:
            self.policy.unet = self.policy.unet.to(device)
        self.drift_model = nn.DataParallel(drift_model).to(device) if drift_model is not None else None

        # freeze
        if not train or drift_model is not None:
            for p in self.policy.unet.parameters():
                p.requires_grad = False
            self.policy.unet.eval()
        if not train and self.drift_model is not None:
            for p in self.drift_model.parameters():
                p.requires_grad = False
            self.drift_model.eval()

        self.checkpointing = checkpointing


    def get_parameters(self):
        return self.policy.unet.parameters()

    def add_langevin(
            self,
            lgv_model=None,
            log_reward=None,
            lgv_clip=1e2,
            lgv_clipping=True,
            finetune_class=1,
    ):

        self.langevin = True
        self.lgv_model = nn.DataParallel(lgv_model).to(self.device)
        self.log_reward = log_reward
        self.lgv_clip = lgv_clip
        self.lgv_clipping = lgv_clipping
        self.finetune_class = finetune_class

        if not self.training:
            for l in self.lgv_model.parameters():
                l.requires_grad = False
            self.lgv_model.eval()

    # def burn_in_drift(self, data, burn_in_iterations, args):
    #     zero_burn_in(self.drift_model.module, data, args, burn_in_iterations=burn_in_iterations)
        # diffusion_burn_in(self.drift_model.module, data, args, burn_in_iterations=burn_in_iterations)

    def run_policy(self, x, t, detach=False):
        """
        in place policy run, modifies self.pfs and self.pflogvar to be used in later operations
        @param x:
        @param t:
        @return:
        """
        # m0 = T.cuda.memory_allocated()

        context = torch.no_grad() if not self.training or self.drift_model is not None or detach else NoContext()
        with context:
            t_ = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)

            # get prob. distribution for next step
            # model_output = self.policy.unet(x, t_).sample
            if self.checkpointing:
                model_output = torch.utils.checkpoint.checkpoint(self.policy.unet, x, t_, use_reentrant=False).sample
            else:
                model_output = self.policy.unet(x, t_).sample

        context = torch.no_grad() if not self.training or detach else NoContext()
        with context:
            if self.drift_model is not None:
                res = self.drift_model(x, t_)
                if isinstance(res, tuple):
                    mean_drift, _ = res
                else:
                    mean_drift = res
                self.pf_mean += mean_drift

            langevin_correction = 0
            if self.langevin:
                scale = self.lgv_model(x, t)
                x.requires_grad_(True)
                with torch.enable_grad():
                    grad_log_r = torch.autograd.grad(self.log_reward(x).log_softmax(1)[:, self.finetune_class].sum(), x)[
                        0].detach()
                    grad_log_r = torch.nan_to_num(grad_log_r)
                    if self.lgv_clipping:
                        grad_log_r = torch.clip(grad_log_r, -self.lgv_clip, self.lgv_clip)

                langevin_correction = scale * grad_log_r

        return model_output, langevin_correction

    def to(self, device):
        self.device = device
        return super().to(device)

    def forward(
            self,
            x,
            t,
            t_next=None,
            ddim_eta=None,
            noise=None,
            backward=False,
            x_0=None,
            clip_output=False,
            detach=False,
    ):

        if backward:
            assert x_0 is not None, "you must provide a proper denoised sample 'x_0' for backward."

            self.noise = noise if noise is not None else torch.randn_like(x)

            # compute jump backward -- can do all at once from x_n/x_start
            x = self.policy.scheduler.add_noise(x_0, self.noise, t_next)

        # --- FORWARD MOTION: compute policy then move to x_{t+1} ----

        # get prob. distribution for next step -- in place, modifies self.pf_mean and self.pflogvar
        model_output, langevin_correction = self.run_policy(x, t, detach)

        results = self.policy.scheduler.step(
            model_output, t, x,
            eta=ddim_eta,
            use_clipped_model_output=clip_output,
            langevin_correction=langevin_correction,
            variance_noise=noise
        )
        self.posterior_mean = results.posterior_mean.to(self.device)
        self.posterior_std = results.posterior_std.to(self.device)
        self.noise = results.noise.to(self.device) if results.noise is not None else results.noise

        if backward:
            return x
        else:
            return results.prev_sample

    def get_logpf(self, x, mean=None, std=None):
        """
        @param x: prev x state
        @param new_x: new x state
        @param pf_mean: if pfs is specified, then it overrides the saved self.pfs
        @param pflogvars: if pflogvars is specified, then it overrides the saved self.pflogvars
        @return: logpf of last call to "forward method"
        """

        mean = mean if mean is not None else self.posterior_mean
        std = std if std is not None else self.posterior_std

        pf_dist = torch.distributions.Normal(mean, std)
        return pf_dist.log_prob(x).sum(tuple(range(1, len(x.shape))))

    def get_logpb(self, t, denoised_x=None, delta_x=None, denoised=False):
        # todo
        pass


class PosteriorPriorDGFN(nn.Module):
    """ Version of posterior-prior dgfn to work with hugging face native library"""
    def __init__(
            self,
            dim,
            prior_policy_model,
            posterior_policy_model,
            drift_model=None,
            traj_length=1000,
            sampling_length=100,
            ddim_sampling_eta=0.,
            mixed_precision=False,
            use_cuda=False,
            transforms=None,
            lora=True,
            push_to_hf=False,
            exp_name='exp',
            dataloader=None,
            checkpointing=True,
            *args,
            **kwargs
    ):

        super().__init__(*args, **kwargs)

        self.use_cuda = use_cuda
        self.device = 'cuda' if self.use_cuda else 'cpu'
        self.mixed_precision = mixed_precision

        self.dataloader = cycle(dataloader) if dataloader is not None else None

        if self.mixed_precision:
            self.context = autocast()
        else:
            self.context = NoContext()

        self.dim = dim
        self.traj_length = traj_length
        self.sampling_length = sampling_length
        self.sampling_step = traj_length / sampling_length
        self.ddim = self.sampling_step > 1.
        self.ddim_sampling_eta = ddim_sampling_eta  # 0-> DDIM, 1-> DDPM

        # ----------------------------------------------
        prior_node = HGFNode(policy_model=prior_policy_model,
                                      x_dim=dim,
                                      sampling_step=self.sampling_step,
                                      ddim=self.ddim,
                                      train=False,
                                      checkpointing=checkpointing,
                                      device=self.device)
        self.register_module('prior_node', prior_node)

        posterior_node = HGFNode(policy_model=posterior_policy_model,
                                 drift_model=drift_model,
                                 x_dim=dim,
                                 sampling_step=self.sampling_step,
                                 ddim=self.ddim,
                                 checkpointing=checkpointing,
                                 train=True,
                                 device=self.device)

        self.register_module('posterior_node', posterior_node)

        prior_params = sum(p.numel() for p in self.prior_node.get_parameters())
        posterior_params = sum(p.numel() for p in self.posterior_node.get_parameters())
        trainable_posterior_params = sum(p.numel() for p in self.posterior_node.get_parameters() if p.requires_grad)
        print(f"\nTotal params: "
              f"\nPRIOR model: {prior_params / 1e6:.2f}M "
              f"\nPOSTERIOR model: {posterior_params / 1e6:.2f}M")
        if drift_model is None:
              print(f"Trainable posterior parameters: {trainable_posterior_params / 1e6:.2f}M/{posterior_params / 1e6:.2f}M  ({trainable_posterior_params*100/posterior_params:.2f}%)\n")
        else:
            drift_params = sum(p.numel() for p in drift_model.parameters())
            trainable_drift_params = sum(p.numel() for p in drift_model.parameters() if p.requires_grad)
            print(f"Trainable drift parameters: {(trainable_drift_params + trainable_drift_params)/ 1e6:.2f}M/{(drift_params + posterior_params) / 1e6:.2f}M\n")

        self.logZ = T.nn.Parameter(T.tensor(0.).to(self.device))
        self.logZ.requires_grad = True
        self.transforms = transforms

        self.lora = lora
        self.push_to_hf = push_to_hf
        self.exp_name = exp_name

        if self.push_to_hf:
            hf_token = os.getenv('HF_TOKEN', None)
            if hf_token is None:
                print("No HuggingFace token was set in 'HF_TOKEN' env. variable. "
                      "Setting push_to_hf to false.")
                self.push_to_hf = False
            else:
                print("HF login succesfull!")
                login(token=hf_token)

                self.hub_model_id = f"{hb.whoami()['name']}/{self.exp_name}"  # xkronosx
                self.repo_id = create_repo(
                    repo_id=self.hub_model_id, exist_ok=True
                ).repo_id

    def to(self, device):
        self.device = device
        self.posterior_node.to(device)
        self.prior_node.to(device)
        return super().to(device)

    def train(self: T, mode: bool = True) -> T:
        super().train()
        self.posterior_node.policy.unet.train()
        self.prior_node.policy.unet.eval()

    def eval(self: T, mode: bool = True) -> T:
        super().eval()
        self.posterior_node.policy.unet.eval()
        self.prior_node.policy.unet.eval()

    def set_loader(self, dataloader):
        self.dataloader = cycle(dataloader)

    def add_classifier(self, classifier):
        self.classifier = classifier

    def get_scheduler(self):
        return self.posterior_node.scheduler

    def get_schedule_args(self):
        return {
            'ddim_eta': self.ddim_sampling_eta,
            'clip_output': self.prior_node.policy.scheduler.config.clip_sample if 'clip_sample' in self.prior_node.policy.scheduler.config else False
        }

    def add_langevin(self, *args, **kwargs):
        self.prior_node.add_langevin(*args, **kwargs)
        self.posterior_node.add_langevin(*args, **kwargs)

    def forward(self, back_and_forth=False, *args, **kwargs):
        with self.context:
            if back_and_forth:
                return self.sample_back_and_forth(*args, **kwargs)
            else:
                return self.sample_fwd(*args, **kwargs)

    def sample_fwd(self, batch_size=None, x_start=None, sample_from_prior=False, detach_freq=0., sampling_length=None):

        assert batch_size is not None, "provide batch_size for sample_fwd"
        sampling_length = sampling_length if sampling_length is not None else self.sampling_length

        return_dict = {}

        normal_dist = torch.distributions.Normal(torch.zeros((batch_size,) + tuple(self.dim), device=self.device),
                                                 torch.ones((batch_size,) + tuple(self.dim), device=self.device))

        x = normal_dist.sample() if x_start is None else x_start
        if self.mixed_precision and 'cuda' in self.device:
            x = x.half()

        return_dict['logpf_posterior'] = normal_dist.log_prob(x).sum(tuple(range(1, len(x.shape)))).to(self.device)
        return_dict['logpf_prior'] = normal_dist.log_prob(x).sum(tuple(range(1, len(x.shape)))).to(self.device)

        scheduler = copy.deepcopy(self.posterior_node.policy.scheduler)
        scheduler.set_timesteps(sampling_length)
        sampling_times = scheduler.timesteps

        times_to_detach = np.random.choice([t for t in sampling_times], int(sampling_length * detach_freq), replace=False)

        for i, t in tqdm(enumerate(sampling_times), total=len(sampling_times)):

            t_specific_args = {
                'noise': None if t < self.traj_length else 0.,
                'detach': t.item() in times_to_detach
            }

            step_args = self.get_schedule_args()
            step_args.update(t_specific_args)

            if sample_from_prior:
                # -- make step in x by prior model
                new_x = self.prior_node(x, t, **step_args).detach()
            else:
                # -- make a step in x by posterior model --
                new_x = self.posterior_node(x, t, **step_args).detach()

                # get posterior pf
                return_dict['logpf_posterior'] += self.posterior_node.get_logpf(x=new_x)

                # ------ compute prior pf for posterior step --------
                # update internal values of pfs and logvar for prior -- inplace
                step_args['noise'] = self.posterior_node.noise  # adjust noise to match posterior
                self.prior_node(x, t, **step_args)

                # get prior pf, given posterior move
                return_dict['logpf_prior'] += self.prior_node.get_logpf(x=new_x)
                # ---------------------------------------------------

            x = new_x

        if self.transforms is not None:
            x = self.transforms(x)
        return_dict['x'] = x

        return return_dict

    def sample_back_and_forth(self, batch_size=None, steps=50, detach_freq=0., sampling_length=None):

        assert self.dataloader is not None, 'please provide a batch of starting samples x'

        x = next(self.dataloader)['images'][:batch_size].to(self.device)

        sampling_length = sampling_length if sampling_length is not None else self.sampling_length

        normal_dist = torch.distributions.Normal(torch.zeros((len(x),) + tuple(self.dim), device=self.device),
                                                 torch.ones((len(x),) + tuple(self.dim), device=self.device))

        scheduler = copy.deepcopy(self.posterior_node.policy.scheduler)
        scheduler.set_timesteps(sampling_length)
        sampling_times = scheduler.timesteps

        return_dict = {}
        # ---------------------------------------------------
        # --------------- Move Backward ---------------------
        # ---------------------------------------------------
        if self.mixed_precision and 'cuda' in self.device:
            x = x.half()

        return_dict['logpf_posterior_b'] = 0.
        return_dict['logpf_prior_b'] = 0.

        backward_sampling_times = list(sampling_times)[::-1][:steps]

        x_start = x[:]
        return_dict['x'] = x_start
        times_to_detach = np.random.choice([t for t in sampling_times], int(sampling_length * detach_freq), replace=False)
        backward_noise = torch.randn_like(x)
        for i, t in tqdm(enumerate(backward_sampling_times), total=len(backward_sampling_times)):

            t_specific_args = {
                'noise': backward_noise,  # fix noise for backward process
                't_next': t + self.traj_length // self.sampling_length,
                'x_0': x_start,
                'backward': True,  # flag backward for backward step
            }

            step_args = self.get_schedule_args()
            step_args.update(t_specific_args)

            # -- make a backward step in x by posterior model --
            new_x = self.posterior_node(x, t, **step_args)

            # get posterior pf
            return_dict['logpf_posterior_b'] += maybe_detach(self.posterior_node.get_logpf(x=x), t, times_to_detach)

            # ------ compute prior pf for posterior step --------
            # update internal values of pfs and logvar for prior -- inplace
            self.prior_node(x, t, **step_args)

            # get prior pf, given posterior move
            return_dict['logpf_prior_b'] += maybe_detach(self.prior_node.get_logpf(x=x), t, times_to_detach)
            # ---------------------------------------------------

            x = new_x.detach()

        # ---------------------------------------------------
        # ------------------ Move Forward  ------------------
        # ---------------------------------------------------

        return_dict['logpf_posterior_f'] = normal_dist.log_prob(x).sum(tuple(range(1, len(x.shape)))).to(self.device)
        return_dict['logpf_prior_f'] = normal_dist.log_prob(x).sum(tuple(range(1, len(x.shape)))).to(self.device)

        forward_sampling_times = backward_sampling_times[::-1]

        for i, t in tqdm(enumerate(forward_sampling_times), total=len(forward_sampling_times)):

            t_specific_args = {'noise': None if t < self.traj_length else 0.}

            step_args = self.get_schedule_args()
            step_args.update(t_specific_args)

            # -- make a step in x by posterior model --
            new_x = self.posterior_node(x, t, **step_args)

            # get posterior pf
            return_dict['logpf_posterior_f'] += maybe_detach(self.posterior_node.get_logpf(x=new_x), t, times_to_detach)

            # ------ compute prior pf for posterior step --------
            # update internal values of pfs and logvar for prior -- inplace
            step_args['noise'] = self.posterior_node.noise  # adjust noice to match posterior
            self.prior_node(x, t, **step_args)

            # get prior pf, given posterior move
            return_dict['logpf_prior_f'] += maybe_detach(self.prior_node.get_logpf(x=new_x), t, times_to_detach)
            # ---------------------------------------------------

            x = new_x.detach()

        if self.transforms is not None:
            x = self.transforms(x)

        return_dict['x_prime'] = x.detach()

        return return_dict

    def sample(self, batch_size=16, sample_from_prior=False):
        return self.sample_fwd(batch_size=batch_size, sample_from_prior=sample_from_prior, x_start=None)['x'].clamp(-1, 1)

    def save(self, folder, push_to_hf, opt, it=0, logZ=0.):

        torch.save({
            "it": it,
            "optimizer_state_dict": opt.state_dict(),
            "logZ": logZ
        }, folder + "checkpoint.tar")

        if isinstance(self.posterior_node.policy.unet, nn.DataParallel):
            model = self.posterior_node.policy.unet.module
        else:
            model = self.posterior_node.policy.unet

        if self.lora:
            model.save_pretrained(folder)

            if self.push_to_hf and push_to_hf:
                # self.posterior_node.policy.unet.module.push_to_hf(self.hub_model_id)
                upload_folder(
                    repo_id=self.repo_id,
                    folder_path=folder,
                    commit_message=f"Iteration {it}",
                    ignore_patterns=["step_*", "epoch_*", "wandb*"],
                )
        else:

            pipeline = DDIMPipeline(unet=model, scheduler=self.posterior_node.policy.scheduler)
            pipeline.save_pretrained(folder)

    def load(self, folder):

        if isinstance(self.posterior_node.policy.unet, nn.DataParallel):
            model = self.posterior_node.policy.unet.module
        else:
            model = self.posterior_node.policy.unet

        if self.lora:
            # attach lora posterior
            lora_weights = load_peft_weights(folder)
            set_peft_model_state_dict(model, lora_weights)

        else:
            pipeline = DDIMPipeline.from_pretrained(folder)
            self.posterior_node.policy = pipeline
            # self.posterior_node.policy.unet = nn.DataParallel(self.posterior_node.policy.unet)


class PosteriorPriorBaselineSampler(nn.Module):

    """ Version of posterior-prior dgfn to work with hugging face native library"""
    def __init__(
            self,
            dim,
            y_dim,
            policy_model,
            scheduler,
            traj_length=1000,
            sampling_length=100,
            ddim_sampling_eta=0.,
            mixed_precision=False,
            use_cuda=False,
            transforms=None,
            lora=True,
            push_to_hf=False,
            exp_name='exp',
            dataloader=None,
            checkpointing=True,
            classifier=None,
            finetune_class=(0,),
            scale=1.,
            mc=False,
            particles=10,
            *args,
            **kwargs
    ):

        super().__init__(*args, **kwargs)

        self.use_cuda = use_cuda
        self.device = 'cuda' if self.use_cuda else 'cpu'
        self.mixed_precision = mixed_precision

        self.dataloader = cycle(dataloader) if dataloader is not None else None

        if self.mixed_precision:
            self.context = autocast()
        else:
            self.context = NoContext()

        self.dim = dim
        self.y_dim = y_dim
        self.traj_length = traj_length
        self.sampling_length = sampling_length
        self.sampling_step = traj_length / sampling_length
        self.ddim = self.sampling_step > 1.
        self.ddim_sampling_eta = ddim_sampling_eta  # 0-> DDIM, 1-> DDPM
        self.scale = scale  # scale of guidance

        self.classifier = classifier

        self.finetune_class = finetune_class

        self.mc = mc
        self.particles = particles

        # ----------------------------------------------
        self.policy = DDPMDPPipeline(unet=policy_model, scheduler=scheduler).to(self.device)

        params = sum(p.numel() for p in self.policy.get_parameters())
        print(f"\nTotal params: "
              f"\nPOLICY model: {params / 1e6:.2f}M ")

        self.transforms = transforms

        self.lora = lora
        self.push_to_hf = push_to_hf
        self.exp_name = exp_name

        if self.push_to_hf:
            hf_token = os.getenv('HF_TOKEN', None)
            if hf_token is None:
                print("No HuggingFace token was set in 'HF_TOKEN' env. variable. "
                      "Setting push_to_hf to false.")
                self.push_to_hf = False
            else:
                print("HF login succesfull!")
                login(token=hf_token)

                self.hub_model_id = f"{hb.whoami()['name']}/{self.exp_name}"  # xkronosx
                self.repo_id = create_repo(
                    repo_id=self.hub_model_id, exist_ok=True
                ).repo_id

    def to(self, device):
        self.device = device
        self.policy.unet.to(device)
        return super().to(device)

    def train(self: T, mode: bool = True) -> T:
        super().train()
        self.policy.unet.eval()

    def eval(self: T, mode: bool = True) -> T:
        super().eval()
        self.policy.unet.eval()

    def add_classifier(self, classifier):
        self.classifier = classifier

    def set_loader(self, dataloader):
        self.dataloader = cycle(dataloader)

    def get_scheduler(self):
        return self.policy.scheduler

    def get_schedule_args(self):
        return {
            'ddim_eta': self.ddim_sampling_eta,
            'clip_output': self.prior_node.scheduler.config.clip_sample if 'clip_sample' in self.prior_node.scheduler.config else False
        }

    def forward(self, batch_size=None, sampling_length=None, sample_from_prior=False, finetune_class=None, mc=None,
                particles=None, *args, **kwargs):

        self.mc = mc if mc is not None else self.mc
        self.particles = particles if particles is not None else self.particles
        finetune_class = self.finetune_class if finetune_class is None else finetune_class

        assert sample_from_prior or self.classifier is not None, "a classifier must be added before sampling from posterior"
        assert sample_from_prior or finetune_class is not None, "a measurement (e.g. class target) must be given before sampling from posterior"
        assert batch_size is not None, "provide batch_size for sampling"
        sampling_length = sampling_length if sampling_length is not None else self.sampling_length

        return_dict = {}

        if isinstance(self.policy.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.policy.unet.config.in_channels,
                self.policy.unet.config.sample_size,
                self.policy.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.policy.unet.config.in_channels, *self.policy.unet.config.sample_size)

        image = randn_tensor(image_shape).to(self.device)  # noise
        measurement = torch.zeros(self.y_dim).to(self.device)
        measurement[finetune_class] = 1
        measurement_noise = randn_tensor(measurement.shape).to(self.device)

        if self.mixed_precision and 'cuda' in self.device:
            image = image.half()

        # set step values
        self.policy.scheduler.set_timesteps(sampling_length)

        for t in self.policy.progress_bar(self.policy.scheduler.timesteps):
            # 1. predict noise model_output
            image = image.requires_grad_(True)
            model_output = self.policy.unet(image, t).sample

            # 2. compute previous image: x_t -> x_t-1
            res = self.policy.scheduler.step(model_output, t, image)
            image_t_minus_1 = res.prev_sample  # x_t-1 according to prior

            if not sample_from_prior:
                x_0_hat = res.pred_original_sample
                noisy_measurement = self.policy.scheduler.add_noise(measurement, measurement_noise, t)

                if self.mc:
                    # shapes = [self.particles] + list(x_0_hat.shape)
                    # batch_size = x_0_hat.shape[0]  # ?
                    sigma_t = res.posterior_std
                    r_t = sigma_t / torch.sqrt(1 + sigma_t ** 2)
                    # x_0_hat_shape = x_0_hat.view(-1).shape[0]
                    # x_0_hat_distr = torch.distributions.MultivariateNormal(torch.zeros(x_0_hat_shape), (r_t ** 2) * (torch.eye(x_0_hat_shape)))
                    # x_0_estimates = torch.reshape(torch.flatten(x_0_hat) + x_0_hat_distr.sample_n(self.particles).to(self.device), shapes)

                    differences = [noisy_measurement - self.policy.scheduler.add_noise(
                        self.classifier(x_0_hat + torch.randn_like(x_0_hat)*r_t),
                        measurement_noise, t) for _ in range(self.particles)]
                    norms = torch.stack([torch.linalg.norm(difference) for difference in differences], dim=0)

                    mc_norm_estimates = torch.logsumexp(-norms, dim=0) - math.log(float(self.particles))

                    guidance = torch.autograd.grad(outputs=mc_norm_estimates, inputs=image)[0]
                else:
                    prediction = self.classifier(x_0_hat)
                    noisy_prediction = self.policy.scheduler.add_noise(prediction, measurement_noise, t)
                    # compute guidance diff. and corresponding norm
                    difference = noisy_measurement - noisy_prediction  # (as is might not work so well for multi-modal distributions)
                    norm = torch.linalg.norm(difference)

                    # print(norm)
                    # norm_grad = torch.autograd.grad(outputs=norm, inputs=image_t_minus_1)[0]
                    norm_grad = torch.autograd.grad(outputs=norm, inputs=image)[0]
                    guidance = -norm_grad

                # apply guidance
                image = (image_t_minus_1 + guidance * self.scale).detach()

            else:
                image = image_t_minus_1.detach()

        return_dict['x'] = image

        return return_dict

    def sample(self, batch_size=16, sample_from_prior=False):
        # important: fid_score uses this function -- make sure the output range is as expected (noramlized=fale -> -1, 1, else 0, 1)
        return self(batch_size=batch_size, sample_from_prior=sample_from_prior)['x'].clip(-1, 1)  # todo reset range to [-1, +1]

    def save(self, folder, push_to_hf, opt, it=0, *args, **kwargs):
        # modify this to have all necessary to resume a run
        torch.save({
            "it": it,
            "optimizer_state_dict": opt.state_dict(),
        }, folder + "checkpoint.tar")

        pipeline = DDIMPipeline(unet=self.policy.unet.module, scheduler=self.policy.scheduler)
        pipeline.save_pretrained(folder)

    def load(self, folder):
        pipeline = DDPMDPPipeline.from_pretrained(folder)
        self.posterior_node = pipeline
        self.policy.unet = nn.DataParallel(self.policy.unet)
