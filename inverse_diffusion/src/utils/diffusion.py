import math
from random import random
from functools import partial
from collections import namedtuple
from utils.simple_io import *

import wandb
from diffusers.models.unets.unet_2d import UNet2DOutput
from diffusers.utils import make_image_grid
from huggingface_hub import create_repo, upload_folder, login

from pathlib import Path
from torch import nn, einsum
from torch.cuda.amp import autocast

from torch.optim import Adam
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator
from denoising_diffusion_pytorch.attend import Attend
from utils.fid_evaluation import SCOREEvaluation
from denoising_diffusion_pytorch.version import __version__
from diffusers import DDPMPipeline, DDIMPipeline, DDIMScheduler, get_cosine_schedule_with_warmup, \
    StableDiffusionPipeline, LMSDiscreteScheduler, StableDiffusionImg2ImgPipeline, DiffusionPipeline, LDMPipeline, \
    ScoreSdeVePipeline, DDPMScheduler

from utils.diffusers.pipelines.ddim_gfn.pipeline_ddim_gfn import DDIMGFNPipeline
from utils.diffusers.pipelines.ddpm_gfn.pipeline_ddpm import DDPMGFNPipeline
from utils.diffusers.schedulers.scheduling_ddpm_gfn import DDPMGFNScheduler
from utils.diffusers.schedulers.scheduling_sde_ve_gfn import ScoreSdeVeGFNScheduler
from utils.simple_io import DictObj, folder_create
# constants
from utils.visualization import plot_samples

import os
import numpy as np
import torch
import torch.nn.functional as F
import huggingface_hub as hb
ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


# helpers functions

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cast_tuple(t, length=1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)


def divisible_by(numer, denom):
    return (numer % denom) == 0


def identity(t, *args, **kwargs):
    return t


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


# small helper modules


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )


def Downsample(dim, dim_out=None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)


# sinusoidal positional embeds


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random=False):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(
            self,
            dim,
            heads=4,
            dim_head=32
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            heads=4,
            dim_head=32,
            flash=False
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend(flash=flash)

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h=self.heads), qkv)

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


# model


class Unet(nn.Module):
    def __init__(
            self,
            dim,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=3,
            self_condition=False,
            resnet_block_groups=8,
            learned_variance=False,
            learned_sinusoidal_cond=False,
            random_fourier_features=False,
            learned_sinusoidal_dim=16,
            attn_dim_head=32,
            attn_heads=4,
            full_attn=(False, False, False, True),
            flash_attn=False
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # attention

        num_stages = len(dim_mults)
        full_attn = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        FullAttention = partial(Attention, flash=flash_attn)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(
                zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                attn_klass(dim_in, dim_head=layer_attn_dim_head, heads=layer_attn_heads),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = FullAttention(mid_dim, heads=attn_heads[-1], dim_head=attn_dim_head[-1])
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(
                zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                attn_klass(dim_out, dim_head=layer_attn_dim_head, heads=layer_attn_heads),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, time, x_self_cond=None):
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[
                                                                     -2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x) + x
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x) + x

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)


# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            model,
            *,
            image_size,
            timesteps=1000,
            sampling_timesteps=None,
            objective='pred_v',
            beta_schedule='sigmoid',
            schedule_fn_kwargs=dict(),
            ddim_sampling_eta=0.,
            auto_normalize=False,
            offset_noise_strength=0.,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
            min_snr_loss_weight=False,  # https://arxiv.org/abs/2303.09556
            min_snr_gamma=5
    ):
        super().__init__()
        # assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        # assert not model.random_or_learned_sinusoidal_cond

        self.model = model

        self.channels = self.model.in_channels
        try:
            self.self_condition = self.model.self_condition
        except:
            self.self_condition = None

        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0',
                             'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        self.beta_schedule = beta_schedule

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters
        self.sampling_timesteps = default(sampling_timesteps,
                                          timesteps)  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # offset noise strength - in blogpost, they claimed 0.1 was ideal

        self.offset_noise_strength = offset_noise_strength

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max=min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    @property
    def device(self):
        return self.betas.device

    def multi_gpu(self):
        self.model = nn.DataParallel(self.model).cuda()
        self.to('cuda')

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
                extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond=None, clip_x_start=False, rederive_pred_noise=False):
        model_output = self.model(x, t, x_self_cond)

        if isinstance(model_output, UNet2DOutput):
            model_output = model_output.sample

        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond=None, clip_denoised=True):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.inference_mode()
    def p_sample(self, x, t: int, x_self_cond=None):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device=device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x, t=batched_times, x_self_cond=x_self_cond,
                                                                          clip_denoised=True)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.inference_mode()
    def p_sample_loop(self, shape, return_all_timesteps=False):
        batch, device = shape[0], self.device

        img = torch.randn(shape, device=device)
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def ddim_sample(self, shape, return_all_timesteps=False):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1,
                               steps=sampling_timesteps + 1)  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=device)
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start=True,
                                                             rederive_pred_noise=True)

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def sample(self, batch_size=16, return_all_timesteps=False):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, image_size, image_size), return_all_timesteps=return_all_timesteps)

    @torch.inference_mode()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device=device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    @autocast(enabled=False)
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise=None, offset_noise_strength=None):
        b, c, h, w = x_start.shape

        noise = default(noise, lambda: torch.randn_like(x_start))

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device=self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        # noise sample

        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.inference_mode():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t, x_self_cond)

        if isinstance(model_out, UNet2DOutput):
            model_out = model_out.sample

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, *args, **kwargs)


class DiffTrainer(object):
    def __init__(
            self,
            diffusion_model,
            dataset,
            dataloader,
            logger,
            *,
            objective='v_prediction',
            results_folder='./results',
            train_batch_size=16,
            gradient_accumulate_every=2,
            train_lr=1e-4,
            train_num_steps=100000,
            ema_update_every=10,
            ema_decay=0.995,
            adam_betas=(0.9, 0.99),
            save_and_sample_every=1000,
            num_samples=25,
            amp=False,
            mixed_precision_type='fp16',
            split_batches=True,
            calculate_fid=False,
            inception_block_idx=2048,
            max_grad_norm=1.,
            workers=0,
            device='cpu',
            dataset_name='',
            data_path='',
            num_fid_samples=50000,
            save_best_and_latest_only=False,
            show_figures=False,
            save_figures=True,
            push_to_hf=False,
            exp_name='exp'
    ):
        super().__init__()

        # accelerator
        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision=mixed_precision_type if amp else 'no',
            log_with="tensorboard",
            project_dir=os.path.join(results_folder, "logs"),
        )

        self.workers = workers
        self.logger = logger
        # model

        self.model = nn.DataParallel(diffusion_model).to(device)
        self.channels = diffusion_model.channels
        is_ddim_sampling = diffusion_model.is_ddim_sampling

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        assert (train_batch_size * gradient_accumulate_every) >= 16, f'your effective batch size (train_batch_size x gradient_accumulate_every) should be at least 16 or above'

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        self.show_figures = show_figures
        self.save_figures = save_figures
        self.max_grad_norm = max_grad_norm

        # dataset and dataloader
        self.dataset_name = dataset_name
        self.data_path = data_path

        self.ds = dataset
        self.dl = cycle(self.accelerator.prepare(dataloader))

        assert len(self.ds) >= 100, 'you should have at least 100 images in your folder. at least 10k images recommended'

        # optimizer

        # self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)
        self.opt = Adam(diffusion_model.parameters())

        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.opt,
            num_warmup_steps=500,
            num_training_steps=self.train_num_steps,
        )

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt, self.lr_scheduler = self.accelerator.prepare(self.model, self.opt, self.lr_scheduler)

        # FID-score computation

        self.calculate_fid = calculate_fid and self.accelerator.is_main_process

        if self.calculate_fid:
            if not is_ddim_sampling:
                self.accelerator.print(
                    "WARNING: Robust FID computation requires a lot of generated samples and can therefore be very time consuming." \
                    "Consider using DDIM sampling to save time."
                )

            self.fid_scorer = SCOREEvaluation(
                batch_size=self.batch_size,
                dl=self.dl,
                sampler=self.ema.ema_model,
                channels=self.channels,
                accelerator=self.accelerator,
                stats_dir=results_folder,
                device=self.results_folder,
                num_fid_samples=num_fid_samples,
                normalize_input=False,
                inception_block_idx=inception_block_idx
            )

        if save_best_and_latest_only:
            assert calculate_fid, "`calculate_fid` must be True to provide a means for model evaluation for `save_best_and_latest_only`."
            self.best_fid = 1e10  # infinite

        self.save_best_and_latest_only = save_best_and_latest_only

        self.push_to_hf = push_to_hf

        if self.push_to_hf:
            hf_token = os.getenv('HF_TOKEN', None)
            if hf_token is None:
                print("No HuggingFace token was set in 'HF_TOKEN' env. variable. Setting push_to_hf to false.")
                self.push_to_hf = False
            else:
                print("HF login succesfull!")
                login(token=hf_token)

                self.hub_model_id = f"{hb.whoami()['name']}/{exp_name}" # xkronosx
                self.repo_id = create_repo(
                    repo_id=self.hub_model_id or Path(self.results_folder).name, exist_ok=True
                ).repo_id

        self.diffusers_scheduler = DDIMScheduler(
            num_train_timesteps=self.model.module.num_timesteps,
            beta_schedule=diffusion_model.beta_schedule,
            prediction_type=objective
        )

        self.diffusers_scheduler.scheduler.config.variance_type = 'fixed_large'
        self.diffusers_scheduler.scheduler.config['variance_type'] = 'fixed_large'
        self.diffusers_pipeline = DDIMPipeline(unet=self.model.module.model, scheduler=self.diffusers_scheduler)

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        torch.save(data, str(self.results_folder / f"{self.ds.path.split('_')[-1]}_training_params_{self.model.module.model.__class__.__name__}.pt"))

        if self.logger is not None:
            self.logger.save()
        # torch.save(self.model.module, f"{self.results_folder}/{self.ds.path.split('_')[-1]}_fwd_policy_{self.model.module.model.__class__.__name__}.pth")

        pipeline = DDIMPipeline(unet=self.model.module.model, scheduler=self.diffusers_scheduler)
        pipeline.save_pretrained(self.results_folder)

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f"{self.ds.path.split('_')[-1]}_training_params_{self.model.module.model.__class__.__name__}.pt"), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        losses = {
            'loss': 0,
        }

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.mean().item()

                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                pbar.set_description(f'loss: {total_loss:.4f}')

                # save progress
                losses['loss'] = loss
                if self.logger is not None:
                    self.logger.log(losses)  # log results

                accelerator.wait_for_everyone()

                self.opt.step()
                self.lr_scheduler.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        with torch.inference_mode():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            # all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))
                            all_images_list = list(map(lambda n: self.diffusers_pipeline(
                                batch_size=n,
                                num_inference_steps=self.model.module.sampling_timesteps,
                                output_type="np.array",
                                return_dict=False,
                                use_clipped_model_output=self.diffusers_scheduler.config.clip_sample
                            )[0], batches))

                        all_images = np.concatenate(all_images_list, axis=0)*2-1

                        folder = ''
                        if self.save_figures:
                            folder = f"{os.path.join(self.results_folder, 'samples')}/"
                            os.makedirs(folder, exist_ok=True)
                        plot_samples(title=f"it: {self.step}",
                                     samples=[x for x in all_images],
                                     filename=f'{folder}it_{self.step}.png',
                                     show=self.show_figures,
                                     save=self.save_figures)

                        # whether to calculate fid

                        if self.calculate_fid:
                            fid_score = self.fid_scorer.fid_score()
                            accelerator.print(f'fid_score: {fid_score}')
                        if self.save_best_and_latest_only:
                            if self.best_fid > fid_score:
                                self.best_fid = fid_score
                                self.save("best")
                            self.save("latest")
                        else:
                            self.save(milestone)

                    self.step += 1

                    if self.step == (self.train_num_steps // 5) and self.push_to_hf:
                        upload_folder(
                            repo_id=self.repo_id,
                            folder_path=self.results_folder,
                            commit_message=f"Iteration {self.step}",
                            ignore_patterns=["step_*", "epoch_*", "wandb*"],
                        )
                pbar.update(1)

        if self.push_to_hf:
            upload_folder(
                repo_id=self.repo_id,
                folder_path=self.results_folder,
                commit_message=f"Iteration {self.step}",
                ignore_patterns=["step_*", "epoch_*", "wandb*"],
            )

        accelerator.print('training complete')
        return self.model


class BurnInWrapper(GaussianDiffusion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, img, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return torch.abs(self.model(img, t))


# trainer class
class TrainingConfig:
    def __init__(self, args):
        self.image_size = args.image_size  # the generated image resolution
        self.train_batch_size = args.batch_size
        self.eval_batch_size = args.plot_batch_size  # how many images to sample during evaluation
        self.num_epochs = args.epochs
        self.gradient_accumulation_steps = args.accumulate_gradient_every
        self.learning_rate = args.lr
        self.lr_warmup_steps = 500
        self.save_image_epochs = 1000
        self.save_model_epochs = 1000
        self.mixed_precision = "fp16" if args.mixed_precision else 'no'  # `no` for float32, `fp16` for automatic mixed precision
        self.output_dir = args.save_folder  # the model name locally and on the HF Hub
        self.inference_steps = args.sampling_length
        self.variance_type = 'fixed_large'
        self.class_name = args.class_name
        self.push_to_wandb = args.push_to_wandb
        self.notes = args.notes

        self.push_to_hf = args.push_to_hf  # whether to upload the saved model to the HF Hub
        self.exp_name = args.exp_name  # the name of the repository to create on the HF Hub
        self.hub_private_repo = True
        self.overwrite_output_dir = True  # overwrite the old model when re-running the notebook
        self.seed = args.seed


def evaluate(batch_size, epoch, pipeline, folder, inference_steps=50, seed=123):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=batch_size,
        generator=torch.manual_seed(seed),
        num_inference_steps=inference_steps,
    ).images

    # Make a grid out of the images
    image_grid = make_image_grid(images, rows=5, cols=6)

    # Save the images
    test_dir = os.path.join(folder, "samples")
    os.makedirs(test_dir, exist_ok=True)
    filename = f"{test_dir}/{epoch:04d}.png"
    image_grid.save(filename)
    return filename


class DiffuserTrainer:

    def __init__(self, config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
        self.config = config
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.lr_scheduler = lr_scheduler

        self.accelerator = Accelerator(
            mixed_precision=self.config.mixed_precision,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            log_with="tensorboard",
            project_dir=os.path.join(self.config.output_dir, "logs"),
        )
        if self.accelerator.is_main_process:
            if self.config.output_dir is not None:
                os.makedirs(self.config.output_dir, exist_ok=True)
            if self.config.push_to_hf:
                hf_token = os.getenv('HF_TOKEN', None)
                if hf_token is None:
                    print("No HuggingFace token was set in 'HF_TOKEN' env. variable. Setting push_to_hf to false.")
                    self.config.push_to_hf = False
                else:
                    print("HF login succesfull!")
                    login(token=hf_token)

                    self.hub_model_id = f"{hb.whoami()['name']}/{self.config.exp_name}_cls{self.config.class_name}"
                    self.repo_id = create_repo(
                        repo_id=self.hub_model_id or Path(self.config.output_dir).name, exist_ok=True
                    ).repo_id

            wandb.init(
                project="_".join(self.config.exp_name.split("_")[:-1]).split('-')[0] if '_' in self.config.exp_name else self.config.exp_name,
                dir=self.config.output_dir,
                resume=True,
                mode='online' if self.config.push_to_wandb else "offline",
                config={k: str(val) if isinstance(val, (list, tuple)) else val for k, val in self.config.__dict__.items()},
                notes=self.config.notes,
                name=self.config.exp_name.split('_')[-1]
            )
            self.checkpoint_dir = f"{self.config.output_dir}checkpoints/"
            self.checkpoint_file = self.checkpoint_dir + "checkpoint.tar"
            folder_create(self.checkpoint_dir, exist_ok=True)

    def train(self):
        # Initialize accelerator and tensorboard logging

        it = 0
        if wandb.run.resumed and file_exists(self.checkpoint_file):
            checkpoint = torch.load(self.checkpoint_file)
            pipeline = DDPMPipeline.from_pretrained(self.checkpoint_dir)

            self.model = pipeline.unet
            self.model.train()
            self.noise_scheduler = pipeline.scheduler

            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            it = checkpoint["it"]
            print(f"***** RESUMING PREVIOUS RUN AT IT={it}")

        # Prepare everything
        # There is no specific order to remember, you just need to unpack the
        # objects in the same order you gave them to the prepare method.
        model, optimizer, train_dataloader, lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.lr_scheduler
        )

        global_step = 0

        # train the model
        while it < self.config.num_epochs:

            progress_bar = tqdm(total=len(train_dataloader), disable=not self.accelerator.is_local_main_process)
            progress_bar.set_description(f"Iteration: {it}")

            for step, batch in enumerate(train_dataloader):
                clean_images = batch["images"]
                # Sample noise to add to the images
                noise = torch.randn(clean_images.shape, device=clean_images.device)
                bs = clean_images.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
                    dtype=torch.int64
                )

                # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)

                with self.accelerator.accumulate(model):
                    # Predict the noise residual
                    noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                    loss = F.mse_loss(noise_pred, noise)
                    self.accelerator.backward(loss)

                    self.accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
                progress_bar.set_postfix(**logs)

                global_step += 1

                if self.accelerator.is_main_process:
                    pipeline = DDPMPipeline(unet=self.accelerator.unwrap_model(model),
                                            scheduler=DDPMScheduler.from_config(self.noise_scheduler.config))

                    img_filename = None
                    # After each epoch you optionally sample some demo images with evaluate() and save the model
                    if (it % self.config.save_image_epochs == 0
                            or it == self.config.num_epochs - 1):
                        img_filename = evaluate(
                            batch_size=self.config.eval_batch_size,
                            epoch=it,
                            pipeline=pipeline,
                            folder=self.config.output_dir,
                            inference_steps=self.config.inference_steps,
                            seed=self.config.seed
                        )

                    if img_filename is not None:
                        image = wandb.Image(img_filename, caption=f"it: {it}")
                        logs['samples'] = image

                    wandb.log(logs, step=global_step)  # log results in wandb

                    # After each epoch you optionally sample some demo images with evaluate() and save the model
                    if (it % self.config.save_model_epochs == 0
                        or it == self.config.num_epochs - 1):
                        torch.save({"it": it, "optimizer_state_dict": optimizer.state_dict()}, self.checkpoint_file)
                        pipeline.save_pretrained(self.checkpoint_dir)

                    if it % 50000 == 0 and self.config.push_to_hf:
                        upload_folder(
                            repo_id=self.repo_id,
                            folder_path=self.config.output_dir,
                            commit_message=f"Iteration {it}",
                            ignore_patterns=["step_*", "epoch_*", "wandb*"],
                        )

                it += 1

        upload_folder(
            repo_id=self.repo_id,
            folder_path=self.config.output_dir,
            commit_message=f"Iteration {it}",
            ignore_patterns=["step_*", "epoch_*", "wandb*"],
        )


def get_diffuser_ddpm(dataset, device):

    if 'mnist' in dataset.lower():
        model_id = "xkronosx/ddpm-mnist-32"
    elif 'cifar' in dataset.lower():
        model_id = "google/ddpm-cifar10-32"
    elif 'celeba' in dataset.lower():
        # model_id = "google/ddpm-ema-celebahq-256"
        model_id = "google/ddpm-celebahq-256"
    else:
        raise NotImplementedError(f"dataset {dataset} is not yet supported for hugging face pretrain diffusers.")

    ddpm = DDPMGFNPipeline.from_pretrained(model_id).to(device)
    ddpm.scheduler.config.variance_type = 'fixed_large'
    ddpm.scheduler.config['variance_type'] = 'fixed_large'

    if isinstance(ddpm, LDMPipeline):
        params = DictObj({
            'traj_length': ddpm.scheduler.config.num_train_timesteps,
            'sampling_length': len(ddpm.scheduler.timesteps),
            'image_size': ddpm.vqvae.sample_size,
            'noise_size': ddpm.unet.sample_size,
            'channels': ddpm.unet.in_channels
        })
    else:
        params = DictObj({
            'traj_length': ddpm.scheduler.config.num_train_timesteps,
            'sampling_length': len(ddpm.scheduler.timesteps),
            'image_size': ddpm.unet.sample_size,
            'noise_size': ddpm.unet.sample_size,
            'channels': ddpm.unet.in_channels
        })

    ddpm.unet.config._name_or_path = model_id
    ddpm.unet.config['_name_or_path'] = model_id

    return ddpm, params


def get_stable_diffuser(dataset, device):

    # Load the SD pipeline and add a hook
    stable_diff = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base").to(device)
    stable_diff.scheduler = LMSDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000
    )
    stable_diff.scheduler.set_timesteps(30)

    def hook_fn(module, input, output):
        module.output = output

    stable_diff.unet.mid_block.register_forward_hook(hook_fn)

    return stable_diff
