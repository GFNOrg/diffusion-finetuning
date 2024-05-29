from diffusers import DDPMScheduler, DDIMScheduler, DDPMPipeline
from diffusers.models.unet_2d import UNet2DModel
from models.denoisers import ScoreNet, Unet
from utils.args import fetch_args
from utils.data_loaders import get_dataset
from utils.diffusers.schedulers.scheduling_ddpm_gfn import DDPMGFNScheduler
from utils.pytorch_utils import *
from utils.diffusion import GaussianDiffusion, DiffTrainer, DiffuserTrainer, TrainingConfig
from diffusers.optimization import get_cosine_schedule_with_warmup

import torch as T
import numpy as np

# get arguments for the run
args, state = fetch_args(exp_prepend="train_prior")

# -------- OVERRIDE ARGUMENTS (if you have to) --------
# args.epochs = 30000
#
# args.algo = 'mle'
# args.t_scale = 1
# args.batch_size = 128
# args.lr = 1e-3
# args.lr_logZ = 1e-1

logtwopi = np.log(2 * 3.14159265358979)
# -----------------------------------------------------

print(args)
logger = Logger(args)
logger.save_args()

seed_experiment(args.seed)

# Get dataset for training -- default args.dataset is 'mnist'

data_dict = get_dataset(
    dataset=args.dataset,
    batch_size=args.batch_size,
    data_path=args.data_path,
    channels=args.channels,
    image_size=args.image_size,
    multi_class_index=args.multi_class_index,
    x_tensor=args.x_tensor,
    y_tensor=args.y_tensor,
    splits=args.splits,
    workers=0 if args is None or 'linux' not in args.system else 4
)
train_dataset = data_dict['train_data']
train_loader = data_dict['train_loader']
x_dim = (args.image_size, args.image_size, args.channels)

model = UNet2DModel(
    sample_size=args.image_size,  # the target image resolution
    in_channels=args.channels,  # the number of input channels, 3 for RGB images
    out_channels=args.channels,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=args.block_out_channels
)
# TO REMOVE
model_id = "google/ddpm-celebahq-256"
ddpm = DDPMPipeline.from_pretrained(model_id)
noise_scheduler = ddpm.scheduler
model = ddpm.unet
model.train()

noise_scheduler = DDPMGFNScheduler.from_config(noise_scheduler.config)
noise_scheduler.variance_type = 'fixed_large'
noise_scheduler.config.variance_type = 'fixed_large'
noise_scheduler.config['variance_type'] = 'fixed_large'
########

# model = UNet2DModel(
#     sample_size=args.image_size,  # the target image resolution
#     in_channels=args.channels,  # the number of input channels, 3 for RGB images
#     out_channels=args.channels,  # the number of output channels
#     layers_per_block=2,  # how many ResNet layers to use per UNet block
#     block_out_channels=args.block_out_channels
# )

print(f'Total params: \nFwd policy model: {(sum(p.numel() for p in model.parameters()) / 1e6):.2f}M ')


# noise_scheduler = DDPMGFNScheduler(
#     num_train_timesteps=args.traj_length,
#     beta_end=0.02,
#     beta_schedule="linear",
#     beta_start=0.0001,
#     clip_sample=True,
#     variance_type='fixed_large'
# )

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=args.epochs,
)

trainer = DiffuserTrainer(
    config=TrainingConfig(args),
    model=model,
    noise_scheduler=noise_scheduler,
    optimizer=optimizer,
    train_dataloader=data_dict['train_loader'],
    lr_scheduler=lr_scheduler
)

trainer.train()

