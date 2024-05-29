from diffusers import get_cosine_schedule_with_warmup

from models.denoisers import ScoreNet, Unet
from utils.args import fetch_args
from utils.data_loaders import get_dataset
from utils.pytorch_utils import *
from utils.diffusion import GaussianDiffusion, DiffTrainer
from diffusers.models.unet_2d import UNet2DModel

import torch as T
import numpy as np

from utils.visualization import plot_samples

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

fwd_policy = UNet2DModel(
    sample_size=args.image_size,  # the target image resolution
    in_channels=args.channels,  # the number of input channels, 3 for RGB images
    out_channels=args.channels,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=args.block_out_channels
)

diffusion = GaussianDiffusion(
    fwd_policy,
    image_size=args.image_size,
    timesteps=args.traj_length,  # number of steps
    sampling_timesteps=args.sampling_length,
    beta_schedule='linear',
    objective='pred_v'
)

trainer = DiffTrainer(
    diffusion,
    train_dataset,
    train_loader,
    logger,
    objective='v_prediction',
    results_folder=args.save_folder,
    train_batch_size=args.batch_size,
    train_lr=1e-4,
    train_num_steps=args.epochs,
    workers=args.workers,
    num_samples=args.plot_batch_size,
    device=args.device,
    show_figures=args.show_figures,
    save_figures=args.save_figures,
    push_to_hf=args.push_to_hf,
    exp_name=args.exp_name,
    inception_block_idx=args.fid_inception_block
)

trainer.train()

