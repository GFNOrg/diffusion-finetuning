from plot_utils import *
import argparse
import torch
import os

from utils import set_seed, fig_to_image
from models import GFN
from gflownet_losses import *
from energies import *
import copy
import matplotlib.pyplot as plt
from tqdm import trange


parser = argparse.ArgumentParser(description='classifier_guidance_posterior')


parser.add_argument('--seed', type=int, default=12345)

parser.add_argument('--name', type=str, default='classifier_guidance')


args = parser.parse_args()

set_seed(args.seed)
if 'SLURM_PROCID' in os.environ:
    args.seed += int(os.environ["SLURM_PROCID"])

plot_data_size = 2000



args.zero_init = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_energy():
    prior = TwentyFiveGaussianMixture(device=device)
    energy = Posterior2DGaussianMixture(device=device)
    return energy, prior




def inference():
    energy, prior_energy = get_energy()
    name = args.name

    gfn_model = GFN(2, 64, 64, 64, 64,
                    trajectory_length=100, clipping=True, lgv_clip=1e2, gfn_clip=1e4,
                    langevin=False, learned_variance=False,
                    partial_energy=False, log_var_range=4.,
                    pb_scale_range=0.1,
                    t_scale=5.0, langevin_scaling_per_dimension=False,
                    conditional_flow_model=False, learn_pb=False,
                    pis_architectures=True, lgv_layers=3,
                    joint_layers=2, zero_init=True, device=device).to(device)

    start_epoch = 0


    checkpoint_path = 'pretrained/prior.pt'

    checkpoint = torch.load(checkpoint_path)
    
    if 'model_state_dict' in checkpoint:
        gfn_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        gfn_model.load_state_dict(checkpoint)

    start_epoch = 0

    prior = copy.deepcopy(gfn_model)
    prior.eval()


    initial_state = torch.zeros(plot_data_size, energy.data_ndim).to(device)
    states, _, _, _ = prior.get_trajectory_fwd_classifier_guidance(initial_state, None, log_r=  prior_energy.log_reward, 
                                                                            log_classifier = energy.log_reward, guid_stren = 1.0)
    samples = states[:, -1]
    gt_samples = energy.sample(plot_data_size)

    fig_contour, ax_contour = get_figure(bounds=(-13., 13.))
    fig_kde, ax_kde = get_figure(bounds=(-13., 13.))
    fig_kde_overlay, ax_kde_overlay = get_figure(bounds=(-13., 13.))

    plot_contours(energy.log_reward, ax=ax_contour, bounds=(-13., 13.), n_contour_levels=150, device=device)
    plot_kde(gt_samples, ax=ax_kde_overlay, bounds=(-13., 13.))
    plot_kde(samples, ax=ax_kde, bounds=(-13., 13.))
    plot_samples(samples, ax=ax_contour, bounds=(-13., 13.))
    plot_samples(samples, ax=ax_kde_overlay, bounds=(-13., 13.))

    fig_contour.savefig(f'output/{name}_contour.png', bbox_inches='tight')
    fig_kde_overlay.savefig(f'output/{name}_kde_overlay.png', bbox_inches='tight')
    fig_kde.savefig(f'output/{name}_kde.png', bbox_inches='tight')

        


if __name__ == '__main__':
    inference()
