from plot_utils import *
import argparse
import torch
import os

from utils import set_seed, fig_to_image, get_gfn_optimizer, get_gfn_backward_loss, get_exploration_std
from models import GFN
from gflownet_losses import *
from energies import *


import matplotlib.pyplot as plt
from tqdm import trange


parser = argparse.ArgumentParser(description='pretrain_prior')
parser.add_argument('--lr_policy', type=float, default=1e-3)
parser.add_argument('--lr_flow', type=float, default=1e-1)
parser.add_argument('--lr_back', type=float, default=1e-3)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--s_emb_dim', type=int, default=64)
parser.add_argument('--t_emb_dim', type=int, default=64)
parser.add_argument('--harmonics_dim', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--epochs', type=int, default=10000)
parser.add_argument('--T', type=int, default=100)
parser.add_argument('--t_scale', type=float, default=5.)
parser.add_argument('--log_var_range', type=float, default=4.)


parser.add_argument('--exploratory', action='store_true', default=False)
parser.add_argument('--langevin', action='store_true', default=False)
parser.add_argument('--langevin_scaling_per_dimension', action='store_true', default=False)
parser.add_argument('--conditional_flow_model', action='store_true', default=False)
parser.add_argument('--learn_pb', action='store_true', default=False)
parser.add_argument('--pb_scale_range', type=float, default=0.1)
parser.add_argument('--learned_variance', action='store_true', default=False)
parser.add_argument('--partial_energy', action='store_true', default=False)
parser.add_argument('--exploration_factor', type=float, default=0.1)
parser.add_argument('--exploration_wd', action='store_true', default=False)
parser.add_argument('--clipping', action='store_true', default=False)
parser.add_argument('--lgv_clip', type=float, default=1e2)
parser.add_argument('--gfn_clip', type=float, default=1e4)
parser.add_argument('--zero_init', action='store_true', default=True)
parser.add_argument('--pis_architectures', action='store_true', default=True)
parser.add_argument('--lgv_layers', type=int, default=3)
parser.add_argument('--joint_layers', type=int, default=2)
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--weight_decay', type=float, default=1e-7)
parser.add_argument('--use_weight_decay', action='store_true', default=False)

parser.add_argument('--name', type=str, default='prior_25gmm', help='Name of the run')
args = parser.parse_args()

set_seed(args.seed)
if 'SLURM_PROCID' in os.environ:
    args.seed += int(os.environ["SLURM_PROCID"])

plot_data_size = 2000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_energy():

    energy = TwentyFiveGaussianMixture(device=device)

    return energy


def plot_step(energy, gfn_model, name):
    batch_size = plot_data_size
    samples = gfn_model.sample(batch_size, energy.log_reward)
    gt_samples = energy.sample(batch_size)

    fig_contour, ax_contour = get_figure(bounds=(-13., 13.))
    fig_kde, ax_kde = get_figure(bounds=(-13., 13.))
    fig_kde_overlay, ax_kde_overlay = get_figure(bounds=(-13., 13.))

    plot_contours(energy.log_reward, ax=ax_contour, bounds=(-13., 13.), n_contour_levels=150, device=device)
    plot_kde(gt_samples, ax=ax_kde_overlay, bounds=(-13., 13.))
    plot_kde(samples, ax=ax_kde, bounds=(-13., 13.))
    plot_samples(samples, ax=ax_contour, bounds=(-13., 13.))
    plot_samples(samples, ax=ax_kde_overlay, bounds=(-13., 13.))

    fig_contour.savefig(f'{name}contour.pdf', bbox_inches='tight')
    fig_kde_overlay.savefig(f'{name}kde_overlay.pdf', bbox_inches='tight')
    fig_kde.savefig(f'{name}kde.pdf', bbox_inches='tight')

    try:
        return {"visualization/contour": wandb.Image(fig_to_image(fig_contour)),
                "visualization/kde_overlay": wandb.Image(fig_to_image(fig_kde_overlay)),
                "visualization/kde": wandb.Image(fig_to_image(fig_kde))}
    except:
        return {}




def train_step(energy, gfn_model, gfn_optimizer, it, exploratory, exploration_factor, exploration_wd):
    gfn_model.zero_grad()

    exploration_std = get_exploration_std(it, exploratory, exploration_factor, exploration_wd)

    # True samples
    samples = energy.sample(args.batch_size).to(device)

    # MLE training
    loss = get_gfn_backward_loss('mle', samples, gfn_model, energy.log_reward,
                                 exploration_std=exploration_std)


    loss.backward()
    gfn_optimizer.step()
    return loss.item()


def train():

    name = f'pretrained/{args.name}.pt'

    energy = get_energy()
 

    config = args.__dict__
    config["Experiment"] = "{args.energy}"


    gfn_model = GFN(energy.data_ndim, args.s_emb_dim, args.hidden_dim, args.harmonics_dim, args.t_emb_dim,
                    trajectory_length=args.T, clipping=args.clipping, lgv_clip=args.lgv_clip, gfn_clip=args.gfn_clip,
                    langevin=args.langevin, learned_variance=args.learned_variance,
                    partial_energy=args.partial_energy, log_var_range=args.log_var_range,
                    pb_scale_range=args.pb_scale_range,
                    t_scale=args.t_scale, langevin_scaling_per_dimension=args.langevin_scaling_per_dimension,
                    conditional_flow_model=args.conditional_flow_model, learn_pb=args.learn_pb,
                    pis_architectures=args.pis_architectures, lgv_layers=args.lgv_layers,
                    joint_layers=args.joint_layers, zero_init=args.zero_init, device=device).to(device)


    gfn_optimizer = get_gfn_optimizer(gfn_model, args.lr_policy, args.lr_flow, args.lr_back, args.learn_pb,
                                      args.conditional_flow_model, args.use_weight_decay, args.weight_decay)

    start_epoch = 0


    print(gfn_model)
    metrics = dict()


    gfn_model.train()
    for i in trange(start_epoch, args.epochs + 1):
        metrics['train/loss'] = train_step(energy, gfn_model, gfn_optimizer, i, args.exploratory, args.exploration_factor, args.exploration_wd)
        if i % 100 == 0:

            images = plot_step(energy, gfn_model, name)
            metrics.update(images)
            plt.close('all')

            #wandb.log(metrics, step=i)
            
            if i % 1000 == 0:
                torch.save({
                    'epoch': i,
                    'model_state_dict': gfn_model.state_dict(),
                    'optimizer_state_dict': gfn_optimizer.state_dict(),
                }, name)
    images = plot_step(energy, gfn_model, name)
    metrics.update(images)
    plt.close('all')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': gfn_model.state_dict(),
        'optimizer_state_dict': gfn_optimizer.state_dict(),
    }, name)




if __name__ == '__main__':
    train()
