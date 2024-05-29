from plot_utils import *
import argparse
import torch
import os

from utils import set_seed, fig_to_image, get_gfn_optimizer, \
    get_gfn_backward_loss, get_exploration_std, get_finetuning_loss
from models import GFN
from gflownet_losses import *
from energies import *
import copy
import matplotlib.pyplot as plt
from tqdm import trange

WANDB = True

if WANDB:
    import wandb

parser = argparse.ArgumentParser(description='finetuning posterior')
parser.add_argument('--lr_policy', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--epochs', type=int, default=5000)
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--kl_weight', type=float, default=1.)
parser.add_argument('--name', type=str, default='rtb_finetuning')
parser.add_argument('--method', type=str, default='rtb')

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

    fig_contour.savefig(f'output/{name}_contour.png', bbox_inches='tight')
    fig_kde_overlay.savefig(f'output/{name}_kde_overlay.png', bbox_inches='tight')
    fig_kde.savefig(f'output/{name}_kde.png', bbox_inches='tight')
    try:
        return {"visualization/contour": wandb.Image(fig_to_image(fig_contour)),
                "visualization/kde_overlay": wandb.Image(fig_to_image(fig_kde_overlay)),
                "visualization/kde": wandb.Image(fig_to_image(fig_kde))}
    except:
        return {}


def train_step(energy, prior, gfn_model, gfn_optimizer, it, method, exploratory, exploration_factor, exploration_wd, beta = 1.0, kl_weight=1.0):
    gfn_model.zero_grad()
    exploration_std = get_exploration_std(it, exploratory, exploration_factor, exploration_wd)
    if method == 'rtb':
        
        loss, kl_div = fwd_train_step(energy, prior, gfn_model, exploration_std, method, beta = beta)
        loss.backward()
        gfn_optimizer.step()
        return loss.item(), kl_div.item()
    elif method == 'rl':
        exploration_std = get_exploration_std(it, False, 0.0, False)

        rl_loss, kl_loss, kl_div = fwd_train_step(energy, prior, gfn_model, exploration_std, method)
        loss = (rl_loss + kl_weight * kl_loss).mean()
        loss.backward()
        gfn_optimizer.step()
        return loss.item(), rl_loss.mean().item(), kl_loss.mean().item(), kl_div.item()

def fwd_train_step(energy, prior, gfn_model, exploration_std, method, return_exp=False, beta=1.0):
    init_state = torch.zeros(args.batch_size, energy.data_ndim).to(device)
    
    if method == 'rtb':
        return get_finetuning_loss('rtb', init_state, prior, gfn_model, energy.log_reward, beta = beta, exploration_std=exploration_std, return_exp=return_exp)
    else:
        return get_finetuning_loss('rl', init_state, prior, gfn_model, energy.log_reward, beta = beta, exploration_std=exploration_std, return_exp=return_exp)





def train():
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


    gfn_optimizer = get_gfn_optimizer(gfn_model, args.lr_policy, 0.1, args.lr_policy, False,
                                      False, False, False)

    start_epoch = 0


    checkpoint_path = 'pretrained/prior.pt'

    checkpoint = torch.load(checkpoint_path)
    
    if 'model_state_dict' in checkpoint:
        gfn_model.load_state_dict(checkpoint['model_state_dict'])
        gfn_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        gfn_model.load_state_dict(checkpoint)

    start_epoch = 0

    prior = copy.deepcopy(gfn_model)
    prior.eval()
    method = args.method

    metrics = dict()


    gfn_model.train()
    for i in trange(start_epoch, args.epochs + 1):
        if method == 'rtb':
            # off-policy: with exploration noise (True)
            loss, kl_div = train_step(energy, prior, gfn_model, gfn_optimizer, i, method, True, 0.5, True, beta=1.0)
            
            metrics['train/loss'] = loss
            metrics['train/kl_div'] = kl_div
        else:
            # on-policy: no exploratinon noise (False)
            loss, rl_loss, kl_loss, kl_div = train_step(energy, prior, gfn_model, gfn_optimizer, i, method, False, 0.0, False, beta=1.0, kl_weight=args.kl_weight)
    
            metrics['train/loss'] = loss
            metrics['train/rl_loss'] = rl_loss
            metrics['train/kl_loss'] = kl_loss
            metrics['train/kl_div'] = kl_div
        
        if i % 100 == 0:

            images = plot_step(energy, gfn_model, name)
            metrics.update(images)
            plt.close('all')

            # you may put logger here
            #########################
            # wandb.log(metrics)
            

            #########################
            print(metrics)

            if i % 1000 == 0:
                torch.save({
                    'epoch': i,
                    'model_state_dict': gfn_model.state_dict(),
                    'optimizer_state_dict': gfn_optimizer.state_dict(),
                }, f'output/{name}.pt')

    images = plot_step(energy, gfn_model, name)
    metrics.update(images)
    plt.close('all')

    torch.save({
        'epoch': args.epochs,
        'model_state_dict': gfn_model.state_dict(),
        'optimizer_state_dict': gfn_optimizer.state_dict(),
    }, f'output/{name}.pt')
        


if __name__ == '__main__':
    train()
