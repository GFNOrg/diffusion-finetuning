import random
import numpy as np
import math
import PIL

from gflownet_losses import *


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def logmeanexp(x, dim=0):
    return x.logsumexp(dim) - math.log(x.shape[dim])


def dcp(tensor):
    return tensor.detach().cpu()


def gaussian_params(tensor):
    mean, logvar = torch.chunk(tensor, 2, dim=-1)
    return mean, logvar


def fig_to_image(fig):
    fig.canvas.draw()

    return PIL.Image.frombytes(
        "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
    )


def get_gfn_optimizer(gfn_model, lr_policy, lr_flow, lr_back, back_model=False, conditional_flow_model=False, use_weight_decay=False, weight_decay=1e-7):
    param_groups = [ {'params': gfn_model.t_model.parameters()},
                     {'params': gfn_model.s_model.parameters()},
                     {'params': gfn_model.joint_model.parameters()},
                     {'params': gfn_model.langevin_scaling_model.parameters()} ]
    if conditional_flow_model:
        param_groups += [ {'params': gfn_model.flow_model.parameters(), 'lr': lr_flow} ]
    else:
        param_groups += [ {'params': [gfn_model.flow_model], 'lr': lr_flow} ]

    if back_model:
        param_groups += [ {'params': gfn_model.back_model.parameters(), 'lr': lr_back} ]

    if use_weight_decay:
        gfn_optimizer = torch.optim.Adam(param_groups, lr_policy, weight_decay=weight_decay)
    else:
        gfn_optimizer = torch.optim.Adam(param_groups, lr_policy)
    return gfn_optimizer





def get_finetuning_loss(mode, init_state, prior, gfn_model, log_reward, beta = 1.0, exploration_std=None, return_exp=False):
    if mode == 'rl':
        rl_loss, kl_loss, kl_div = fwd_rl(init_state, prior, gfn_model, log_reward, exploration_std)
        return rl_loss, kl_loss, kl_div
    elif mode == 'rtb':
        return fwd_rtb(init_state, prior, gfn_model, log_reward, exploration_std, beta = beta, return_exp=return_exp)

    return None

def get_gfn_backward_loss(mode, samples, gfn_model, log_reward, exploration_std=None):

    loss = bwd_mle(samples, gfn_model, log_reward, exploration_std)
    return loss


def get_exploration_std(iter, exploratory, exploration_factor=0.1, exploration_wd=False):
    if exploratory is False:
        return None
    if exploration_wd:
        if iter < 500:
            exploration_std = exploration_factor
        else:
            exploration_std = exploration_factor * max(0, 1. - iter / 4500.)
    else:
        exploration_std = exploration_factor
    expl = lambda x: exploration_std
    return expl



