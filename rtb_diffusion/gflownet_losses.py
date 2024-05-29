import torch
from torch.distributions import Normal


def fwd_rl(initial_state, prior, gfn, log_reward_fn, exploration_std=None, return_exp = False):

    states, log_p_posterior, log_pbs, log_fs = gfn.get_trajectory_fwd(initial_state, exploration_std, log_reward_fn)
    log_p_prior = prior.get_trajectory_fwd_off(states.detach(), log_reward_fn).detach()

    
    with torch.no_grad():
        log_r = log_reward_fn(states[:, -1]).detach()
    reward = log_r.exp()
    adv = reward - reward.mean()

    kl_loss = (log_p_posterior.sum(-1) - log_p_prior.sum(-1))**2
    kl_div = (log_p_posterior.sum(-1) - log_p_prior.sum(-1)).mean()
    
    reinforce_loss = -adv * log_p_posterior.sum(-1)

    return reinforce_loss, kl_loss, kl_div

def fwd_rtb(initial_state, prior, gfn, log_reward_fn, exploration_std=None, return_exp = False, beta = 1.0):
    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_fwd(initial_state, exploration_std, log_reward_fn)
    log_p_prior = prior.get_trajectory_fwd_off(states.detach(), log_reward_fn).detach()
    with torch.no_grad():
        log_r = log_reward_fn(states[:, -1]).detach()
    kl_div = (log_pfs.sum(-1) - log_p_prior.sum(-1)).mean()

    loss = 0.5 * ((log_pfs.sum(-1) + log_fs[:, 0] - log_p_prior.sum(-1) - beta * log_r) ** 2)
    if return_exp:
        return loss.mean(), states, log_pfs, log_pbs, log_r, kl_div
    return loss.mean(), kl_div


def bwd_mle(samples, gfn, log_reward_fn, exploration_std=None):
    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_bwd(samples, exploration_std, log_reward_fn)
    loss = -log_pfs.sum(-1)
    return loss.mean()


