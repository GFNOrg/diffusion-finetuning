import seaborn as sns
import torch
import matplotlib.pyplot as plt
import itertools
import numpy as np
from einops import rearrange


def get_figure(bounds=(-10., 10.)):
    fig, ax = plt.subplots(1, figsize=(16, 16))
    ax.axis('off')
    ax.set_autoscale_on(False)
    ax.set_xlim([bounds[0], bounds[1]])
    ax.set_ylim([bounds[0], bounds[1]])
    return fig, ax


def plot_contours(log_prob, ax=None, bounds=(-10., 10.), grid_width_n_points=200, n_contour_levels=50,
                  log_prob_min=-1000., device=torch.device('cuda')):
    """Plot contours of a log_prob_func that is defined on 2D"""
    if ax is None:
        fig, ax = plt.subplots(1)
    x_points_dim1 = torch.linspace(bounds[0], bounds[1], grid_width_n_points)
    x_points_dim2 = x_points_dim1
    x_points = torch.tensor(list(itertools.product(x_points_dim1, x_points_dim2)))
    log_p_x = log_prob(x_points.to(device)).detach().cpu()
    log_p_x = torch.clamp_min(log_p_x, log_prob_min)
    log_p_x = log_p_x.reshape((grid_width_n_points, grid_width_n_points))
    x_points_dim1 = x_points[:, 0].reshape((grid_width_n_points, grid_width_n_points)).numpy()
    x_points_dim2 = x_points[:, 1].reshape((grid_width_n_points, grid_width_n_points)).numpy()
    if n_contour_levels:
        ax.contour(x_points_dim1, x_points_dim2, log_p_x, levels=n_contour_levels)
    else:
        ax.contour(x_points_dim1, x_points_dim2, log_p_x)


def plot_samples(samples, ax=None, bounds=(-10., 10.), alpha=0.5):
    if ax is None:
        fig, ax = plt.subplots(1)
    samples = torch.clamp(samples, bounds[0], bounds[1])
    samples = samples.cpu().detach()
    ax.scatter(samples[:, 0], samples[:, 1], alpha=alpha, marker="o", s=10)


def plot_kde(samples, ax=None, bounds=(-10., 10.)):
    if ax is None:
        fig, ax = plt.subplots(1)
    samples = samples.cpu().detach()
    sns.kdeplot(x=samples[:, 0], y=samples[:, 1], cmap="Blues", fill=True, ax=ax, clip=bounds)


def viz_many_well(mw_energy, samples=None, num_samples=5000):
    if samples is None:
        samples = mw_energy.sample(num_samples)

    x13 = samples[:, 0:3:2].detach().cpu()
    fig_samples_x13, ax_samples_x13 = viz_sample2d(x13, "samples", f"distx13.png", lim=3)
    fig_kde_x13, ax_kde_x13 = viz_kde2d(x13, "kde", f"kdex13.png", lim=3)

    lim = 3
    alpha = 0.8
    n_contour_levels = 20

    def logp_func(x_2d):
        x = torch.zeros((x_2d.shape[0], mw_energy.data_ndim)).to(mw_energy.device)
        x[:, 0] = x_2d[:, 0]
        x[:, 2] = x_2d[:, 1]
        return -mw_energy.energy(x).detach().cpu()

    x13 = samples[:, 0:3:2]
    contour_img_path = f"contourx13.png"
    fig_contour_x13, ax_contour_x13 = viz_contour_sample2d(x13, contour_img_path, logp_func, lim=lim, alpha=alpha,
                                                           n_contour_levels=n_contour_levels)

    x23 = samples[:, 1:3].detach().cpu()
    fig_samples_x23, ax_samples_x23 = viz_sample2d(x23, "samples", f"distx23.png", lim=3)
    fig_kde_x23, ax_kde_x23 = viz_kde2d(x23, "kde", f"kdex23.png", lim=3)

    def logp_func(x_2d):
        x = torch.zeros((x_2d.shape[0], mw_energy.data_ndim)).to(mw_energy.device)
        x[:, 1] = x_2d[:, 0]
        x[:, 2] = x_2d[:, 1]
        return -mw_energy.energy(x).detach().cpu()

    x23 = samples[:, 1:3]
    contour_img_path2 = f"contourx23.png"
    fig_contour_x23, ax_contour_x23 = viz_contour_sample2d(x23, contour_img_path2, logp_func, lim=lim, alpha=alpha,
                                                           n_contour_levels=n_contour_levels)

    return fig_samples_x13, ax_samples_x13, fig_kde_x13, ax_kde_x13, fig_contour_x13, ax_contour_x13, fig_samples_x23, ax_samples_x23, fig_kde_x23, ax_kde_x23, fig_contour_x23, ax_contour_x23


def traj_plot1d(traj_len, samples, xlabel, ylabel, title="", fsave="img.png"):
    samples = rearrange(samples, "t b d -> b t d").cpu()
    inds = np.linspace(0, samples.shape[1], traj_len, endpoint=False, dtype=int)
    samples = samples[:, inds]
    plt.figure()
    for i, sample in enumerate(samples):
        plt.plot(np.arange(traj_len), sample.flatten(), marker="x", label=f"sample {i}")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(fsave)
    plt.close()


########### 2D plot
def viz_sample2d(points, title, fsave, lim=7.0, sample_num=50000):
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    if title is not None:
        ax.set_title(title)
    ax.plot(
        points[:sample_num, 0],
        points[:sample_num, 1],
        linewidth=0,
        marker=".",
        markersize=1,
    )
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    return fig, ax


def viz_kde2d(points, title, fname, lim=7.0, sample_num=2000):
    fig, ax = plt.subplots(1, 1, figsize=(7, 7), dpi=200)
    if title is not None:
        ax.set_title(title)
    sns.kdeplot(
        x=points[:sample_num, 0], y=points[:sample_num, 1],
        cmap="coolwarm", fill=True, ax=ax
    )
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    return fig, ax


def viz_coutour_with_ax(ax, log_prob_func, lim=3.0, n_contour_levels=None):
    grid_width_n_points = 100
    log_prob_min = -1000.0
    x_points_dim1 = torch.linspace(-lim, lim, grid_width_n_points)
    x_points_dim2 = x_points_dim1
    x_points = torch.tensor(list(itertools.product(x_points_dim1, x_points_dim2)))
    log_p_x = log_prob_func(x_points).detach().cpu()
    log_p_x = torch.clamp_min(log_p_x, log_prob_min)
    log_p_x = log_p_x.reshape((grid_width_n_points, grid_width_n_points))
    x_points_dim1 = x_points[:, 0].reshape((grid_width_n_points, grid_width_n_points)).numpy()
    x_points_dim2 = x_points[:, 1].reshape((grid_width_n_points, grid_width_n_points)).numpy()
    if n_contour_levels:
        ax.contour(x_points_dim1, x_points_dim2, log_p_x, levels=n_contour_levels)
    else:
        ax.contour(x_points_dim1, x_points_dim2, log_p_x)


def viz_contour_sample2d(points, fname, log_prob_func,
                         lim=3.0, alpha=0.7, n_contour_levels=None):
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    viz_coutour_with_ax(ax, log_prob_func, lim=lim, n_contour_levels=n_contour_levels)

    samples = torch.clamp(points, -lim, lim)
    samples = samples.cpu().detach()
    ax.plot(samples[:, 0], samples[:, 1],
            linewidth=0, marker=".", markersize=1.5, alpha=alpha)

    return fig, ax
