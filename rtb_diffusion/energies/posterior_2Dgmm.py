import matplotlib.pyplot as plt

import torch
import torch.distributions as D
from torch.distributions.mixture_same_family import MixtureSameFamily

from .base_set import BaseSet

from . import twenty_five_gmm


class Posterior2DGaussianMixture(BaseSet):
    def __init__(self, device, scale=0.5477222, dim=2):
        super().__init__()
        self.device = device
        self.data = torch.tensor([0.0])
        self.data_ndim = 2

        self.prior = twenty_five_gmm.TwentyFiveGaussianMixture(device, dim=2)

        mean_ls = [
            [-10., -5.], [-5., -10.], [-5., 0.],
            [10., -5.], [0., 0.], [0., 5.],
            [5., -5.], [5., 0.], [5., 10.],
        ]

        nmode = len(mean_ls)
        mean = torch.stack([torch.tensor(xy) for xy in mean_ls])
        comp = D.Independent(D.Normal(mean.to(self.device), torch.ones_like(mean).to(self.device) * scale), 1)
        
        probs = torch.Tensor([4, 10, 4, 5, 10, 5, 4, 15, 4]).to(self.device)
        probs = probs / probs.sum()
        mix = D.Categorical(probs=probs)
        self.gmm = MixtureSameFamily(mix, comp)
        self.data_ndim = dim

    def gt_logz(self):
        return 0.

    def energy(self, x):
        en =  -(self.gmm.log_prob(x).flatten()) - self.prior.energy(x) #- self.prior.gmm.log_prob(x).flatten())
        #print("x shape: ", x.shape)
        #print("en shape: ", en.shape)
        #exit()
        return en 
    
    def sample(self, batch_size):
        return self.gmm.sample((batch_size,))

    def viz_pdf(self, fsave="ou-density.png"):
        x = torch.linspace(-15, 15, 100).to(self.device)
        y = torch.linspace(-15, 15, 100).to(self.device)
        X, Y = torch.meshgrid(x, y)
        x = torch.stack([X.flatten(), Y.flatten()], dim=1)  # ?

        density = self.unnorm_pdf(x)
        return x, density

    def __getitem__(self, idx):
        del idx
        return self.data[0]
