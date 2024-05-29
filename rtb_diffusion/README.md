# Relative Trajectory Balance for Posterior Diffusion Sampler

This repository builds upon the original source code developed in:

[On Diffusion Models for Amortized Inference: Benchmarking and Improving Stochastic Control and Sampling](https://arxiv.org/abs/2402.05098)

## Overview

This repository provides a posterior diffusion sampler where the prior model is a 2D Gaussian mixture model with 25 modes (25gmm), and the posterior is a Gaussian mixture model with 9 modes, with the modes containing reweighted densities. This setup can be extended to various prior-reward combinations by incorporating custom energy functions into the `energy/` directory.

## Dependencies

Ensure you have the following libraries installed:

- `torch`
- `einops`
- `pot`
- `matplotlib`

## Getting Started

#### RTB finetuning:

To fine-tune the posterior using Relative Trajectory Balance (RTB), run:

```
python finetune_posterior.py --method rtb --name "save_rtb_finetune"
```

#### RL finetuning:

For reinforcement learning (RL) based fine-tuning, run:


```
python finetune_posterior.py --method rl --kl_weight 0.01 --name "save_rl_finetune"
```

#### Classifier Guidance:

To use classifier guidance for fine-tuning the posterior, run:


```
python classifer_guidance_posterior.py
```

#### prior pretraning ####

To pretrain the prior model, run:



```
python pretrain_prior.py 
```
