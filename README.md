# Amortizing intractable inference in diffusion models\\for vision, language, and control

This repository contains code for learning posteriors with diffusion priors and arbitrary constraints using _relative trajectory balance_ (RTB) introduced in

**Amortizing intractable inference in diffusion models for vision, language, and control**

Siddarth Venkatraman*, Moksh Jain*, Luca Scimeca*, Minsu Kim*, Marcin Sendera*, Mohsin Hasan, Luke Rowe, Sarthak Mittal, Pablo Lemos, Emmanuel Bengio, Alexandre Adam, Jarrid Rector-Brooks, Yoshua Bengio, Glen Berseth, Nikolay Malkin

[arXiv](https://arxiv.org/abs/)


The code and documentation for running the experiments is structured in subdirectories corresponding to each experiment.

- Class-conditional posterior sampling from unconditional diffusion priors (§3.1) in `inverse_diffusion/`
- Fine-tuning a text-to-image diffusion model (§3.2) in `dpok_finetuning/`
- Text infilling with discrete diffusion language models (§3.3) in `diffusion_lm/`
- KL-constrained policy search in offline reinforcement learning (§3.4) in `offline_RL/`
- Learning posterior of diffusion model sampling a mixture of 25 Gaussians (§1) in `rtb_diffusion/`
