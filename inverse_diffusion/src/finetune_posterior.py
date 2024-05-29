from models.classifiers import CNN, ResNet
from models.langevin import LangevinModel
from utils.args import fetch_args
from utils.gfn_diffusion import load_PPDGFN_from_diffusers,load_PPDGFN_from_diffusers, GFNFinetuneTrainer, ReinforceFinetuneTrainer
from utils.pytorch_utils import seed_experiment, train_classifier, print_gpu_memory
from utils.gfn_diffusion import load_PPDGFN_from_diffusers, GFNFinetuneTrainer, \
    load_DDPM_from_diffusers, ReinforceFinetuneTrainer, BaselineGuidance
from utils.pytorch_utils import seed_experiment, train_classifier, get_train_classifier
from utils.simple_io import *

import torch as T
import numpy as np
import torch.nn as nn


# get arguments for the run
args, state = fetch_args(exp_prepend='train_posterior')
print(f"Running experiment on '{args.device}'")

# -------- OVERRIDE ARGUMENTS (if you have to) --------
# args.dataset = 'mnist'
# args.t_scale = 1
# args.batch_size = 128
# args.lr = 1e-5
# args.lr_logZ = 5e-2
args.learn_var = True

logtwopi = np.log(2 * 3.14159265358979)

seed_experiment(args.seed)

# ------------  Load pretrained PosteriorPriorGFN from pretrained diffusion model  ---------

if args.method in['gfn', "reinforce"]:
    sampler = load_PPDGFN_from_diffusers(args)  # note, this function may change some of the args in place
else:
    # todo baseline case
    sampler = load_DDPM_from_diffusers(args)
print(args)

# -------- Train a reward model (if not already pretrained) ---------
classifier = get_train_classifier(args, scheduler=sampler.get_scheduler())

# ---- add langevin if gfn and user specified (here because we need a classifier first) ----
if args.method == 'gfn' and args.langevin:

    log_reward = classifier if args.langevin else lambda x: T.zeros((x.shape[0],), device=args.device)

    problem_dim = int(args.channels * (args.image_size ** 2))
    lgv_model = LangevinModel(problem_dim, args.lgv_t_dim, args.lgv_hidden_dim, 1,
                              args.lgv_num_layers, args.lgv_zero_init)

    sampler.add_langevin(log_reward=log_reward,
                         lgv_model=lgv_model,
                         lgv_clip=args.lgv_clip,
                         lgv_clipping=args.lgv_clipping)
else:
    sampler.add_classifier(classifier=classifier)  # add classifier for classifier guidance

# ------------  Set learning params  ---------
if args.method == 'gfn':
    params = [param for param in sampler.posterior_node.get_parameters() if param.requires_grad]
    opt = T.optim.Adam([{'params': params,
                         'lr': args.lr},
                        {'params': [sampler.logZ],
                         'lr': args.lr_logZ,
                         'weight_decay':args.z_weight_decay}])

    Trainer = GFNFinetuneTrainer
else:
    opt = None
    Trainer = BaselineGuidance


# ------------  Train posterior  ---------
if args.method in ['gfn', 'dp', 'lgd_mc']:
    trainer = Trainer(
        sampler=sampler,
        classifier=classifier,
        optimizer=opt,
        finetune_class=args.finetune_class,
        save_folder=args.save_folder,
        config=args
    )
elif args.method == 'reinforce':
    trainer = ReinforceFinetuneTrainer(
        sampler=sampler,
        classifier=classifier,
        optimizer=opt,
        finetune_class=args.finetune_class,
        save_folder=args.save_folder,
        config=args
    )
else:
    raise ValueError(f"Method '{args.method}' not recognized.")

trainer.run(
    finetune_class=args.finetune_class,
    epochs=args.epochs,
    back_and_forth=args.back_and_forth,
)
