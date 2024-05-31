from utils.data_loaders import CELEB_CLASSES, DATASETS
from utils.simple_io import *
import warnings
warnings.filterwarnings("ignore")


import argparse
import platform
import os
import torch
from distutils.util import strtobool

if platform.system() == "Windows":
    home_folder = ""  # Specify the home folder path for Windows, if different from the default
    system = 'win'
elif platform.system() == "Darwin":  # Darwin is the system name for macOS
    home_folder = os.path.expanduser("~")  # Home folder path for macOS
    system = 'mac'
else:
    # This will cover Linux and other Unix-like systems
    home_folder = os.path.expanduser("~")  # Home folder path for Linux/Unix
    system = 'linux'


def fetch_args(experiment_run=True, exp_prepend='exp'):
    parser = argparse.ArgumentParser(description='PyTorch Feature Combination Mode')

    parser.add_argument('-md', '--model', default='UNet', type=str, help="'UNet', 'ScoreNet' supported at the moment.")
    parser.add_argument('-ds', '--dataset', default='mnist', type=str, help="'mnist', 'cifar10' and 'UTKFace' supported at the moment.")
    parser.add_argument('-sf', '--save_folder', type=str, default="./../results", help='Path to save results to.')
    parser.add_argument('-pf', '--load_path', type=str, default="./models/pretrained", help="Folder to keep pretrained 'best' weights.")
    parser.add_argument('-dp', '--data_path', type=str, default=f"{home_folder}/data", help="Folder containing datasets.")
    parser.add_argument('-ptm', '--prior_training_mode', type=str, default=f"dpm", help="modality of training for 'train_prior' script. Gfn is gfn mle training, while anything else will do regular diffusion training.")
    parser.add_argument('-pm', '--pretrain_models', type=strtobool, default=False, help='Pretrain models.')
    parser.add_argument('-rp', '--replace', type=strtobool, default=True, help='Replace run logs.') # todo: replace true
    parser.add_argument('--baseline_reinforce',type=str, default='whitening', help='debugging for a proper backprop.')

    parser.add_argument('-fs', '--show_figures', type=strtobool, default=False, help='show plots.')
    parser.add_argument('-fo', '--save_figures', type=strtobool, default=True, help='save plots.')
    parser.add_argument('-sw', '--save_model_weights', type=strtobool, default=False, help='save model weights.')
    parser.add_argument('--plot_batch_size', default=30, type=int)

    parser.add_argument('-en', '--exp_name', type=str, default='', help='Experiment name.')

    parser.add_argument('--method', default='gfn', type=str, help='Method to use for training (reinforce, gfn, dp, lgd_mc, , etc.).')
    # REINFORCE Hiperparams
    parser.add_argument('--reward_weight', default=10, type=float, help='Reward weight for REINFORCE.')
    parser.add_argument('--kl_weight', default=0.01, type=float, help='KL weight for REINFORCE.')
    # cls guidance Hyperparams
    parser.add_argument('--particles', default=10, type=int, help='No. of particles for the MC-based approaches (e.g., lgd_mc)')
    parser.add_argument('--scale', default=1., type=float, help='No. of particles for the MC-based approaches (e.g., lgd_mc)')


    parser.add_argument('--channels', default=3, type=int, help='Image channels.')
    # Optimization options
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='Number of data loading workers (default: 4).')

    # GFN/Training Parameters
    parser.add_argument('--epochs', default=1000, type=int, metavar='N', help='Number of epochs to run')
    parser.add_argument('--algo', type=str, default='mle', help="Algorithm used for the gfn training objective.")
    parser.add_argument('-bs', '--batch_size', type=int, default=128, help="Training Batch Size.")
    parser.add_argument('-tbs', '--test_batch_size', default=16, type=int, help='Test batchsize.')
    parser.add_argument('--lr', '--learning_rate', default=1e-5, type=float, help='Initial learning rate.')
    parser.add_argument('--lr_logZ', default=1e-1, type=float, help='Learning rate for logZ.')
    parser.add_argument('--z_weight_decay', default=0, type=float, help='Weight decay for logZ.')

    parser.add_argument('--traj_length', default=100, type=int, help='Legth of trajectory.')
    parser.add_argument('--sampling_length', default=100, type=int, help='Legth of sampling traj. If sampling_length < traj_length then DDPM automatically kicks in.')
    parser.add_argument('--learning_cutoff', default=1e-1, type=float, help='Cut-off for allowed closeness of prior/posterior given inpferfect TB assumption.')
    parser.add_argument('--learn_var', default=False, type=strtobool, help='Whether to learn the variance. Mind the models will give a tuple with two outputs if flagged to True.')
    parser.add_argument('--loss_type', type=str, default='l2', help='Loss type to use for regular diffusion training.')
    parser.add_argument('-age', '--accumulate_gradient_every', default=1, type=int, help='Number of iterations to accumulate gradient for.')
    parser.add_argument('--detach_freq', default=0., type=float, help='Fraction of steps on which not to train')
    parser.add_argument('--detach_cut_off', default=1., type=float, help='Fraction of steps to keep from t=1 (full noise).')

    parser.add_argument('--back_and_forth', default=False, type=strtobool, help='Whether to train based on back and forth trajectories.')
    parser.add_argument('--bf_length', default=50, type=int, help='backward steps in the back and forth learning algoritm')
    parser.add_argument('--drift_only', default=False, type=strtobool, help='Whether to train only with drift model.')
    parser.add_argument('--drift_burn_in', default=1000, type=int, help='How many iterations to burn in the drift.')
    parser.add_argument('--mixed_precision', default=False, type=strtobool, help='Whether to train with mixed precision.')
    parser.add_argument('--checkpointing', default=True, type=strtobool, help='Uses checkpointing to save memory in exchange for compute.')

    parser.add_argument('--multi_class_index', default=0, type=int, help='Index of class type for multiclass datasets.')
    parser.add_argument('--finetune_class', default='0', type=str, help='Index of class to finetune.')
    parser.add_argument('--class_name', default='not_specified', type=str, help='Name of class to finetune.')
    parser.add_argument('--classifier_model', default='resnet', type=str, help='resnet or cnn.')
    parser.add_argument('--classifier_depth', default=18, type=int, help='resnet depth.')
    parser.add_argument('--classifier_pretrained', default=False, type=bool, help='loads and freezes weights from pretrained classifier on torch vision (if available).')

    parser.add_argument('--langevin', action='store_true', default=False)
    parser.add_argument('--lgv_clip', type=float, default=1e2)
    parser.add_argument('--lgv_clipping', action='store_true', default=False)
    parser.add_argument('--lgv_t_dim', default=64, type=int)
    parser.add_argument('--lgv_hidden_dim', default=256, type=int)
    parser.add_argument('--lgv_num_layers', default=3, type=int)
    parser.add_argument('--lgv_zero_init', action='store_true', default=False)

    # FID
    parser.add_argument('--compute_fid', default=False, type=strtobool, help='Compute FID.')
    parser.add_argument('--num_fid_samples', default=50000, type=int, help='Number of FID samples to use for fid computation.')

    # lora
    parser.add_argument('--lora', default=True, type=strtobool, help='low rank approximation training.')
    parser.add_argument('--rank', default=32, type=int, help='lora rank.')

    # hugging face
    parser.add_argument('--push_to_hf', default=False, type=strtobool)
    parser.add_argument('--push_to_wandb', default=True, type=strtobool)

    # wandb
    parser.add_argument('--notes', default="", type=str)

    # Miscs
    parser.add_argument('--seed', default=912, type=int, help='Manual seed.')

    args = parser.parse_args()
    state = {k: v for k, v in args._get_kwargs()}

    # --------- multi-gpu special ---------------
    args.use_cuda = torch.cuda.device_count() != 0
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.system = system

    if not args.use_cuda or 'win' in args.system or args.method in ['dp', 'lgd_mc']:
        args.workers = 0

    # if 'linux' in args.system and args.use_cuda:
    #     # logic for gpu partition on cluster
    #     if "posterior" in exp_prepend and torch.cuda.get_device_properties(0).total_memory / 1e9 < 70:
    #         print(f"* NOTE: detected GPU with {torch.cuda.get_device_properties(0).total_memory / 1e9: .2f}Gb of memory, "
    #               f"The batch size is going to be halfed from{args.batch_size} to {args.batch_size/2} to prevent memory overflow. "
    #               f"The effective batch size will be the same")
    #         args.batch_size = int(args.batch_size / 2)
    #         args.accumulate_gradient_every *= 2

    # --------- DATA SPECIFIC ARGS ---------------
    # -- fix naming
    args.supported_datasets = list(DATASETS.keys())  # todo add support for ['flower', 'celebA', 'AFHQ']
    dti = [i for i, dt in enumerate(args.supported_datasets) if args.dataset.lower() in dt.lower()]
    if len(dti) > 0:
        args.dataset = args.supported_datasets[dti[0]]
    else:
        raise NotImplementedError(f"the dataset {args.dataset} is not yet supported.")

    # -- impose data specific args
    args.splits = DATASETS[args.dataset]['splits']
    args.image_size = DATASETS[args.dataset]['image_size']
    args.noise_size = DATASETS[args.dataset]['noise_size']
    args.channels = DATASETS[args.dataset]['channels']
    args.x_tensor = DATASETS[args.dataset]['x_tensor']
    args.y_tensor = DATASETS[args.dataset]['y_tensor']
    args.classifier_model = DATASETS[args.dataset]['classifier_model']
    args.classifier_depth = DATASETS[args.dataset]['classifier_depth']
    args.num_classes = DATASETS[args.dataset]['num_classes']
    args.multi_class_index = DATASETS[args.dataset]['multi_class_index']
    args.block_out_channels = DATASETS[args.dataset]['block_out_channels']
    args.fid_inception_block = DATASETS[args.dataset]['fid_inception_block']

    if args.finetune_class[0].isdigit() and 'celeb' not in args.dataset.lower():
        args.class_name = args.finetune_class
        args.finetune_class = [int(args.finetune_class)]
    else:
        if 'mnist' in args.dataset.lower() and args.finetune_class == 'odd':
            args.class_name = 'Odd'
            args.finetune_class = [1, 3, 5, 7, 9]
        elif 'mnist' in args.dataset.lower() and args.finetune_class == 'even':
            args.class_name = 'Even'
            args.finetune_class = [0, 2, 4, 6, 8]
        elif 'face' in args.dataset.lower() and args.finetune_class == 'young':
            args.class_name = 'Young'
            args.finetune_class = list(range(30))
        elif 'face' in args.dataset.lower() and args.finetune_class == 'adult':
            args.class_name = 'Adult'
            args.finetune_class = list(range(30, 69))
        elif 'face' in args.dataset.lower() and args.finetune_class == 'elder':
            args.class_name = 'Elder'
            args.finetune_class = list(range(70, 117))
        elif 'celeb' in args.dataset.lower():
            if args.finetune_class not in CELEB_CLASSES:
                if args.finetune_class == '0':   # probably just left the default -- give a default valid class
                    args.class_name = args.y_tensor.capitalize()
                else:
                    print(f"provide a valid value '{args.multi_class_index}' for '{args.dataset}' dataset. Allowed values are:")
                    for value in CELEB_CLASSES: print(f"  {value}")
                    raise ValueError(f"Invalid 'finetune_class'")
            else:
                args.class_name = args.finetune_class.capitalize()
                args.y_tensor = args.finetune_class

            args.multi_class_index = args.y_tensor
            args.finetune_class = [1]
        else:
            raise ValueError(f"class '{args.finetune_class}' for dataset '{args.dataset}' is not supported.")

    # --------- FINETUNING SPECIAL ARGS---------------
    # ---------  gradient accumulation ---------
    if args.batch_size * args.accumulate_gradient_every < 32:
        args.accumulate_gradient_every = int(32/args.batch_size)
        print(f"*forcing '--accumulate_gradient_every' to {args.accumulate_gradient_every}"
              f"\n*effective batch size: {args.batch_size * args.accumulate_gradient_every}")

    # ------ FOLDER CREATION AND SETTINGS -----------------
    # create exp_name if it wasn't given, then choose and create a folder to save exps based on parameters
    if len(args.exp_name) == 0:
        args.exp_name = f"{args.dataset}_{args.model}"
    args.exp_name = f"{exp_prepend}_" + args.exp_name

    if experiment_run:
        num = 0
        save_folder_name = f"{args.save_folder}/{args.exp_name}_{num}/"
        if not args.replace:
            num = 0
            while folder_exists(save_folder_name):
                num += 1
                save_folder_name = f"{args.save_folder}/{args.exp_name}_{num}/"
        args.save_folder = save_folder_name
        # args.load_path += f"/{args.exp_name}_{num}"

    folder_create(args.save_folder, exist_ok=True)
    folder_create(args.load_path, exist_ok=True)
    folder_create(f"{args.save_folder}/samples/", exist_ok=True)

    os.environ["DEEPLAKE_DOWNLOAD_PATH"] = args.data_path + '/'

    return args, state
