

import torch as T
from accelerate import Accelerator
from utils.fid_evaluation import SCOREEvaluation
from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel, load_peft_weights, set_peft_model_state_dict
from huggingface_hub import hf_hub_download

from models.denoisers import ScoreNet
from models.samplers import PosteriorPriorDGFN, PosteriorPriorDGFN, PosteriorPriorBaselineSampler
from utils.data_loaders import cycle, get_dataset
from utils.diffusers.schedulers.scheduling_ddim_gfn import DDIMGFNSchedulerOutput, DDIMGFNScheduler
from utils.diffusion import get_diffuser_ddpm, get_stable_diffuser
from utils.pytorch_utils import Logger
from utils.simple_io import get_filenames, load_dict_from_file, DictObj
from utils.visualization import plot_samples, compare_samples
from torchvision import transforms
from utils.simple_io import *

import torch
import torch.nn as nn
import copy
import os
import wandb


def load_PPDGFN_from_diffusion(args):
    """posterior policy diffusion gfn"""
    # Use pretrained model for both prior and (to burn in the) posterior policy
    assert args.model.lower() in ["scorenet", "unet"], f"Model '{args.model}' not currently supported."

    # check pretrained models
    pretrained_policy_models = get_filenames(args.load_path,
                                             starts_with=f'{args.dataset}_fwd_policy_{args.model.lower()}',
                                             ends_with='.pth')
    assert len(pretrained_policy_models) > 0, f"A prior denoising model must be trained first! No model found at {args.load_path}"

    # load model with the highest epoch
    diff_model = T.load(f'{args.load_path}/{pretrained_policy_models[0]}', map_location=T.device('cpu'))
    print(f"\nLoaded prior: {pretrained_policy_models[0]}, "
          f"Traj Length: {diff_model.num_timesteps}, "
          f"Sampling Length: {diff_model.sampling_timesteps}\n")

    tfs = None
    if diff_model.sampling_timesteps != args.sampling_length:
        print(f"#### NOTE: the pretrained model's sampling length is {diff_model.sampling_timesteps}, not {args.sampling_length}. Samples might not look as intended.")
    if diff_model.num_timesteps != args.traj_length:
        print(f"#### NOTE: the pretrained model's trajectory length is {diff_model.num_timesteps}, not {args.traj_length}. The length was reset to {diff_model.num_timesteps} for consistency.")
        args.traj_length = diff_model.num_timesteps
    if args.image_size != diff_model.image_size or args.channels != diff_model.channels:
        print(f"#### NOTE: the pretrained model's image size is {(diff_model.channels, diff_model.image_size, diff_model.image_size)}, not {(args.channels, args.image_size, args.image_size)}. The image size was reset to the pretrain model's.")
        tfs = transforms.Resize(args.image_size)
        args.image_size = diff_model.image_size
        args.channels = diff_model.channels

    x_dim = (args.channels, args.image_size, args.image_size)
    prior_policy_model = diff_model.model

    # ------------  Load pretrained prior model  ---------
    if args.drift_only:

        posterior_policy_model = prior_policy_model  # -> will be frozen
        drift_model = ScoreNet(
            input_size=args.image_size,
            channels=args.channels,
            # learn_var=args.learn_var
        ).to(args.device)
    else:
        posterior_policy_model = copy.deepcopy(prior_policy_model)  # hard copy -> for training
        drift_model = None

    diff_gfn = PosteriorPriorDGFN(dim=x_dim,
                                  traj_length=args.traj_length,
                                  sampling_length=args.sampling_length,
                                  # NOTE: is sampling_length < traj_length then DDPM sampling kicks in
                                  prior_policy_model=prior_policy_model,
                                  posterior_policy_model=posterior_policy_model,
                                  drift_model=drift_model,
                                  beta_schedule='linear',
                                  mixed_precision=args.mixed_precision,
                                  use_cuda=args.use_cuda,
                                  transforms=tfs)

    if args.mixed_precision and 'cuda' in args.device:
        diff_gfn.half()

    diff_gfn.posterior_node.logvarrange = 0
    diff_gfn.prior_node.logvarrange = 0

    return diff_gfn


def load_PPDGFN_from_diffusers(args):
    """posterior policy diffusion gfn from hf diffusers """

    # get pretrained prior
    ddpm, params = get_diffuser_ddpm(args.dataset, args.device)

    tfs = None
    if params.sampling_length != args.sampling_length:
        print(f"#### NOTE: the pretrained model's sampling length is {len(ddpm.scheduler.timesteps)}, not {args.sampling_length}. Samples might not look as intended.")
    ddpm.scheduler.set_timesteps(args.sampling_length)

    if params.traj_length != args.traj_length:
        print(f"#### NOTE: the pretrained model's trajectory length is {ddpm.scheduler.config.num_train_timesteps}, not {args.traj_length}. The length was reset to {ddpm.scheduler.config.num_train_timesteps} for consistency.")
        args.traj_length = params.traj_length

    if args.image_size != params.image_size or args.channels != params.channels:
        print(f"#### NOTE: the pretrained model's image size is {(params.channels, params.image_size, params.image_size)}, not {(args.channels, args.image_size, args.image_size)}. The image size was reset to the pretrain model's.")
        from torchvision import transforms
        tfs = transforms.Resize(args.image_size)
        args.image_size = params.image_size
        args.channels = params.channels

    if args.noise_size != params.noise_size:
        print(f"#### NOTE: the pretrained model's noise is {(params.channels, params.noise_size, params.noise_size)}, not {(args.channels, args.noise_size, args.noise_size)}. The noise size was reset to the pretrain model's.")
        args.noise_size = params.noise_size

    x_dim = (args.channels, args.image_size, args.image_size)
    prior_policy_model = ddpm

    # ------------  Load pretrained prior model  ---------
    if args.drift_only:
        posterior_policy_model = prior_policy_model  # -> will be frozen
        drift_model = ScoreNet(
            input_size=args.image_size,
            channels=args.channels,
            # learn_var=args.learn_var
        ).to(args.device)
    else:
        posterior_policy_model = copy.deepcopy(prior_policy_model)  # hard copy -> for training
        drift_model = None

        if args.lora:
            unet_lora_config = LoraConfig(
                r=args.rank,
                lora_alpha=args.rank,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
            posterior_policy_model.unet = get_peft_model(posterior_policy_model.unet, unet_lora_config)

    diff_gfn = PosteriorPriorDGFN(dim=x_dim,
                                  traj_length=args.traj_length,
                                  sampling_length=args.sampling_length,
                                  # NOTE: is sampling_length < traj_length then DDPM sampling kicks in
                                  prior_policy_model=prior_policy_model,
                                  posterior_policy_model=posterior_policy_model,
                                  drift_model=drift_model,
                                  mixed_precision=ddpm.unet.dtype == torch.float16,
                                  use_cuda=args.use_cuda,
                                  detach_cut_off=args.detach_cut_off,
                                  transforms=tfs,
                                  lora=args.lora,
                                  push_to_hf=args.push_to_hf,
                                  exp_name=args.exp_name,
                                  checkpointing=args.checkpointing)

    if args.mixed_precision and 'cuda' in args.device:
        diff_gfn.half()

    diff_gfn.posterior_node.logvarrange = 0
    diff_gfn.prior_node.logvarrange = 0

    return diff_gfn


def load_DDPM_from_diffusers(args):
    """stable diffusion gfn from hf diffusers"""

    # get pretrained prior
    ddpm, params = get_diffuser_ddpm(args.dataset, args.device)

    tfs = None
    if params.sampling_length != args.sampling_length:
        print(f"#### NOTE: the pretrained model's sampling length is {len(ddpm.scheduler.timesteps)}, not {args.sampling_length}. Samples might not look as intended.")
    ddpm.scheduler.set_timesteps(args.sampling_length)

    if params.traj_length != args.traj_length:
        print(f"#### NOTE: the pretrained model's trajectory length is {ddpm.scheduler.config.num_train_timesteps}, not {args.traj_length}. The length was reset to {ddpm.scheduler.config.num_train_timesteps} for consistency.")
        args.traj_length = params.traj_length

    if args.image_size != params.image_size or args.channels != params.channels:
        print(f"#### NOTE: the pretrained model's image size is {(params.channels, params.image_size, params.image_size)}, not {(args.channels, args.image_size, args.image_size)}. The image size was reset to the pretrain model's.")
        from torchvision import transforms
        tfs = transforms.Resize(args.image_size)
        args.image_size = params.image_size
        args.channels = params.channels

    if args.noise_size != params.noise_size:
        print(f"#### NOTE: the pretrained model's noise is {(params.channels, params.noise_size, params.noise_size)}, not {(args.channels, args.noise_size, args.noise_size)}. The noise size was reset to the pretrain model's.")
        args.noise_size = params.noise_size

    x_dim = (args.channels, args.image_size, args.image_size)
    policy_model = ddpm.unet
    policy_model.eval()
    scheduler = ddpm.scheduler

    sampler = PosteriorPriorBaselineSampler(
        dim=x_dim,
        y_dim=args.num_classes,
        traj_length=args.traj_length,
        sampling_length=args.sampling_length,
        policy_model=policy_model,
        scheduler=scheduler,
        mixed_precision=ddpm.unet.dtype == torch.float16,
        use_cuda=args.use_cuda,
        transforms=tfs,
        lora=args.lora,
        checkpointing=args.checkpointing,
        push_to_hf=args.push_to_hf,
        exp_name=args.exp_name,
        finetune_class=args.finetune_class,
        scale=args.scale,
        mc=True if args.method == 'lgd_mc' else False,
        particles=args.particles
    )

    if args.mixed_precision and 'cuda' in args.device:
        sampler.half()

    return sampler


def load_adapted_model(dataset, model_id, device='cpu'):
    # load prior
    ddpm, params = get_diffuser_ddpm(dataset, device)

    # attach lora posterior
    config = PeftConfig.from_pretrained(model_id)
    pft_model = PeftModel(ddpm.unet, config)
    lora_weights = load_peft_weights(model_id)
    set_peft_model_state_dict(pft_model, lora_weights)
    return ddpm


class FinetuneTrainer:

    """general class to achieve finetuning - connects to wandb (save run logs) and hugging face (push repo for easy loading later)"""
    def __init__(
            self,
            sampler,
            classifier,
            config,
            finetune_class,
            save_folder,
            optimizer
    ):

        wandb_key = os.getenv('WANDB_API_KEY', None)
        if wandb_key is None:
            print("NOTE: WANDB_API_KEY has not been set in the environment. Wandb tracking is off.")
            self.push_to_wandb = False
        else:
            self.push_to_wandb = True
            wandb.login(key=wandb_key)

        self.sampler = sampler
        self.opt = optimizer
        self.config = config

        self.classifier = classifier

        self.save_folder = save_folder

        # ------------------------------------------------------------------------------------------------
        self.accelerator = Accelerator(
            split_batches=True,
            mixed_precision='fp16' if config.mixed_precision else 'no',
            log_with="wandb" if self.push_to_wandb else "tensorboard",
            gradient_accumulation_steps=config.accumulate_gradient_every,
            cpu=not config.use_cuda,
            project_dir=config.save_folder
        )

        # WANDB & HF

        if self.accelerator.is_main_process:
            wandb.init(
                project="_".join(self.config.exp_name.split("_")[:-1]).split('-')[0] if '_' in self.config.exp_name else self.config.exp_name,
                dir=self.config.save_folder,
                resume=True,
                mode='online' if self.config.push_to_wandb else "offline",
                config={k: str(val) if isinstance(val, (list, tuple)) else val for k, val in self.config.__dict__.items()},
                notes=self.config.notes,
                name=self.config.exp_name.split('_')[-1]
            )
            self.checkpoint_dir = f"{self.config.save_folder}checkpoints/"
            self.checkpoint_file = self.checkpoint_dir + "checkpoint.tar"
            folder_create(self.checkpoint_dir, exist_ok=True)

            # -------------------------------------------------------------------------------------------------
            # custom logger for easy tracking and plots later
            self.logger = Logger(config)
            self.logger.save_args()

        with self.accelerator.main_process_first():

            # --------  ----  Load relevant dataset  ---------
            if config.back_and_forth or config.compute_fid:
                self.data_dict = get_dataset(
                    dataset=config.dataset,
                    batch_size=config.batch_size,
                    data_path=config.data_path,
                    channels=config.channels,
                    image_size=config.image_size,
                    x_tensor=config.x_tensor,
                    y_tensor=config.y_tensor,
                    splits=config.splits,
                    multi_class_index=config.multi_class_index,
                    workers=config.workers,
                    filter_data=config.compute_fid,
                    # returns an additional sliced view of the data by class, for true samples,
                    filter_labels=finetune_class
                )
                self.sampler.set_loader(self.accelerator.prepare(self.data_dict['train_loader']))

        if config.compute_fid:
            # ------- FID -------------------
            true_class_dataset = cycle(self.accelerator.prepare(self.data_dict[f'train_class_view']))
            num_fid_samples = min(len(self.data_dict[f'train_class_view'].dataset), self.config.num_fid_samples)

            if num_fid_samples < 50000:
                print(f"WARNING: computing FID score with {num_fid_samples} samples. At least 50000 samples are suggested.")

            if 'test_class_view' in self.data_dict.keys():
                true_class_dataset_test = cycle(self.accelerator.prepare(self.data_dict[f'test_class_view']))
                if len(self.data_dict[f'test_class_view'].dataset) < num_fid_samples:
                    print(f"WARNING: number of elements in true class for TEST dataset "
                          f"is {len(self.data_dict[f'test_class_view'].dataset)} < {num_fid_samples}")
            else:
                true_class_dataset_test = None

            self.fid_scorer = SCOREEvaluation(
                batch_size=config.batch_size,
                dl=true_class_dataset,
                dl_test=true_class_dataset_test,
                sampler=sampler,
                channels=config.channels,
                accelerator=self.accelerator,
                stats_dir=self.config.save_folder,
                device=config.device,
                num_fid_samples=num_fid_samples,
                normalize_input=False,
                inception_block_idx=config.fid_inception_block
            )

        self.x_dim = (config.image_size, config.image_size, config.channels)

        self.sampler, self.opt = self.accelerator.prepare(self.sampler, self.opt)

    def run(self, finetune_class, epochs=5000, **sampler_kwargs):

        it = 0

        self.resume()

        assert self.config.accumulate_gradient_every > 0, "must set 'accumulate_gradient_every' > 0"

        # ------------  Train posterior  ---------
        while it < epochs:

            self.sampler.train()
            for si in range(self.config.accumulate_gradient_every):
                print(f"it {it} [{si + 1}/{self.config.accumulate_gradient_every}] : ".endswith(""))

                # do a training step (sample posterior, score it and backprop)
                loss, results_dict = self.sampler_step(finetune_class, **sampler_kwargs)
                self.accelerator.backward(loss)

            self.opt.step()
            self.opt.zero_grad()

            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:

                if self.config.compute_fid and it % 100 == 0:
                    scores = self.fid_scorer.fid_score()
                    for key in scores.keys():
                        results_dict[key] = scores[key]

                self.logger.log(results_dict)  # log results locally

                if it % 10 == 0:
                    self.logger.print(it)  # print progress to terminal
                    self.logger.save()  # save logs file locally

                    self.sampler.save(
                        folder=self.checkpoint_dir,
                        opt=self.opt,
                        push_to_hf=self.config.lora and it % 1000 == 0,
                        it=it
                    )

                img_filename = None
                if it % 100 == 0:
                    img_filename = self.generate_plots(finetune_class, it=it)

                # save
                log_data = {k: val.cpu().mean().item() if isinstance(val, torch.Tensor) else val for k, val in results_dict.items()}
                if img_filename is not None and self.config.save_figures:
                    image = wandb.Image(img_filename, caption=f"it: {it}")
                    log_data['samples'] = image

                wandb.log(data=log_data, step=it)  # log results in wandb

                if it % 1000 == 0 and not self.config.lora:
                    print(
                        f"#################\n"
                        f"DIVERGENCE {torch.FloatTensor([(param1 - param2).abs().sum() for param1, param2 in zip(self.sampler.prior_node.get_parameters(), self.sampler.posterior_node.get_parameters()) if param2.requires_grad]).sum()}\n"
                        f"#################\n"
                    )

            it += 1

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.sampler.save(
                folder=self.checkpoint_dir,
                opt=self.opt,
                push_to_hf=True,
                it=it
            )
        self.accelerator.end_training()
        print("training ended")

    def resume(self):
        raise NotImplementedError()

    def sampler_step(self, *args, **kwargs):
        raise NotImplementedError()

    def generate_plots(self, *args, **kwargs):
        raise NotImplementedError()


class GFNFinetuneTrainer(FinetuneTrainer):

    def __init__(
            self,
            sampler,
            classifier,
            config,
            finetune_class,
            save_folder,
            optimizer
    ):
        super().__init__(sampler, classifier, config, finetune_class, save_folder, optimizer)

    def resume(self):
        """handles resuming of training from experiment folder"""
        if wandb.run.resumed and file_exists(self.checkpoint_file):
            checkpoint = torch.load(self.checkpoint_file)
            self.sampler.load(self.checkpoint_dir)
            self.opt.load_state_dict(checkpoint["optimizer_state_dict"])
            self.sampler.logZ = checkpoint["logZ"]
            it = checkpoint["it"]
            print(f"***** RESUMING PREVIOUS RUN AT IT={it}")

    def sampler_step(self, finetune_class, back_and_forth=False, *args, **kwargs):
        """handles one training step"""
        if not back_and_forth:
            # ------ do regular finetuning ------
            # sample x
            results_dict = self.sampler(batch_size=self.config.batch_size, sample_from_prior=False, detach_freq=self.config.detach_freq)

            # get reward
            logr_x_prime = self.classifier(results_dict['x'].float()).log_softmax(1)[:, finetune_class].max(-1)[0]

            # compute loss for posterior
            loss = 0.5 * (((results_dict['logpf_posterior'] + self.sampler.logZ - results_dict[
                'logpf_prior'] - logr_x_prime) ** 2)
                          - self.config.learning_cutoff).relu().mean()

            if self.accelerator.is_main_process:
                results_dict['PF_divergence'] = (
                            results_dict['logpf_posterior'] - results_dict['logpf_prior']).mean().item()

        else:
            # ------ do back and forth finetuning ----
            # get some samples from data
            # sample x
            results_dict = self.sampler(back_and_forth=True, steps=self.config.bf_length, detach_freq=self.config.detach_freq)

            # get reward
            logr_x = self.classifier(results_dict['x']).log_softmax(1)[:, finetune_class]
            logr_x_prime = self.classifier(results_dict['x_prime'].float()).log_softmax(1)[:, finetune_class]

            if self.accelerator.is_main_process:
                results_dict['ratio'] = logr_x_prime / logr_x

            # compute loss for posterior
            loss = .5 * (((results_dict['logpf_posterior_b'] - results_dict['logpf_posterior_f']
                           - results_dict['logpf_prior_b'] + results_dict['logpf_prior_f']
                           - logr_x + logr_x_prime
                           ) ** 2) - self.config.learning_cutoff).relu().mean()

        # log additional stuff & save
        results_dict['loss'] = loss
        results_dict['logr'] = logr_x_prime
        results_dict['logZ'] = self.sampler.logZ

        if 'x' in results_dict.keys(): del results_dict['x']

        return loss, results_dict

    def generate_plots(self, finetune_class, it=0):
        """generate plots for current prior/posterior modeled distribution"""
        with T.no_grad():
            print("generating samples")

            if not self.config.back_and_forth:
                xs = self.sampler(batch_size=self.config.plot_batch_size, sampling_length=self.config.sampling_length)['x']
                logrs = self.classifier(xs.float()).log_softmax(1)[:, finetune_class].max(-1)[0].detach().cpu().numpy()

                print("generating prior-posterior samples")
                xs_prior = self.sampler(
                    batch_size=self.config.plot_batch_size,
                    sample_from_prior=True,
                    sampling_length=self.config.sampling_length,
                )['x']
                prior_logrs = self.classifier(xs_prior.float()).log_softmax(1)[:, finetune_class].max(-1)[0].cpu().detach().numpy()

                img_filename = f'{self.save_folder}/samples/{self.config.exp_name}_prior_posterior_comparison_{it}.png'
                compare_samples(
                    samples1=[x.cpu().permute(1, 2, 0) for x in xs_prior],
                    samples2=[x.cpu().permute(1, 2, 0) for x in xs],
                    sample1_title='Prior samples',
                    sample2_title='Posterior samples',
                    sample1_logs=[f'{pr:.2f}' for pr in prior_logrs],
                    sample2_logs=[f'{pr:.2f}' for pr in logrs],
                    show=self.config.show_figures,
                    save=self.config.save_figures,
                    filename=img_filename
                )
            else:
                results = self.sampler(
                    batch_size=self.config.plot_batch_size,
                    back_and_forth=True,
                    steps=self.config.bf_length,
                    detach_freq=self.config.detach_freq,
                    sampling_length=self.config.sampling_length
                )

                xs = results['x']
                xs_posterior = results['x_prime']

                prior_logrs = self.classifier(xs).log_softmax(1)[:, finetune_class].max(-1)[0].cpu().detach().numpy()
                posterior_logrs = self.classifier(xs_posterior.float()).log_softmax(1)[:, finetune_class].max(-1)[
                    0].cpu().detach().numpy()

                img_filename = f'{self.save_folder}/samples/{self.config.exp_name}_prior_posterior_comparison_bf_{it}.png'
                compare_samples(
                    samples1=[x.cpu().permute(1, 2, 0) for x in xs],
                    samples2=[x.cpu().permute(1, 2, 0) for x in xs_posterior],
                    sample1_title='Original Samples',
                    sample2_title='Posterior samples',
                    sample1_logs=[f'{pr:.2f}' for pr in prior_logrs],
                    sample2_logs=[f'{pr:.2f}' for pr in posterior_logrs],
                    show=self.config.show_figures,
                    save=self.config.save_figures,
                    filename=img_filename
                )
        return img_filename


def diffusion_resample(args, exp_paths, batch_size=16, device='cpu'):

    print("Select experiments to regenerate:")
    for i in range(len(exp_paths)):
        print(f"[{i}]: {exp_paths[i].split('/')[-1]}")

    hub = False
    key = input("*info: insert experiment number of match keyword for multiple experiments\n"
                "       insert 'x' to supply a hugging_face repo instead.\n\n"
                "Experiment key: ")
    if key[0] == 'x' and len(key) < 3:
        key = input("Hugging_face repo handle: ")
        paths = [key]
        try:
            exp_name = key.split('/')[-1]
            file_path = hf_hub_download(key, 'run_args.json')
            args[exp_name] = DictObj(load_dict_from_file(file_path))
            hub = True

        except Exception as e:
            print(f"Failed to retrieve 'run_args.json' from the repo: '{key}'")
            raise e

    else:
        if all([k.isdigit() for k in key]):
            paths = [exp_paths[int(key)]]
        else:
            paths = [pth for pth in exp_paths if key in pth.split('/')[-1]]

    for exp_path in paths:
        exp_name = exp_path.split('/')[-1]

        while True:
            sampling_length = input(f"{exp_path.split('/')[-1]}: Sampling Length: ")
            try: sampling_length = int(sampling_length); break;
            except: pass

        if args[exp_name].lora:

            if not hub:
                # if local, check if the experiments contain anything
                saved_models = get_filenames(exp_path, contains=['adapter'], ends_with='.safetensors')
                if len(saved_models) == 0:
                    continue

            ddpm_prior = get_diffuser_ddpm(
                dataset=args[exp_name].dataset,
                device=device
            )
            ddpm_prior.scheduler.set_timesteps(sampling_length)

            ddpm_posterior = load_adapted_model(
                dataset=args[exp_name].dataset,
                model_id=exp_path,
                device=device
            )
            ddpm_posterior.scheduler.set_timesteps(sampling_length)

            prior_posterior_gfn = PosteriorPriorDGFN(
                dim=(args[exp_name].channels, args[exp_name].image_size, args[exp_name].image_size),
                traj_length=args[exp_name].traj_length,
                sampling_length=args[exp_name].sampling_length,
                # NOTE: is sampling_length < traj_length then DDPM sampling kicks in
                prior_policy_model=ddpm_prior,
                posterior_policy_model=ddpm_posterior,
                drift_model=None,
                time_spacing=ddpm_posterior.scheduler.config.timestep_spacing,
                mixed_precision=ddpm_posterior.unet.dtype == torch.float16,
                use_cuda='cpu' not in device,
                lora=args[exp_name].lora,
                checkpointing=args[exp_name].checkpointing,
                push_to_hf=False,
                exp_name=args[exp_name].exp_name
            )

        else:

            saved_models = get_filenames(exp_path, contains=['posterior'], ends_with='.pth')
            if len(saved_models) == 0:
                continue

            # load whole model
            prior_posterior_gfn = torch.load(exp_path + "/" + saved_models[0], map_location=torch.device('cpu')).to(device)
            if isinstance(prior_posterior_gfn.prior_gfn_node.policy.unet, nn.DataParallel):
                prior_posterior_gfn.prior_gfn_node.policy.unet = prior_posterior_gfn.prior_gfn_node.policy.unet.module.to(device)
                prior_posterior_gfn.posterior_gfn_node.policy.unet = prior_posterior_gfn.posterior_gfn_node.policy.unet.module.to(device)

        with torch.no_grad():
            xs_prior = prior_posterior_gfn(batch_size=batch_size, sample_from_prior=True, sampling_length=sampling_length)['x']
            xs = prior_posterior_gfn(batch_size=batch_size, sampling_length=sampling_length)['x']

        compare_samples(samples1=[x.cpu().permute(1, 2, 0) for x in xs_prior],
                        samples2=[x.cpu().permute(1, 2, 0) for x in xs],
                        sample1_title='Prior samples',
                        sample2_title='Posterior samples',
                        show=True,
                        save=True,
                        filename=f'{exp_path}_prior_posterior_comparison_final.png')


class ReinforceFinetuneTrainer:

    def __init__(self, sampler, classifier, config, finetune_class, save_folder, optimizer=None):

        wandb_key = os.getenv('WANDB_API_KEY', None)
        if wandb_key is None:
            print("NOTE: WANDB_API_KEY has not been set in the environment. Wandb tracking is off.")
            self.push_to_wandb = False
        else:
            self.push_to_wandb = True
            wandb.login(key=wandb_key)

        self.sampler = sampler
        self.opt = optimizer
        self.config = config

        self.classifier = classifier

        self.save_folder = save_folder

        # ------------------------------------------------------------------------------------------------
        self.accelerator = Accelerator(
            split_batches=True,
            mixed_precision='fp16' if config.mixed_precision else 'no',
            log_with="wandb" if self.push_to_wandb else "tensorboard",
            gradient_accumulation_steps=config.accumulate_gradient_every,
            cpu=not config.use_cuda,
            project_dir=config.save_folder
        )

        # WANDB & HF

        if self.accelerator.is_main_process:
            wandb.init(
                project="_".join(self.config.exp_name.split("_")[:-1]).split('-')[0] if '_' in self.config.exp_name else self.config.exp_name,
                dir=self.config.save_folder,
                resume=True,
                mode='online' if self.config.push_to_wandb else "offline",
                config={k: str(val) if isinstance(val, (list, tuple)) else val for k, val in self.config.__dict__.items()},
                notes=self.config.notes,
                name=self.config.exp_name.split('_')[-1]
            )
            self.checkpoint_dir = f"{self.config.save_folder}checkpoints/"
            self.checkpoint_file = self.checkpoint_dir + "checkpoint.tar"
            folder_create(self.checkpoint_dir, exist_ok=True)

            # -------------------------------------------------------------------------------------------------
            # custom logger for easy tracking and plots later
            self.logger = Logger(config)
            self.logger.save_args()

        # ------------  Load relevant dataset  ---------

        if config.back_and_forth or config.compute_fid:
            self.data_dict = get_dataset(
                dataset=config.dataset,
                batch_size=config.batch_size,
                data_path=config.data_path,
                channels=config.channels,
                image_size=config.image_size,
                x_tensor=config.x_tensor,
                y_tensor=config.y_tensor,
                splits=config.splits,
                multi_class_index=config.multi_class_index,
                workers=0 if config is None or 'linux' not in config.system else 4,
                filter_data=config.compute_fid,
                # returns an additional sliced view of the data by class, for true samples,
                filter_labels=finetune_class
            )
            self.sampler.set_loader(self.accelerator.prepare(self.data_dict['train_loader']))

        if config.compute_fid:
            # ------- FID -------------------

            true_class_dataset = cycle(self.accelerator.prepare(self.data_dict[f'train_class_view']))
            num_fid_samples = min(len(self.data_dict[f'train_class_view'].dataset), self.config.num_fid_samples)

            if num_fid_samples < 50000:
                print(f"WARNING: computing FID score with {num_fid_samples} samples. At least 50000 samples are suggested.")

            if 'test_class_view' in self.data_dict.keys():
                true_class_dataset_test = cycle(self.accelerator.prepare(self.data_dict[f'test_class_view']))
                if len(self.data_dict[f'test_class_view'].dataset) < num_fid_samples:
                    print(f"WARNING: number of elements in true class for TEST dataset "
                          f"is {len(self.data_dict[f'test_class_view'].dataset)} < {num_fid_samples}")
            else:
                true_class_dataset_test = None

            self.fid_scorer = SCOREEvaluation(
                batch_size=config.batch_size,
                dl=true_class_dataset,
                dl_test=true_class_dataset_test,
                sampler=sampler,
                channels=config.channels,
                accelerator=self.accelerator,
                stats_dir=self.config.save_folder,
                device=config.device,
                num_fid_samples=num_fid_samples,
                normalize_input=False,
                inception_block_idx=config.fid_inception_block,
            )

        self.x_dim = (config.image_size, config.image_size, config.channels)

        self.sampler, self.opt = self.accelerator.prepare(self.sampler, self.opt)

    def run(self, finetune_class, epochs=5000, **sampler_kwargs):

        it = 0

        if wandb.run.resumed and file_exists(self.checkpoint_file):
            checkpoint = torch.load(self.checkpoint_file)
            self.sampler.load(self.checkpoint_dir)
            self.opt.load_state_dict(checkpoint["optimizer_state_dict"])
            with torch.no_grad():
                self.sampler.logZ.copy_(float(checkpoint["logZ"]))

            it = checkpoint["it"]
            print(f"***** RESUMING PREVIOUS RUN AT IT={it}")

        assert self.config.accumulate_gradient_every > 0, "must set 'accumulate_gradient_every' > 0"
        # ------------  Train posterior  ---------
        while it < epochs:

            self.sampler.train()
            for si in range(self.config.accumulate_gradient_every):
                print(f"it {it} [{si + 1}/{self.config.accumulate_gradient_every}] : ".endswith(""))

                # do a training step (sample posterior, score it and backprop)
                loss, results_dict = self.sampler_step(finetune_class, **sampler_kwargs)
                self.accelerator.backward(loss)

            self.opt.step()
            self.opt.zero_grad()

            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:

                if self.config.compute_fid and it % 100 == 0:
                    scores = self.fid_scorer.fid_score()
                    for key in scores.keys():
                        results_dict[key] = scores[key]

                self.logger.log(results_dict)  # log results locally

                if it % 10 == 0:
                    self.logger.print(it)  # print progress to terminal
                    self.logger.save()  # save logs file locally

                    self.sampler.save(
                        folder=self.checkpoint_dir,
                        opt=self.opt,
                        push_to_hf=self.config.lora and it % 1000 == 0,
                        it=it
                    )

                img_filename = None
                if it % 100 == 0:
                    img_filename = self.generate_plots(finetune_class, it=it)

                # save
                log_data = {k: val.cpu().mean().item() if isinstance(val, torch.Tensor) else val for k, val in results_dict.items()}
                if img_filename is not None and self.config.save_figures:
                    image = wandb.Image(img_filename, caption=f"it: {it}")
                    log_data['samples'] = image

                wandb.log(data=log_data, step=it)  # log results in wandb

                if it % 1000 == 0 and not self.config.lora:
                    print(
                        f"#################\n"
                        f"DIVERGENCE {torch.FloatTensor([(param1 - param2).abs().sum() for param1, param2 in zip(self.sampler.prior_gfn_node.get_parameters(), self.sampler.posterior_gfn_node.get_parameters()) if param2.requires_grad]).sum()}\n"
                        f"#################\n"
                    )

            it += 1

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.sampler.save(
                folder=self.checkpoint_dir,
                opt=self.opt,
                push_to_hf=True,
                it=it
            )
        self.accelerator.end_training()
        print("training ended")

    def sampler_step(self, finetune_class, back_and_forth=False):

        results_dict = self.sampler(batch_size=self.config.batch_size, sample_from_prior=False,
                                    detach_freq=self.config.detach_freq)
        
        # get reward
        logr_x_prime = self.classifier(results_dict['x'].float()).log_softmax(1)[:, finetune_class].max(-1)[0]

        reward = self.config.reward_weight * logr_x_prime.exp()

        baseline = reward.mean()

        # reward normalibation method for REINFORCE to minimize policy gradience variance
        if self.config.baseline_reinforce == 'whitening':
            adv = (reward - baseline)/(reward.std() + 1e-8)
        elif self.config.baseline_reinforce == 'mean_normalization':
            adv = (reward - baseline)
        else:
            adv = reward

        reinforce_loss = (-adv*(results_dict['logpf_posterior']))
        kl_loss = ((results_dict['logpf_posterior'] - results_dict['logpf_prior'].detach())**2)

        # this is sample based estimation of KL divergence where sample width is 1
        loss = (reinforce_loss + self.config.kl_weight*kl_loss).mean()

        if self.accelerator.is_main_process:
            results_dict['PF_divergence'] = (
                        results_dict['logpf_posterior'] - results_dict['logpf_prior']).mean().item()

        # log additional stuff & save
        results_dict['loss'] = loss
        results_dict['logr'] = logr_x_prime
        results_dict['kl_loss'] = kl_loss
        results_dict['reinforce_loss'] = reinforce_loss

        if 'x' in results_dict.keys(): del results_dict['x']

        return loss, results_dict


class BaselineGuidance(GFNFinetuneTrainer):
    def __init__(
            self,
            sampler,
            classifier,
            config,
            finetune_class,
            save_folder,
            optimizer=None
    ):
        super().__init__(sampler, classifier, config, finetune_class, save_folder, optimizer=optimizer)

    def run(self, finetune_class, epochs=5000, **sampler_kwargs):

        # ------------  Train posterior  ---------
        for it, scale in enumerate([.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]):
            #
            # logrs = []
            # if self.config.batch_size == 32:
            #     its = 5
            # elif self.config.batch_size == 16:
            #     its = 10
            # else:
            #     its = 20
            logrs = []
            for i in range(20):
                xs = self.sampler(batch_size=self.config.batch_size, sampling_length=self.config.sampling_length, finetune_class=finetune_class)['x']
                logrs += self.classifier(xs.float()).log_softmax(1)[:, finetune_class].max(-1)[0].detach().cpu().numpy().tolist()

            self.accelerator.wait_for_everyone()
            logr = np.mean(logrs)

            if self.accelerator.is_main_process:
                self.sampler.scale = scale

                img_filename = self.generate_plots(finetune_class, it=it)

                results_dict = {}
                scores = self.fid_scorer.fid_score(grad=True)
                for key in scores.keys():
                    results_dict[key] = scores[key]

                self.logger.log(results_dict)  # log results locally

                self.logger.print(it)  # print progress to terminal
                self.logger.save()  # save logs file locally

                # save
                log_data = {k: val.cpu().mean().item() if isinstance(val, torch.Tensor) else val for k, val in results_dict.items()}
                if img_filename is not None and self.config.save_figures:
                    image = wandb.Image(img_filename, caption=f"it: {it}")
                    log_data['samples'] = image

                log_data['scale'] = scale
                log_data['logr'] = logr
                wandb.log(data=log_data, step=it)  # log results in wandb

        self.accelerator.end_training()
        print("training ended")

    def resume(self):
        """handles resuming of training from experiment folder"""
        if wandb.run.resumed and file_exists(self.checkpoint_file):
            with self.accelerator.main_process_first():
                checkpoint = torch.load(self.checkpoint_file)
                self.sampler.load(self.checkpoint_dir)
                self.opt.load_state_dict(checkpoint["optimizer_state_dict"])
                with torch.no_grad():
                    self.sampler.logZ.copy_(float(checkpoint["logZ"]))

                it = checkpoint["it"]
                print(f"***** RESUMING PREVIOUS RUN AT IT={it}")

    def generate_plots(self, finetune_class, it=0):
        """generate plots for current prior/posterior modeled distribution"""

        print("generating samples")

        if not self.config.back_and_forth:
            xs = self.sampler(batch_size=self.config.plot_batch_size, sampling_length=self.config.sampling_length, finetune_class=finetune_class)['x']
            logrs = self.classifier(xs.float()).log_softmax(1)[:, finetune_class].max(-1)[0].detach().cpu().numpy()

            print("generating prior-posterior samples")
            xs_prior = self.sampler(
                batch_size=self.config.plot_batch_size,
                sample_from_prior=True,
                sampling_length=self.config.sampling_length
            )['x']
            prior_logrs = self.classifier(xs_prior.float()).log_softmax(1)[:, finetune_class].max(-1)[0].cpu().detach().numpy()

            img_filename = f'{self.save_folder}/samples/{self.config.exp_name}_prior_posterior_comparison_{it}.png'
            compare_samples(
                samples1=[x.cpu().permute(1, 2, 0) for x in xs_prior],
                samples2=[x.cpu().permute(1, 2, 0) for x in xs],
                sample1_title='Prior samples',
                sample2_title='Posterior samples',
                sample1_logs=[f'{pr:.2f}' for pr in prior_logrs],
                sample2_logs=[f'{pr:.2f}' for pr in logrs],
                show=self.config.show_figures,
                save=self.config.save_figures,
                filename=img_filename
            )
        else:
            results = self.sampler(
                batch_size=self.config.plot_batch_size,
                back_and_forth=True,
                steps=self.config.bf_length,
                detach_freq=self.config.detach_freq,
                sampling_length=self.config.sampling_length
            )

            xs = results['x']
            xs_posterior = results['x_prime']

            prior_logrs = self.classifier(xs).log_softmax(1)[:, finetune_class].max(-1)[0].cpu().detach().numpy()
            posterior_logrs = self.classifier(xs_posterior.float()).log_softmax(1)[:, finetune_class].max(-1)[
                0].cpu().detach().numpy()

            img_filename = f'{self.save_folder}/samples/{self.config.exp_name}_prior_posterior_comparison_bf_{it}.png'
            compare_samples(
                samples1=[x.cpu().permute(1, 2, 0) for x in xs],
                samples2=[x.cpu().permute(1, 2, 0) for x in xs_posterior],
                sample1_title='Original Samples',
                sample2_title='Posterior samples',
                sample1_logs=[f'{pr:.2f}' for pr in prior_logrs],
                sample2_logs=[f'{pr:.2f}' for pr in posterior_logrs],
                show=self.config.show_figures,
                save=self.config.save_figures,
                filename=img_filename
            )
        return img_filename


def diffusion_resample(args, exp_paths, batch_size=16, device='cpu'):

    print("Select experiments to regenerate:")
    for i in range(len(exp_paths)):
        print(f"[{i}]: {exp_paths[i].split('/')[-1]}")

    hub = False
    key = input("*info: insert experiment number of match keyword for multiple experiments\n"
                "       insert 'x' to supply a hugging_face repo instead.\n\n"
                "Experiment key: ")
    if key[0] == 'x' and len(key) < 3:
        key = input("Hugging_face repo handle: ")
        paths = [key]
        try:
            exp_name = key.split('/')[-1]
            file_path = hf_hub_download(key, 'run_args.json')
            args[exp_name] = DictObj(load_dict_from_file(file_path))
            hub = True

        except Exception as e:
            print(f"Failed to retrieve 'run_args.json' from the repo: '{key}'")
            raise e

    else:
        if all([k.isdigit() for k in key]):
            paths = [exp_paths[int(key)]]
        else:
            paths = [pth for pth in exp_paths if key in pth.split('/')[-1]]

    for exp_path in paths:
        exp_name = exp_path.split('/')[-1]

        while True:
            sampling_length = input(f"{exp_path.split('/')[-1]}: Sampling Length: ")
            try: sampling_length = int(sampling_length); break;
            except: pass

        if args[exp_name].lora:
            model_path = exp_path + '/checkpoints/'

            if not hub:
                # if local, check if the experiments contain anything
                saved_models = get_filenames(model_path, contains=['adapter'], ends_with='.safetensors')
                if len(saved_models) == 0:
                    print(f"no '.safetensors' found at '{exp_path+'/checkpoints/'}' ")
                    continue

            ddpm_prior, _ = get_diffuser_ddpm(
                dataset=args[exp_name].dataset,
                device=device
            )
            ddpm_prior.scheduler.set_timesteps(sampling_length)

            ddpm_posterior = load_adapted_model(
                dataset=args[exp_name].dataset,
                model_id=model_path,
                device=device
            )
            ddpm_posterior.scheduler.set_timesteps(sampling_length)

            prior_posterior_gfn = PosteriorPriorDGFN(
                dim=(args[exp_name].channels, args[exp_name].image_size, args[exp_name].image_size),
                traj_length=args[exp_name].traj_length,
                sampling_length=args[exp_name].sampling_length,
                # NOTE: is sampling_length < traj_length then DDPM sampling kicks in
                prior_policy_model=ddpm_prior,
                posterior_policy_model=ddpm_posterior,
                drift_model=None,
                mixed_precision=ddpm_posterior.unet.dtype == torch.float16,
                use_cuda='cpu' not in device,
                lora=args[exp_name].lora,
                checkpointing=args[exp_name].checkpointing,
                push_to_hf=False,
                exp_name=args[exp_name].exp_name
            )

        else:

            saved_models = get_filenames(exp_path, contains=['posterior'], ends_with='.pth')
            if len(saved_models) == 0:
                continue

            # load whole model
            prior_posterior_gfn = torch.load(exp_path + "/" + saved_models[0], map_location=torch.device('cpu')).to(device)
            if isinstance(prior_posterior_gfn.prior_node.policy.unet, nn.DataParallel):
                prior_posterior_gfn.prior_node.policy.unet = prior_posterior_gfn.prior_node.policy.unet.module.to(device)
                prior_posterior_gfn.posterior_node.policy.unet = prior_posterior_gfn.posterior_node.policy.unet.module.to(device)

        with torch.no_grad():
            xs_prior = prior_posterior_gfn(batch_size=batch_size, sample_from_prior=True, sampling_length=sampling_length)['x']
            xs = prior_posterior_gfn(batch_size=batch_size, sampling_length=sampling_length)['x']

        compare_samples(samples1=[x.cpu().permute(1, 2, 0) for x in xs_prior],
                        samples2=[x.cpu().permute(1, 2, 0) for x in xs],
                        sample1_title='Prior samples',
                        sample2_title='Posterior samples',
                        show=True,
                        save=True,
                        filename=f'{exp_path}/samples/prior_posterior_comparison_final.png')

