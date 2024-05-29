Code is based on [`louaaron/Score-Entropy-Discrete-Diffusion`](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion).

To train the reward model, please follow instructions at [GFNOrg/gfn-lm-tuning](https://github.com/GFNOrg/gfn-lm-tuning/tree/main/infill_subj_arithmetic).

The entry point to the code is `finetune_cond_data.py`. The following command with the right path to the reward model should run the experiments in the paper

```bash
python finetune_cond_data.py reward_type=story sampling_len=15 likelihood_model=<path_to_reward> wandb_mode=online loss_type=vargrad save_dir="<output_base_path>"
```

For evaluation use the `eval.py` script. 
```bash
python eval.py --load_checkpoint_path <path_to_checkpoint> --save_path <output_path>.pkl.gz
```

