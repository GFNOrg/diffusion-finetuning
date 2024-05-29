# Inverse Problems with score based diffusion
Inverse Problems with score based diffusion



## File Structure

- `src/`: Contains all the source code.
  - `src/models`: Code for denoising and diffusion models.
    - `src/models/pretrained/`: Should contain pretrained weights.
  - `src/utils/`: Custom utility libraries.
  - Code to run experiments.
- `results/`: Stores local results (can be redirected).


### Dataset Generation

To prepare datasets for experiments, run `create_data.py` in `src`. Set `PATH_TO_DATA` to the desired storage location.

Supported datasets: `mnist`, `cifar`, `utkface`.

##### Example Command

```bash
python create_data.py --data_path PATH_TO_DATA
```


### Training a Prior Model Locally
Train a prior model locally using `train_prior.py`. The results will be in saved in `./results/` folder unless the `--save_folder` argument is set. Supported models are currently `ScoreNet`, `UNet`. 
###### NOTE: most arguments are only examplar, and will quickly become outdated. Check the ".sh" scripts for up-to-date running examples.
##### Example Command
```bash
python train_prior.py --batch_size 64 --traj_length 200 --data_path PATH_TO_DATA --load_path ./src/models/pretrained --dataset 'mnist' --lr 1e-3 --epochs 5000 --sampling_length 100 --model UNet --exp_name first_experiment
```


### Fine-Tuning a Pretrained Model

To fine-tune, move pretrained weights to `src/models/pretrained/`. Use `finetune_posterior.py` for fine-tuning.
##### Example Command
```bash
python finetune_posterior.py --traj_length 200 --data_path PATH_TO_DATA --load_path ./src/models/pretrained --dataset mnist --lr 1e-5 --sampling_length 100 --batch_size 32 --epochs 10000 --model UNet --finetune_class 7 --exp_name second_experiment
```