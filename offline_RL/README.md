# QFlow Offline 

IQL codebase adapted from public github repo: https://github.com/gwthomas/IQL-PyTorch


### Install Mujoco

Install Mujoco, following the instructions here: https://github.com/openai/mujoco-py?tab=readme-ov-file#install-mujoco

### Train BC prior

```
python train_bc.py --env-id 'halfcheetah-medium-replay-v2' --diffusion-steps 75 --n-epochs 1500 [--track]
```

### Train Q function with IQL

```
cd 'IQL_PyTorch'
python3 main.py --env-name 'halfcheetah-medium-replay-v2' [--track]
```

### Policy extraction with RTB
For reproducing results in paper, use alpha values from Table G.2 in the paper.

```
python qflow_offline.py --env-id 'halfcheetah-medium-replay' --alpha 0.05 --diffusion-steps 75 --batch-size 64 --num-eval 10 [--track]
```