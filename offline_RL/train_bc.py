import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import d4rl
import random
import argparse
from distutils.util import strtobool
import os
import time
from model import DiffusionModel
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

class D4RLDataset(Dataset):
    def __init__(self, data):
        self.states = data['observations']
        self.actions = data['actions']

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        states = self.states[idx]
        actions = self.actions[idx]
        return states, actions

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="advantage-diffusion",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default='swish',
        help="the entity (team) of wandb's project")

    parser.add_argument("--env-id", type=str, default="hopper-medium-expert",
        help="the id of the environment")
    parser.add_argument("--diffusion-steps", type=int, default=75)
    parser.add_argument("--batch-size", type=float, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--schedule", type=str, default='linear')
    parser.add_argument("--n-epochs", type=int, default=100000)
    parser.add_argument("--predict", type=str, default='epsilon')
    parser.add_argument("--policy-net", type=str, default='mlp')
    parser.add_argument("--num-eval", type=int, default=10)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    filename = args.env_id+"_"+args.exp_name
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    env = gym.make(args.env_id)
    dataset = env.get_dataset()
    #if args.predict == 'epsilon':
    dataset['actions'] = np.arctanh(np.clip(dataset['actions'],-0.99,0.99))
    data = D4RLDataset(dataset)
    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=True)
    
    model = DiffusionModel(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0], diffusion_steps=args.diffusion_steps, predict=args.predict, policy_net=args.policy_net).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    best_eval = -10000
    for epoch in range(args.n_epochs):
        loss_epoch = 0.0
        for states, actions in dataloader:
            states = states.float().to(device)
            actions = actions.float().to(device)
            optimizer.zero_grad()
            loss = model.compute_loss(states, actions).mean()
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()
            #print(loss.item(), epoch)
        
        if epoch % 15 == 0:
            torch.save(model.state_dict(), "bc_models/"+filename+".pth")
        with torch.no_grad():
            if epoch % 15 == 0:
                avg_reward = 0.0
                for i in range(args.num_eval):
                    s = env.reset()
                    done = False
                    while not done:
                        s_tensor = torch.tensor(s).float().to(device).unsqueeze(0)
                        a = model.sample(s_tensor).detach().cpu().numpy()
                        a = torch.tanh(torch.tensor(a)).detach().cpu().numpy()[0]
                        s, r, done, _ = env.step(a)
                        avg_reward += r
                avg_reward /= args.num_eval
                avg_reward = env.get_normalized_score(avg_reward)*100
                wandb.log({"loss": loss_epoch/len(dataloader), "avg_reward": avg_reward})
                if avg_reward > best_eval:
                    best_eval = avg_reward
                    torch.save(model.state_dict(), "bc_models/"+filename+"_best.pth")
            else:
                wandb.log({"loss": loss_epoch/len(dataloader)})