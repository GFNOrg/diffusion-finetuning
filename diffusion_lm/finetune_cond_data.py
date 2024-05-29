import torch
import gzip
import pickle
import os

import torch.nn.functional as F
from load_model import load_model
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, get_constant_schedule_with_warmup, AutoModelForCausalLM
from gfn import PosteriorPriorGFN
import random
import numpy as np
import copy
from tqdm import tqdm
from omegaconf import OmegaConf
from transformers import AutoTokenizer, RobertaForSequenceClassification
from dataclasses import dataclass, field
import editdistance
import heapq
import wandb
import pandas as pd

def seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)

class StoryLLReward:
    def __init__(self, name, batch_size):
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(name)
        self.model.eval()
        self.batch_size = batch_size
    
    def __call__(self, input, str_postfix, tokenizer, **kwargs):
        rewards = []
        for i in range(0, len(input), self.batch_size):
            input_batch = input[i:i+self.batch_size]
            
            str_input = tokenizer.batch_decode(input_batch)
            
            encoded_inp = self.tokenizer(str_input, return_tensors="pt", padding=True)
            encoded_postfix = self.tokenizer(str_postfix, return_tensors="pt", padding=True)
            # encoded_gen = self.tokenizer(str_gen, return_tensors="pt", padding=True)
            prefix_lens = encoded_inp["attention_mask"].sum(1) - encoded_postfix["attention_mask"].sum(1)
            prefix_lens = prefix_lens[0].item()
            # import pdb; pdb.set_trace();
            with torch.no_grad():
                out = self.model(encoded_inp.input_ids, attention_mask=encoded_inp.attention_mask).logits.log_softmax(-1)[:, :-1]
                toks = encoded_inp.input_ids[:, 1:].unsqueeze(-1)
                log_probs = torch.gather(out, dim=-1, index=toks).squeeze(-1)
                log_probs = log_probs[:, prefix_lens:]
                log_probs = log_probs.sum(-1)
            rewards.extend(log_probs.numpy().tolist())
        return torch.tensor(rewards)


def main():
    conf = OmegaConf.load("configs/ft.yaml")
    cli_conf = OmegaConf.from_cli()
    cfg = OmegaConf.merge(conf, cli_conf)
    seed_everything(cfg.seed)
    wandb.init(project="discrete_diff_ft", config=OmegaConf.to_container(cfg, resolve=True), mode=cfg.wandb_mode)

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    device = torch.device('cuda')
    if not os.path.exists(f"{cfg.save_dir}/{cfg.exp_name}"):
        os.makedirs(f"{cfg.save_dir}/{cfg.exp_name}")
    df = pd.read_csv("data/stories.csv")
    data = []
    for index in range(100, 1000):
        row = df.iloc[index]
        story = {}
        story["str_query"] = "Beginning: " + row["sentence1"] + " " + row["sentence2"] + " " + row["sentence3"]
        story["str_query"] += "\nEnding: " + row["sentence5"] + "\nMiddle:"
        story["str_sol"] =  row["sentence5"]
        story["str_rationale"] = " " + row["sentence4"]
        story["beginning"] = row["sentence1"] + " " + row["sentence2"] + " " + row["sentence3"]
        # print(row['sentence1'])
        # if index == config.dataset.max_data:
        #     break
        data.append(story)
    # prefix_ids = tokenizer("David noticed he had put on a lot of weight recently. He examined his habits to try and figure out the reason. He realized he'd been eating too much fast food lately.").input_ids
    # # suffix_ids = tokenizer("After a few weeks, he started to feel much better.").input_ids
    # suffix_str = ["After a few weeks, he started to feel much better."] * cfg.batch_size
    # input_ids = prefix_ids # + suffix_ids
    # input_locs = list(range(len(prefix_ids))) #+ list(range(60-len(suffix_ids), 60))
    
    # input_ids = torch.tensor(input_ids, device="cuda")[None].repeat(cfg.batch_size, 1)
    # mask = torch.ones((cfg.batch_size, cfg.seq_len))
    # mask[:, input_locs] = 0



    prior_model, graph, noise = load_model(cfg.model_name, device)
    eval_model = GPT2LMHeadModel.from_pretrained('gpt2-large').to(device)
    eval_model.eval()
    posterior_model = copy.deepcopy(prior_model)
    prior_model.eval()
    posterior_model.eval()
    
    if cfg.reward_type == "dummy":
        reward_model = lambda x, tok: torch.zeros(x.shape[0], device=x.device)
    elif cfg.reward_type=="story":
        reward_model = StoryLLReward(cfg.likelihood_model, cfg.reward_batch_size)

    gfn = PosteriorPriorGFN(prior_model, posterior_model, reward_model, graph, noise, "analytic", 0, cfg.sampling_len, 1e-5, device)
    grad_accumulation = cfg.grad_accumulation
    if cfg.loss_type == "tb":
        opt = torch.optim.Adam([{'params': gfn.posterior_model.parameters(), 'lr': cfg.learning_rate}, {'params': gfn.logZ, "lr": cfg.learning_rate_Z}])
    elif cfg.loss_type == "vargrad":
        opt = torch.optim.Adam(gfn.posterior_model.parameters(), lr=cfg.learning_rate)
    scheduler = get_constant_schedule_with_warmup(opt, cfg.warmup_steps)
    batch_size = cfg.batch_size
    pb = tqdm(range(cfg.num_steps))
    
    opt.zero_grad()
    for i in pb:
        if i < cfg.reward_temp_sched_steps:
            reward_invtemp = cfg.reward_invtemp_start + (cfg.reward_invtemp_end - cfg.reward_invtemp_start) * i / cfg.reward_temp_sched_steps 
        
        idx = np.random.choice(len(data), 1)
        story = data[idx[0]]
        prefix_ids = tokenizer(story["beginning"]).input_ids
        # suffix_ids = tokenizer(story["str_sol"]).input_ids
        cfg.seq_len = len(prefix_ids) + len(tokenizer([story["str_rationale"]]).input_ids[0])
        suffix_str = [story["str_sol"]] * cfg.batch_size
        # input_ids = torch.tensor(prefix_ids, device="cuda")[None].repeat(cfg.batch_size, 1)
        input_ids = torch.tensor(prefix_ids, device="cuda")[None].repeat(cfg.batch_size, 1)
        input_locs = list(range(len(prefix_ids))) # + list(range(cfg.seq_len - len(suffix_ids), cfg.seq_len))
        mask = torch.ones((cfg.batch_size, cfg.seq_len))
        mask[:, input_locs] = 0

        def projector(inp):
            inp = inp.clone()
            with torch.no_grad():
                inp[:, input_locs] = input_ids
            return inp

        results = gfn.sample_fwd((batch_size, cfg.seq_len), sample_from_prior=i % 5 == 0, detach_prob=0.7, tokenizer=tokenizer, projector=projector)
        samples = results['x'][mask.bool()].reshape(-1, mask.int().sum(1)[0].item())
        re_samples = results['x']
        # traj = results['traj']
        logreward = reward_model(re_samples, suffix_str, tokenizer).to(device)
        fwd_r = logreward.mean().item()
        if cfg.loss_type == "tb":
            forward_loss = 0.5 * ((results["posterior_log_probs"].sum(1) + gfn.logZ - results["prior_log_probs"].sum(1) - 1 * logreward) ** 2 - cfg.cutoff).relu().mean()
        elif cfg.loss_type == "vargrad":
            term = (results["posterior_log_probs"].sum(1) - results["prior_log_probs"].sum(1) - reward_invtemp * logreward)
            forward_loss = 0.5 * ((term + term.mean()) ** 2 - cfg.cutoff).relu().mean()
        loss = forward_loss

        loss = loss / grad_accumulation
        loss.backward()

        norm = torch.nn.utils.clip_grad_norm_(gfn.posterior_model.parameters(), 0.1)
        if (i + 1) % grad_accumulation == 0:
            opt.step()
            opt.zero_grad()
            scheduler.step()
        
        with torch.no_grad():
            eval_out = eval_model(samples, labels=samples)
            pplx = eval_out.loss.exp().mean()
        
        update_str = f"Forward Loss: {forward_loss.item():.3f}, Fwd Log Reward: {fwd_r:.3f}, PPLX: {pplx.item():.3f}"
        pb.set_description(update_str)
        update_dict = {"fwd_loss": forward_loss.item(), "fwd_logreward": fwd_r, "pplx": pplx.item()}
        wandb.log(update_dict)
        
        if i % 50 == 0:
            print("Posterior: ", tokenizer.batch_decode(samples))
            print("Posterior Log Reward: ", fwd_r)
            print("Posterior PPLX: ", pplx.item())
            with torch.no_grad():
                prior_samples = gfn.sample_fwd((batch_size, cfg.seq_len), sample_from_prior=True, detach_prob=1, tokenizer=tokenizer, projector=projector)['x'][mask.bool()].reshape(-1, mask.int().sum(1)[0].item())
                print("Prior: ", tokenizer.batch_decode(prior_samples))
                prior_reward = reward_model(prior_samples, suffix_str, tokenizer).mean().item()
                print("Prior Log Reward: ", prior_reward)
                with torch.no_grad():
                    eval_out = eval_model(prior_samples, labels=prior_samples)
                    prior_pplx = eval_out.loss.exp().mean()
                print("Prior PPLX: ", prior_pplx.item())
                wandb.log({"prior_logreward": prior_reward, "prior_pplx": prior_pplx.item()}, commit=False)
            with gzip.open(f"{cfg.save_dir}/{cfg.exp_name}/posterior_{i}.pkl.gz", "wb") as f:
                pickle.dump(posterior_model, f)

if __name__=="__main__":
    main()