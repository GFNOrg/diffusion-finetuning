# Taken from https://github.com/GFNOrg/gfn-lm-tuning/blob/main/infill_subj_arithmetic/eval_infill.py

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer
import random
import gzip
import pickle
from load_model import load_model
from gfn import PosteriorPriorGFN

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.gleu_score import sentence_gleu
from bert_score import BERTScorer

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="gpt2")
parser.add_argument("--save_path", type=str, default="prompting.pkl.gz")
parser.add_argument("--temp", type=float, default=1.0)
parser.add_argument("--max_eval_len", type=int, default=12)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--eval_baseline", type=bool, default=False)
parser.add_argument("--prior", type=bool, default=False)
parser.add_argument("--load_checkpoint_path", type=str, default=None)
parser.add_argument("--query_type", type=str, default="infill")
args = parser.parse_args()


def seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(args.seed)
scorer = BERTScorer(model_type="microsoft/deberta-xlarge-mnli", lang="en", rescale_with_baseline=True)
load_checkpoint_path = args.load_checkpoint_path

tokenizer = AutoTokenizer.from_pretrained("gpt2")

df = pd.read_csv("./data/stories.csv")
data = []

for index in range(0, 100):
    row = df.iloc[index]
    story = {}
    if args.query_type == "infill":
        story["str_query"] = "Beginning: " + row["sentence1"] + " " + row["sentence2"] + " " + row["sentence3"]
        story["str_query"] += "\nEnding: " + row["sentence5"] + "\nMiddle:"
    elif args.query_type == "beginning":
        story["str_query"] = row["sentence1"] + " " + row["sentence2"] + " " + row["sentence3"]
    story["str_sol"] = row["sentence5"]
    story["str_rationale"] = row["sentence4"]
    story["beginning"] = row["sentence1"] + " " + row["sentence2"] + " " + row["sentence3"]

    data.append(story)

train_queries = []
train_rationales = []
train_sols = []
train_weight = []
train_beginning = []

test_queries = []
test_num_sols = []
test_sols = []
test_beginning = []

for i, sample in enumerate(data):
    train_queries.append(sample['str_query'])
    train_rationales.append(sample['str_rationale'])
    train_sols.append(sample['str_sol'])
    train_weight.append(1)
    train_beginning.append(sample['beginning'])


for sample in data:
    test_queries.append(sample['str_query'])
    if "num_sol" in sample:
        test_num_sols.append(sample['num_sol'])
    else:
        test_num_sols = None
    test_sols.append(sample['str_sol'])
    test_beginning.append(sample['beginning'])

encoded_train_queries = [tokenizer(query, return_tensors='pt')['input_ids'].cuda() for query in train_queries]
encoded_train_sols = [tokenizer(answer, return_tensors='pt')['input_ids'].cuda() for answer in train_sols]
encoded_test_queries = [tokenizer(query, return_tensors='pt')['input_ids'].cuda() for query in test_queries]
encoded_train_beginning = [tokenizer(query, return_tensors='pt')['input_ids'].cuda() for query in train_beginning]
encoded_test_beginning = [tokenizer(query, return_tensors='pt')['input_ids'].cuda() for query in test_beginning]

pad_token_id = tokenizer.eos_token_id
eos_token_id = tokenizer.eos_token_id

def compute_metrics(ref, cands):
    # BLEU
    ref_tokens = ref.split()
    cands_tokens = [cand.split() for cand in cands]
    bleu_scores = [sentence_bleu([ref_tokens], cand, smoothing_function=SmoothingFunction().method1) for cand in cands_tokens]

    # GLEU
    gleu_scores = [sentence_gleu([ref_tokens], cand) for cand in cands_tokens]

    # BERTScore
    P, R, F1 = scorer.score(cands, [ref]*len(cands), verbose=False)
    bertscore_f1 = F1.tolist()

    return bleu_scores, gleu_scores, bertscore_f1


def load_models(model):
    prior_model, graph, noise = load_model("louaaron/sedd-small", torch.device("cuda"))
    with gzip.open(args.load_checkpoint_path, "rb") as f:
        posterior_model = pickle.load(f)
    
    gfn = PosteriorPriorGFN(prior_model, posterior_model, None, graph, noise, "analytic", 10, 16, 1e-5, torch.device("cuda"))
    return gfn


def eval_baseline(gfn, tokenizer, temp=1.0, max_eval_len=12, batch_size=100):
    metrics = {
        "bleu": [],
        "gleu": [],
        "bertscore_f1": []
    }
    examples = []
    for idx in tqdm(range(len(encoded_train_queries))):
        # encoded_input = encoded_train_queries[idx]
        story = data[idx]
        prefix_ids = tokenizer(story["beginning"]).input_ids
        suffix_ids = tokenizer(story["str_sol"]).input_ids
        seq_len = len(prefix_ids) + len(tokenizer([story["str_rationale"]]).input_ids[0]) + len(suffix_ids)
        suffix_str = [story["str_sol"]] * batch_size
        input_ids = torch.tensor(prefix_ids+suffix_ids, device="cuda")[None].repeat(batch_size, 1)
        input_locs = list(range(len(prefix_ids))) + list(range(seq_len - len(suffix_ids), seq_len))
        mask = torch.ones((batch_size, seq_len))
        mask[:, input_locs] = 0
        def projector(inp):
            inp = inp.clone()
            with torch.no_grad():
                inp[:, input_locs] = input_ids
            return inp
        with torch.no_grad():
            results = gfn.sample_fwd((batch_size, seq_len), sample_from_prior=True, detach_prob=1, tokenizer=tokenizer, projector=projector)
        torch.cuda.empty_cache()
        samples = results['x'][mask.bool()].reshape(-1, mask.int().sum(1)[0].item())
        generated_text = tokenizer.batch_decode(samples, skip_special_tokens=True)
        generated_text = [gt[1:] for gt in generated_text]
        # import pdb; pdb.set_trace();
        examples.append((data[idx]["beginning"],data[idx]["str_sol"], data[idx]["str_rationale"], generated_text))
        bleu, gleu, bertscore_f1 = compute_metrics(train_rationales[idx], generated_text)
        metrics["bleu"].append(bleu)
        metrics["gleu"].append(gleu)
        metrics["bertscore_f1"].append(bertscore_f1)
    return metrics, examples

def eval_model(gfn, tokenizer, sample_from_prior=False, batch_size=100):
    metrics = {
        "bleu": [],
        "gleu": [],
        "bertscore_f1": []
    }
    examples = []
    for idx in tqdm(range(len(encoded_train_queries))):
        story = data[idx]
        prefix_ids = tokenizer(story["beginning"]).input_ids
        seq_len = len(prefix_ids) + len(tokenizer([story["str_rationale"]]).input_ids[0])
        input_ids = torch.tensor(prefix_ids, device="cuda")[None].repeat(batch_size, 1)
        input_locs = list(range(len(prefix_ids)))
        mask = torch.ones((batch_size, seq_len))
        mask[:, input_locs] = 0
        
        def projector(inp):
            inp = inp.clone()
            with torch.no_grad():
                inp[:, input_locs] = input_ids
            return inp
        
        with torch.no_grad():
            results = gfn.sample_fwd((batch_size, seq_len), sample_from_prior=sample_from_prior, detach_prob=1, tokenizer=tokenizer, projector=projector)
        samples = results['x'][mask.bool()].reshape(-1, mask.int().sum(1)[0].item())
        
        torch.cuda.empty_cache()
        samples = results['x'][mask.bool()].reshape(-1, mask.int().sum(1)[0].item())
        generated_text = tokenizer.batch_decode(samples, skip_special_tokens=True)
        generated_text = [gt[1:] for gt in generated_text]
        full_text = tokenizer.batch_decode(results['x'], skip_special_tokens=True)
        examples.append((data[idx]["beginning"],data[idx]["str_sol"], data[idx]["str_rationale"], generated_text, full_text))
        bleu, gleu, bertscore_f1 = compute_metrics(train_rationales[idx], generated_text)
        metrics["bleu"].append(bleu)
        metrics["gleu"].append(gleu)
        metrics["bertscore_f1"].append(bertscore_f1)

    return metrics, examples

gfn = load_models(None)
# eval_metrics, examples = eval_model(model, tokenizer, temp=args.temp, max_eval_len=args.max_eval_len, batch_size=args.max_eval_len)
if args.eval_baseline:
    print("Baseline")
    eval_metrics, examples = eval_baseline(gfn, tokenizer, batch_size=100)
else:
    print("GFN", args.prior)
    eval_metrics, examples = eval_model(gfn, tokenizer, sample_from_prior=args.prior)

import pickle
import gzip
with gzip.open(args.save_path, "wb") as f:
    pickle.dump(examples, f)
print("BLEU: ", np.mean(eval_metrics["bleu"]))
print("GLEU: ", np.mean(eval_metrics["gleu"]))
print("BERTScore F1: ", np.mean(eval_metrics["bertscore_f1"]))