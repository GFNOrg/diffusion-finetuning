import torch
import argparse

from load_model import load_model
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import torch.nn.functional as F
import sampling


def main():
    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument("--model_path", default="louaaron/sedd-small", type=str)
    parser.add_argument("--dataset", default="wikitext103", type=str)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=512)
    args = parser.parse_args()

    
    device = torch.device('cuda')
    model, graph, noise = load_model(args.model_path, device)
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    eval_model = GPT2LMHeadModel.from_pretrained("gpt2-large").to(device)
    sampling_fn = sampling.get_pc_sampler(
        graph, noise, (args.batch_size, 128), 'analytic', args.steps, device=device
    )
    samples = sampling_fn(model)
    with torch.no_grad():
        eval_out = eval_model(samples, labels=samples)
        logits = eval_out.logits.transpose(-1, -2)
        pplx = F.cross_entropy(logits[..., :-1], samples[..., 1:], reduction="none").exp().mean()
    text_samples = tokenizer.batch_decode(samples)
    
    for i in text_samples:
        print(i)
        print("=================================================")

    print(f"Perplexity: {pplx.item()}")

if __name__=="__main__":
    main()