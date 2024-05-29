import os
import json
import torch
from PIL import Image
import ImageReward as imagereward
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from argparse import ArgumentParser
import utils

def evaluate_image_reward(image_folder, prompt, output_file="image_rewards.json"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the ImageReward model
    image_reward = imagereward.load("ImageReward-v1.0")
    image_reward.requires_grad_(False)
    image_reward.to(device, dtype=torch.float16)

    rewards = 0.0
    n = 0

    for image_file in os.listdir(image_folder):
        if image_file.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, image_file)
            image = Image.open(image_path).convert("RGB")

            # Get image reward
            blip_reward, _ = utils.image_reward_get_reward(image_reward, image, prompt, torch.float16)#image_reward(image, prompt, dtype=torch.float16)

            # Store the rewards in a dictionary
            rewards += blip_reward.item()
            n += 1

    print("Avg reward: ", rewards/n)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--img_dir")
    parser.add_argument("--prompt")
    args = parser.parse_args()
    image_folder = args.img_dir
    prompt = args.prompt
    evaluate_image_reward(image_folder, prompt)
