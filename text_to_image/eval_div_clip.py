import torch
import torchvision.transforms as transforms
from PIL import Image
import glob
from torch.nn.functional import cosine_similarity
from argparse import ArgumentParser
from transformers import CLIPModel, CLIPProcessor  # pylint: disable=g-multiple-import
from transformers import CLIPTokenizer  # pylint: disable=g-multiple-import

def main(args):
    # Load the pre-trained CLIP model
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    reward_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    reward_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    # Assume CUDA is available and use GPU for computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load images
    image_paths = glob.glob(args.img_dir + "*.png")  # Adjust the path and extension
    images = [Image.open(x).convert('RGB') for x in image_paths]

    # Process images
    inputs = reward_processor(images=images, return_tensors="pt")
    pixels = inputs.pixel_values.to(device)

    with torch.no_grad():
        features = model.get_image_features(pixels)

    # Compute pairwise cosine similarity
    n = features.size(0)
    similarity_matrix = torch.zeros((n, n), device=device)
    for i in range(n):
        for j in range(i + 1, n):
            similarity = cosine_similarity(features[i].unsqueeze(0), features[j].unsqueeze(0))
            similarity_matrix[i][j] = similarity
            similarity_matrix[j][i] = similarity  # since cosine similarity is symmetric

    # Calculate average pairwise cosine similarity (excluding self-similarity)
    upper_tri_indices = torch.triu_indices(row=n, col=n, offset=1)  # Offset 1 to exclude diagonal
    average_similarity = torch.mean(similarity_matrix[upper_tri_indices[0], upper_tri_indices[1]])

    # Print the average cosine similarity
    print("Average Pairwise Cosine Similarity:", average_similarity.item())

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--img_dir", default="output_img/green_rabbit/kl_0.01/")
    args = parser.parse_args()
    main(args)