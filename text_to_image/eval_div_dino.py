import torch
import torchvision.transforms as transforms
from PIL import Image
import timm
import glob
from torch.nn.functional import cosine_similarity
from argparse import ArgumentParser

def main(args):
    # Load the pre-trained DINO model
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    model.eval()

    # Assume CUDA is available and use GPU for computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Transformations for the input images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load images
    image_paths = glob.glob(args.img_dir+"*.png")  # Adjust the path and extension
    images = [transform(Image.open(x).convert('RGB')).unsqueeze(0).to(device) for x in image_paths]

    # Extract features
    with torch.no_grad():
        features = [model(x).squeeze(0) for x in images]

    # Compute pairwise cosine similarity
    n = len(features)
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
  main(parser.parse_args())