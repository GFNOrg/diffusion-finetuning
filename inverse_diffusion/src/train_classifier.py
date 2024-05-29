from models.classifiers import CNN, ResNet
from utils.args import fetch_args
from utils.data_loaders import get_dataset, cycle
from utils.pytorch_utils import seed_experiment, train_classifier,  NoContext
from utils.simple_io import *
from torch.cuda.amp import autocast

import torch as T
import numpy as np
import gc

# get arguments for the run
args, state = fetch_args(exp_prepend='')
print(f"Running experiment on '{args.device}'")

seed_experiment(args.seed)

if 'cnn' in args.classifier_model.lower():
    classifier = CNN(
        input_size=args.image_size,
        channels=args.channels,
        num_classes=args.num_classes
    )
else:
    classifier = ResNet(
        input_size=args.image_size,
        channels=args.channels,
        num_classes=args.num_classes,
        depth=args.classifier_depth,
        finetune=args.classifier_pretrained
    )

pretrained_classifer = get_filenames(
    args.load_path,
    contains=[args.dataset, "classifier", f"_{args.multi_class_index}_"],
    ends_with='.pth'
)

# if len(pretrained_classifer) == 0:
print(f"No classifier '{args.model}' found for '{args.dataset}' data, \nTraining classifier...")
trained_params, best_epoch, classifier = train_classifier(
    model=classifier,
    batch_size=128,
    data_path=args.data_path,
    dataset=args.dataset,
    channels=args.channels,
    image_size=args.image_size,
    x_tensor=args.x_tensor,
    y_tensor=args.y_tensor,
    multi_class_index=args.multi_class_index,
    workers=0 if args is None or 'linux' not in args.system else 4,
    use_cuda=args.use_cuda
)
T.save(trained_params, f'{args.load_path}/{args.dataset}_classifier_{args.multi_class_index}_{best_epoch}.pth')
# else:
#     print(f"A classifier already exists for {args.dataset}, classes: {args.multi_class_index}: {args.finetune_class}!")