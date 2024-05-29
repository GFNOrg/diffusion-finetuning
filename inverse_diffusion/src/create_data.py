from utils.args import fetch_args
from utils.simple_io import *
from PIL import Image

import torchvision
import gdown
import deeplake

# get arguments for the run
args, state = fetch_args()

if 'celeb' in args.dataset.lower():
    dt_id = 'celeb-a'
else:
    dt_id = args.dataset.lower()


if args.dataset.lower() in ['mnist', 'celeba', 'cifar10']:
    splits = ['train', 'test']
    for split in splits:
    # use deep lake
        deeplake.dataset(f"hub://activeloop/{dt_id}-{split}", access_method='local')

elif 'utkface' in args.dataset.lower():
    def setup_utkface_dataset(args):
        # Define Google Drive links
        # utkface_link = 'https://drive.google.com/file/d/1W-vm-rgSDsPA015wQQ9vWzquR_KvgBwe'
        # crop1_link = 'https://drive.google.com/file/d/19GNs2OPm0zvkR99nFlXNo1aQTB22HFCj'

        # Define destination paths
        utkface_zip_path = f'{args.data_path}/UTKFace.zip'
        crop1_zip_path = f'{args.data_path}/crop1.zip'

        # a file
        utkface_id = "1W-vm-rgSDsPA015wQQ9vWzquR_KvgBwe"
        crop1_id = "19GNs2OPm0zvkR99nFlXNo1aQTB22HFCj"
        gdown.download(id=utkface_id, output=utkface_zip_path)
        gdown.download(id=crop1_id, output=crop1_zip_path)

        # Extract ZIP files
        extract_zip(utkface_zip_path, f'{args.data_path}')
        extract_zip(crop1_zip_path, f'{args.data_path}')


    # Assuming fetch_args() provides necessary paths
    args, state = fetch_args()
    setup_utkface_dataset(args)

    #example run > python create_data.py --dataset utkface --data_path $SCRATCH/data/
