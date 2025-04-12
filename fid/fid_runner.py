import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset

# Set up paths
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STYLE_FEATS_PATH = "/kaggle/input/style-feats-adain-1000/style_feats_adain_1000.npy"
EDM_FID_SCRIPT = os.path.join(BASE_PATH, "fid", "edm", "fid.py")

# Add repo to path
sys.path.append(BASE_PATH)
sys.path.append(os.path.join(BASE_PATH, 'experiments'))

from adaIN.model import vgg as original_vgg, decoder as original_decoder
from experiments.style_transfer import NSTTransform

def load_models(device):
    vgg = original_vgg
    decoder = original_decoder
    vgg.load_state_dict(torch.load(f'{BASE_PATH}/experiments/adaIN/vgg_normalised.pth', map_location=device))
    vgg = torch.nn.Sequential(*list(vgg.children())[:31])
    decoder.load_state_dict(torch.load(f'{BASE_PATH}/experiments/adaIN/decoder.pth', map_location=device))
    return vgg.to(device).eval(), decoder.to(device).eval()

def stylize_images(dataset_name, alpha_min, alpha_max, probability, num_to_sample, pixels, device):
    print(f"Loading NPZ and sampling {num_to_sample} images...")
    npz_map = {
        'c10': '/kaggle/input/1m-cifar10/1mcifar10.npz',
        'c100': '/kaggle/input/1m-cifar100/1mcifar100.npz',
        'tin': '/kaggle/input/1m-tiny/tiny_edm_1m.npz'
    }
    npz_path = npz_map[dataset_name]
    save_dir = f'/kaggle/working/styled_fid_images/{dataset_name}_{num_to_sample}'
    os.makedirs(save_dir, exist_ok=True)

    data = np.load(npz_path)
    all_images = data['image']
    indices = torch.randperm(len(all_images))[:num_to_sample]
    sampled_images = all_images[indices.numpy()]

    images_tensor = torch.from_numpy(sampled_images).permute(0, 3, 1, 2).float() / 255.0
    loader = DataLoader(TensorDataset(images_tensor), batch_size=128, shuffle=False)

    print("Stylizing images...")
    vgg, decoder = load_models(device)
    style_feats = torch.from_numpy(np.load(STYLE_FEATS_PATH)).to(device)

    nst = NSTTransform(
        style_feats=style_feats,
        vgg=vgg,
        decoder=decoder,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        probability=probability,
        pixels=pixels
    )

    with torch.no_grad():
        count = 0
        for batch in tqdm(loader):
            x = batch[0].to(device)
            y = nst(x).cpu()
            y = (y * 255).clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()
            for img in y:
                Image.fromarray(img).save(os.path.join(save_dir, f"{count:06d}.png"))
                count += 1

    print(f"Saved {count} stylized images to: {save_dir}")
    return save_dir

def extract_training_set(dataset_name):
    print("Extracting real training images...")
    output_dir = f"/kaggle/working/{dataset_name}-training"
    os.makedirs(output_dir, exist_ok=True)

    if dataset_name == "c10":
        dataset = CIFAR10(root='/kaggle/working', train=True, download=True)
    elif dataset_name == "c100":
        dataset = CIFAR100(root='/kaggle/working', train=True, download=True)
    elif dataset_name == "tin":
        dataset = ImageFolder(root="/kaggle/input/tinyimagenet/tiny-imagenet-200/train", transform=None)
    else:
        raise ValueError("Invalid dataset name")

    for idx, (img, _) in tqdm(enumerate(dataset), total=len(dataset)):
        Image.fromarray(np.array(img)).save(os.path.join(output_dir, f"{idx:06d}.png"))

    print(f"Saved real training set to: {output_dir}")
    return output_dir

def calculate_ref_stats(real_train_dir, dataset_name):
    ref_out = f"/kaggle/working/{dataset_name}-training-ref.npz"
    print(f"Calculating FID reference stats...")
    os.system(f"python {EDM_FID_SCRIPT} ref --data {real_train_dir} --dest {ref_out}")
    return ref_out

def calculate_fid(styled_dir, ref_path, dataset_name, num_to_sample):
    print(f"Calculating FID for {dataset_name}...")
    os.system(f"python {EDM_FID_SCRIPT} calc --images {styled_dir} "
              f"--ref {ref_path} --num {num_to_sample} --seed 42 --batch 64")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['c10', 'c100', 'tin'])
    parser.add_argument('--alpha_min', type=float, default=1.0)
    parser.add_argument('--alpha_max', type=float, default=1.0)
    parser.add_argument('--probability', type=float, default=1.0)
    parser.add_argument('--num_to_sample', type=int)
    parser.add_argument('--pixels', type=int)
    args = parser.parse_args()

    dataset = args.dataset
    alpha_min = args.alpha_min
    alpha_max = args.alpha_max
    probability = args.probability
    num_to_sample = args.num_to_sample if args.num_to_sample else (100000 if dataset == 'tin' else 50000)
    pixels = args.pixels if args.pixels else (64 if dataset == 'tin' else 32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    styled_dir = stylize_images(dataset, alpha_min, alpha_max, probability, num_to_sample, pixels, device)
    real_dir = extract_training_set(dataset)
    ref_path = calculate_ref_stats(real_dir, dataset)
    calculate_fid(styled_dir, ref_path, dataset, num_to_sample)

if __name__ == "__main__":
    main()
