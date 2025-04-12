# ============================
# FILE: fid/run_fid.py
# ============================

import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Run full FID pipeline")

    # CLI flags passed from the user
    parser.add_argument('--dataset', required=True, choices=['c10', 'c100', 'tin'])
    parser.add_argument('--alpha_min', type=float, default=1.0)
    parser.add_argument('--alpha_max', type=float, default=1.0)
    parser.add_argument('--probability', type=float, default=1.0)
    parser.add_argument('--num', type=int, help='Number of images to sample')
    parser.add_argument('--pixels', type=int, help='Image size (32 or 64)')
    parser.add_argument('--run_fid', action='store_true', help='Trigger FID computation')

    args = parser.parse_args()

    # Build command string to pass args to the actual pipeline
    cmd = f"python /kaggle/working/model-based-data-augmentation/fid/fid_runner.py --dataset {args.dataset} --alpha_min {args.alpha_min} --alpha_max {args.alpha_max} --probability {args.probability}"

    if args.num:
        cmd += f" --num_to_sample {args.num}"
    if args.pixels:
        cmd += f" --pixels {args.pixels}"
    if args.run_fid:
        cmd += f" --run_fid"

    print(f"[INFO] Running pipeline with: {cmd}")
    os.system(cmd)

if __name__ == "__main__":
    main()