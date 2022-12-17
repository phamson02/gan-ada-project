"""This module is fol calculation of datasets' statistics for FID evaluation"""

from utils.fid_score import calculate_activation_statistics
from argparse import ArgumentParser
import numpy as np
from utils.inception_score import InceptionV3
import torch
import os
import glob

def main(args):
    save_path = "datasets_stats"
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = InceptionV3([block_idx]).to(device)
    paths = glob.glob(os.path.join(args.path, "*.png"))
    if len(paths) == 0:
        paths = glob.glob(os.path.join(args.path, "*.jpg"))
    paths = paths[:int(len(paths)*args.portion)]
    
    assert len(paths) !=0
    
    mu, sigma = calculate_activation_statistics(paths, model, device=device)
    
    os.makedirs(save_path, exist_ok=True)
    np.savez(f"{save_path}/{args.dataset_name}.npz", mu=mu, sigma=sigma)
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-nm", "--dataset_name", default=None, type=str, help="name of the current dataset")
    parser.add_argument("-p", "--path", default=None, type=str, help="path to dataset dir")
    parser.add_argument("-pr", "--portion", default=1.0, type=float, help="portion of training data for calculating stats")
    # parser.add_argument("-sp", "--save_path", default=None, type=str, help="path for saving statistics")
    args = parser.parse_args()
    main(args)