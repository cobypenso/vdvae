import numpy as np
import imageio
import os
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from data import set_up_data
import utils
from train_helpers import set_up_hyperparams, load_vaes, load_opt, accumulate_stats, save_model, update_ema
import matplotlib.pyplot as plt

def sample(H, vae, ema_vae, logprint):
    fname = "sample2.png"
    mb = 8
    batches = []
    for t in [1.0] * 10:
        batches.append(ema_vae.forward_uncond_samples(mb, t=t).cpu().view(-1, 32, 32))
    n_rows = len(batches)
    im = utils.clusters_to_images(torch.from_numpy(np.concatenate(batches, axis=0))).permute(0,2,3,1)
    im = im.reshape((n_rows, mb, 32, 32, 3)).permute(0, 2, 1, 3, 4).reshape([n_rows * 32, mb * 32, 3])
    logprint(f'printing samples to {fname}')
    imageio.imwrite(fname, im)
    
def main():
    H, logprint = set_up_hyperparams()
    H, data_train, data_valid_or_test, preprocess_fn = set_up_data(H)
    
    vae, ema_vae = load_vaes(H, logprint)
    sample(H, vae, ema_vae, logprint)

if __name__ == "__main__":
    main()