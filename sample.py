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

def show_samples(samples):
    out = utils.tile_images(samples)
    plt.imsave("samples.png", out)

def sample(H, vae, ema_vae, logprint):
    samples = ema_vae.forward_uncond_samples(8, t=None)
    show_samples(samples)
    
def main():
    H, logprint = set_up_hyperparams()
    H, data_train, data_valid_or_test, preprocess_fn = set_up_data(H)
    
    vae, ema_vae = load_vaes(H, logprint)
    sample(H, vae, ema_vae, logprint)

if __name__ == "__main__":
    main()