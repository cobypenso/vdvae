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
from vae import VAE
from train_helpers import restore_params

def sample(H, vae, logprint, epoch):
    fname = "./samples_" + H.model_type + "/model_epoch_" + str(epoch) + "/"
    # The choice of 100 batch size and 100 iteration is arbitrary, as long as at the end
    #  we end up with 10000 samples
    mb = 10
    batches = []
    for i in range(1000):
        samples = vae.forward_uncond_samples(mb)
        if samples is None:
            while (samples is None):
                samples = vae.forward_uncond_samples(mb)        
        
        samples = samples.cpu().view(-1, 32, 32)
        # Save here the sampled images. #
        for j in range(mb):
            im = utils.clusters_to_images(samples[j]).squeeze().permute(1,2,0)
            imageio.imwrite(fname + "pic_" + str(i * mb + j) + ".png", im)
        # ----------------------------- #
    

def load_trained_vaes(H, logprint, epoch):
    vae = VAE(H)
    model_type = H.model_type

    restore_path = "./saved_models/" + model_type + "/test/epoch_" + str(epoch) + "_-model.th"

    logprint(f'Restoring vae from {restore_path}')
    restore_params(vae, restore_path)

    vae = vae.cuda()

    if len(list(vae.named_parameters())) != len(list(vae.parameters())):
        raise ValueError('Some params are not named. Please name all params.')
    total_params = 0
    for name, p in vae.named_parameters():
        total_params += np.prod(p.shape)
    logprint(total_params=total_params, readable=f'{total_params:,}')
    return vae

def main():
    H, logprint = set_up_hyperparams()
    H, data_train, data_valid_or_test, preprocess_fn = set_up_data(H)
    
    # Load the models from various epochs #
    epochs = [0,1,5,10,15,20,25,30,35,40,45,50]
    
    for epoch in epochs:
        vae = load_trained_vaes(H, logprint, epoch)
        sample(H, vae, logprint, epoch)

if __name__ == "__main__":
    main()