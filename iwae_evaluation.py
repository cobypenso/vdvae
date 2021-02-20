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


def iwae_calc(x, H, ema_vae, logprint, K = 1):
    '''
        Pass image through the VAE and calculate the log likelihood 
        more accurently than the ELBO estimator -> IWAE

        @param: K - number of samples for iwae computation --> the bigger the better(accuracy wise)
        @param: x - the image to compute iwae on
        @param: ema_vae - vae model
    '''
    x = x[0]
    sum = 0
    
    for i in range(K):
        # Get the latents and stats
        stats = ema_vae.forward_get_latents(x)
        log_q_zGx = [s['log_q_zGx'].cpu() for s in stats]
        log_p_z = [s['log_p_z'].cpu() for s in stats]

        # add all the log_q_zGx to form the total log_q_zGx, same for log_p_z
        # calculate log_p_z
        log_p_z = torch.mean(torch.Tensor([torch.mean(p) for p in log_p_z]))

        # calculate log_q_zGx
        log_q_zGx = torch.mean(torch.Tensor([torch.mean(q) for q in log_q_zGx]))
        
        # Get the ouput of the decoder given specific latents
        mb = x.shape[0]
        zs = [s['z'].cuda() for s in stats]
        xhat, p_xGz = ema_vae.forward_samples_set_latents(mb, zs, t=1.0, prob_vector=True)
    
        #calculate log_p_xGz
        log_p_xGz = torch.mean(torch.log(p_xGz).gather(1, xhat[:,None,:,:]))

        #calculate (p_xGz * p_z / q_zGx) by calculating exp(log_p_xGz + log_p_z - log_zGx)
        iwae_one_sample = torch.exp(log_p_xGz + log_p_z - log_q_zGx)
        print (iwae_one_sample)
        sum += iwae_one_sample
        
    iwae = -torch.log(sum / K)
    stats = ema_vae.forward(x[None, :], x[None, :])
    print(stats['elbo'])
    print(iwae)

    return iwae
    
    
def main():
    H, logprint = set_up_hyperparams()
    H, data_train, data_valid_or_test, preprocess_fn = set_up_data(H)
    
    # -- Calculate Log{P(x)} for every x in the dataset -- #
    dict_x_nll = {}
    vae, ema_vae = load_vaes(H, logprint)
    for idx, x in enumerate(data_train):
        nll = iwae_calc(x, H, ema_vae, logprint)
        dict_x_nll[x] = nll
    
    # -- Save results to file -- #
    
    
if __name__ == "__main__":
    main()