import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from data import set_up_data
import utils
from train_helpers import set_up_hyperparams, load_vaes, load_opt, accumulate_stats, save_model, update_ema


def iwae_calc(x, H, ema_vae, logprint, K = 200):
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
        stats = ema_vae.forward(x[None, :], x[None, :])
        sum += torch.exp(stats['elbo'])
        
    iwae = torch.log(sum / K)
    stats = ema_vae.forward(x[None, :], x[None, :])
    print('ELBO: ', stats['elbo'], 'vs. IWAE: ', iwae)

    return iwae

def elbo_calc(x, H, ema_vae, logprint, K = 200):
    '''
        Pass image through the VAE and calculate the log likelihood 
        with ELBO approximation

        @param: K - number of samples for elbo computation --> the bigger the better(law of big numbers)
        @param: x - the image to compute iwae on
        @param: ema_vae - vae model
    '''
    x = x[0]
    sum = 0
    
    for i in range(K):
        stats = ema_vae.forward(x[None, :], x[None, :])
        sum += stats['elbo']
        
    elbo = sum / K

    return elbo

def iwae_calc_manual(x, H, ema_vae, logprint, K = 200):
    x = x[0]
    sum = 0
    
    for i in range(K):
        # Get the latents and stats
        stats = ema_vae.forward_get_latents(x)
        log_q_zGx = [s['log_q_zGx'] for s in stats]
        log_p_z = [s['log_p_z'] for s in stats]
    
        # Get the ouput of the decoder given specific latents
        mb = 1
        zs = [s['z'].cuda() for s in stats]
        xhat, p_xGz = ema_vae.forward_samples_set_latents(mb, zs, t=1.0, prob_vector=True)
    
        #calculate log_p_xGz
        log_p_xGz = torch.mean(torch.log(p_xGz).gather(1, xhat[:,None,:,:]))

        # add all the log_q_zGx to form the total log_q_zGx, same for log_p_z
        # calculate log_p_z
        log_p_z = torch.mean(torch.Tensor([torch.mean(p) for p in log_p_z]))

        # calculate log_q_zGx
        log_q_zGx = torch.mean(torch.Tensor([torch.mean(q) for q in log_q_zGx]))

        #calculate (p_xGz * p_z / q_zGx) by calculating exp(log_p_xGz + log_p_z - log_zGx)
        iwae_one_sample = torch.exp(log_p_xGz + log_p_z - log_q_zGx)
        print (iwae_one_sample)
        sum += iwae_one_sample
    
    iwae = -torch.log(sum/K)
    return iwae
    
def main():
    H, logprint = set_up_hyperparams()
    H, data_train, data_valid_or_test, preprocess_fn = set_up_data(H)
    
    # -- Calculate Log{P(x)} for every x in the dataset -- #
    dict_x_nll = {}
    vae, ema_vae = load_vaes(H, logprint)

    count = 0 
    total_iter = 0

    for idx, x in enumerate(data_train):
        iwae = iwae_calc(x, H, ema_vae, logprint)
        elbo = elbo_calc(x, H, ema_vae, logprint)

        if iwae <= elbo:
            count += 1
        total_iter += 1

        dict_x_nll[x] = iwae

        if total_iter % 10 == 0:
            print(count / total_iter * 100,'%')
    
    # -- Save results to file -- #
    
    
if __name__ == "__main__":
    main()