import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from data import set_up_data
import utils
from train_helpers import set_up_hyperparams, load_vaes, load_opt, accumulate_stats, save_model, update_ema
import pickle
from vae import VAE
from train_helpers import restore_params

def iwae_calc(x, H, ema_vae, logprint, K = 200):
    '''
        Pass image through the VAE and calculate the log likelihood 
        more accurently than the ELBO estimator -> IWAE

        @param: K - number of samples for iwae computation --> the bigger the better(accuracy wise)
        @param: x - the image to compute iwae on
        @param: ema_vae - vae model
    '''
    x = x[0].cuda()
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
    x = x[0].cuda()
    sum = 0
    
    for i in range(K):
        stats = ema_vae.forward_with_sum_nll(x[None, :], x[None, :])
        sum += stats['elbo']
        
    elbo = sum / K

    return elbo

def iwae_calc_manual(x, H, ema_vae, logprint, K = 200):
    x = x[0].cuda()
    sum = 0
    
    for i in range(K):
        # Get the latents and stats
        stats = ema_vae.forward_get_latents(x)
        log_q_zGx = [s['log_q_zGx'] for s in stats]
        log_p_z = [s['log_p_z'] for s in stats]
    
        # Get the ouput of the decoder given specific latents
        mb = 1
        zs = [s['z'].cuda() for s in stats]
        xhat, log_p_xGz, px_z = ema_vae.forward_samples_set_latents(mb, zs, t=1.0, prob_vector=True, x = x)
        
        #calculate log_p_xGz
        # x_as_ind = x.to(dtype=torch.int64)
        # log_p_xGz = torch.mean(torch.log(px_z).gather(1, x_as_ind.reshape(xhat.shape)[:,None,:,:]))
        # add all the log_q_zGx to form the total log_q_zGx, same for log_p_z
        # calculate log_p_z
        count = len(log_p_z)
        for i in log_p_z:
            count += np.prod(i.shape)
        log_p_z = torch.sum(torch.Tensor([torch.sum(p) for p in log_p_z])) / 97
        count = len(log_q_zGx)
        for i in log_q_zGx:
            count += np.prod(i.shape)
        log_q_zGx = torch.sum(torch.Tensor([torch.sum(q) for q in log_q_zGx])) / 97

        #calculate (p_xGz * p_z / q_zGx) by calculating exp(log_p_xGz + log_p_z - log_zGx)
        iwae_one_sample = torch.exp(log_p_xGz + log_p_z - log_q_zGx)
        sum += iwae_one_sample
    
    iwae = -torch.log(sum/K)
    return iwae


def load_trained_vaes(H, logprint, epoch):
    vae = VAE(H)
    model_type = H.model_type
    if model_type != 'large' and model_type != 'larger' and model_type != 'medium':
        logprint(f'No such model type!!')
        raise Exception('No such model type!')

    restore_path = "./saved_models/" + model_type + "_model/test/epoch_" + str(epoch) + "_-model.th"
    restore_path_ema = "./saved_models/" + model_type + "_model/test/epoch_" + str(epoch) + "_-model-ema.th"

    logprint(f'Restoring vae from {restore_path}')
    restore_params(vae, restore_path)

    ema_vae = VAE(H)
    logprint(f'Restoring ema vae from {restore_path_ema}')
    restore_params(ema_vae, restore_path_ema)
    ema_vae.requires_grad_(False)

    vae = vae.cuda()
    ema_vae = ema_vae.cuda()

    if len(list(vae.named_parameters())) != len(list(vae.parameters())):
        raise ValueError('Some params are not named. Please name all params.')
    total_params = 0
    for name, p in vae.named_parameters():
        total_params += np.prod(p.shape)
    logprint(total_params=total_params, readable=f'{total_params:,}')
    return vae, ema_vae

def main():
    H, logprint = set_up_hyperparams()
    H, data_train, data_valid_or_test, preprocess_fn = set_up_data(H)
    
    # Load the models from various epochs #
    epochs = [250, 225, 200, 175, 150, 125, 100, 75, 50, 25, 0]
    
    for epoch in epochs:
        vae, ema_vae = load_trained_vaes(H, logprint, epoch)
        # -- Calculate -Log{P(x)} for every x in the dataset -- #
        elbo_list = []
        for x in data_train:
            # iwae = iwae_calc_manual(x, H, ema_vae, logprint, K=1)
            elbo = elbo_calc(x, H, ema_vae, logprint, K=5)
            elbo_list.append(elbo)
        # -- Save results to file -- #
        fname = H.model_type + "_model_elbo_calc_epoch_" + str(epoch) +".p"
        pickle.dump(elbo_list, open(fname, "wb"))
        print ("Epoch calc ended")
        
if __name__ == "__main__":
    main()