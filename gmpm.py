#############################################################################################################

# Written by Eyal Betzalel 20.1.2021

#############################################################################################################

# In this section I will load generated dataset with NLL values and return unified table for all of them

# Inputs :
# 1. ImageGPT file path
# 2. VDVAE file path
# 3. same_model_compare_flag - if true two path are in the same format and belong to same model

# Output :
#1. Unified table for all results

#############################################################################################################

# import :

import os
import numpy as np
import pickle
import argparse
from tqdm import tqdm
import h5py
# args :

parser = argparse.ArgumentParser()

parser.add_argument(
    "--imagegpt_res",
    type=str,
    help="Path to ImageGPT results"
)

parser.add_argument(
    "--vdvae_data",
    type=str,
    help="Path to VDVAE data"
)

parser.add_argument(
    "--vdvae_res",
    type=str,
    help="Path to VDVAE results"
)

parser.add_argument(
    "--data-type",
    type=str,
    help="Apply on test or train dataset"
)

args = parser.parse_args()

def load_file(path):

    if path.endswith('.p'):
        path = open(path, "rb")
        model_table = pickle.load(path)

    elif path.endswith('.npy'):

        model_table = np.load(path)
        nll_vec = model_table[:, 1024].tolist()
        dataset = model_table[:, :1024].tolist()

        model_table = []
        for i in range(len(nll_vec)):
            model_table.append((np.asarray(dataset[i]), nll_vec[i]))
    elif path.endswith('.h5'):
        data = []
        flagOneFile = 0
        with h5py.File(path, "r") as f:
            a_group_key = list(f.keys())[0]
            # Get the data
            temp = list(f[a_group_key])
            data.append(temp[1:])
            flagOneFile = 1
        data_flat = [item for sublist in data for item in sublist]
        data_flat = np.stack(data_flat, axis=0)
        precent_train_test_split = 0.7
        train = data_flat[:int(np.floor(precent_train_test_split * data_flat.shape[0])), :]
        np.random.seed(42)
        tr_va_split_indices = np.random.permutation(train.shape[0])
        if args.data_type == 'test':
            model_table = train[tr_va_split_indices[-5000:]]
        else:
            model_table = train[tr_va_split_indices[:-5000]]
    return model_table


def add_element(dict, key, value):
    if key not in dict:
        dict[key] = []
    dict[key].append(value)


def kl_divergence(p, q):
    return (1.0 / len(p)) * sum(np.log(p[i] / q[i]) for i in range(len(p)))


def tot_var_dist(p, q):
    return (1.0 / len(p)) * sum(0.5 * np.abs((p[i] / q[i]) - 1) for i in range(len(p)))


def js_divergence(p, q):
    m = [0.5 * (p[i] + q[i]) for i in range(len(p))]
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


nll2prob = lambda a: np.exp(-1 * a)/1024

# Import datasets:

imagegpt_dataset = load_file(args.imagegpt_res)
vdvae_data = load_file(args.vdvae_data)
import ipdb; ipdb.set_trace()
for root, dirs, files in os.walk(args.vdvae_res):
    for file in files:
        
        if file.endswith(".p"):

            print(" Working on : %s" % file)

            # Import current model evaluation results:

            print(" == Load data ==")

            vdvae_res = load_file(os.path.join(root, file))

            # Join data to NLL results :

            model_vdvae = []
            model_vdvae = [(vdvae_data[i], vdvae_res[i]) for i in range(len(vdvae_data))]

            # Sort ImageGPT and vdvae results in the same order --> Create hashmap :

            d = {}

            print(" == Creating hashmaps ==")

            for i in tqdm(range(len(model_vdvae))):
                #key = tuple(map(tuple, model_vdvae[i][0].astype(np.int)))
                key = tuple(model_vdvae[i][0].astype(np.int))
                add_element(d, key, i)

            sorted_list = []
            [sorted_list.append(0) for _ in range(len(model_vdvae))]

            # Check if lists are coordinated and sort ImageGPT list:

            print(" == Sorting ImageGPT ==")
            for i in tqdm(range(len(imagegpt_dataset))):
                #curr = np.expand_dims(imagegpt_dataset[i][0], axis=0)
                key = tuple(imagegpt_dataset[i][0].astype(np.int))
                value = d.get(key)

                if value == None:
                    continue

                value = value[0]
                sorted_list[value] = (imagegpt_dataset[i][0], imagegpt_dataset[i][1], model_vdvae[value][1].cpu())
            # Calc f - divergence  :
            
            # Clip non-regularity for calculation stability
            for j in range(len(sorted_list)):
                if sorted_list[j][2]>20:
                    sorted_list[j] = (sorted_list[j][0],sorted_list[j][1], 20)
                elif sorted_list[j][2]<0:
                    sorted_list[j] = (sorted_list[j][0],sorted_list[j][1], 0.1)
                

            p = []
            q = []

            print(" == Calculating f div ==")

            for i in tqdm(range(len(sorted_list))):
                if sorted_list[i] == 0:
                    continue
                p.append(nll2prob(sorted_list[i][1]))
                q.append(nll2prob(sorted_list[i][2]))
            
            kl_d = kl_divergence(p, q)
            rev_kl_d = kl_divergence(q, p)
            tot_var_dist_res = tot_var_dist(p, q)
            js_divergence_res = js_divergence(p, q)
            curr_run_str = file[:-2]

            # Saving results to text file :

            file_name = curr_run_str + ".txt"
            file_name = os.path.join(root, file_name)
            file = open(file_name,"a")

            curr_run_str = curr_run_str
            kl_d_str = "kl_d = " + str(kl_d)
            rev_kl_d_str = "rev_kl_d = " + str(rev_kl_d)
            tot_var_dist_str = "tot_var_dist = " + str(tot_var_dist_res)
            js_divergence_str = "js_divergence = " + str(js_divergence_res)

            file.write('%r\n%r\n%r\n%r\n%r\n' % (curr_run_str, kl_d_str, rev_kl_d_str, tot_var_dist_str, js_divergence_str))
            file.close()