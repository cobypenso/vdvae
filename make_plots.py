import pickle
import matplotlib.pyplot as plt

# Load IS #
small_is = pickle.load(open('./samples_small_model_lower_lr/Inception_Score_for_small_models.p','rb'))
medium_is = pickle.load(open('./samples_medium_model_lower_lr/Inception_Score_for_medium_models.p','rb')) 
# Load FID #
small_fid = pickle.load(open('./samples_small_model_lower_lr/FID_for_small_models.p','rb')) 
medium_fid = pickle.load(open('./samples_medium_model_lower_lr/FID_for_medium_models.p','rb')) 

# Load Distances #
small_distance = pickle.load(open('small_distance_metrics.p','rb'))
medium_distance = pickle.load(open('medium_distance_metrics.p','rb')) 

epochs = [1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]

# KL Plotting #
kl = []
for i in epochs:
    kl.append(small_distance[str(i)]['kl'])
plt.plot(epochs, kl, label = 'small', marker = 'o')
kl = []
for i in epochs:
    kl.append(medium_distance[str(i)]['kl'])
plt.plot(epochs, kl, label = 'medium', marker = 'o')

plt.xlim(0,101)
plt.ylim(3,5)
plt.xlabel('Epoch')
plt.ylabel('KL Divergence')
plt.legend()
plt.savefig("KL_summary.png")

# IS Plotting #
plt.clf()
IS = []
for i in epochs:
    IS.append(small_is['small_model_lower_lr'][i])
plt.plot(epochs, IS, label = 'small', marker = 'o')
IS = []
for i in epochs:
    IS.append(medium_is['medium_model_lower_lr'][i])
plt.plot(epochs, IS, label = 'medium', marker = 'o')
plt.savefig("IS_summary.png")

# FID Plotting #
plt.clf()
FID = []
for i in epochs:
    FID.append(float((small_fid['small_model_lower_lr'][i])[6:-1]))
plt.plot(epochs, FID, label = 'small', marker = 'o')
FID = []
for i in epochs:
    FID.append(float((medium_fid['medium_model_lower_lr'][i])[6:-1]))
plt.plot(epochs, FID, label = 'medium', marker = 'o')

# KL with different seeds
# Load Distances #
seed_0 = pickle.load(open('small_seed_0_distance_metrics.p','rb'))
seed_10 = pickle.load(open('small_seed_10_distance_metrics.p','rb'))
seed_20 = pickle.load(open('small_seed_20_distance_metrics.p','rb'))
seed_30 = pickle.load(open('small_seed_30_distance_metrics.p','rb'))
seed_40 = pickle.load(open('small_seed_40_distance_metrics.p','rb'))

epochs = [1,5,10,15,20,25,30,35,40]
# KL Plotting #
plt.clf()
kl = []
for i in epochs:
    kl.append(seed_0[str(i)]['kl'])
plt.plot(epochs, kl, label = 'seed 0', marker = 'o')

kl = []
for i in epochs:
    kl.append(seed_10[str(i)]['kl'])
plt.plot(epochs, kl, label = 'seed 10', marker = 'o')

kl = []
for i in epochs:
    kl.append(seed_20[str(i)]['kl'])
plt.plot(epochs, kl, label = 'seed 20', marker = 'o')
kl = []
for i in epochs:
    kl.append(seed_30[str(i)]['kl'])
plt.plot(epochs, kl, label = 'seed 30', marker = 'o')
kl = []
for i in epochs:
    kl.append(seed_40[str(i)]['kl'])
plt.plot(epochs, kl, label = 'seed 40', marker = 'o')

plt.xlim(0,41)
plt.ylim(3.3,5)
plt.xlabel('Epoch')
plt.ylabel('KL Divergence')
plt.legend()
plt.savefig("KL_seeds_summary.png")

plt.xlim(0,101)
plt.ylim(25, 100)
plt.xlabel('Epoch')
plt.ylabel('FID')
plt.legend()
plt.savefig("FID_summary.png")