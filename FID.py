import subprocess
import pickle

dict_is = {'medium_model_lower_lr':{}}
for model_type in ['medium_model_lower_lr']:
    for i in [0,1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]:
        fname = './samples_' + model_type + '/model_epoch_' + str(i)
        result = subprocess.run(['python','-m', 'pytorch_fid', './save', fname], stdout=subprocess.PIPE)
        (dict_is[model_type])[i] = result.stdout

print("Calculating FID...")
print(dict_is)
pickle.dump(dict_is, open('FID_for_medium_models.p', "wb"))