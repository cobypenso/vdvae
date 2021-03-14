import argparse
import torch
from torchvision.datasets.vision import VisionDataset
from torchvision import datasets, transforms
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
from torchvision.models.inception import inception_v3
import numpy as np
from scipy.stats import entropy
from PIL import Image
import pickle

def make_dataset(dir):
    import os
    images = []
    d = os.path.expanduser(dir)

    if not os.path.exists(dir):
        print('path does not exist')

    for root, _, fnames in sorted(os.walk(d)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            images.append(path)
    return images 
    
class SamplesForlder(VisionDataset):
    def __init__(self, root, transform=None):
        #Call make_dataset to collect files. 
        self.samples = make_dataset(root)
        self.transform = transform
        
    def __getitem__(self, index: int):
        sample = self.samples[index]
        sample = self.loader(sample)
        
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.samples)
        
    def loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
        
    

def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]
        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

# args :

parser = argparse.ArgumentParser()

parser.add_argument(
    "--path",
    type=str,
    help="Path to ImageGPT results",
    default='./samples_medium/model_epoch_0/'
)

args = parser.parse_args()

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])
                                
dict_is = {'medium':{}, 'large':{}, 'larger':{}}
for model_type in ['medium', 'large', 'larger']:
    for i in [0,25,50,75,100,125,150,175,200,225,250]:
        fname = './samples_' + model_type + '/model_epoch_' + str(i)
        dataset = SamplesForlder(fname, transform=transform)
        mean_split_scores, std_split_scores = inception_score(dataset, cuda=True, batch_size=32, resize=True, splits=1)
        (dict_is[model_type])[i] = mean_split_scores

print("Calculating Inception Score...")
print(dict_is)
pickle.dump(dict_is, open('Inception_Score_for_all_models.p', "wb"))

"""Computes the inception score of the generated images imgs
imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
cuda -- whether or not to run on GPU
batch_size -- batch size for feeding into Inception v3
splits -- number of splits
"""