# %%
import glob
from itertools import chain
import os
import random
import zipfile
import time
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from linformer import Linformer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
#from tqdm.notebook import tqdm


#from vit_pytorch.efficient import ViT
from vit_pytorch import ViT

import argparse
arg_parser = argparse.ArgumentParser() 
arg_parser.add_argument('--image_path',
                            type=str,
                            default='')
arg_parser.add_argument('--cpu_or_gpu',
                            type=str,
                            default='cpu')
arg_parser.add_argument('--label_type',
                            type=str,
                            default='PHI')
arg_parser.add_argument('--input_model_path',
                            type=str,
                            default='')
arg_parser.add_argument('--output_ana_path',
                            type=str,
                            default='')
      
args = arg_parser.parse_args()
print(args)


#
# Nethod to convert path name containing slope to the actual slope
def get_slope_from_path(path):
#
# Extract the part of the filesname which contains the slope (which is in degrees*100)
    label_theta = path.split('THETA')[1]
    label_phi = path.split('PHI')[1]

    if args.label_type == 'PHI':
        label = label_phi
    else:
        label = label_theta
#
# Now normalize (take out the factor of 100, then dived by max which is 180 degrees)
    label = int(label)/100.0/180.0
#
    return label

#
# Utility function to go from normalized value back to degrees
def convert_to_degrees(val):
    val_conv = val*180.0
    return val_conv

# %%
print(f"Torch: {torch.__version__}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device,cpu_or_gpu",device,args.cpu_or_gpu)
if args.cpu_or_gpu == 'cpu':
    new_model = torch.load(args.input_model_path, map_location=torch.device('cpu'))
elif args.cpu_or_gpu == 'gpu':
    #device = 'cuda'
    new_model = torch.load(args.input_model_path)
else:
    print("neither cpu or gpu requested!")
    exit()



# %%
# Image augmentation
train_transforms = transforms.Compose(
    [
        transforms.Resize((96, 256)),
        #transforms.RandomResizedCrop(100),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize((96, 256)),
        #transforms.CenterCrop(100),
        transforms.ToTensor(),
    ]
)


test_transforms = transforms.Compose(
    [
        transforms.Resize((96, 256)),
        #transforms.CenterCrop(100),
        transforms.ToTensor(),
    ]
)


def delta_phi_tensor(output, target):
    new_output = convert_to_degrees(output.clone())*math.pi/180.0
    new_target = convert_to_degrees(target.clone())*math.pi/180.0
    dphi = torch.atan2(torch.sin(new_output - new_target), torch.cos(new_output - new_target))
    dphi *= 180.0/math.pi
    return dphi


def delta_phi(output, target):
    dphi = math.atan2(math.sin(output*math.pi/180.0 - target*math.pi/180.0), math.cos(output*math.pi/180.0 - target*math.pi/180.0))
    dphi *= 180.0/math.pi
    return dphi


# %%
# Load data
class SlopeDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        #print('img shape',img.size)
        img_transformed = self.transform(img)
#
# Extract the part of the filesname which contains the slope (which is in degrees*100)
        label = get_slope_from_path(img_path)

        return img_transformed, label
#
image_list = glob.glob(args.image_path)

batch_size=16

image_data = SlopeDataset(image_list, transform=train_transforms)
image_loader = DataLoader(dataset = image_data, batch_size=batch_size, shuffle=False )
#test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True)
print('len(image_data), len(image_loader)', len(image_data), len(image_loader))
# %%

rows = []
t0 = time.time()
for data, label in image_loader:
    data = data.to(device)
    label = label.to(device).float()
#
# This step is necessary when #output classes=1 (like for regression) 
# so that output and label have samne dimension
    label = label.unsqueeze(1)
    #print("data",data.size())
    output = new_model(data).float()
#
# Convert both back from normalized result to degrees
    label = convert_to_degrees(label)
    output = convert_to_degrees(output)
    #dphi = delta_phi_tensor(output, label)
    print("delta-t",time.time()-t0)
    t0 = time.time()
    for o,l in zip(output,label):
        o = o.item()
        if o<0.0:
            o += 360.0
        l = l.item()
        dp = delta_phi(o,l)
        print("o,l,dphi:",o,l,dp)
        rows.append({'output':o,'label':l,'dphi':dp})
#
df = pd.DataFrame(rows)
df.to_csv(args.output_ana_path,index=False)
print("done")