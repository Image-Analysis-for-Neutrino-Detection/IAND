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


import os, psutil
def printmem():
    process = psutil.Process(os.getpid())
    print("   memory:",round(process.memory_info().rss/(10**9),3),'Gbytes')  # in bytes 


#from vit_pytorch.efficient import ViT
from vit_pytorch import ViT

import argparse
arg_parser = argparse.ArgumentParser() 
arg_parser.add_argument('--INPUT_DIR',
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
train_labels = []
train_images =[]
test_labels = []
test_images =[]
num = 0
print("Start reading training data")
t0 = time.time()
num_train = 0
num_test = 0
fnames = glob.glob(args.INPUT_DIR+'/*.pkl')
train_rows = []
test_rows = []
for fname in sorted(fnames):
    df = pd.read_pickle(fname)
    print(fname,df.columns)
    for row in df.itertuples():
        num += 1
        run = row.run
        event = row.event
        phi = int(100*row.interaction_phi)
        theta = int(100*row.interaction_eta)
        #print(phi,row.interaction_phi,row.image.shape)
        final_image = row.image[0:256,:]
        #break
# Normalize the image with values between 0 and 255
        normalized_image = (final_image - np.min(final_image)) * (255.0 / (np.max(final_image) - np.min(final_image)))/255.0
        normalized_image = normalized_image.astype('float32')

# Convert the normalized numpy array to an image using Pillow
        image = normalized_image # Image.fromarray(normalized_image)
        image = np.resize(image,(1,256,96))
        if args.label_type == 'PHI':
            label = row.interaction_phi/360.0
        elif args.label_type == 'THETA':
            label = row.interaction_eta/360.0
        else:
            print("incoreect label type:",args.label)
            exit()
#
        digit = str(event)[-1]
        if num<20:
            print("run,event,digit",run,event,digit,image.shape)
        if digit == '1':
            test_rows.append({'label':label,'image':image})
            #test_labels.append(label)
            #test_images.append(image)
        elif digit == '2':
            train_rows.append({'label':label,'image':image})
            #train_labels.append(label)
            #train_images.append(image)

    print('processed ',num,time.time()-t0)
    printmem()
    t0 = time.time()
    if len(train_rows)>5000:
        break

train_df = pd.DataFrame(train_rows)
test_df = pd.DataFrame(test_rows)

print("Number training samples:",len(train_df))
print("Number testing samples:",len(test_df))
printmem()


def MSE_deltaphi2(output, target):
#    dphi = torch.atan2(torch.sin(output*360*math.pi/180.0 - target*360*math.pi/180.0), torch.cos(output*360*math.pi/180.0 - target*360*math.pi/180.0))
#    dphi = torch.modulus(dphi,2.0*math.pi)
    loss = 1.0 - torch.cos((output - target)*(2.0*math.pi))
    loss = torch.mean(loss)
#    loss = torch.mean((loss)**2)
    return loss

def my_dphi(output, target): 
    cossim = torch.cos((output - target)*(2.0*math.pi))
    dphi = torch.acos(cossim)*180/math.pi
    cossim = 1.0 - cossim
    return dphi,cossim
 
def MSE_deltaphi(output, target):
#    dphi = torch.atan2(torch.sin(output*360*math.pi/180.0 - target*360*math.pi/180.0), torch.cos(output*360*math.pi/180.0 - target*360*math.pi/180.0))
#    dphi = torch.modulus(dphi,2.0*math.pi)
    dphi = (output - target)
#    dphi = (dphi - 180) / 180.0
#    dphi = (torch.remainder(dphi+180,360.0) - 180.0)/180.0
    loss = torch.mean((dphi)**2)
    return loss

def MSE_mine(output, target):
#    dphi = torch.atan2(torch.sin(output*360*math.pi/180.0 - target*360*math.pi/180.0), torch.cos(output*360*math.pi/180.0 - target*360*math.pi/180.0))
#    dphi = torch.modulus(dphi,2.0*math.pi)
    loss = torch.mean((output - target)**2)
    return loss

from torch.utils.data import Dataset, DataLoader

# create custom dataset class
class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __getitem__(self, index):
        row = self.dataframe.iloc[index].to_numpy()
        #features = row[5:]
        #label = row[2]
        features = torch.from_numpy(np.array(row[1]))
        label = torch.from_numpy(np.array(row[0]))
        return features, label

    def __len__(self):
        return len(self.dataframe)

# define data set object
train_dataset = CustomDataset(dataframe=train_df)
test_dataset = CustomDataset(dataframe=test_df)
printmem()
# create DataLoader object of DataSet object
workers = 0
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True,num_workers=workers)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True,num_workers=workers)
printmem()


epoch_loss = 0
num_loss = 0
epoch_val_loss = 0
num_val_loss = 0
num_good_val = 0
num_all_val = 0

rows = []
t0 = time.time()
data_type = 'train'
with torch.no_grad():
    first = True
    step = 0
    ts = time.time()
    for sample in train_dataloader:
        step += 1
        if step % 100 == 0:
            print("   step:",step,time.time()-ts)
            ts = time.time()
        data = sample[0]
        label = sample[1]

        data = data.to(device)
        label = label.to(device).float()
        label = label.unsqueeze(1)

        val_output = new_model(data)
        val_loss = MSE_deltaphi2(val_output, label)
        for o,l in zip(val_output,label):
            num_all_val += 1
            dphi,cossim = my_dphi(o,l)
            o = o.item()*360
            if o<0.0:
                o += 360.0
            l = l.item()*360
            rows.append({'data_type':data_type,'output':o,'label':l,'dphi':dphi.item(),'cossim':cossim.item()})

t0 = time.time()
data_type = 'test'
with torch.no_grad():
    epoch_val_accuracy = 0
    epoch_val_loss = 0
    num_val_loss = 0
    num_good_val = 0
    num_all_val = 0
    first = True
    step = 0
    ts = time.time()
    for sample in test_dataloader:
        step += 1
        if step % 100 == 0:
            print("   step:",step,time.time()-ts)
            ts = time.time()
        data = sample[0]
        label = sample[1]

        data = data.to(device)
        label = label.to(device).float()
        label = label.unsqueeze(1)

        val_output = new_model(data)
        val_loss = MSE_deltaphi2(val_output, label)
        for o,l in zip(val_output,label):
            num_all_val += 1
            dphi,cossim = my_dphi(o,l)
            o = o.item()*360
            if o<0.0:
                o += 360.0
            l = l.item()*360
            rows.append({'data_type':data_type,'output':o,'label':l,'dphi':dphi.item(),'cossim':cossim.item()})

#
df = pd.DataFrame(rows)
df.to_csv(args.output_ana_path,index=False)
print("done")