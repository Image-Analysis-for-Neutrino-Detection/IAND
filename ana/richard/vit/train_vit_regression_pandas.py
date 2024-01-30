
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

import tarfile
from io import BytesIO
from pathlib import Path
from typing import Union
from torchvision.transforms.functional import to_tensor

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
arg_parser.add_argument('--label_type',
                            type=str,
                            default='PHI')
arg_parser.add_argument('--output_model_path',
                            type=str,
                            default='')
      
args = arg_parser.parse_args()
print(args)



# %%
print(f"Torch: {torch.__version__}")
# Training settings
batch_size = 64
epochs = 100
lr = 3e-5
gamma = 0.7
seed = 42
resolution = 5.0
# %%
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed)
# %%
device = 'cuda'

#
# Utility function to go from normalized value back to degrees
def convert_to_degrees(val):
    val_conv = val*360.0
    if val_conv<0:
        val_conv += 360.0
    return val_conv


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
        else:
            train_rows.append({'label':label,'image':image})
            #train_labels.append(label)
            #train_images.append(image)

    print('processed ',num,time.time()-t0)
    printmem()
    t0 = time.time()
    # if len(train_rows)>50000:
    #     break
train_df = pd.DataFrame(train_rows)
test_df = pd.DataFrame(test_rows)

print("Number training samples:",len(train_df))
print("Number testing samples:",len(test_df))
printmem()

# %%
# Vision transformer
# model = ViT(
#     dim=128,
#     image_size=224,
#     patch_size=32,
#     num_classes=2,
#     transformer=efficient_transformer,
#     channels=3,
# )
# model = ViT(
#     image_size = 224,
#     patch_size = 32,
#     num_classes = 2,
#     dim = 128,
#     depth = 6,
#     heads = 16,
#     mlp_dim = 2048,
#     dropout = 0.1,
#     emb_dropout = 0.1
# ).to(device)

#
# Standard
# model = ViT(
#     image_size = (96, 256),
#     channels=1,
#     patch_size = 16,
#     num_classes = 1,
#     dim = 128,
#     depth = 12,
#     heads = 10,
#     mlp_dim = 128,
#     dropout = 0.1,
#     emb_dropout = 0.1
# ).to(device)

model = ViT(
    image_size = (256, 96),
    channels=1,
    patch_size = 16,
    num_classes = 1,
    dim = 256,
    depth = 12,
    heads = 16,
    mlp_dim = 256,
    dropout = 0.1,
    emb_dropout = 0.1
).to(device)

print("model",model)

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

# %%
# Training
# loss function: CrossEntropyLoss for classification, MSELoss for regression
#criterion = nn.CrossEntropyLoss()
#criterion = nn.MSELoss()
criterion = nn.MSELoss()
#criterion2 = nn.CosineSimilarity() 

#back-propagation on the above *loss* will try cos(angle) = 0. But I want angle between the vectors to be 0 or cos(angle) = 1.

# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

num_batches = int(len(train_labels)/batch_size)
print("Number of batches:",num_batches)
num_batches_test = int(len(test_labels)/batch_size)
print("Number of batches test:",num_batches_test)

epoch_val_loss_min = 1000000.0
time_epoch = time.time()
for epoch in range(epochs):
    epoch_loss = 0
    num_loss = 0
    epoch_accuracy = 0
    print("epoch",epoch)

#    for data, label in tqdm(train_loader):
    num_good = 0
    num_all = 0
    first = True
    step = 0
    ts = time.time()
    for sample in train_dataloader:
        step += 1
        if step % 100 == 0:
            print("   batch count:",step,time.time()-ts)
            ts = time.time()
        data = sample[0]
        label = sample[1]
        #label = torch.from_numpy(np.array(label))
        #print('data size',data.size())
        data = data.to(device)
        label = label.to(device).float()
#
# This step is necessary when #output classes=1 (like for regression) 
# so that output and label have samne dimension
        label = label.unsqueeze(1)
        #print("data",data.size())
        output = model(data).float()
        if first:
            print('output/label dimensions: ',output.size(),label.size())

        old_loss = criterion(output, label)
        my_loss = MSE_mine(output, label)
        loss = MSE_deltaphi2(output, label)
        #loss = torch.mean(torch.abs(criterion2(label*2.0*math.pi,output*2.0*math.pi)))
        #loas = 1.0-loss

        #loss = MSE_deltaphi(output, label)
        if num_all<64:
            print('loss,my_loss,mydphi_loss ',old_loss,my_loss,loss)
        this_num = 0
        for o,l in zip(output,label):
            num_all += 1
            this_num += 1
            if first and this_num<10:
                dphi,cossim = my_dphi(o,l)
                print("training output,pred:",convert_to_degrees(o).item(),convert_to_degrees(l).item(),dphi.item(),cossim.item())
            if abs(convert_to_degrees(o-l))<resolution:
                num_good += 1
        first = False
        num_good += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #acc = (output.argmax(dim=1) == label).float().mean()
        #epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss
        num_loss += 1
    epoch_loss /= num_loss
    print("   finished first loop",num_all,num_good )
    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        num_val_loss = 0
        num_good_val = 0
        num_all_val = 0
        first = True
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

            val_output = model(data)
            #val_loss = criterion(val_output, label)
            val_loss = MSE_deltaphi2(val_output, label)
            for o,l in zip(val_output,label):
                num_all_val += 1
                if first:
                    print("validation output,pred:",convert_to_degrees(o),convert_to_degrees(l))
                if abs(convert_to_degrees(o-l))<resolution:
                    num_good_val += 1
            first = False

            #acc = (val_output.argmax(dim=1) == label).float().mean()
            #epoch_val_accuracy += acc / len(valid_loader)
            epoch_val_loss += val_loss
            num_val_loss += 1
        epoch_val_loss /= num_val_loss
    print("   epoch time:",time.time()-time_epoch)
    time_epoch = time.time()
    print(
        f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - val_loss : {epoch_val_loss:.4f}\n"
    )
    print("   num_all,num_good:",num_all,num_good,"; num_all_val,num_good_val",num_all_val,num_good_val)
    printmem()
    if epoch_val_loss < epoch_val_loss_min:
        epoch_val_loss_min = epoch_val_loss
        print("Validation loss decreased, writing out model!")
        t0 = time.time()
        torch.save(model, args.output_model_path)
        print("... time to write out:",time.time()-t0)


# %%
