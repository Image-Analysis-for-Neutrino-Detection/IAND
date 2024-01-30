
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


#from vit_pytorch.efficient import ViT
from vit_pytorch import ViT

import argparse
arg_parser = argparse.ArgumentParser() 
arg_parser.add_argument('--train_path',
                            type=str,
                            default='')
arg_parser.add_argument('--test_path',
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

train_path = args.train_path
valid_path = args.test_path
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
# Now normalize (take out the factor of 100, then dived by max which is 360 degrees)
    label = (int(label)/100.0) / 360.0
#
    return label

#
# Utility function to go from normalized value back to degrees
def convert_to_degrees(val):
    val_conv = val*360.0
    if val_conv<0:
        val_conv += 360.0
    return val_conv


# %%
input_array_train = []
tar = tarfile.open(train_path)
num = 0
for member in tar:
    if num<10:
        print(member.name)
    num += 1
    label = get_slope_from_path(member.name)
    input_array_train.append((train_path,member.name, label))
input_array_test = []
tar = tarfile.open(valid_path)
num = 0
for member in tar:
    if num<10:
        print(member.name)
    num += 1
    label = get_slope_from_path(member.name)
    input_array_test.append((valid_path,member.name, label))

print("Number training samples:",len(input_array_train))
print("Number testing samples:",len(input_array_test))


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
# %%
# Load data

class SlopeDataset(Dataset):
    def __init__(self, root_dir: Union[str, Path],transform=None,input_array=[]):
        super(SlopeDataset, self).__init__()
        self.root_dir = Path(root_dir)
        self.tar_files = {}  # Initialise an empty dict
        # Initialise self.input_array from a metadata file containing info about the tar files
        # such that self.input_array[i] contains a tuple (tar_file_name, image_name, label)
        self.input_array = input_array
        self.transform = transform

    def __len__(self):
        return len(self.input_array)

    def __getitem__(self, index):
        tar_file_name, image_name, label = self.input_array[index]
        if tar_file_name in self.tar_files:  # If file has been opened before then use it
            tar_file = self.tar_files[tar_file_name]
        else:  # If not then open the tar file and store the TarFile object for future use
            tar_file = tarfile.open(self.root_dir / tar_file_name)
            self.tar_files[tar_file_name] = tar_file
        # The complicated code to extract a png file from a tar file
        #img = Image.open(BytesIO(tar_file.extractfile(f'{image_name}').read())).convert('RGB')
        img = tar_file.extractfile(f'{image_name}')
        img = img.read()
        img = Image.frombytes('L', (96,256), img, 'raw')
        return to_tensor(img), label



#
train_data = SlopeDataset(train_path, transform=train_transforms, input_array=input_array_train)
valid_data = SlopeDataset(valid_path, transform=test_transforms, input_array=input_array_test)
#test_data = SlopeDataset(test_list, transform=test_transforms)
train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True )
valid_loader = DataLoader(dataset = valid_data, batch_size=batch_size, shuffle=True)
#test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True)
print(len(train_data), len(train_loader))
print(len(valid_data), len(valid_loader))


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
    image_size = (96, 256),
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

epoch_val_loss_min = 1000000.0
for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0
    print("epoch",epoch)

#    for data, label in tqdm(train_loader):
    num_good = 0
    num_all = 0
    first = True
    step = 0
    ts = time.time()
    for data, label in train_loader:
        step += 1
        if step % 100 == 0:
            print("   step:",step,time.time()-ts)
            ts = time.time()
        data = data.to(device)
        label = label.to(device).float()
#
# This step is necessary when #output classes=1 (like for regression) 
# so that output and label have samne dimension
        label = label.unsqueeze(1)
        print("data",data.size())
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
        for o,l in zip(output,label):
            num_all += 1

            if first:
                dphi1 = convert_to_degrees(o-l)
                dphi2 = (torch.remainder(dphi1+180,360.0) - 180.0)
                dphi3 = dphi2/180.0
                print()
                print("training output,pred:",convert_to_degrees(o),convert_to_degrees(l))
            if abs(convert_to_degrees(o-l))<resolution:
                num_good += 1
        first = False
        num_good += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #acc = (output.argmax(dim=1) == label).float().mean()
        #epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)
    print("   finished first loop",num_all,num_good )
    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        num_good_val = 0
        num_all_val = 0
        first = True
        for data, label in valid_loader:
            
            data = data.to(device)
            label = label.to(device)
            label = label.unsqueeze(1)

            val_output = model(data)
            #val_loss = criterion(val_output, label)
            val_loss = MSE_deltaphi(val_output, label)
            for o,l in zip(val_output,label):
                num_all_val += 1
                if first:
                    print("validation output,pred:",convert_to_degrees(o),convert_to_degrees(l))
                if abs(convert_to_degrees(o-l))<resolution:
                    num_good_val += 1
            first = False

            #acc = (val_output.argmax(dim=1) == label).float().mean()
            #epoch_val_accuracy += acc / len(valid_loader)
            epoch_val_loss += val_loss / len(valid_loader)

    print(
        f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - val_loss : {epoch_val_loss:.4f}\n"
    )
    print("   num_all,num_good:",num_all,num_good,"; num_all_val,num_good_val",num_all_val,num_good_val)

    if epoch_val_loss < epoch_val_loss_min:
        print("Validation loss decreased, writing out model!")
        t0 = time.time()
        torch.save(model, args.output_model_path)
        print("... time to write out:",time.time()-t0)


# %%
