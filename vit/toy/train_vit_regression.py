
# %%
import glob
from itertools import chain
import os
import random
import zipfile

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
arg_parser.add_argument('--train_dir',
                            type=str,
                            default='')
arg_parser.add_argument('--test_dir',
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
# os.makedirs('data', exist_ok=True)
# train_dir = 'data/train'
# test_dir = 'data/test'
# with zipfile.ZipFile('train.zip') as train_zip:
#     train_zip.extractall('data')
    
# with zipfile.ZipFile('test.zip') as test_zip:
#     test_zip.extractall('data')
#train_list = glob.glob('pngs_slopes/*.png')
#test_list = glob.glob('pngs_slopes_test/*.png')
train_list = glob.glob(args.train_dir+'/*.png')
test_list = glob.glob(args.test_dir+'/*.png')
#train_list = glob.glob('pngs_slopes_xnoise2/*.png')
#test_list = glob.glob('pngs_slopes_xnoise_test2/*.png')
print(f"Train Data: {len(train_list)}")
print(f"Test Data: {len(test_list)}")

random.shuffle(train_list)
random.shuffle(test_list)

#
# Nethod to convert path name containing slope to the actual slope
def get_slope_from_path(path):
#
# Extract the part of the filesname which contains the slope (which is in degrees*100)
    label = path.split('_')[-1].split('.')[0]
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
labels = []
n1 = 0
n2 = 0
for path in train_list:
    label = get_slope_from_path(path)
    n1 += 1 
    if n1<10:
        print(path,label,type(label))


# %%
print(labels[:10])
# %%
train_list, valid_list = train_test_split(train_list, 
                                          test_size=0.2,
                                          random_state=seed)
# %%
print(f"Train Data: {len(train_list)}")
print(f"Validation Data: {len(valid_list)}")
print(f"Test Data: {len(test_list)}")
# %%
# Image augmentation
train_transforms = transforms.Compose(
    [
        transforms.Resize((100,100)),
        #transforms.RandomResizedCrop(100),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize(100),
        #transforms.CenterCrop(100),
        transforms.ToTensor(),
    ]
)


test_transforms = transforms.Compose(
    [
        transforms.Resize(100),
        #transforms.CenterCrop(100),
        transforms.ToTensor(),
    ]
)
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
        img_transformed = self.transform(img)
#
# Extract the part of the filesname which contains the slope (which is in degrees*100)
        label = get_slope_from_path(img_path)

        return img_transformed, label
#
train_data = SlopeDataset(train_list, transform=train_transforms)
valid_data = SlopeDataset(valid_list, transform=test_transforms)
test_data = SlopeDataset(test_list, transform=test_transforms)
train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True )
valid_loader = DataLoader(dataset = valid_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True)
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

model = ViT(
    image_size = 100,
    patch_size = 20,
    num_classes = 1,
    dim = 128,
    depth = 6,
    heads = 10,
    mlp_dim = 128,
    dropout = 0.1,
    emb_dropout = 0.1
).to(device)

# %%
# Training
# loss function: CrossEntropyLoss for classification, MSELoss for regression
#criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0
    print("epoch",epoch)

#    for data, label in tqdm(train_loader):
    num_good = 0
    num_all = 0
    first = True
    for data, label in train_loader:
        data = data.to(device)
        label = label.to(device).float()
#
# This step is necessary when #output classes=1 (like for regression) 
# so that output and label have samne dimension
        label = label.unsqueeze(1)
        output = model(data).float()
        if first:
            print('output/label dimensions: ',output.size(),label.size())

        loss = criterion(output, label)

        for o,l in zip(output,label):
            num_all += 1
            if first:
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
            val_loss = criterion(val_output, label)
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
# %%
