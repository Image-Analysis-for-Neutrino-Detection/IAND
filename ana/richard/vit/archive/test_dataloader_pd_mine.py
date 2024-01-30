# %%
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import math
#df = pd.read_pickle('/fs/scratch/PAS2159/neutrino/signal_fixed/dataframe_converted/volts_images/df_simple_power_images_99.pkl')

fnames = glob.glob('/fs/scratch/PAS2159/neutrino/signal_fixed/dataframe_converted/volts_images/df_simple_power_images_99.pkl')
train_rows = []
test_rows = []
num = 0
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
#        normalized_image = (final_image - np.min(final_image)) * (255.0 / (np.max(final_image) - np.min(final_image)))
        normalized_image = (final_image - np.min(final_image)) * (255.0 / (np.max(final_image) - np.min(final_image)))/255.0
        normalized_image = normalized_image.astype('float32')
        if num<2:
            print(np.max(normalized_image))

# Convert the normalized numpy array to an image using Pillow
        image = normalized_image # Image.fromarray(normalized_image)
        image = np.resize(image,(1,256,96))
        label = row.interaction_phi/360.0
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
    if num>50000:
        break
df = pd.DataFrame(train_rows)
print(df.columns)
# %%
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

data = CustomDataset(dataframe=df)
dataloader = DataLoader(data)
for sample in dataloader:
    print(sample[1].item(),sample[0].shape)


# %%
