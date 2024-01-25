# %%
import pandas as pd
import numpy as np
import glob
from PIL import Image
import matplotlib.pyplot as plt
from skimage.io import imsave
import time

fnames = glob.glob('/fs/ess/PAS2159/neutrino/signal_fixed/dataframe_converted/volts_images/*.pkl')

output_dir = '/fs/ess/PAS2159/neutrino/signal_fixed/dataframe_converted/volts_images/images/'
output_dir = '/fs/scratch/PAS2159/neutrino/signal_fixed/dataframe_converted/volts_images/images'
t0 = time.time()
for fname in sorted(fnames):
    df = pd.read_pickle(fname)
    print(fname,df.columns)
    for row in df.itertuples():
        phi = int(100*row.interaction_phi)
        theta = int(100*row.interaction_eta)
        #print(phi,row.interaction_phi,row.image.shape)
        final_image = row.image[0:255,:]
        #break
# Normalize the image with values between 0 and 255
        normalized_image = (final_image - np.min(final_image)) * (255.0 / (np.max(final_image) - np.min(final_image)))
        normalized_image = normalized_image.astype('uint8')

# Convert the normalized numpy array to an image using Pillow
        image = Image.fromarray(normalized_image)
        filepath = output_dir + '/image_THETA'+str(theta)+'THETA_PHI'+str(phi)+'PHI.png'
        image.save(filepath)

        #plt.imshow(image) #Needs to be in row,col order

    print("   time:",time.time()-t0)
    t0 = time.time()


# %%
