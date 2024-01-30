# %%
import pandas as pd
import numpy as np
import glob

from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from skimage.io import imsave
import time
from random import random 
import tarfile

import argparse
arg_parser = argparse.ArgumentParser() 
arg_parser.add_argument('--INPUT_DIR',
                            type=str,
                            default='')
arg_parser.add_argument('--OUTPUT_TAR_TRAIN',
                            type=str,
                            default='')
arg_parser.add_argument('--OUTPUT_TAR_TEST',
                            type=str,
                            default='')
arg_parser.add_argument('--TEST_PERCENT',
                            type=float,
                            default=0.1)

args = arg_parser.parse_args()
print(args)

tar_train = tarfile.open(args.OUTPUT_TAR_TRAIN, 'w')
tar_test = tarfile.open(args.OUTPUT_TAR_TEST, 'w')

fnames = glob.glob(args.INPUT_DIR+'/*.pkl')

t0 = time.time()
num_train = 0
num_test = 0
for fname in sorted(fnames):
    df = pd.read_pickle(fname)
    print(fname,df.columns)
    for row in df.itertuples():
        run = row.run
        event = row.event
        phi = int(100*row.interaction_phi)
        theta = int(100*row.interaction_eta)
        #print(phi,row.interaction_phi,row.image.shape)
        final_image = row.image[0:256,:]
        #break
# Normalize the image with values between 0 and 255
        normalized_image = (final_image - np.min(final_image)) * (255.0 / (np.max(final_image) - np.min(final_image)))
        normalized_image = normalized_image.astype('uint8')

# Convert the normalized numpy array to an image using Pillow
        image = Image.fromarray(normalized_image)
        #filepath = output_dir + '/image_run_'+str(run)+'_event_'+str(event)+'_THETA'+str(theta)+'THETA_PHI'+str(phi)+'PHI.png'
        #image.save(filepath)
        name = 'image_run_'+str(run)+'_event_'+str(event)+'_THETA'+str(theta)+'THETA_PHI'+str(phi)+'PHI.png'
        img_fileobj = BytesIO(image.tobytes())

        tarinfo = tarfile.TarInfo(name=name)
        tarinfo.size = img_fileobj.getbuffer().nbytes
        if random()<args.TEST_PERCENT:
            tar_test.addfile(tarinfo, img_fileobj)
            num_test += 1
        else:
            tar_train.addfile(tarinfo, img_fileobj)
            num_train += 1

        

        #plt.imshow(image) #Needs to be in row,col order

    print("   time:",time.time()-t0,'; num_train,num_test:',num_train,num_test)
    t0 = time.time()

tar_train.close()
tar_test.close()
# %%