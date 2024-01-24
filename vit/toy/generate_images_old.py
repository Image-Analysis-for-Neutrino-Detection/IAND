# %%
# Imports for this and the following sections:
import PIL
import glob
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def generate_data(this_many,lim = 10,prefix=''):

    for i in tqdm(range(this_many)):

        # create plot with limits
        plt.figure(figsize=(5,5))
        plt.xlim([-lim,lim])
        plt.ylim([-lim,lim])

        # add tilted rectangle
        angle = np.random.uniform(low=0, high=180)
        plt.gca().add_patch(Rectangle((0,0),lim-1,1,angle=angle,facecolor='k'))
        plt.gca().add_patch(Rectangle((0,0),-lim+1,1,angle=angle,facecolor='k'))

        # add scatter plot as noise
        xs = np.random.uniform(low=-lim, high=lim, size=50)
        ys = np.random.uniform(low=-lim, high=lim, size=50)
        plt.scatter(xs, ys, s=100, c='k')

        # tidy up
        plt.gca().set_aspect('equal')
        plt.gca().get_yaxis().set_visible(False)
        plt.gca().get_xaxis().set_visible(False)

        plt.savefig(f'{prefix}sample{i:04}_angle_{int(angle*100):05}.png',dpi=20)
        plt.close()

# %%
#Generate 10000 such pngs:
generate_data(10000,prefix='pngs/')

#%%

generate_data(200,prefix='pngs_test/')

# %%
