# %%
# Imports for this and the following sections:
import PIL
import glob
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

#
# This geberates two classes of images:
# - lines (with noise) of positive slope (10-40 degrees)
# - lines (with noise) of negative slope (140-170 degrees)

def generate_data(this_many_each,lim = 10,prefix=''):

    for slope in ['positive','negative']:
        if slope == 'positive':
            min_angle = 10.
            max_angle = 40.
        else:
            min_angle = 140.
            max_angle = 170.

        for i in tqdm(range(this_many_each)):

            # create plot with limits
            plt.figure(figsize=(5,5))
            plt.xlim([-lim,lim])
            plt.ylim([-lim,lim])

            # add tilted rectangle
            angle = np.random.uniform(low=min_angle, high=max_angle)
#            plt.gca().add_patch(Rectangle((0,0),lim-1,1,angle=angle,facecolor='k'))
#            plt.gca().add_patch(Rectangle((0,0),-lim+1,1,angle=angle,facecolor='k'))
            plt.gca().add_patch(Rectangle((0,0),lim-.5,.5,angle=angle,facecolor='k'))
            plt.gca().add_patch(Rectangle((0,0),-lim+.5,.5,angle=angle,facecolor='k'))

            # add scatter plot as noise
#            xs = np.random.uniform(low=-lim, high=lim, size=50)
#            ys = np.random.uniform(low=-lim, high=lim, size=50)
            xs = np.random.uniform(low=-lim, high=lim, size=1500)
            ys = np.random.uniform(low=-lim, high=lim, size=1500)
            plt.scatter(xs, ys, s=50, c='k')

            # tidy up
            plt.gca().set_aspect('equal')
            plt.gca().get_yaxis().set_visible(False)
            plt.gca().get_xaxis().set_visible(False)

#            plt.savefig(f'{prefix}sample{i:04}_slope_{slope}_angle_{int(angle*100):05}.png',dpi=20)
            fname = f'{prefix}sample{i:04}_slope_{slope}_angle_{int(angle*100):05}.png'
            plt.savefig(fname,dpi=20)
            plt.close()
            img = PIL.Image.open(fname)
            newimg = img.convert('RGB')
            newimg.save(fname,'PNG')


# %%
#Generate 10000 such pngs:
generate_data(2000,prefix='pngs_slopes_xnoise2/')

#%%

generate_data(200,prefix='pngs_slopes_xnoise_test2/')

# %%
itype = 'pos'  # or 'neg'
this_png = glob.glob('pngs_slopes_xnoise2/sample*0010*'+itype+'*.png')
print(this_png)
# %%
img = np.array(PIL.Image.open(this_png[0]))
plt.imshow(img[:,:,0])

# %%
print(img.shape)
# %%
