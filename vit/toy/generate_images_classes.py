# %%
# Imports for this and the following sections:
import PIL
import glob
import numpy as np
import math

from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
#
# Set the parameters
num_train_samples = 5000
num_test_samples = 250
num_line_points = 50
#main_dir = 'classification/low_noise/'
#noise_ratio = 0.5
main_dir = 'classification/high_noise/'
noise_ratio = 2.0

num_noise_points = int(num_line_points*noise_ratio)
#
def generate_data(this_many,lim = 10,prefix=''):
    for slope_type in ['positive','negative']:
        if slope_type == 'positive':
            min_angle = 10.
            max_angle = 40.
        else:
            min_angle = 140.
            max_angle = 170.
        for i in tqdm(range(this_many)):

            # create plot with limits
            plt.figure(figsize=(5,5))
            plt.xlim([-lim,lim])
            plt.ylim([-lim,lim])

            angle = np.random.uniform(low=min_angle, high=max_angle)
            slope = math.tan(math.pi*angle/180.0)
            intercept = 0.0
            xp = np.random.uniform(low=-lim, high=lim, size=num_line_points)
            yp = xp*slope + intercept

            # add scatter plot as noise
            xs = np.random.uniform(low=-lim, high=lim, size=num_noise_points)
            ys = np.random.uniform(low=-lim, high=lim, size=num_noise_points)
            #xs = np.array([])
            #ys = np.array([])
            xp = np.concatenate([xp,xs])
            yp = np.concatenate([yp,ys])

            plt.scatter(xp, yp, s=10, c='k')
            if i == 0:
                #print(xp,yp)
                plt.show()

            # tidy up
            plt.gca().set_aspect('equal')
            plt.gca().get_yaxis().set_visible(False)
            plt.gca().get_xaxis().set_visible(False)

#            fname = f'{prefix}sample{i:04}_angle_{int(angle*100):05}.png'
            fname = f'{prefix}sample{i:04}_slope_{slope_type}_angle_{int(angle*100):05}.png'
            plt.savefig(fname,dpi=20)
            plt.close()
            img = PIL.Image.open(fname)
            newimg = img.convert('RGB')
            newimg.save(fname,'PNG')

# %%
#Generate pngs:
generate_data(num_train_samples,prefix=main_dir + '/train/')
generate_data(num_test_samples,prefix=main_dir + '/test/')

# %%
this_png = glob.glob(main_dir + '/train/*0008*.png')
print(this_png)
img = np.array(PIL.Image.open(this_png[0]))
plt.imshow(img[:,:,0])
print(img.shape)
# %%

