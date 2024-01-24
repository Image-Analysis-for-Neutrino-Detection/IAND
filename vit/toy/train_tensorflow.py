# %%
# Imports for this and the following sections:
import PIL
import glob
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

#List all `png`s we have generated:
pngs = glob.glob('pngs/sample*.png')

ims = {}
for png in pngs:
    ims[png]=np.array(PIL.Image.open(png))

#Let's call the arrays created from pngs `questions`
questions = np.array([each for each in ims.values()]).astype(np.float32)
#Read in the slopes to an array:
solutions = np.array([float(each.split('_')[-1].split('.')[0])/100 for each in ims]).astype(np.float32)


# %%
#Check the first color channel of the first image:

#Check the slope on the image above:
plt.imshow(questions[0][:,:,0])
print(solutions[0])
# %%

model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(3, 3, activation='relu'),
  tf.keras.layers.Conv2D(3, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(2),
  tf.keras.layers.Conv2D(3, 3, activation='relu'),
  tf.keras.layers.Conv2D(3, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(2),
  tf.keras.layers.Conv2D(3, 3, activation='relu'),
  tf.keras.layers.Conv2D(3, 3, activation='relu'),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(units=512, activation='relu'),
  tf.keras.layers.Dense(units=256, activation='relu'),
  tf.keras.layers.Dense(units=64, activation='relu'),
  tf.keras.layers.Dense(units=1)
])

model.compile(loss='mean_squared_error', optimizer="adam")

history = model.fit(questions, solutions, epochs=10, batch_size=200, verbose=1)


#
# %%

#Read in pngs:
test_pngs = glob.glob('pngs_test/*.png')

test_ims = {}
for png in test_pngs:
    test_ims[png]=np.array(PIL.Image.open(png))

#Prepare test questions and solutions as before:
test_questions = np.array([each for each in test_ims.values()]).astype(np.float32)
test_solutions = np.array([float(each.split('_')[-1].split('.')[0])/100
                            for each in test_ims]).astype(np.float32)

#Apply model:
test_answers = model.predict(test_questions)

for ts,ta in zip(test_solutions,test_answers):
    print(ts,ta)
# %%
