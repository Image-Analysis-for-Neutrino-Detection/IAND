# %%
import tarfile
from PIL import Image
import tarfile
from io import BytesIO
from pathlib import Path
from typing import Union
from torchvision.transforms.functional import to_tensor
import matplotlib.pyplot as plt

input_array_all = []
tar_file_name = '/fs/scratch/PAS2159/neutrino/signal_fixed/dataframe_converted/volts_images/images_train_all.tar'
tar_file = tarfile.open(tar_file_name)
image_name = 'image_run_1_event_0_THETA8905THETA_PHI20393PHI.png'
num = 0
for member in tar:
    if num<10:
        print(member.name)
    num += 1
    image = tar_file.extractfile(f'{image_name}')
    print('1',type(image))

    image = image.read()
    print('2',type(image),len(image))
    image = Image.frombytes('L', (96,256), image, 'raw')
    print('2',type(image))
    plt.imshow(image)
    #img = Image.open(BytesIO(tar_file.extractfile(f'{image_name}').read())).convert('RGB')

    break

# %%

from PIL import Image, TarIO

fp = TarIO.TarIO("/fs/scratch/PAS2159/neutrino/signal_fixed/dataframe_converted/volts_images/images_train_all.tar", "image_run_1_event_0_THETA8905THETA_PHI20393PHI.png")
im = Image.open(fp)

# %%
