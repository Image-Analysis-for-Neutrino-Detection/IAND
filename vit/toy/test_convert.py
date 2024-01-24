# %%
import PIL
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
def example():
    base = Image.open('test.jpg').convert('RGBA')
    txt = Image.new('RGBA', base.size, (255,255,255,0))
    fnt = ImageFont.truetype('/Library/Fonts/Chalkduster.ttf', 40)
    drw = ImageDraw.Draw(txt)
    drw.text((10,10), "HELLO", font=fnt, fill=(255,0,0,128))
    result= Image.alpha_composite(base, txt)
    result.convert('RGB')
    print ('mode after convert = %s'%result.mode)
    result.save('test1.jpg','JPEG')


# %%
testimg = 'sample0943_slope_negative_angle_15198.png'
img = PIL.Image.open(testimg)
img_arr = np.array(img)
print(img_arr.shape)
plt.imshow(img_arr[:,:,:])

# %%
newimg = img.convert('RGB')
img_arr = np.array(newimg)
print(img_arr.shape)
plt.imshow(img_arr[:,:,:])
# %%
