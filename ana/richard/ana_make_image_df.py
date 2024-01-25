# %%

import numpy as np
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
import pandas as pd
import glob
from collections import defaultdict
from collections import Counter
import time
from functools import partial
from itertools import repeat
from itertools import combinations
def nested_defaultdict(default_factory, depth=1):
    result = partial(defaultdict, default_factory)
    for _ in repeat(None, depth - 1):
        result = partial(defaultdict, result)
    return result()
import scipy.signal

import matplotlib
import math
matplotlib.rcParams.update({'font.size': 16})
from math import atan2,degrees,sqrt,acos,pi
#
# %%
import argparse
arg_parser = argparse.ArgumentParser() 
arg_parser.add_argument('--input_dir',
                            type=str,
                            default='')
arg_parser.add_argument('--output_dir',
                            type=str,
                            default='')
arg_parser.add_argument('--run_number',
                            type=int,
                            default=-1)
      
args = arg_parser.parse_args()
print(args)
t00 = time.time()
input_dir = args.input_dir
output_dir = args.output_dir
run_number = args.run_number

#
# Some utility functions
def asSpherical(xyz):
    #takes list xyz (single coord)
    x       = xyz[0]
    y       = xyz[1]
    z       = xyz[2]
    r       =  sqrt(x*x + y*y + z*z)
    theta   =  acos(z/r)*180/ pi #to degrees
    phi     =  atan2(y,x)*180/ pi
    if phi<0.0:
        phi += 360.0
    return (r,theta,phi)

def AngleBtw2Points(pointA, pointB):
  changeInX = pointB[0] - pointA[0]
  changeInY = pointB[1] - pointA[1]
  #print(pointA,pointB,changeInX,changeInX)
  phi_radians = atan2(changeInY,changeInX)
  dphi = degrees(phi_radians) #remove degrees if you want your answer in radians
  if dphi < 0:
     dphi += 360.0
  return dphi
# %%
#
# Load the antenna map
df_antenna_map = pd.read_csv('../data/antenna_id_from_useful_channel.csv')
hpol_list = []
vpol_list = []
useful_channel_to_antenna = defaultdict(lambda:-1)
for row in df_antenna_map.itertuples():
    useful_channel_to_antenna[row.UsefulChanIndexH] = row.IceMCAnt
    useful_channel_to_antenna[row.UsefulChanIndexV] = row.IceMCAnt
    hpol_list.append(row.UsefulChanIndexH)
    vpol_list.append(row.UsefulChanIndexV)
print(useful_channel_to_antenna)       

# Load the anita antennaa coords
df_anita_antenna = pd.read_csv('../data/anitaIIIPhotogrammetry.csv',skiprows=1)
print(len(df_anita_antenna),df_anita_antenna.columns)
antenna_id_to_xyz = {}
antenna_id_to_phi_radians = {}
antenna_id_to_phi_degrees = {}
zring = {}
for index,row in df_anita_antenna.iterrows():
    print(int(row['  Z (in)  ']))
    zval = int(row['  Z (in)  '])
    if abs(zval) <= 2.0:
        zval = '0'
    elif abs(zval-106) <= 2.0:
        zval = '106'
    elif abs(zval-144) <= 2.0:
        zval = '144'
    elif abs(zval+43) <= 2.0:
        zval = '-43'
    zring[row['An']-1] = zval
        
    antenna_id_to_xyz[row['An']-1] = (row['  X (in)  '],row['  Y (in)  '],row['  Z (in)  '])
    antenna_id_to_phi_radians[row['An']-1] = math.atan2(row['  Y (in)  '],row['  X (in)  '])
    degree_val = math.degrees(antenna_id_to_phi_radians[row['An']-1])
    if degree_val < 0:
        degree_val += 360
    antenna_id_to_phi_degrees[row['An']-1] = degree_val
# %%
#
# Load the dataframes
fnames = glob.glob(input_dir + '/*_'+str(run_number)+'.pkl')
print('fnames',fnames)
li = []
for fname in fnames:
    dft = pd.read_pickle(fname)
    li.append(dft)
df = pd.concat(li, axis=0, ignore_index=True)

# %%
drows = []
draw_event = 3
t0 = time.time()
for row in df.itertuples():
    run_number = row.run
    event = row.event
    neu_energy = row.neu_energy
    interaction_phi = row.interaction_phi
    interaction_eta = row.interaction_eta
    event_times = row.event_times
    event_volts = row.event_volts
    event_chan_ids = row.event_chan_ids
#
    p_array = np.square(event_volts)
    image_arr = defaultdict(list)

    for cid,v_array in zip(event_chan_ids,event_volts):
        p_array = np.square(v_array)
        #print('p_array',p_array.shape)
        p_array = p_array.tolist()
        if cid in useful_channel_to_antenna:
            antenna_id = useful_channel_to_antenna[cid]
            phi = antenna_id_to_phi_degrees[antenna_id]
            (x,y,z) = antenna_id_to_xyz[antenna_id]
            if cid in hpol_list:
                phi -= 0.05
            else:
                phi += 0.05
        image_arr[phi] = p_array
    image_map = np.empty([260,48*2], dtype = float)
    xv = []
    yv = []
    rows = []
    channel = 0
    for phi in sorted(image_arr):
        tchannel = 0
        for pval in image_arr[phi]:
            rows.append({'event':event,'channel':channel,'tchannel':tchannel,'power':pval})
            image_map[tchannel][channel] = pval
            #print('power',tchannel,channel,pval)
            tchannel += 1
        channel += 1
    df_plot = pd.DataFrame(rows)
    drows.append({'run':run_number, 'event':event,
                  'neu_energy':neu_energy,'interaction_phi':interaction_phi,
                  'interaction_eta':interaction_eta,'image':image_map})
    if False: #event <= draw_event:  #event==draw_event:
        print('neu_energy,interaction_phi,interaction_eta',round(neu_energy,2),round(interaction_phi,2),round(interaction_eta,2))
        fig = px.density_heatmap(df_plot,x='tchannel', y='channel',z='power',nbinsx=260,nbinsy=48*2,histfunc='min')
        fig.show()
        #for p in sorted(power_by_phi):
        #    print('   ',p,power_by_phi[p])
       
#    if event >= draw_event:
#        break
#
# Now form dataframe of event images
df_images = pd.DataFrame(drows)
print("start write")

df_images.to_pickle(output_dir + '/df_simple_power_images_'+str(run_number)+'.pkl')
print("done")
print("Total events:",len(df))
print('Total time:',time.time()-t00)
print('just images time:',time.time()-t0)

# %%
