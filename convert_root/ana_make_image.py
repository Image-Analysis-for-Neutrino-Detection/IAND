# %%
import ROOT
import numpy as np
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
import pandas as pd

from collections import defaultdict
from collections import Counter

from functools import partial
from itertools import repeat
from itertools import combinations
def nested_defaultdict(default_factory, depth=1):
    result = partial(defaultdict, default_factory)
    for _ in repeat(None, depth - 1):
        result = partial(defaultdict, result)
    return result()
import scipy.signal
"""
from collections import defaultdict
import os
from fnmatch import fnmatch
import pymap3d as pm
from mpl_toolkits.basemap import Basemap
"""
import matplotlib
import uproot
import math
matplotlib.rcParams.update({'font.size': 16})
#%% 
from math import atan2,degrees,sqrt,acos,pi

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
#
# Load the antenna map
df_antenna_map = pd.read_csv('antenna_id_from_useful_channel.csv')
hpol_list = []
vpol_list = []
useful_channel_to_antenna = defaultdict(lambda:-1)
for row in df_antenna_map.itertuples():
    useful_channel_to_antenna[row.UsefulChanIndexH] = row.IceMCAnt
    useful_channel_to_antenna[row.UsefulChanIndexV] = row.IceMCAnt
    hpol_list.append(row.UsefulChanIndexH)
    vpol_list.append(row.UsefulChanIndexV)
print(useful_channel_to_antenna)       

#
# %%
# Load the anita antennaa coords
df_anita_antenna = pd.read_csv('/users/PAS1043/osu7903/neutrino_research/anita/components/icemc/data/anitaIIIPhotogrammetry.csv',skiprows=1)
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

for cid in sorted(antenna_id_to_phi_degrees,key=antenna_id_to_phi_degrees.get):
    print(cid,antenna_id_to_phi_degrees[cid])

# %%
anita_geom = uproot.open('/users/PAS1043/osu7903/neutrino_research/anita/share/anitaMap/anitageom.root')
print("anita_geom Tree")
for key in anita_geom.keys():
    print("   ",key)
anita = anita_geom['anita;1']
print(anita)

#
# %%
print("truthAnitaTree")
truthAnitaTree = uproot.open('/fs/ess/PAS2159/neutrino/'+dir+'/SimulatedAnitaTruthFile.root:truthAnitaTree')
for key in truthAnitaTree.keys():
    print("   ",key)

truth_nuMom = truthAnitaTree['truth/nuMom'].array(library="np")
truth_nuPos = truthAnitaTree['truth/nuPos[3]'].array(library="np")
truth_balloonPos = truthAnitaTree['truth/balloonPos[3]'].array(library="np")
truth_balloonDir = truthAnitaTree['truth/balloonDir[3]'].array(library="np")

#print(truth_balloonPos[:10])
#print(truth_nuPos[:10])

interaction_eta_from_balloon = []
interaction_phi_from_balloon = []
neutrino_energy = []
num = 0
for nuMom,nupos,bpos in zip(truth_nuMom,truth_nuPos,truth_balloonPos):
    cVec = (nupos[0]-bpos[0],nupos[1]-bpos[1],nupos[2]-bpos[2])
    (r,theta,phi) = asSpherical(cVec)
    dphi = AngleBtw2Points(bpos,nupos)
    if num<4:
       print('dphi ',dphi,nupos[0]-bpos[0],nupos[1]-bpos[1],theta,phi)
    num += 1
    interaction_eta_from_balloon.append(theta)
    interaction_phi_from_balloon.append(dphi)
    neutrino_energy.append(nuMom)

print('interaction_phi_from_balloon',len(interaction_phi_from_balloon))
# %%
#example using uproot
print("SimulatedAnitaEventFile")
#dir = 'signal_1M'
dir = 'signal_100k_new'
events = uproot.open('/fs/ess/PAS2159/neutrino/'+dir+'/SimulatedAnitaEventFile.root:eventTree')
# %%
print("events Tree")
for key in events.keys():
    print("   ",key)

#
# %%
event_RawAnitaEvent_chanId = events['event/RawAnitaEvent/chanId[108]'].array(library="np")
event_RawAnitaEvent_mean = events['event/RawAnitaEvent/mean[108]'].array(library="np")
event_RawAnitaEvent_chanId = events['event/RawAnitaEvent/chanId[108]'].array(library="np")

event_times_array = events['event/fTimes[108][260]'].array(library="np")
event_volts_array = events['event/fVolts[108][260]'].array(library="np")
event_power_array = np.square(event_volts_array)
avg_power = np.average(event_power_array, axis=2)
max_avg_power_index = np.argmax(avg_power,axis=1)
print("event_power_array",event_power_array.shape,avg_power.shape,max_avg_power_index.shape)
print(event_RawAnitaEvent_chanId.shape)
print(event_RawAnitaEvent_mean.shape)
num_events = event_RawAnitaEvent_chanId.shape[0]
print('num_events',num_events)

# %%

draw_event = 2
#
# Generate the channel vs time, power as value image
drows = []
for event in range(num_events):
    interaction_eta = interaction_eta_from_balloon[event]
    interaction_phi = interaction_phi_from_balloon[event]
    neu_energy = neutrino_energy[event]
    image_arr = defaultdict(list)

    for cid,v_array in zip(event_RawAnitaEvent_chanId[event],event_volts_array[event]):
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
    drows.append({'event':event,'neu_energy':neu_energy,'interaction_phi':interaction_phi,'interaction_eta':interaction_eta,'image':image_map})
    if event <= draw_event:  #event==draw_event:
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
df_images.to_hdf('/fs/ess/PAS2159/neutrino/'+dir+'/power_images.hdf',key='images')
print("start write")
df_images.to_pickle('/fs/ess/PAS2159/neutrino/'+dir+'/df_power_images.pkl')
print("done")
#   
# Generate the avg power vs phi map
# %%
for event in range(num_events):
    power_by_phi = defaultdict(float)
    print(event_RawAnitaEvent_chanId[event])
    phivals = []
    powervals = []
    ant_type = []
    zvals = []
    for cid,cm,ap in zip(event_RawAnitaEvent_chanId[event],event_RawAnitaEvent_mean[event],avg_power[event]):
        if cid in useful_channel_to_antenna:
            antenna_id = useful_channel_to_antenna[cid]
            phi = antenna_id_to_phi_degrees[antenna_id]
            (x,y,z) = antenna_id_to_xyz[antenna_id]
            if cid in hpol_list:
                phi -= 0.05
                ant_type.append('Horizontal')
            else:
                phi += 0.05
                ant_type.append('Vertical')
            power_by_phi[phi] = ap
            phivals.append(phi)
            powervals.append(ap)
            zvals.append(zring[antenna_id])
            
    
    if True:  #event==draw_event:
        fig = px.scatter(x=phivals, y=powervals,color=zvals,symbol=ant_type)
        fig.show()
        #for p in sorted(power_by_phi):
        #    print('   ',p,power_by_phi[p])
       
    if event >= draw_event:
        break

#
# %%
print('done')
#
# %%
event_times_array = events['event/fTimes[108][260]'].array(library="np")
event_volts_array = events['event/fVolts[108][260]'].array(library="np")
event_power_array = np.square(event_volts_array)

# %%

print("event_power_array shape",event_power_array.shape)
#print("truth_payloadPhi shape",truth_payloadPhi.shape)

avg_power = np.average(event_power_array, axis=2)
max_avg_power_index = np.argmax(avg_power,axis=1)
print(avg_power.shape)
print(max_avg_power_index.shape)
print(max_avg_power_index)
# %%
fig = px.scatter(x=max_avg_power_index,y=truth_payloadPhi,title = "Phi vs Index",
                 labels = {'x': 'max power index', 'y':'Phi'})

fig.show() 
 
# %%
ev_num = 0
chan = 0
fig = px.line(x=event_times_array[ev_num,chan,:],y=event_power_array[ev_num,chan,:],title = "Pwoer vs Time",
                 labels = {'x': 'time', 'y':'Voltage'})

fig.show() 
 

# %%
