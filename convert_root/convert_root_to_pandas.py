# %%
import numpy as np
import pandas as pd

import uproot
import math
from math import atan2,degrees,sqrt,acos,pi

# input_dir = '/fs/ess/PAS2159/neutrino/signal_fixed/'
# output_dir = '/fs/ess/PAS2159/neutrino/signal_fixed/dataframe_converted'
# run_number = 1

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

input_dir = args.input_dir
output_dir = args.output_dir
run_number = args.run_number

#%% 

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
# %%
print("truthAnitaTree")
truthAnitaTree = uproot.open(input_dir + '/SimulatedAnitaTruthFile'+str(run_number)+'.root:truthAnitaTree')
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
events = uproot.open(input_dir + '/SimulatedAnitaEventFile'+str(run_number)+'.root:eventTree')

print("events Tree")
for key in events.keys():
    print("   ",key)

event_times_array = events['event/fTimes[108][260]'].array(library="np")
event_volts_array = events['event/fVolts[108][260]'].array(library="np")
event_chan_ids_array = events['event/RawAnitaEvent/chanId[108]'].array(library="np")
print(event_times_array.shape)

# %%

num_events = event_times_array.shape[0]
print('num_events',num_events)
# %%
#
# Generate the channel vs time, power as value image
drows = []
for event in range(num_events):
    interaction_eta = interaction_eta_from_balloon[event]
    interaction_phi = interaction_phi_from_balloon[event]
    neu_energy = neutrino_energy[event]
    event_times = event_times_array[event]
    event_volts = event_volts_array[event]
    event_chan_ids = event_chan_ids_array[event]
    drows.append({'run':run_number, 'event':event,
                    'neu_energy':neu_energy,'interaction_phi':interaction_phi,'interaction_eta':interaction_eta,
                    'event_chan_ids':event_chan_ids,
                    'event_times':event_times,'event_volts':event_volts})
#
# Now form dataframe of event images
df_out = pd.DataFrame(drows)
df_out.to_pickle(output_dir + '/df_root_neutrino_'+str(run_number)+'.pkl')
print("done")

# %%
