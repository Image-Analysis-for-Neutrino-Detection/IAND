import matplotlib
#matplotlib.use('TkAgg')
matplotlib.rcParams.update({'font.size': 20})

import numpy as np
import pandas as pd
from collections import defaultdict
import itertools
import math
from math import ceil
from scipy import signal
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import matplotlib.pyplot as plt

#df = pd.read_pickle('data/df_root_neutrino_19.pkl')
#columns ['run', 'event', 'run_internal', 'weight', 'neu_energy','interaction_phi', 'interaction_eta', 'event_chan_ids', 'event_times','event_volts']


class Antenna:
	def __init__(self,row):
		extra_z = 0.0
		phase_center = 0.0 #m
		self.x = row[1]*0.0254-phase_center/np.sqrt(2)
		self.y = row[2]*0.0254-phase_center/np.sqrt(2)
		self.z = row[3]*0.0254+extra_z
		self.az = (row[5]+360)%360

		self.slice = int(ceil((self.az+360.0)%360.0)/30)
		
def polar2cart(r, theta, phi):
    return [
         r * math.sin(theta) * math.cos(phi),
         r * math.sin(theta) * math.sin(phi),
         r * math.cos(theta)
    ]

class Detector:

    def __init__(self,raw_df):
        self.df = raw_df
        #loads x,y,z and azimuthal center from this .csv file where the columns are 1,2,3 and 5 respectivly
        self.photogram = np.loadtxt('../config/anitaIIIPhotogrammetry.csv',skiprows=2,delimiter=',',usecols=range(0,9))
        
        #loads channel mapping, first value is physical antenna number, second value is horizontal channel id in root file, third value is vertical chanel id in root file.
        self.channelmap = np.loadtxt('../config/ChannelMapping.txt',usecols=range(3,6),skiprows=1)
        #print('channelmap is ',self.channelmap)

        self.numEvents = len(raw_df)
        self.num_chans = len(self.photogram)
        self.channels = defaultdict()

        for ant in range(0,self.num_chans):
            self.channels[ant]=self.GetChan(ant)

    def GetChan(self,id):
        return(Antenna(self.photogram[id]))

    def LoadAntennaNum(self,channel):
        #channel goes from 0 to 108
        ind = np.where(self.channelmap[:,1:]==channel)
        #ind is a tuple of two arrays, the first is an array with the index of rows where element is found, the second is an array with the index of the columns where the element is found
        try:
            if(ind[1][0]==0):
                self.pol='H'
            else:
                self.pol='V'
            return(ind[0][0])
        except IndexError:
            return(555)
        
        
    def LoadWaveform(self,ev):
        self.evnum = ev
        self.wf = defaultdict(dict)
        self.t = defaultdict()
        self.upsample = 30
        for i in range(0,108):
            ant = self.LoadAntennaNum(i)
            self.t[ant]=self.df.loc[ev,'event_times'][i]
            #print(self.t)
            self.wf[ant][self.pol]=self.df.loc[ev,'event_volts'][i]
            #print(self.wf)
            self.wf[ant][self.pol],self.t[ant]=signal.resample(self.wf[ant][self.pol],len(self.wf[ant][self.pol])*self.upsample,t=self.t[ant])

        self.dt = self.t[0][1]-self.t[0][0]
        self.center = len(self.t[0])
        #print("Finished " + str(ev))


class Correlator:
    def __init__(self,data_frame):
        self.timemaps = {}
        self.zoommaps = {}
        self.SingleMaps = {}
        self.c = 299792458
        self.polarizations = ['V','H']

        #Main instrument has 96 channels (48x2 for including Horz and Vert channels of each antenna)
        self.channels = np.linspace(0,47,48,dtype='int')
        #print('channels are :',self.channels)

        #how many sections:
        self.NumSectors = 12
        #all the combinations of ALL the pairs (even the ones that don't make sense)
        self.pairs = list(itertools.combinations(self.channels,2))

        #For main instrument, remove pairs that are far away
        #self.pairs = RemoveBadPairs(self.pairs,40)

        #print('number of pairs ',len(self.pairs))
        self.det = Detector(data_frame)
        self.DefineSectors()

        self.MakeTimemaps()

        self.polarizations = ['V','H']

        self.det.LoadWaveform(0)
        self.LoadTimeIndices()

        
    def DefineSectors(self):

        #main instrument channels 0 through 95
        #nadir channels 96 to 107
        #each sector is centered on angle 0, 30, etc. so spans from +-15,etc.
        #assume antennas can see +-45 degrees?
        #self.NumSectors = 12
        #self.ChannelsPerSector = 16 
        self.SectorAngles = {}
        self.AntennaView=75
        self.sectorWidth = 75#360/self.NumSectors #degrees
        self.sectorCenters = 360/self.NumSectors
        self.allSectors=defaultdict(list)

        for ch in range(0,self.det.num_chans):
            ch_az = self.det.channels[ch].az
            #print(ch,ch_az)
            for sec in range(0,self.NumSectors):
                sectormin = (sec*self.sectorCenters-self.sectorWidth)#-self.AntennaView)
                sectormax = (sec*self.sectorCenters+self.sectorWidth)#+self.AntennaView)
                #print(sectormin,sectormax)
                if(sectormin<=ch_az<=sectormax):
                    self.allSectors[sec].append(ch)
        #print('here the sectors are:', self.allSectors)

    def MakeTimemaps(self):
        #calculate the time delays between each pair of antennas:

        radius = 10000 #m

        self.azspace = 360
        self.alspace = 360

        self.azlist = np.linspace(0,360,self.azspace)*np.pi/180.0
        self.allist = np.linspace(90,180,self.alspace)*np.pi/180.0		
        cart = np.zeros([len(self.allist),len(self.azlist),3])
        for azi, az in enumerate(self.azlist):
            for ali, al in enumerate(self.allist):

                cart[ali,azi]= polar2cart(radius,al,az)

        thismap = {}
        for ch in self.channels:
            #print(ch)
            thismap[ch]= np.zeros([len(self.allist),len(self.azlist)])
            thismap[ch]=np.sqrt((self.det.channels[ch].x-cart[:,:,0])**2+(self.det.channels[ch].y-cart[:,:,1])**2+(self.det.channels[ch].z-cart[:,:,2])**2)/self.c*1e9

        for pair in self.pairs:
            self.timemaps[pair]=thismap[pair[1]]-thismap[pair[0]]


    def CalculateBestAzEl(self,dv,thisaz=None,thisal=None):

        if type(thisaz) is not np.ndarray:
            thisaz=self.azlist
        if type(thisal) is not np.ndarray:
            thisal=self.allist
        min_dval=np.unravel_index(dv[~np.isnan(dv)].argmax(), dv.shape)

        az_best = thisaz[min_dval[1]]*180.0/np.pi
        al_best= thisal[min_dval[0]]*180.0/np.pi

        return(az_best,al_best)

    def LoadTimeIndices(self):
        #for the correlator- do this only once. Essentially converting the time delay maps to a index-based matrix
        self.tmap_inds = defaultdict()

        for antennas in self.pairs:
            self.tmap_inds[antennas]=np.rint(self.timemaps[antennas]/self.det.dt+self.det.center).astype(int)


    def FastCorrelate(self,kZoom=0):
        #The main correlator function!
        #Options:
        #kZoom: whether you want to zoom in around the peak or not. If you do, it loads the zoom maps.
        #kZoom will only work if you already ran the program over the whole map once!

        if(kZoom==0):
            #self.MakeTimemaps()
            #self.LoadTimeIndices()

            #this_timemap = self.timemaps
            this_tmap = self.tmap_inds
            this_az = self.azlist
            this_al = self.allist
            this_azspace = self.azspace
            this_alspace = self.alspace

        else:
            #print('another check :', self.best_az_noNadir_sector,self.best_al_noNadir_sector)
            self.MakeZoomedTimemaps(self.best_az,self.best_al)
            self.LoadZoomIndices()

            #this_timemap = self.zoommaps
            this_tmap = self.tmap_zoom
            this_az = self.azzoom
            this_al = self.alzoom
            this_azspace = self.azzspacezoom
            this_alspace = self.alzspacezoom

        #take each waveform and normalize for correlation:
        """
        for ch in self.channels:
            for polar in polarizations:
                self.det.wf[ch][polar]=self.det.wf[ch][polar]/np.max(self.det.wf[ch][polar])
                self.det.wf[ch][polar]=self.det.wf[ch][polar]/np.std(self.det.wf[ch][polar])
                self.det.wf[ch][polar]=self.det.wf[ch][polar]-np.mean(self.det.wf[ch][polar])
        """
        #set up matrices:
        amplitude_corrector = 1.0/(float(len(self.det.wf[0]['V'])))
        t0 = time.time()
        dvals_main = np.zeros([this_alspace,this_azspace])

        counter_main = np.zeros([this_alspace,this_azspace])
        t0 = time.time()
        all_delays = []
        all_zdiff = []
        #loop over antennas:
        for antennas in self.pairs:

            #loop over polarizations:
            for polar in self.polarizations:

                #correlate the two signals:
                sig0 = self.det.wf[antennas[0]][polar]
                sig1 = self.det.wf[antennas[1]][polar]
                cor = amplitude_corrector*signal.correlate(sig1,sig0)
                
                """
                if((np.max(sig0)>75)&(np.max(sig1)>75)):
                    print('antennas: ',antennas)
                    cor2 = amplitude_corrector*signal.correlate(sig1,sig0,mode='same')
                    print('delay is ', np.argmax(cor2)*dt-50.0)
                    print('expected distance is ', (np.argmax(cor2)*dt-50.0)*1e-9*299792458)
                    print('true distance is ', self.det.channels[antennas[0]].z-self.det.channels[antennas[1]].z)
                    print('')
                    all_delays.append(np.argmax(cor2)*dt-50.0)
                    all_zdiff.append(self.det.channels[antennas[0]].z-self.det.channels[antennas[1]].z)
                    
                    plt.figure(1)
                    plt.plot(self.det.t[0],sig0)
                    plt.plot(self.det.t[0],sig1)
                    plt.figure(2)
                    dvals_main += cor[this_tmap[antennas]]
                    counter_main+=1
                    plt.imshow(cor[this_tmap[antennas]],aspect='auto',extent=[np.min(this_az)*180.0/np.pi,np.max(this_az)*180.0/np.pi,np.max(this_al)*-180.0/np.pi+90,np.min(this_al)*180.0/np.pi-90])
                    plt.colorbar()
                    plt.show()
                """

                
                #then loop over sections. Only add info if antenna is pointing in the direction of the section (roughly)
                for sec in range(0,self.NumSectors):

                    #for zoomed in maps, don't even add the section at all unless it's the specific section we're zoomed in around
                    checkAngle = sec*self.sectorCenters
                    if((this_az[0]!=0)&(this_az[-1]*180.0/np.pi!=360)&((checkAngle+self.AntennaView<this_az[0]*180/np.pi)|(checkAngle-self.AntennaView>this_az[-1]*180/np.pi))):
                        continue

                    #otherwise, add to appropriate part of map.
                    #First add to main map (no nadirs):
                    if((antennas[0] in self.allSectors[sec]) &(antennas[1] in self.allSectors[sec])):
                        if(kZoom==0):
                            leftIndex = int((sec*self.sectorCenters-self.AntennaView)*this_azspace/360)
                            rightIndex = int((sec*self.sectorCenters+self.AntennaView)*this_azspace/360)
                            if(leftIndex<0):
                                dvals_main[:,leftIndex:] += cor[this_tmap[antennas][:,leftIndex:]]
                                counter_main[:,leftIndex:] += 1
                                dvals_main[:,:rightIndex] += cor[this_tmap[antennas][:,:rightIndex]]
                                counter_main[:,:rightIndex] += 1

                            else:
                                dvals_main[:,leftIndex:rightIndex] += cor[this_tmap[antennas][:,leftIndex:rightIndex]]
                                counter_main[:,leftIndex:rightIndex]+=1
                        else:
                            dvals_main += cor[this_tmap[antennas]]
                            counter_main+=1

                

        plt.figure(2)
        plt.scatter(all_delays,all_zdiff)
        plt.xlabel('Delay between channels [ns]')
        plt.ylabel('Difference in height [m]')
        plt.show()
        #then finally we're done! these are the maps for nadirs and main:
        dvals_main=dvals_main/counter_main

        self.best_az,self.best_al= self.CalculateBestAzEl(dvals_main,this_az,this_al)

        t2 = time.time()
        #print('time elapsed: ',t2-t0)		


        #make the maps:
        plt.figure(1,figsize=(12,8))
        ax= plt.gca()

        #im = ax.imshow(dvals_main,aspect='auto',vmin=min_v,vmax=max_v,extent=[np.min(this_az)*180.0/np.pi,np.max(this_az)*180.0/np.pi,np.max(this_al)*-180.0/np.pi+90,np.min(this_al)*180.0/np.pi-90])
        im = ax.imshow(dvals_main,aspect='auto',extent=[np.min(this_az)*180.0/np.pi,np.max(this_az)*180.0/np.pi,np.max(this_al)*-180.0/np.pi+90,np.min(this_al)*180.0/np.pi-90])
        #ax.scatter(self.best_az_Nadir_sector,'x',90-self.best_al_Nadir_sector,color='white',alpha=0.5)

        divider = make_axes_locatable(ax)
        #plt.title('Main Only')
        plt.xlabel('Azimuth [Deg]')
        plt.ylabel('Elevation [Deg]')	
        cax = divider.append_axes("right", size="5%",pad=0.01)
        plt.colorbar(im, cax=cax)
        #plt.savefig('test'+str(self.det.evnum)+'.png')
        plt.close()


        return(dvals_main)