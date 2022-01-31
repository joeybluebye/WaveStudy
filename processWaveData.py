"""
Created on Thu Sep 30 21:01:57 2021

The purpose of this program is to process wave (hiphop) data into a nodal point
model using dynamic time warping. For the later purpose of running a "real-time"
simulation showing intermittent concurrent haptic augmented feedback 
(based on the model) on a still body and on one attempting a wave. 

Main problem solved by this code is dealing with missing data when the elbows
turned away from the camera, a spline fixes it enough to do further analysis.

It sets all waves in the same direction in preparation for DTW. (by calculating
peak prominence and determining the order)

As a bonus there is an animation showing how the DTW averaged the waves together.

@author: Joey Wijffels
"""

print("this will take about one hour, sit back and relax")

import pandas as pd
import numpy as np
import scipy as sy
from scipy import interpolate
import scipy.io as sio
import glob
import os as os
import matplotlib.pyplot as plt
import time

class wavedata():
    def __init__(self, trialnumber,cuttime, read,cut, splined,reverse,onedirection,dtw,comb,prominences):
        
        self.trialnumber = trialnumber
        self.cuttime = cuttime
        self.read = read 
        self.cut = cut 
        self.splined = splined 
        self.reverse = reverse
        self.onedirection = onedirection
        self.dtw = dtw
        self.comb = comb
        self.prominences = prominences
             
start = time.time()

# get to the folder where a subjects data is
os.chdir('C:/Users/Equipo/Documents/Science_work/WaveMLchallenge/Stage_wave/data/wavepilot')
# Get a list of all the csv files
csv_files = glob.glob('*3d*') # files include head and fingers! 

# load start and stop (manually selected in matlab 2018) 
cuttimes = sio.loadmat("metadata_pilot.mat")

def loadwaves(w): 
    trialdata = pd.read_csv(csv_files[w], skiprows = [0,1,2,3])
    trialdata2  = trialdata.iloc[:,:]
    return trialdata2
end = time.time()
print(end - start)

wavesList = []
cuttime = []

for w in range(len(csv_files)):
    # prep the list to fit data
    wavesList.append(wavedata(w,[],loadwaves(w),[],[],[],[],[],[],[]))

for w in range(len(csv_files)): # average the markers together some have 3 or 1 point, x y z

    wavesList[w].comb =np.zeros((len(wavesList[w].read),27))
    for f in range(len(wavesList[w].read)):   
        for t in range(3):
# L finger marker 10
            wavesList[w].comb[f][0 + (1*t)] = np.array(wavesList[w].read)[f, 27 + (1*t)]
# L wrist marker 7,8,9
            wavesList[w].comb[f][3 + (1*t)] = np.nanmean([np.array(wavesList[w].read)[f,18 + (1*t)], np.array(wavesList[w].read)[f,21 + (1*t)], np.array(wavesList[w].read)[f,24 + (1*t)]])
# L elbow marker 4,5,6
            wavesList[w].comb[f][6 + (1*t)] = np.nanmean([np.array(wavesList[w].read)[f,9 + (1*t)], np.array(wavesList[w].read)[f,12 + (1*t)], np.array(wavesList[w].read)[f,15 + (1*t)]])
# L shoulder marker 1,2,3
            wavesList[w].comb[f][9 + (1*t)] = np.nanmean([np.array(wavesList[w].read)[f,0 + (1*t)], np.array(wavesList[w].read)[f,3 + (1*t)], np.array(wavesList[w].read)[f,6 + (1*t)]])
# Head marker 11
            wavesList[w].comb[f][12 + (1*t)] = np.array(wavesList[w].read)[f, 30 + (1*t)]
# R shoulder marker 12,13,14
            wavesList[w].comb[f][15 + (1*t)] = np.nanmean([np.array(wavesList[w].read)[f,33 + (1*t)], np.array(wavesList[w].read)[f,36 + (1*t)], np.array(wavesList[w].read)[f,39 + (1*t)]])
# R elbow marker 15,16,17
            wavesList[w].comb[f][18 + (1*t)] = np.nanmean([np.array(wavesList[w].read)[f,42 + (1*t)], np.array(wavesList[w].read)[f,45 + (1*t)], np.array(wavesList[w].read)[f,48 + (1*t)]])
# R wrist marker 18,19,20
            wavesList[w].comb[f][21 + (1*t)] = np.nanmean([np.array(wavesList[w].read)[f,51 + (1*t)], np.array(wavesList[w].read)[f,54 + (1*t)], np.array(wavesList[w].read)[f,57 + (1*t)]])
# R Hand marker 21   
            wavesList[w].comb[f][24 + (1*t)] = np.array(wavesList[w].read)[f, 60 + (1*t)]

for w in range(len(csv_files)):   
    wavesList[w].read = []
    
end = time.time()
print(end - start)
print("starting to spline")

for w in range(len(csv_files)):
     x = np.array(pd.DataFrame(wavesList[w].comb).index)
     y = np.array(wavesList[w].comb)
     idx_finite = np.array(np.isfinite(y))
     n = len(idx_finite)
     ynew_finite = np.zeros((n,len(y[1])))
     for i in range(len(y[1])):
         if np.isnan(y[:,i]).all():
             print(w)
             wavesList[w].comb = wavesList[w-1].comb
             wavesList[w].prominences = wavesList[w-1].prominences
             break

         f_finite = interpolate.Rbf(x[idx_finite[:,i]], y[idx_finite[:,i]][:,i], function = "cubic")
         ynew_finite[:,i] = f_finite(x)

     if x[idx_finite[:,i]].any:
         # cuts away irrelevant data
         wavesList[w].splined = ynew_finite
         cuttime.append([cuttimes["start"][w - 1],cuttimes["stop"][w - 1]])
         wavesList[w].cut = wavesList[w].splined[cuttime[w][0][0]:cuttime[w][1][0]] 
         print(w)
         
     for i in range(len(y[1])): 
         # quick fix making all peaks positive by adding 1000. spline will return empty otherwise.
         prominence=sy.signal.find_peaks(ynew_finite[:,i]+1000, height = 0, prominence=(None,3))
         wavesList[w].prominences.append(prominence)
         revYnew = np.flipud(wavesList[w].cut)
         wavesList[w].reverse = revYnew

segmentsZup = [2,5,8,11,14,17,20,23,26] 
for w in range(len(csv_files)):   
    wavesList[w].comb = []

# set all waves in one direction (based on which arm moves first)
print("getting that order right for you")

for w in range(len(csv_files)):
    # uses the finger but if its peak is unclear will take the wrist instead
    try:
        first =    wavesList[w].prominences[segmentsZup[0]][0][np.argmax(wavesList[w].prominences[segmentsZup[0]][1]["prominences"])]
    except:
        first =    wavesList[w].prominences[segmentsZup[1]][0][np.argmax(wavesList[w].prominences[segmentsZup[1]][1]["prominences"])]
        print("small problem")
        print(w)
    try:
        last = wavesList[w].prominences[segmentsZup[-1]][0][np.argmax(wavesList[w].prominences[segmentsZup[-1]][1]["prominences"])]
    except: 
        last = wavesList[w].prominences[segmentsZup[-2]][0][np.argmax(wavesList[w].prominences[segmentsZup[-2]][1]["prominences"])]
        print("small problem")
        print(w)

    if first > last:
        wavesList[w].onedirection = wavesList[w].reverse 
    else:
        wavesList[w].onedirection = wavesList[w].splined
    wavesList[w].dtw.append([0,wavesList[w].onedirection])

# delete some data to save memory
for w in range(len(csv_files)):   
    wavesList[w].reverse = []
    wavesList[w].splined = []
    wavesList[w].cut = []

for w in range(len(csv_files)):
    k = [[[],[],[],[],[],[],[],[],[]],[wavesList[w].dtw[0][1][:,segmentsZup[0]],wavesList[w].dtw[0][1][:,segmentsZup[1]],wavesList[w].dtw[0][1][:,segmentsZup[2]],wavesList[w].dtw[0][1][:,segmentsZup[3]],wavesList[w].dtw[0][1][:,segmentsZup[4]],wavesList[w].dtw[0][1][:,segmentsZup[5]],wavesList[w].dtw[0][1][:,segmentsZup[6]],wavesList[w].dtw[0][1][:,segmentsZup[7]],wavesList[w].dtw[0][1][:,segmentsZup[8]]]]
    wavesList[w].dtw.append(k)
    
    del wavesList[w].dtw[0]

end = time.time()
print(end - start)
# dtw
print("starting DTW")
from dtaidistance import dtw

segments = [0,1,2,3,4,5,6,7,8] 

for cycle in range(0,6):
    
    print("cycle")   
    print(cycle)
    end = time.time()
    print(end - start)    
    for w in range(32): 
        if (w < (32/(2**cycle))): 
            paths = [[],[],[],[],[],[],[],[],[]]
            distance = [[],[],[],[],[],[],[],[],[]]
            best_path = [[],[],[],[],[],[],[],[],[]]
            e = [[],[],[],[],[],[],[],[],[]]
            f = [[],[],[],[],[],[],[],[],[]]
            s1 = [[],[],[],[],[],[],[],[],[]]
            s2 = [[],[],[],[],[],[],[],[],[]]
            sc= [[],[],[],[],[],[],[],[],[]]       
            for s in range(0,len(segments)):
                distance[s], paths[s] = dtw.warping_paths(wavesList[w*2].dtw[cycle][1][segments[s]], wavesList[w*2+1].dtw[cycle][1][segments[s]])                   
                best_path[s] = dtw.best_path(paths[s]) 
                e[s]=np.array(best_path[s])[:,0]
                f[s]=np.array(best_path[s])[:,1]     
                s1[s] = wavesList[w*2].dtw[cycle][1][s][e[s]]
                s2[s] = wavesList[w*2+1].dtw[cycle][1][s][f[s]]             
                sc[s] = (s1[s]+s2[s]) /2
            wavesList[w].dtw.append([[[],[],[],[],[],[],[],[],[]],sc,cycle + 1,best_path])
            print(w)
       
end = time.time()
print(end - start)    

#plot
print("building up the plot")
ArmSpan = 0,1,2,3,4,5,6,7,8
segmentsZup = [2,5,8,11,14,17,20,23,26] 

# 9 segments, 64 waves, requiring 7 cycles, reduced to a 100 samples. 
Z = np.array(np.zeros((9,64,7,100)))
Z[:,:,:,:] = np.nan
Y = np.array(np.zeros((9,64,7,100)))
Y[:,:,:,:] = np.nan
X = np.array(np.zeros((9,64,7,100)))
X[:,:,:,:] = np.nan

# reduce samples for animating
for f in range(100):
    for c in range(7): 
        
        for w in range(len(csv_files)/2):
            if (w < ((len(csv_files)/2)/(2**c))):

                for s in range(len(ArmSpan)):
                    wavesList[w].dtw[c][0][s]=np.around(np.linspace(0,len(wavesList[w].dtw[c][1][ArmSpan[s]]),100)).astype(int)
                    Z[s][w][c][f] = wavesList[w].dtw[c][1][ArmSpan[s]][wavesList[w].dtw[c][0][ArmSpan[s]][f]-1] # -1¿¿
                    Y[s][w][c][f] = wavesList[w].onedirection[:,segmentsZup[s]-1][wavesList[w].dtw[0][0][ArmSpan[s]][f]-1]
                    X[s][w][c][f] = wavesList[w].onedirection[:,ArmSpan[s]-2][wavesList[w].dtw[0][0][ArmSpan[s]][f]-1]
# zscore (Y not important)
minZ, maxZ = np.nanmin(Z), np.nanmax(Z)
Z = (Z - minZ)/(maxZ - minZ)
minX, maxX = np.nanmin(X), np.nanmax(X)
X = (X - minX)/(maxX - minX)

fig = plt.figure()
ax = plt.axes(projection="3d")
for f in range(100):
    ax.clear()

    ax.plot_wireframe(X[0:,0:64,0,f],Y[0:,0:64,0,f],    Z[0:,0:64,0,f], color='green')
    ax.plot_wireframe(X[0:,0:32,1,f],Y[0:,0:32,0,f],    Z[0:,0:32,1,f], color='red')
    ax.plot_wireframe(X[0:,0:16,2,f],Y[0:,0:16,0,f],    Z[0:,0:16,2,f], color='yellow')
    ax.plot_wireframe(X[0:,0:8,3,f],Y[0:,0:8,0,f],    Z[0:,0:8,2,f], color='blue')
    ax.plot_wireframe(X[0:,0:4,4,f],Y[0:,0:4,0,f],    Z[0:,0:4,2,f], color='pink')
    ax.plot_wireframe(X[0:,0:2,5,f],Y[0:,0:2,0,f],    Z[0:,0:2,2,f], color='orange')
    ax.plot_wireframe(X[0:,0:1,6,f],Y[0:,0:1,0,f],    Z[0:,0:1,2,f], color='black')

    ax.set_ylabel("ArmSpan")
    ax.set_zlabel("height (mm)") 
    ax.set_xlabel("waves")
    ax.set_ylim([-750, 1000])
    ax.view_init(elev=33, azim=19)    
    plt.pause(0.01)