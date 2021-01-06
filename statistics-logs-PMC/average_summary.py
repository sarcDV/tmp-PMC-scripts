import glob, os
import math
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import mannwhitneyu
from scipy.stats import wilcoxon
from sklearn.metrics import cohen_kappa_score
from scipy import signal
import warnings
warnings.filterwarnings("ignore")

contrasts = ['PD', 'T1', 'T2', 'T2s05', 'T2s035', 'T2s025']
OFF = np.zeros((6,6))
ON = np.zeros((6,6))
for ii in range(0, len(contrasts)):
	OFF[ii,::] = np.load(str(contrasts[ii])+'-off-sum.npy')
	ON[ii,::] = np.load(str(contrasts[ii])+'-on-sum.npy')

print(np.mean(OFF[:,0])	, OFF[:,0])
	
#######################################################
coff_ = "#3274A1"##"#333333"
con_ = "#E1812C"####"#b2b2b2"
csame_ ="#ffffff"#"#b2b2b2" ###"#01d1a1"
cedge_ ="#000000"

blim_ = 0
width = 0.35
fig = plt.figure() #figsize = (5.5,5))##1)
# grid = plt.GridSpec(2, 4, hspace=0.2, wspace=0.2)
line_labels = [ "PMC OFF", 
                "PMC ON"]
# Create the bar labels
bar_labels = ['Displ-x', 'Displ-y', 'Displ-z', 'Rot-Roll','Rot-Yaw','Rot-Pitch'] #,'ALL CONTRASTS']
bar_finlabels = ['Displ-x', 'Displ-y', 'Displ-z', 'Rot-Roll','Rot-Yaw','Rot-Pitch'] #,'ALL\nCONT.']
assey_ = np.double([1,2,3,4,5,6])-width/2
## READER 1 ###
ax1 = plt.subplot(111)#(grid[0,0])# plt.subplot(2,4,1)
read1off_ = [np.mean(OFF[:,0]),np.mean(OFF[:,1]), np.mean(OFF[:,2]), np.mean(OFF[:,3]), np.mean(OFF[:,4]), np.mean(OFF[:,5])]
read1on_ = [np.mean(ON[:,0]),np.mean(ON[:,1]), np.mean(ON[:,2]), np.mean(ON[:,3]), np.mean(ON[:,4]), np.mean(ON[:,5])]

xerroroff_ = [np.std(OFF[:,0]),np.std(OFF[:,1]), np.std(OFF[:,2]), np.std(OFF[:,3]), np.std(OFF[:,4]), np.std(OFF[:,5])]
xerroron_ = [np.std(ON[:,0]),np.std(ON[:,1]), np.std(ON[:,2]), np.std(ON[:,3]), np.std(ON[:,4]), np.std(ON[:,5])]          

# l1 = ax1.barh(assey_, read1off_, xerr=xerroroff_,  height=width,  color=coff_ , edgecolor=cedge_, capsize=3)
# l2 = ax1.barh(assey_+width, read1on_, xerr=xerroron_, height=width,  color= con_, edgecolor=cedge_, capsize=3)  #, hatch='/' )

l1 = ax1.barh(assey_, read1off_,  xerr=np.asarray(xerroroff_)/2,  height=width,  color=coff_ , edgecolor=cedge_, capsize=3)
l2 = ax1.barh(assey_+width, read1on_, xerr=np.asarray(xerroron_)/2, height=width,  color= con_, edgecolor=cedge_, capsize=3)  #, hatch='/' )

plt.xticks(fontsize=10, fontweight='bold')
plt.yticks(assey_+width/2, bar_finlabels, fontsize=10, fontweight='bold')
plt.xlabel('[mm] for displacements and [°] for rotations', fontsize=10, fontweight='bold')
plt.title('Averages for all contrasts', fontsize=10, fontweight='bold')
ax1.legend([l1, l2],     # The line objects
           labels=line_labels,   # The labels for each line
           # loc="upper right",# "upper center",   # Position of legend
           # borderaxespad=0.1,    # Small spacing around legend box
           #title=""# "Legend Title"  # Title for the legend
           fontsize=9)
ax1.grid(True)
plt.show()
