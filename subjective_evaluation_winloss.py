# -*- coding: utf-8 -*-
import glob, os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from scipy.stats import wilcoxon
from sklearn.metrics import cohen_kappa_score
from scipy import signal
import warnings
warnings.filterwarnings("ignore")

# matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)

""" 
- only the question on motion artifacts
- take only second column!!!
R1 : Daniel
R2 : Falk
R3 : Hendrik
R4 : Renat

Structure example:

R1_off 6 columns [T1, T2, PD, T2s05, T2s035, T2s025]
R1_on  6 columns [T1, T2, PD, T2s05, T2s035, T2s025]
...

readers = ['Daniel', 'Falk', 'Hendrik', 'Renat']
R1_t1off = np.reshape(np.asarray(
	open('./subjective-image-assessment/PIQA_Daniel/results_iqa_T1_OFF.txt').read().split()),(21,3))
R1_t1on = np.reshape(np.asarray(
	open('./subjective-image-assessment/PIQA_Daniel/results_iqa_T1_ON.txt').read().split()),(21,3))
# print(R1_t1off, R1_t1on)
"""
# ----------------------------------------
def ccorrel(p1, p2):
    a = p1
    b = p2
    a = (a - np.mean(a)) / (np.std(a) * len(a))
    b = (b - np.mean(b)) / (np.std(b))
    c = np.correlate(a, b)
    
    return c

def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])

def percentage_eval(in1_, in2_):
	onebettertwo=0
	same_=0
	twobetterone=0
	for ii in range(0, len(in1_)):
		if in1_[ii] == in2_[ii]:
			same_=same_#+1
		elif in1_[ii] > in2_[ii]:
			onebettertwo=onebettertwo+1
		else:
			twobetterone=twobetterone+1

	onebettertwo=-(onebettertwo/len(in1_))*100
	same_=same_#(same_/len(in1_))*100
	twobetterone=(twobetterone/len(in1_))*100

	return truncate(onebettertwo,2), truncate(same_,2), truncate(twobetterone,2)

def convert_pval(inval):
    if inval < 0.01:
        outval = "p < 0.01" 
    elif inval >= 0.01:
        outval = "p = "+str(truncate(inval,2))
    return outval


# -----------------------------------------------
## read data per reader
R1_off, R2_off, R3_off, R4_off = [],[],[],[]
R1_on, R2_on, R3_on, R4_on = [],[],[],[]
readers = ['Daniel', 'Falk', 'Hendrik', 'Renat']
sessions = ['OFF', 'ON']
contrasts =  ['T1', 'T2', 'PD', '05', '035', '025']
path1_ = './subjective-image-assessment/PIQA_'
path2_ = '/results_iqa_'
path3_ = '.txt'


for reader in readers:
	for contrast in contrasts:
		for session in sessions:
			filepath_ = path1_+str(reader)+path2_+str(contrast)+'_'+str(session)+path3_
			# # print(filepath_)
			tmp_ = np.reshape(np.asarray(open(filepath_).read().split()),(21,3))
			# select only last column:
			tmp_ = tmp_[:,2]
			if reader == 'Daniel':
				if session == 'OFF':
					R1_off.append(tmp_)
				else:
					R1_on.append(tmp_)

			if reader == 'Falk':
				if session == 'OFF':
					R2_off.append(tmp_)
				else:
					R2_on.append(tmp_)

			if reader == 'Hendrik':
				if session == 'OFF':
					R3_off.append(tmp_)
				else:
					R3_on.append(tmp_)

			if reader == 'Renat':
				if session == 'OFF':
					R4_off.append(tmp_)
				else:
					R4_on.append(tmp_)


 
R1_off, R2_off, R3_off, R4_off = np.asarray(R1_off),  np.asarray(R2_off), np.asarray(R3_off), np.asarray(R4_off)
R1_on, R2_on, R3_on, R4_on = np.asarray(R1_on),  np.asarray(R2_on), np.asarray(R3_on), np.asarray(R4_on)

###############################################
### cross-correlation #########################
###############################################
# interpretation of kappa:
# kappa     Agreement
#  < 0 		less than chance agreement
# 0.01–0.20 Slight agreement
# 0.21– 0.40 Fair agreement
# 0.41–0.60 Moderate agreement
# 0.61–0.80 Substantial agreement
# 0.81–0.99 Almost perfect agreement
# print(R1_off)
OFF_ = np.array([R1_off, R2_off, R3_off, R4_off])
ON_ = np.array([R1_on, R2_on, R3_on, R4_on])

list_ = []

# : 4, 6, 21 
temp_ = np.concatenate((OFF_, ON_), axis=2)
for c in range(0, temp_.shape[1]):
	R12_ = cohen_kappa_score(temp_[0,c,:], temp_[1,c,:])
	R13_ = cohen_kappa_score(temp_[0,c,:], temp_[2,c,:])
	R14_ = cohen_kappa_score(temp_[0,c,:], temp_[3,c,:])

	R21_ = cohen_kappa_score(temp_[1,c,:], temp_[0,c,:])
	R23_ = cohen_kappa_score(temp_[1,c,:], temp_[2,c,:])
	R24_ = cohen_kappa_score(temp_[1,c,:], temp_[3,c,:])

	R31_ = cohen_kappa_score(temp_[2,c,:], temp_[0,c,:])
	R32_ = cohen_kappa_score(temp_[2,c,:], temp_[1,c,:])
	R34_ = cohen_kappa_score(temp_[2,c,:], temp_[3,c,:])

	R41_ = cohen_kappa_score(temp_[3,c,:], temp_[0,c,:])
	R42_ = cohen_kappa_score(temp_[3,c,:], temp_[1,c,:])
	R43_ = cohen_kappa_score(temp_[3,c,:], temp_[2,c,:])

	# print(R12_, R13_, R14_)
	# print(R21_, R23_, R24_)
	# print(R31_, R32_, R34_)
	# print(R41_, R42_, R43_)

	list_.append([R12_, R13_, R14_,R21_, R23_, R24_,R31_, R32_, R34_,R41_, R42_, R43_])

list_ = np.asarray(list_)
print(np.mean(list_))

###############################################
### read the weights ##########################
###############################################

woff_, won_ = np.zeros((21,)), np.zeros((21,))# [], []
contrastsw_ =  ['T1', 'T2', 'PD', 'T2s05', 'T2s035', 'T2s025']
for contrast in contrastsw_:
	filepath_ = './statistics-logs-PMC/weights/'+str(contrast)+'_weights.txt'
	
	tmp_ = np.reshape(np.asarray(open(filepath_).read().split()),(21,3))
	
	woff_ = np.vstack((woff_, tmp_[:,1]))
	won_ = np.vstack((won_, tmp_[:,2]))
	

woff_, won_ = woff_[1:,:], won_[1:, :]

# print((np.double(woff_)*np.double(R1_off))+np.double(R1_off))

R1_off = (np.double(woff_)*np.double(R1_off))+np.double(R1_off)
R1_on = (np.double(won_)*np.double(R1_on))+np.double(R1_on)

R2_off = (np.double(woff_)*np.double(R2_off))+np.double(R2_off)
R2_on = (np.double(won_)*np.double(R2_on))+np.double(R2_on)

R3_off = (np.double(woff_)*np.double(R3_off))+np.double(R3_off)
R3_on = (np.double(won_)*np.double(R3_on))+np.double(R3_on)

R4_off = (np.double(woff_)*np.double(R4_off))+np.double(R4_off)
R4_on = (np.double(won_)*np.double(R4_on))+np.double(R4_on)

#######################################################
## evaluation #########################################
#######################################################

R1_eval_, R2_eval_, R3_eval_, R4_eval_ = [],[],[],[]

for ii in range(0,len(contrasts)):
	R1offbon, R1same_, R1onboff = percentage_eval(R1_off[ii,:], R1_on[ii,:])
	R1_eval_.append([R1offbon, R1same_, R1onboff])

	R2offbon, R2same_, R2onboff = percentage_eval(R2_off[ii,:], R2_on[ii,:])
	R2_eval_.append([R2offbon, R2same_, R2onboff])

	R3offbon, R3same_, R3onboff = percentage_eval(R3_off[ii,:], R3_on[ii,:])
	R3_eval_.append([R3offbon, R3same_, R3onboff])

	R4offbon, R4same_, R4onboff = percentage_eval(R4_off[ii,:], R4_on[ii,:])
	R4_eval_.append([R4offbon, R4same_, R4onboff])

R1_eval_, R2_eval_,R3_eval_,R4_eval_ = np.float64(np.asarray(R1_eval_)), np.float64(np.asarray(R2_eval_)), np.float64(np.asarray(R3_eval_)), np.float64(np.asarray(R4_eval_))
# print(R1_eval_.shape, R1_eval_[0,:])
# print(R2_eval_.shape, R2_eval_[0,:])
# print(R3_eval_.shape, R3_eval_[0,:])
# print(R4_eval_.shape, R4_eval_[0,:])
# x = np.array([1.,2.,3.])
#######################################################
## PLOT ###############################################
#######################################################
coff_ = "#3274A1"##"#333333"
con_ = "#E1812C"####"#b2b2b2"
csame_ ="#ffffff"#"#b2b2b2" ###"#01d1a1"
cedge_ ="#000000"

blim_ = -90.
uplim_ = 90.
width = 0.5
fig = plt.figure(figsize = (10,4))##1)
grid = plt.GridSpec(2, 4, hspace=0.2, wspace=0.2)
line_labels = [ "PMC OFF better\nthan PMC ON", 
			    "PMC ON better\n than PMC OFF"]
# Create the bar labels
bar_labels = ['T1-w', 'T2-w', 'PD-w', 'T2*-w (05)','T2*-w (035)','T2*-w (025)','ALL CONT.']
bar_finlabels = ['T1-w', 'T2-w', 'PD-w', 'T2*-w\n(05)','T2*-w\n(035)','T2*-w\n(025)','ALL\nCONT.']
assey_ = [1,2,3,4,5,6,7]
## READER 1 ###
ax1 = plt.subplot(grid[0,0])# plt.subplot(2,4,1)
read1off_ = [R1_eval_[0,0],R1_eval_[1,0],R1_eval_[2,0],R1_eval_[3,0],R1_eval_[4,0],R1_eval_[5,0],
			np.mean([R1_eval_[0,0],R1_eval_[1,0],R1_eval_[2,0],R1_eval_[3,0],R1_eval_[4,0],R1_eval_[5,0]])]
read1on_ = [R1_eval_[0,2],R1_eval_[1,2],R1_eval_[2,2],R1_eval_[3,2],R1_eval_[4,2],R1_eval_[5,2],
			np.mean([R1_eval_[0,2],R1_eval_[1,2],R1_eval_[2,2],R1_eval_[3,2],R1_eval_[4,2],R1_eval_[5,2]])]
err1off_ = [0.,0.,0.,0.,0.,0.,
			np.std([R1_eval_[0,0],R1_eval_[1,0],R1_eval_[2,0],R1_eval_[3,0],R1_eval_[4,0],R1_eval_[5,0]])]
err1on_ = [0.,0.,0.,0.,0.,0.,
			np.std([R1_eval_[0,2],R1_eval_[1,2],R1_eval_[2,2],R1_eval_[3,2],R1_eval_[4,2],R1_eval_[5,2]])]

l1 = ax1.barh(assey_, read1off_, xerr=err1off_,  color=coff_ , edgecolor=cedge_, capsize=3)
l2 = ax1.barh(assey_, read1on_,   xerr=err1on_, color= con_, edgecolor=cedge_, capsize=3)  #, hatch='/' )
plt.yticks(assey_, bar_labels, fontsize=10, fontweight='bold')
ax1.set_xticklabels([])
ax1.set_xlim(blim_, uplim_)
plt.title('READER 1', fontsize=10, fontweight='bold')

### ----------------------------------------------------------
### ----------------------------------------------------------
### ----------------------------------------------------------
fig.legend([l1, l2],     # The line objects
           labels=line_labels,   # The labels for each line
           loc="upper right",# "upper center",   # Position of legend
           # borderaxespad=0.1,    # Small spacing around legend box
           #title=""# "Legend Title"  # Title for the legend
           fontsize=8)

ax1.grid(True)
### ----------------------------------------------------------
### ----------------------------------------------------------
### ----------------------------------------------------------
## READER 2 ###
ax1 = plt.subplot(grid[0,1])#plt.subplot(2,4,2)
read2off_ = [R2_eval_[0,0],R2_eval_[1,0],R2_eval_[2,0],R2_eval_[3,0],R2_eval_[4,0],R2_eval_[5,0],
			np.mean([R2_eval_[0,0],R2_eval_[1,0],R2_eval_[2,0],R2_eval_[3,0],R2_eval_[4,0],R2_eval_[5,0]])]
read2on_ =  [R2_eval_[0,2],R2_eval_[1,2],R2_eval_[2,2],R2_eval_[3,2],R2_eval_[4,2],R2_eval_[5,2],
			np.mean([R2_eval_[0,2],R2_eval_[1,2],R2_eval_[2,2],R2_eval_[3,2],R2_eval_[4,2],R2_eval_[5,2]])]

err2off_ = [0.,0.,0.,0.,0.,0.,
			np.std([R2_eval_[0,0],R2_eval_[1,0],R2_eval_[2,0],R2_eval_[3,0],R2_eval_[4,0],R2_eval_[5,0]])]
err2on_ = [0.,0.,0.,0.,0.,0.,
			np.std([R2_eval_[0,2],R2_eval_[1,2],R2_eval_[2,2],R2_eval_[3,2],R2_eval_[4,2],R2_eval_[5,2]])]
l1 = ax1.barh(assey_, read2off_, xerr=err2off_,  color=coff_ , edgecolor=cedge_, capsize=3)
l2 = ax1.barh(assey_, read2on_, xerr=err2on_,    color= con_, edgecolor=cedge_, capsize=3)  #, hatch='/' )
ax1.set_xlim(blim_, uplim_)
ax1.set_xticklabels([])
plt.yticks(assey_, [], fontsize=10)
plt.title('READER 2', fontsize=10, fontweight='bold')
ax1.grid(True)

## READER 3 ###
ax1 =plt.subplot(grid[1,0])# plt.subplot(2,4,5)
read3off_ = [R3_eval_[0,0],R3_eval_[1,0],R3_eval_[2,0],R3_eval_[3,0],R3_eval_[4,0],R3_eval_[5,0],
			np.mean([R3_eval_[0,0],R3_eval_[1,0],R3_eval_[2,0],R3_eval_[3,0],R3_eval_[4,0],R3_eval_[5,0]])]
read3on_ = [R3_eval_[0,2],R3_eval_[1,2],R3_eval_[2,2],R3_eval_[3,2],R3_eval_[4,2],R3_eval_[5,2],
			np.mean([R3_eval_[0,2],R3_eval_[1,2],R3_eval_[2,2],R3_eval_[3,2],R3_eval_[4,2],R3_eval_[5,2]])]

err3off_ = [0.,0.,0.,0.,0.,0.,
			np.std([R3_eval_[0,0],R3_eval_[1,0],R3_eval_[2,0],R3_eval_[3,0],R3_eval_[4,0],R3_eval_[5,0]])]
err3on_ = [0.,0.,0.,0.,0.,0.,
			np.std([R3_eval_[0,2],R3_eval_[1,2],R3_eval_[2,2],R3_eval_[3,2],R3_eval_[4,2],R3_eval_[5,2]])]
l1 = ax1.barh(assey_, read3off_, xerr=err3off_,  color=coff_ , edgecolor=cedge_, capsize=3)
l2 = ax1.barh(assey_, read3on_,xerr=err3on_,    color= con_, edgecolor=cedge_, capsize=3)  #, hatch='/' )
ax1.set_xlim(blim_, uplim_)
plt.xticks(fontsize=10, fontweight='bold')
plt.yticks(assey_, bar_labels, fontsize=10, fontweight='bold')
plt.xlabel('(%)', fontsize=10, fontweight='bold')
plt.title('READER 3', fontsize=10, fontweight='bold')
ax1.grid(True)

## READER 4 ###
ax1 = plt.subplot(grid[1,1])#plt.subplot(2,4,6)
read4off_ = [R4_eval_[0,0],R4_eval_[1,0],R4_eval_[2,0],R4_eval_[3,0],R4_eval_[4,0],R4_eval_[5,0],
			np.mean([R4_eval_[0,0],R4_eval_[1,0],R4_eval_[2,0],R4_eval_[3,0],R4_eval_[4,0],R4_eval_[5,0]])]
read4on_ = [R4_eval_[0,2],R4_eval_[1,2],R4_eval_[2,2],R4_eval_[3,2],R4_eval_[4,2],R4_eval_[5,2],
			np.mean([R4_eval_[0,2],R4_eval_[1,2],R4_eval_[2,2],R4_eval_[3,2],R4_eval_[4,2],R4_eval_[5,2]])]
err4off_ = [0.,0.,0.,0.,0.,0.,
			np.std([R4_eval_[0,0],R4_eval_[1,0],R4_eval_[2,0],R4_eval_[3,0],R4_eval_[4,0],R4_eval_[5,0]])]
err4on_ = [0.,0.,0.,0.,0.,0.,
			np.std([R4_eval_[0,2],R4_eval_[1,2],R4_eval_[2,2],R4_eval_[3,2],R4_eval_[4,2],R4_eval_[5,2]])]
l1 = ax1.barh(assey_, read4off_, xerr=err4off_,  color=coff_ , edgecolor=cedge_ , capsize=3)
l2 = ax1.barh(assey_, read4on_, xerr=err4on_, color= con_, edgecolor=cedge_, capsize=3)  #, hatch='/' )
ax1.set_xlim(blim_, uplim_)
plt.xticks(fontsize=10, fontweight='bold')
plt.yticks(assey_, [], fontsize=10)
plt.xlabel('(%)', fontsize=10, fontweight='bold')
plt.title('READER 4', fontsize=10, fontweight='bold')
ax1.grid(True)
########################################################
## ALL READERS #########################################
########################################################

ax1 = plt.subplot(grid[:,2:])#plt.subplot(2,4,6)
readsoff_ = np.asarray([np.mean([R1_eval_[0,0],R2_eval_[0,0],R3_eval_[0,0],R4_eval_[0,0]]),
						np.mean([R1_eval_[1,0],R2_eval_[1,0],R3_eval_[1,0],R4_eval_[1,0]]),
						np.mean([R1_eval_[2,0],R2_eval_[2,0],R3_eval_[2,0],R4_eval_[2,0]]),
						np.mean([R1_eval_[3,0],R2_eval_[3,0],R3_eval_[3,0],R4_eval_[3,0]]),
						np.mean([R1_eval_[4,0],R2_eval_[4,0],R3_eval_[4,0],R4_eval_[4,0]]),
						np.mean([R1_eval_[5,0],R2_eval_[5,0],R3_eval_[5,0],R4_eval_[5,0]])])

readsoff_ = np.hstack((readsoff_, np.mean(readsoff_)))

readson_ = np.asarray([np.mean([R1_eval_[0,2],R2_eval_[0,2],R3_eval_[0,2],R4_eval_[0,2]]),
						np.mean([R1_eval_[1,2],R2_eval_[1,2],R3_eval_[1,2],R4_eval_[1,2]]),
						np.mean([R1_eval_[2,2],R2_eval_[2,2],R3_eval_[2,2],R4_eval_[2,2]]),
						np.mean([R1_eval_[3,2],R2_eval_[3,2],R3_eval_[3,2],R4_eval_[3,2]]),
						np.mean([R1_eval_[4,2],R2_eval_[4,2],R3_eval_[4,2],R4_eval_[4,2]]),
						np.mean([R1_eval_[5,2],R2_eval_[5,2],R3_eval_[5,2],R4_eval_[5,2]])])
readson_ = np.hstack((readson_, np.mean(readson_)))

xerroroff_ =  np.asarray([np.std([R1_eval_[0,0],R2_eval_[0,0],R3_eval_[0,0],R4_eval_[0,0]]),
						np.std([R1_eval_[1,0],R2_eval_[1,0],R3_eval_[1,0],R4_eval_[1,0]]),
						np.std([R1_eval_[2,0],R2_eval_[2,0],R3_eval_[2,0],R4_eval_[2,0]]),
						np.std([R1_eval_[3,0],R2_eval_[3,0],R3_eval_[3,0],R4_eval_[3,0]]),
						np.std([R1_eval_[4,0],R2_eval_[4,0],R3_eval_[4,0],R4_eval_[4,0]]),
						np.std([R1_eval_[5,0],R2_eval_[5,0],R3_eval_[5,0],R4_eval_[5,0]])])
xerroroff_ = np.hstack((xerroroff_, np.mean(xerroroff_)))

xerroron_ = np.asarray([np.std([R1_eval_[0,2],R2_eval_[0,2],R3_eval_[0,2],R4_eval_[0,2]]),
						np.std([R1_eval_[1,2],R2_eval_[1,2],R3_eval_[1,2],R4_eval_[1,2]]),
						np.std([R1_eval_[2,2],R2_eval_[2,2],R3_eval_[2,2],R4_eval_[2,2]]),
						np.std([R1_eval_[3,2],R2_eval_[3,2],R3_eval_[3,2],R4_eval_[3,2]]),
						np.std([R1_eval_[4,2],R2_eval_[4,2],R3_eval_[4,2],R4_eval_[4,2]]),
						np.std([R1_eval_[5,2],R2_eval_[5,2],R3_eval_[5,2],R4_eval_[5,2]])])
xerroron_ = np.hstack((xerroron_, np.mean(xerroron_)))

l1 = ax1.barh(assey_, readsoff_, xerr=xerroroff_, color=coff_ , edgecolor=cedge_, capsize=3)
l2 = ax1.barh(assey_, readson_,  xerr=xerroron_, color= con_, edgecolor=cedge_, capsize=3)  #, hatch='/' )
ax1.set_xlim(blim_, uplim_)
ax1.yaxis.tick_right()
plt.xticks(fontsize=10, fontweight='bold')
plt.yticks(assey_, bar_finlabels, fontsize=10, fontweight='bold')
plt.xlabel('(%)', fontsize=10, fontweight='bold')
plt.title('ALL READERS', fontsize=12, fontweight='bold')


#######################################################
# compare samples #####################################
#######################################################
t1_off_ = np.concatenate((R1_off[0,:], R2_off[0,:], R3_off[0,:], R4_off[0,:]))
t2_off_ = np.concatenate((R1_off[1,:], R2_off[1,:], R3_off[1,:], R4_off[1,:]))
pd_off_ = np.concatenate((R1_off[2,:], R2_off[2,:], R3_off[2,:], R4_off[2,:]))
t2s05_off_ = np.concatenate((R1_off[3,:], R2_off[3,:], R3_off[3,:], R4_off[3,:]))
t2s035_off_ = np.concatenate((R1_off[4,:], R2_off[4,:], R3_off[4,:], R4_off[4,:]))
t2s025_off_ = np.concatenate((R1_off[5,:], R2_off[5,:], R3_off[5,:], R4_off[5,:]))

t1_on_ = np.concatenate((R1_on[0,:], R2_on[0,:], R3_on[0,:], R4_on[0,:]))
t2_on_ = np.concatenate((R1_on[1,:], R2_on[1,:], R3_on[1,:], R4_on[1,:]))
pd_on_ = np.concatenate((R1_on[2,:], R2_on[2,:], R3_on[2,:], R4_on[2,:]))
t2s05_on_ = np.concatenate((R1_on[3,:], R2_on[3,:], R3_on[3,:], R4_on[3,:]))
t2s035_on_ = np.concatenate((R1_on[4,:], R2_on[4,:], R3_on[4,:], R4_on[4,:]))
t2s025_on_ = np.concatenate((R1_on[5,:], R2_on[5,:], R3_on[5,:], R4_on[5,:]))


stackoffall_ = np.concatenate((t1_off_, t2_off_, pd_off_, t2s05_off_, t2s035_off_, t2s025_off_))
stackonall_ = np.concatenate((t1_on_, t2_on_, pd_on_, t2s05_on_, t2s035_on_, t2s025_on_))

stackoff_ = np.array([t1_off_, t2_off_, pd_off_, t2s05_off_, t2s035_off_, t2s025_off_, stackoffall_])
stackon_ = np.array([t1_on_, t2_on_, pd_on_, t2s05_on_, t2s035_on_, t2s025_on_, stackonall_])

pvalues = []
for ii in range(0, len(stackoff_)):
    stat, p = mannwhitneyu(stackoff_[ii], stackon_[ii])
    print('Statistics=%.3f, p=%.5f' % (stat, p))
    pvalues.append(p)
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Same distribution (fail to reject H0)')
    else:
        print('Different distribution (reject H0)')

# print(pvalues)
########################################################################################
########################################################################################
don_, doff_  = 25, -3
sh_ = 0.15

props = dict(boxstyle='round', facecolor='wheat', alpha=1)
for ii in range(0, len(readson_)):
    ax1.text(readson_[ii]-don_, ii+1+sh_+0.05,str(truncate(readson_[ii],2)), fontweight='bold', fontsize=9, bbox=props)
    ax1.text(readsoff_[ii]-doff_,ii+1+sh_+0.05,str(truncate(readsoff_[ii]*(-1),2)), fontweight='bold', fontsize=9, bbox=props)
    ax1.text(blim_-2.5, ii+1,str(convert_pval(pvalues[ii])), fontweight='bold', fontsize=7, bbox=props)
########################################################################################
########################################################################################
ax1.grid(True)
plt.show()

########################################################################################
########################################################################################
########################################################################################
"""
x = np.array([1.,2.,3.])

coff_ = "#3274A1"##"#333333"
con_ = "#E1812C"####"#b2b2b2"
csame_ ="#ffffff"#"#b2b2b2" ###"#01d1a1"
cedge_ ="#000000"

blim_ = -75.
uplim_ = 75.
width = 0.5
# Create the legend

#fig.legend([l1, l2, l3, l4],     # The line objects
#           labels=line_labels,   # The labels for each line
#           loc="center right",   # Position of legend
#           borderaxespad=0.1,    # Small spacing around legend box
#           title="Legend Title"  # Title for the legend
#           )

fig = plt.figure(figsize = (7,7))##1)
line_labels = [ "PMC OFF received a better score than PMC ON", 
			    "PMC OFF and PMC ON received the same score", 
			    "PMC ON received a better score than PMC OFF"]
# ---------------------------------------------------------------------------------- #########
## READER 1 ###
## T1-w ####
ax1 = plt.subplot(571)
l1 = ax1.bar(1, np.float64(R1_eval_[0,0]), width=0.25, color=coff_ , edgecolor=cedge_)
l2 = ax1.bar(1.25, np.float64(R1_eval_[0,1]), width=0.25, color=csame_, edgecolor=cedge_)
l3 = ax1.bar(1.5, np.float64(R1_eval_[0,2]), width=0.25, color= con_, edgecolor=cedge_)  #, hatch='/' )
ax1.set_ylim(blim_, uplim_)
ax1.set_xticklabels([])
ax1.set_xticks([])
# ax1.set_xlabel('T1-w', fontsize=8, fontweight='bold')
ax1.set_ylabel('READER 1', rotation=90, fontsize=8, fontweight='bold')
fig.legend([l1, l2, l3],     # The line objects
           labels=line_labels,   # The labels for each line
           loc="upper center",# "upper center",   # Position of legend
           # borderaxespad=0.1,    # Small spacing around legend box
           #title=""# "Legend Title"  # Title for the legend
           fontsize=8
           
           )
ax1.text(0.465,68.5,'(%)', fontweight='bold')
plt.yticks(fontweight='bold')
ax1.grid(True)

## T2-w ####
ax2 = plt.subplot(572)
ax2.bar(1, np.float64(R1_eval_[1,0]), width=0.25, color=coff_ , edgecolor=cedge_)
ax2.bar(1.25, np.float64(R1_eval_[1,1]), width=0.25, color=csame_, edgecolor=cedge_)
ax2.bar(1.5, np.float64(R1_eval_[1,2]), width=0.25, color= con_, edgecolor=cedge_)  #, hatch='/' )
ax2.set_ylim(blim_, uplim_)
ax2.set_xticklabels([])
ax2.set_xticks([])
ax2.set_yticklabels([])
# ax2.set_xlabel('T2-w', fontsize=8, fontweight='bold')
ax2.grid(True)

## PD-w ####
ax3 = plt.subplot(573)
ax3.bar(1, np.float64(R1_eval_[2,0]), width=0.25, color=coff_ , edgecolor=cedge_)
ax3.bar(1.25, np.float64(R1_eval_[2,1]), width=0.25, color=csame_, edgecolor=cedge_)
ax3.bar(1.5, np.float64(R1_eval_[2,2]), width=0.25, color= con_, edgecolor=cedge_)  #, hatch='/' )
ax3.set_ylim(blim_, uplim_)
ax3.set_xticklabels([])
ax3.set_xticks([])
ax3.set_yticklabels([])
# ax3.set_xlabel('PD-w', fontsize=8, fontweight='bold')
ax3.grid(True)

## T2s05-w ####
ax4 = plt.subplot(574)
ax4.bar(1, np.float64(R1_eval_[3,0]), width=0.25, color=coff_ , edgecolor=cedge_)
ax4.bar(1.25, np.float64(R1_eval_[3,1]), width=0.25, color=csame_, edgecolor=cedge_)
ax4.bar(1.5, np.float64(R1_eval_[3,2]), width=0.25, color= con_, edgecolor=cedge_)  #, hatch='/' )
ax4.set_ylim(blim_, uplim_)
ax4.set_xticklabels([])
ax4.set_xticks([])
ax4.set_yticklabels([])
# ax4.set_xlabel('T2*-w (05)', fontsize=8, fontweight='bold')
ax4.grid(True)

## T2s035-w ####
ax5 = plt.subplot(575)
ax5.bar(1, np.float64(R1_eval_[4,0]), width=0.25, color=coff_ , edgecolor=cedge_)
ax5.bar(1.25, np.float64(R1_eval_[4,1]), width=0.25, color=csame_, edgecolor=cedge_)
ax5.bar(1.5, np.float64(R1_eval_[4,2]), width=0.25, color= con_, edgecolor=cedge_)  #, hatch='/' )
ax5.set_ylim(blim_, uplim_)
ax5.set_xticklabels([])
ax5.set_xticks([])
ax5.set_yticklabels([])
# ax5.set_xlabel('T2*-w (035)', fontsize=8, fontweight='bold')
ax5.grid(True)

## T2s025-w ####
ax6 = plt.subplot(576)
ax6.bar(1, np.float64(R1_eval_[5,0]), width=0.25, color=coff_ , edgecolor=cedge_)
ax6.bar(1.25, np.float64(R1_eval_[5,1]), width=0.25, color=csame_, edgecolor=cedge_)
ax6.bar(1.5, np.float64(R1_eval_[5,2]), width=0.25, color= con_, edgecolor=cedge_)  #, hatch='/' )
ax6.set_ylim(blim_, uplim_)
ax6.set_xticklabels([])
ax6.set_xticks([])
ax6.set_yticklabels([])
# ax6.set_xlabel('T2*-w (025)', fontsize=8, fontweight='bold')
ax6.grid(True)

## average all contrasts R1 ####
ax7 = plt.subplot(577)
# print(np.mean(np.float64(R1_eval_[:,0])))
ax7.bar(1, np.mean(np.float64(R1_eval_[:,0])), width=0.25, color=coff_ , edgecolor=cedge_ , yerr=np.std(np.float64(R1_eval_[:,0])))
ax7.bar(1.25, np.mean(np.float64(R1_eval_[:,1])), width=0.25, color=csame_, edgecolor=cedge_, yerr=np.std(np.float64(R1_eval_[:,1])))
ax7.bar(1.5, np.mean(np.float64(R1_eval_[:,2])), width=0.25, color= con_, edgecolor=cedge_, yerr=np.std(np.float64(R1_eval_[:,2])))  #, hatch='/' )
ax7.set_ylim(blim_, uplim_)
ax7.set_xticklabels([])
ax7.set_xticks([])
ax7.set_yticklabels([])
# ax7.set_xlabel('Average', fontsize=8, fontweight='bold')
ax7.grid(True)

# --------------------------------------------------- #
# ---------------------------------------------------------------------------------- #########
## READER 2 ###
## T1-w ####
ax8 = plt.subplot(578)
ax8.bar(1, np.float64(R2_eval_[0,0]), width=0.25, color=coff_ , edgecolor=cedge_)
ax8.bar(1.25, np.float64(R2_eval_[0,1]), width=0.25, color=csame_, edgecolor=cedge_)
ax8.bar(1.5, np.float64(R2_eval_[0,2]), width=0.25, color= con_, edgecolor=cedge_)  #, hatch='/' )
ax8.set_ylim(blim_, uplim_)
ax8.set_xticklabels([])
ax8.set_xticks([])
# ax8.set_xlabel('T1-w', fontsize=8, fontweight='bold')
ax8.set_ylabel('READER 2', rotation=90, fontsize=8, fontweight='bold')
ax8.text(0.465,68.5,'(%)', fontweight='bold')
plt.yticks(fontweight='bold')
ax8.grid(True)

## T2-w ####
ax9 = plt.subplot(579)
ax9.bar(1, np.float64(R2_eval_[1,0]), width=0.25, color=coff_ , edgecolor=cedge_)
ax9.bar(1.25, np.float64(R2_eval_[1,1]), width=0.25, color=csame_, edgecolor=cedge_)
ax9.bar(1.5, np.float64(R2_eval_[1,2]), width=0.25, color= con_, edgecolor=cedge_)  #, hatch='/' )
ax9.set_ylim(blim_, uplim_)
ax9.set_xticklabels([])
ax9.set_xticks([])
ax9.set_yticklabels([])
# ax9.set_xlabel('T2-w', fontsize=8, fontweight='bold')
ax9.grid(True)

## PD-w ####
ax10 = plt.subplot(5,7,10)
ax10.bar(1, np.float64(R2_eval_[2,0]), width=0.25, color=coff_ , edgecolor=cedge_)
ax10.bar(1.25, np.float64(R2_eval_[2,1]), width=0.25, color=csame_, edgecolor=cedge_)
ax10.bar(1.5, np.float64(R2_eval_[2,2]), width=0.25, color= con_, edgecolor=cedge_)  #, hatch='/' )
ax10.set_ylim(blim_, uplim_)
ax10.set_xticklabels([])
ax10.set_xticks([])
ax10.set_yticklabels([])
# ax10.set_xlabel('PD-w', fontsize=8, fontweight='bold')
ax10.grid(True)

## T2s05-w ####
ax11 = plt.subplot(5,7,11)
ax11.bar(1, np.float64(R2_eval_[3,0]), width=0.25, color=coff_ , edgecolor=cedge_)
ax11.bar(1.25, np.float64(R2_eval_[3,1]), width=0.25, color=csame_, edgecolor=cedge_)
ax11.bar(1.5, np.float64(R2_eval_[3,2]), width=0.25, color= con_, edgecolor=cedge_)  #, hatch='/' )
ax11.set_ylim(blim_, uplim_)
ax11.set_xticklabels([])
ax11.set_xticks([])
ax11.set_yticklabels([])
# ax11.set_xlabel('T2*-w (05)', fontsize=8, fontweight='bold')
ax11.grid(True)

## T2s035-w ####
ax12 = plt.subplot(5,7,12)
ax12.bar(1, np.float64(R2_eval_[4,0]), width=0.25, color=coff_ , edgecolor=cedge_)
ax12.bar(1.25, np.float64(R2_eval_[4,1]), width=0.25, color=csame_, edgecolor=cedge_)
ax12.bar(1.5, np.float64(R2_eval_[4,2]), width=0.25, color= con_, edgecolor=cedge_)  #, hatch='/' )
ax12.set_ylim(blim_, uplim_)
ax12.set_xticklabels([])
ax12.set_xticks([])
ax12.set_yticklabels([])
# ax12.set_xlabel('T2*-w (035)', fontsize=8, fontweight='bold')
ax12.grid(True)

## T2s025-w ####
ax13 = plt.subplot(5,7,13)
ax13.bar(1, np.float64(R2_eval_[5,0]), width=0.25, color=coff_ , edgecolor=cedge_)
ax13.bar(1.25, np.float64(R2_eval_[5,1]), width=0.25, color=csame_, edgecolor=cedge_)
ax13.bar(1.5, np.float64(R2_eval_[5,2]), width=0.25, color= con_, edgecolor=cedge_)  #, hatch='/' )
ax13.set_ylim(blim_, uplim_)
ax13.set_xticklabels([])
ax13.set_xticks([])
ax13.set_yticklabels([])
# ax13.set_xlabel('T2*-w (025)', fontsize=8, fontweight='bold')
ax13.grid(True)

## average all contrasts R1 ####
ax14 = plt.subplot(5,7,14)
# print(np.mean(np.float64(R2_eval_[:,0])))
ax14.bar(1, np.mean(np.float64(R2_eval_[:,0])), width=0.25, color=coff_ , edgecolor=cedge_ , yerr=np.std(np.float64(R2_eval_[:,0])))
ax14.bar(1.25, np.mean(np.float64(R2_eval_[:,1])), width=0.25, color=csame_, edgecolor=cedge_, yerr=np.std(np.float64(R2_eval_[:,1])))
ax14.bar(1.5, np.mean(np.float64(R2_eval_[:,2])), width=0.25, color= con_, edgecolor=cedge_, yerr=np.std(np.float64(R2_eval_[:,2])))  #, hatch='/' )
ax14.set_ylim(blim_, uplim_)
ax14.set_xticklabels([])
ax14.set_xticks([])
ax14.set_yticklabels([])
# ax14.set_xlabel('Average for Reader 2', fontsize=8, fontweight='bold')
ax14.grid(True)

# --------------------------------------------------- #
# --------------------------------------------------- #
# ---------------------------------------------------------------------------------- #########
## READER 3 ###
## T1-w ####
ax15 = plt.subplot(5,7,15)
ax15.bar(1, np.float64(R3_eval_[0,0]), width=0.25, color=coff_ , edgecolor=cedge_)
ax15.bar(1.25, np.float64(R3_eval_[0,1]), width=0.25, color=csame_, edgecolor=cedge_)
ax15.bar(1.5, np.float64(R3_eval_[0,2]), width=0.25, color= con_, edgecolor=cedge_)  #, hatch='/' )
ax15.set_ylim(blim_, uplim_)
ax15.set_xticklabels([])
ax15.set_xticks([])
# ax15.set_xlabel('T1-w', fontsize=8, fontweight='bold')
ax15.set_ylabel('READER 3', rotation=90, fontsize=8, fontweight='bold')
ax15.text(0.465,68.5,'(%)', fontweight='bold')
plt.yticks(fontweight='bold')
ax15.grid(True)

## T2-w ####
ax16 = plt.subplot(5,7,16)
ax16.bar(1, np.float64(R3_eval_[1,0]), width=0.25, color=coff_ , edgecolor=cedge_)
ax16.bar(1.25, np.float64(R3_eval_[1,1]), width=0.25, color=csame_, edgecolor=cedge_)
ax16.bar(1.5, np.float64(R3_eval_[1,2]), width=0.25, color= con_, edgecolor=cedge_)  #, hatch='/' )
ax16.set_ylim(blim_, uplim_)
ax16.set_xticklabels([])
ax16.set_xticks([])
ax16.set_yticklabels([])
# ax16.set_xlabel('T2-w', fontsize=8, fontweight='bold')
ax16.grid(True)

## PD-w ####
ax17 = plt.subplot(5,7,17)
ax17.bar(1, np.float64(R3_eval_[2,0]), width=0.25, color=coff_ , edgecolor=cedge_)
ax17.bar(1.25, np.float64(R3_eval_[2,1]), width=0.25, color=csame_, edgecolor=cedge_)
ax17.bar(1.5, np.float64(R3_eval_[2,2]), width=0.25, color= con_, edgecolor=cedge_)  #, hatch='/' )
ax17.set_ylim(blim_, uplim_)
ax17.set_xticklabels([])
ax17.set_xticks([])
ax17.set_yticklabels([])
# ax17.set_xlabel('PD-w', fontsize=8, fontweight='bold')
ax17.grid(True)

## T2s05-w ####
ax18 = plt.subplot(5,7,18)
ax18.bar(1, np.float64(R3_eval_[3,0]), width=0.25, color=coff_ , edgecolor=cedge_)
ax18.bar(1.25, np.float64(R3_eval_[3,1]), width=0.25, color=csame_, edgecolor=cedge_)
ax18.bar(1.5, np.float64(R3_eval_[3,2]), width=0.25, color= con_, edgecolor=cedge_)  #, hatch='/' )
ax18.set_ylim(blim_, uplim_)
ax18.set_xticklabels([])
ax18.set_xticks([])
ax18.set_yticklabels([])
# ax18.set_xlabel('T2*-w (05)', fontsize=8, fontweight='bold')
ax18.grid(True)

## T2s035-w ####
ax19 = plt.subplot(5,7,19)
ax19.bar(1, np.float64(R3_eval_[4,0]), width=0.25, color=coff_ , edgecolor=cedge_)
ax19.bar(1.25, np.float64(R3_eval_[4,1]), width=0.25, color=csame_, edgecolor=cedge_)
ax19.bar(1.5, np.float64(R3_eval_[4,2]), width=0.25, color= con_, edgecolor=cedge_)  #, hatch='/' )
ax19.set_ylim(blim_, uplim_)
ax19.set_xticklabels([])
ax19.set_xticks([])
ax19.set_yticklabels([])
# ax19.set_xlabel('T2*-w (035)', fontsize=8, fontweight='bold')
ax19.grid(True)

## T2s025-w ####
ax20 = plt.subplot(5,7,20)
ax20.bar(1, np.float64(R3_eval_[5,0]), width=0.25, color=coff_ , edgecolor=cedge_)
ax20.bar(1.25, np.float64(R3_eval_[5,1]), width=0.25, color=csame_, edgecolor=cedge_)
ax20.bar(1.5, np.float64(R3_eval_[5,2]), width=0.25, color= con_, edgecolor=cedge_)  #, hatch='/' )
ax20.set_ylim(blim_, uplim_)
ax20.set_xticklabels([])
ax20.set_xticks([])
ax20.set_yticklabels([])
# ax20.set_xlabel('T2*-w (025)', fontsize=8, fontweight='bold')
ax20.grid(True)

## average all contrasts R3 ####
ax21 = plt.subplot(5,7,21)
# print(np.mean(np.float64(R3_eval_[:,0])))
ax21.bar(1, np.mean(np.float64(R3_eval_[:,0])), width=0.25, color=coff_ , edgecolor=cedge_ , yerr=np.std(np.float64(R3_eval_[:,0])))
ax21.bar(1.25, np.mean(np.float64(R3_eval_[:,1])), width=0.25, color=csame_, edgecolor=cedge_, yerr=np.std(np.float64(R3_eval_[:,1])))
ax21.bar(1.5, np.mean(np.float64(R3_eval_[:,2])), width=0.25, color= con_, edgecolor=cedge_, yerr=np.std(np.float64(R3_eval_[:,2])))  #, hatch='/' )
ax21.set_ylim(blim_, uplim_)
ax21.set_xticklabels([])
ax21.set_xticks([])
ax21.set_yticklabels([])
# ax21.set_xlabel('Average for Reader 3', fontsize=8, fontweight='bold')
ax21.grid(True)

# --------------------------------------------------- #
# ---------------------------------------------------------------------------------- #########
## READER 4 ###
## T1-w ####
ax22 = plt.subplot(5,7,22)
ax22.bar(1, np.float64(R4_eval_[0,0]), width=0.25, color=coff_ , edgecolor=cedge_)
ax22.bar(1.25, np.float64(R4_eval_[0,1]), width=0.25, color=csame_, edgecolor=cedge_)
ax22.bar(1.5, np.float64(R4_eval_[0,2]), width=0.25, color= con_, edgecolor=cedge_)  #, hatch='/' )
ax22.set_ylim(blim_, uplim_)
ax22.set_xticklabels([])
ax22.set_xticks([])
# ax22.set_xlabel('T1-w', fontsize=8, fontweight='bold')
ax22.set_ylabel('READER 4', rotation=90, fontsize=8, fontweight='bold')
ax22.text(0.465,68.5,'(%)', fontweight='bold')
plt.yticks(fontweight='bold')
ax22.grid(True)

## T2-w ####
ax23 = plt.subplot(5,7,23)
ax23.bar(1, np.float64(R4_eval_[1,0]), width=0.25, color=coff_ , edgecolor=cedge_)
ax23.bar(1.25, np.float64(R4_eval_[1,1]), width=0.25, color=csame_, edgecolor=cedge_)
ax23.bar(1.5, np.float64(R4_eval_[1,2]), width=0.25, color= con_, edgecolor=cedge_)  #, hatch='/' )
ax23.set_ylim(blim_, uplim_)
ax23.set_xticklabels([])
ax23.set_xticks([])
ax23.set_yticklabels([])
# ax23.set_xlabel('T2-w', fontsize=8, fontweight='bold')
ax23.grid(True)

## PD-w ####
ax24 = plt.subplot(5,7,24)
ax24.bar(1, np.float64(R4_eval_[2,0]), width=0.25, color=coff_ , edgecolor=cedge_)
ax24.bar(1.25, np.float64(R4_eval_[2,1]), width=0.25, color=csame_, edgecolor=cedge_)
ax24.bar(1.5, np.float64(R4_eval_[2,2]), width=0.25, color= con_, edgecolor=cedge_)  #, hatch='/' )
ax24.set_ylim(blim_, uplim_)
ax24.set_xticklabels([])
ax24.set_xticks([])
ax24.set_yticklabels([])
# ax24.set_xlabel('PD-w', fontsize=8, fontweight='bold')
ax24.grid(True)

## T2s05-w ####
ax25 = plt.subplot(5,7,25)
ax25.bar(1, np.float64(R4_eval_[3,0]), width=0.25, color=coff_ , edgecolor=cedge_)
ax25.bar(1.25, np.float64(R4_eval_[3,1]), width=0.25, color=csame_, edgecolor=cedge_)
ax25.bar(1.5, np.float64(R4_eval_[3,2]), width=0.25, color= con_, edgecolor=cedge_)  #, hatch='/' )
ax25.set_ylim(blim_, uplim_)
ax25.set_xticklabels([])
ax25.set_xticks([])
ax25.set_yticklabels([])
# ax25.set_xlabel('T2*-w (05)', fontsize=8, fontweight='bold')
ax25.grid(True)

## T2s035-w ####
ax26 = plt.subplot(5,7,26)
ax26.bar(1, np.float64(R4_eval_[4,0]), width=0.25, color=coff_ , edgecolor=cedge_)
ax26.bar(1.25, np.float64(R4_eval_[4,1]), width=0.25, color=csame_, edgecolor=cedge_)
ax26.bar(1.5, np.float64(R4_eval_[4,2]), width=0.25, color= con_, edgecolor=cedge_)  #, hatch='/' )
ax26.set_ylim(blim_, uplim_)
ax26.set_xticklabels([])
ax26.set_xticks([])
ax26.set_yticklabels([])
# ax26.set_xlabel('T2*-w (035)', fontsize=8, fontweight='bold')
ax26.grid(True)

## T2s025-w ####
ax27 = plt.subplot(5,7,27)
ax27.bar(1, np.float64(R4_eval_[5,0]), width=0.25, color=coff_ , edgecolor=cedge_)
ax27.bar(1.25, np.float64(R4_eval_[5,1]), width=0.25, color=csame_, edgecolor=cedge_)
ax27.bar(1.5, np.float64(R4_eval_[5,2]), width=0.25, color= con_, edgecolor=cedge_)  #, hatch='/' )
ax27.set_ylim(blim_, uplim_)
ax27.set_xticklabels([])
ax27.set_xticks([])
ax27.set_yticklabels([])
# ax27.set_xlabel('T2*-w (025)', fontsize=8, fontweight='bold')
ax27.grid(True)

## average all contrasts R4 ####
ax28 = plt.subplot(5,7,28)
# print(np.mean(np.float64(R4_eval_[:,0])))
ax28.bar(1, np.mean(np.float64(R4_eval_[:,0])), width=0.25, color=coff_ , edgecolor=cedge_ , yerr=np.std(np.float64(R4_eval_[:,0])))
ax28.bar(1.25, np.mean(np.float64(R4_eval_[:,1])), width=0.25, color=csame_, edgecolor=cedge_, yerr=np.std(np.float64(R4_eval_[:,1])))
ax28.bar(1.5, np.mean(np.float64(R4_eval_[:,2])), width=0.25, color= con_, edgecolor=cedge_, yerr=np.std(np.float64(R4_eval_[:,2])))  #, hatch='/' )
ax28.set_ylim(blim_, uplim_)
ax28.set_xticklabels([])
ax28.set_xticks([])
ax28.set_yticklabels([])
# ax28.set_xlabel('Average for Reader 4', fontsize=8, fontweight='bold')
ax28.grid(True)

# --------------------------------------------------- #
# --------------------------------------------------- #
# ---------------------------------------------------------------------------------- #########
## Averages per contrast ###
## T1-w ####
ax29 = plt.subplot(5,7,29)

t1_ = np.vstack((np.float64(R1_eval_[0,:]), np.float64(R2_eval_[0,:]), np.float64(R3_eval_[0,:]), np.float64(R4_eval_[0,:])))
# print(t1_.shape,'\n', t1_)
ax29.bar(1, np.mean(t1_[:,0]), width=0.25, color=coff_ , edgecolor=cedge_, yerr=np.std(t1_[:,0]))
ax29.bar(1.25, np.mean(t1_[:,1]), width=0.25, color=csame_, edgecolor=cedge_, yerr=np.std(t1_[:,1]))
ax29.bar(1.5,np.mean(t1_[:,2]), width=0.25, color= con_, edgecolor=cedge_, yerr=np.std(t1_[:,2]))  #, hatch='/' )
ax29.set_ylim(blim_, uplim_)
ax29.set_xticklabels([])
ax29.set_xticks([])
ax29.set_xlabel('T1-w', fontsize=8, fontweight='bold')
ax29.set_ylabel('ALL\n READERS', rotation=90, fontsize=8, fontweight='bold')
ax29.text(0.465,68.5,'(%)', fontweight='bold')
plt.yticks(fontweight='bold')
ax29.grid(True)

## T2-w ####
ax30 = plt.subplot(5,7,30)
t2_ = np.vstack((np.float64(R1_eval_[1,:]), np.float64(R2_eval_[1,:]), np.float64(R3_eval_[1,:]), np.float64(R4_eval_[1,:])))
# print(t2_.shape,'\n', t2_)
ax30.bar(1, np.mean(t2_[:,0]), width=0.25, color=coff_ , edgecolor=cedge_, yerr=np.std(t2_[:,0]))
ax30.bar(1.25, np.mean(t2_[:,1]), width=0.25, color=csame_, edgecolor=cedge_, yerr=np.std(t2_[:,1]))
ax30.bar(1.5, np.mean(t2_[:,2]), width=0.25, color= con_, edgecolor=cedge_, yerr=np.std(t2_[:,2]))  #, hatch='/' )
ax30.set_ylim(blim_, uplim_)
ax30.set_xticklabels([])
ax30.set_xticks([])
ax30.set_yticklabels([])
ax30.set_xlabel('T2-w', fontsize=8, fontweight='bold')
ax30.grid(True)

## PD-w ####
ax31 = plt.subplot(5,7,31)
pd_ = np.vstack((np.float64(R1_eval_[2,:]), np.float64(R2_eval_[2,:]), np.float64(R3_eval_[2,:]), np.float64(R4_eval_[2,:])))
# print(pd_.shape,'\n', pd_)
ax31.bar(1, np.mean(pd_[:,0]), width=0.25, color=coff_ , edgecolor=cedge_, yerr=np.std(pd_[:,0]))
ax31.bar(1.25,np.mean(pd_[:,1]), width=0.25, color=csame_, edgecolor=cedge_, yerr=np.std(pd_[:,1]))
ax31.bar(1.5, np.mean(pd_[:,2]), width=0.25, color= con_, edgecolor=cedge_, yerr=np.std(pd_[:,2]))  #, hatch='/' )
ax31.set_ylim(blim_, uplim_)
ax31.set_xticklabels([])
ax31.set_xticks([])
ax31.set_yticklabels([])
ax31.set_xlabel('PD-w', fontsize=8, fontweight='bold')
ax31.grid(True)

## T2s05-w ####
ax32 = plt.subplot(5,7,32)
t205_ = np.vstack((np.float64(R1_eval_[3,:]), np.float64(R2_eval_[3,:]), np.float64(R3_eval_[3,:]), np.float64(R4_eval_[3,:])))
# print(t205_.shape,'\n', t205_)
ax32.bar(1, np.mean(t205_[:,0]), width=0.25, color=coff_ , edgecolor=cedge_, yerr=np.std(t205_[:,0]))
ax32.bar(1.25, np.mean(t205_[:,1]), width=0.25, color=csame_, edgecolor=cedge_, yerr=np.std(t205_[:,1]))
ax32.bar(1.5, np.mean(t205_[:,2]), width=0.25, color= con_, edgecolor=cedge_, yerr=np.std(t205_[:,2]))  #, hatch='/' )
ax32.set_ylim(blim_, uplim_)
ax32.set_xticklabels([])
ax32.set_xticks([])
ax32.set_yticklabels([])
ax32.set_xlabel('T2*-w \n(05)', fontsize=8, fontweight='bold')
ax32.grid(True)

## T2s035-w ####
ax33 = plt.subplot(5,7,33)
t2035_ = np.vstack((np.float64(R1_eval_[4,:]), np.float64(R2_eval_[4,:]), np.float64(R3_eval_[4,:]), np.float64(R4_eval_[4,:])))
# print(t2035_.shape,'\n', t2035_)
ax33.bar(1, np.mean(t2035_[:,0]), width=0.25, color=coff_ , edgecolor=cedge_, yerr=np.std(t2035_[:,0]))
ax33.bar(1.25, np.mean(t2035_[:,1]), width=0.25, color=csame_, edgecolor=cedge_, yerr=np.std(t2035_[:,1]))
ax33.bar(1.5, np.mean(t2035_[:,2]), width=0.25, color= con_, edgecolor=cedge_, yerr=np.std(t2035_[:,2]))  #, hatch='/' )
ax33.set_ylim(blim_, uplim_)
ax33.set_xticklabels([])
ax33.set_xticks([])
ax33.set_yticklabels([])
ax33.set_xlabel('T2*-w \n(035)', fontsize=8, fontweight='bold')
ax33.grid(True)

## T2s025-w ####
ax34 = plt.subplot(5,7,34)
t2025_ = np.vstack((np.float64(R1_eval_[5,:]), np.float64(R2_eval_[5,:]), np.float64(R3_eval_[5,:]), np.float64(R4_eval_[5,:])))
# print(t2025_.shape,'\n', t2025_)
ax34.bar(1, np.mean(t2025_[:,0]), width=0.25, color=coff_ , edgecolor=cedge_, yerr=np.std(t2025_[:,0]))
ax34.bar(1.25, np.mean(t2025_[:,1]), width=0.25, color=csame_, edgecolor=cedge_, yerr=np.std(t2025_[:,1]))
ax34.bar(1.5, np.mean(t2025_[:,2]), width=0.25, color= con_, edgecolor=cedge_, yerr=np.std(t2025_[:,2]))  #, hatch='/' )
ax34.set_ylim(blim_, uplim_)
ax34.set_xticklabels([])
ax34.set_xticks([])
ax34.set_yticklabels([])
ax34.set_xlabel('T2*-w \n(025)', fontsize=8, fontweight='bold')
ax34.grid(True)

## average all contrasts R4 ####
ax35 = plt.subplot(5,7,35)
allcon_= np.vstack((t1_,t2_,pd_,t205_, t2035_, t2025_))
# print(allcon_.shape,'\n', allcon_)
ax35.bar(1, np.mean(allcon_[:,0]), width=0.25, color=coff_ , edgecolor=cedge_ , yerr=np.std(allcon_[:,0]))
ax35.bar(1.25, np.mean(allcon_[:,1]), width=0.25, color=csame_, edgecolor=cedge_, yerr=np.std(allcon_[:,1]))
ax35.bar(1.5, np.mean(allcon_[:,2]), width=0.25, color= con_, edgecolor=cedge_, yerr=np.std(allcon_[:,2]))  #, hatch='/' )
ax35.set_ylim(blim_, uplim_)
ax35.set_xticklabels([])
ax35.set_xticks([])
ax35.set_yticklabels([])
ax35.set_xlabel('All \ncontrasts', fontsize=8, fontweight='bold')
ax35.set_facecolor('#61FF33')####('#E5ECF6')
ax35.grid(True)

print( np.mean(allcon_[:,0]), np.mean(allcon_[:,1]), np.mean(allcon_[:,2]))
# --------------------------------------------------- #
plt.yticks(fontsize=8, fontweight='bold')
# plt.tight_layout()
plt.show()
"""