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

R1_off = np.double(R1_off) # (np.double(woff_)*np.double(R1_off))+np.double(R1_off)
R1_on = np.double(R1_on) # (np.double(won_)*np.double(R1_on))+np.double(R1_on)

R2_off = np.double(R2_off) # (np.double(woff_)*np.double(R2_off))+np.double(R2_off)
R2_on = np.double(R2_on) # (np.double(won_)*np.double(R2_on))+np.double(R2_on)

R3_off = np.double(R3_off) # (np.double(woff_)*np.double(R3_off))+np.double(R3_off)
R3_on = np.double(R3_on) # (np.double(won_)*np.double(R3_on))+np.double(R3_on)

R4_off = np.double(R4_off) # (np.double(woff_)*np.double(R4_off))+np.double(R4_off)
R4_on = np.double(R4_on) # (np.double(won_)*np.double(R4_on))+np.double(R4_on)

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
don_, doff_  = 15, +4
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