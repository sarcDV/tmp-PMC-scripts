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

# print(R1_off, R2_off, R3_off, R4_off)
# print(R1_on, R2_on, R3_on, R4_on)


stack_off_ = np.double(np.concatenate((R1_off, R2_off, R3_off, R4_off), axis=1))
stack_on_ = np.double(np.concatenate((R1_on, R2_on, R3_on, R4_on), axis=1))

print(np.mean(stack_off_, axis= 1),np.std(stack_off_, axis= 1))
print(np.mean(stack_on_, axis= 1),np.std(stack_on_, axis= 1))

meanoff_, stdoff_ = np.mean(stack_off_, axis= 1),np.std(stack_off_, axis= 1)
meanon_, stdon_ = np.mean(stack_on_, axis= 1),np.std(stack_on_, axis= 1)

OFF_all = np.double(np.concatenate((R1_off, R2_off, R3_off, R4_off)).flatten())
ON_all = np.double(np.concatenate((R1_on, R2_on, R3_on, R4_on)).flatten())

print(np.mean(OFF_all), np.std(OFF_all))
print(np.mean(ON_all), np.std(ON_all))

meanoff_ = np.hstack((meanoff_, np.mean(OFF_all)))
meanon_ = np.hstack((meanon_, np.mean(ON_all)))

stdoff_ = np.hstack((stdoff_, np.std(OFF_all)))
stdon_ = np.hstack((stdon_, np.std(ON_all)))

print(meanoff_, meanon_)

print(stack_off_.shape)

pvalues = []
for ii in range(0, (stack_off_.shape[0])):
    stat, p = mannwhitneyu(stack_off_[ii, :], stack_on_[ii,:])
    print('Statistics=%.3f, p=%.5f' % (stat, p))
    pvalues.append(p)
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Same distribution (fail to reject H0)')
    else:
        print('Different distribution (reject H0)')

stat, p = mannwhitneyu(OFF_all, ON_all)
print('Statistics=%.3f, p=%.5f' % (stat, p))
pvalues.append(p)
# interpret
alpha = 0.05
if p > alpha:
    print('Same distribution (fail to reject H0)')
else:
    print('Different distribution (reject H0)')



#######################################################
## PLOT ###############################################
#######################################################
coff_ = "#3274A1"##"#333333"
con_ = "#E1812C"####"#b2b2b2"
csame_ ="#ffffff"#"#b2b2b2" ###"#01d1a1"
cedge_ ="#000000"

blim_ = -90.
uplim_ = 90.
width = 0.35
fig = plt.figure(figsize = (10,4))##1)

line_labels = [ "PMC OFF", 
			    "PMC ON"]
# Create the bar labels
bar_labels = ['T1-w', 'T2-w', 'PD-w', 'T2*-w (05)','T2*-w (035)','T2*-w (025)','ALL CONT.']
bar_finlabels = ['T1-w', 'T2-w', 'PD-w', 'T2*-w\n(05)','T2*-w\n(035)','T2*-w\n(025)','ALL\nCONT.']
assey_ = np.double([1,2,3,4,5,6,7])-width/2
## READER 1 ###
ax1 = plt.subplot(111)# plt.subplot(2,4,1)

l1 = ax1.bar(assey_, meanoff_, yerr=stdoff_, width=width,  color=coff_ , edgecolor=cedge_, capsize=3)
l2 = ax1.bar(assey_+width, meanon_,   yerr=stdon_,width=width, color= con_, edgecolor=cedge_, capsize=3)  #, hatch='/' )
# plt.yticks(assey_, bar_labels, fontsize=10, fontweight='bold')
plt.xticks(assey_+width/2, bar_labels, fontsize=10, fontweight='bold')
plt.yticks( fontsize=10, fontweight='bold')
# ax1.set_xticklabels([])
# ax1.set_xlim(blim_, uplim_)
plt.grid()
plt.title('Average scores for contrast', fontsize=10, fontweight='bold')
fig.legend([l1, l2],     # The line objects
           labels=line_labels,   # The labels for each line
           loc="upper right",# "upper center",   # Position of legend
           borderaxespad=0.1,    # Small spacing around legend box
           #title=""# "Legend Title"  # Title for the legend
           fontsize=8)

########################################################################################
########################################################################################
don_, doff_  = 15, +4
sh_ = 0.15

props = dict(boxstyle='round', facecolor='wheat', alpha=1)
for ii in range(0, len(pvalues)):
    ax1.text( ii+0.5+width, 4,str(convert_pval(pvalues[ii])), fontweight='bold', fontsize=7, bbox=props)
########################################################################################
########################################################################################
plt.show()