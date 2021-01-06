# -*- coding: utf-8 -*-
import glob, os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def ccorrel(p1, p2):
    a = p1
    b = p2
    a = (a - np.mean(a)) / (np.std(a) * len(a))
    b = (b - np.mean(b)) / (np.std(b))
    c = np.correlate(a, b)
    
    return c

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
			same_=same_+1
		elif in1_[ii] > in2_[ii]:
			onebettertwo=onebettertwo+1
		else:
			twobetterone=twobetterone+1

	onebettertwo=(onebettertwo/len(in1_))*100
	same_=(same_/len(in1_))*100
	twobetterone= 100-(onebettertwo+same_)##(twobetterone/len(in1_))*100

	return truncate(onebettertwo,2), truncate(same_,2), truncate(twobetterone,2)


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


R1_off, R2_off, R3_off, R4_off = np.uint8(R1_off), np.uint8(R2_off), np.uint8(R3_off), np.uint8(R4_off)
R1_on, R2_on, R3_on, R4_on = np.uint8(R1_on), np.uint8(R2_on), np.uint8(R3_on), np.uint8(R4_on)

"""
print(ccorrel(np.uint8(R1_off[0,:]), np.uint8(R1_off[0,:])))#, 
print(ccorrel(np.uint8(R1_off[0,:]), np.uint8(R2_off[0,:])))#, 
print(ccorrel(np.uint8(R1_off[0,:]), np.uint8(R3_off[0,:])))#, 
print(ccorrel(np.uint8(R1_off[0,:]), np.uint8(R4_off[0,:])))#, 
#  print(np.uint8(R1_off[0,:]))
"""

## reshape and concatenate off on:

R1_ = np.concatenate((np.reshape(R1_off, (R1_off.shape[0]*R1_off.shape[1])),
					  np.reshape(R1_on, (R1_on.shape[0]*R1_on.shape[1]))))

R2_ = np.concatenate((np.reshape(R2_off, (R2_off.shape[0]*R2_off.shape[1])),
					  np.reshape(R2_on, (R2_on.shape[0]*R2_on.shape[1]))))

R3_ = np.concatenate((np.reshape(R3_off, (R3_off.shape[0]*R3_off.shape[1])),
					  np.reshape(R3_on, (R3_on.shape[0]*R3_on.shape[1]))))

R4_ = np.concatenate((np.reshape(R4_off, (R4_off.shape[0]*R4_off.shape[1])),
					  np.reshape(R4_on, (R4_on.shape[0]*R4_on.shape[1]))))


R1toall_= np.array([ccorrel(R1_, R1_),ccorrel(R1_, R2_),ccorrel(R1_, R3_),ccorrel(R1_, R4_)]).squeeze()
R2toall_= np.array([ccorrel(R2_, R1_),ccorrel(R2_, R2_),ccorrel(R2_, R3_),ccorrel(R2_, R4_)]).squeeze()
R3toall_= np.array([ccorrel(R3_, R1_),ccorrel(R3_, R2_),ccorrel(R3_, R3_),ccorrel(R3_, R4_)]).squeeze()
R4toall_= np.array([ccorrel(R4_, R1_),ccorrel(R4_, R2_),ccorrel(R4_, R3_),ccorrel(R4_, R4_)]).squeeze()

all_ = np.concatenate((R1toall_, R2toall_, R3toall_, R4toall_))
## remove ones:
arr = all_[all_ <=0.98]
print(np.mean(all_))
print(np.mean(arr), np.std(arr), arr.max(), arr.min())
### 0.75 0.02 0.79 0.72

x= [1,2,3,4]
labels=['R1','R2','R3','R4']

fig = plt.figure(figsize = (10,4))

ax1 = plt.subplot(141)
ax1.bar([1,2,3,4],100*R1toall_)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontweight='bold')
plt.yticks(fontweight='bold')
ax1.set_title('Correlation of R1,\nwith [R2,R3,R4]', fontsize=10, fontweight='bold')
ax1.text(-0.5,105.5,'(%)', fontweight='bold')
# ax1.grid(True)

ax2 = plt.subplot(142)
ax2.bar([1,2,3,4],100*R2toall_)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontweight='bold')
# plt.yticks(fontweight='bold')
ax2.set_title('Correlation of R2,\nwith [R1,R3,R4]', fontsize=10, fontweight='bold')
ax2.set_yticklabels([])

ax3 = plt.subplot(143)
ax3.bar([1,2,3,4],100*R3toall_)
ax3.set_xticks(x)
ax3.set_xticklabels(labels, fontweight='bold')
# plt.yticks(fontweight='bold')
ax3.set_title('Correlation of R3,\nwith [R1,R2,R4]', fontsize=10, fontweight='bold')
ax3.set_yticklabels([])

ax4 = plt.subplot(144)
ax4.bar([1,2,3,4],100*R4toall_)
ax4.set_xticks(x)
ax4.set_xticklabels(labels, fontweight='bold')
# plt.yticks(fontweight='bold')
ax4.set_title('Correlation of R4,\nwith [R1,R2,R3]', fontsize=10, fontweight='bold')
ax4.set_yticklabels([])

plt.show()