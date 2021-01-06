# -*- coding: utf-8 -*-
import glob, os
import math
import numpy as np
import matplotlib.pyplot as plt

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

R1_eval_, R2_eval_= np.double(np.asarray(R1_eval_)), np.double(np.asarray(R2_eval_)),
R3_eval_,R4_eval_ = np.double(np.asarray(R3_eval_)), np.double(np.asarray(R4_eval_))


# --------------------------------------------------------------- 
group_names=['A','B','C']
# ['PMC OFF better than PMC ON', 'PMC OFF same as PMC ON', 'PMC ON better than PMC OFF']
# Create colors
a, b, c=[plt.cm.Reds, plt.cm.Blues,  plt.cm.Greens]
# First Ring (outside)
fig = plt.figure(figsize = (14,10))##1)
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
ax1 = plt.subplot(571)
mypie, _ = ax1.pie(R1_eval_[0,:], radius=1., colors=[a(0.85), b(0.85), c(0.85)] )
#ax1.set_ylabel('READER 1', rotation=90, fontsize=8, fontweight='bold')
# ----
ax1 = plt.subplot(572)
mypie, _ = ax1.pie(R1_eval_[1,:], radius=1.,  colors=[a(0.8), b(0.8), c(0.8)] )
# ----
ax1 = plt.subplot(573)
mypie, _ = ax1.pie(R1_eval_[2,:], radius=1.,  colors=[a(0.75), b(0.75), c(0.75)] )
# ----
ax1 = plt.subplot(574)
mypie, _ = ax1.pie(R1_eval_[3,:], radius=1.,  colors=[a(0.7), b(0.7), c(0.7)] )
# ----
ax1 = plt.subplot(575)
mypie, _ = ax1.pie(R1_eval_[4,:], radius=1.,  colors=[a(0.65), b(0.65), c(0.65)] )
# ----
ax1 = plt.subplot(576)
mypie, _ = ax1.pie(R1_eval_[5,:], radius=1.,  colors=[a(0.6), b(0.6), c(0.6)] )
# ----
ax1 = plt.subplot(577)
mypie, _ = ax1.pie(np.mean(np.double(R1_eval_),0), radius=1.25,  colors=[a(0.9), b(0.9), c(0.9)] )
plt.setp( mypie, width=0.5, edgecolor='white')
subgroup_size = np.concatenate((R1_eval_[:,0], R1_eval_[:,1], R1_eval_[:,2]), axis=0)# np.reshape(np.double(R1_eval_)/6., 6*3)
mypie2, _ = ax1.pie(subgroup_size, radius=1.25-0.5, 
								   colors=[a(0.85), a(0.8), a(0.75), a(0.7), a(0.65), a(0.6),
										   b(0.85), b(0.8), b(0.75), b(0.7), b(0.65), b(0.6),
										   c(0.85), c(0.8), c(0.75), c(0.7), c(0.65), c(0.6),])
plt.setp( mypie2, width=0.5, edgecolor='white')
#plt.margins(0,0)

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
ax1 = plt.subplot(578)
mypie, _ = ax1.pie(R2_eval_[0,:], radius=1., colors=[a(0.85), b(0.85), c(0.85)] )
#ax1.set_ylabel('READER 2', rotation=90, fontsize=8, fontweight='bold')
# ----
ax1 = plt.subplot(579)
mypie, _ = ax1.pie(R2_eval_[1,:], radius=1.,  colors=[a(0.8), b(0.8), c(0.8)])
# ----
ax1 = plt.subplot(5,7,10)
mypie, _ = ax1.pie(R2_eval_[2,:], radius=1.,  colors=[a(0.75), b(0.75), c(0.75)] )
# ----
ax1 = plt.subplot(5,7,11)
mypie, _ = ax1.pie(R2_eval_[3,:], radius=1.,  colors=[a(0.7), b(0.7), c(0.7)])
# ----
ax1 = plt.subplot(5,7,12)
mypie, _ = ax1.pie(R2_eval_[4,:], radius=1.,  colors=[a(0.65), b(0.65), c(0.65)] )
# ----
ax1 = plt.subplot(5,7,13)
mypie, _ = ax1.pie(R2_eval_[5,:], radius=1.,  colors=[a(0.6), b(0.6), c(0.6)] )
# ----
ax1 = plt.subplot(5,7,14)
mypie, _ = ax1.pie(np.mean(np.double(R2_eval_),0), radius=1.25,  colors=[a(0.9), b(0.9), c(0.9)] )
plt.setp( mypie, width=0.5, edgecolor='white')
subgroup_size = np.concatenate((R2_eval_[:,0], R2_eval_[:,1], R2_eval_[:,2]), axis=0)# np.reshape(np.double(R1_eval_)/6., 6*3)
mypie2, _ = ax1.pie(subgroup_size, radius=1.25-0.5, 
								  colors=[a(0.85), a(0.8), a(0.75), a(0.7), a(0.65), a(0.6),
										   b(0.85), b(0.8), b(0.75), b(0.7), b(0.65), b(0.6),
										   c(0.85), c(0.8), c(0.75), c(0.7), c(0.65), c(0.6),])
plt.setp( mypie2, width=0.5, edgecolor='white')
#plt.margins(0,0)

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
ax1 = plt.subplot(5,7,15)
mypie, _ = ax1.pie(R3_eval_[0,:], radius=1., colors=[a(0.85), b(0.85), c(0.85)] )
#ax1.set_ylabel('READER 3', rotation=90, fontsize=8, fontweight='bold')
# ----
ax1 = plt.subplot(5,7,16)
mypie, _ = ax1.pie(R3_eval_[1,:], radius=1.,  colors=[a(0.8), b(0.8), c(0.8)] )
# ----
ax1 = plt.subplot(5,7,17)
mypie, _ = ax1.pie(R3_eval_[2,:], radius=1.,  colors=[a(0.75), b(0.75), c(0.75)] )
# ----
ax1 = plt.subplot(5,7,18)
mypie, _ = ax1.pie(R3_eval_[3,:], radius=1.,  colors=[a(0.7), b(0.7), c(0.7)] )
# ----
ax1 = plt.subplot(5,7,19)
mypie, _ = ax1.pie(R3_eval_[4,:], radius=1.,  colors=[a(0.65), b(0.65), c(0.65)] )
# ----
ax1 = plt.subplot(5,7,20)
mypie, _ = ax1.pie(R3_eval_[5,:], radius=1.,  colors=[a(0.6), b(0.6), c(0.6)] )
# ----
ax1 = plt.subplot(5,7,21)
mypie, _ = ax1.pie(np.mean(np.double(R3_eval_),0), radius=1.25,  colors=[a(0.9), b(0.9), c(0.9)] )
plt.setp( mypie, width=0.5, edgecolor='white')
subgroup_size = np.concatenate((R3_eval_[:,0], R3_eval_[:,1], R3_eval_[:,2]), axis=0)# np.reshape(np.double(R1_eval_)/6., 6*3)
mypie2, _ = ax1.pie(subgroup_size, radius=1.25-0.5, 
								   colors=[a(0.85), a(0.8), a(0.75), a(0.7), a(0.65), a(0.6),
										   b(0.85), b(0.8), b(0.75), b(0.7), b(0.65), b(0.6),
										   c(0.85), c(0.8), c(0.75), c(0.7), c(0.65), c(0.6),])
plt.setp(mypie2, width=0.5, edgecolor='white')
#plt.margins(0,0)

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
ax1 = plt.subplot(5,7,22)
mypie, _ = ax1.pie(R4_eval_[0,:], radius=1., colors=[a(0.85), b(0.85), c(0.85)] )
#ax1.set_ylabel('READER 4', rotation=90, fontsize=8, fontweight='bold')
# ----
ax1 = plt.subplot(5,7,23)
mypie, _ = ax1.pie(R4_eval_[1,:], radius=1.,  colors=[a(0.8), b(0.8), c(0.8)] )
# ----
ax1 = plt.subplot(5,7,24)
mypie, _ = ax1.pie(R4_eval_[2,:], radius=1.,  colors=[a(0.75), b(0.75), c(0.75)] )
# ----
ax1 = plt.subplot(5,7,25)
mypie, _ = ax1.pie(R4_eval_[3,:], radius=1.,  colors=[a(0.7), b(0.7), c(0.7)] )
# ----
ax1 = plt.subplot(5,7,26)
mypie, _ = ax1.pie(R4_eval_[4,:], radius=1.,  colors=[a(0.65), b(0.65), c(0.65)] )
# ----
ax1 = plt.subplot(5,7,27)
mypie, _ = ax1.pie(R4_eval_[5,:], radius=1.,  colors=[a(0.6), b(0.6), c(0.6)] )
# ----
ax1 = plt.subplot(5,7,28)
mypie, _ = ax1.pie(np.mean(np.double(R4_eval_),0), radius=1.25,  colors=[a(0.9), b(0.9), c(0.9)] )
plt.setp( mypie, width=0.5, edgecolor='white')
subgroup_size = np.concatenate((R4_eval_[:,0], R4_eval_[:,1], R4_eval_[:,2]), axis=0)# np.reshape(np.double(R1_eval_)/6., 6*3)
mypie2, _ = ax1.pie(subgroup_size, radius=1.25-0.5, 
								   colors=[a(0.85), a(0.8), a(0.75), a(0.7), a(0.65), a(0.6),
										   b(0.85), b(0.8), b(0.75), b(0.7), b(0.65), b(0.6),
										   c(0.85), c(0.8), c(0.75), c(0.7), c(0.65), c(0.6),])
plt.setp( mypie2, width=0.5, edgecolor='white')
#plt.margins(0,0)


# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
ax1 = plt.subplot(5,7,29)
t1_ = np.vstack((R1_eval_[0,:],R2_eval_[0,:], R3_eval_[0,:], R4_eval_[0,:]))
mypie, _ = ax1.pie(np.mean(t1_, axis=0), radius=1.25, colors=[a(0.85), b(0.85), c(0.85)] )
#ax1.set_ylabel('ALL READERS', rotation=90, fontsize=8, fontweight='bold')
#ax1.set_xlabel('T1-w', fontsize=8, fontweight='bold')
# ----
ax1 = plt.subplot(5,7,30)
t2_ = np.vstack((R1_eval_[1,:],R2_eval_[1,:], R3_eval_[1,:], R4_eval_[1,:]))
mypie, _ = ax1.pie(np.mean(t2_, axis=0), radius=1.25, colors=[a(0.8), b(0.8), c(0.8)] )
#ax1.set_xlabel('T2-w', fontsize=8, fontweight='bold')
# ----
ax1 = plt.subplot(5,7,31)
pd_ = np.vstack((R1_eval_[2,:],R2_eval_[2,:], R3_eval_[2,:], R4_eval_[2,:]))
mypie, _ = ax1.pie(np.mean(pd_, axis=0), radius=1.25, colors=[a(0.75), b(0.75), c(0.75)] )
#ax1.set_xlabel('PD-w', fontsize=8, fontweight='bold')
# ----
ax1 = plt.subplot(5,7,32)
t205_ = np.vstack((R1_eval_[3,:],R2_eval_[3,:], R3_eval_[3,:], R4_eval_[3,:]))
mypie, _ = ax1.pie(np.mean(t205_, axis=0), radius=1.25, colors=[a(0.7), b(0.7), c(0.7)] )
#ax1.set_xlabel('T2*-w \n(05)', fontsize=8, fontweight='bold')
# ----
ax1 = plt.subplot(5,7,33)
t2035_ = np.vstack((R1_eval_[4,:],R2_eval_[4,:], R3_eval_[4,:], R4_eval_[4,:]))
mypie, _ = ax1.pie(np.mean(t2035_, axis=0), radius=1.25, colors=[a(0.65), b(0.65), c(0.65)] )
#ax1.set_xlabel('T2*-w \n(035)', fontsize=8, fontweight='bold')
# ----
ax1 = plt.subplot(5,7,34)
t2025_ = np.vstack((R1_eval_[5,:],R2_eval_[5,:], R3_eval_[5,:], R4_eval_[5,:]))
mypie, _ = ax1.pie(np.mean(t2025_, axis=0), radius=1.25, colors=[a(0.6), b(0.6), c(0.6)] )
#ax1.set_xlabel('T2*-w \n(025)', fontsize=8, fontweight='bold')
# ----
ax1 = plt.subplot(5,7,35)
allcon_= np.vstack((t1_,t2_,pd_,t205_, t2035_, t2025_))
mypie, _ = ax1.pie(np.mean(allcon_,0), radius=1.5,  colors=[a(0.9), b(0.9), c(0.9)] )
print(np.mean(allcon_,0))
#ax1.set_xlabel('All contrasts', fontsize=8, fontweight='bold')
plt.setp( mypie, width=0.5, edgecolor='white')
subgroup_ =np.vstack((np.mean(t1_,0), np.mean(t2_,0), np.mean(pd_,0),
					  np.mean(t205_,0), np.mean(t2035_,0), np.mean(t2025_,0)))#, axis=0))

subgroup_size = np.concatenate((subgroup_[:,0], subgroup_[:,1], subgroup_[:,2]), axis=0)
mypie2, _ = ax1.pie(subgroup_size, radius=1.5-0.5, 
								   colors=[a(0.85), a(0.8), a(0.75), a(0.7), a(0.65), a(0.6),
										   b(0.85), b(0.8), b(0.75), b(0.7), b(0.65), b(0.6),
										   c(0.85), c(0.8), c(0.75), c(0.7), c(0.65), c(0.6),])
plt.setp( mypie2, width=0.5, edgecolor='white')

#plt.tight_layout()
plt.show()
# ------------------------------------------------------------
# --------------------------


"""
# Libraries
# import matplotlib.pyplot as plt
 
# Make data: I have 3 groups and 7 subgroups
group_names=['groupA', 'groupB', 'groupC']
group_size=[12,11,30]
subgroup_names=['A.1', 'A.2', 'A.3', 'B.1', 'B.2', 'C.1', 'C.2', 'C.3', 'C.4', 'C.5']
subgroup_size=[4,3,5,6,5,10,5,5,4,6]
 
# Create colors
a, b, c=[plt.cm.Blues, plt.cm.Reds, plt.cm.Greens]
 
# First Ring (outside)
fig, ax = plt.subplots()
ax.axis('equal')
mypie, _ = ax.pie(group_size, radius=1.3, labels=group_names, colors=[a(0.6), b(0.6), c(0.6)] )
plt.setp( mypie, width=0.3, edgecolor='white')
 
# Second Ring (Inside)
mypie2, _ = ax.pie(subgroup_size, radius=1.3-0.3, labels=subgroup_names, labeldistance=0.7, colors=[a(0.5), a(0.4), a(0.3), b(0.5), b(0.4), c(0.6), c(0.5), c(0.4), c(0.3), c(0.2)])
plt.setp( mypie2, width=0.4, edgecolor='white')
plt.margins(0,0)
 
# show it
# plt.show()
# Change color
squarify.plot(sizes=[13,22,35,5], label=["group A", "group B", "group C", "group D"], color=["red","green","blue", "grey"], alpha=.4 )
plt.axis('off')
plt.show()
# If you have a data frame?
import pandas as pd
df = pd.DataFrame({'nb_people':[8,3,4,2], 'group':["group A", "group B", "group C", "group D"] })
squarify.plot(sizes=df['nb_people'], label=df['group'], alpha=.8 )
plt.axis('off')
plt.show()

import squarify    # pip install squarify (algorithm for treemap)
import matplotlib.gridspec as gridspec
# If you have 2 lists
fig = plt.figure(figsize = (9,2))##1)
# -----
ax1 = plt.subplot(131)
squarify.plot(sizes=[60,10,10,10,10,10,10], 
			   label=["PMC OFF \n better than \n PMC ON",
			   		  "T1-w", "T2-w",
			   		  "PD-w", "T2*-w \n (05)",
			   		  "T2*-w \n (035) ", "T2*-w\n (025) "], 
			    color=["#951b1e","#ab2022","#b82325","#c62728","#d5312e","#e43d33","#ee4b3b"],
			    #[a(0.9),a(0.85), a(0.8), a(0.75), a(0.7), a(0.65), a(0.6)],
			   	alpha=.7 )
plt.axis('off')

# -----
ax1 = plt.subplot(132)
squarify.plot(sizes=[60,10,10,10,10,10,10], 
			   label=["PMC OFF \n same as \n PMC ON",
			   		  "T1-w", "T2-w",
			   		  "PD-w", "T2*-w \n (05)",
			   		  "T2*-w \n (035) ", "T2*-w\n (025) "],
			    color=[b(0.9),b(0.85), b(0.8), b(0.75), b(0.7), b(0.65), b(0.6)],
			   	alpha=.7 )
plt.axis('off')
# -----
ax1 = plt.subplot(133)
squarify.plot(sizes=[60,10,10,10,10,10,10], 
			   label=["PMC ON \n better than \n PMC OFF",
			   		  "T1-w", "T2-w",
			   		  "PD-w", "T2*-w \n (05)",
			   		  "T2*-w \n (035) ", "T2*-w\n (025) "],
			    color=[c(0.9),c(0.85), c(0.8), c(0.75), c(0.7), c(0.65), c(0.6)],
			   	alpha=.7 )
plt.axis('off')
plt.subplots_adjust(wspace=None, hspace=None)
plt.show()

"""