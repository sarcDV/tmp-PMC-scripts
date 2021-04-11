# -*- coding: utf-8 -*-
import glob, os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

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

def homebarplot(pmc_off,pmc_on, titolo):
	people = ("R1","R2","R3","R4","Ravg")
	num_people = len(people)
	xmin, xmax = 0, 10
	## time_spent = np.random.uniform(low=5, high=100, size=num_people)
	## proficiency = np.abs(time_spent / 12. + np.random.normal(size=num_people))
	xerr_off = [0. , 0., 0. ,0., np.std(pmc_off)]
	xerr_on =  [0. , 0., 0. ,0., np.std(pmc_on)]
	pmc_off = np.hstack((pmc_off, np.mean(pmc_off)))
	pmc_on = np.hstack((pmc_on, np.mean(pmc_on)))
	pos = np.arange(num_people) + .5 # bars centered on the y axis
	fig, (ax_left, ax_right, ax_diff) = plt.subplots(ncols=3)
	## ---------------------------
	ax_left.barh(pos, pmc_off, xerr = xerr_off,  align="center", facecolor="#3274A1")
	# ax_left.set_yticks([])
	ax_left.set_yticks(pos)
	ax_left.set_yticklabels(people, ha="center",  x=-0.08, fontsize=10, fontweight='bold')
	ax_left.set_xlabel("PMC OFF", fontsize=10, fontweight='bold')
	ax_left.invert_xaxis()
	ax_left.grid(True)
	ax_left.set_xlim([xmax, xmin])
	## ---------------------------
	ax_right.barh(pos, pmc_on, xerr = xerr_on, align="center", facecolor="#E1812C")
	ax_right.set_yticks([])
	# ax_right.set_yticks(pos)
	### x moves tick labels relative to left edge of axes in axes units
	## ax_right.set_yticklabels(people, ha="center",  x=-0.08)
	ax_right.set_xlabel("PMC ON", fontsize=10, fontweight='bold')
	ax_right.grid(True)
	ax_right.set_xlim([xmin,xmax])
	## ---------------------------
	## print(pmc_on - pmc_off)
	ax_diff.barh(pos, pmc_on-pmc_off, align="center", facecolor="green")
	ax_diff.set_yticks([])
	# # ax_right.set_yticks(pos)
	# ### x moves tick labels relative to left edge of axes in axes units
	# ax_diff.set_yticklabels(people, ha="center", x=-0.08)
	ax_diff.set_xlabel("PMC ON - PMC OFF", fontsize=10, fontweight='bold')
	ax_diff.grid(True)
	tmpmax =  (np.max(np.abs(pmc_on-pmc_off)))
	ax_diff.set_xlim([-tmpmax-0.75,tmpmax+0.75])
	## --------------------------
	## plt.suptitle("Learning Python")
	plt.suptitle(titolo, fontsize=12, fontweight='bold')
	## plt.show()
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

scans = ["T1-w", "T2-w", "PD", "T2*-w (05)", "T2*-w (035)", "T2*-w (025)"] 
# soggetti = ["au70", "dl43", "dn20", "dp20", "iy25", "kc73", "me21", "mf79", "nv85", "pa30",
# 			"qg37", "sc17", "um68", "ut70", "vk04", "vq83", "ws31", "ww25", "xi27", "xx54", 
# 			"yv98" ]


# print(R1_off.shape, R2_off.shape, R3_off.shape, R4_off.shape)
# print(R1_on.shape, R2_on.shape, R3_on.shape, R4_on.shape)
# print(R1_off.dtype)

# for sequence in range(0,6):	
# 	for subject in range(0,21):
# 		pmc_off = np.float64([ R1_off[sequence, subject],  R1_off[sequence, subject], 
# 					R3_off[sequence, subject],  R4_off[sequence, subject] ])
# 		pmc_on = np.float64([ R1_on[sequence, subject],  R1_on[sequence, subject], 
# 					R3_on[sequence, subject],  R4_on[sequence, subject] ])
# 		# print(pmc_on-pmc_off)
# 		print(scans[sequence]+" "+soggetti[subject])
# 		titolo = scans[sequence]+" "+soggetti[subject]
# 		homebarplot(pmc_off, pmc_on, titolo)
# 		figure = plt.gcf()	
# 		plt.savefig(str(soggetti[subject])+"_"+str(contrasts[sequence])+".png")
# 		plt.close()

# gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1]) 
# ax0 = plt.subplot(gs[0])

soggetti = ["au70", "dl43", "dn20", "dp20", "iy25", "kc73", "me21", "mf79", "nv85", "pa30",
			"qg37", "sc17", "um68", "ut70", "vk04", "vq83", "ws31", "ww25", "xi27", "xx54", 
			"yv98" ]

# soggetti = ["ID-01", "ID-02", "ID-03", "ID-04", "ID-05", "ID-06", "ID-07", "ID-08", "ID-09", "ID-10",
# 			"ID-11", "ID-12", "ID-13", "ID-14", "ID-15", "ID-16", "ID-17", "ID-18", "ID-19", "ID-20", 
# 			"ID-21" ]

# readers_str = ["$R1_{avg}$", "$R2_{avg}$","$R3_{avg}$","$R4_{avg}$"]
readers_str = ["R1", "R2", "R3", "R4"]

fig = plt.figure(figsize=(7,8),constrained_layout=True)
widths = [1, 1, 1]
heights = [0.5, 1.5, 4]
spec = fig.add_gridspec(ncols=3, nrows=3, width_ratios=widths,height_ratios=heights)
zz = 5 ## choose contrast
xmin, xmax = 0, 10
## ---------------------------------------------



## ---------------------------------------------
# ax1 = plt.subplot(321)
ax1 = fig.add_subplot(spec[0, 0])
data_off = np.hstack((np.float64(R1_off[zz,:]),np.float64(R2_off[zz,:]), np.float64(R3_off[zz,:]),np.float64(R4_off[zz,:])))      
ax1.barh(1.0, np.mean(data_off), xerr=np.std(data_off),facecolor="#3274A1")
pos_avg = np.float64([0.0, 1.0, 2.0])
avgn_ = ["","", ""]
ax1.set_yticks(pos_avg)
ax1.set_yticklabels(avgn_, x = -0.06, ha="center", fontsize=10, fontweight='bold')
ax1.set_xticklabels([])
ax1.set_xlim([xmin, xmax])
ax1.invert_xaxis()
ax1.set_title('PMC OFF', fontsize=10, fontweight='bold')
ax1.set_ylim([0.5,1.5])
props = dict(boxstyle='round', facecolor='white', alpha=1)

ax1.text(4.75,1,r'$\mu$='+str(truncate(np.mean(data_off),2))+r'$\pm$'+str(truncate(np.std(data_off),2)), fontweight='bold', fontsize=8, bbox=props)


## ---------------------------------------------
# ax2 = plt.subplot(322)
ax2 = fig.add_subplot(spec[0, 1])
data_on = np.hstack((np.float64(R1_on[zz,:]),np.float64(R2_on[zz,:]), np.float64(R3_on[zz,:]),np.float64(R4_on[zz,:])))      
ax2.barh(1.0, np.mean(data_on), xerr=np.std(data_on) , facecolor="#E1812C")
## ax2.set_yticklabels([])
pos_avg = np.float64([0.0, 1.0, 2.0])
avg_ = ["","MEAN", ""]
ax2.set_yticks(pos_avg)
ax2.set_yticklabels(avg_, x = -0.1, ha="center", fontsize=8, fontweight='bold')
ax2.set_title('PMC ON', fontsize=10, fontweight='bold')
ax2.set_xticklabels([])
ax2.set_xlim([xmin, xmax])
ax2.set_ylim([0.5,1.5])

ax2.text(0.5,1,r'$\mu$='+str(truncate(np.mean(data_on),2))+r'$\pm$'+str(truncate(np.std(data_on),2)), fontweight='bold', fontsize=8, bbox=props)

## ---------------------------------------------
# ax3 = plt.subplot(323)
ax3 = fig.add_subplot(spec[1, 0])
pos_read = np.float64([1.,2.,3.,4.])
avg_r_off = np.float64([np.mean(np.float64(R1_off[zz,:])), np.mean(np.float64(R2_off[zz,:])),
			 np.mean(np.float64(R3_off[zz,:])), np.mean(np.float64(R4_off[zz,:]))])
std_r_off = ([np.std(np.float64(R1_off[zz,:])), np.std(np.float64(R2_off[zz,:])),
			 np.std(np.float64(R3_off[zz,:])), np.std(np.float64(R4_off[zz,:]))])

ax3.barh(pos_read, avg_r_off, xerr = std_r_off, facecolor="#3274A1")
ax3.set_yticklabels([])
ax3.set_xticklabels([])
ax3.set_xlim([xmin, xmax])
ax3.invert_xaxis()

ax3.text(3.75,1,r'$\mu$='+str(truncate(avg_r_off[0],2))+r'$\pm$'+str(truncate(std_r_off[0],2)), fontweight='bold', fontsize=6, bbox=props)
ax3.text(3.75,2,r'$\mu$='+str(truncate(avg_r_off[1],2))+r'$\pm$'+str(truncate(std_r_off[1],2)), fontweight='bold', fontsize=6, bbox=props)
ax3.text(3.75,3,r'$\mu$='+str(truncate(avg_r_off[2],2))+r'$\pm$'+str(truncate(std_r_off[2],2)), fontweight='bold', fontsize=6, bbox=props)
ax3.text(3.75,4,r'$\mu$='+str(truncate(avg_r_off[3],2))+r'$\pm$'+str(truncate(std_r_off[3],2)), fontweight='bold', fontsize=6, bbox=props)

## ---------------------------------------------
# ax4 = plt.subplot(324)
ax4 = fig.add_subplot(spec[1, 1])
avg_r_on = np.float64([np.mean(np.float64(R1_on[zz,:])), np.mean(np.float64(R2_on[zz,:])),
			 np.mean(np.float64(R3_on[zz,:])), np.mean(np.float64(R4_on[zz,:]))])
std_r_on = np.float64([np.std(np.float64(R1_on[zz,:])), np.std(np.float64(R2_on[zz,:])),
			 np.std(np.float64(R3_on[zz,:])), np.std(np.float64(R4_on[zz,:]))])

ax4.barh(pos_read, avg_r_on, xerr = std_r_on, facecolor="#E1812C")
ax4.set_yticklabels([])
ax4.set_xticklabels([])
ax4.set_xlim([xmin, xmax])
ax4.set_yticks(pos_read)
ax4.set_yticklabels( readers_str, x = -0.1, ha="center", fontsize=8, fontweight='bold')

ax4.text(0.5,1,r'$\mu$='+str(truncate(avg_r_on[0],2))+r'$\pm$'+str(truncate(std_r_on[0],2)), fontweight='bold', fontsize=6, bbox=props)
ax4.text(0.5,2,r'$\mu$='+str(truncate(avg_r_on[1],2))+r'$\pm$'+str(truncate(std_r_on[1],2)), fontweight='bold', fontsize=6, bbox=props)
ax4.text(0.5,3,r'$\mu$='+str(truncate(avg_r_on[2],2))+r'$\pm$'+str(truncate(std_r_on[2],2)), fontweight='bold', fontsize=6, bbox=props)
ax4.text(0.5,4,r'$\mu$='+str(truncate(avg_r_on[3],2))+r'$\pm$'+str(truncate(std_r_on[3],2)), fontweight='bold', fontsize=6, bbox=props)


## ---------------------------------------------
## ax5 = plt.subplot(325)
ax5 = fig.add_subplot(spec[2, 0])
larghezza = 0.2
pos = np.arange(len(soggetti)) + 1 

ax5.barh(pos-larghezza, np.float64(R1_off[zz,:]), height=larghezza, align="center", facecolor="#3274A1")#, hatch="|")
ax5.barh(pos , np.float64(R2_off[zz,:]), height=larghezza, align="center", facecolor="#3274A1") #, hatch="/")
ax5.barh(pos+larghezza , np.float64(R3_off[zz,:]), height=larghezza , align="center", facecolor="#3274A1")#, hatch="+")
ax5.barh(pos+2*larghezza , np.float64(R4_off[zz,:]), height=larghezza , align="center", facecolor="#3274A1") #, hatch="-")
ax5.set_xlim([xmin, xmax])
ax5.invert_xaxis()
ax5.set_yticklabels([])

## ---------------------------------------------
## ax6 = plt.subplot(326)
ax6 = fig.add_subplot(spec[2, 1])
ax6.barh(pos-larghezza, np.float64(R1_on[zz,:]), height=larghezza, align="center", facecolor="#E1812C")
ax6.barh(pos , np.float64(R2_on[zz,:]), height=larghezza, align="center", facecolor="#E1812C" )
ax6.barh(pos+larghezza , np.float64(R3_on[zz,:]), height=larghezza , align="center", facecolor="#E1812C")
ax6.barh(pos+2*larghezza , np.float64(R4_on[zz,:]), height=larghezza , align="center", facecolor="#E1812C")

ax6.set_xlim([xmin, xmax])
ax6.set_yticks(pos+0.2)
ax6.set_yticklabels( soggetti, ha="center",  x=-0.1, fontsize=8, fontweight='bold')

## ---------------------------------------------
# ax2 = plt.subplot(322)
axd1 = fig.add_subplot(spec[0, 2])
axd1.barh(1.0, np.mean(data_on)- np.mean(data_off),  facecolor="green")
axd1.set_title(r'$\Delta =$'+'(PMC ON - PMC OFF)', fontsize=10, fontweight='bold')
axd1.set_yticklabels([])
axd1.set_xticklabels([])
axd1.set_ylim([0.5,1.5])
axd1.set_xlim([-5.1, 5.1])
axd1.grid(True)

stack_off_ = np.double(np.hstack((R1_off[zz,:].squeeze(), R2_off[zz,:].squeeze(), R3_off[zz,:].squeeze(), R4_off[zz,:].squeeze())))
stack_on_ = np.double(np.hstack((R1_on[zz,:].squeeze(), R2_on[zz,:].squeeze(), R3_on[zz,:].squeeze(), R4_on[zz,:].squeeze())))

stat, p = mannwhitneyu(stack_off_, stack_on_)
print('Statistics=%.3f, p=%.5f' % (stat, p))


delta_ = np.mean(data_on) - np.mean(data_off)
axd1.text(-4.75,0.75,r'$\Delta$='+str(truncate(np.mean(delta_),2))+"\n"+str(convert_pval(p)),
		 fontweight='bold', fontsize=8, bbox=props)
## ---------------------------------------------
# ax2 = plt.subplot(322)
axd2 = fig.add_subplot(spec[1, 2])
axd2.barh(pos_read, (avg_r_on-avg_r_off),  facecolor="green")
axd2.set_xticklabels([])
axd2.set_yticklabels([])
# axd2.set_ylim([0.5,1.5])
axd2.set_xlim([-5.1, 5.1])
axd2.grid(True)

deltar_ = avg_r_on -avg_r_off
stat1, p1 = mannwhitneyu(R1_off[zz,:], R1_on[zz,:])
stat2, p2 = mannwhitneyu(R2_off[zz,:], R2_on[zz,:])
stat3, p3 = mannwhitneyu(R3_off[zz,:], R3_on[zz,:])
stat4, p4 = mannwhitneyu(R4_off[zz,:], R4_on[zz,:])
axd2.text(-4.5,0.75,r'$\Delta$='+str(truncate(deltar_[0],2))+"\n"+str(convert_pval(p1)), fontweight='bold', fontsize=6, bbox=props)
axd2.text(-4.5,1.75,r'$\Delta$='+str(truncate(deltar_[1],2))+"\n"+str(convert_pval(p2)), fontweight='bold', fontsize=6, bbox=props)
axd2.text(-4.5,2.75,r'$\Delta$='+str(truncate(deltar_[2],2))+"\n"+str(convert_pval(p3)), fontweight='bold', fontsize=6, bbox=props)
axd2.text(-4.5,3.75,r'$\Delta$='+str(truncate(deltar_[3],2))+"\n"+str(convert_pval(p4)), fontweight='bold', fontsize=6, bbox=props)


## ---------------------------------------------
# ax2 = plt.subplot(322)
axd3 = fig.add_subplot(spec[2, 2])
axd3.barh(pos-larghezza, np.float64(R1_on[zz,:])-np.float64(R1_off[zz,:]), height=larghezza, align="center", facecolor="green")
axd3.barh(pos , np.float64(R2_on[zz,:])-np.float64(R1_off[zz,:]), height=larghezza, align="center", facecolor="green" )
axd3.barh(pos+larghezza , np.float64(R3_on[zz,:])-np.float64(R1_off[zz,:]), height=larghezza , align="center", facecolor="green")
axd3.barh(pos+2*larghezza , np.float64(R4_on[zz,:])-np.float64(R1_off[zz,:]), height=larghezza , align="center", facecolor="green")
# tmp1_ = np.float64(R1_on[zz,:])-np.float64(R1_off[zz,:])
# tmp2_ = np.float64(R2_on[zz,:])-np.float64(R1_off[zz,:])
# tmp3_ = np.float64(R3_on[zz,:])-np.float64(R1_off[zz,:])
# tmp4_ = np.float64(R4_on[zz,:])-np.float64(R4_off[zz,:])

# tmpmax_ = np.max([np.abs(tmp1_),np.abs(tmp2_), np.abs(tmp3_), np.abs(tmp4_)])
# print(tmpmax_)
axd3.set_yticklabels([])
# axd2.set_ylim([0.5,1.5])
axd3.set_xlim([-5.1, 5.1])
axd3.grid(True)
## --------------------------------
# axd1.set_xlim([-tmpmax_-0.25,tmpmax_+0.75])
# axd2.set_xlim([-tmpmax_-0.75,tmpmax_+0.75])
# axd3.set_xlim([-tmpmax_-0.75,tmpmax_+0.75])
## ----------------------------------------------
plt.suptitle(str(scans[zz]), fontsize=16, fontweight='bold')
# plt.tight_layout()
plt.show()


## ----------------------------------------------
## ----------------------------------------------
## ----------------------------------------------