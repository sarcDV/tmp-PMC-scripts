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

def convert_pval(inval):
    if inval < 0.01:
        outval = "p < 0.01" 
    elif inval >= 0.01:
        outval = "p = "+str(truncate(inval,2))
    return outval


def data_loader(input_off, input_on):
	time_x = np.squeeze(np.array(input_off['logdata_OFF_realtime']))
	y_off_tr_x =np.squeeze(np.array(input_off['logdata_OFF_realtimex']))
	y_off_tr_y =np.squeeze(np.array(input_off['logdata_OFF_realtimey']))
	y_off_tr_z =np.squeeze(np.array(input_off['logdata_OFF_realtimez']))
	
	y_on_tr_x =np.squeeze(np.array(input_on['logdata_ON_realtimex']))
	y_on_tr_y =np.squeeze(np.array(input_on['logdata_ON_realtimey']))
	y_on_tr_z =np.squeeze(np.array(input_on['logdata_ON_realtimez']))

	y_off_rot_x =np.squeeze(np.array(input_off['logdata_OFF_realtimeRx']))
	y_off_rot_y =np.squeeze(np.array(input_off['logdata_OFF_realtimeRy']))
	y_off_rot_z =np.squeeze(np.array(input_off['logdata_OFF_realtimeRz']))
	
	y_on_rot_x =np.squeeze(np.array(input_on['logdata_ON_realtimeRx']))
	y_on_rot_y =np.squeeze(np.array(input_on['logdata_ON_realtimeRy']))
	y_on_rot_z =np.squeeze(np.array(input_on['logdata_ON_realtimeRz']))
	
	check_size_off_ = int(np.asarray(y_off_tr_x.shape))
	check_size_on_ = int(np.asarray(y_on_tr_x.shape))
	
	if (check_size_off_ > check_size_on_):
		y_off_tr_x = y_off_tr_x[check_size_off_ -check_size_on_:]
		y_off_tr_y = y_off_tr_y[check_size_off_ -check_size_on_:]
		y_off_tr_z = y_off_tr_z[check_size_off_ -check_size_on_:]
		y_off_rot_x = y_off_rot_x[check_size_off_ -check_size_on_:]
		y_off_rot_y = y_off_rot_y[check_size_off_ -check_size_on_:]
		y_off_rot_z = y_off_rot_z[check_size_off_ -check_size_on_:]
	elif (check_size_off_ < check_size_on_):
		y_on_tr_x = y_on_tr_x[check_size_on_ -check_size_off_:]
		y_on_tr_y = y_on_tr_y[check_size_on_ -check_size_off_:]
		y_on_tr_z = y_on_tr_z[check_size_on_ -check_size_off_:]
		y_on_rot_x = y_on_rot_x[check_size_on_ -check_size_off_:]
		y_on_rot_y = y_on_rot_y[check_size_on_ -check_size_off_:]
		y_on_rot_z = y_on_rot_z[check_size_on_ -check_size_off_:]
	
	return y_off_tr_x, y_off_tr_y, y_off_tr_z, y_off_rot_x, y_off_rot_y, y_off_rot_z, \
		   y_on_tr_x, y_on_tr_y, y_on_tr_z, y_on_rot_x, y_on_rot_y, y_on_rot_z, time_x


sessions = ['Off']#['Off', 'On']
contrasts = ['PD']#['PD', 'T1', 'T2', 'T2s05', 'T2s035', 'T2s025']

offtrx, offtry, offtrz, offrx, offry, offrz = [], [], [], [], [], []
ontrx, ontry, ontrz, onrx, onry, onrz = [], [], [], [], [], []

## PD analysis:
if contrasts[0] == 'PD':
	corr_size_ = 19502
elif contrasts[0] == 'T1':
	corr_size_ = 80807
elif contrasts[0] == 'T2':
	corr_size_ = 19502
elif contrasts[0] == 'T2s05':
	corr_size_ = 11038
elif contrasts[0] == 'T2s035':
	corr_size_ = 15358
else:
	corr_size_ = 21118
# corr_size_ = 19502 ## PD and T2 :19502 
# corr_size_ = 80807 ## T1
# corr_size_ = 11038 ## T2s05
# corr_size_ = 15358 ## T2s035
# corr_size_ = 21118 ## T2s025

#filetxt = str(contrasts[0])+"_"+str(sessions[0])+"_haralick.txt"
#myfile = open(filetxt, 'w')
for cont_ in contrasts:
	for sess in sessions:
		for file in sorted(os.listdir("./"+str(sess)+"/zz_tmp_mat/"+str(cont_)+"/")):
			if file.endswith(".mat"):
				# print(file) 
				pmc_off = os.path.join("./"+str(sess)+"/zz_tmp_mat/"+str(cont_)+"/", file)
				pmc_on = os.path.join("./On/zz_tmp_mat/"+str(cont_)+"/", file.replace('OFF','ON'))
				# print(pmc_off, pmc_on)
				tmp_data_OFF = sio.loadmat(pmc_off)
				tmp_data_ON = sio.loadmat(pmc_on)

				ftrx, ftry, ftrz, frx, fry, frz, ntrx, ntry, ntrz, nrx, nry, nrz, time_x = data_loader(tmp_data_OFF, tmp_data_ON)
				# print(time_x.shape)
				# print(ftrx.shape, ntrx.shape)
				# print(frx.shape, nrx.shape)
				## -----------------------------------------
				# - pmc off -
				ftrx = ftrx[len(ftrx)-corr_size_:]
				ftry = ftry[len(ftry)-corr_size_:]
				ftrz = ftrz[len(ftrz)-corr_size_:]
				frx = frx[len(frx)-corr_size_:]
				fry = fry[len(fry)-corr_size_:]
				frz = frz[len(frz)-corr_size_:]

				# - pmc on -
				ntrx = ntrx[len(ntrx)-corr_size_:]
				ntry = ntry[len(ntry)-corr_size_:]
				ntrz = ntrz[len(ntrz)-corr_size_:]
				nrx = nrx[len(nrx)-corr_size_:]
				nry = nry[len(nry)-corr_size_:]
				nrz = nrz[len(nrz)-corr_size_:]
				## -----------------------------------------
				offtrx.append(ftrx), offtry.append(ftry), offtrz.append(ftrz), 
				offrx.append(frx), offry.append(fry), offrz.append(frz)

				ontrx.append(ntrx), ontry.append(ntry), ontrz.append(ntrz), 
				onrx.append(nrx), onry.append(nry), onrz.append(nrz)
				## ---------------------------------------------

time_x = time_x[len(time_x)-corr_size_:]
time_x = time_x - time_x[0]

offtrx, offtry, offtrz = np.asarray(offtrx),  np.asarray(offtry),  np.asarray(offtrz) 
offrx, offry, offrz = np.asarray(offrx), np.asarray(offry),  np.asarray(offrz) 

ontrx, ontry, ontrz = np.asarray(ontrx),  np.asarray(ontry),  np.asarray(ontrz) 
onrx, onry, onrz = np.asarray(onrx), np.asarray(onry),  np.asarray(onrz)  
# print(ontrx.shape)
# print(np.mean(offtrx, axis=0).shape)

## testing
# ------------------------------------------------- #
# ------------------------------------------------- #
# ------------------------------------------------- #
# --- x ----
offtrx_ = np.mean(offtrx, axis=0)
offtrx_ = offtrx_ - offtrx_[0]
offtrxstd_ =  np.std(offtrx, axis=0) 
offtrxstd_ = offtrxstd_ - offtrxstd_[0]

ontrx_ = np.mean(ontrx, axis=0)
ontrx_ = ontrx_ -ontrx_[0]
ontrxstd_ = np.std(ontrx, axis=0)
ontrxstd_ = ontrxstd_ -ontrxstd_[0]

# --- y ----
offtry_ = np.mean(offtry, axis=0)
offtrystd_ =  np.std(offtry, axis=0) 
offtry_ = offtry_ - offtry_[0]
offtrystd_ = offtrystd_ - offtrystd_[0]

ontry_ = np.mean(ontry, axis=0)
ontry_ = ontry_ -ontrx_[0]
ontrystd_ = np.std(ontry, axis=0)
ontrystd_ = ontrystd_ -ontrystd_[0]

# --- z ----
offtrz_ = np.mean(offtrz, axis=0)
offtrzstd_ =  np.std(offtrz, axis=0) 
offtrz_ = offtrz_ - offtrz_[0]
offtrzstd_ = offtrzstd_ - offtrzstd_[0]

ontrz_ = np.mean(ontrz, axis=0)
ontrz_ = ontrz_ -ontrz_[0]
ontrzstd_ = np.std(ontrz, axis=0)
ontrzstd_ = ontrzstd_ -ontrzstd_[0]
# ------------------------------------------------- #
# ------------------------------------------------- #
# ------------------------------------------------- #
# --- rx ----
offrx_ = np.mean(offrx, axis=0)
offrx_ = offrx_ - offrx_[0]
offrxstd_ =  np.std(offrx, axis=0) 
offrxstd_ = offrxstd_ - offrxstd_[0]

onrx_ = np.mean(onrx, axis=0)
onrx_ = onrx_ -onrx_[0]
onrxstd_ = np.std(onrx, axis=0)
onrxstd_ = onrxstd_ -onrxstd_[0]

# --- ry ----
offry_ = np.mean(offry, axis=0)
offrystd_ =  np.std(offry, axis=0) 
offry_ = offry_ - offry_[0]
offrystd_ = offrystd_ - offrystd_[0]

onry_ = np.mean(onry, axis=0)
onry_ = onry_ -onry_[0]
onrystd_ = np.std(onry, axis=0)
onrystd_ = onrystd_ -onrystd_[0]

# --- rz ----
offrz_ = np.mean(offrz, axis=0)
offrz_ = offrz_ - offrz_[0]
offrzstd_ =  np.std(offrz, axis=0) 
offrzstd_ = offrzstd_ - offrzstd_[0]

onrz_ = np.mean(onrz, axis=0)
onrz_ = onrz_ -onrz_[0]
onrzstd_ = np.std(onrz, axis=0)
onrzstd_ = onrzstd_ -onrzstd_[0]

# --------------------------------------------------------------------------- #

# print('Translation x PMC OFF', offtrx_.shape, 'Translation x PMC ON', ontrx_.shape)
# print('Translation y PMC OFF', offtry_.shape, 'Translation y PMC ON', ontry_.shape)
# print('Translation z PMC OFF', offtrz_.shape, 'Translation z PMC ON', ontrz_.shape)

# print('Rotation x PMC OFF', offrx_.shape,'Rotation x PMC ON', onrx_.shape)
# print('Rotation y PMC OFF', offry_.shape,'Rotation y PMC ON', onry_.shape)
# print('Rotation z PMC OFF', offrz_.shape,'Rotation z PMC ON', onrz_.shape)
"""
bofftrx_,  bofftrxstd_ = np.sqrt(np.sum((offtrxstd_-offtrx_)**2)), np.max(offtrxstd_**2)
bofftry_,  bofftrystd_ = np.sqrt(np.sum((offtrystd_-offtry_)**2)), np.max(offtrystd_**2)
bofftrz_,  bofftrzstd_ = np.sqrt(np.sum((offtrzstd_-offtrz_)**2)), np.max(offtrzstd_**2)

boffrx_,  boffrxstd_ = np.sqrt(np.sum((offrxstd_-offrx_)**2)), np.max(offrxstd_**2)
boffry_,  boffrystd_ = np.sqrt(np.sum((offrystd_-offry_)**2)), np.max(offrystd_**2)
boffrz_,  boffrzstd_ = np.sqrt(np.sum((offrzstd_-offrz_)**2)), np.max(offrzstd_**2)

bontrx_,  bontrxstd_ = np.sqrt(np.sum((ontrxstd_-ontrx_)**2)), np.max(ontrxstd_**2)
bontry_,  bontrystd_ = np.sqrt(np.sum((ontrystd_-ontry_)**2)), np.max(ontrystd_**2)
bontrz_,  bontrzstd_ = np.sqrt(np.sum((ontrzstd_-ontrz_)**2)), np.max(ontrzstd_**2)

bonrx_,  bonrxstd_ = np.sqrt(np.sum((onrxstd_-onrx_)**2)), np.max(onrxstd_**2)
bonry_,  bonrystd_ = np.sqrt(np.sum((onrystd_-onry_)**2)), np.max(onrystd_**2)
bonrz_,  bonrzstd_ = np.sqrt(np.sum((onrzstd_-onrz_)**2)), np.max(onrzstd_**2)
"""
bofftrx_,  bofftrxstd_ = np.sqrt(np.sum((offtrxstd_)**2)), np.max(offtrxstd_**2)
bofftry_,  bofftrystd_ = np.sqrt(np.sum((offtrystd_)**2)), np.max(offtrystd_**2)
bofftrz_,  bofftrzstd_ = np.sqrt(np.sum((offtrzstd_)**2)), np.max(offtrzstd_**2)

boffrx_,  boffrxstd_ = np.sqrt(np.sum((offrxstd_)**2)), np.max(offrxstd_**2)
boffry_,  boffrystd_ = np.sqrt(np.sum((offrystd_)**2)), np.max(offrystd_**2)
boffrz_,  boffrzstd_ = np.sqrt(np.sum((offrzstd_)**2)), np.max(offrzstd_**2)

bontrx_,  bontrxstd_ = np.sqrt(np.sum((ontrxstd_)**2)), np.max(ontrxstd_**2)
bontry_,  bontrystd_ = np.sqrt(np.sum((ontrystd_)**2)), np.max(ontrystd_**2)
bontrz_,  bontrzstd_ = np.sqrt(np.sum((ontrzstd_)**2)), np.max(ontrzstd_**2)

bonrx_,  bonrxstd_ = np.sqrt(np.sum((onrxstd_)**2)), np.max(onrxstd_**2)
bonry_,  bonrystd_ = np.sqrt(np.sum((onrystd_)**2)), np.max(onrystd_**2)
bonrz_,  bonrzstd_ = np.sqrt(np.sum((onrzstd_)**2)), np.max(onrzstd_**2)

pvalues = []
stack_off_ = np.vstack((offtrx_,offtry_,offtrz_,offrx_,offry_,offrz_))
stack_on_ = np.vstack((ontrx_,ontry_,ontrz_,onrx_,onry_,onrz_))


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

#######################################################
coff_ = "#3274A1"##"#333333"
con_ = "#E1812C"####"#b2b2b2"
csame_ ="#ffffff"#"#b2b2b2" ###"#01d1a1"
cedge_ ="#000000"

blim_ = 0
maxstd_ = np.max([bofftrxstd_, bofftrystd_, bofftrzstd_, bontrxstd_, bontrystd_, bontrzstd_,
				 boffrxstd_, boffrystd_, boffrzstd_, bonrxstd_, bonrystd_, bonrzstd_])
uplim_ = np.max([bofftrx_, bofftry_, bofftrz_, bontrx_, bontry_, bontrz_,
				 boffrx_, boffry_, boffrz_, bonrx_, bonry_, bonrz_]) + 5 # maxstd_ +5
#
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
read1off_ = [bofftrx_, bofftry_, bofftrz_, boffrx_, boffry_, boffrz_]
read1on_ = [bontrx_, bontry_, bontrz_, bonrx_, bonry_, bonrz_]

xerroroff_ = [bofftrxstd_, bofftrystd_, bofftrzstd_, boffrxstd_, boffrystd_, boffrzstd_]
xerroron_ = [bontrxstd_, bontrystd_, bontrzstd_, bonrxstd_, bonrystd_, bonrzstd_]          

# l1 = ax1.barh(assey_, read1off_, xerr=xerroroff_,  height=width,  color=coff_ , edgecolor=cedge_, capsize=3)
# l2 = ax1.barh(assey_+width, read1on_, xerr=xerroron_, height=width,  color= con_, edgecolor=cedge_, capsize=3)  #, hatch='/' )

l1 = ax1.barh(assey_, read1off_,  height=width,  color=coff_ , edgecolor=cedge_, capsize=3)
l2 = ax1.barh(assey_+width, read1on_,  height=width,  color= con_, edgecolor=cedge_, capsize=3)  #, hatch='/' )

plt.xticks(fontsize=10, fontweight='bold')
plt.yticks(assey_+width/2, bar_finlabels, fontsize=10, fontweight='bold')
plt.xlabel('[mm] for displacements and [°] for rotations', fontsize=10, fontweight='bold')
ax1.set_xlim(blim_, uplim_)
if contrasts[0] == 'PD':
	title_ = 'PD-w scans'
elif contrasts[0] == 'T1':
	title_ = 'T1-w scans'
elif contrasts[0] == 'T2':
	title_ = 'T2-w scans'
elif contrasts[0] == 'T2s05':
	title_ = 'T2*-w (05) scans'
elif contrasts[0] == 'T2s035':
	title_ = 'T2*-w (035) scans'
else:
	title_ = 'T2*-w (025) scans'

plt.title(str(title_), fontsize=10, fontweight='bold')
ax1.legend([l1, l2],     # The line objects
           labels=line_labels,   # The labels for each line
           # loc="upper right",# "upper center",   # Position of legend
           # borderaxespad=0.1,    # Small spacing around legend box
           #title=""# "Legend Title"  # Title for the legend
           fontsize=9)

########################################################################################
########################################################################################
# props = dict(boxstyle='round', facecolor='wheat', alpha=1)
# for ii in range(0, len(pvalues)):
#     ax1.text( uplim_+1, assey_[ii]+width/2, str(convert_pval(pvalues[ii])), fontweight='bold', fontsize=7, bbox=props)
########################################################################################
########################################################################################
ax1.grid(True)
## save to file:

# np.save(str(contrasts[0])+'-off-sum.npy', read1off_)
# np.save(str(contrasts[0])+'-on-sum.npy', read1on_)


plt.show()
