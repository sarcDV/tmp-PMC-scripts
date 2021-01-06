import glob, os
import math
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt 


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
contrasts = ['T1']#['PD', 'T1', 'T2', 'T2s05', 'T2s035', 'T2s025']

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

#filetxt = str(contrasts[0])+"_"+str(sessions[0])+"_haralick.txt"
#myfile = open(filetxt, 'w')
for cont_ in contrasts:
	for sess in sessions:
		for file in sorted(os.listdir("./"+str(sess)+"/zz_tmp_mat/"+str(cont_)+"/")):
			if file.endswith(".mat"):
				# print(file) 
				pmc_off = os.path.join("./"+str(sess)+"/zz_tmp_mat/"+str(cont_)+"/", file)
				pmc_on = os.path.join("./On/zz_tmp_mat/"+str(cont_)+"/", file.replace('OFF','ON'))
				print(pmc_off, pmc_on)
				tmp_data_OFF = sio.loadmat(pmc_off)
				tmp_data_ON = sio.loadmat(pmc_on)

				ftrx, ftry, ftrz, frx, fry, frz, ntrx, ntry, ntrz, nrx, nry, nrz, time_x = data_loader(tmp_data_OFF, tmp_data_ON)
				print(time_x.shape)
				print(ftrx.shape, ntrx.shape)
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

# trlowlim_, truplim_ = -1.5,1.5
# rtlowlim_, rtuplim_ = -.6, .6
axtrlowlim_, axtruplim_ = np.min((offtrx_-(offtrxstd_))[0::30]),  np.max((offtrx_+(offtrxstd_))[0::30]) # -1.5,1.5
aytrlowlim_, aytruplim_ = np.min((offtry_-(offtrystd_))[0::30]),  np.max((offtry_+(offtrystd_))[0::30]) # -1.5,1.5
aztrlowlim_, aztruplim_ = np.min((offtrz_-(offtrzstd_))[0::30]),  np.max((offtrz_+(offtrzstd_))[0::30]) # -1.5,1.5

axrtlowlim_, axrtuplim_ = np.min((offrx_-(offrxstd_))[0::30]),  np.max((offrx_+(offrxstd_))[0::30]) # -1.5,1.5
ayrtlowlim_, ayrtuplim_ = np.min((offry_-(offrystd_))[0::30]),  np.max((offry_+(offrystd_))[0::30]) # -1.5,1.5
azrtlowlim_, azrtuplim_ = np.min((offrz_-(offrzstd_))[0::30]),  np.max((offrz_+(offrzstd_))[0::30]) # -1.5,1.5

bxtrlowlim_, bxtruplim_ = np.min((ontrx_-(ontrxstd_))[0::30]),  np.max((ontrx_+(ontrxstd_))[0::30]) # -1.5,1.5
bytrlowlim_, bytruplim_ = np.min((ontry_-(ontrystd_))[0::30]),  np.max((ontry_+(ontrystd_))[0::30]) # -1.5,1.5
bztrlowlim_, bztruplim_ = np.min((ontrz_-(ontrzstd_))[0::30]),  np.max((ontrz_+(ontrzstd_))[0::30]) # -1.5,1.5

bxrtlowlim_, bxrtuplim_ = np.min((onrx_-(onrxstd_))[0::30]),  np.max((onrx_+(onrxstd_))[0::30]) # -1.5,1.5
byrtlowlim_, byrtuplim_ = np.min((onry_-(onrystd_))[0::30]),  np.max((onry_+(onrystd_))[0::30]) # -1.5,1.5
bzrtlowlim_, bzrtuplim_ = np.min((onrz_-(onrzstd_))[0::30]),  np.max((onrz_+(onrzstd_))[0::30]) # -1.5,1.5

trlowlim_ = np.min([axtrlowlim_, aytrlowlim_, aztrlowlim_, bxtrlowlim_, bytrlowlim_, bztrlowlim_])
truplim_ = np.max([axtruplim_, aytruplim_, aztruplim_, bxtruplim_, bytruplim_, bztruplim_])

rtlowlim_ = np.min([axrtlowlim_, ayrtlowlim_, azrtlowlim_, bxrtlowlim_, byrtlowlim_, bzrtlowlim_])
rtuplim_ = np.max([axrtuplim_, ayrtuplim_, azrtuplim_, bxrtuplim_, byrtuplim_, bzrtuplim_]) 

coff_ = "#3274A1"  ## "#333333" ## '#ff3434'
con_ = "#E1812C"  ## "#b2b2b2" ## '#0b850b'

#c = "#333333"
#c2 = "#b2b2b2"
linewid_ = 1.6
markers_ = 1.6
fsl_ = 12

plt.figure(figsize=(16,8))
### 
plt.subplot(231)
plt.plot(time_x, offtrx_, color=coff_, label='Displ-x OMTS OFF' )
plt.plot(time_x[0::30], (offtrx_+(offtrxstd_))[0::30], ':', linewidth=linewid_, ms=markers_, color=coff_)
plt.plot(time_x[0::30], (offtrx_-(offtrxstd_))[0::30], ':', linewidth=linewid_, ms=markers_, color=coff_)
plt.plot(time_x, ontrx_, color=con_, label='Displ-x OMTS ON' )
plt.plot(time_x[0::30], (ontrx_+(ontrxstd_))[0::30], ':', linewidth=linewid_, ms=markers_, color=con_)
plt.plot(time_x[0::30], (ontrx_-(ontrxstd_))[0::30], ':', linewidth=linewid_, ms=markers_, color=con_)
plt.grid(color='white', linewidth=0.82)
# plt.xlabel('Time (seconds)', fontsize=12, fontweight='bold')
plt.ylabel('Displacements (mm)', fontsize=12, fontweight='bold')
ax = plt.gca()
ax.set_facecolor('#E5ECF6')
plt.xlim([0,time_x[-1]])
plt.ylim([trlowlim_, truplim_])
plt.xticks(fontsize=0, color='white')
plt.yticks(fontsize=10, fontweight='bold')
plt.legend(fontsize=fsl_)
# ----------------------------------------------------------------------------------------
plt.subplot(232)
plt.plot(time_x, offtry_, color=coff_, label='Displ-y OMTS OFF' )
plt.plot(time_x[0::30], (offtry_+(offtrystd_))[0::30], ':', linewidth=linewid_, ms=markers_, color=coff_)
plt.plot(time_x[0::30], (offtry_-(offtrystd_))[0::30], ':', linewidth=linewid_, ms=markers_, color=coff_)
plt.plot(time_x, ontrx_, color=con_, label='Displ-y OMTS ON' )
plt.plot(time_x[0::30], (ontry_+(ontrystd_))[0::30], ':', linewidth=linewid_, ms=markers_, color=con_)
plt.plot(time_x[0::30], (ontry_-(ontrystd_))[0::30], ':', linewidth=linewid_, ms=markers_, color=con_)
plt.grid(color='white', linewidth=0.82)
# plt.xlabel('Time (seconds)', fontsize=12, fontweight='bold')
# plt.ylabel('Displacements (mm)', fontsize=12, fontweight='bold')
ax = plt.gca()
ax.set_facecolor('#E5ECF6')
plt.xlim([0,time_x[-1]])
plt.ylim([trlowlim_, truplim_])
plt.xticks(fontsize=0, color='white')
plt.yticks(fontsize=0, color='white')
plt.legend(fontsize=fsl_)
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
plt.subplot(233)
plt.plot(time_x, offtrz_, color=coff_, label='Displ-z OMTS OFF' )
plt.plot(time_x[0::30], (offtrz_+(offtrzstd_))[0::30], ':', linewidth=linewid_, ms=markers_, color=coff_)
plt.plot(time_x[0::30], (offtrz_-(offtrzstd_))[0::30], ':', linewidth=linewid_, ms=markers_, color=coff_)
plt.plot(time_x, ontrx_, color=con_, label='Displ-y OMTS ON' )
plt.plot(time_x[0::30], (ontrz_+(ontrzstd_))[0::30], ':', linewidth=linewid_, ms=markers_, color=con_)
plt.plot(time_x[0::30], (ontrz_-(ontrzstd_))[0::30], ':', linewidth=linewid_, ms=markers_, color=con_)
plt.grid(color='white', linewidth=0.82)
# plt.xlabel('Time (seconds)', fontsize=12, fontweight='bold')
# plt.ylabel('Displacements (mm)', fontsize=12, fontweight='bold')
# plt.box(False)
ax = plt.gca()
ax.set_facecolor('#E5ECF6')
plt.xlim([0,time_x[-1]])
plt.ylim([trlowlim_, truplim_])
plt.xticks(fontsize=0, color='white')
plt.yticks(fontsize=0, color='white')
plt.legend(fontsize=fsl_)

# ----------------------------------------------------------------------------------------
plt.subplot(236)
plt.plot(time_x, offrx_, color=coff_, label='Pitch OMTS OFF' )
plt.plot(time_x[0::30], (offrx_+(offrxstd_))[0::30], ':', linewidth=linewid_, ms=markers_, color=coff_)
plt.plot(time_x[0::30], (offrx_-(offrxstd_))[0::30], ':', linewidth=linewid_, ms=markers_, color=coff_)
plt.plot(time_x, onrx_, color=con_, label='Pitch OMTS ON' )
plt.plot(time_x[0::30], (onrx_+(onrxstd_))[0::30], ':', linewidth=linewid_, ms=markers_, color=con_)
plt.plot(time_x[0::30], (onrx_-(onrxstd_))[0::30], ':', linewidth=linewid_, ms=markers_, color=con_)
plt.grid(color='white', linewidth=0.82)
plt.xlabel('Time (seconds)', fontsize=12, fontweight='bold')
#plt.ylabel('Rotations (degrees)', fontsize=12, fontweight='bold')
ax = plt.gca()
ax.set_facecolor('#E5ECF6')
plt.xlim([0,time_x[-1]])
plt.ylim([rtlowlim_, rtuplim_])
plt.yticks(fontsize=0, color='white')
plt.xticks(fontsize=10, fontweight='bold')
plt.legend(fontsize=fsl_)
# ----------------------------------------------------------------------------------------
plt.subplot(235)
plt.plot(time_x, offry_, color=coff_, label='Yaw OMTS OFF' )
plt.plot(time_x[0::30], (offry_+(offrystd_))[0::30], ':', linewidth=linewid_, ms=markers_, color=coff_)
plt.plot(time_x[0::30], (offry_-(offrystd_))[0::30], ':', linewidth=linewid_, ms=markers_, color=coff_)
plt.plot(time_x, onry_, color=con_, label='Yaw OMTS ON' )
plt.plot(time_x[0::30], (onry_+(onrystd_))[0::30], ':', linewidth=linewid_, ms=markers_, color=con_)
plt.plot(time_x[0::30], (onry_-(onrystd_))[0::30], ':', linewidth=linewid_, ms=markers_, color=con_)
plt.grid(color='white', linewidth=0.82)
plt.xlabel('Time (seconds)', fontsize=12, fontweight='bold')
# plt.ylabel('Rotations (degrees)', fontsize=12, fontweight='bold')
ax = plt.gca()
ax.set_facecolor('#E5ECF6')
plt.xlim([0,time_x[-1]])
plt.ylim([rtlowlim_, rtuplim_])
plt.yticks(fontsize=0, color='white')
plt.xticks(fontsize=10, fontweight='bold')
plt.legend(fontsize=fsl_)
# ----------------------------------------------------------------------------------------
plt.subplot(234)
plt.plot(time_x, offrz_, color=coff_, label='Roll OMTS OFF' )
plt.plot(time_x[0::30], (offrz_+(offrzstd_))[0::30], ':', linewidth=linewid_, ms=markers_, color=coff_)
plt.plot(time_x[0::30], (offrz_-(offrzstd_))[0::30], ':', linewidth=linewid_, ms=markers_, color=coff_)
plt.plot(time_x, onrz_, color=con_, label='Roll OMTS ON' )
plt.plot(time_x[0::30], (onrz_+(onrzstd_))[0::30], ':', linewidth=linewid_, ms=markers_, color=con_)
plt.plot(time_x[0::30], (onrz_-(onrzstd_))[0::30], ':', linewidth=linewid_, ms=markers_, color=con_)
plt.grid(color='white', linewidth=0.82)
plt.xlabel('Time (seconds)', fontsize=12, fontweight='bold')
plt.ylabel('Rotations (°)', fontsize=12, fontweight='bold')
plt.yticks(fontsize=10, fontweight='bold')
plt.xticks(fontsize=10, fontweight='bold')
ax = plt.gca()
ax.set_facecolor('#E5ECF6')
plt.xlim([0,time_x[-1]])
plt.ylim([rtlowlim_, rtuplim_])
plt.legend(fontsize=fsl_)

# ----------------------------------------------------------------------------------------
plt.tight_layout()
plt.show()




