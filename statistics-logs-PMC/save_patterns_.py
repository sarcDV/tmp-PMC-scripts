import glob, os
import math
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt 
from scipy.signal import find_peaks
from scipy.signal import medfilt
from skimage.feature.peak import peak_local_max
import scipy.stats as stats

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


def find_peaks_thr(vect1_, thr_, dist_):
	vect2_ = np.array([0])# np.zeros((vect1_.shape))
	
	for ii in range(dist_,len(vect1_)):
		if np.abs(vect1_[ii]-vect1_[ii-dist_])>thr_:
			vect2_=np.concatenate((vect2_,np.array([ii])), axis=0)
			#vect2_[ii] = 1 #vect1_[ii]

	return vect2_[1:]


sessions = ['Off']#['Off', 'On']
selection = 0 # 0 = PD, 1 = T1, 2 = T2, 3 = T2s05, 4 = T2s035, 5 = T2s025
list_contrasts = ["PD", "T1", "T2", "T2s05", "T2s035", "T2s025"]
corrections_ = [19502 ,80807, 19502, 11038, 15358, 21118] 
contrasts = list_contrasts[selection] #['PD', 'T1', 'T2', 'T2s05', 'T2s035', 'T2s025']
corr_size_ = corrections_[selection]
print(contrasts, corr_size_)

offtrx, offtry, offtrz, offrx, offry, offrz = [], [], [], [], [], []
ontrx, ontry, ontrz, onrx, onry, onrz = [], [], [], [], [], []

## PD analysis:
# corr_size_ = 19502 ## PD and T2 :19502 
# corr_size_ = 80807 ## T1
# corr_size_ = 11038 ## T2s05
# corr_size_ = 15358 ## T2s035
# corr_size_ = 21118 ## T2s025

#filetxt = str(contrasts[0])+"_"+str(sessions[0])+"_haralick.txt"
#myfile = open(filetxt, 'w')

# number of peaks divided by 1/3 len(time_x)

list_sub = []
for sess in sessions:
	for file in sorted(os.listdir("./"+str(sess)+"/zz_tmp_mat/"+str(contrasts)+"/")):
		if file.endswith(".mat"):
			subname_ = file[0:4]
			list_sub.append(subname_)
			# print(subname_)
			pmc_off = os.path.join("./"+str(sess)+"/zz_tmp_mat/"+str(contrasts)+"/", file)
			pmc_on = os.path.join("./On/zz_tmp_mat/"+str(contrasts)+"/", file.replace('OFF','ON'))
			#print(pmc_off, pmc_on)
			tmp_data_OFF = sio.loadmat(pmc_off)
			tmp_data_ON = sio.loadmat(pmc_on)

			ftrx, ftry, ftrz, frx, fry, frz, ntrx, ntry, ntrz, nrx, nry, nrz, time_x = data_loader(tmp_data_OFF, tmp_data_ON)
			# print(time_x.shape)
			#print(ftrx.shape, ntrx.shape)
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

# print(list_sub)

offtrx, offtry, offtrz = np.asarray(offtrx),  np.asarray(offtry),  np.asarray(offtrz) 
offrx, offry, offrz = np.asarray(offrx), np.asarray(offry),  np.asarray(offrz) 

ontrx, ontry, ontrz = np.asarray(ontrx),  np.asarray(ontry),  np.asarray(ontrz) 
onrx, onry, onrz = np.asarray(onrx), np.asarray(onry),  np.asarray(onrz)  

for ssub in range(0,21):
	# print(list_sub[ssub])
	plt.close()
	testx_ = offtrx[ssub,:]-offtrx[0,0]
	testy_ = offtry[ssub,:]-offtry[0,0]
	testz_ = offtrz[ssub,:]-offtrz[0,0]

	testrx_ = offrx[ssub,:]-offrx[0,0]
	testry_ = offry[ssub,:]-offry[0,0]
	testrz_ = offrz[ssub,:]-offrz[0,0]

	testfx_ = ontrx[ssub,:]-ontrx[0,0]
	testfy_ = ontry[ssub,:]-ontry[0,0]
	testfz_ = ontrz[ssub,:]-ontrz[0,0]

	testfrx_ = onrx[ssub,:]-onrx[0,0]
	testfry_ = onry[ssub,:]-onry[0,0]
	testfrz_ = onrz[ssub,:]-onrz[0,0]

	# print(testx_.max(), testy_.max(), testz_.max(), testrx_.max(), testry_.max(), testz_.max())
	# max and min translations:
	maxtr_ = np.array([testx_.max(), testy_.max(), testz_.max(), 
					testfx_.max(), testfy_.max(), testfz_.max()])
	mintr_ = np.array([testx_.min(), testy_.min(), testz_.min(), 
					testfx_.min(), testfy_.min(), testfz_.min()])

	# max and min rotations:
	maxrt_ = np.array([testrx_.max(), testry_.max(), testrz_.max(), 
					testfrx_.max(), testfry_.max(), testfrz_.max()])
	minrt_ = np.array([testrx_.min(), testry_.min(), testrz_.min(), 
					testfrx_.min(), testfry_.min(), testfrz_.min()])

	vthr_ = .25
	vdist_ = 5#0
	aoff_ = []
	aon_ = []

	coff_ = "#3274A1"##"#333333"
	con_ = "#E1812C"
	# -------------------------------------------------------
	plt.figure(figsize=(16,8))
	# -------------------------------------------------------
	# -------------------------------------------------------
	# -------------------------------------------------------
	# plt.suptitle(str(contrasts)+' - subject: '+ str(list_sub[ssub]) )
	plt.subplot(231)
	testpeaks_ = find_peaks_thr(testx_, thr_ =vthr_, dist_ = vdist_)
	plt.plot(time_x, testx_, label='Displ-x PMC OFF', color=coff_)
	plt.plot(time_x[testpeaks_], testx_[testpeaks_], 'o', color='red', label='Peaks PMC OFF')
	testpeaksf_ = find_peaks_thr(testfx_, thr_ =vthr_, dist_ = vdist_)
	plt.plot(time_x, testfx_, label='Displ-x PMC ON', color=con_)
	plt.plot(time_x[testpeaksf_], testfx_[testpeaksf_], 'o', color='green', label='Peaks PMC ON')
	plt.axis([0, time_x[-1], mintr_.min()-0.25 , maxtr_.max()+0.25])
	plt.ylabel('Displacements (mm)', fontsize=12, fontweight='bold')
	plt.xticks(fontsize=0, color='white')
	plt.yticks(fontsize=10, fontweight='bold')
	plt.legend()
	ax = plt.gca()
	ax.set_facecolor('#E5ECF6')
	aoff_.append(np.asarray(testpeaks_.shape))
	aon_.append(np.asarray(testpeaksf_.shape))
	plt.grid(color='white', linewidth=0.82)
	# -------------------------------------------------------
	# -------------------------------------------------------
	# -------------------------------------------------------
	plt.subplot(232)
	plt.plot(time_x, testy_, label='Displ-y PMC OFF', color=coff_)
	testpeaks_ = find_peaks_thr(testy_, thr_ =vthr_, dist_ = vdist_)
	plt.plot(time_x[testpeaks_], testy_[testpeaks_], 'o', color='red', label='Peaks PMC OFF')
	plt.plot(time_x, testfy_, label='Displ-y PMC ON', color=con_)
	testpeaksf_ = find_peaks_thr(testfy_, thr_ =vthr_, dist_ = vdist_)
	plt.plot(time_x[testpeaksf_], testfy_[testpeaksf_], 'o', color='green', label='Peaks PMC ON')
	plt.axis([0,  time_x[-1], mintr_.min()-0.25 , maxtr_.max()+0.25])
	plt.yticks(fontsize=0, color='white')
	plt.xticks(fontsize=0, color='white')
	ax = plt.gca()
	ax.set_facecolor('#E5ECF6')
	plt.legend()
	aoff_.append(np.asarray(testpeaks_.shape))
	aon_.append(np.asarray(testpeaksf_.shape))
	plt.grid(color='white', linewidth=0.82)
	# -------------------------------------------------------
	# -------------------------------------------------------
	# -------------------------------------------------------
	plt.subplot(233)
	plt.plot(time_x, testz_, label='Displ-x PMC OFF', color=coff_)
	testpeaks_ = find_peaks_thr(testz_, thr_ =vthr_, dist_ = vdist_)
	plt.plot(time_x[testpeaks_], testz_[testpeaks_], 'o', color='red', label='Peaks PMC OFF')
	plt.plot(time_x, testfz_, label='Displ-x PMC ON', color=con_)
	testpeaksf_ = find_peaks_thr(testfz_, thr_ =vthr_, dist_ = vdist_)
	plt.plot(time_x[testpeaksf_], testfz_[testpeaksf_], 'o', color='green', label='Peaks PMC ON')
	plt.axis([0,  time_x[-1], mintr_.min()-0.25 , maxtr_.max()+0.25])
	plt.legend()
	plt.yticks(fontsize=0, color='white')
	plt.xticks(fontsize=0, color='white')
	ax = plt.gca()
	ax.set_facecolor('#E5ECF6')
	aoff_.append(np.asarray(testpeaks_.shape))
	aon_.append(np.asarray(testpeaksf_.shape))
	plt.grid(color='white', linewidth=0.82)
	# -------------------------------------------------------
	# -------------------------------------------------------
	# -------------------------------------------------------
	plt.subplot(234)
	plt.plot(time_x, testrx_, label='Roll PMC OFF', color=coff_)
	testpeaks_ = find_peaks_thr(testrx_, thr_ =vthr_, dist_ = vdist_)
	plt.plot(time_x[testpeaks_], testrx_[testpeaks_], 'o', color='red', label='Peaks PMC OFF')
	plt.plot(time_x, testfrx_, label='Roll PMC ON', color=con_)
	testpeaksf_ = find_peaks_thr(testfrx_, thr_ =vthr_, dist_ = vdist_)
	plt.plot(time_x[testpeaksf_], testfrx_[testpeaksf_], 'o', color='green', label='Peaks PMC ON')
	plt.axis([0,  time_x[-1], minrt_.min()-0.25 , maxrt_.max()+0.25])
	plt.ylabel('Rotations (°)', fontsize=12, fontweight='bold')
	plt.xlabel('Time (seconds)', fontsize=12, fontweight='bold')
	plt.legend()
	plt.yticks(fontsize=10, fontweight='bold')
	plt.xticks(fontsize=10, fontweight='bold')
	ax = plt.gca()
	ax.set_facecolor('#E5ECF6')
	aoff_.append(np.asarray(testpeaks_.shape))
	aon_.append(np.asarray(testpeaksf_.shape))
	plt.grid(color='white', linewidth=0.82)
	# -------------------------------------------------------
	# -------------------------------------------------------
	# -------------------------------------------------------
	plt.subplot(235)
	plt.plot(time_x, testry_, label='Yaw PMC OFF', color=coff_)
	testpeaks_ = find_peaks_thr(testry_, thr_ =vthr_, dist_ = vdist_)
	plt.plot(time_x[testpeaks_], testry_[testpeaks_], 'o', color='red', label='Peaks PMC OFF')
	plt.plot(time_x, testfry_, label='Yaw PMC ON', color=con_)
	testpeaksf_ = find_peaks_thr(testfry_, thr_ =vthr_, dist_ = vdist_)
	plt.plot(time_x[testpeaksf_], testfry_[testpeaksf_], 'o', color='green', label='Peaks PMC ON')
	plt.axis([0,  time_x[-1], minrt_.min()-0.25 , maxrt_.max()+0.25])
	plt.xlabel('Time (seconds)', fontsize=12, fontweight='bold')
	plt.yticks(fontsize=0, color='white')
	plt.xticks(fontsize=10, fontweight='bold')
	plt.legend()
	ax = plt.gca()
	ax.set_facecolor('#E5ECF6')
	aoff_.append(np.asarray(testpeaks_.shape))
	aon_.append(np.asarray(testpeaksf_.shape))
	plt.grid(color='white', linewidth=0.82)
	# -------------------------------------------------------
	# -------------------------------------------------------
	# -------------------------------------------------------
	plt.subplot(236)
	plt.plot(time_x, testrz_, label='Pitch PMC OFF', color=coff_)
	testpeaks_ = find_peaks_thr(testrz_, thr_ =vthr_, dist_ = vdist_)
	plt.plot(time_x[testpeaks_], testrz_[testpeaks_], 'o', color='red', label='Peaks PMC OFF')
	plt.plot(time_x, testfrz_, label='Pitch PMC ON', color=con_)
	testpeaksf_ = find_peaks_thr(testfrz_, thr_ =vthr_, dist_ = vdist_)
	plt.plot(time_x[testpeaksf_], testfrz_[testpeaksf_], 'o', color='green', label='Peaks PMC ON')
	plt.axis([0,  time_x[-1], minrt_.min()-0.25 , maxrt_.max()+0.25])
	plt.xlabel('Time (seconds)', fontsize=12, fontweight='bold')
	plt.yticks(fontsize=0, color='white')
	plt.xticks(fontsize=10, fontweight='bold')
	plt.legend()
	ax = plt.gca()
	ax.set_facecolor('#E5ECF6')
	# print(testpeaks_.shape, testpeaksf_.shape)
	aoff_.append(np.asarray(testpeaks_.shape))
	aon_.append(np.asarray(testpeaksf_.shape))
	plt.grid(color='white', linewidth=0.82)
	plt.tight_layout()	
	plt.show()
	# print("\n")
	aoff_ = np.asarray(aoff_).squeeze()
	aon_ = np.asarray(aon_).squeeze()
	
	print(str(list_sub[ssub]), np.sum(aoff_)/corr_size_,  np.sum(aon_)/corr_size_)
	# print(str(list_sub[ssub])," OFF ", np.sum(aoff_)/corr_size_)
	# print( str(list_sub[ssub])," ON ", np.sum(aon_)/corr_size_)

	# print(aoff_.shape, aoff_, np.mean(aoff_), np.sum(aoff_)/corr_size_, np.sum(aoff_)/time_x[-1])
	# print(aon_.shape, aon_, np.mean(aon_), np.sum(aon_)/corr_size_, np.sum(aon_)/time_x[-1])
	# plt.show()	
	# figManager = plt.get_current_fig_manager()
	# figManager.window.showMaximized()
	
	# figure = plt.gcf()
	# plt.savefig(str(list_sub[ssub])+"_T2025.pdf")#, dpi=150)
	
	# plt.savefig(str(list_sub[ssub])+"_"+str(contrasts)+".png")
	# 
	