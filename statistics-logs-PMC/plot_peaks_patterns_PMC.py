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


sessions = ['Off']#['Off', 'On']
contrasts = ['PD']#['PD', 'T1', 'T2', 'T2s05', 'T2s035', 'T2s025']

offtrx, offtry, offtrz, offrx, offry, offrz = [], [], [], [], [], []
ontrx, ontry, ontrz, onrx, onry, onrz = [], [], [], [], [], []

## PD analysis:
corr_size_ = 19502 ## PD and T2 :19502 
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
def peaks_finder(input_):#=[intrx, intry, intrz, inrx, inry, inrz]):
	peaks_ = np.array([0])
	for ii in input_:
		# print(ii.shape)
		test_ = ii-ii[0]
		prom_ =  .2*np.abs(stats.kurtosis(test_))
		# 100*np.abs(np.var(test_))
		#.2*np.abs(stats.kurtosis(test_)) #2.*np.abs(np.mean(test_))
		test_ = medfilt(test_, kernel_size=25)
		# peaks, _= find_peaks(test_, threshold=0.5*np.abs(np.mean(test_)), prominence=prom_)
		peaks, _= find_peaks(test_,  prominence=prom_)
		npeaks, _ = find_peaks(1-test_, prominence=prom_)
		concpeaks_ = np.concatenate((peaks, npeaks))
		peaks_ = np.concatenate((peaks_, concpeaks_))

	# print(peaks_.shape)
	# peaks_ = np.unique
	return np.unique(peaks_[1:])
# --- x ----
# offtrx_ = np.mean(offtrx, axis=0)
# offtrx_ = offtrx_ - offtrx_[0]
# offtrxstd_ =  np.std(offtrx, axis=0) 
# offtrxstd_ = offtrxstd_ - offtrxstd_[0]
"""
print(offtrx.shape)
test_ = offtrx[0,:]-offtrx[0,0]
print(np.mean(test_), np.std(test_))
prom_ = np.abs(np.mean(test_))
#peaks, properties = find_peaks(test_, prominence=prom_)#0.15)# height=-0.2)#, prominence=1, width=1.5)
#npeaks, nproperties = find_peaks(1-test_, prominence=prom_)#0.15)#, height=0.3)
peaks, _= find_peaks(test_, prominence=prom_)
npeaks, _ = find_peaks(1-test_, prominence=prom_)

concpeaks_ = np.concatenate((peaks, npeaks))
#print(peaks)#, properties)
#skpeaks =peak_local_max(test_, min_distance=500)
plt.figure()
plt.plot(test_)
plt.plot(concpeaks_, test_[concpeaks_],"o")
#plt.plot(peaks, test_[peaks],"x")
#plt.plot(npeaks, test_[npeaks],"o")
#plt.plot(skpeaks, test_[skpeaks], "d")
plt.show()
"""
# print(offtrx.shape)
ssub = 0
eee_ = peaks_finder(input_=[offtrx[ssub,:], offtry[ssub,:], offtrz[ssub,:], 
							offrx[ssub,:], offry[ssub,:], offrz[ssub,:]])

fff_ = peaks_finder(input_=[ontrx[ssub,:], ontry[ssub,:], ontrz[ssub,:], 
							onrx[ssub,:], onry[ssub,:], onrz[ssub,:]])

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

print(eee_.shape, fff_.shape)

#plt.figure()
#plt.subplot(211)
#plt.plot(test_)
#plt.subplot(212)
#plt.plot(medfilt(test_, kernel_size=25))

plt.figure()
plt.subplot(231)
plt.plot(testx_)
plt.plot(eee_, testx_[eee_],"o")
plt.plot(testfx_)
plt.plot(fff_, testfx_[fff_], "o")
plt.grid()

plt.subplot(232)
plt.plot(testy_)
plt.plot(eee_, testy_[eee_],"o")
plt.plot(testfy_)
plt.plot(fff_, testfy_[fff_], "o")
plt.grid()

plt.subplot(233)
plt.plot(testz_)
plt.plot(eee_, testz_[eee_],"o")
plt.plot(testfz_)
plt.plot(fff_, testfz_[fff_], "o")
plt.grid()

plt.subplot(234)
plt.plot(testrx_)
plt.plot(eee_, testrx_[eee_],"o")
plt.plot(testfrx_)
plt.plot(fff_, testfrx_[fff_], "o")
plt.grid()

plt.subplot(235)
plt.plot(testry_)
plt.plot(eee_, testry_[eee_],"o")
plt.plot(testfry_)
plt.plot(fff_, testfry_[fff_], "o")
plt.grid()

plt.subplot(236)
plt.plot(testrz_)
plt.plot(eee_, testrz_[eee_],"o")
plt.plot(testfrz_)
plt.plot(fff_, testfrz_[fff_], "o")
plt.grid()


plt.show()

