import glob, os
import math
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import norm

def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])

# ------------------------------------------
def calculate_diff(inarray):
	diff_ = []
	for ii in range(1,len(inarray)):
		diff_.append(inarray[ii]-inarray[ii-1])

	return np.asarray(diff_)


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
contrasts = ['T2s025']#['PD', 'T1', 'T2', 'T2s05', 'T2s035', 'T2s025']


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

filetxt = str(contrasts[0])+"_distr_mean_std.txt"
myfile = open(filetxt, 'w')
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
				
				offtrx_ = ftrx - ftrx[0]
				offtry_ = ftry - ftry[0]
				offtrz_ = ftrz - ftrz[0]

				offrx_ = frx - frx[0]
				offry_ = fry - fry[0]
				offrz_ = frz - frz[0]


				ontrx_ = ntrx - ntrx[0]
				ontry_ = ntry - ntry[0]
				ontrz_ = ntrz - ntrz[0]

				onrx_ = nrx - nrx[0]
				onry_ = nry - nry[0]
				onrz_ = nrz - nrz[0]

				## --------------------------------------------
				tmp_offtrx = calculate_diff(offtrx_[0::30])
				tmp_ontrx = calculate_diff(ontrx_[0::30])

				tmp_offtry = calculate_diff(offtry_[0::30])
				tmp_ontry = calculate_diff(ontry_[0::30])

				tmp_offtrz = calculate_diff(offtrz_[0::30])
				tmp_ontrz = calculate_diff(ontrz_[0::30])

				tmp_offrx = calculate_diff(offrx_[0::30])
				tmp_onrx = calculate_diff(onrx_[0::30])

				tmp_offry = calculate_diff(offry_[0::30])
				tmp_onry = calculate_diff(onry_[0::30])

				tmp_offrz = calculate_diff(offrz_[0::30])
				tmp_onrz = calculate_diff(onrz_[0::30])

				muofftrx, stdofftrx = norm.fit(tmp_offtrx)
				muofftry, stdofftry = norm.fit(tmp_offtry)
				muofftrz, stdofftrz = norm.fit(tmp_offtrz)

				muoffrx, stdoffrx = norm.fit(tmp_offrx)
				muoffry, stdoffry = norm.fit(tmp_offry)
				muoffrz, stdoffrz = norm.fit(tmp_offrz)

				muontrx, stdontrx = norm.fit(tmp_ontrx)
				muontry, stdontry = norm.fit(tmp_ontry)
				muontrz, stdontrz = norm.fit(tmp_ontrz)

				muonrx, stdonrx = norm.fit(tmp_onrx)
				muonry, stdonry = norm.fit(tmp_onry)
				muonrz, stdonrz = norm.fit(tmp_onrz)

				text_ = [file[0:4], muofftrx, stdofftrx, muofftry, stdofftry, muofftrz, stdofftrz,
									muoffrx, stdoffrx, muoffry, stdoffry, muoffrz, stdoffrz,
									muontrx, stdontrx, muontry, stdontry, muontrz, stdontrz,
									muonrx, stdonrx, muonry, stdonry, muonrz, stdonrz]
				tmp_line_ = str(text_)
				tmp_line_ = tmp_line_.replace('[','')
				tmp_line_ = tmp_line_.replace(']','')
				print(tmp_line_)
				myfile.write("%s\n" % str(tmp_line_))
				