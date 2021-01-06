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
contrasts = ['PD']#['PD', 'T1', 'T2', 'T2s05', 'T2s035', 'T2s025']


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


				nbins=512 #'auto'

				coff_ = "#3274A1"  ## "#333333" ## '#ff3434'
				con_ = "#E1812C"  ## "#b2b2b2" ## '#0b850b'
				plt.figure(figsize=(16,8))
				# 
				plt.subplot(231)
				plt.hist(tmp_offtrx, bins=nbins, alpha=0.6, color=coff_)
				muoff, stdoff = norm.fit(tmp_offtrx)
				xminoff, xmaxoff = plt.xlim()
				xoff = np.linspace(xminoff, xmaxoff, 100)
				poff = norm.pdf(xoff, muoff, stdoff)
				plt.plot(xoff, poff, color=coff_, linewidth=2)

				plt.hist(tmp_ontrx, bins=nbins, alpha=0.6, color=con_)
				muon, stdon = norm.fit(tmp_ontrx)
				xminon, xmaxon = plt.xlim()
				xon = np.linspace(xminon, xmaxon, 100)
				pon = norm.pdf(xon, muon, stdon)
				plt.plot(xon, pon, color=con_, linewidth=2)

				# plt.xlim([-0.02,0.02])
				plt.title('Displacements x')
				xmin, xmax = plt.xlim()
				ymin, ymax = plt.ylim()
				plt.text(xmin-(0.20*xmin), ymax-(0.10*ymax), 'PMC OFF',  color=coff_, fontweight='bold', fontsize=11)
				plt.text(xmin-(0.20*xmin), ymax-(0.20*ymax), 'Mean = '+str(truncate(muoff,4)),  color=coff_, fontweight='bold', fontsize=11)
				plt.text(xmin-(0.20*xmin), ymax-(0.30*ymax), 'Std = '+str(truncate(stdoff,4)),  color=coff_, fontweight='bold', fontsize=11)

				plt.text(xmax-(0.70*xmax), ymax-(0.10*ymax), 'PMC ON',  color=con_, fontweight='bold', fontsize=11)
				plt.text(xmax-(0.70*xmax), ymax-(0.20*ymax), 'Mean = '+str(truncate(muon,4)),  color=con_, fontweight='bold', fontsize=11)
				plt.text(xmax-(0.70*xmax), ymax-(0.30*ymax), 'Std = '+str(truncate(stdon,4)),  color=con_, fontweight='bold', fontsize=11)
				plt.grid()


				# 
				plt.subplot(232)
				
				plt.hist(tmp_offtry, bins=nbins, alpha=0.6, color=coff_)
				muoff, stdoff = norm.fit(tmp_offtry)
				xminoff, xmaxoff = plt.xlim()
				xoff = np.linspace(xminoff, xmaxoff, 100)
				poff = norm.pdf(xoff, muoff, stdoff)
				plt.plot(xoff, poff, color=coff_, linewidth=2)

				plt.hist(tmp_ontry, bins=nbins, alpha=0.6, color=con_)
				muon, stdon = norm.fit(tmp_ontry)
				xminon, xmaxon = plt.xlim()
				xon = np.linspace(xminon, xmaxon, 100)
				pon = norm.pdf(xon, muon, stdon)
				plt.plot(xon, pon, color=con_, linewidth=2)

				# plt.xlim([-0.02,0.02])
				plt.title('Displacements y')
				xmin, xmax = plt.xlim()
				ymin, ymax = plt.ylim()
				plt.text(xmin-(0.20*xmin), ymax-(0.10*ymax), 'PMC OFF',  color=coff_, fontweight='bold', fontsize=11)
				plt.text(xmin-(0.20*xmin), ymax-(0.20*ymax), 'Mean = '+str(truncate(muoff,4)),  color=coff_, fontweight='bold', fontsize=11)
				plt.text(xmin-(0.20*xmin), ymax-(0.30*ymax), 'Std = '+str(truncate(stdoff,4)),  color=coff_, fontweight='bold', fontsize=11)

				plt.text(xmax-(0.70*xmax), ymax-(0.10*ymax), 'PMC ON',  color=con_, fontweight='bold', fontsize=11)
				plt.text(xmax-(0.70*xmax), ymax-(0.20*ymax), 'Mean = '+str(truncate(muon,4)),  color=con_, fontweight='bold', fontsize=11)
				plt.text(xmax-(0.70*xmax), ymax-(0.30*ymax), 'Std = '+str(truncate(stdon,4)),  color=con_, fontweight='bold', fontsize=11)
				plt.grid()

				# 
				plt.subplot(233)
				
				plt.hist(tmp_offtrz, bins=nbins, alpha=0.6, color=coff_)
				muoff, stdoff = norm.fit(tmp_offtrz)
				xminoff, xmaxoff = plt.xlim()
				xoff = np.linspace(xminoff, xmaxoff, 100)
				poff = norm.pdf(xoff, muoff, stdoff)
				plt.plot(xoff, poff, color=coff_, linewidth=2)

				plt.hist(tmp_ontrz, bins=nbins, alpha=0.6, color=con_)
				muon, stdon = norm.fit(tmp_ontrz)
				xminon, xmaxon = plt.xlim()
				xon = np.linspace(xminon, xmaxon, 100)
				pon = norm.pdf(xon, muon, stdon)
				plt.plot(xon, pon, color=con_, linewidth=2)

				# plt.xlim([-0.02,0.02])
				plt.title('Displacements z')
				xmin, xmax = plt.xlim()
				ymin, ymax = plt.ylim()
				plt.text(xmin-(0.20*xmin), ymax-(0.10*ymax), 'PMC OFF',  color=coff_, fontweight='bold', fontsize=11)
				plt.text(xmin-(0.20*xmin), ymax-(0.20*ymax), 'Mean = '+str(truncate(muoff,4)),  color=coff_, fontweight='bold', fontsize=11)
				plt.text(xmin-(0.20*xmin), ymax-(0.30*ymax), 'Std = '+str(truncate(stdoff,4)),  color=coff_, fontweight='bold', fontsize=11)

				plt.text(xmax-(0.70*xmax), ymax-(0.10*ymax), 'PMC ON',  color=con_, fontweight='bold', fontsize=11)
				plt.text(xmax-(0.70*xmax), ymax-(0.20*ymax), 'Mean = '+str(truncate(muon,4)),  color=con_, fontweight='bold', fontsize=11)
				plt.text(xmax-(0.70*xmax), ymax-(0.30*ymax), 'Std = '+str(truncate(stdon,4)),  color=con_, fontweight='bold', fontsize=11)
				plt.grid()

				# 
				plt.subplot(234)
				
				plt.hist(tmp_offrx, bins=nbins, alpha=0.6, color=coff_)
				muoff, stdoff = norm.fit(tmp_offrx)
				xminoff, xmaxoff = plt.xlim()
				xoff = np.linspace(xminoff, xmaxoff, 100)
				poff = norm.pdf(xoff, muoff, stdoff)
				plt.plot(xoff, poff, color=coff_, linewidth=2)

				plt.hist(tmp_onrx, bins=nbins, alpha=0.6, color=con_)
				muon, stdon = norm.fit(tmp_onrx)
				xminon, xmaxon = plt.xlim()
				xon = np.linspace(xminon, xmaxon, 100)
				pon = norm.pdf(xon, muon, stdon)
				plt.plot(xon, pon, color=con_, linewidth=2)

				# plt.xlim([-0.02,0.02])
				plt.title('Rotations x')
				xmin, xmax = plt.xlim()
				ymin, ymax = plt.ylim()
				plt.text(xmin-(0.20*xmin), ymax-(0.10*ymax), 'PMC OFF',  color=coff_, fontweight='bold', fontsize=11)
				plt.text(xmin-(0.20*xmin), ymax-(0.20*ymax), 'Mean = '+str(truncate(muoff,4)),  color=coff_, fontweight='bold', fontsize=11)
				plt.text(xmin-(0.20*xmin), ymax-(0.30*ymax), 'Std = '+str(truncate(stdoff,4)),  color=coff_, fontweight='bold', fontsize=11)

				plt.text(xmax-(0.70*xmax), ymax-(0.10*ymax), 'PMC ON',  color=con_, fontweight='bold', fontsize=11)
				plt.text(xmax-(0.70*xmax), ymax-(0.20*ymax), 'Mean = '+str(truncate(muon,4)),  color=con_, fontweight='bold', fontsize=11)
				plt.text(xmax-(0.70*xmax), ymax-(0.30*ymax), 'Std = '+str(truncate(stdon,4)),  color=con_, fontweight='bold', fontsize=11)
				
				plt.grid()


				# 
				plt.subplot(235)
				
				plt.hist(tmp_offry, bins=nbins, alpha=0.6, color=coff_)
				muoff, stdoff = norm.fit(tmp_offry)
				xminoff, xmaxoff = plt.xlim()
				xoff = np.linspace(xminoff, xmaxoff, 100)
				poff = norm.pdf(xoff, muoff, stdoff)
				plt.plot(xoff, poff, color=coff_, linewidth=2)

				plt.hist(tmp_onry, bins=nbins, alpha=0.6, color=con_)
				muon, stdon = norm.fit(tmp_onry)
				xminon, xmaxon = plt.xlim()
				xon = np.linspace(xminon, xmaxon, 100)
				pon = norm.pdf(xon, muon, stdon)
				plt.plot(xon, pon, color=con_, linewidth=2)

				# plt.xlim([-0.02, 0.02])
				plt.title('Rotations y')
				xmin, xmax = plt.xlim()
				ymin, ymax = plt.ylim()
				plt.text(xmin-(0.20*xmin), ymax-(0.10*ymax), 'PMC OFF',  color=coff_, fontweight='bold', fontsize=11)
				plt.text(xmin-(0.20*xmin), ymax-(0.20*ymax), 'Mean = '+str(truncate(muoff,4)),  color=coff_, fontweight='bold', fontsize=11)
				plt.text(xmin-(0.20*xmin), ymax-(0.30*ymax), 'Std = '+str(truncate(stdoff,4)),  color=coff_, fontweight='bold', fontsize=11)

				plt.text(xmax-(0.70*xmax), ymax-(0.10*ymax), 'PMC ON',  color=con_, fontweight='bold', fontsize=11)
				plt.text(xmax-(0.70*xmax), ymax-(0.20*ymax), 'Mean = '+str(truncate(muon,4)),  color=con_, fontweight='bold', fontsize=11)
				plt.text(xmax-(0.70*xmax), ymax-(0.30*ymax), 'Std = '+str(truncate(stdon,4)),  color=con_, fontweight='bold', fontsize=11)
				plt.grid()

				# 
				plt.subplot(236)
				
				plt.hist(tmp_offrz, bins=nbins, alpha=0.6, color=coff_)
				muoff, stdoff = norm.fit(tmp_offrz)
				xminoff, xmaxoff = plt.xlim()
				xoff = np.linspace(xminoff, xmaxoff, 100)
				poff = norm.pdf(xoff, muoff, stdoff)
				plt.plot(xoff, poff, color=coff_, linewidth=2)

				plt.hist(tmp_onrz, bins=nbins, alpha=0.6, color=con_)
				muon, stdon = norm.fit(tmp_onrz)
				xminon, xmaxon = plt.xlim()
				xon = np.linspace(xminon, xmaxon, 100)
				pon = norm.pdf(xon, muon, stdon)
				plt.plot(xon, pon, color=con_, linewidth=2)

				# plt.xlim([-0.02, 0.02])
				plt.title('Rotations z')
				xmin, xmax = plt.xlim()
				ymin, ymax = plt.ylim()
				plt.text(xmin-(0.20*xmin), ymax-(0.10*ymax), 'PMC OFF',  color=coff_, fontweight='bold', fontsize=11)
				plt.text(xmin-(0.20*xmin), ymax-(0.20*ymax), 'Mean = '+str(truncate(muoff,4)),  color=coff_, fontweight='bold', fontsize=11)
				plt.text(xmin-(0.20*xmin), ymax-(0.30*ymax), 'Std = '+str(truncate(stdoff,4)),  color=coff_, fontweight='bold', fontsize=11)

				plt.text(xmax-(0.70*xmax), ymax-(0.10*ymax), 'PMC ON',  color=con_, fontweight='bold', fontsize=11)
				plt.text(xmax-(0.70*xmax), ymax-(0.20*ymax), 'Mean = '+str(truncate(muon,4)),  color=con_, fontweight='bold', fontsize=11)
				plt.text(xmax-(0.70*xmax), ymax-(0.30*ymax), 'Std = '+str(truncate(stdon,4)),  color=con_, fontweight='bold', fontsize=11)
				
				plt.grid()
				# ------------------------------------------
				plt.suptitle(str(contrasts[0])+' '+str(file[0:4]))
				plt.tight_layout()
				# plt.show()
				# figure = plt.gcf()
				# plt.savefig(str(list_sub[ssub])+"_T2025.pdf")#, dpi=150)
	
				# plt.savefig(str(contrasts[0])+"_"+str(file[0:4])+".png") 

