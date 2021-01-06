import glob, os
import math
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt 
import cv2
from scipy.ndimage import gaussian_filter
from skimage import feature
import scipy.ndimage as ndi
import scipy

def AES(img):
	""" 
	AES(k)=sqrt{sum[E(Iij)(Gx^2(Iij)+Gy^2(Iij))]}/sum[E(Iij)]
		
		E(Iij)--> binary mask of the edges extracted using Canny edge detector 
		Gx -----> [-1 -1 -1; 0 0 0; 1 1 1]
		Gy -----> [-1 0 1; -1 0 1; -1 0 1]
		G x , and G y represent the centered gradient kernels along x and y, respectively
		The mean value across all the slices was considered for the analysis. When blurring increases,
		AES values decrease.
	"""
	Gx = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
	Gy = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
	## 
	x = np.zeros((img.shape[2]))
	## 
	for kk in range(0, img.shape[2]):
		s = img[:,:,kk]
		E = feature.canny(s, sigma=0.6)
		E = E.astype(np.float64)
		filtx = cv2.filter2D(s, -1, Gx)
		filty = cv2.filter2D(s, -1, Gy)
		aes_ = np.sqrt((E*((filtx**2)+(filty**2))).sum()) / E.sum()
		x[kk]=aes_
		
	x = x[~np.isnan(x)]

	return x
# -------------------------------
# -------------------------------

sessions = ['OFF', 'ON']
contrasts = ['PD', 'T1', 'T2', 'T2star-05', 'T2star-035', 'T2star-025']


for sess in sessions:
	for cont_ in contrasts:
		filetxt = str(cont_)+"_"+str(sess)+".txt"
		myfile = open(filetxt, 'w')
		for file in sorted(os.listdir("./"+str(sess)+"/"+str(cont_)+"/")):
			if file.endswith(".nii.gz"):
				filepath_ = os.path.join("./"+str(sess)+"/"+str(cont_)+"/", file)
				img = nib.load(str(filepath_)).get_fdata()
				print('... processing ... ', file, img.shape)
				## normalize:
				img = img/img.max()
				## call AES function
				eval_ = AES(img)
				tmp_line_ = str(eval_)
				tmp_line_ = tmp_line_.replace('[','')
				tmp_line_ = tmp_line_.replace(']','')
				tmp_lineb_ = (str(file), tmp_line_)
				myfile.write("%s\n" % str(tmp_lineb_))
				# --------------------------------------
#    myfile.close()
##########################################################
##########################################################
##########################################################

"""
sessions = ['OFF']#['OFF', 'ON']
contrasts = ['T2star-025']#['PD', 'T1', 'T2', 'T2star-05', 'T2star-035', 'T2star-025']

filetxt = str(contrasts[0])+"_"+str(sessions[0])+".txt"
myfile = open(filetxt, 'w')
for cont_ in contrasts:
	for sess in sessions:
		for file in sorted(os.listdir("./"+str(sess)+"/"+str(cont_)+"/")):
			if file.endswith(".nii.gz"):
				filepath_ = os.path.join("./"+str(sess)+"/"+str(cont_)+"/", file)
				img = nib.load(str(filepath_)).get_fdata()
				print('... processing ... ', file, img.shape)
				## normalize:
				img = img/img.max()
				## call AES function
				eval_ = AES(img)
				tmp_line_ = str(eval_)
				tmp_line_ = tmp_line_.replace('[','')
				tmp_line_ = tmp_line_.replace(']','')
				tmp_lineb_ = (str(file), tmp_line_)
				myfile.write("%s\n" % str(tmp_lineb_))
				# myfile.write("%s\n" % tmp_line_)
				
# myfile.close()
"""
