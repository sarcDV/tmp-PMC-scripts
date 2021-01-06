## average edge strength 
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
from skimage.filters.rank import entropy
from skimage.filters.rank import gradient
from skimage.morphology import disk
from skimage.feature import greycomatrix, greycoprops
from skimage.measure import shannon_entropy
#  skimage.measure.shannon_entropy(image, base=2)

def Haralick_eval(img):
	x = np.zeros((img.shape[2]))
	distances = [3, 5, 7, 9, 11, 21]
	angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
	## 
	for kk in range(0, img.shape[2]):
		s = img[:,:,kk]
		s = s/s.max()
		tmp_ =(np.uint8(s*255))
		glcm = greycomatrix(tmp_, 
                    distances=distances, 
                    angles=angles,
                    symmetric=False,
                    normed=True)

		tmp= glcm.sum(axis=2).sum(axis=2)
		x[kk]= shannon_entropy(tmp)
		
	x = x[~np.isnan(x)]

	return x

sessions = ['ON']#['OFF', 'ON']
contrasts = ['PD']#['PD', 'T1', 'T2', 'T2star-05', 'T2star-035', 'T2star-025']

filetxt = str(contrasts[0])+"_"+str(sessions[0])+"_haralick.txt"
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
				## call Haralick function
				eval_ = Haralick_eval(img)
				tmp_line_ = str(eval_)
				tmp_line_ = tmp_line_.replace('[','')
				tmp_line_ = tmp_line_.replace(']','')
				myfile.write("%s\n" % tmp_line_)
				"""
				tmp_line_ = str([file, str(np.mean(eval_)), str(np.std(eval_)), str(eval_)])
				tmp_line_ = tmp_line_.replace(".nii.gz", ' ')
				tmp_line_ = tmp_line_.replace('[','')
				tmp_line_ = tmp_line_.replace(']','')
				tmp_line_ = tmp_line_.replace("'",'')
				tmp_line_ = tmp_line_.replace("n",' ')
				tmp_line_ =  tmp_line_.replace("\ ",' ')
				tmp_line_ = tmp_line_.replace(",", ' ')
				# print(tmp_line_)
				myfile.write("%s\n" % tmp_line_)
				"""

myfile.close()
