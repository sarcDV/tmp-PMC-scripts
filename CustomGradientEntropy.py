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

def GradEntr_eval(img):
	x = np.zeros((img.shape[2]))
	## 
	for kk in range(0, img.shape[2]):
		s = img[:,:,kk]
		# s = s/s.max()
		# s =(np.uint8(s*255))
		entr_ = gradient(s, disk(3))
		entr_ = entr_/entr_.max()
		out = entropy(entr_, disk(3))
		x[kk]= out.sum()/(out.shape[0]*out.shape[1])
		
	x = x[~np.isnan(x)]

	return x

# ---------------------------------------------
# ---------------------------------------------
# ---------------------------------------------

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
				## call grade function
				eval_ = GradEntr_eval(img)
				tmp_line_ = str(eval_)
				tmp_line_ = tmp_line_.replace('[','')
				tmp_line_ = tmp_line_.replace(']','')
				tmp_lineb_ = (str(file), tmp_line_)
				myfile.write("%s\n" % str(tmp_lineb_))

myfile.close()	
"""
sessions = ['ON']#['OFF', 'ON']
contrasts = ['T2star-025']#['PD', 'T1', 'T2', 'T2star-05', 'T2star-035', 'T2star-025']

filetxt = str(contrasts[0])+"_"+str(sessions[0])+"_GradientEntropy.txt"
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
				## call grade function
				eval_ = GradEntr_eval(img)
				tmp_line_ = str(eval_)
				tmp_line_ = tmp_line_.replace('[','')
				tmp_line_ = tmp_line_.replace(']','')
				myfile.write("%s\n" % tmp_line_)
				
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
				

myfile.close()	

a = nib.load('./OFF/PD/au70_PD_OFF.nii.gz').get_fdata()
a = a/a.max()
max_ = np.asarray(a.shape).max()
s = a[:,:,0]

entr_ = gradient(s, disk(3))
entr_ = entr_/entr_.max()
print(entr_.max(), entr_.min())
out = entropy(entr_, disk(3))


print(out.sum()/(out.shape[0]*out.shape[1]))

plt.figure()
plt.subplot(131)
plt.imshow(s)
plt.subplot(132)
plt.imshow(entr_)
plt.subplot(133)
plt.imshow(out)
plt.show()

"""