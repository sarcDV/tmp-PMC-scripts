import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import seaborn as sns
import nibabel as nib
import cv2
from scipy.ndimage import gaussian_filter
from skimage.filters.rank import entropy
from skimage.filters.rank import gradient
from skimage.morphology import disk
from skimage import feature
import scipy.ndimage as ndi
import scipy
from skimage.feature import greycomatrix, greycoprops
from skimage.measure import shannon_entropy
import warnings
warnings.filterwarnings("ignore")

def varpercent(valstart_, valend_):
	return ((valend_ -valstart_)/valstart_)*100


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
# --------------------------------------------------------------------------------------------
#  load nii files
"""
class_1_ = nib.load('./ref_vols_for_metrics_testing/xi27_class-1_T1.nii.gz').get_fdata()
class_2_ = nib.load('./ref_vols_for_metrics_testing/xi27_class-2_T1.nii.gz').get_fdata()
class_3_ = nib.load('./ref_vols_for_metrics_testing/xi27_class-3_T1.nii.gz').get_fdata()

## normalize
class_1_ = class_1_/class_1_.max()
class_2_ = class_2_/class_2_.max()
class_3_ = class_3_/class_3_.max()
##  calculation metrics
aes_1_, aes_2_, aes_3_ = AES(class_1_), AES(class_2_), AES(class_3_)
cge_1_, cge_2_, cge_3_ = GradEntr_eval(class_1_), GradEntr_eval(class_2_), GradEntr_eval(class_3_)
hta_1_, hta_2_, hta_3_ = Haralick_eval(class_1_), Haralick_eval(class_2_), Haralick_eval(class_3_)

np.savetxt('AES_class_1_.txt', aes_1_, delimiter=' ')
np.savetxt('AES_class_2_.txt', aes_2_, delimiter=' ')
np.savetxt('AES_class_3_.txt', aes_3_, delimiter=' ')

np.savetxt('CGE_class_1_.txt', cge_1_, delimiter=' ')
np.savetxt('CGE_class_2_.txt', cge_2_, delimiter=' ')
np.savetxt('CGE_class_3_.txt', cge_3_, delimiter=' ')

np.savetxt('HTA_class_1_.txt', hta_1_, delimiter=' ')
np.savetxt('HTA_class_2_.txt', hta_2_, delimiter=' ')
np.savetxt('HTA_class_3_.txt', hta_3_, delimiter=' ')
"""
## --------------------------------------------------------------------------------------------

subj =['dl43','vq83','xi27']
n_ =2 

aes_1_ = 100*np.loadtxt('./ref_vols_for_metrics_testing/'+str(subj[n_])+'/AES_class_1_.txt')
aes_2_ = 100*np.loadtxt('./ref_vols_for_metrics_testing/'+str(subj[n_])+'/AES_class_2_.txt')
aes_3_ = 100*np.loadtxt('./ref_vols_for_metrics_testing/'+str(subj[n_])+'/AES_class_3_.txt')


cge_1_ = np.loadtxt('./ref_vols_for_metrics_testing/'+str(subj[n_])+'/CGE_class_1_.txt')
cge_2_ = np.loadtxt('./ref_vols_for_metrics_testing/'+str(subj[n_])+'/CGE_class_2_.txt')
cge_3_ = np.loadtxt('./ref_vols_for_metrics_testing/'+str(subj[n_])+'/CGE_class_3_.txt')

# labels = ['OMTS ON\nNo intentional\nMotion', 'Sample 2', 'Sample 3']
labels = ['1', '2', '3']
ind = np.arange(len(labels))
width = 0.5
plt.figure(figsize = (6,3))
# plt.subplot(331)
# plt.imshow(sample_n_, cmap='gray', vmin=0, vmax=0.25)
# plt.text(150,-10, 'Sample 1')
# plt.text(650,-10, 'Sample 2')
# plt.text(1150,-10, 'Sample 3')
# plt.axis('off')
fs_=10
plt.subplot(121)
plt.boxplot([aes_1_, aes_2_, aes_3_], notch=True, showfliers=False)
plt.xticks([1, 2, 3],labels, fontsize=fs_, fontweight='bold')
# plt.xticks([1, 2, 3],['','',''], fontsize=fs_, fontweight='bold')
plt.yticks(fontsize=fs_, fontweight='bold')
plt.ylim(0,0.7)
plt.grid(True)
# plt.title('Average Edge Strength', fontsize=fs_, fontweight='bold')
plt.subplot(122)
plt.boxplot([cge_1_, cge_2_, cge_3_], notch=True,showfliers=False)
plt.xticks([1, 2, 3],labels, fontsize=fs_, fontweight='bold')
#plt.xticks([1, 2, 3],['','',''], fontsize=fs_, fontweight='bold')
plt.ylim(0,2.5)
plt.yticks(fontsize=fs_, fontweight='bold')
plt.grid(True)
#plt.title('Gradient Entropy', fontsize=fs_, fontweight='bold')
plt.tight_layout()
plt.show()

"""
subj =['dl43','vq83','xi27']

class1_aes, class2_aes, class3_aes = [],[],[]
class1_cge, class2_cge, class3_cge = [],[],[]

for ii in subj:
	aes_1_ = 100*np.loadtxt('./ref_vols_for_metrics_testing/'+str(ii)+'/AES_class_1_.txt')
	aes_2_ = 100*np.loadtxt('./ref_vols_for_metrics_testing/'+str(ii)+'/AES_class_2_.txt')
	aes_3_ = 100*np.loadtxt('./ref_vols_for_metrics_testing/'+str(ii)+'/AES_class_3_.txt')

	class1_aes.append(aes_1_)
	class2_aes.append(aes_2_)
	class3_aes.append(aes_3_)

	cge_1_ = np.loadtxt('./ref_vols_for_metrics_testing/'+str(ii)+'/CGE_class_1_.txt')
	cge_2_ = np.loadtxt('./ref_vols_for_metrics_testing/'+str(ii)+'/CGE_class_2_.txt')
	cge_3_ = np.loadtxt('./ref_vols_for_metrics_testing/'+str(ii)+'/CGE_class_3_.txt')

	class1_cge.append(cge_1_)
	class2_cge.append(cge_2_)
	class3_cge.append(cge_3_)

class1_aes, class2_aes, class3_aes = np.asarray(class1_aes), np.asarray(class2_aes), np.asarray(class3_aes)
class1_cge, class2_cge, class3_cge = np.asarray(class1_cge), np.asarray(class2_cge), np.asarray(class3_cge)
"""

# plt.figure(1)
# ax1 = plt.subplot(211)
# ax1.boxplot(class1_aes, notch=True, showfliers=False, positions=[0.5,1.,1.5])
# ax1.boxplot(class2_aes, notch=True, showfliers=False, positions=[2,2.5,3])
# ax1.boxplot(class3_aes, notch=True, showfliers=False, positions=[3.5,4.0,4.5])
# ax1.set_xlim(0, 5)
# ax1.set_xticklabels([])
# ax1.set_xticks([])
# ax1.grid(True)

## ----

#ax2 = plt.subplot(212)
#ax2.boxplot(class1_cge[0,:], notch=True, showfliers=False, positions=[0.5,1.,1.5])
#ax2.boxplot(class2_cge[0,:], notch=True, showfliers=False, positions=[2,2.5,3])
#ax2.boxplot(class3_cge[0,:], notch=True, showfliers=False, positions=[3.5,4.0,4.5])
#ax2.set_xlim(0, 5)
#ax2.set_xticklabels([])
#ax2.set_xticks([])
#ax2.grid(True)
# plt.show()

#bplot_aes_ = [np.mean(aes_1_), np.mean(aes_2_), np.mean(aes_3_)]
#bplot_cge_ = [np.mean(cge_1_), np.mean(cge_2_), np.mean(cge_3_)]
#bplot_hta_ = [np.mean(hta_1_), np.mean(hta_2_), np.mean(hta_3_)]

#bplot_aessd_ = [np.std(aes_1_), np.std(aes_2_), np.std(aes_3_)]
#bplot_cgesd_ = [np.std(cge_1_), np.std(cge_2_), np.std(cge_3_)]
#bplot_htasd_ = [np.std(hta_1_), np.std(hta_2_), np.std(hta_3_)]




"""

"""

# plt.bar(ind, menMeans, width, yerr=menStd)
"""
bb_aes_ = [varpercent(np.mean(aes_1_), np.mean(np.mean(aes_1_))), 
		   varpercent(np.mean(aes_1_), np.mean(np.mean(aes_2_))), 
		   varpercent(np.mean(aes_1_), np.mean(np.mean(aes_3_)))]

bb_cge_ = [varpercent(np.mean(cge_1_), np.mean(np.mean(cge_1_))), 
		   varpercent(np.mean(cge_1_), np.mean(np.mean(cge_2_))), 
		   varpercent(np.mean(cge_1_), np.mean(np.mean(cge_3_)))]

bb_hta_ = [varpercent(np.mean(hta_1_), np.mean(np.mean(hta_1_))), 
		   varpercent(np.mean(hta_1_), np.mean(np.mean(hta_2_))), 
		   varpercent(np.mean(hta_1_), np.mean(np.mean(hta_3_)))]

plt.figure()
plt.subplot(231)
plt.boxplot([aes_1_, aes_2_, aes_3_], notch=True, showfliers=False)
# plt.xticks_label(labels)
plt.grid(True)
plt.title('Average Edge Strength')
# plt.bar(ind, bplot_aes_)#, width, yerr=bplot_aessd_)
plt.subplot(232)
plt.boxplot([cge_1_, cge_2_, cge_3_], notch=True,showfliers=False)
plt.grid(True)
plt.title('Gradient Entropy')
# plt.bar(ind, bplot_cge_)#, width, yerr=bplot_cgesd_)
plt.subplot(233)
plt.boxplot([hta_1_, hta_2_, hta_3_],notch=True, showfliers=False)
plt.grid(True)
plt.title('Haralick''s Entropy')
# plt.bar(ind, bplot_hta_)#, width, yerr=bplot_htasd_)

plt.subplot(234)
plt.bar(ind, bb_aes_)
plt.subplot(235)
plt.bar(ind, bb_cge_)
plt.subplot(236)
plt.bar(ind, bb_hta_)

#plt.tight_layout()
plt.show()
"""
# ------------------------
"""  calculation metrics
aes_1_, aes_2_, aes_3_ = AES(class_1_), AES(class_2_), AES(class_3_)
cge_1_, cge_2_, cge_3_ = GradEntr_eval(class_1_), GradEntr_eval(class_2_), GradEntr_eval(class_3_)
hta_1_, hta_2_, hta_3_ = Haralick_eval(class_1_), Haralick_eval(class_2_), Haralick_eval(class_3_)

np.savetxt('AES_class_1_.txt', aes_1_, delimiter=' ')
np.savetxt('AES_class_2_.txt', aes_2_, delimiter=' ')
np.savetxt('AES_class_3_.txt', aes_3_, delimiter=' ')

np.savetxt('CGE_class_1_.txt', cge_1_, delimiter=' ')
np.savetxt('CGE_class_2_.txt', cge_2_, delimiter=' ')
np.savetxt('CGE_class_3_.txt', cge_3_, delimiter=' ')

np.savetxt('HTA_class_1_.txt', hta_1_, delimiter=' ')
np.savetxt('HTA_class_2_.txt', hta_2_, delimiter=' ')
np.savetxt('HTA_class_3_.txt', hta_3_, delimiter=' ')

"""