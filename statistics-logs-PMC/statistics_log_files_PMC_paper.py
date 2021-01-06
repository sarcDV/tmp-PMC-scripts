## statistics for PMC paper
#from plotly.graph_objs import Bar, Scatter, Layout, Figure
import os
import glob
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt 
import scipy.stats as stats

#import plotly
#from plotly import tools

empty_off_ = []
empty_on_ = []

empty_rot_off_ = []
empty_rot_on_ = []

test_tr_off_ = []
test_tr_on_ = []
# ----------------------------------------------
# PD, T1, T2, T2s05, T2s035, T2s025
# ----------------------------------------------
for file in list(glob.glob('./Off/zz_tmp_mat/T2s05/*.mat')): 
	file_off = file
	file_on = file.replace('Off', 'On')
	file_on = file_on.replace('_OFF', '_ON')
	# print('\n')
	# print(file_off, file_on)                                      
	
	temporal_data_OFF = sio.loadmat(file_off)
	temporal_data_ON = sio.loadmat(file_on)

	y_off_tr_x =np.squeeze(np.array(temporal_data_OFF['logdata_OFF_realtimex']))
	y_off_tr_y =np.squeeze(np.array(temporal_data_OFF['logdata_OFF_realtimey']))
	y_off_tr_z =np.squeeze(np.array(temporal_data_OFF['logdata_OFF_realtimez']))
	
	y_on_tr_x =np.squeeze(np.array(temporal_data_ON['logdata_ON_realtimex']))
	y_on_tr_y =np.squeeze(np.array(temporal_data_ON['logdata_ON_realtimey']))
	y_on_tr_z =np.squeeze(np.array(temporal_data_ON['logdata_ON_realtimez']))

	y_off_rot_x =np.squeeze(np.array(temporal_data_OFF['logdata_OFF_realtimeRx']))
	y_off_rot_y =np.squeeze(np.array(temporal_data_OFF['logdata_OFF_realtimeRy']))
	y_off_rot_z =np.squeeze(np.array(temporal_data_OFF['logdata_OFF_realtimeRz']))
	
	y_on_rot_x =np.squeeze(np.array(temporal_data_ON['logdata_ON_realtimeRx']))
	y_on_rot_y =np.squeeze(np.array(temporal_data_ON['logdata_ON_realtimeRy']))
	y_on_rot_z =np.squeeze(np.array(temporal_data_ON['logdata_ON_realtimeRz']))
	
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


	# --------------------------------------------------------------------------- #
	# print('Translation x PMC OFF', y_off_tr_x.shape, 'Translation x PMC ON', y_on_tr_x.shape)
	# print('Translation y PMC OFF', y_off_tr_y.shape, 'Translation y PMC ON', y_on_tr_y.shape)
	# print('Translation z PMC OFF', y_off_tr_z.shape, 'Translation z PMC ON', y_on_tr_z.shape)

	# print('Rotation x PMC OFF',y_off_rot_x.shape,'Rotation x PMC ON', y_on_rot_x.shape)
	# print('Rotation y PMC OFF',y_off_rot_y.shape,'Rotation y PMC ON', y_on_rot_y.shape)
	# print('Rotation z PMC OFF',y_off_rot_z.shape,'Rotation z PMC ON', y_on_rot_z.shape)


	total_rot_off = np.sqrt(y_off_rot_x**2+y_off_rot_y**2+y_off_rot_z**2)
	total_rot_on = np.sqrt(y_on_rot_x**2+y_on_rot_y**2+y_on_rot_z**2)

	total_tr_off = np.sqrt(y_off_tr_x**2+y_off_tr_y**2+y_off_tr_z**2)
	total_tr_on = np.sqrt(y_on_tr_x**2+y_on_tr_y**2+y_on_tr_z**2)

	test_tr_off_.append(total_tr_off)
	test_tr_on_.append(total_tr_on)
	#plt.figure()
	#plt.subplot(121)
	#plt.plot(total_tr_off)
	#plt.plot(total_tr_on)
	#plt.subplot(122)
	#plt.plot(total_rot_off)
	#plt.plot(total_rot_on)
	#plt.show()
	# -------------------------------------------------------------------------
	# stats.mannwhitneyu(x, y, use_continuity=True, alternative=None)
	# Compute the Mann-Whitney rank test on samples x and y.
	# Use only when the number of observation in each sample is > 20 and you have
	# 2 independent samples of ranks. Mann-Whitney U is significant if the u-obtained
	# is LESS THAN or equal to the critical value of U.
	# This test corrects for ties and by default uses a continuity correction.
	# 
	# -------------------------------------------------------------------------
	# scipy.stats.ttest_ind(a, b, axis=0, equal_var=True, nan_policy='propagate')
	# Calculate the T-test for the means of two independent samples of scores.
	# This is a two-sided test for the null hypothesis that 2 independent samples
	# have identical average (expected) values. This test assumes that the populations 
	# have identical variances by default.
	# We can use this test, if we observe two independent samples from the same or different population,
	# e.g. exam scores of boys and girls or of two ethnic groups. The test measures whether 
	# the average (expected) value differs significantly across samples. 
	# If we observe a large p-value, for example larger than 0.05 or 0.1, then we cannot 
	# reject the null hypothesis of identical average scores. If the p-value is smaller 
	# than the threshold, e.g. 1%, 5% or 10%, then we reject the null hypothesis of equal averages.
	# -------------------------------------------------------------------------
	# scipy.stats.ttest_rel(a, b, axis=0, nan_policy='propagate')
	# Calculate the t-test on TWO RELATED samples of scores, a and b.
	# This is a two-sided test for the null hypothesis that 2 related or repeated samples
	# have identical average (expected) values.

	# print('Total translation PMC OFF: mean, stand. dev., skewness, ', np.mean(total_tr_off), np.std(total_tr_off), stats.skew(total_tr_off))
	# print('Total translation PMC ON: mean, stand. dev., skewness, ', np.mean(total_tr_on), np.std(total_tr_on), stats.skew(total_tr_on))

	# print('Total rotation PMC OFF: mean, stand. dev., skewness, ', np.mean(total_rot_off), np.std(total_rot_off), stats.skew(total_rot_off))
	# print('Total rotation PMC ON: mean, stand. dev., skewness, ', np.mean(total_rot_on), np.std(total_rot_on), stats.skew(total_rot_on))

	empty_off_.append(np.mean(total_tr_off))
	empty_on_.append(np.mean(total_tr_on))

	empty_rot_off_.append(np.mean(total_rot_off))
	empty_rot_on_.append(np.mean(total_rot_on))

	# print('\n')

empty_off_= np.asarray(empty_off_)
empty_on_= np.asarray(empty_on_)

empty_rot_off_= np.asarray(empty_rot_off_)
empty_rot_on_= np.asarray(empty_rot_on_)

print(empty_on_.shape)
print('Mean translation OFF', np.mean(empty_off_), 'Mean translation ON',np.mean(empty_on_))
print('Std translation OFF',np.std(empty_off_), 'Std translation ON',np.std(empty_on_))

print('Mean rotation OFF',np.mean(empty_rot_off_), 'Mean rotation ON', np.mean(empty_rot_on_))
print('Std rotation OFF',np.std(empty_rot_off_), 'Std rotation ON', np.std(empty_rot_on_))

print(stats.mannwhitneyu(empty_off_, empty_on_))#, use_continuity=True, alternative='greater'))
print(stats.ttest_ind(empty_off_, empty_on_))

#test_tr_off_ = np.asarray(test_tr_off_)
#test_tr_off_ = np.reshape(test_tr_off_, (test_tr_off_.shape[0]*test_tr_off_.shape[1]))
#print(test_tr_off_.shape, np.mean(test_tr_off_[:]))#, stats.skew(test_tr_off_))
	
#test_tr_on_ = np.asarray(test_tr_on_)
#test_tr_on_ = np.reshape(test_tr_on_, (test_tr_on_.shape[0]*test_tr_on_.shape[1]))
#print(test_tr_on_.shape)#, np.mean(test_tr_on_), stats.skew(test_tr_on_))

# print(stats.mannwhitneyu(test_tr_off_/test_tr_off_.max(), test_tr_on_/test_tr_on_.max(), use_continuity=True, alternative=None))
# ---------------------------------------------------------------- #	
#(statistic=1139093738835.0, pvalue=0.0)
#(statistic=1407336259482.0, pvalue=6.278526737597417e-284)
# ---------------------------------------------------------------- #	
"""
### reading data ####
./On/zz_tmp_mat/T1/vq83_T1_ON.mat
{'__header__': b'MATLAB 5.0 MAT-file, Platform: MACI64, Created on: Fri Oct 14 17:29:51 2016', 
'__version__': '1.0', '__globals__': [], 
'stat_motion': array([[10.41825162, -0.09759215,  1.40612446,  0.89638853, -0.07613847,
        24.06594934, -0.29821027,  0.49515867]]), 
'log_file': array(['vq83_T1_ON.log'], dtype='<U14'), 
'logData_ON': array([[(array([[1724043, 1724176, 1724177, ..., 1782014, 1782015, 1782015]],
      dtype=int32), array([[2.84649282e+09, 2.84817296e+09, 2.84818558e+09, ...,
        3.57885621e+09, 3.57886884e+09, 3.57886884e+09]]), array([[0.49647, 0.50602, 0.50506, ..., 0.30487, 0.28699, 0.28699]]), array([[-0.82022, -0.84613, -0.83743, ...,  2.4657 ,  2.4626 ,  2.4626 ]]), array([[-0.058928, -0.10963 , -0.10668 , ..., -4.0002  , -4.0197  ,
        -4.0197  ]]), array([[ 0.00419294,  0.00441969,  0.00443584, ..., -0.00919073,
        -0.00911946, -0.00911946]]), array([[-0.00243901, -0.00238143, -0.00244817, ..., -0.00068546,
        -0.00071788, -0.00071788]]), array([[-0.00451786, -0.00459341, -0.00459966, ..., -0.0020706 ,
        -0.00213407, -0.00213407]]), array([[0.99997795, 0.99997675, 0.99997652, ..., 0.99995536, 0.99995577,
        0.99995577]]), array([[ 0.47921385,  0.50520652,  0.50702022, ..., -1.05335623,
        -1.04520205, -1.04520205]]), array([[-0.28165551, -0.27521308, -0.28287179, ..., -0.07636341,
        -0.08002969, -0.08002969]]), array([[-0.51653889, -0.52516141, -0.52583897, ..., -0.23798534,
        -0.24528732, -0.24528732]]), array([], shape=(0, 0), dtype=uint8), array([], shape=(0, 0), dtype=uint8), array([[0, 0, 0, ..., 0, 0, 0]], dtype=uint8), array([[  5.1074, 150.43  , 107.83  ]]), array([[ 0.51587862, -0.47745332,  0.52106823,  0.48414397]]), array([[1, 1, 1, ..., 1, 1, 1]], dtype=uint8))]],
      dtype=[('frame', 'O'), ('ts', 'O'), ('x', 'O'), ('y', 'O'), ('z', 'O'), ('qx', 'O'), ('qy', 'O'), ('qz', 'O'), ('qr', 'O'), ('Rx', 'O'), ('Ry', 'O'), ('Rz', 'O'), ('deltaMM', 'O'), ('deltaDegs', 'O'), ('rejected', 'O'), ('initialPose_v', 'O'), ('initialPose_q', 'O'), ('off', 'O')]), 'FR': array([[80]], dtype=uint8), 'n_start': array([[3]], dtype=uint8), 'logdata_ON_realtime': array([[0.000000e+00, 1.250000e-02, 1.250000e-02, ..., 7.229625e+02,
        7.229750e+02, 7.229750e+02]]), 
      'logdata_ON_realtimex': array([[ 0.     , -0.00061, -0.00061, ..., -0.20019, -0.21807, -0.21807]]), 
      'logdata_ON_realtimey': array([[ 0.     , -0.01445, -0.01445, ...,  3.30313,  3.30003,  3.30003]]), 
      'logdata_ON_realtimez': array([[ 0.00000e+00,  1.90000e-03,  1.90000e-03, ..., -3.89352e+00,
        -3.91302e+00, -3.91302e+00]]), 
      'logdata_ON_realtimeRx': array([[ 0.00000000e+00,  5.39170743e-04,  5.39170743e-04, ...,
        -1.56037644e+00, -1.55222227e+00, -1.55222227e+00]]), 
      'logdata_ON_realtimeRy': array([[0.        , 0.00338377, 0.00338377, ..., 0.20650838, 0.2028421 ,
        0.2028421 ]]), 
      'logdata_ON_realtimeRz': array([[0.        , 0.00101382, 0.00101382, ..., 0.28785363, 0.28055165,
        0.28055165]]), 
      'test_rot': array([[0.        , 0.00053917, 0.00053917, ..., 0.28785363, 0.28055165,
        0.28055165]]), 
      'test_tran': array([[ 0.00000e+00, -6.10000e-04, -6.10000e-04, ..., -3.89352e+00,
        -3.91302e+00, -3.91302e+00]]), 
      'max_rot_y': array([[3.63866915]]), 
      'min_rot_y': array([[-8.79322971]]), 
      'max_tran_y': array([[19.32543]]), 
      'min_tran_y': array([[-18.78932]]), 
      'max_ON': array([[ 1.96594   , 18.82543   ,  3.04868   ,  3.13866915,  1.15029539,
         1.23964497]]), 
      'min_ON': array([[ -2.71466   ,  -3.11007   , -18.28932   ,  -8.29322971,
         -0.51622611,  -0.2173444 ]]), 
      'total_motionON': array([[ 4.6806    , 21.9355    , 21.338     , 11.43189886,  1.6665215 ,
         1.45698937]]), 
      'mean_ON': array([[-0.10721632,  0.68105553, -0.81624976, -0.38413965, -0.01932809,
         0.06032536]]), 
      'var_ON': array([[0.09092982, 3.78761008, 3.75123714, 0.75393246, 0.0247201 ,
        0.02831713]]), 
      'std_ON': array([[0.30154572, 1.94617833, 1.93681108, 0.86829284, 0.15722626,
        0.16827694]]), 
      'med_ON': array([[-0.10101   ,  0.21246   , -0.30313   , -0.19003874, -0.06836586,
        -0.0067462 ]]), 
      'kurt_ON': array([[17.84639013, 40.27020706, 33.41923717, 36.24234433,  9.00702164,
         7.61049574]]), 
      'skew_ON': array([[-1.34901502,  5.36674373, -4.57508853, -4.72486997,  1.83979492,
         1.65317325]]), 
      'y_ON': array([[10.41825162],
       [-0.09759215],
       [ 1.40612446],
       [ 0.89638853],
       [-0.07613847],
       [24.06594934],
       [-0.29821027]]), 
      'AX1': array([[482.06201172]]), 
      'ON_tqi': array([[0.49515867]]), 
      'pathstr': array([], dtype='<U1'), 'name': array(['vq83_T1_ON'], dtype='<U10'), 
      'ext': array(['.log'], dtype='<U4')}
"""
