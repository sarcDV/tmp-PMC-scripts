import scipy.io as sio
import numpy as np

# load the data
#temporal_data = sio.loadmat('data_842-3.mat')
# temporal_data_OFF = sio.loadmat('./Off/zz_tmp_mat/PD/au70_PD_OFF.mat')
temporal_data_OFF = sio.loadmat('./Off/zz_tmp_mat/T2s025/vq83_T2s025_OFF.mat')
#('./Off/zz_tmp_mat/T1/au70_T1_OFF.mat')
# -------------------------------------------------------------------
# PMC OFF

random_x = np.array(temporal_data_OFF['logdata_OFF_realtime']) 
# Rotations
random_y0 = np.array(temporal_data_OFF['logdata_OFF_realtimeRx'])
random_y1 = np.array(temporal_data_OFF['logdata_OFF_realtimeRy'])
random_y2 = np.array(temporal_data_OFF['logdata_OFF_realtimeRz'])
# Translations
random_t0 = np.array(temporal_data_OFF['logdata_OFF_realtimex'])
random_t1 = np.array(temporal_data_OFF['logdata_OFF_realtimey'])
random_t2 = np.array(temporal_data_OFF['logdata_OFF_realtimez'])
# resize for the plots
random_x.resize(random_x.size)
random_y0.resize(random_y0.size)
random_y1.resize(random_y1.size)
random_y2.resize(random_y2.size)
random_t0.resize(random_t0.size)
random_t1.resize(random_t1.size)
random_t2.resize(random_t2.size)
# -------------------------------------------------------------------

# PMC ON
#temporal_data_ON = sio.loadmat('./On/zz_tmp_mat/PD/au70_PD_ON.mat')
temporal_data_ON = sio.loadmat('./On/zz_tmp_mat/T2s025/vq83_T2s025_ON.mat')
#('./On/zz_tmp_mat/T1/au70_T1_ON.mat')
time_x = np.array(temporal_data_ON['logdata_ON_realtime']) 
# Rotations
rot_y0 = np.array(temporal_data_ON['logdata_ON_realtimeRx'])
rot_y1 = np.array(temporal_data_ON['logdata_ON_realtimeRy'])
rot_y2 = np.array(temporal_data_ON['logdata_ON_realtimeRz'])
# Translations
tra_t0 = np.array(temporal_data_ON['logdata_ON_realtimex'])
tra_t1 = np.array(temporal_data_ON['logdata_ON_realtimey'])
tra_t2 = np.array(temporal_data_ON['logdata_ON_realtimez'])
# resize for the plots
time_x.resize(time_x.size)
rot_y0.resize(rot_y0.size)
rot_y1.resize(rot_y1.size)
rot_y2.resize(rot_y2.size)
tra_t0.resize(tra_t0.size)
tra_t1.resize(tra_t1.size)
tra_t2.resize(tra_t2.size)

import matplotlib.pyplot as plt 


plt.figure(1) #figsize = (6,9))
# plt.suptitle('Motion Patterns - Subject 1 - PD scan', fontsize=22, fontweight='bold')
plt.subplot(211)
plt.plot(random_x, random_y0, color='#0b850b',label='Pitch')
plt.plot(random_x, random_y1, color='#1515ff',label='Yaw')
plt.plot(random_x, random_y2, color='#ffa500',label='Roll')
plt.grid(color='white', linewidth=0.82)
# plt.xlim([0,315])
# plt.ylim([-0.3,0.2])
# plt.xticks( fontsize=13, fontweight='bold')
# plt.yticks( fontsize=13, fontweight='bold')
# plt.xlabel('Time (seconds)')
# plt.ylabel('Rotations (degrees)', fontsize=16, fontweight='bold')
ax = plt.gca()
ax.set_facecolor('#E5ECF6')

plt.legend(prop={'size': 13, 'weight':'bold'})
# plt.title('OMTS Off', fontsize=18, fontweight='bold')
# --------------------------------------------------------
plt.subplot(212)
plt.plot(time_x, rot_y0, color='#0b850b',label='Pitch')
plt.plot(time_x, rot_y1, color='#1515ff',label='Yaw')
plt.plot(time_x, rot_y2, color='#ffa500',label='Roll')
plt.grid(color='white', linewidth=0.82)
# plt.xlim([0,315])
# plt.xlabel('Time (seconds)', fontsize=16, fontweight='bold')
# plt.ylim([-0.3,0.2])
# plt.ylabel('Rotations (degrees)', fontsize=16, fontweight='bold')
# plt.xticks( fontsize=13, fontweight='bold')
# plt.yticks( fontsize=13, fontweight='bold')
ax = plt.gca()
ax.set_facecolor('#E5ECF6')
# plt.legend(prop={'size': 13, 'weight':'bold'})
# plt.title('OMTS On', fontsize=18, fontweight='bold')
#plt.show()
# --------------------------------------------------------
# --------------------------------------------------------

plt.figure(2)
plt.subplot(211)
plt.plot(random_x, random_t0, color='#ff3434',label='Displ-x')
plt.plot(random_x, random_t1, color='#0cab2e',label='Displ-y')
plt.plot(random_x, random_t2, color='#000066',label='Displ-z')
plt.grid(color='white', linewidth=0.82)
#plt.xlim([0,315])
#plt.ylim([-0.6,0.8])
#plt.xticks( fontsize=13, fontweight='bold')
#plt.yticks( fontsize=13, fontweight='bold')
#plt.xlabel('Time (seconds)', fontsize=16, fontweight='bold')
#plt.ylabel('Displacements (mm)', fontsize=16, fontweight='bold')
ax = plt.gca()
ax.set_facecolor('#E5ECF6')
#plt.legend(prop={'size': 13, 'weight':'bold'})
#plt.title('OMTS Off', fontsize=18, fontweight='bold')

# --------------------------------------------------------
plt.subplot(212)
plt.plot(time_x, tra_t0, color='#ff3434',label='Displ-x')
plt.plot(time_x, tra_t1, color='#0cab2e',label='Displ-y')
plt.plot(time_x, tra_t2, color='#000066',label='Displ-z')
plt.grid(color='white', linewidth=0.82)
#plt.xticks( fontsize=13, fontweight='bold')
#plt.yticks( fontsize=13, fontweight='bold')
#plt.xlim([0,315])
#plt.xlabel('Time (seconds)', fontsize=16, fontweight='bold')
#plt.ylim([-0.6,0.8])
#plt.ylabel('Displacements (mm)', fontsize=16, fontweight='bold')
ax = plt.gca()
ax.set_facecolor('#E5ECF6')
#plt.legend(prop={'size': 13, 'weight':'bold'})
#plt.title('OMTS On', fontsize=18, fontweight='bold')
plt.show()