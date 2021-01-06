# -*- coding: utf-8 -*-
import plotly
from plotly import subplots #tools
from plotly.graph_objs import Bar, Scatter, Layout, Figure
import scipy.io as sio
import numpy as np

import plotly
from plotly import tools
'''
     %% --------
   
max_OFF = [ max(logdata_OFF_realtimex(:)),max(logdata_OFF_realtimey(:)),max(logdata_OFF_realtimez(:)),...
           max(logdata_OFF_realtimeRx(:)),max(logdata_OFF_realtimeRy(:)),max(logdata_OFF_realtimeRz(:))];  

min_OFF = [ min(logdata_OFF_realtimex(:)),min(logdata_OFF_realtimey(:)),min(logdata_OFF_realtimez(:)),...
           min(logdata_OFF_realtimeRx(:)),min(logdata_OFF_realtimeRy(:)),min(logdata_OFF_realtimeRz(:)),];  
             
total_motionOFF =  max_OFF - min_OFF;  


mean_OFF = [ mean(logdata_OFF_realtimex(:)),mean(logdata_OFF_realtimey(:)),mean(logdata_OFF_realtimez(:)),...
           mean(logdata_OFF_realtimeRx(:)),mean(logdata_OFF_realtimeRy(:)),mean(logdata_OFF_realtimeRz(:))]
       
var_OFF = [ var(logdata_OFF_realtimex(:)),var(logdata_OFF_realtimey(:)),var(logdata_OFF_realtimez(:)),...
           var(logdata_OFF_realtimeRx(:)),var(logdata_OFF_realtimeRy(:)),var(logdata_OFF_realtimeRz(:))]

std_OFF = [ std(logdata_OFF_realtimex(:)),std(logdata_OFF_realtimey(:)),std(logdata_OFF_realtimez(:)),...
           std(logdata_OFF_realtimeRx(:)),std(logdata_OFF_realtimeRy(:)),std(logdata_OFF_realtimeRz(:))]  

med_OFF = [ median(logdata_OFF_realtimex(:)),median(logdata_OFF_realtimey(:)),median(logdata_OFF_realtimez(:)),...
           median(logdata_OFF_realtimeRx(:)),median(logdata_OFF_realtimeRy(:)),median(logdata_OFF_realtimeRz(:))]              

kurt_OFF = [ kurtosis(logdata_OFF_realtimex(:)),kurtosis(logdata_OFF_realtimey(:)),kurtosis(logdata_OFF_realtimez(:)),...
           kurtosis(logdata_OFF_realtimeRx(:)),kurtosis(logdata_OFF_realtimeRy(:)),kurtosis(logdata_OFF_realtimeRz(:))]              
  
%mom_OFF = [ moment(logdata_OFF_realtimex(:)),moment(logdata_OFF_realtimey(:)),moment(logdata_OFF_realtimez(:)),...
%           moment(logdata_OFF_realtimeRx(:)),moment(logdata_OFF_realtimeRy(:)),moment(logdata_OFF_realtimeRz(:))]              
 
skew_OFF = [ skewness(logdata_OFF_realtimex(:)),skewness(logdata_OFF_realtimey(:)),skewness(logdata_OFF_realtimez(:)),...
           skewness(logdata_OFF_realtimeRx(:)),skewness(logdata_OFF_realtimeRy(:)),skewness(logdata_OFF_realtimeRz(:))]              
    
'''
temporal_data_OFF = sio.loadmat('./Off/zz_tmp_mat/T1/au70_T1_OFF.mat')
temporal_data_ON = sio.loadmat('./On/zz_tmp_mat/T1/au70_T1_ON.mat')
# total motion
y_OFF=np.array(temporal_data_OFF['total_motionOFF'])
y_OFF.resize(y_OFF.size)

y_ON=np.array(temporal_data_ON['total_motionON'])
y_ON.resize(y_ON.size)

# mean values
mean_OFF=np.array(temporal_data_OFF['mean_OFF'])
mean_OFF.resize(mean_OFF.size)

mean_ON=np.array(temporal_data_ON['mean_ON'])
mean_ON.resize(mean_ON.size)

# std values
std_OFF=np.array(temporal_data_OFF['std_OFF'])
std_OFF.resize(std_OFF.size)

std_ON=np.array(temporal_data_ON['std_ON'])
std_ON.resize(std_ON.size)

# variance values
var_OFF=np.array(temporal_data_OFF['var_OFF'])
var_OFF.resize(var_OFF.size)

var_ON=np.array(temporal_data_ON['var_ON'])
var_ON.resize(var_ON.size)

# median values
med_OFF=np.array(temporal_data_OFF['med_OFF'])
med_OFF.resize(med_OFF.size)

med_ON=np.array(temporal_data_ON['med_ON'])
med_ON.resize(med_ON.size)

# kurtosis values
kurt_OFF=np.array(temporal_data_OFF['kurt_OFF'])
kurt_OFF.resize(kurt_OFF.size)

kurt_ON=np.array(temporal_data_ON['kurt_ON'])
kurt_ON.resize(kurt_ON.size)

# skewness values
skew_OFF=np.array(temporal_data_OFF['skew_OFF'])
skew_OFF.resize(skew_OFF.size)

skew_ON=np.array(temporal_data_ON['skew_ON'])
skew_ON.resize(skew_ON.size)
################################################################################
# total motion
################################################################################
trace0 = Bar(
    x=['Displ x (mm)', 'Displ y (mm)', 'Displ z (mm)', 'Rot x (°)', 'Rot y (°)', 'Rot z (°)'], 
    #y=[20.5, 13.5, 25, 16, 18, 22, 66],
    y=y_OFF,
    name='PMC OFF',
    marker=dict(
        color='rgb(49,130,189)'
    )
)
trace1 = Bar(
    x=['Displ x (mm)', 'Displ y (mm)', 'Displ z (mm)', 'Rot x (°)', 'Rot y (°)', 'Rot z (°)'],
    #y=[19, 14, 22, 14, 16, 19,44],
    y=y_ON,
    name='PMC ON',
    marker=dict(
        color='rgb(204,204,204)',
    )
)
################################################################################
# mean values
################################################################################
trace2 = Bar(
    x=['Displ x (mm)', 'Displ y (mm)', 'Displ z (mm)', 'Rot x (°)', 'Rot y (°)', 'Rot z (°)'],
    #y=[19, 14, 22, 14, 16, 19,44],
    y=mean_OFF,
    name='PMC OFF'#,
    #marker=dict(
    #    color='rgb(49,130,189)',
    #)
)

trace3 = Bar(
    x=['Displ x (mm)', 'Displ y (mm)', 'Displ z (mm)', 'Rot x (°)', 'Rot y (°)', 'Rot z (°)'],
    #y=[19, 14, 22, 14, 16, 19,44],
    y=mean_ON,
    name='PMC ON'#,
#    marker=dict(
#        color='rgb(204,204,204)',
#    )
)
################################################################################
# std values
################################################################################
trace4 = Bar(
    x=['Displ x (mm)', 'Displ y (mm)', 'Displ z (mm)', 'Rot x (°)', 'Rot y (°)', 'Rot z (°)'],
    #y=[19, 14, 22, 14, 16, 19,44],
    y=std_OFF,
    name='PMC OFF'#,
    #marker=dict(
    #    color='rgb(204,204,204)',
    #)
)

trace5 = Bar(
    x=['Displ x (mm)', 'Displ y (mm)', 'Displ z (mm)', 'Rot x (°)', 'Rot y (°)', 'Rot z (°)'],
    #y=[19, 14, 22, 14, 16, 19,44],
    y=std_ON,
    name='PMC ON'#,
    #marker=dict(
    #    color='rgb(204,204,204)',
    #)
)
################################################################################
# variance values
################################################################################
trace6 = Bar(
    x=['Displ x (mm)', 'Displ y (mm)', 'Displ z (mm)', 'Rot x (°)', 'Rot y (°)', 'Rot z (°)'],
    #y=[19, 14, 22, 14, 16, 19,44],
    y=var_OFF,
    name='PMC OFF'#,
    #marker=dict(
    #    color='rgb(204,204,204)',
    #)
)

trace7 = Bar(
    x=['Displ x (mm)', 'Displ y (mm)', 'Displ z (mm)', 'Rot x (°)', 'Rot y (°)', 'Rot z (°)'],
    #y=[19, 14, 22, 14, 16, 19,44],
    y=var_ON,
    name='PMC ON'#,
    #marker=dict(
    #    color='rgb(204,204,204)',
    #)
)
################################################################################
# median values
################################################################################
trace8 = Bar(
    x=['Displ x (mm)', 'Displ y (mm)', 'Displ z (mm)', 'Rot x (°)', 'Rot y (°)', 'Rot z (°)'],
    #y=[19, 14, 22, 14, 16, 19,44],
    y=med_OFF,
    name='PMC OFF'#,
    #marker=dict(
    #    color='rgb(204,204,204)',
    #)
)

trace9 = Bar(
    x=['Displ x (mm)', 'Displ y (mm)', 'Displ z (mm)', 'Rot x (°)', 'Rot y (°)', 'Rot z (°)'],
    #y=[19, 14, 22, 14, 16, 19,44],
    y=med_ON,
    name='PMC ON'#,
    #marker=dict(
    #    color='rgb(204,204,204)',
    #)
)
################################################################################
# kurtosis values
################################################################################
trace10 = Bar(
    x=['Displ x (mm)', 'Displ y (mm)', 'Displ z (mm)', 'Rot x (°)', 'Rot y (°)', 'Rot z (°)'],
    #y=[19, 14, 22, 14, 16, 19,44],
    y=kurt_OFF,
    name='PMC OFF'#,
    #marker=dict(
    #    color='rgb(204,204,204)',
    #)
)

trace11 = Bar(
    x=['Displ x (mm)', 'Displ y (mm)', 'Displ z (mm)', 'Rot x (°)', 'Rot y (°)', 'Rot z (°)'],
    #y=[19, 14, 22, 14, 16, 19,44],
    y=kurt_ON,
    name='PMC ON'#,
    #marker=dict(
    #    color='rgb(204,204,204)',
    #)
)
############
# total motion, average motion, standard deviation,
# variance, median, kurtosis or skewness
#fig = tools.make_subplots(rows=2, cols=3, subplot_titles=('Total Motion', 'Mean values', 'Standard deviation',
#                                                          'Variance', 'Median','Kurtosis'  ))
fig = subplots.make_subplots(rows=2, cols=3, subplot_titles=('Total Motion', 'Mean values', 'Standard deviation',
                                                          'Variance', 'Median','Kurtosis'  ))
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 1, 2)
fig.append_trace(trace4, 1, 3)
fig.append_trace(trace5, 1, 3)

fig.append_trace(trace6, 2, 1)
fig.append_trace(trace7, 2, 1)
fig.append_trace(trace8, 2, 2)
fig.append_trace(trace9, 2, 2)
fig.append_trace(trace10, 2, 3)
fig.append_trace(trace11, 2, 3)

plotly.offline.plot(fig, filename='statistical_mp_au70.html')
'''
data = [trace0, trace1]
layout = Layout(
    title='Total Motion',
    xaxis=dict(
        # set x-axis' labels direction at 45 degree angle
        tickangle=-45,
    ),
    barmode='group',
)
fig = Figure(data=data, layout=layout)
#plot_url = py.plot(fig, filename='angled-text-bar')
#plotly.offline.plot(data, filename='scatter-plot-with-colorscale')
plotly.offline.plot(fig, filename='statistical_mp_au70.html')
'''