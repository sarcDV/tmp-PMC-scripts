import plotly
from plotly import subplots, tools
from plotly.graph_objs import Scatter, Layout, Figure
import scipy.io as sio
import numpy as np

# load the data
#temporal_data = sio.loadmat('data_842-3.mat')
temporal_data_OFF = sio.loadmat('./Off/zz_tmp_mat/PD/au70_PD_OFF.mat')
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
temporal_data_ON = sio.loadmat('./On/zz_tmp_mat/PD/au70_PD_ON.mat')
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
# Create traces

trace0 = Scatter(
    x = random_x,
    y = random_y0,
    mode = 'lines',
    line = dict(
        color = ('green')),
    name = 'Rotation x-axis'
)

trace1 = Scatter(
    x = random_x,
    y = random_y1,
    mode = 'lines',
    line = dict(
        color = ('blue')),
    name = 'Rotation y-axis'
)

trace2 = Scatter(    
    x = random_x,
    y = random_y2, 
    mode = 'lines',
    line = dict(
        color = ('orange')),
    name = 'Rotation z-axis'
)

trace3 = Scatter(
    x = random_x,
    y= random_t0,
    mode = 'lines',
    line = dict(
        color = ('red')),
    name = 'Displacement x-axis'
    )

trace4 = Scatter(
    x = random_x,
    y= random_t1,
    mode = 'lines',
    line = dict(
        color = ('rgb(76, 153, 0)')),
    name = 'Displacement y-axis'
    )

trace5 = Scatter(
    x = random_x,
    y= random_t2,
    mode = 'lines',
    line = dict(
        color = ('rgb(0, 0, 102)')),
    name = 'Displacement z-axis'
    )

trace6 = Scatter(
    x = time_x,
    y = rot_y0,
    mode = 'lines',
    line = dict(
        color = ('green')),
   #     color = ('rgb(22, 96, 167)'),
   #     width = 4,
   #     dash = 'dash')
    name = 'Rotation x-axis', 
    showlegend = False 
)

trace7 = Scatter(
    x = time_x,
    y = rot_y1,
    mode = 'lines',
    line = dict(
        color = ('blue')),
    name = 'Rotation y-axis', 
    showlegend = False 
)

trace8 = Scatter(    
    x = time_x,
    y = rot_y2, 
    mode = 'lines',
    line = dict(
        color = ('orange')),
    name = 'Rotation z-axis', 
    showlegend = False 
)

trace9 = Scatter(
    x = time_x,
    y= tra_t0,
    mode = 'lines',
    line = dict(
        color = ('red')),
    name = 'Displacement x-axis', 
    showlegend = False 
    )

trace10 = Scatter(
    x = time_x,
    y= tra_t1,
    mode = 'lines',
    line = dict(
        color = ('rgb(76, 153, 0)')),
    name = 'Displacement y-axis', 
    showlegend = False 
    )

trace11 = Scatter(
    x = time_x,
    y= tra_t2,
    mode = 'lines',
    line = dict(
        color = ('rgb(0, 0, 102)')),
    name = 'Displacement z-axis', 
    showlegend = False 
    )


    
#fig = Figure(data=data)
fig = subplots.make_subplots(rows=2, cols=2, subplot_titles=('PMC OFF', 'PMC ON'))

fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 1)
fig.append_trace(trace3, 2, 1)
fig.append_trace(trace4, 2, 1)
fig.append_trace(trace5, 2, 1)

fig.append_trace(trace6, 1, 2)
fig.append_trace(trace7, 1, 2)
fig.append_trace(trace8, 1, 2)
fig.append_trace(trace9, 2, 2)
fig.append_trace(trace10, 2, 2)
fig.append_trace(trace11, 2, 2)

# All of the axes properties here: https://plot.ly/python/reference/#XAxis

fig['layout']['xaxis1'].update(title='Time(seconds)')
fig['layout']['xaxis2'].update(title='Time(seconds)')
fig['layout']['xaxis3'].update(title='Time(seconds)')
fig['layout']['xaxis4'].update(title='Time(seconds)')


# All of the axes properties here: https://plot.ly/python/reference/#YAxis

fig['layout']['yaxis1'].update(title='Rotations (degrees)', range=[-0.3, 0.3])
fig['layout']['yaxis2'].update(title='Rotations (degrees)', range=[-0.3, 0.3])
fig['layout']['yaxis3'].update(title='Displacement (mm)', range=[-0.8, 0.8])
fig['layout']['yaxis4'].update(title='Displacement (mm)', range=[-0.8, 0.8])


fig['layout'].update(title='Motion Patterns')



plotly.offline.plot(fig, filename='au70_T1.html')
