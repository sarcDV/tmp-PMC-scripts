import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from scipy.stats import mannwhitneyu
from scipy.stats import wilcoxon
import warnings
warnings.filterwarnings("ignore")

def newline(p1, p2):
    ax = plt.gca()
    xmin, xmax = ax.get_xbound()

    if(p2[0] == p1[0]):
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
        ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])

    l = mlines.Line2D([xmin,xmax], [ymin,ymax], color="#cccccc", linestyle="--")
    ax.add_line(l)
    return l


def repeat_string(string, length):
    empty_string_ = []
    for ii in range(0, length):
        empty_string_.append(string)

    stringa_ = np.asarray(empty_string_)

    return stringa_

from scipy import stats

def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])


def percentage_eval(in1_, in2_):
    onebettertwo=0
    same_=0
    twobetterone=0
    for ii in range(0, len(in1_)):
        if in1_[ii] == in2_[ii]:
            same_=same_#+1
        elif in1_[ii] > in2_[ii]:
            onebettertwo=onebettertwo+1
        else:
            twobetterone=twobetterone+1

    onebettertwo=(onebettertwo/len(in1_))*100
    same_=(same_/len(in1_))*100
    twobetterone=-(twobetterone/len(in1_))*100

    return truncate(twobetterone,2), truncate(same_,2), truncate(onebettertwo,2)

def convert_pval(inval):
    if inval < 0.01:
        outval = "p < 0.01" 
    elif inval >= 0.01:
        outval = "p = "+str(truncate(inval,2))
    return outval
###############################################
### read the weights ##########################
###############################################

woff_, won_ = np.zeros((21,)), np.zeros((21,))# [], []
contrastsw_ =  ['T1', 'T2', 'PD', 'T2s05', 'T2s035', 'T2s025']
for contrast in contrastsw_:
    filepath_ = './statistics-logs-PMC/weights/'+str(contrast)+'_weights.txt'
    
    tmp_ = np.reshape(np.asarray(open(filepath_).read().split()),(21,3))
    
    woff_ = np.vstack((woff_, tmp_[:,1]))
    won_ = np.vstack((won_, tmp_[:,2]))
    

woff_, won_ = woff_[1:,:], won_[1:, :]

##############################################
### read the scores and adjust the scores ####
##############################################
contrastssc_ =  ['T1', 'T2', 'PD', 'T2star-05', 'T2star-035', 'T2star-025']

t1_off_, t2_off_, pd_off_ = np.zeros((1)), np.zeros((1)), np.zeros((1))
t2s05_off_, t2s035_off_, t2s025_off_ = np.zeros((1)), np.zeros((1)), np.zeros((1))
t1_on_, t2_on_, pd_on_ = np.zeros((1)), np.zeros((1)), np.zeros((1))
t2s05_on_, t2s035_on_, t2s025_on_ = np.zeros((1)), np.zeros((1)), np.zeros((1))
count_ = 0
for contrast in contrastssc_:
    filepathoff_ = './CustomGradientEntropy_analysis/'+str(contrast)+'_OFF.txt'
    testoff_ = pd.read_csv(filepathoff_, header=None)
    filepathon_ = './CustomGradientEntropy_analysis/'+str(contrast)+'_ON.txt'
    teston_ = pd.read_csv(filepathon_, header=None)
    # print(test_)
    for ii in range(0,21):
        # print(testoff_[0][ii].split(" "))
        tmpoff_ = list(testoff_[0][ii].split(" "))
        tmpoff_ = (np.asarray(list(filter(None, tmpoff_))))
        tmpon_ = list(teston_[0][ii].split(" "))
        tmpon_ = (np.asarray(list(filter(None, tmpon_))))
        
        if count_ == 0:
            aoff_ = np.double(tmpoff_[1:]) # -np.double(woff_[count_, ii])
            aon_ = np.double(tmpon_[1:])# -np.double(won_[count_, ii])
            ## -----------
            check_size_off_ = int(np.asarray(aoff_.shape))
            check_size_on_ = int(np.asarray(aon_.shape))
            if (check_size_off_ > check_size_on_):
                aoff_  = aoff_[check_size_off_ -check_size_on_:]
            elif (check_size_off_ < check_size_on_):
                aon_ = aon_[check_size_on_ -check_size_off_:]
            ## -----------
            t1_off_ = np.concatenate((t1_off_, aoff_))
            t1_on_= np.concatenate((t1_on_, aon_))
        elif count_ == 1:
            aoff_ = np.double(tmpoff_[1:])# -np.double(woff_[count_, ii])#
            aon_ = np.double(tmpon_[1:])# -np.double(won_[count_, ii])
            t2_off_ = np.concatenate((t2_off_, aoff_))
            t2_on_= np.concatenate((t2_on_, aon_))
        elif count_ == 2:
            aoff_ = np.double(tmpoff_[1:])#-np.double(woff_[count_, ii])
            aon_ = np.double(tmpon_[1:])#-np.double(won_[count_, ii])
            pd_off_ = np.concatenate((pd_off_, aoff_))
            pd_on_= np.concatenate((pd_on_, aon_))
        elif count_ == 3:
            aoff_ = np.double(tmpoff_[1:])#-np.double(woff_[count_, ii])
            aon_ = np.double(tmpon_[1:])#-np.double(won_[count_, ii])
            t2s05_off_ = np.concatenate((t2s05_off_, aoff_))
            t2s05_on_= np.concatenate((t2s05_on_, aon_))
        elif count_ == 4:
            aoff_ = np.double(tmpoff_[1:]) # -np.double(woff_[count_, ii])
            aon_ = np.double(tmpon_[1:]) # -np.double(won_[count_, ii])
            t2s035_off_ = np.concatenate((t2s035_off_, aoff_))
            t2s035_on_= np.concatenate((t2s035_on_, aon_))
        else:
            aoff_ = np.double(tmpoff_[1:])#-np.double(woff_[count_, ii])
            aon_ = np.double(tmpon_[1:])#-np.double(won_[count_, ii])
            t2s025_off_ = np.concatenate((t2s025_off_, aoff_))
            t2s025_on_= np.concatenate((t2s025_on_, aon_))
        # print(np.double(tmpoff_[1:])*np.double(woff_[count_, ii]))
        # print((tmpoff_[1:]))
        # print(woff_[count_, ii])

    count_ = count_+1


#######################################################
## evaluation #########################################
#######################################################

T1_eval_= []
T2_eval_, PD_eval_= [],[]
T2s05_eval_, T2s035_eval_, T2s025_eval_ = [],[],[]


R1offbon, R1same_, R1onboff = percentage_eval(t1_off_, t1_on_)
T1_eval_.append([R1offbon, R1same_, R1onboff])


R2offbon, R2same_, R2onboff = percentage_eval(t2_off_, t2_on_)
T2_eval_.append([R2offbon, R2same_, R2onboff])

R3offbon, R3same_, R3onboff = percentage_eval(pd_off_, pd_on_)
PD_eval_.append([R3offbon, R3same_, R3onboff])

R4offbon, R4same_, R4onboff = percentage_eval(t2s05_off_, t2s05_on_)
T2s05_eval_.append([R4offbon, R4same_, R4onboff])

R5offbon, R5same_, R5onboff = percentage_eval(t2s035_off_, t2s035_on_)
T2s035_eval_.append([R5offbon, R5same_, R5onboff])

R6offbon, R6same_, R6onboff = percentage_eval(t2s025_off_, t2s025_on_)
T2s025_eval_.append([R6offbon, R6same_, R6onboff])


T1_eval_     = np.float64(np.asarray(T1_eval_))
T2_eval_     = np.float64(np.asarray(T2_eval_))
PD_eval_     = np.float64(np.asarray(PD_eval_))
T2s05_eval_  = np.float64(np.asarray(T2s05_eval_)) 
T2s035_eval_ = np.float64(np.asarray(T2s035_eval_))
T2s025_eval_ = np.float64(np.asarray(T2s025_eval_))

#######################################################
# compare samples #####################################
#######################################################
stackoffall_ = np.concatenate((t1_off_, t2_off_, pd_off_, t2s05_off_, t2s035_off_, t2s025_off_))
stackonall_ = np.concatenate((t1_on_, t2_on_, pd_on_, t2s05_on_, t2s035_on_, t2s025_on_))

stackoff_ = np.array([t1_off_, t2_off_, pd_off_, t2s05_off_, t2s035_off_, t2s025_off_, stackoffall_])
stackon_ = np.array([t1_on_, t2_on_, pd_on_, t2s05_on_, t2s035_on_, t2s025_on_, stackonall_])

pvalues = []
for ii in range(0, len(stackoff_)):
    stat, p = mannwhitneyu(stackoff_[ii], stackon_[ii])
    print('Statistics=%.3f, p=%.5f' % (stat, p))
    pvalues.append(p)
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Same distribution (fail to reject H0)')
    else:
        print('Different distribution (reject H0)')

print(pvalues)
#######################################################
## PLOT ###############################################
#######################################################
coff_ = "#3274A1"##"#333333"
con_ = "#E1812C"####"#b2b2b2"
csame_ ="#ffffff"#"#b2b2b2" ###"#01d1a1"
cedge_ ="#000000"

blim_ = -75.
uplim_ = 75.
width = 0.5
fig = plt.figure(figsize = (5.5,5))##1)
# grid = plt.GridSpec(2, 4, hspace=0.2, wspace=0.2)
line_labelx = [ "PMC OFF better\nthan PMC ON", 
                "PMC ON better\nthan PMC OFF"]
# Create the bar labels
bar_labels = ['T1-w', 'T2-w', 'PD-w', 'T2*-w (05)','T2*-w (035)','T2*-w (025)','ALL CONTRASTS']
bar_finlabels = ['T1-w', 'T2-w', 'PD-w', 'T2*-w\n(05)','T2*-w\n(035)','T2*-w\n(025)','ALL\nCONT.']
assey_ = [1,2,3,4,5,6,7]
## READER 1 ###
ax1 = plt.subplot(111)#(grid[0,0])# plt.subplot(2,4,1)
read1off_ = [T1_eval_[0,0],T2_eval_[0,0],PD_eval_[0,0],T2s05_eval_[0,0],T2s035_eval_[0,0],T2s025_eval_[0,0],
            np.mean([T1_eval_[0,0],T2_eval_[0,0],PD_eval_[0,0],T2s05_eval_[0,0],T2s035_eval_[0,0],T2s025_eval_[0,0]])]
read1on_ = [T1_eval_[0,2],T1_eval_[0,2],PD_eval_[0,2],T2s05_eval_[0,2],T2s035_eval_[0,2],T2s025_eval_[0,2],
            np.mean([T1_eval_[0,2],T1_eval_[0,2],PD_eval_[0,2],T2s05_eval_[0,2],T2s035_eval_[0,2],T2s025_eval_[0,2]])]

xerroroff_ = [0.,0.,0.,0.,0.,0.,
            np.std([T1_eval_[0,0],T2_eval_[0,0],PD_eval_[0,0],T2s05_eval_[0,0],T2s035_eval_[0,0],T2s025_eval_[0,0]])]
xerroron_ = [0.,0.,0.,0.,0.,0.,
            np.std([T1_eval_[0,2],T1_eval_[0,2],PD_eval_[0,2],T2s05_eval_[0,2],T2s035_eval_[0,2],T2s025_eval_[0,2]])]          

l1 = ax1.barh(assey_, read1off_, xerr=xerroroff_, color=coff_ , edgecolor=cedge_, capsize=3)
l2 = ax1.barh(assey_, read1on_,  xerr=xerroron_,color= con_, edgecolor=cedge_, capsize=3)  #, hatch='/' )
plt.xticks(fontsize=10, fontweight='bold')
plt.yticks(assey_, bar_finlabels, fontsize=10, fontweight='bold')
plt.xlabel('(%)', fontsize=10, fontweight='bold')
ax1.set_xlim(blim_, uplim_)
plt.title('Gradient \nEntropy', fontsize=10, fontweight='bold')
fig.legend([l1, l2],     # The line objects
           labels=line_labelx,   # The labels for each line
           loc="upper right",# "upper center",   # Position of legend
           # borderaxespad=0.1,    # Small spacing around legend box
           #title=""# "Legend Title"  # Title for the legend
           fontsize=9)
###
don_, doff_  = 17.5, -3
sh_ = 0.15

props = dict(boxstyle='round', facecolor='wheat', alpha=1)
for ii in range(0, len(pvalues)):
    ax1.text(read1on_[ii]-don_, ii+1+sh_+0.05,str(truncate(read1on_[ii],2)), fontweight='bold', fontsize=9, bbox=props)
    ax1.text(read1off_[ii]-doff_,ii+1+sh_+0.05,str(truncate(read1off_[ii]*(-1),2)), fontweight='bold', fontsize=9, bbox=props)
    ax1.text(uplim_-4.5, ii+0.95,str(convert_pval(pvalues[ii])), fontweight='bold', fontsize=7, bbox=props)
    """
    if ((read1off_[ii]*(-1)) > read1on_[ii] ):
        print("off better than on")
        ax1.text(read1on_[ii]+don_/4, ii+1+sh_+0.05,str(convert_pval(pvalues[ii])), fontweight='bold', fontsize=10)
    else:
        print("on better than off")
        ax1.text(read1off_[ii]+8*doff_, ii+1+sh_+0.05,str(convert_pval(pvalues[ii])), fontweight='bold', fontsize=10)    
    """

ax1.grid(True)

plt.show()

