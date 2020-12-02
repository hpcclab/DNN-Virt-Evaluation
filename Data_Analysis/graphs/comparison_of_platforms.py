import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import scipy.stats as st



dir = '../Yolov3/experiment_result/'

platforms = dict()

for platform in os.listdir(dir):
    
    platforms.update({platform:None})
    executions = []
    
    for file in os.listdir(dir+platform):
       with open(dir+platform+'/'+file, newline='') as csvfile:
           reader = csv.reader(csvfile, delimiter=',')
           for row in reader:
               if row[0][:6] == 'output':
                   executions.append(float(row[2]))
    platforms[platform] = executions
    

stat_platforms = dict()
for key, value in platforms.items():    
    mean_key = 'mean_'+key
    CI_key = 'CI_'+key    
    mean = np.mean(value)    
    CI=st.t.interval(alpha=0.95, df=len(value)-1, loc=np.mean(value),
                     scale=st.sem(value))     
    #print('Key:' + key + '  mean = '+str(mean) +' CI = '+str(CI))    
    stat_platforms.update({CI_key:CI[1]-CI[0], mean_key:mean})

labels = ['BM', 'CN', 'VM', 'VMCN']


Cores = [2,4,8,16,32,64]
vanilla_means = [stat_platforms['mean_64-vanilla-BM'],
                 stat_platforms['mean_64-vanilla-CN'],
                 stat_platforms['mean_64-vanilla-VM'],
                 stat_platforms['mean_64-vanilla-VMCN']
                 ]
GPU_means = [    stat_platforms['mean_64-GPU-BM'],
                 stat_platforms['mean_64-GPU-CN'],
                 stat_platforms['mean_64-GPU-VM'],
                 stat_platforms['mean_64-GPU-VMCN']
                 ]
pinned_means  = [0,
                 stat_platforms['mean_64-pinned-CN'],
                 stat_platforms['mean_64-pinned-VM'],
                 stat_platforms['mean_64-pinned-VMCN']
                 ]

vanilla_yerr = [stat_platforms['CI_64-vanilla-BM'],
                 stat_platforms['CI_64-vanilla-CN'],
                 stat_platforms['CI_64-vanilla-VM'],
                 stat_platforms['CI_64-vanilla-VMCN']
                 ]

GPU_yerr = [    stat_platforms['CI_64-GPU-BM'],
                 stat_platforms['CI_64-GPU-CN'],
                 stat_platforms['CI_64-GPU-VM'],
                 stat_platforms['CI_64-GPU-VMCN']
                 ]



pinned_yerr  = [0,
                 stat_platforms['CI_64-pinned-CN'],
                 stat_platforms['CI_64-pinned-VM'],
                 stat_platforms['CI_64-pinned-VMCN']
                 ]


x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars

#hatch = [ '/', '\', '|', '-', '+', 'x', 'o', 'O', '.', '*']
hatch = ['///' , '\\\\\\' , '|','---','xx','o','O','..','*']

fig, ax = plt.subplots()

rects1 = ax.bar(x - width, GPU_means, width=width, yerr=GPU_yerr,
                capsize=4, hatch=hatch[0],fill=False,
                edgecolor= 'darkred',label='GPU')

rects2 = ax.bar(x , pinned_means,width=width, yerr=pinned_yerr,
                capsize=4,hatch=hatch[1],fill=False, 
                edgecolor='green',label='Pinned')
r = x.astype(np.float)
r[0] = r[0] - width
rects3 = ax.bar( r+ width , vanilla_means, width=width, yerr=vanilla_yerr,
                capsize=4,hatch=hatch[7], fill=False,
                edgecolor='navy',label='Vanilla')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Inference Time [sec]',fontsize=14,fontweight = 'bold')
ax.set_xlabel('Execution Platforms',fontsize=14,fontweight = 'bold')
#ax.set_title('Inference Time for Execution Platforms',fontsize=16,fontweight = 'bold')
x = x.astype(np.float)
x[0] = x[0] - width/2.

ax.set_ylim(0,250)
ax.set_xticks(x)
ax.set_xticklabels(labels,fontsize = 14)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:5.0f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


# autolabel(rects1)
# autolabel(rects2)
# autolabel(rects3)


fig.tight_layout()
#plt.savefig('comparison.png')
plt.show()