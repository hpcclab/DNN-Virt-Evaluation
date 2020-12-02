import matplotlib
import matplotlib.pyplot as plt
import numpy as np
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
cores = [2,4,8,16,32,64]
VM   = []
BM   = []
CN   = []
VMCN = []

VM_yerr = []
BM_yerr = []
CN_yerr = []
VMCN_yerr=[]
for experiment in stat_platforms:
    
    core_no = int(experiment.split('_')[1].split('-')[0])
    core_type = experiment.split('_')[1].split('-')[1]
    platform = experiment.split('_')[1].split('-')[2]   
    
    if core_type =='GPU' and experiment.split('_')[0]=='mean':
        #print('Numer of Cores = '+ str(core_no) + '  platform = ' + platform)
        if platform == 'VM':
            VM.append([core_no,
                       stat_platforms['mean_'+str(core_no)+'-'+
                                      core_type+'-'+platform]])
            
        if platform == 'BM':
            BM.append([core_no,
                       stat_platforms['mean_'+str(core_no)+'-'+
                                      core_type+'-'+platform]])
            
        if platform == 'CN':
            CN.append([core_no,
                       stat_platforms['mean_'+str(core_no)+'-'+
                                      core_type+'-'+platform]])
            
        if platform == 'VMCN':
            VMCN.append([core_no,
                       stat_platforms['mean_'+str(core_no)+'-'+
                                      core_type+'-'+platform]])
            
    if core_type =='GPU' and experiment.split('_')[0]=='CI':
        #print('Numer of Cores = '+ str(core_no) + '  platform = ' + platform)
        if platform == 'VM':
            VM_yerr.append([core_no,
                       stat_platforms['CI_'+str(core_no)+'-'+
                                      core_type+'-'+platform]])
            
        if platform == 'BM':
            BM_yerr.append([core_no,
                       stat_platforms['CI_'+str(core_no)+'-'+
                                      core_type+'-'+platform]])
            
        if platform == 'CN':
            CN_yerr.append([core_no,
                       stat_platforms['CI_'+str(core_no)+'-'+
                                      core_type+'-'+platform]])
            
        if platform == 'VMCN':
            VMCN_yerr.append([core_no,
                       stat_platforms['CI_'+str(core_no)+'-'+
                                    core_type+'-'+platform]])
print(VM)
VM = np.array(VM)
BM = np.array(BM)
CN = np.array(CN)
VMCN = np.array(VMCN)

VM   = VM[VM[:,0].argsort()]
BM   = BM[BM[:,0].argsort()]
CN   = CN[CN[:,0].argsort()]
VMCN = VMCN[VMCN[:,0].argsort()]

VM_yerr   = np.sort(np.array(VM_yerr),axis=0)
BM_yerr   = np.sort(np.array(BM_yerr),axis=0)
CN_yerr  = np.sort(np.array(CN_yerr),axis=0)
VMCN_yerr = np.sort(np.array(VMCN_yerr),axis=0)

markers = ['o', 'v' , 'x',4]
linestyles = ['--',(0, (3, 1, 1, 1)),'-.','-']
colors = ['darkred','blue','darkgreen','goldenrod']
fig, ax = plt.subplots()
ax.set_xticks(cores)
ax.errorbar(BM[:,0],BM[:,1],yerr = BM_yerr[:,1], capsize=5, elinewidth=2,linestyle=linestyles[0],color = colors[0],marker=markers[0],linewidth=2, markersize=8,label=labels[0]) 
ax.errorbar(CN[:,0],CN[:,1],yerr = CN_yerr[:,1], capsize=5, elinewidth=2,linestyle=linestyles[1],color = colors[1],marker=markers[1],linewidth=2, markersize=8,label=labels[1]) 
ax.errorbar(VM[:,0],VM[:,1],yerr = VM_yerr[:,1], capsize=5, elinewidth=2,linestyle=linestyles[2],color = colors[2],marker=markers[2],linewidth=2, markersize=8,label=labels[2]) 
ax.errorbar(VMCN[:,0],VMCN[:,1],yerr = VMCN_yerr[:,1], capsize=5, elinewidth=2,linestyle=linestyles[3],color = colors[3],marker=markers[3],linewidth=2, markersize=8,label=labels[3])     

ax.set_ylabel('Inference Time [sec]',fontsize=14)
ax.set_xlabel('Number of CPU Cores',fontsize=14)
ax.grid('minor') 
ax.legend()     
fig.tight_layout()
plt.savefig('comparison_GPU_Cores.png')
plt.savefig('comparison_GPU_Cores.pdf')
# =============================================================================
# Cores = [2,4,8,16,32,64]
# vanilla_means = [stat_platforms['mean_64-vanilla-BM'],
#                  stat_platforms['mean_64-vanilla-CN'],
#                  stat_platforms['mean_64-vanilla-VM'],
#                  stat_platforms['mean_64-vanilla-VMCN']
#                  ]
# GPU_means = [    stat_platforms['mean_64-GPU-BM'],
#                  stat_platforms['mean_64-GPU-CN'],
#                  stat_platforms['mean_64-GPU-VM'],
#                  stat_platforms['mean_64-GPU-VMCN']
#                  ]
# pinned_means  = [0,
#                  stat_platforms['mean_64-pinned-CN'],
#                  stat_platforms['mean_64-pinned-VM'],
#                  stat_platforms['mean_64-pinned-VMCN']
#                  ]
# 
# vanilla_yerr = [stat_platforms['CI_64-vanilla-BM'],
#                  stat_platforms['CI_64-vanilla-CN'],
#                  stat_platforms['CI_64-vanilla-VM'],
#                  stat_platforms['CI_64-vanilla-VMCN']
#                  ]
# 
# GPU_yerr = [    stat_platforms['CI_64-GPU-BM'],
#                  stat_platforms['CI_64-GPU-CN'],
#                  stat_platforms['CI_64-GPU-VM'],
#                  stat_platforms['CI_64-GPU-VMCN']
#                  ]
# 
# 
# 
# pinned_yerr  = [0,
#                  stat_platforms['CI_64-pinned-CN'],
#                  stat_platforms['CI_64-pinned-VM'],
#                  stat_platforms['CI_64-pinned-VMCN']
#                  ]
# 
# 
# x = np.arange(len(labels))  # the label locations
# width = 0.2  # the width of the bars
# 
# #hatch = [ '/', '\', '|', '-', '+', 'x', 'o', 'O', '.', '*']
# hatch = ['///' , '\\\\\\' , '|','---','xx','o','O','..','*']
# 
# fig, ax = plt.subplots()
# 
# rects1 = ax.bar(x - width, GPU_means, width=width, yerr=GPU_yerr,
#                 capsize=4, hatch=hatch[0],fill=False,
#                 edgecolor= 'darkred',label='GPU')
# 
# rects2 = ax.bar(x , pinned_means,width=width, yerr=pinned_yerr,
#                 capsize=4,hatch=hatch[1],fill=False, 
#                 edgecolor='green',label='Pinned')
# r = x.astype(np.float)
# r[0] = r[0] - width
# rects3 = ax.bar( r+ width , vanilla_means, width=width, yerr=vanilla_yerr,
#                 capsize=4,hatch=hatch[7], fill=False,
#                 edgecolor='navy',label='Vanilla')
# 
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Inference Time [sec]',fontsize=14,fontweight = 'bold')
# ax.set_xlabel('Execution Platforms',fontsize=14,fontweight = 'bold')
# #ax.set_title('Inference Time for Execution Platforms',fontsize=16,fontweight = 'bold')
# x = x.astype(np.float)
# x[0] = x[0] - width/2.
# 
# ax.set_ylim(0,250)
# ax.set_xticks(x)
# ax.set_xticklabels(labels,fontsize = 14)
# ax.legend()
# 
# 
# def autolabel(rects):
#     """Attach a text label above each bar in *rects*, displaying its height."""
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate('{:5.0f}'.format(height),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),  # 3 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom')
# 
# 
# # autolabel(rects1)
# # autolabel(rects2)
# # autolabel(rects3)
# 
# 
# fig.tight_layout()
# #plt.savefig('comparison.png')
# plt.show()
# =============================================================================