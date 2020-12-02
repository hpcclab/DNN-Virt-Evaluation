import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import csv
import os
import scipy.stats as st


def read_data(path): 
    stat_experiments = dict()    
    for experiment in os.listdir(path):
        key_mean = 'mean_'+experiment    
        key_CI  = 'CI_'+experiment
        stat_experiments.update({key_mean:None, key_CI:None})
        executions = []    
        for file in os.listdir(path+experiment):
           with open(path+experiment+'/'+file, newline='') as csvfile:
               reader = csv.reader(csvfile, delimiter=',')
               for row in reader:
                   if row[0][:6] == 'output':
                       executions.append(float(row[2]))
        mean = np.mean(executions)
        CI=st.t.interval(alpha=0.95, df=len(executions)-1, loc=mean,
                         scale=st.sem(executions))
        stat_experiments[key_mean] = mean
        stat_experiments[key_CI]   =CI[1]-CI[0]
    return stat_experiments


def get_results_for_core_type(core_type,stat_experiments):
    
    results = dict({'BM-mean':[],'CN-mean':[],'VM-mean':[] ,'VMCN-mean':[],
                    'BM-CI':[],'CN-CI':[],'VM-CI':[] ,'VMCN-CI':[]})
    for experiment,stat_value in stat_experiments.items():
        
        stat_type = experiment.split('_')[0]
        core_no = int(experiment.split('_')[1].split('-')[0])
        platform = experiment.split('_')[1].split('-')[2]
        key = platform+'-'+stat_type
        
        if core_type == experiment.split('_')[1].split('-')[1]:
            
            results[key].append([core_no,stat_value])
    for key,value in results.items():       
        if core_type !='pinned' or key.split('-')[0] != 'BM':            
            value = np.array(value)    
            results[key] = value[value[:,0].argsort()]
            
    return results

def plot_core_type(core_type,results):    
    
    labels = ['BM', 'CN', 'VM', 'VMCN']
    xticks  = [2,4,8,16,32,64]
    markers = ['o', 'v' , 'x',4]
    linestyles = ['--',(0, (3, 1, 1, 1)),'-.','-']
    colors = ['darkred','blue','darkgreen','goldenrod']
    fig, ax = plt.subplots()
    ax.set_xticks(xticks)
    
    if core_type != 'pinned':
        ax.errorbar(results['BM-mean'][:,0],results['BM-mean'][:,1],
                yerr = results['BM-CI'][:,1],
                capsize=5, elinewidth=2,linestyle=linestyles[0],color = colors[0],
                marker=markers[0],linewidth=2,markersize=8,label=labels[0]) 
    
    ax.errorbar(results['CN-mean'][:,0],results['CN-mean'][:,1],
                yerr = results['CN-CI'][:,1],
                capsize=5, elinewidth=2,linestyle=linestyles[1],color = colors[1],
                marker=markers[1],linewidth=2, markersize=8,label=labels[1]) 
    
    ax.errorbar(results['VM-mean'][:,0],results['VM-mean'][:,1],
                yerr = results['VM-CI'][:,1],
                capsize=5, elinewidth=2,linestyle=linestyles[2],color = colors[2],
                marker=markers[2],linewidth=2, markersize=8,label=labels[2]) 
    
    ax.errorbar(results['VMCN-mean'][:,0],results['VMCN-mean'][:,1],
                yerr = results['VMCN-CI'][:,1],
                capsize=5, elinewidth=2,linestyle=linestyles[3],color = colors[3],
                marker=markers[3],linewidth=2, markersize=8,label=labels[3])     
    
    ax.set_ylabel('Inference Time [sec]',fontsize=14)
    ax.set_xlabel('Number of CPU Cores',fontsize=14)
    ax.grid('minor') 
    ax.legend()     
    fig.tight_layout()    
    plt.savefig(core_type+'_vs_no_of_cores'+'.pdf')