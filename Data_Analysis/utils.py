import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import csv
import os
import scipy.stats as st


def confidence(alpha, data ):
    n = len(data)    
    data_mean = np.mean(data)
    data_std = np.std(data,ddof=1)        
    cut_off = data_std * 3
    lower, upper = data_mean - cut_off, data_mean + cut_off
    outliers_removed = [x for x in data if x > lower and x < upper]    
    mean = np.mean(outliers_removed)     
    s = np.std(outliers_removed,ddof=1)
    se = s/np.sqrt(len(outliers_removed))
    t = st.t.ppf((1+alpha)/2.0,len(outliers_removed)-1)    
    CI = [mean - t * se , mean + t* se]  
    
    return mean, CI
    
    
    

def read_data(path,app): 
    stat_experiments = dict()    
    for experiment in os.listdir(path):
        key_mean = 'mean_'+experiment    
        key_CI  = 'CI_'+experiment
        stat_experiments.update({key_mean:None, key_CI:None})
        executions = [] 
        exec_outliers_removed=[]
        for file in os.listdir(path+experiment):
           with open(path+experiment+'/'+file, newline='') as csvfile:
               reader = csv.reader(csvfile, delimiter=',')    
               
               for row in reader:
                   if app == 'FaceNet':
                       if row[0][:5] != 'video':
                           execution = float(row[4])
                           executions.append(execution)
    
                   elif app == 'Yolo':
                       if row[0][:6] == 'output':                           
                           execution = float(row[2])                           
                           executions.append(execution)  
                           
        mean , CI = confidence(0.95,executions)        
        stat_experiments[key_mean] = mean
        stat_experiments[key_CI]   =0.5*(CI[1]-CI[0])
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

def plot_core_type(core_type,results,app):    
    
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
    if app == 'FaceNet':
        name = core_type+'FaceNet.pdf'
    elif app == 'Yolo':
        name = core_type+'Yolo.pdf'
    plt.savefig('./graphs/'+name)