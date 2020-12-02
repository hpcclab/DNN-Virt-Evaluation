from utils import *

# Path to the directory of the experimental results
path_to_results ='../FaceNet/'

# read_data: it is a function that reads files in the directory and returns
# a dictionary contains the mean and CI for each experiment
stat_experiments = read_data(path_to_results)

# The result of experiments for a specific core type (i.e GPU, pinned, and 
# vanilla) is given by get_results_for_core_type. 
# The inputs of this function is the desired core type (e.g 'GPU') and the
# experimental results (which is a dictionary)
results_GPU = get_results_for_core_type('GPU',stat_experiments)

# plot the results of exeuting tasks on GPU for different number of CPU cores
plot_core_type('GPU',results_GPU)

results_vanilla = get_results_for_core_type('vanilla',stat_experiments)
plot_core_type('vanilla',results_vanilla)

results_pinned = get_results_for_core_type('pinned',stat_experiments)
plot_core_type('pinned',results_pinned)
