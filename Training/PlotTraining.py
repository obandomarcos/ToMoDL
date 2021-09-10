import pickle 
os.chdir('/home/marcos/DeepOPT/')
sys.path.append('Utilities/')
sys.path.append('OPTmodl/')
import matplotlib.pyplot as plt
import numpy as np
from folders_cluster

with open(results_folder+'FBP_error_projections.pkl', 'rb')
    train_info = pickle.load(f)

projections = [20, 30, 40, 60, 90, 120, 180]
error_test = []
error_test_std = []

error_fbp_test = []
error_fbp_test_std = []

for train_proj in train_info.values():
    
   error_test.append(train_proj['test'][-1])
   #error_test_std.append(train_proj['test_std'][-1])
   
   error_fbp_test.append(train_proj['test_fbp'][-1])
   error_fbp_test_std.append(train_proj['test_fbp_std'][-1])

#%% Plot
fig, ax = plt.subplots(1,2, figsize = (8,6))

ax.errorbar([i for i in range(len(projections))], error_test, yerr=error_test_std, fmt = '*', markersize = 8, label = 'Error MODL')
ax.grid()
ax.errorbar([i for i in range(len(projections))], error_fbp_test, yerr=error_fbp_test_std, fmt = '*', markersize = 8, label = 'Error FBP')
ax.set_xticks([i-1 for i in range(len(projections)+1)])
ax.set_xlim([-1, len(projections)])
ax.set_xlabel('# of projections', fontsize = 14)
ax.set_ylabel('MSE', fontsize = 14)
ax.set_xticklabels([0]+list(dict.fromkeys(projections)))

fig.savefig()

