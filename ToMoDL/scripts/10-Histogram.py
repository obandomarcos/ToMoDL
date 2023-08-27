import wandb
import matplotlib.pyplot as plt
import numpy as np

user_project_name = 'omarcos/deepopt/'
artifact_name_dpf = 'run-3fv8g9iw-table_dpf:v0'
artifact_name_part = 'run-3fv8g9iw-table_part:v0'

acceleration_factor = 22
testing_name_group = 'x{}_test2'.format(acceleration_factor)
run = wandb.init(project = 'deepopt', group = testing_name_group, job_type = 'Dataset Evaluation + K-Folding', name = '3 fold')

table_dpf = run.use_artifact(user_project_name+artifact_name_dpf, type='run_table').get('table_dpf')
table_part = run.use_artifact(user_project_name+artifact_name_part, type='run_table').get('table_part')

k_folds = np.arange(4).astype(int)

fig_dpf, ax_dpf = plt.subplots(1,1, figsize = (8,6))
fig_part, ax_part = plt.subplots(1,1, figsize = (8,6))

axs = [ax_dpf, ax_part]
label_dpf = ['1-day post fertilisation', '3-day post fertilisation', '5-day post fertilisation']
label_part = ['Head', 'Body', 'Upper tail', 'Lower tail']

labels = [label_dpf, label_part]

tables = [table_dpf, table_part]

for ax_num, ax in enumerate(axs):

    table = tables[ax_num]
    label = labels[ax_num]

    for k_fold, qty in table.iterrows():
        
        if k_fold == 0 and ax_num == 0:

            ax.bar(k_fold-0.2, qty[0], 0.2, label = label[0], color = 'red')
            ax.bar(k_fold, qty[1], 0.2, label = label[1], color = 'green')
            ax.bar(k_fold+0.2, qty[2], 0.2, label = label[2], color = 'blue')
            
        elif k_fold == 0 and ax_num == 1:

            ax.bar(k_fold-0.15, qty[0], 0.1, label = label[0], color = 'red')
            ax.bar(k_fold-0.05, qty[1], 0.1, label = label[1], color = 'green')
            ax.bar(k_fold+0.05, qty[2], 0.1, label = label[2], color = 'blue')
            ax.bar(k_fold+0.15, qty[3], 0.1, label = label[3], color = 'orange')

        elif k_fold != 0 and ax_num == 1:

            ax.bar(k_fold-0.15, qty[0], 0.1, color = 'red')
            ax.bar(k_fold-0.05, qty[1], 0.1, color = 'green')
            ax.bar(k_fold+0.05, qty[2], 0.1, color = 'blue')
            ax.bar(k_fold+0.15, qty[3], 0.1, color = 'orange')
        
        else:

            ax.bar(k_fold-0.2, qty[0], 0.2, color = 'red')
            ax.bar(k_fold, qty[1], 0.2, color = 'green')
            ax.bar(k_fold+0.2, qty[2], 0.2, color = 'blue')

    ax.legend()
    ax.set_xticks(k_folds)
    
    if ax_num == 0:
        ax.set_yticks(np.arange(4))
        ax.set_title('Test dataset distribution - Fish dpf - x{}'.format(acceleration_factor))
    else:
        ax.set_yticks(np.arange(4))
        ax.set_title('Test dataset distribution - Fish part - x{}'.format(acceleration_factor))

    ax.set_xlabel('# of Fold')
    ax.set_ylabel('# of dataset')

wandb.log({'parts_distribution': wandb.Image(fig_part)})
wandb.log({'dpf_distribution': wandb.Image(fig_dpf)})

fig_part.savefig('./logs/parts_x{}.pdf'.format(acceleration_factor), bbox_inches = 'tight')
fig_dpf.savefig('./logs/dpf_x{}.pdf'.format(acceleration_factor), bbox_inches = 'tight')



