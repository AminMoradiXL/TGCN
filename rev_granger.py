import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
import seaborn as sns
import matplotlib.pyplot as plt

agent_name = 'PLS'

path_ = f'Dataset/Attack_dataset/rte_case14_realistic/rho_attack_rte_case14_realistic_{agent_name}.csv'
data = pd.read_csv(path_)  
data.drop(columns = 'Unnamed: 0', inplace = True)

data_with_noise = data + np.random.normal(0, 0.1, size=data.shape)

p_values = np.zeros((data.shape[1], data.shape[1])) 

for i, feature1 in enumerate(data.columns):
    for j, feature2 in enumerate(data.columns):
        # if i != j: 
        if True: 
            combined_data = np.column_stack((data_with_noise[feature1].values, data_with_noise[feature2].values))
            
            result = grangercausalitytests(combined_data, maxlag=6)
            
            p_value = result[2][0]['ssr_ftest'][1]
            p_values[i, j] = p_value

# Step 4: Plot Heatmap
plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(p_values, xticklabels=data.columns, yticklabels=data.columns, cmap='RdBu', annot=False)
plt.xlabel('Powerline ID')
plt.ylabel('Powerline ID')
colorbar = heatmap.collections[0].colorbar
colorbar.set_label('Granger Causality', rotation=270, labelpad=20)
# plt.savefig(f'Plotting_new_figs/correlation and granger/{agent_name}_Granger.png',dpi=600)
plt.show()

# plt.figure(figsize=(10, 8))
# heatmap = sns.heatmap(np.corrcoef(data.T), xticklabels=data.columns, yticklabels=data.columns, cmap='RdBu', annot=False)
# plt.xlabel('Powerline ID')
# plt.ylabel('Powerline ID')
# colorbar = heatmap.collections[0].colorbar
# colorbar.set_label('Correlation', rotation=270, labelpad=20)
# # plt.savefig(f'Plotting_new_figs/correlation and granger/{agent_name}_Correlation.png',dpi=600)
# plt.show()

# how a past history of a signal is effecting other one's future