import numpy as np 
import os
import pandas as pd 

data_dir = os.path.join(".", "data", "DrosophilaMelanogaster") 
prior_file_name = "NetREX_female_reference_network.txt" 

prior_network_path = os.path.join(data_dir, prior_file_name)

prior_network_df = pd.read_csv(prior_network_path, header=None, index_col=False, sep="\t") 

names_1 = set(prior_network_df[0]) 
names_2 = set(prior_network_df[1]) 

names = list(names_1.union(names_2)) 

name_index_dict = {}
i = 0

for name in names: 
    if not name in name_index_dict:
        name_index_dict[name] = i
        i = i + 1 


n = len(names)

network = np.zeros((n,n))    

for i in range(len(prior_network_df)):  
    tf = prior_network_df.loc[i, 0]
    target = prior_network_df.loc[i, 1] 

    tf_ind = name_index_dict[tf]
    target_ind= name_index_dict[target] 

    network[tf_ind, target_ind] = 1 

column_sum = network.sum(axis=0)
print(column_sum) 

row_sum = network.sum(axis=1)
print(row_sum) 

print(np.where(column_sum == 0, 1, 0) + np.where(row_sum == 0, 1, 0))

print(network.sum()) 
