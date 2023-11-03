import csv
import os
import pandas as pd  

genes_file_name = "subnetwork_genes_intersect.txt"  
cpms_file_name = "cpm_updated.txt"   

folder_path = os.path.join(".")   
cpm_path = os.path.join(folder_path, cpms_file_name)  

genes = set()

with open(os.path.join(folder_path, genes_file_name)) as gene_list_file:
    for line in csv.reader(gene_list_file): 
        genes.add(line[0]) 

cpm_df = pd.read_csv(cpm_path, delimiter="\t", index_col=0)

print(cpm_df) 

full_index = cpm_df.index 
my_index = [] 

for id in full_index: 
    if id in genes: 
        my_index.append(id) 

cpm_df = cpm_df.filter(items=my_index, axis=0) 

all_columns = list(cpm_df.columns) 
types = ["V", "C"]
for type in types:
    type_columns = []

    for column in all_columns:
        my_type = column.split("_")[1]
        if  my_type == type:
            type_columns.append(column) 
    
    cpm_type_df = cpm_df[type_columns] 
    cpm_type_df.to_csv(os.path.join(folder_path, "cpm_" + type + ".txt"), sep="\t")     