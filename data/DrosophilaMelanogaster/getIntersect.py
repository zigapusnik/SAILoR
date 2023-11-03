import csv
import os 
import pandas as pd 

folder_path = os.path.join(".")   
time_series_file_name = "cpm_updated.txt"    

subnetwork_genes_file_name = "subnetwork_genes.txt" 
subnetwork_genes_intersect_file_name = "subnetwork_genes_intersect.txt"  

time_series_genes = set()
with open(os.path.join(folder_path, time_series_file_name)) as file_time_series:
    next(file_time_series) 
    for line in csv.reader(file_time_series, delimiter="\t"):
        time_series_genes.add(line[0])


subnetwork_genes_df = pd.read_csv(os.path.join(folder_path, subnetwork_genes_file_name), sep="\t") 
subnetwork_genes_set = set(subnetwork_genes_df["GeneID"]) 


genes_intersect = time_series_genes.intersection(subnetwork_genes_set) 
print("Number of genes in subnetwork:" + str(len(subnetwork_genes_set))) 
print("Number of genes in intersect: " + str(len(genes_intersect)))   

with open(os.path.join(folder_path, subnetwork_genes_intersect_file_name), "w") as intersect_genes_file: 
    for gene in genes_intersect:
        intersect_genes_file.write(gene + "\n")  