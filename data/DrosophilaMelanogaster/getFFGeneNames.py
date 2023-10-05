import csv
import os


#NetREX_female_prediciton_ranks_full.txt 

#folder_path = os.path.join(".", "data", "Drosophila melanogaster") 
folder_path = os.path.join(".")  
prior_network_file_name = "NetREX_female_prediciton_ranks_full.txt"

save_names_file_name = "NetREX_female_prediciton_ranks_full_names.txt"  
save_TFs_file_name = "NetREX_female_prediciton_ranks_TFs_names.txt" 

gene_names = set() 
TFs = set()  

#flybase api to get gene names
with open(os.path.join(folder_path, prior_network_file_name)) as prior_network_file:
    next(prior_network_file)  
    for line in csv.reader(prior_network_file, delimiter="\t"):
        TF = line[0]
        gene_names.add(TF)  
        gene = line[1]  
        gene_names.add(gene)    
        TFs.add(TF) 

gene_names_list = list(gene_names) 
print(gene_names_list)
print(len(gene_names_list))  

with open(os.path.join(folder_path, save_names_file_name), "w+") as save_names_file:
    for name in gene_names_list:
        save_names_file.write(f"{name}\n")  

with open(os.path.join(folder_path, save_TFs_file_name), "w+") as TFs_file:
    for TF in TFs:
        TFs_file.write(f"{TF}\n")   



