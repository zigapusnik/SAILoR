import csv
import os 

folder_path = os.path.join(".")   
prior_genes_file_name = "found.txt" 
time_series_file_name = "raw_corrected_counts_filtered_updated.txt"  
genes_file_name = "genes.txt"   
TFs_IDs_file_name = "TFs.txt"     
TFs_names_file_name = "NetREX_female_prediciton_ranks_TFs_names.txt"    

prior_file_name = "NetREX_female_prediciton_ranks_full.txt" 
prior_intersect_file_name = "NetREX_female_prediciton_ranks.txt" 

mapping = {}
prior_genes = set()
with open(os.path.join(folder_path, prior_genes_file_name)) as file_prior:
    for line in csv.reader(file_prior, delimiter="\t"):
        prior_genes.add(line[1]) 
        mapping[line[0]] = line[1] 

time_series_genes = set()
with open(os.path.join(folder_path, time_series_file_name)) as file_time_series:
    next(file_time_series) 
    for line in csv.reader(file_time_series, delimiter="\t"):
        time_series_genes.add(line[0])

genes_intersect = time_series_genes.intersection(prior_genes)
print("Number of genes in intersect: " + str(len(genes_intersect)))

tf_ids = set()  
tf_name_id_lines = []    
with open(os.path.join(folder_path, TFs_names_file_name)) as tf_names_file: 
    for line in csv.reader(tf_names_file):  
        tf_name = line[0] 
        tf_id = mapping.get(tf_name, "")  
        tf_ids.add(tf_id) 

        if tf_id in genes_intersect: 
            my_line = tf_id + "\n" 
            tf_name_id_lines.append(my_line)     

with open(os.path.join(folder_path, TFs_IDs_file_name), "w+") as tf_IDs_file:  
    for line in tf_name_id_lines: 
        tf_IDs_file.write(line)    

ranks_intersect = []
rank = 1
with open(os.path.join(folder_path, prior_file_name)) as ranked_full_file:
    next(ranked_full_file) 
    for line in csv.reader(ranked_full_file, delimiter="\t"):
        tf = line[0]
        gene = line[1]
    
        tf_id = mapping.get(tf, "") 
        gene_id = mapping.get(gene, "")   

        if tf_id in genes_intersect and gene_id in genes_intersect and tf_id in tf_ids and gene_id in tf_ids:
            ranks_intersect.append((tf_id, gene_id, rank))
            rank = rank + 1 


with open(os.path.join(folder_path, prior_intersect_file_name), "w+") as ranked_file:  
    ranked_file.write("TF\tGene\trank\n")
    for interaction in ranks_intersect: 
        ranked_file.write(interaction[0] + "\t" + interaction[1] + "\t" + str(interaction[2]) + "\n")   

with open(os.path.join(folder_path, genes_file_name), "w+") as genes_file:  
    for gene in genes_intersect: 
        genes_file.write(gene+"\n")       
