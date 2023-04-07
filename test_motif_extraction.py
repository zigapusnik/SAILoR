import csv	
import os
import numpy as np


import motif_extraction 

#construct adjacency matrix based on provided gold standard tsv file	   
def construct_adj_matrix(size, file):  
	adj_matrix = np.zeros((size, size)).astype(int) 	
	with open(file) as fd:
		rd = csv.reader(fd, delimiter="\t", quotechar='"') 		
		genes = {}
		in_tot = 0
		for row in rd:
			src = row[0]        
			dst = row[1]       		 		
			edge = int(row[2])   
			 
			if src in genes:
				i = genes.get(src)
			else:
				i = in_tot
				genes[src] = i
				in_tot = in_tot +  1
			if dst in genes:
				j = genes.get(dst)
			else:
				j = in_tot
				genes[dst] = j
				in_tot = in_tot +  1
				
			if edge == 1:
				adj_matrix[i, j] = 1   
				#adj_matrix[j, i] = 1   
	return adj_matrix  	

matrices = [] 

size = 16

for ex in range(1, 11):
    example = ex
    test_folder = os.path.join("..", "data", "EcoliExtractedNetworks", str(size))      
    test_file_name = "Ecoli-" + str(example) + "_goldstandard.tsv"  
    test_file_path = os.path.join(test_folder, test_file_name)  	

    adj_mat = construct_adj_matrix(size, test_file_path)
    matrices.append(adj_mat)   
    print(adj_mat)    

motif_extraction.extract_motifs(matrices, depth=3)  