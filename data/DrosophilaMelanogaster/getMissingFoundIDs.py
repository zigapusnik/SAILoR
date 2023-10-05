import csv
import os


#41467_2018_6382_MOESM4_ESM.txt females without ovaries 
#41467_2018_6382_MOESM5_ESM.txt males without testicles  

#folder_path = os.path.join(".", "data", "Drosophila melanogaster") 
folder_path = os.path.join(".")  
file_name = "NetREX_female_prediciton_ranks_full_names_ids.txt"  

gene_names = set() 

i = 0

unknowns = []
my_lines = []

with open(os.path.join(folder_path, file_name)) as file:
    next(file) 
    for line in csv.reader(file, delimiter="\t"):
        if line[1] == '':
            unknowns.append(line[0]) 
        else: 
            my_lines.append(line)


with open(os.path.join(folder_path, "missing.txt"), "w+") as file:
    for unknown in unknowns: 
        file.write(f"{unknown}\n") 

with open(os.path.join(folder_path, "found.txt"), "w+") as file:
    for line in my_lines: 
        file.write("\t".join(line) + "\n")   
