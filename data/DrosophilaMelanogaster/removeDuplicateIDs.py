import csv
import os


#folder_path = os.path.join(".", "data", "Drosophila melanogaster") 
folder_path = os.path.join(".")  
file_name = "NetREX_female_prediciton_ranks_full_names_ids_validation_table.txt"
 
items = {}
duplicates = []


with open(os.path.join(folder_path, file_name)) as file:
    next(file) 
    for line in csv.reader(file, delimiter="\t"): 
        item, id, symbol = line

        if item in items:
            items[item] = items[item] + 1
        else:
            items[item] = 1

for item, count in items.items():
    if count > 1:
        duplicates.append(item) 

lines = []

with open(os.path.join(folder_path, file_name)) as file:
    for item, id, symbol in csv.reader(file, delimiter="\t"): 

        if item not in duplicates:
            lines.append(item + "\t" + id + "\n")      

i = 0
with open(os.path.join(folder_path, "found.txt"), "a") as file:
    for line in lines:
        i = i + 1
        file.write(line) 

print(str(i) + " additional inseritons")  