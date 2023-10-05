import pandas as pd
import numpy as np  
import os 

from Bio import SeqIO

def getN(raw_reads_df, cpm_df):
    N_df = raw_reads_df.div(cpm_df)*pow(10, 6)   
    #Ns = list(N_df.iloc[0,:])  
    return N_df   

def transalte_type(value, mappings):
    values = value.split("_")
    for map in mappings:
        if values[0] in map:
            values[0] = map[0]

    a,b,c = values 
    values = [c,a,b]  

    name = "_".join(values)  
    name = name.replace(".", "_")
    return name 

def translate_type_two(value, mappings):
    values = value.split("_")
    for map in mappings:
        if values[1] in map:
            values[1] = map[0] 
    name = "_".join(values)  
    return name     


folder_path = os.path.join(".")  
cpm_file_name = "cpm_updated.txt" 
raw_reads_name = "raw_corrected_counts_filtered_updated.txt"    
fragment_lengths_file_name = "fragment_lengths.txt" 
transcripts_fasta_file_name = "FlyBase_transcripts.fasta"   

cpm_file_path = os.path.join(folder_path, cpm_file_name) 
raw_reads_path = os.path.join(folder_path, raw_reads_name) 
fragment_lengths_path = os.path.join(folder_path, fragment_lengths_file_name) 
transcripts_fasta_path = os.path.join(folder_path, transcripts_fasta_file_name)  
transcripts_path = os.path.join(folder_path, transcripts_fasta_file_name)   

cmp_df = pd.read_csv(cpm_file_path, delimiter="\t", index_col=0)  
raw_reads_df = pd.read_csv(raw_reads_path, delimiter="\t", index_col=0)        

type_mappings = [("S", "SPmin", "SP-"),("V", "V", "V"),("C", "SPplus", "SP+")]   

tmp_file_name = os.path.join(folder_path, "tpm.txt")      

#check that gene order matches  
if not cmp_df.index.equals(raw_reads_df.index):
    print("CMP, raw_gene_counts series do not match")
    exit() 
#check that experiment order matches
if not cmp_df.columns.equals(raw_reads_df.columns):
    print("CMP, raw_gene_counts columns do not match")
    exit()   

#get numbers of reads
Ns = getN(raw_reads_df, cmp_df)

#get fragment lengths
fragment_lengths_df = pd.read_csv(fragment_lengths_path, delimiter="\t") 
fragment_lengths_df = fragment_lengths_df[["Experiment", "Avg"]]  
fragment_lengths_df["Experiment"] = fragment_lengths_df["Experiment"].map(lambda x: transalte_type(x, type_mappings))   
column_names = cmp_df.columns 
index_ = cmp_df.index 
fragment_lengths_list = [fragment_lengths_df.loc[fragment_lengths_df["Experiment"] == column, "Avg"].iloc[0] for column in column_names]
fragment_lengths_array = np.array([fragment_lengths_list]) 
fragment_lengths_df = pd.DataFrame(np.repeat(fragment_lengths_array, len(index_), axis=0), columns=column_names, index=index_)    

#get transcripts lengths 
transcripts_dict = {} #{gene: {exon: length}}
transcripts = SeqIO.parse(open(transcripts_path), "fasta") 
for transcript in transcripts: 
    description = transcript.description           
    values = description.split("; ")  

    name = values[0].split(" ")[0] 
    length_found = False
    parent_found = False

    for value in values: 
        field, val = value.split("=")
        if field == "length":
            length = val 
            length_found = True
        if field == "parent":
            parent = val 
            parent_found = True 
        if length_found and parent_found:
            #print(name + " " + parent + " " + length ) 
            break 

    if parent in transcripts_dict:
        sub_dict = transcripts_dict[parent]
    else:
        sub_dict = {}

    sub_dict[name] = length
    transcripts_dict[parent] = sub_dict  

mean_transcript_lengths = {} 
for parent, transcripts in transcripts_dict.items():  
    transcript_num = len(transcripts)
    sum_length = 0 
    for tr, lngth in transcripts.items(): 
        sum_length = sum_length + float(lngth)    

    mean_transcript_lengths[parent] = sum_length/transcript_num

genes_list = list(raw_reads_df.index) 
length_array = np.zeros((len(genes_list), 1))    

missing_lengths_indices = [] 
total_found = 0
total_sum = 0

i = 0
for gene in genes_list: 
    if gene in mean_transcript_lengths:
        length = mean_transcript_lengths[gene]
        length_array[i,0] = length  
        total_found = total_found + 1
        total_sum = total_sum + length 
    else:
        missing_lengths_indices.append(i) 
    i = i + 1

print("Number of missing transcripts: " + str(len(missing_lengths_indices))) 
mean_length_float = total_sum/total_found 
print("Mean transcript length: " + str(mean_length_float)) 
for ind in missing_lengths_indices:
    length_array[ind,0] = mean_length_float

experiment_columns = raw_reads_df.columns
lengths = np.repeat(length_array, len(experiment_columns), axis=1) 
gene_lengths_df = pd.DataFrame(lengths, index=genes_list, columns=experiment_columns)


#print(Ns) 
#print(raw_reads_df)  
#print(fragment_lengths_df) 

effective_lengths_df = gene_lengths_df - fragment_lengths_df.astype(float) + 1 

#print(gene_lengths_df) 
#print(fragment_lengths_df) 
#print(effective_lengths_df) 

fpkm_df = raw_reads_df.div(effective_lengths_df.multiply(Ns)) 
fpkm_df = fpkm_df*pow(10,9) 

#print(fpkm_df) 

fpkm_sum = fpkm_df.sum()   

#print(fpkm_sum)  

tpm = fpkm_df.div(fpkm_sum, axis=1) 
tpm = tpm*pow(10,6)  


#tpm is nan where cpm is 0, since number of reads is undefined
#in this case tpm should also be zero 
tpm = tpm.fillna(0)  

tpm.to_csv(tmp_file_name, sep="\t") 