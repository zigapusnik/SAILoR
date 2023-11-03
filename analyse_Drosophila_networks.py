import inferBNsDynamics
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt 
import networkx as nx
import numpy as np 
import os 
import pandas as pd 
import re  
import run_case_study_Drosophila

from functools import reduce
from sklearn.cluster import KMeans  

def getStructure(BNfilePath, DNfilePath, gene_dict = {}, getSign=True):     
    bool_expressions = inferBNsDynamics.getBooleanExpressions(BNfilePath)  
    target_genes, eval_expressions = getTargetGenesEvalExpressions(bool_expressions)   
    lines = []  

    for targetGene, evalExpr in zip(target_genes, eval_expressions): 
        #print(evalExpr)      
        regs = list(set(re.findall(r"(FBgn\d+)", evalExpr)))     
        regNum = len(regs)        
        actRep = {}  
        for reg in regs:
           actRep[reg] = {"activates":0, "represses":0}    
         
        #print(targetGene)   
        #print(regs)     
        #print(regNums)      
        
        if getSign:
            #generate truth table for given Boolean expression and regulators
            #iterate over all input vectors  
            for iv in range(2**regNum):  
                ivBinary = [int(x) for x in ("{:0"+ str(regNum) + "b}").format(iv)]   

                #initialize inputs
                for bit, reg in zip(ivBinary, regs): 
                    exec(reg + " = " + str(bit))  
                
                #evaluate expression
                val = eval(evalExpr)     

                #count activations and repressions 
                for bit, reg in zip(ivBinary, regs):    
                    actRepReg = actRep[reg]
                    if bit == val:
                        #activation 
                        actRepReg["activates"] = actRepReg["activates"] + 1   
                    else: 
                        #repression 
                        actRepReg["represses"] = actRepReg["represses"] + 1  

            #generate all +/- connections for target gene            
            for reg, actRepReg in actRep.items():
                sign = "(+)"
                if actRepReg["represses"] > actRepReg["activates"]:
                    sign = "(-)"
                lines.append(str(targetGene) + " <- " + str(reg) + sign + "\n")    
        else: 
            for reg in regs:
                lines.append(str(targetGene) + " <- " + str(reg) + "\n")           
            
    with open(DNfilePath, "w+") as fileStructureHandle:     
        fileStructureHandle.writelines(lines)           

def getEdges(DNfilePath, gene_nums_dict, gene_names):

    adj_matrix = np.zeros((len(gene_nums_dict), len(gene_nums_dict)))

    G = nx.DiGraph()

    with open(DNfilePath) as structureFile:
        lines = structureFile.readlines()     

    for line in lines:     
        edge_args = line.split(" ")
        edge = (edge_args[0], edge_args[2][:-4])
        edge_sign = int(edge_args[2][-3:-2] + "1")  
        gene_num_0 = gene_nums_dict[edge[0]]
        gene_num_1 = gene_nums_dict[edge[1]]

        adj_matrix[gene_num_0, gene_num_1] = edge_sign  
        G.add_edge(gene_names[gene_num_0], gene_names[gene_num_1], weight=edge_sign)

    return adj_matrix, G              

def plotNetwork(network_path, dicrected_network_path, gene_nums_dict, gene_names, gene_roles_dict, unique_roles):
    getStructure(network_path, dicrected_network_path)
    edges_matrix, G = getEdges(dicrected_network_path, gene_nums_dict, gene_names) 

    colors = []
    color_map = plt.get_cmap("Set3") 
    for node in G.nodes():
        role = gene_roles_dict[node]
        color = color_map(unique_roles.index(role)) 
        colors.append(color) 

    edge_colors = []
    for edge in G.edges(): 
        if G[edge[0]][edge[1]]["weight"] == 1:
            edge_colors.append("black")
        else: 
            edge_colors.append("red")   

    fig, ax = plt.subplots(figsize=(10,10))
    #G = nx.from_numpy_array(edges_matrix, create_using=nx.DiGraph)  

    custom_lines = [Line2D([0], [0], color=color_map(index), lw=4) for index in range(len(unique_roles))]          
    ax.legend(custom_lines, unique_roles, loc="lower right") 

    pos = nx.nx_agraph.graphviz_layout(G, prog="neato") #dot za SAILoR
    nx.draw(G, pos=pos, node_size=500, node_color = colors, edge_color=edge_colors, with_labels = True, width=1.5, arrowsize=20) #node_color="#1a1a1a"  



    fig.tight_layout()  
    plt.show()  

    return None   

def getTargetGenesEvalExpressions(bool_expressions):  
    target_genes = [] 
    eval_expressions = []  
    for k in range(0, len(bool_expressions)):  
        expr = bool_expressions[k]   
        gene_name = re.search(r'.+', expr[:expr.find(" = ")]).group()
        eval_expr =  expr[expr.find("= ") + 2:]
        target_genes.append(gene_name)   
        eval_expressions.append(eval_expr) 
    return target_genes, eval_expressions 
 
def evalBooleanModel(model_path, test_series): 
    gene_indices = {name:index for index, name in enumerate(test_series.columns.tolist())} 
    rows, columns = test_series.shape 
    simulations = np.array(test_series.copy())  #set initial states      
    bool_expressions = inferBNsDynamics.getBooleanExpressions(model_path)       
    target_genes, eval_expressions = getTargetGenesEvalExpressions(bool_expressions)  
    gene_names =  test_series.columns.tolist()

	#intialize genes to false
    for gene_name in gene_names:     
        exec(gene_name + " = False")         


    for time_stamp in range(1, rows):     
		#dynamically allocate variables  
        for target_gene in target_genes:  
            gene_num = gene_indices[target_gene]
            exec(target_gene + " = " + str(simulations[time_stamp - 1, gene_num]))     
     		
		#initialize simulation to false  
        ex_row = [0]*columns   
		#evaluate all expression  
        for k in range(0, len(bool_expressions)):      
            gene_name = target_genes[k]    
            eval_expr = eval_expressions[k]              
            gene_num = gene_indices[gene_name] 
               
            ex_row[gene_num] = int(eval(eval_expr))         	    	 	   
         
        simulations[time_stamp,:] = ex_row          

    return simulations    

def iterativeKmeans(data, d=3):     
    data = np.array(data)                
    data = np.reshape(data, (-1,1)) #reshape to array with one feature  
    while d > 0:       
        clusters = pow(2, d) 
        kmeans = KMeans(n_clusters=clusters, random_state=0).fit(data)     
        data = kmeans.cluster_centers_[kmeans.labels_]   
        d = d - 1        
    #binarize     	
    boolV = kmeans.cluster_centers_[0,0] > kmeans.cluster_centers_[1,0] 
    centers = np.array([int(boolV), int(not boolV)])      
    return pd.Series(centers[kmeans.labels_].tolist())       

def binarise(timeSeriesDf):
    columns = timeSeriesDf.columns.tolist()  
    columns.remove("Time")            
    data = timeSeriesDf.loc[:, columns]   
    binarised_df = data.apply(iterativeKmeans, axis=0)   
    binarised_df = binarised_df.astype(int) 
    binarised_df = pd.concat([binarised_df, timeSeriesDf.loc[:, ["Time"]]], axis=1)              
    return binarised_df 

results_dir = os.path.join(".", "results", "DrosophilaMelanogaster") 
data_dir = os.path.join(".", "data", "DrosophilaMelanogaster")        
gene_ids_path = os.path.join(".", "data", "DrosophilaMelanogaster", "subnetwork_genes_intersect.txt")  
gene_names_path = os.path.join(".", "data", "DrosophilaMelanogaster", "subnetwork_genes_names.txt")  

roles_df = pd.read_csv(os.path.join(".", "data", "DrosophilaMelanogaster", "subnetwork_genes.txt"), sep="\t")
unique_roles = reduce(lambda re, x: re + [x] if x not in re else re, list(roles_df["Function"]), [])   

with open(gene_ids_path, "r") as gene_ids_file: 
    lines = gene_ids_file.readlines()  

i = 0 
gene_nums_dict = {}  
gene_roles_dict = {} 
gene_names = [] 
with open(gene_names_path, "r") as gene_names_file:  
    lines = gene_names_file.readlines() 

i = 0
for line in lines:
    id_name = line[:-1].split("\t")  
    gene_id = id_name[0] 
    name = id_name[1] 
    gene_names.append(name)       
    role = roles_df[roles_df["GeneID"] == gene_id]["Function"].iloc[0]   
    gene_roles_dict[name] = role   
    gene_nums_dict[gene_id] = i       
    i = i + 1 

groups = ["V", "C"]          
for group in groups:  

    network_file_name = "cpm_" + group + "_network.txt"    
    network_path = os.path.join(results_dir, network_file_name)   
    binarised_file_learn_name = "Dm_timeseries_" + group + "_cpm_binarised.tsv"  

    directed_network_path = os.path.join(results_dir, "cpm_" + group + "_directed_network.txt")    

    binarised_learn_path = os.path.join(data_dir, binarised_file_learn_name)
    binarised_test_df = pd.read_csv(binarised_learn_path, sep="\t")  

    binarised_test_0_df = binarised_test_df.loc[binarised_test_df["Experiment"] == 0] 
    binarised_test_1_df = binarised_test_df.loc[binarised_test_df["Experiment"] == 1]

    plotNetwork(network_path, directed_network_path, gene_nums_dict, gene_names, gene_roles_dict, unique_roles)         

    columns = binarised_test_df.columns.tolist() 
    columns.remove("Time")   
    columns.remove("Experiment")       

    binarised_test_0_df = binarised_test_0_df[columns]  
    binarised_test_1_df = binarised_test_1_df[columns]   

    print(binarised_test_0_df.shape) 
    print(group)     

    time_points = binarised_test_0_df.shape[0] - 1 
    gene_num = binarised_test_0_df.shape[1]
    total = time_points*gene_num

    simulations = evalBooleanModel(network_path, binarised_test_0_df)   
    erros = np.absolute(np.subtract(simulations, np.array(binarised_test_0_df)))  

    individual_errors = erros[1:time_points+1,:]
    individual_errors = individual_errors.sum(axis=0)/time_points 
    print(individual_errors)

    total_errors = erros.sum()          
    print(1 - total_errors/total)    
    print(1 - np.sum(individual_errors)/gene_num)

    simulations = evalBooleanModel(network_path, binarised_test_1_df) 
    erros = np.absolute(np.subtract(simulations, np.array(binarised_test_1_df)))
    individual_errors = erros[1:time_points+1,:]
    individual_errors = individual_errors.sum(axis=0)/time_points

    print(individual_errors) 

    total_errors = erros.sum()       
    print(1 - total_errors/total)    
    print(1 - np.sum(individual_errors)/gene_num)        
 
    print("---------------------------------------------------------")    