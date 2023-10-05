from copy import copy
import itertools
import matplotlib.pyplot as plt  
import networkx as nx 
import numpy as np 
import os  
import pandas as pd 
import re   
import sklearn.metrics as metrics      

from inferBNsDynamics import getBooleanExpressions, getTargetGenesEvalExpressions 

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN) 


#calculate metrices for given adjacency matrices	
def getMetrics(edgesTrue, edgesPred, networkSize):   

    adj_true = np.zeros((networkSize, networkSize))   
    adj_pred = np.zeros((networkSize, networkSize)) 

    for edge in edgesTrue: 
        adj_true[edge[0]-1, edge[1]-1] = 1
        adj_true[edge[1]-1, edge[0]-1] = 1   

    for edge in edgesPred:       
        adj_pred[int(edge[0])-1, int(edge[1])-1] = 1
        adj_pred[int(edge[1])-1, int(edge[0])-1] = 1        
		
    #y_true = adj_true.flatten()    
    #y_pred = adj_pred.flatten() 
    y_true = adj_true[np.triu_indices(networkSize)]           
    y_pred = adj_pred[np.triu_indices(networkSize)]      



    accuracy = metrics.accuracy_score(y_true, y_pred) 
    precision = metrics.precision_score(y_true, y_pred) 
    recall = metrics.recall_score(y_true, y_pred) 
    f1 = metrics.f1_score(y_true, y_pred) 
    mcc = metrics.matthews_corrcoef(y_true, y_pred)

    TP,FP,TN,FN = perf_measure(y_true, y_pred) #works same as metrics            

    TPR = float(TP)/(TP + FN) 
    TNR = float(TN)/(TN + FP)      

    bm = TPR + TNR - 1.     

    print("Acurracy: " + str(accuracy))
    print("Precision: " + str(precision))
    #print("My precision: " + str(TP/float(TP + FP)))    
    print("Recall: " + str(recall))
    #print("My recall: " + str(TP/float(TP + FN)))     	 
    print("F1 score: " + str(f1))    

    return {"Accuracy": accuracy, "Precision": precision,"Recall": recall,"F1": f1,"MCC": mcc,"BM": bm}                                  


def getEdgesGold(timeSeriesFile, goldStandardFile): 
    df = pd.read_csv(timeSeriesFile, sep="\t", decimal=",")  
    df = df.apply(pd.to_numeric)   
    df = df.dropna() 
    df = df.drop(columns=["Time"]) 
	
    columns = df.columns 
    columnNumbers = np.arange(1, len(df.columns)+1) 
    geneNums = {}

    for c, n in zip(columns,columnNumbers): 
        geneNums[c] = n               
    
    edges = []   

    with open(goldStandardFile) as goldStructureFile:   
        lines = goldStructureFile.read().splitlines()  

    #print(timeSeriesFile)
    #print(goldStandardFile) 

    for line in lines:  
        items = re.findall(r'[^\s]+', line)   
        if items[2] == '1':   
            edges.append((geneNums[items[1]], geneNums[items[0]])) #switch positions to target <- regulator                

    return edges         

def getEdges(DNfilePath):
    print(DNfilePath) 

    edges = [] #list of tuples 
    with open(DNfilePath) as structureFile:
        lines = structureFile.read().splitlines()     

    for line in lines:     
        edge = tuple(re.findall(r'[0-9]+', line))           
        edges.append(edge)          
    return edges       

def plotDirectedNetwork(edges):   
    G = nx.DiGraph()     
    G.add_edges_from(edges)     
    G = G.reverse(copy=True)   
    pos = nx.spring_layout(G) # positions for all nodes       
    nx.draw(G, pos, with_labels=True)   
    plt.show()     


def getStructure(BNfilePath, DNfilePath, getSign=False):   
    bool_expressions = getBooleanExpressions(BNfilePath)  
    target_genes, eval_expressions = getTargetGenesEvalExpressions(bool_expressions)   
    lines = [] 

    for targetGene, evalExpr in zip(target_genes, eval_expressions): 
        #print(evalExpr)      
        regs = list(set(re.findall(r"(Gene\d+)", evalExpr)))     
        regNums = {}  
        regNum = len(regs)        
        actRep = {}  
        for reg in regs:
           reR = re.match(r"Gene(\d+)", reg) 
           actRep[reg] = {"activates":0, "represses":0}   
           regNums[reg] = int(reR.group(1))  
         
        #print(targetGene)   
        #print(regs)     
        #print(regNums)      
        
        if getSign:
            #generate truth table for given Boolean expression and regulators
            #iterate over all input vectors  
            for iv in range(2**regNum):  
                ivBinary = [int(x) for x in ("{:0"+ str(regNum) + "b}").format(iv)]   
                #print(ivBinary)  

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
                lines.append(str(targetGene) + " <- " + str(regNums[reg]) + sign + "\n")    
        else: 
            for regName, regNum in regNums.items():
                lines.append(str(targetGene) + " <- " + str(regNum) + "\n")          
            
    with open(DNfilePath, "w+") as fileStructureHandle:     
        fileStructureHandle.writelines(lines)          

if __name__ == "__main__":  
    #set working directory
    os.chdir(os.path.dirname(__file__))         
    #organisms = ["EcoliExtractedNetworks"] 
    organism = "Ecoli"     
    results_name = "EcoliExtractedNetworks"
    data_folder = "EcoliExtractedNetworks"    
    methods = ["SAILoR"]                    
    networkSize = 64                                                                                

    for method in methods: 
        results_path = os.path.join(".", "results", results_name, str(networkSize), method) 
        data_path = os.path.join(".", "data", data_folder, str(networkSize))     
        
        all_results = {}    

        #iterate over all files 
        for BNfileName in os.listdir(results_path):
            if BNfileName.endswith("dynamics.tsv"):
                print(BNfileName)  
                BNfilePath = os.path.join(results_path, BNfileName)       
                reM = re.match(results_name + "-([0-9]+)_([0-9]+)_dynamics.tsv", BNfileName)    
                if reM:  
                    networkNum = int(reM.group(1))  
                    crossIteration = int(reM.group(2))   
                    DNfilePath = os.path.join(results_path, organism + "-" + str(networkNum) + "_" + str(crossIteration) + "_structure.tsv")  
                    timeSeriesFilePath = os.path.join(data_path, organism +"-" + str(networkNum) +"_dream4_timeseries.tsv")           
                    goldStandardFilePath = os.path.join(data_path, organism+"-" + str(networkNum) +"_goldstandard.tsv")   

                    if networkNum not in all_results.keys():
                        all_results[networkNum] = pd.DataFrame(columns = ["Accuracy", "Precision", "Recall", "F1", "MCC", "BM"])                   

                    if not os.path.exists(DNfilePath):      
                        #convert Boolean network to directed graph and save structure       
                        getStructure(BNfilePath, DNfilePath)  

                    #print(DNfilePath)
                    #print(timeSeriesFilePath)
                    #print(goldStandardFilePath)       
                    edges = getEdges(DNfilePath)  
                    #display directed network                       
                    #plotDirectedNetwork(edges)   
                    edgesGold = getEdgesGold(timeSeriesFilePath, goldStandardFilePath)     
                    my_metrics = getMetrics(edgesGold, edges, networkSize)   
                    all_results[networkNum] = all_results[networkNum].append(my_metrics, ignore_index = True)  
                    #print(my_metrics)  
                    #print(all_results[networkNum])    

        print(all_results.items()) 
        #iterate over all dataframes and write them into seperate files 
        for number_key, df_value in all_results.items():  
            df_value.to_csv(os.path.join(results_path, "results_structure_network_" + str(number_key) + ".tsv"), sep="\t", index=False)           

