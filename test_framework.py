from deap import creator, base, tools, algorithms

import framework
import matplotlib 
import matplotlib.pyplot as plt 
import multiprocessing   
import numpy as np 
import os  
import pickle   

def main():  
    subfolder = "DREAM4" # "EcoliExtractedNetworks"              
    net_nums = [1] #range(1,11)          
    net_size = 10 #32 #64 #10 #100                   
    
    imprvs = {}              
    dstcs = {}               
    mtrcs = {}    
    base = {}          

    for net_num in net_nums:
        dstcs[net_num] = []   
        mtrcs[net_num] = []
        base[net_num]  = []  
        imprvs[net_num] = []        

        scenario = subfolder + "_" + str(net_size) + "_" + str(net_num)   
        
        #10 repetitions 
        for runNum in range(10):   
 
            scenario = scenario + "_" + str(runNum) 
            savePath = os.path.join(".", "results", scenario) 

            distances, metrics, baseMetric, imprMetric = runFramework(net_num, net_size, subfolder, savePath = savePath)   

            dstcs[net_num].append(distances)  
            mtrcs[net_num].append(metrics) 
            base[net_num].append(baseMetric)     
            imprvs[net_num].append(imprMetric)     

    folder = os.path.join(".", "results", subfolder + "_" + str(net_size))       

    #create directory if does not exists    
    #if not os.path.exists(folder): 
    #    os.makedirs(folder)   
    #with open(os.path.join(folder, "dump_results.pkl"), "wb") as file:      
    #    pickle.dump((dstcs, mtrcs, base, imprvs), file)        


def getPaths(net_num, net_size, subfolder):     
    steadyStatesPaths = [] 
    referencePaths = []  
    goldNetPath = None      
    
    if subfolder == "DREAM4":
        folder_path = os.path.join(".", "data", subfolder, f"insilico_size{net_size}_{net_num}")  
        timeSeriesPaths = [os.path.join(folder_path, f"insilico_size{net_size}_{net_num}_timeseries.tsv")]


        path_knockdowns = os.path.join(folder_path, f"insilico_size{net_size}_{net_num}_knockdowns.tsv") 
        path_knockouts = os.path.join(folder_path, f"insilico_size{net_size}_{net_num}_knockouts.tsv") 
        path_wildtype = os.path.join(folder_path, f"insilico_size{net_size}_{net_num}_wildtype.tsv")   

        binarisedPath = os.path.join(folder_path, f"insilico_size{net_size}_{net_num}_binarised.tsv")  

        steadyStatesPaths = [path_knockdowns, path_knockouts, path_wildtype]        
        path_gold_standard = os.path.join(".", "data", subfolder, "DREAM4_gold_standards")  

        #exclude network 1
        for i in range(1,6):
            if i != net_num:
                path = os.path.join(path_gold_standard, f"insilico_size{net_size}_{i}_goldstandard.tsv")  
                referencePaths.append(path)   
            else:
                goldNetPath = os.path.join(path_gold_standard, f"insilico_size{net_size}_{i}_goldstandard.tsv")

    elif subfolder == "EcoliExtractedNetworks":      
        folder_path = os.path.join(".", "data", subfolder, str(net_size))         
        timeSeriesPaths = [os.path.join(folder_path, f"Ecoli-{net_num}_dream4_timeseries.tsv")]  
        steadyStatesPaths = []       

        for i in range(1,11):  
            if i != net_num:
                path = os.path.join(folder_path, f"Ecoli-{i}_goldstandard.tsv")   
                referencePaths.append(path) 
            else:
                goldNetPath = os.path.join(folder_path, f"Ecoli-{i}_goldstandard.tsv")          
    
        binarisedPath = os.path.join(folder_path, f"Ecoli-{net_num}_dream4_binarised.tsv")     

    return timeSeriesPaths, steadyStatesPaths, referencePaths, goldNetPath, binarisedPath 

def runFramework(net_num, net_size, subfolder, savePath = None, debug = True):      
    timeSeriesPaths, steadyStatesPaths, referencePaths, goldNetPath, binarisedPath = getPaths(net_num, net_size, subfolder) 
    
    decoder = framework.ContextSpecificDecoder(timeSeriesPaths, steadyStatesPaths = steadyStatesPaths, referenceNetPaths = referencePaths, goldNetPath = goldNetPath, savePath=savePath)       
    fronts = decoder.getNetworkCandidates()  

    print("Number of networks in first Pareto front")   
    print(len(fronts[0]))  
    first_front =  framework.getUniqueSubjects(fronts[0]) 
    print("Number of unique networks in first Pareto front") 
    print(len(first_front))  

    #for subject in first_front: 
    #    print(subject.getAdjacencyMatrix())          

    distances, metrics, baseMetric = decoder.test(first_front)     

    if debug:
        accuracies = [metric["Accuracy"] for metric in metrics]
        precisions = [metric["Precision"] for metric in metrics]
        recals = [metric["Recall"] for metric in metrics]  
        f1Scores = [metric["F1"] for metric in metrics] 
        mccs = [metric["MCC"] for metric in metrics] 
        bms = [metric["BM"] for metric in metrics]  

        plt.figure()
        plt.scatter(distances[0], f1Scores)
        plt.xlabel("Distances weights") 
        plt.ylabel("F1 scores") 
        plt.show()   
        plt.figure()
        plt.scatter(distances[1], f1Scores) 
        plt.xlabel("Distances topology") 
        plt.ylabel("F1 scores") 
        plt.show() 
        plt.figure()
        plt.scatter(distances[2], f1Scores)
        plt.xlabel("Distances combined")   
        plt.ylabel("F1 scores") 
        plt.show()       

    indices = np.argmin(distances, axis=1)       
    topMetric = metrics[indices[2]] 

    accuracyImpr = topMetric["Accuracy"] - baseMetric["Accuracy"]
    precisionImpr = topMetric["Precision"] - baseMetric["Precision"] 
    racallImpr = topMetric["Recall"] - baseMetric["Recall"] 
    f1Impr = topMetric["F1"] - baseMetric["F1"]  
    mccImpr = topMetric["MCC"] - baseMetric["MCC"]   
    bmImpr = topMetric["BM"] - baseMetric["BM"]   

    imprv = {"Accuracy": accuracyImpr, "Precision": precisionImpr, "Recall": racallImpr, "F1": f1Impr, "MCC": mccImpr, "BM": bmImpr}
    print(imprv)  

    return distances, metrics, baseMetric, imprv 

if __name__ == "__main__": 
    matplotlib.use('TkAgg')    
    main() 

    