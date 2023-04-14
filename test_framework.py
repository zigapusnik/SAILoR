from deap import creator, base, tools, algorithms

import framework
import matplotlib.pyplot as plt 
import multiprocessing   
import numpy as np 
import os  
import pickle   

def main():  
    subfolderProperties = {"DREAM4": {"net_nums": range(1,6), "net_sizes": [10]}} #,"EcoliExtractedNetworks": {"net_nums": range(1,11), "net_sizes": [16]}}     
    imprvs = {}    
    dstcs = {}   
    mtrcs = {} 
    base = {}       
    for subfolder in subfolderProperties:
        net_nums = subfolderProperties[subfolder]["net_nums"]
        net_sizes = subfolderProperties[subfolder]["net_sizes"]  
        for net_size in net_sizes: 
            imprvsFolderSize = {"Accuracy": [], "Precision": [], "Recall": [], "F1": [], "MCC": [], "BM": []}    
            dstcs[subfolder + "_" + str(net_size)] = []   
            mtrcs[subfolder + "_" + str(net_size)] = []
            base[subfolder + "_" + str(net_size)]  = []   

            for net_num in net_nums:
                for runNum in range(5): 
                    distances, metrics, baseMetric, imprMetric = runFramework(net_num, net_size, subfolder, runNum = runNum)  
                    
                    dstcs[subfolder + "_" + str(net_size)].append(distances)
                    mtrcs[subfolder + "_" + str(net_size)].append(metrics)
                    base[subfolder + "_" + str(net_size)].append(baseMetric) 

                    for key in imprMetric:    
                        imprvsFolderSize[key].append(imprMetric[key])  
            
            imprvs[subfolder + "_" + str(net_size)] = imprvsFolderSize     

    folder = os.path.join(".", "results")
    file = open(os.path.join(folder, "dump_results.pkl")) 
    pickle.sump((), file)
    file.close() 

    boxPlotPath = os.path.join(folder, "boxplots.pdf")  
    plotBoxPlots(imprvs, savePath=boxPlotPath)          

def plotBoxPlots(imprvs, savePath = None):  
    f1sDict = {}
    for folder_Size in imprvs:   
        f1sDict[folder_Size] = imprvs[folder_Size]["F1"]

    plt.figure()  
    fig, ax = plt.subplots()
    ax.boxplot(f1sDict.values())     
    ax.set_xticklabels(f1sDict.keys())  
    if savePath is not None:
        plt.savefig(savePath)     
    else:      
        plt.show()       
    plt.clf()   

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

def runFramework(net_num, net_size, subfolder, runNum = None, debug = False):   
    timeSeriesPaths, steadyStatesPaths, referencePaths, goldNetPath, binarisedPath = getPaths(net_num, net_size, subfolder) 
    scenario = subfolder + "_" + str(net_size) + "_" + str(net_num) 
    if runNum is not None:
        scenario = scenario + "_" + str(runNum) 

    savePath = os.path.join(".", "results", scenario)   
    decoder = framework.ContextSpecificDecoder(timeSeriesPaths, steadyStatesPaths = steadyStatesPaths, referenceNetPaths = referencePaths, goldNetPath = goldNetPath, savePath = savePath)    
    distances, metrics, baseMetric = decoder.run()    

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

    return distances, metrics, baseMetric, {"Accuracy": accuracyImpr, "Precision": precisionImpr, "Recall": racallImpr, "F1": f1Impr, "MCC": mccImpr, "BM": bmImpr} 

if __name__ == "__main__": 
    main() 

    