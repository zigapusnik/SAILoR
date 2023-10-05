from deap import creator, base, tools, algorithms

import SAILoR 
import matplotlib.pyplot as plt 
import numpy as np 
import os    
import pickle   


def parameter_sweep():  
    #60 repetitions of randomised parameter search    
    #Random Search for Hyper-Parameter Optimization (Bergstra and Bengio, 2012)
    for i in range(60):    

        print(f"Parameter optimization iteration {i}")  

        #draw samples from uniform distribution      
        w1 = np.random.uniform()   
        w2 = np.random.uniform() 
        w3 = np.random.uniform()
        w4 = np.random.uniform()    

        #set parameter values
        obj2Weights = np.array([w1, w2, w3, w4])   
        print(f"Obj2Weights values: {obj2Weights}")     

        #call main function     
        main(obj2Weights=obj2Weights)       


def main(obj2Weights = None):    
    subfolder = "EcoliExtractedNetworks"          
    net_nums = range(1,11)            
    net_size = 16 #16 #32 #64                                                     
    
    imprvs = {}                        
    dstcs = {}                      
    mtrcs = {}            
    base = {}               

    parameter_substring = ""

    if obj2Weights is not None:
        parameter_substring = "obj2Weights_" + np.array2string(obj2Weights, formatter={'float_kind':lambda x: "%.4f" % x})         

    for net_num in net_nums:
        dstcs[net_num] = []    
        mtrcs[net_num] = []  
        base[net_num]  = []    
        imprvs[net_num] = []              

        scenario = subfolder + "_" + str(net_size) + "_" + str(net_num)   
        
        #10 repetitions         
        for runNum in range(10):        
            print(f"Iteration within test scenario {runNum}")    
            scenario = scenario + "_" + str(runNum)  
            savePath = os.path.join(".", "results", scenario) 

            distances, metrics, baseMetric, imprMetric = runFramework(net_num, net_size, subfolder, savePath = savePath, obj2Weights = obj2Weights)    

            dstcs[net_num].append(distances)   
            mtrcs[net_num].append(metrics)  
            base[net_num].append(baseMetric)      
            imprvs[net_num].append(imprMetric)      

    folder = os.path.join(".", "results", subfolder + "_" + str(net_size) + parameter_substring)         

    #create directory if does not exists     
    
    if not os.path.exists(folder):  
        os.makedirs(folder)    
    with open(os.path.join(folder, "dump_results.pkl"), "wb") as file:       
        pickle.dump((dstcs, mtrcs, base, imprvs), file)                
        

def getPaths(net_num, net_size, subfolder):      
    steadyStatesPaths = [] 
    referencePaths = []  
    goldNetPath = None      
    
    subfolder_path = os.path.join(".", "data", subfolder)
    folder_path = os.path.join(subfolder_path, str(net_size)) 
    timeSeriesPath = os.path.join(folder_path, f"Ecoli-{net_num}_dream4_timeseries.tsv")  
    steadyStatesPaths = []         

    for i in range(1,11):  
        if i == net_num: 
            goldNetPath = os.path.join(folder_path, f"Ecoli-{i}_goldstandard.tsv")  
        else:
            path = os.path.join(subfolder_path, str(net_size), f"Ecoli-{i}_goldstandard.tsv")   
            referencePaths.append(path)     

    binarisedPath = os.path.join(folder_path, f"Ecoli-{net_num}_dream4_binarised.tsv")     

    return timeSeriesPath, steadyStatesPaths, referencePaths, goldNetPath, binarisedPath 

def runFramework(net_num, net_size, subfolder, savePath = None, obj2Weights=None, debug = False):               
    timeSeriesPath, steadyStatesPaths, referencePaths, goldNetPath, binarisedPath = getPaths(net_num, net_size, subfolder) 
    
    decoder = SAILoR.ContextSpecificDecoder(timeSeriesPath, steadyStatesPaths = steadyStatesPaths, referenceNetPaths = referencePaths, goldNetPath = goldNetPath, savePath=savePath, obj2Weights=obj2Weights, decimal=",")       
    fronts = decoder.getNetworkCandidates()           

    #decoder.run()   

    first_front =  SAILoR.getUniqueSubjects(fronts[0])                
    
    #print("Number of unique networks in first Pareto front")   
    #print(len(first_front))             

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
    main()    
    #parameter_sweep()           