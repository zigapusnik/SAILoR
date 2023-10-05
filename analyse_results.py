import ast
import math  
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle    
import re     

from scipy.interpolate import interp2d

def plotBoxPlots(data, savePath = None):  
    fig = plt.figure()   
    plt.boxplot(data)     
    #ax.set_xticklabels(f1sDict.keys())  
    if savePath is not None:
        plt.savefig(savePath)     
    else:       
        plt.show()          
    #plt.clf()        

def readResults(subfolder_name):
    with open(os.path.join(subfolder_name, "dump_results.pkl"), "rb") as file:  
        dstcs, mtrcs, base, imprvs = pickle.load(file)               


    imprvs_dict = {"Accuracy": [], "Precision": [], "Recall": [], "F1": [], "MCC": [], "BM": []}
    for net_num, imprv_list in imprvs.items():  
        #print(imprv_list)  
        i = 0
        for imprv in imprv_list: 
            for key, value in imprv.items(): 

                #if value of inferred networks is nan increase is negative base value    
                if math.isnan(value):    
                    value = base[net_num][i][key]    
                    value = -value  
                    print(f"nan value detected for network {net_num}")  

                if not math.isnan(value):
                    imprvs_dict[key].append(value)    

            i = i + 1 
    return imprvs_dict["F1"]



def main():
    subfolder = "EcoliExtractedNetworks" #"DREAM4" #"EcoliExtractedNetworks"               
    parameter = "obj2Weights"   
    net_nums = range(1, 11)         
    net_size = 64 #16 #32 #64         
    alphas_to_values = pd.DataFrame(data=None, columns=["alpha1", "alpha2", "alpha3", "alpha4", "median", "min", "max"]) 
        
    all_subfolders = os.walk(os.path.join(".", "results", "14_07_2023"))     
    subfolder_start = os.path.join(".", "results", "14_07_2023", subfolder + "_" + str(net_size) + parameter + "_" )  #EcoliExtractedNetworks_16obj2Weights_[0.3027 0.1308 0.5053 0.6698] # + "+ "[0.9923 0.2326 0.6033 0.3385]"    

    for subfolder in all_subfolders:    
        subfolder_name = subfolder[0]        

        if subfolder_start == subfolder_name[0:len(subfolder_start)]: 
            print(subfolder_name)   
            alphas_string = subfolder_name[len(subfolder_start):]
            alphas_string = alphas_string.replace(" ", ", ")  
            alphas = ast.literal_eval(alphas_string)  
            print(alphas) 
            results = np.array(readResults(subfolder_name))  
            alphas_to_values = pd.concat([alphas_to_values, pd.DataFrame([{"alpha1": alphas[0], "alpha2": alphas[1], "alpha3": alphas[2], "alpha4": alphas[3], "median": np.median(results) , "min": np.min(results) , "max": np.max(results) }])], ignore_index=True)

            #plotBoxPlots(results)        


    alphas_to_values = alphas_to_values.sort_values('median')

    print(alphas_to_values)  

    a1 = list(alphas_to_values["alpha1"])
    a2 = list(alphas_to_values["alpha2"])
    a3 = list(alphas_to_values["alpha3"])
    a4 = list(alphas_to_values["alpha4"])
    values = list(alphas_to_values["median"])

    print(np.max(values))
    print(np.min(values))

    print(len(values)) 

    f = interp2d(a1, a2, values, kind="linear") 

    step = 0.1
    x_coords = np.arange(0, 1 + step, step)  
    y_coords = np.arange(0, 1 + step, step) 

    print()

    z = f(x_coords, y_coords)  


    fig = plt.imshow(z, extent=[0,1,0,1], origin="lower")
    
    fig.axes.set_autoscale_on(False)
    
    plt.colorbar(fig)  
    plt.scatter(a1,a2,100)    

    plt.show()

    ax = plt.axes(projection='3d')

    ax.scatter3D(a1, a2, values)

    plt.show() 


    plt.scatter(a1, values)
    plt.show()

    plt.scatter(a2, values)
    plt.show()

    plt.scatter(a3, values)
    plt.show()

    plt.scatter(a4, values)
    plt.show()

if __name__ == "__main__":      
    main()           