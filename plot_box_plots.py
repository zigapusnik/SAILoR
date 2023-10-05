import ast
import math  
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle    
import re     

from scipy.interpolate import interp2d

def plotBoxPlots(all_data, net_sizes, metrics):  

    colors = ["#1F77B4", "#FF7F0E", "#2CA02C"] 

    fig, axs = plt.subplots(2, 3)
    i = 0
    val = 0

    for metric in metrics:

        j = i//3
        k = i%3

        plot_data = [x[metric] for x in all_data]

        bp = axs[j][k].boxplot(plot_data, widths=0.6)    
        
        axs[j][k].title.set_text(metric)   
        axs[j][k].set_xticks(range(1, len(plot_data) + 1), net_sizes)

        axs[j][k].set_xlim(-val, 4+val)

        # Hide the right and top spines
        axs[j][k].spines[['right', 'top']].set_visible(False)

        for l, (median, box) in enumerate(zip(bp["medians"], bp["boxes"])):
            median.set(color=colors[l], linewidth=1.25) 
            box.set(color=colors[l], linewidth=1.25) 

        for l, (whisker, cap) in enumerate(zip(bp["whiskers"], bp["caps"])):
            ind = l//2
            whisker.set(color=colors[ind], linewidth=1.25)
            cap.set(color=colors[ind], linewidth=1.25) 

        print(bp["fliers"][0]) 

        for l, flier in enumerate(bp["fliers"]):
            flier.set(markerfacecolor=colors[l], markeredgecolor=colors[l], markersize=3)

        i = i + 1

    fig.tight_layout() 
    plt.show()          
    
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
    return imprvs_dict



def main():
    subfolder = "EcoliExtractedNetworks" #"DREAM4" #"EcoliExtractedNetworks"                
    net_nums = range(1, 11)         
    net_sizes = [16, 32, 64]        
        
    all_subfolders = os.walk(os.path.join(".", "results", "results_18_07_2023"))     
    subfolder_start = os.path.join(".", "results", "results_18_07_2023", subfolder + "_")  #EcoliExtractedNetworks_16obj2Weights_[0.3027 0.1308 0.5053 0.6698] # + "+ "[0.9923 0.2326 0.6033 0.3385]"   

    combined_results = []
    net_sizes_order = []

    metrics = ["Accuracy", "Precision", "Recall", "F1", "BM", "MCC"]  

    for subfolder in all_subfolders:    
        subfolder_name = subfolder[0]        

        if subfolder_start == subfolder_name[0:len(subfolder_start)]: 
            net_size = int(subfolder_name[len(subfolder_start):])
            net_sizes_order.append(net_size)

            results = readResults(subfolder_name)  
            combined_results.append(results)

    print(net_sizes_order)

    if not net_sizes_order == sorted(net_sizes_order):
        indices = np.argsort(net_sizes_order)
        combined_results = list(np.array(combined_results)[indices]) 

    plotBoxPlots(combined_results, net_sizes, metrics)          

if __name__ == "__main__":      
    main()           