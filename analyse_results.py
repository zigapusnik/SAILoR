import math  
import matplotlib.pyplot as plt
import os
import pickle    
import re     

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
    plotBoxPlots(imprvs_dict["Precision"])  


def main():
    subfolder = "EcoliExtractedNetworks" #"DREAM4" #"EcoliExtractedNetworks"               
    parameter = "obj2Weights"   
    net_nums = range(1, 11)         
    net_size = 64 #16 #32 #64 #10 #100           
        
    all_subfolders = os.walk(os.path.join(".", "results")) 
    subfolder_start = os.path.join(".", "results", subfolder + "_" + str(net_size) + parameter + "_" )  #EcoliExtractedNetworks_16obj2Weights_[0.3027 0.1308 0.5053 0.6698] # + "+ "[0.9923 0.2326 0.6033 0.3385]"   

    for subfolder in all_subfolders:    
        subfolder_name = subfolder[0]     

        if subfolder_start == subfolder_name[0:len(subfolder_start)]:
            print(subfolder_name) 
            readResults(subfolder_name) 

if __name__ == "__main__":    
    main()       