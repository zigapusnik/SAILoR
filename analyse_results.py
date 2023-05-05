import math  
import matplotlib.pyplot as plt
import os
import pickle       

def plotBoxPlots(data, savePath = None):  
    fig = plt.figure()   
    plt.boxplot(data)     
    #ax.set_xticklabels(f1sDict.keys())  
    if savePath is not None:
        plt.savefig(savePath)     
    else:       
        plt.show()          
    #plt.clf()        

def main():
    subfolder = "EcoliExtractedNetworks" #"DREAM4" #"EcoliExtractedNetworks"               
    net_nums = range(1, 11)        
    net_size = 32 #16 #32 #64 #10 #100             

    folder = os.path.join(".", "results", subfolder + "_" + str(net_size))      

    with open(os.path.join(folder, "dump_results.pkl"), "rb") as file:  
        dstcs, mtrcs, base, imprvs = pickle.load(file)               


    imprvs_dict = {"Accuracy": [], "Precision": [], "Recall": [], "F1": [], "MCC": [], "BM": []}
    for net_num, imprv_list in imprvs.items():  
        #print(imprv_list)  
        i = 0
        for imprv in imprv_list: 
            for key, value in imprv.items():    
                if math.isnan(value):    
                    value = base[net_num][i][key]    
                    value = -value  
                    print(f"nan value detected for network {net_num}")  
                imprvs_dict[key].append(value)    

            i = i + 1 
    plotBoxPlots(imprvs_dict["F1"])          

if __name__ == "__main__":   
    main()    