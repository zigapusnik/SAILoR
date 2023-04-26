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
    subfolder = "DREAM4" #"EcoliExtractedNetworks" #"DREAM4" #"EcoliExtractedNetworks"               
    net_nums = range(1, 11)     
    net_size = 100 #16 #32 #64 #10 #100       

    folder = os.path.join(".", "results", subfolder + "_" + str(net_size))    

    with open(os.path.join(folder, "dump_results.pkl"), "rb") as file:  
        dstcs, mtrcs, base, imprvs = pickle.load(file)              

    imprvs_dict = {"Accuracy": [], "Precision": [], "Recall": [], "F1": [], "MCC": [], "BM": []}
    for net_num, imprv_list in imprvs.items(): 
        print(imprv_list) 
        for imprv in imprv_list:
            for key, value in imprv.items(): 
                if not math.isnan(value): 
                    imprvs_dict[key].append(value)   

    plotBoxPlots(imprvs_dict["BM"])  

if __name__ == "__main__":
    main() 