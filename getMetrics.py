import matplotlib 
import matplotlib.pyplot as plt  
import numpy as np 
import os    
import pandas as pd    
import re
import scipy.stats as st
import seaborn as sns 
import time 

def show(fig):
    import io
    import plotly.io as pio 
    from PIL import Image
    buf = io.BytesIO()
    pio.write_image(fig, buf)
    img = Image.open(buf)
    img.show()         

def plot_charts(organism, size_data_dict, save_path): #metrics, metrics_w_units                 
    br = "\n"   

    #data_dict 
    sizes = list(size_data_dict.keys()) 
    size_fu = sizes[0]
    print(sizes) 
    print(sizes[0])  

    methods = list(size_data_dict[size_fu][organism].keys()) #get all methods 
    print(methods)   
    print(size_data_dict[size_fu][organism]) 

    metrics = size_data_dict[size_fu][organism][methods[0]].keys().tolist() 
    print(metrics)         


    fig = plt.figure(figsize=(9, 9))               
    for indx, metric in enumerate(metrics):         
            
        categories = [*methods, methods[0]]     

        plot_data_all = []
        lower_all = [] 
        upper_all = []         

        for size, data_dict in size_data_dict.items():
            plot_data = [] 
            lower = []
            upper = [] 
            for i, method in enumerate(methods): 
                plot_data.append(data_dict[organism][method][metric]["mean"]) 
                lower.append(data_dict[organism][method][metric][0]) 
                upper.append(data_dict[organism][method][metric][1])        
    
            plot_data = [*plot_data, plot_data[0]]     
            plot_data_all.append(plot_data) 
    
            lower = [*lower, lower[0]]     
            lower_all.append(lower) 
    
            upper = [*upper, upper[0]]     
            upper_all.append(upper)                              

        for i, category in enumerate(categories):
            if category == "BestFit":
                categories[i] = "Best-Fit"        

        if metric != "Time [s]":   

            #print(str(metric) + " " + str(indx))      

            offset = 2*(60/360.)*np.pi            
            label_loc = list(np.linspace(start=offset, stop=2*np.pi + offset, num=len(plot_data)))        
            label_loc = [loc % (2*np.pi) for loc in label_loc]             

            ax = plt.subplot(3, 3, indx, polar=True)        
            min_val = float('inf')   
            max_val = float('-inf')  

            for plot_data, lower, upper, size in zip(plot_data_all, lower_all, upper_all, sizes): 
                plt.plot(label_loc, plot_data, label = "Network size " + str(size))
                plt.fill_between(label_loc, upper, lower, facecolor=plt.gca().lines[-1].get_color(), alpha=0.25)       

                if min(lower) < min_val:
                    min_val = min(lower)

                if max(upper) > max_val: 
                    max_val = max(upper)  

            plt.title(str(metric), y=1.05, weight=600)                                          
            lines, labels = plt.thetagrids(np.degrees(label_loc), labels=categories) 
            handles, labels = ax.get_legend_handles_labels()
               

    fig.legend(handles, labels, loc='lower center')   
    fig.tight_layout()   
    fig.subplots_adjust(wspace=0.45, hspace=0.45)         

    plt.savefig(os.path.join(save_path, organism + "_Metrics" + ".eps"), format='eps')        
    plt.savefig(os.path.join(save_path, organism + "_Metrics" + ".pdf"), format='pdf')              


    #plot running time on plots                
    plt.figure(figsize=(6, 6))   
    fig, ax = plt.subplots()   

    colors = [0]*3
    cmap = matplotlib.cm.get_cmap('tab10') 
    all_colors = list(cmap.colors) 

    colors[0] = cmap.colors[0]
    colors[1] = cmap.colors[1]
    colors[2] = cmap.colors[2]


    metric = "Time [s]"   
    num = len(methods)     
    width = 0.13  # the width of the bars 

    my_methods = ["BestFit", "SAILoR", "MIBNI", "GABNI", "REVEAL", "ATEN", "LogBTF"]  
    my_names = {"BestFit": "Best-Fit", "SAILoR": "SAILoR", "MIBNI": "MIBNI", "GABNI": "GABNI", "REVEAL": "REVEAL", "ATEN": "ATEN", "LogBTF": "LogBTF"}   
    my_names_arr = ["Best-Fit", "Best-Fit", "Best-Fit", "SAILoR", "SAILoR", "SAILoR", "MIBNI", "MIBNI", "MIBNI", "GABNI", "GABNI", "GABNI", "REVEAL",  "REVEAL",  "REVEAL", "ATEN", "ATEN", "ATEN", "LogBTF", "LogBTF", "LogBTF"]

    for i, method in enumerate(my_methods): 
        x = [] 
        y = []  
        ymin = []
        ymax = [] 
        my_labels = []

        for j, size in enumerate(sizes):
            x.append(j) 
            y.append(size_data_dict[size][organism][method][metric]["mean"])  
            ymin.append(size_data_dict[size][organism][method][metric][0])   
            ymax.append(size_data_dict[size][organism][method][metric][1]) 

            my_labels.append(re.sub('^0m', '', re.sub('^0h', '', time.strftime('%#Hh%#Mm%#Ss', time.gmtime(round(y[-1]))))))  

            rect = ax.bar(x[-1] + (i - int(num/2))*width, (y[-1]), width, yerr=[[(y[-1] - ymin[-1])], [(ymax[-1] - y[-1])]], label=my_names[method], capsize=6, color=colors[j], alpha=0.7)          
            ax.bar_label(rect, labels=[my_labels[j]], padding=5) 
    

    pos = []
    for bar in ax.patches:
        pos.append(bar.get_x()+bar.get_width()/2.)

    ax.set_yscale("log") 

    ax.set_xticks(pos)  
    ax.set_xticklabels(my_names_arr)
    plt.xticks(rotation=45)

    plt.ylabel('Time [s]')                 
    ax.spines['top'].set_visible(False) 
    ax.spines['right'].set_visible(False)  
    plt.legend([f"Network size {size}" for size in sizes]) 
    plt.tight_layout()             
    plt.savefig(os.path.join(save_path, organism + "_" + metric + ".eps"), format='eps')  
    plt.savefig(os.path.join(save_path, organism + "_" + metric + ".pdf"), format='pdf')                
            
if __name__ == '__main__':

    sns.set_style("white")   

	#set working directory
    os.chdir(os.path.dirname(__file__))    

    organisms = ["EcoliExtractedNetworks"]               
    methods = ["SAILoR", "BestFit", "GABNI", "MIBNI", "REVEAL", "ATEN", "LogBTF"]          

    networkSizes = [16, 32, 64]                  
    networkNum = 10         
    testSize = 56   
    
    for organism in organisms:  
        organism_path =  os.path.join(".", "results", organism)  
        res_size_data = {}     

        for networkSize in networkSizes:
            res_data = {}  
            res_data[organism] = {} 

            for method in methods:
                results_path =  os.path.join(organism_path, str(networkSize))       
                results_method_path = os.path.join(results_path, method)     

                all_size_data_df = pd.DataFrame()      
                all_size_structure_df = pd.DataFrame()            

                print(results_method_path)         
                for net_num in range(1, networkNum+1): 
                    result_file_path = os.path.join(results_method_path, "results_network_" + str(net_num) + ".tsv")  
                    results_structure_file_path = os.path.join(results_method_path, "results_structure_network_" + str(net_num) + ".tsv")        
                    #print(result_file_path)           
                    if os.path.isfile(result_file_path):
                        results_df = pd.read_csv(result_file_path, sep='\t')    
                        results_df.replace([np.inf, -np.inf], np.nan, inplace=True)
                        results_df.dropna(inplace=True)   
                        all_size_data_df = all_size_data_df.append(results_df)                    
                    else:         
                        continue   
                    if os.path.isfile(results_structure_file_path):   
                        results_df = pd.read_csv(results_structure_file_path, sep='\t')  
                        all_size_structure_df = all_size_structure_df.append(results_df)         
                    else: 
                        continue 


                all_size_data_df["errors"] = all_size_data_df["errors"].apply(lambda x: 1 - x / ((testSize - 1)*networkSize))   
                all_size_data_df = all_size_data_df.rename(columns={"errors":"Dynamic accuracy", "time": "Time [s]"})         
                
                mean0 = all_size_data_df.agg(["mean"])
                mean1 = all_size_structure_df.agg(["mean"])  
                mean_all = pd.concat([mean0, mean1], axis=1)  
                print(mean_all) 

                all_size_data_df_confidence = all_size_data_df.apply(lambda x: st.norm.interval(alpha=0.95, loc=np.mean(x), scale=st.sem(x)), axis=0)
                all_size_structure_df_confidence = all_size_structure_df.apply(lambda x: st.norm.interval(alpha=0.95, loc=np.mean(x), scale=st.sem(x)), axis=0)    
                all_size_df_confidence = pd.concat([all_size_data_df_confidence, all_size_structure_df_confidence], axis=1)
 
                mean_all = pd.concat([mean_all, all_size_df_confidence], axis=0)                  
                res_data[organism][method] = mean_all   


            res_size_data[networkSize] = res_data  

        plot_charts(organism, res_size_data, organism_path)                   



