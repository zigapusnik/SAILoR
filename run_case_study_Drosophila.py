###########################################################
#### Drosophila melanogaster case study with SAILoR (Structure-Aware Inference of Logic Rules) ####
########################################################### 

#Copyright 2023 Žiga Pušnik   

#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os 
import pandas as pd 
import SAILoR 

def preprocess_timeseries(ts_paths):
    df_list = [] 
    for ts_path in ts_paths:
        df = pd.read_csv(ts_path, sep="\t", index_col = 0)      
        df_T = df.transpose() 
        #normalize  
        df_T = (df_T - df_T.min())/(df_T.max() - df_T.min())   
        df_T.insert(0, 'Time', 0)  
        df_T["Time"] = df_T.index 
        df_T = df_T.reset_index(drop=True) 
        df_list.append(df_T)  

    return pd.concat(df_list, axis=0, ignore_index = 0)  

#utilize A and B time series for network inference from V and C groups
if __name__ == "__main__":


    data_folder = os.path.join(".", "data", "DrosophilaMelanogaster") 
    reference_network_file_name = "NetREX_female_reference_network.txt"  
    ranked_interractions_file_name = "NetREX_female_prediciton_ranks.txt"    
    genes_list_file_name = "genes.txt"  
  
    groups = ["V", "C"]           

    for group in groups:

        #preprocess data if file with normalized time series does not exists
        time_series_file_name = "Dm_timeseries_" + group + "_cpm.txt"  
        time_series_path = os.path.join(data_folder, time_series_file_name) 

        ranked_interractions_path = os.path.join(data_folder, ranked_interractions_file_name) 
        reference_network_path = os.path.join(data_folder, reference_network_file_name) 

        print(time_series_path) 

        if not os.path.exists(time_series_path):
            time_series_A_file_name = "cpm_A_" + group + "_interpolated.txt"  
            time_series_B_file_name = "cpm_B_" + group + "_interpolated.txt"  

            A_file_path = os.path.join(data_folder, time_series_A_file_name)
            B_file_path = os.path.join(data_folder, time_series_B_file_name)

            df = preprocess_timeseries([A_file_path, B_file_path])
                  
            df.to_csv(time_series_path, sep="\t", index=False)      


        #time_series_df = pd.read_csv(time_series_path, sep="\t", index_col=False)  
        genes_list_path = os.path.join(data_folder, genes_list_file_name)     
        
        referenceNetPaths =  [os.path.join(data_folder, "prior_networks", "prior-" + str(ind) + ".tsv") for ind in range(1,11)]  

        #run SAILoR with  
        decoder = SAILoR.ContextSpecificDecoder(time_series_path, referenceNetPaths = referenceNetPaths, initialPop = [3,5], initialPopProb = [0.99, 0.01]) #parameters 
        best = decoder.run()   

        boolean_expressions = []    
        for bfun in best:    
            boolean_expressions.append(bfun[4])         

        output_file_name = "cpm_" + group + "_network.txt"
        output_file_path = os.path.join(".", "results", "DrosophilaMelanogaster", output_file_name)   

        with open(output_file_path, "w+") as outFile:   
            outFile.writelines([expression + "\n" for expression in boolean_expressions])       