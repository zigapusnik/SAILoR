
import scipy.interpolate as interpolate  
import matplotlib.pyplot as plt 
import numpy as np 
import os 
import pandas as pd 
import re
import sys


folder_path = os.path.join(".") 

groups = ["V", "C"] 
experiments = ["A", "B"]    
time_points = np.arange(1, 9, 1)           
interpolation = "cubic-spline" # "B-spline" "cubic-spline"
plot = False       

for group in groups:   
    print(group) 
    cpm_type_file = "cpm_" + group + ".txt" 
    cpm_type_df = pd.read_csv(os.path.join(folder_path, cpm_type_file), delimiter="\t", index_col=0) 

    my_index = cpm_type_df.index 

    columns = cpm_type_df.columns 
    for experiment in experiments:
        print(experiment)

        a = np.zeros((len(my_index), len(time_points)))  
        b = np.zeros((len(my_index), len(time_points))) 

        experiment_columns = []
        experiment_time_points = []  

        re_string = experiment + "_" + group + "_(.+)"
        for column in columns:
            res = re.match(re_string, column)
            if res:
                timepoint = res.groups()[0] 
                timepoint = timepoint.replace("_", ".") 
                experiment_time_points.append(float(timepoint))

                experiment_columns.append(column)  

        cpm_type_experiment_df = cpm_type_df[experiment_columns] 
        cpm_type_experiment_interpolated_df = pd.DataFrame(b, index=my_index, columns=time_points)      

        for gene_num in my_index:
            cpm_row = cpm_type_experiment_df.loc[gene_num].to_numpy()

            if interpolation == "cubic-spline":
                cpm_cs = interpolate.CubicSpline(experiment_time_points, cpm_row, bc_type="clamped") 
                cpm_type_experiment_interpolated_df.loc[gene_num] = cpm_cs(time_points)   

                if plot:
                    plt.plot(experiment_time_points, cpm_row, 'bo', label='Original points')
                    plt.plot(time_points, cpm_cs(time_points), 'r', label='Cubic-Spline')
                    plt.legend(loc='best')
                    plt.show()   

            elif interpolation == "B-spline":
                #cubic spline interpolation 
                t_cpm,c_cpm,k_cpm = interpolate.splrep(experiment_time_points, cpm_row, s=0, k=3) 
                cpm_bspline = interpolate.BSpline(t_cpm, c_cpm, k_cpm)  
                cpm_type_experiment_interpolated_df.loc[gene_num] = cpm_bspline(time_points) 
                cpm_type_experiment_interpolated_df.loc[gene_num][experiment_time_points[:-1]] = cpm_row[:-1]  

                if plot:
                    plt.plot(experiment_time_points, cpm_row, 'bo', label='Original points')
                    plt.plot(time_points, cpm_bspline(time_points), 'r', label='BSpline')
                    plt.legend(loc='best')
                    plt.show()  

            else: 
                sys.exit() 

        #trim negative values to 0
        cpm_type_experiment_interpolated_df[cpm_type_experiment_interpolated_df < 0] = 0 

        #write to csv files  
        cpm_type_experiment_interpolated_df.to_csv(os.path.join(folder_path, "cpm_" + experiment + "_" + group + "_interpolated.txt"), sep="\t")  
