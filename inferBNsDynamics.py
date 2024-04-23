import csv          
import itertools
import numpy as np 
import os  
import pandas as pd
import re  
import subprocess 
import time     

from functools import reduce
from sklearn.cluster import KMeans 

def iterativeKmeans(data, d=3):  
	data = np.array(data)    		

	while d > 0:  
		data = np.reshape(data, (-1,1)) #reshape to array with one feature  
		clusters = pow(2, d) 
		kmeans = KMeans(n_clusters=clusters, random_state=0, n_init=10).fit(data)   
		data = kmeans.cluster_centers_[kmeans.labels_] 
		d = d - 1  
	#binarize 	
	boolV = kmeans.cluster_centers_[0,0] > kmeans.cluster_centers_[1,0] 
	centers = np.array([int(boolV), int(not boolV)])    
	return pd.Series(centers[kmeans.labels_].tolist())        


def evalGabniModel(model_path, test_series):
	rows, columns = test_series.shape 
	simulations = test_series.iloc[[0]].copy()  #set initial states    
	print(simulations) 

	with open(model_path) as f: 
		bool_expressions = f.readlines()         

	target_genes =  [] 
	regulators  =  []  
	eval_maps  =  []  

	for k in range(0, len(bool_expressions)):  
		expr = bool_expressions[k].rstrip()    
		gene_num = int(re.search(r'\d+', expr[:expr.find("(")]).group()) 
		e_string = re.sub("([^{}=, ]+(?=[ ]*=[ ]*))", r"'\1'", expr[expr.find("= ") + 2:]).replace("=",":") 
		e_map =  eval(e_string)     
		target_genes.append(gene_num)    
		regs = eval(expr[expr.find("("):expr.find(")")+1]) 
		if not isinstance (regs, tuple):  
			regs = [regs]
		regs = list(regs)  

		regulators.append([x - 1 for x in regs])               
		eval_maps.append(e_map)     

	for time_stamp in range(1, rows): 
		ex_row = [0]*columns   
		for k in range(0, len(target_genes)):         
			gene_num = target_genes[k]     
			eval_map = eval_maps[k] 
			regs = regulators[k]       

			keys = list(map(int, simulations.iloc[time_stamp - 1, regs])) 
			key =  reduce((lambda x, y: str(x) + str(y)), keys, "")   
			if key in eval_map:
				val = eval_map.get(key) 
			else:
				val = simulations.iat[time_stamp - 1, gene_num - 1]
			ex_row[gene_num - 1] = val     

		simulations = simulations.append([ex_row], ignore_index = True)   

	erros = simulations.sub(test_series)   
	return np.absolute(erros.to_numpy()).sum()   

def getTargetGenesEvalExpressions(bool_expressions):  
	target_genes = [] 
	eval_expressions = []  
	for k in range(0, len(bool_expressions)):  
		expr = bool_expressions[k]   
		gene_num = int(re.search(r'\d+', expr[:expr.find(" = ")]).group())
		eval_expr =  expr[expr.find("= ") + 2:]
		target_genes.append(gene_num)   
		eval_expressions.append(eval_expr) 
	return target_genes, eval_expressions 

def getBooleanExpressions(model_path):
	bool_expressions = []
	with open(model_path) as f:
		bool_expressions = [line.replace("!"," not ").replace("&"," and ").replace("||", " or ").strip() for line in f]  
	return bool_expressions     

def evalBooleanModel(model_path, test_series): 
	rows, columns = test_series.shape 
	simulations = test_series.iloc[[0]].copy()  #set initial states      
	bool_expressions = getBooleanExpressions(model_path)       
	target_genes, eval_expressions = getTargetGenesEvalExpressions(bool_expressions)         

	#intialize genes to false
	for k in range(0, columns):   
		gene_num = k + 1    
		exec("Gene" + str(gene_num) + " = False")     

	for time_stamp in range(1, rows):  
		#dynamically allocate variables  
		for k in range(0, len(target_genes)):    
			gene_num = target_genes[k]   
			exec("Gene" + str(gene_num) + " = " + str(simulations.iat[time_stamp - 1, gene_num - 1]))    
		
		#initialize simulation to false  
		ex_row = [0]*columns   
		#evaluate all expression  
		for k in range(0, len(bool_expressions)):      
			gene_num = target_genes[k]    
			eval_expr = eval_expressions[k]     
			#print(eval_expr)   
			ex_row[gene_num - 1] = int(eval(eval_expr))         	    	 	   

		simulations = simulations.append([ex_row], ignore_index = True)    

	erros = simulations.sub(test_series) 
	return np.absolute(erros.to_numpy()).sum()    

def evalLogBTFModel(model_path, test_series):     
	coefficient_matrix = np.loadtxt(model_path, dtype='float', delimiter=' ')
	rows, columns = test_series.shape  
	simulations = np.zeros((rows, columns))    
	simulations[0,:] = test_series.iloc[[0]].copy()    

	zero_indices = np.where(~coefficient_matrix.any(axis=0))[0]       

	for time_stamp in range(1, rows):    
		prediciton = np.matmul(simulations[time_stamp-1,:], coefficient_matrix[1:,:]) + coefficient_matrix[0,:] #add thresholds  
		prediciton[prediciton >= 0] = 1
		prediciton[prediciton < 0] = 0 
		simulations[time_stamp,:] = prediciton

		#if entire column of coefficient matrix is zero keep previous value   
		for i in zero_indices:
			simulations[time_stamp,i] = simulations[time_stamp-1,i]       

	errors = np.subtract(simulations, test_series.to_numpy())       
	return np.absolute(errors).sum()              

if __name__ == '__main__':  

	#set working directory
	os.chdir(os.path.dirname(__file__))  

	organisms = ["EcoliExtractedNetworks"]   

	methods = ["SAILoR"]                                                              
	networkSize = 64                    
	networkNum = 10                       
     
	for organism, method in itertools.product(organisms, methods):  
		data_path = os.path.join(".", "data", organism, str(networkSize))   
		results_path =  os.path.join(".", "results", organism, str(networkSize))     
		results_method_path = os.path.join(results_path, method)   

		for i in range(1, networkNum + 1):  

			data_file = "Ecoli" + "-" + str(i) + "_dream4_timeseries.tsv" 
			binarized_file = os.path.join(results_path, data_file)   
			results_file = os.path.join(results_method_path, "results_network_" + str(i) + ".tsv")  

			#get reference paths 
			referencePaths = []     
			for j in range(1,networkNum + 1):   
				if i != j:   
					ref_path = os.path.join(data_path, f"Ecoli-{i}_goldstandard.tsv")   
					referencePaths.append(ref_path)              			

			execution_times = list()         
			dynamic_errors = list() 		
			
			df = pd.read_csv(os.path.join(data_path, data_file), sep="\t", decimal=",")  
			df = df.apply(pd.to_numeric) 
			df = df.dropna() 
			my_columns = list(df.columns)   
			my_columns.remove("Time")        		
			#df.columns = np.arange(len(df.columns))  #drop column names 

			new_columns = ["Gene" + str(i) for i in np.arange(1, len(my_columns) + 1)]        
			df = df.rename(columns={a:b for a, b in zip(my_columns, new_columns)}) #drop column names except Time               

			if not os.path.exists(binarized_file): 
				#df = df.drop(columns=["Time"])  
				bin_df = df[new_columns].apply(iterativeKmeans, axis=0)   
				bin_df.to_csv(binarized_file, index=False, sep="\t", header=None)              

			bin_df = pd.read_csv(binarized_file, sep="\t", header=None)    

			rows, columns = df.shape     
			seriesSize = rows      
			test_size = 56                    
			crossIterations = int(seriesSize/test_size) 
			print(crossIterations) 
			print("Cross-validation iterations: " + str(crossIterations))         

			if method == "GABNI":       	 
				evalFunction = evalGabniModel
			elif method == "LogBTF":
				evalFunction = evalLogBTFModel
			else:
				evalFunction = evalBooleanModel   
			
			for j in range(crossIterations):   
				
				crossIterationFilePath = "Ecoli" + "-" + str(i) + "-" + str(j) + "_dream4_timeseries.tsv" 
				crossIterationBinFilePath = "Ecoli" + "-" + str(i) + "-" + str(j) + "_dream4_timeseriesBinarised.tsv" 
				seriesPath = os.path.join(data_path, crossIterationFilePath)  
				seriesBinPath = os.path.join(data_path, crossIterationBinFilePath)    

				#prepare time series for inference  
				dynamics_file = os.path.join(results_method_path, organism + "-" + str(i) + "_" + str(j) + "_dynamics.tsv")
				structure_file = os.path.join(results_method_path, organism + "-" + str(i) + "_" + str(j) + "_structure.tsv")      
	
				drop_rows = range(j*test_size, min((j + 1)*test_size, seriesSize))    

				test_series = df.iloc[drop_rows]    
				test_series = test_series.reset_index(drop=True)     

				test_series_bin = bin_df.iloc[drop_rows]     
				test_series_bin = test_series_bin.reset_index(drop=True)     
				      
				infer_series = df.drop(drop_rows)     
				infer_series = infer_series.reset_index(drop=True)
				delta_t = infer_series["Time"][1] - infer_series["Time"][0]     
				infer_series["Time"] = np.array([delta_t*k for k in range(infer_series["Time"].size)])      				   

				infer_series_bin = bin_df.drop(drop_rows) 
				infer_series_bin = infer_series_bin.reset_index(drop=True)   

				print(test_series_bin.shape) 
				print(infer_series_bin.shape)       
				print(infer_series.shape)      
	 
				if method == "MIBNI": 
					cmd = r"java -classpath implementations MIBNI.Mibni" + " \"" + infer_series_bin.to_csv(header=None, sep="\t", index=None) + "\" " + dynamics_file  
				elif method == "GABNI": 	
					cmd = r"java -classpath implementations\GABNI gabni.GA" + " \"" + infer_series_bin.to_csv(header=None, sep="\t", index=None) + "\" " + dynamics_file + " " +  structure_file    
				elif method == "ATEN":    
					#write to file due to R arguments bug 
					f = open("tmp_ATEN.txt", "w")   
					infer_series_bin.to_csv(f, header=None, sep="\t", index=None) 
					f.close()    
					cmd = r"Rscript  implementations\ATEN\runAten.R" + " " + f.name + " " + dynamics_file
				elif method == "LogBTF":	
					f = open("tmp_LogBTF.txt", "w")   
					infer_series_bin.to_csv(f, header=None, sep="\t", index=None)  
					f.close()   
					cmd = r"Rscript " + os.path.join(".", "implementations", "LogBTF", "runLogBTF.R") + " \"" + f.name + "\" \"" + dynamics_file + "\""          
				elif method == "BestFit": 
					cmd = r"python2 implementations\BooleanModeling2post\BinInfer.py learn-method=BESTFIT solutions=1 iterations=1 verbose=2" + " input=\"" + infer_series_bin.to_csv(sep="\t", index=None) + "\" outputFile="+dynamics_file
				elif method == "REVEAL": 
					#cmd = r"python implementations\REVEAL\reveal.py " + " --input \"" + infer_series.to_csv(sep="\t", index=None) + "\" --outputFile="+dynamics_file
					cmd = r"python2 implementations\BooleanModeling2post\BinInfer.py learn-method=REVEAL solutions=1 iterations=1 verbose=2" + " input=\"" + infer_series_bin.to_csv(sep="\t", index=None) + "\" outputFile="+dynamics_file
				elif method == "SAILoR": 

					#save to file if file does not exists 
					#if not os.path.exists(seriesPath): 
					infer_series.to_csv(seriesPath, sep="\t", index=False)   
					#if not os.path.exists(seriesBinPath): 
					infer_series_bin.to_csv(seriesBinPath, sep="\t", index=False)     

					cmd = "python SAILoR.py" + " --timeSeriesPath \"" + seriesPath  + "\" --binarisedPath \"" + seriesBinPath + "\" --referencePaths \"" + str(referencePaths) + "\" --outputFilePath \"" + dynamics_file + "\" --decimal ,"        
				else:                  
					break                         	   

				#put inference method in try except block    
				try: 
					print(cmd)
					start = time.time()     
					p = subprocess.call(cmd, shell=True)          
					end = time.time()  
					elapsed = end - start    
					errs = evalFunction(dynamics_file, test_series_bin)       
				except Exception as error:
					print("An error occurred:", error)
					elapsed = float('Inf') 
					errs = float('Inf')    

				execution_times.append(elapsed)          
				dynamic_errors.append(errs)       

			rslt_df = pd.DataFrame(list(zip(execution_times, dynamic_errors)), columns=["time", "errors"])    
			rslt_df.to_csv(results_file, index=False, sep="\t", float_format='%.2f')                    
