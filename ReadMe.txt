####################################################################
#### code for SAILoR (Structure-Aware Inference of Logic Rules) ####
####################################################################

SAILoR.py contains python code for Boolean inference method SAILoR. 

SAILoR can be run from a separate python file, e.g. see run_case_study_Drosophila  


	decoder = SAILoR.ContextSpecificDecoder(time_series_path, referenceNetPaths = referenceNetPaths) #initializes required data structures and objects
	best = decoder.run() #runs the method and returns the Boolean function that matches binarized time series data from time_series_path 


SAILoR can be run directly from python interpreter, e.g. see inferBNsDynamics.py 
	
	python SAILoR.py --timeSeriesPath "seriesPath" --binarisedPath "seriesBinPath" --referencePaths "referencePaths" --outputFilePath "output_file" --decimal "," #default decimal character is "."  