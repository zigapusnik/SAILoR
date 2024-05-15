####################################################################
#### code for SAILoR (Structure-Aware Inference of Logic Rules) ####
####################################################################

SAILoR.py contains python code for Boolean inference method SAILoR. 

SAILoR can be run from a separate python file, e.g. see run_case_study_Drosophila  


decoder = SAILoR.ContextSpecificDecoder(time_series_path, referenceNetPaths = referenceNetPaths) #initializes required data structures and objects
best = decoder.run() #runs the method and returns the Boolean function that matches binarized time series data from time_series_path 


SAILoR can be run directly from python interpreter, e.g. see inferBNsDynamics.py 
	
python SAILoR.py --timeSeriesPath "seriesPath" --binarisedPath "seriesBinPath" --referencePaths "referencePaths" --outputFilePath "output_file" --decimal "," #default decimal character is "."  

-----

License
Source code of SAILoR with all accompanying data and results is freely available under open source MIT license.  

MIT License
#Copyright 2024 Žiga Pušnik   

#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
