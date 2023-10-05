This is the NetREX version 2017.04.14 that was used to generate results
for the manuscript:

Yijie Wang, Dong-Yeon Cho, Hangnoh Lee, Justin Fear, Brian Oliver, and Teresa M Przytycka. NetREX: Network Rewiring using EXpression - Towards Contexts-specific Regulatory Networks


--------------------------------------------------------------------------------
Installation 
--------------------------------------------------------------------------------
Include ./Matlab into the matlab path. (in the current folder and use Matlab command “addpath(./Matlab)”)

--------------------------------------------------------------------------------
Parameters and a brief explanation of each one
--------------------------------------------------------------------------------
Input.NumGene = N # number of genes in the network
Input.NumTF  = M  # number of TFs in the network
Input.NumExp = L # number of observations in the expression data

Input.S0 # a noisy prior network: N*M dimensions
Input.A0 # initial TF activities(TFAs), assign random values is fine: M*L dimensions
Input.E # gene expression data or observed data: N*L dimensions

Input.KeepEdge # number of edges kept in the prior 
Input.AddEdge # number of edges added to the prior
Input.mu # positive number to avoid TFAs reach boundary, “1” can be used as default
Input.kappa # control the power the graph embedding term,  “1” can be used as default
Input.xi # avoid regulatory potentials to reach boundary. “1” can be used as default
Input.C # bound for regulatory potentials, “2” can be used as default
Input.M # bound for TFAs, the maximum value in Input.E

--------------------------------------------------------------------------------
Output Files
--------------------------------------------------------------------------------
Following the example, you can get a network with edge ranking as the weights. The order of genes and TFs are the same to Input.S0

--------------------------------------------------------------------------------
Other Resources
--------------------------------------------------------------------------------
Data used for E.coli and Drosophila are included in ./Data folder

./Data/Inputs contains the input data for all the experiments. 

./Data?Outputs contains predicted networks for female fly and male fly without testis.

./Data/PPI&GOscore contains the protein protein interactions and GO terms (IC >=2) data for different fly data.

--------------------------------------------------------------------------------
Example
--------------------------------------------------------------------------------
Example: run “Run_NetREX_ToyExample.m” in folder ./Example 
The toy example is corresponding to Fig. 1b in the manuscript. 
