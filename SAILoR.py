###########################################################
#### SAILoR (Structure-Aware Inference of Logic Rules) ####
########################################################### 

#Copyright 2023 Žiga Pušnik   

#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from dynGENIE3 import *
from deap import creator, base, tools, algorithms
#from inferelator import inferelator_workflow 
from sklearn.cluster import KMeans  
from qm import QuineMcCluskey  
from triadic_census import count_triads, count_local_triads    

import ast
import argparse  
import csv
import networkx as nx
import numpy as np 
import matplotlib.pyplot as plt 
import multiprocessing
import os  
import pandas as pd 
import pickle 
import sklearn.metrics as metrics 
import sys  
import time    


def inferBooleanNetwork(adjM, nNodes, bin_df, shift_bin_df, experiments_df, geneNames, qm, experiments):    
    b_functions = [] 
    for target in range(nNodes):   

        regulators, gT, bfun, bexpr, tT = inferBooleanFunction(adjM[:, target], target, bin_df, shift_bin_df, experiments_df, geneNames, qm, experiments)

        b_fun = (target, regulators, gT, bfun, bexpr, tT)            
        b_functions.append(b_fun)            

    print("Boolean Network Inferred")  
    return b_functions      

def inferBooleanFunction(regulatorsArray, target, bin_df, shift_bin_df, experiments_df, geneNames, qm, experiments):
        gT = None        
        bfun = None  
        regulators = np.argwhere(regulatorsArray == 1).flatten()            

        if regulators.size != 0:  
            gT =  getGeneralisedTruthTable(target, regulators, bin_df, shift_bin_df, experiments_df, experiments)
            minterms = list(gT[gT["T"] > gT["F"]]["inputVector"])   
            dont_cares = list(gT[gT["T"] == gT["F"]]["inputVector"])     
            if len(minterms) != 0 or len(dont_cares) != 0:   
                bfun = qm.simplify(minterms, dont_cares, num_bits=len(regulators))   
        """
        else:  
            print(f"No regulators found for {self.geneNames[target]}")             
        """ 

        bexpr = getExpression(bfun, target, regulators, geneNames)  
        tT = getTruthTable(bexpr, regulators, geneNames)  

        return regulators, gT, bfun, bexpr, tT     


#returns truth table based on Boolean expression  
def getTruthTable(bexpr, regulators, geneNames):  
    numReg = len(regulators)          
    numRows = pow(2, numReg)            
    rowValues = list(range(numRows))     
    regNames = [geneNames[regulator] for regulator in regulators]  

    tT = pd.DataFrame(rowValues, columns = ["inputVector"])  
    tT["value"] = 0
    tT["neg_value"] = 1     

    expr = bexpr[bexpr.find("= ")+1:]   

    if len(regulators != 0): 
        for row in rowValues:  
            #set regulators value 
            regVals = getBinaryFromDecimal(row, numReg)    
            for regName, regValue in zip(regNames, regVals):
                exec(regName + " = " + str(regValue)) 
            value =  int(eval(expr))
            tT.loc[row, "value"] = value 
            tT.loc[row, "neg_value"] = abs(1- value)     

    else:   
        tT["value"] = int(eval(expr))          
    return tT    

def getExpression(bfun, target, regulators, geneNames): 
    expr = geneNames[target] + " = "

    if bfun is None:
        return expr + "0" 

    regulator_names = [geneNames[regulator] for regulator in regulators]        
    expressions = [] 
    #epi ... essential prime implicant   
    for epi in bfun: 
        #list of regulators names, "" for    
        epi_expr = [regulator_names[i] if c != '-' else "" for i, c in enumerate(epi)]
        epi_expr = ["not " + epi_expr[i] if c == '0' else epi_expr[i] for i, c in enumerate(epi)] 
        epi_expr = list(filter(lambda c: c != "", epi_expr))     

        if len(epi_expr) == 0: 
            epi_expr = "1"   

        expressions.append(" and ".join(epi_expr))          
    return expr + " or ".join(expressions)     

def getGeneralisedTruthTable(target, regulators, bin_df, shift_bin_df, experiments_df, experiments):   
    reg_shift_df = shift_bin_df.iloc[:,regulators]    
    target_df = bin_df.iloc[:,target]     
    numReg = len(regulators)       
    numRows = pow(2, numReg)         
    rowValues = list(range(numRows))          

    gT = pd.DataFrame(rowValues, columns = ["inputVector"])
    #create zero columns for T and F   
    gT["T"] = 0          
    gT["F"] = 0              

    for experiment in range(experiments):
        sel = experiments_df["Experiment"] == experiment
        df_shift_exp = reg_shift_df[sel]     
        target_df_exper = target_df[sel]

        selT = target_df_exper == 1         
        selF = target_df_exper == 0          

        #exclude first time point   
        selT.iloc[0] = False  
        selF.iloc[0] = False           

        df_shift_exp_vectors = df_shift_exp.agg(''.join, axis=1)  
        df_shift_exp_vectors = df_shift_exp_vectors.apply(getDecimalFromBinary)      

        vectorsTrue = df_shift_exp_vectors[selT] 
        vectorsFalse = df_shift_exp_vectors[selF]

        countsTrue = vectorsTrue.value_counts()          
        countsFalse = vectorsFalse.value_counts()  

        trueIndices = countsTrue.index.values 
        falseIndices = countsFalse.index.values     

        gT.loc[trueIndices, "T"] = gT.loc[trueIndices,"T"] + countsTrue 
        gT.loc[falseIndices, "F"] = gT.loc[falseIndices,"F"] + countsFalse      

    return gT        

#returns dynamic accuraccy based on training time series data 
#bNetwork ... list of nodes
#bNetwork[i] ... [target, regulators, generalised truth table, boolean function, boolean expression, truth table]  
def getDynamicAccuracy(bNetwork, bin_df, experiments_df, geneNames, experiments, index): 
    total_errors = 0
    timeSteps = 0 

    numNodes = bin_df.shape[1] 
    #simulate Boolean model for each experiment   
    for experiment in range(experiments):  
        sel = experiments_df["Experiment"] == experiment 
        bin_df_exp = bin_df[sel]   
        simNum = len(bin_df_exp.index)     
        initialState = bin_df_exp.iloc[0,:]    

        bin_df_exp = bin_df_exp.to_numpy()                     
        simulation = simulateBooleanModel(bNetwork, list(initialState), simNum, geneNames)                                  

        errors = np.absolute(simulation - bin_df_exp).sum()    
        total_errors = total_errors + errors  
        timeSteps = timeSteps + simNum - 1 #exclude first time point    

    #return dynamic accuracy
    return 1 -  total_errors/(timeSteps*numNodes), index       

#simulates Boolean network based on provided mode  
# mode = 0 ... exec and eval (dynamic evaluation of Boolean expressions) 
# mode = 1 ... truth table lookup 
# mode = 2 ... semi-tensor product      
#TO DO ... different modes of simulation - exec and eval, lookup from truth table, semi-tensor product   
def simulateBooleanModel(bNetwork, initialState, simNum, geneNames, mode=2):     
    num_nodes = len(bNetwork)       
    simulations = np.zeros((simNum, num_nodes), dtype=int)           
    #set initial state       
    simulations[0] = initialState       
    
    #exec and eval
    if mode == 0:
        expressions = {}    
        for node in bNetwork:
            target = node[0]
            bexpr = node[4]  
            expr = bexpr[bexpr.find("= ")+1:]  
            expressions[target] = expr  

        #exec and eval to dynamically evaluate Boolean models 
        for time_stamp in range(1, simNum): 
            for node in bNetwork:
                target = node[0]
                exec(geneNames[target] + " = " + str(simulations[time_stamp - 1, target]))        

            for node in bNetwork:
                target = node[0]  
                exp = expressions[target]   
                simulations[time_stamp, target] = int(eval(exp)) 

    #truth table lookup
    elif mode == 1:  
        for time_stamp in range(1, simNum): 
            for node in bNetwork:
                target = node[0]
                regulators = node[1]  
                tT = node[5]     

                if len(regulators) > 0:  
                    ind = getDecimalFromBinary(''.join(map(str, list(simulations[time_stamp-1, regulators]))))   #use tuple instead of int         
                else:
                    ind = 0 
                simulations[time_stamp, target] = tT.iloc[ind]["value"] 

    #semi-tensor product  
    elif mode == 2:
        #vector representation of states, i.e. 1 = [1 0]^T, 0 = [0 1]^T   
        initialStateVectorized = np.array([initialState, initialState]) 
        initialStateVectorized[1] = initialStateVectorized[1] - 1
        previousStateVectorized = np.abs(initialStateVectorized)  
        currentStateVectorized = np.zeros((2, num_nodes))      

        Tts = {}    
        for node in bNetwork:
            target = node[0] 
            tT = node[5] 
            #vectorize truth table, transpose it and flip columns 
            tT = np.flip(tT[["value", "neg_value"]].to_numpy().transpose(), axis=1)        
            Tts[target] = tT        

        for time_stamp in range(1, simNum):  
            for node in bNetwork:  
                target = node[0]   
                regulators = node[1]     
                tT = Tts[target]      
                currentStateVectorized[:,target]  = getBooleanBySemiTensorProduct(tT, previousStateVectorized[:,regulators])[:,0]  
                simulations[time_stamp, target] = currentStateVectorized[0,target]      
            previousStateVectorized = currentStateVectorized          
        
    return  simulations  


class Network:     
    #nNodes ... network size
    #adjM ... np.array representing adjacency matrix
    #refFile ... file path to reference network  
    #geneIndices ... dictionary - mapping gene names to indices of adjacency matrix
    #geneNames ... dictionary - mapping from adj indices to gene names   
    def __init__(self, nNodes=0, adjM=None, refFile = None, geneIndices = None, geneNames = None):
        self.nNodes = nNodes
        self.adjM = None 
        self.geneIndices = geneIndices  
        self.geneNames = geneNames  
        self.triadicCensus = None 
        self.normalisedTriC = None 
        self.nxG = None 
        self.edge_prob = None  
        self.out_degs_dist = None  
        self.in_degs_dist = None 
        self.in_nums = None      

        if refFile is not None:
            self.constructReferenceNetwork(refFile, geneIndices=geneIndices, geneNames=geneNames)        

        if adjM is not None:
            self.setAdjacencyMatrix(adjM)  
            self.setTriadicCensus()    

    #set number of nodes
    def setnNodes(self, nNodes):
        self.nNodes = nNodes  
        self.nNodesSquared = nNodes*nNodes

    def setMaxRegs(self, maxRegs):
        self.maxRegs = maxRegs   

    def normaliseTriadicCensus(self):
        #normalise triadic census   
        sumT = np.sum(self.triadicCensus) 
        if sumT > 0: 
            self.normalisedTriC = self.triadicCensus/np.sum(self.triadicCensus)           
        else: 
            self.normalisedTriC = self.triadicCensus               

    #sets triadic census count and orbit count for each node
    def setTriadicCensus(self, debug = False):
        start = time.time()     
        self.triadicCensus, self.triad_pair_count = self.countTriads()      
        end = time.time()     
        elapsed = end - start     
        if debug:      
            print(f"Counting triads: {elapsed} seconds elapsed!")             
        
        self.normaliseTriadicCensus()  
  

    def setAdjacencyMatrix(self, adjM): 
        self.adjM = adjM 
        self.setGraph()   
  

    def updateTriadicCensus(self): 
        adj = self.adjM     
        #get nodes with different edges    
        a, b = self.diff_a, self.diff_b      
        if a != b:       
            subgraph_nodes = set()    
            #get node a and b neighbourhood
            neighbours_a = nx.all_neighbors(self.nxG, a)    
            neighbours_b = nx.all_neighbors(self.nxG, b)     
            
            subgraph_nodes.update(neighbours_a)  

            if adj[a,b] == 1 or adj[b,a] == 1:
                #union of all neigbours
                subgraph_nodes.update(neighbours_b)  
            else:
                #intersect of all neighbours
                subgraph_nodes = subgraph_nodes.intersection(neighbours_b) 

            #remove self loops 
            if a in subgraph_nodes:
                subgraph_nodes.remove(a)
            if b in subgraph_nodes:
                subgraph_nodes.remove(b) 

            #update triadic census:   
            triad_pair_count = self.countLocalTriads(a,b,subgraph_nodes)             
            triad_pair_count_diff = triad_pair_count  - self.triad_pair_count[a,b,:] - self.triad_pair_count[b,a,:]    

            #set values for triangular indices    
            self.triad_pair_count[a,b,:] = 0         
            self.triad_pair_count[b,a,:] = triad_pair_count           
            
            self.triadicCensus = self.triadicCensus + triad_pair_count_diff[3:]      
            self.normaliseTriadicCensus()               
                
    #create networkx graph representation
    def setGraph(self):
        if self.adjM is not None:
            self.nxG = nx.from_numpy_array(self.adjM, create_using=nx.DiGraph)  

    def countTriads(self):    
        if self.nxG is not None:
            return count_triads(self.nxG)     
        else: 
            return None    
        
    def countLocalTriads(self, v, u, nodelist):
        if self.nxG is not None:
            return count_local_triads(self.nxG, v, u, nodelist)   
        else: 
            return None             

    def getTriadicCensus(self):
        return self.triadicCensus 
    
    def getNormalisedTriadicCensus(self):
        return self.normalisedTriC  

    def getAdjacencyMatrix(self): 
        return self.adjM  
    
    def setInDegs(self):
        maxRegs = self.maxRegs 
        inDegs = np.zeros(maxRegs + 1)
        if self.adjM is not None: 

            inNums = self.in_nums
            
            if inNums is None:    
                inNums = np.sum(self.adjM, axis=0).astype(int)                       
                inNums[inNums > maxRegs] = maxRegs  
                self.in_nums = inNums
                
            for inNum in inNums:
                inDegs[inNum] += 1     
        
        self.in_degs = inDegs
        self.in_degs_dist = inDegs/self.nNodes         
    
    def updateInDegs(self, diff_b, edge_diff):  
        b_deg_before = self.in_nums[diff_b] 
        b_deg_after = b_deg_before + edge_diff 

        #update node in degree
        self.in_nums[diff_b] = b_deg_after  

        self.in_degs[b_deg_before] = self.in_degs[b_deg_before] - 1   
        self.in_degs[b_deg_after] = self.in_degs[b_deg_after] + 1    

        self.in_degs_dist = self.in_degs/self.nNodes   

    #returns distribution of nodes in-degrees
    def getInDegs(self):   
        if self.in_degs_dist is None:
            self.setInDegs()
        return self.in_degs_dist 
    
    def setOutDegs(self):  
        maxRegulates = self.maxRegs
        outDegs = np.zeros(maxRegulates + 1) 
        if self.adjM is not None: 
            outNums = np.sum(self.adjM, axis=1).astype(int)       
            outNums[outNums > maxRegulates] = maxRegulates 
            self.out_nums = outNums   

            for outNum in outNums:
                outDegs[outNum] += 1

        self.out_degs = outDegs 
        self.out_degs_dist = outDegs/self.nNodes   

        return self.out_degs_dist  

    def updateOutDegs(self, diff_a, edge_diff):   
        a_deg_before = self.out_nums[diff_a]  
        a_deg_after = a_deg_before + edge_diff    

        if a_deg_after > self.maxRegs: 
            a_deg_after = self.maxRegs  

        self.out_degs[a_deg_before] = self.out_degs[a_deg_before] - 1    
        self.out_degs[a_deg_after] = self.out_degs[a_deg_after] + 1   

        self.out_degs_dist = self.out_degs/self.nNodes        

    #returns distribution of nodes out-degrees
    def getOutDegs(self):
        if self.out_degs_dist is None: 
            self.setOutDegs() 
        return self.out_degs_dist  

    #set edge probability 
    def setEdgeProb(self): 
        self.sum_adj = np.sum(self.adjM)       
        self.edge_prob = self.sum_adj/self.nNodesSquared        

    #update edge probability       
    def updateEdgeProb(self, edge_diff):           
        self.sum_adj = self.sum_adj + edge_diff        
        self.edge_prob = self.sum_adj/self.nNodesSquared          

    #returns edge probability between two nodes  
    def getEdgeProb(self):  
        if self.edge_prob is None: 
            self.setEdgeProb() 
        return self.edge_prob   
    
    #construct adjacency matrix based on provided reference file	   
    def constructReferenceNetwork(self, file, geneIndices=None, geneNames=None):    
        if geneIndices is None: 
            geneIndices = {}   
            geneNames = {}   
            gene_ind = 0 

            try:
                with open(file) as fd:
                    rd = csv.reader(fd, delimiter="\t", quotechar='"') 		
                    for row in rd:
                        g1 = row[0]
                        g2 = row[1]   

                        if g1 not in geneIndices:
                            geneIndices[g1] = gene_ind
                            geneNames[gene_ind] = g1
                            gene_ind = gene_ind + 1 
                        if g2 not in geneIndices: 
                            geneIndices[g2] = gene_ind 
                            geneNames[gene_ind] = g2 
                            gene_ind = gene_ind + 1  
 
            except:
                print(f"Error while reading {file}!")  
            
        nNodes = len(geneIndices)  

        adj_matrix = np.zeros((nNodes, nNodes)).astype(int) 	                    
        try:    
            with open(file) as fd:
                rd = csv.reader(fd, delimiter="\t", quotechar='"')  	
                for row in rd:   
                    g1 = row[0] #source 
                    g2 = row[1] #destination  
                    edge = int(row[2])      

                    if edge == 1:

                        if g1 not in geneIndices or g2 not in geneIndices:
                            sys.exit("Error: gene names do not match!")  

                        i = geneIndices[g1]  
                        j = geneIndices[g2]      
                        adj_matrix[i, j] = edge     
        except:
            print(f"Error while reading {file}!")
        
        self.setAdjacencyMatrix(adj_matrix)   
        self.setTriadicCensus()   
        
        self.geneIndices = geneIndices  
        self.geneNames = geneNames         
        self.nNodes = (self.adjM.shape)[0]      
        self.nNodesSquared = self.nNodes *self.nNodes    	


class GeneticSolver:
    #nGen ... number of generations
    #nSub ... number of subjects - population size
    #cxP  ... crossover probability
    #mutP ... mutation probability - defined as a ratio of expected changes in adjacency matrix      
    #networkProperties ... network properties extracted from reference networks, expression data and user defined 
    #initialPop ... list containing modes to generate initial population 
    #initialPopProb ... mode probabilities for constructing initial generation    
    def __init__(self, networkProperties, obj2Weights = None, nGen=10, nSub=1000, cxP=1, mutP=1, initialPop = [3, 5], initialPopProb = [0.99, 0.01], exactNetworksIndices = None):                         
        self.netProperties = networkProperties     
        self.nNodes = self.netProperties.nNodes    
        self.nNodesSquared = self.nNodes*self.nNodes 

        self.obj2Weights = obj2Weights    

        self.exactNetworksIndices = exactNetworksIndices 
        
        self.nGen = nGen        
        self.nSub = nSub    
        self.cxP = cxP    
        self.mutP = mutP                       
        self.plotParetoPerGeneration = False                 
        self.plotPopulationPerGeneration = False                                              

        self.initialPop = initialPop     

        if 7 in initialPop: 
            print("Warning: gene names must match when constructing subjects directly from reference networks!")       

        self.initialPopProb = initialPopProb                                                                                                         
        #self.initialPop ... modes of generating initial population    
        #0 ... random (equal probability for ede/non-edge)
        #1 ... random (by folowing distribution of number of regulators)
        #2 ... based on reg. weights (select proportionate to edge probability and distribution of number of regulators)
        #3 ... based on reg. weights (select top k regulators based on distribution of number of regulators - in-degree)  
        #4 ... based on reg. weights (select top k regulations based on distribution of number of regulations - out-degree)           
        #5 ... based on top k ranked regulations, k is obtained from expected number of edges  
        #6 ... based on reg. weights and given threshold, if threshold is not given use dynamic threshold 
        #7 ... extracted from reference networks
        #list of modes ... each subject is generated with randomly selected mode     
                 
        #create multiobjective fitness to maximize objective (1) and minimize it (-1)  
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))         
        creator.create("Individual", Network, fitness=creator.FitnessMulti)       

        toolbox = base.Toolbox()        
        toolbox.register("individual", self.generate_subject)
        toolbox.register("population", self.generate_population)
        toolbox.register("evaluate", self.eval_subject)  
        toolbox.register("mate", self.crossover)  
        toolbox.register("mutate", self.mutation)     
        toolbox.register("select", tools.selNSGA2)  
        toolbox.register("sortNondominated", tools.sortNondominated)   

        #cpu_count = multiprocessing.cpu_count()
        #print(f"Available workers {cpu_count}") 
        #pool = multiprocessing.Pool(cpu_count) 
        #toolbox.register("map", pool.map) 
        self.toolbox = toolbox
        
    def run(self, debug = False):     

        #generate initial population 
        population = self.toolbox.population() 
        print("Initial population generated!")  

        if debug:
            print("Number of unique subjects: " + str(len(getUniqueSubjects(population))))  

        #evaluate initial population 
        fitnesses = self.toolbox.map(self.toolbox.evaluate, population)  
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit        

        for gen in range(self.nGen):

            start = time.time()
            print(f"NSGA-II generation {gen + 1}")     

            if self.plotPopulationPerGeneration:
                scatterPlotSubjects(population)      

            offspring = algorithms.varAnd(population, self.toolbox, cxpb=self.cxP, mutpb = self.mutP)       
            population.extend(offspring) #union of population and offspring         

            # Evaluate subjects with an invalid fitness 
            invalid_sub = [sub for sub in population if not sub.fitness.valid] 
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_sub)
            for ind, fit in zip(invalid_sub, fitnesses):
                ind.fitness.values = fit       

            if debug: 
                print(f"Length of invalid subjects is {len(invalid_sub)}") 

                avgFit = [0, 0]   
                for ind, fit in zip(invalid_sub, fitnesses):
                    avgFit[0] = avgFit[0] + fit[0]
                    avgFit[1] = avgFit[1] + fit[1]  
            
                avgFit[0] = avgFit[0]/len(invalid_sub) 
                avgFit[1] = avgFit[1]/len(invalid_sub)  
                print("Average fitness is:")
                print(avgFit)     

            #Tournament selection based on dominance (D) between two individuals
            #If the two individuals do not interdominate the selection is made based on crowding distance (CD)  
            population = self.toolbox.select(population, self.nSub)          

            if self.plotParetoPerGeneration:
                fronts = self.toolbox.sortNondominated(population, self.nSub, first_front_only = True) 
                paretoFront = fronts[0] 
                scatterPlotSubjects(paretoFront, gen=gen)    
            end = time.time()  
            elapsed = end - start 
            print(f"Generation: {elapsed} seconds elapsed")  
        start = time.time()  
        fronts = self.toolbox.sortNondominated(population, self.nSub, first_front_only = False)  
        end = time.time()  
        elapsed = end - start
        if debug:        
            print(f"Sorting individuals into nondominated levels: {elapsed} seconds elapsed") 
        #scatterPlotSubjects(fronts[0])        
        return fronts          

    def eval_subject(self, subject, debug = False):      
        netProperties = self.netProperties  
        #in-degree distribution    
        avgInDegs = netProperties.avgInDegs 
        #out-degree distribution    
        avgOutDegs = netProperties.avgOutDegs        
        #favorize less sparsely connected networks  
        expEdgeProb = 1*netProperties.expEdgeProb  
        avgTriC = netProperties.avgTriC    
        rankedDictionary = netProperties.rankedDictionary        

        adjM = subject.getAdjacencyMatrix()    
        indcs0 = np.where(adjM == 0)
        indcs1 = np.where(adjM == 1)  
        nCon = self.nNodesSquared      

        nonRegulations = len(indcs0[0])  
        regulations = len(indcs1[0])        

        sumN = regulations*(regulations + 1)/2 
        sumK = nonRegulations*(regulations + nCon + 1)/2    
        rankedList = [rankedDictionary[(a,b)] for (a,b) in zip(indcs1[0], indcs1[1])]   
        nonRankedList = [rankedDictionary[(a,b)] for (a,b) in zip(indcs0[0], indcs0[1])]   
        obj1 = sum(rankedList)/sumN - sum(nonRankedList)/sumK                     

        obj2 = 0 
        obj2List = []  
        if self.obj2Weights is not None: 
            obj2Weights = self.obj2Weights
            
            if debug:   
                print(f"Values for weights of topological properties: {obj2Weights}")   
        else: 
            #if weights for topological loss functions is not defined use default values
            obj2Weights = np.array([0.285, 0.0604, 0.7872, 0.3377])                                        

        #cost functions based on comparison of distributions are based on overlap score     
        outDegDist = subject.getOutDegs()          
        outDegOverlap = np.minimum(avgOutDegs, outDegDist)
        outDegCost = 1 - np.sum(outDegOverlap)    
        obj2List.append(outDegCost)       

        inDegDist = subject.getInDegs()     
        inDegOverlap = np.minimum(avgInDegs, inDegDist)    
        inDegCost = 1 - np.sum(inDegOverlap)           
        obj2List.append(inDegCost)               
        
        triC = subject.getNormalisedTriadicCensus()          
        triOverlap = np.minimum(avgTriC,triC)              
        triCost = 1 - np.sum(triOverlap)      
        obj2List.append(triCost)         

        eProb = subject.getEdgeProb()    
        eProbCost = np.abs(expEdgeProb - eProb)
        if eProb < expEdgeProb:        
            eProbCost = eProbCost/expEdgeProb 
        else:  
            eProbCost = eProbCost/(1 - expEdgeProb)      
               
        obj2List.append(eProbCost)                            

        obj2List = np.array(obj2List)                    
        obj2 = np.dot(obj2Weights, obj2List) 

        return obj1, obj2         

    #initialize individual   
    def generate_population(self):    
        popList = []
        for i in range(self.nSub):
            popList.append(self.generate_subject()) 
        return popList
    
    def generateRandomAdj(self):
        return np.random.randint(2, size=(self.nNodes, self.nNodes))              

    def generateTopRegulationsAdj(self, stochasticEdgeNumber = True): 
        netProperties = self.netProperties
        expectedEdgeProb = netProperties.expEdgeProb     
        (regulatorIndices, regulatedIndices, _) = netProperties.rankedListWeightsTuple 
        totalEdges = self.nNodes*self.nNodes
        regNum =  round(expectedEdgeProb*totalEdges)  
        
        if stochasticEdgeNumber:
            varNum = int(regNum/4)
            regNum = regNum +  np.random.randint(varNum) - round(varNum/2.0)      

        adj = np.zeros((self.nNodes, self.nNodes))   
        adj[(regulatorIndices[:regNum], regulatedIndices[:regNum])] = 1    
        return adj    


    #generate random adjacency matrix based on regulators degree distribution as in reference networks
    #if regWeights is provided select regulators based on weights 
    def generateRandomRegDistAdj(self, regWeights = None):  
        netProperties = self.netProperties
        regWeights = netProperties.regWeights 
        avgRegDegs = netProperties.avgRegDegs   
        maxRegs = netProperties.maxRegs
        adj = np.zeros((self.nNodes, self.nNodes)) 
        
        #for every regulator randomly choose number of regulations 
        for i in range(self.nNodes):       
            #select number of regulators given avgRegDegs 
            regNum = np.random.choice(maxRegs + 1, 1, replace=False, p=avgRegDegs)[0]
            #select and assign regulators     
            regNums = np.random.choice(self.nNodes, regNum, replace=False, p = regWeights[:,i] if regWeights is not None else None)          
            adj[regNums, i] = 1 

        return adj       
    
    #generate adjacency matrix based on threshold
    def generateRegWeightsThresholdAdj(self, threshold = None):     
        netProperties = self.netProperties
        regWeights = netProperties.regWeights 
     
        if threshold is None: 
            threshold = 1.5/self.nNodes  

        return (regWeights >= threshold).astype(int)     
    
    #generate adjacency matrix based on regWeights   
    def generateRegWeightsAdj(self):
        netProperties = self.netProperties
        regWeights = netProperties.regWeights  
        return self.generateRandomRegDistAdj(regWeights = regWeights)      

    #select top k regulators based on distribution of number of regulators 
    def generateRegWeightsTopKAdjInDegree(self):   
        netProperties = self.netProperties
        avgInDegs = netProperties.avgInDegs   
        maxRegs = netProperties.maxRegs 
        regWeightsSortIndicesInDegree = netProperties.regWeightsSortIndicesInDegree 

        adj = np.zeros((self.nNodes, self.nNodes))

        for i in range(self.nNodes):         
            #select number of regulators given probs
            regNum = np.random.choice(maxRegs + 1, 1, replace=False, p=avgInDegs)[0]
            #use at least one regulator 
            if regNum == 0:
                regNum = 1

            indcs = regWeightsSortIndicesInDegree[:,i][-regNum:] #select last regNum regulators with highest scores     
            
            for ind in indcs:   
                adj[ind, i] = 1             
        return adj 

    def generateRegWeightsTopKAdjOutDegree(self): 
        netProperties = self.netProperties   
        avgOutDegs = netProperties.avgOutDegs  
        maxRegs = netProperties.maxRegs    
        regWeightsSortIndicesOutDegree = netProperties.regWeightsSortIndicesOutDegree  

        adj = np.zeros((self.nNodes, self.nNodes))

        for i in range(self.nNodes):
            regNum = np.random.choice(maxRegs + 1, 1, replace=False, p=avgOutDegs)[0]

            indcs = regWeightsSortIndicesOutDegree[i,:][-regNum:] #select last regNum regulators with highest scores 
            
            for ind in indcs:    
                adj[i, ind] = 1             
        return adj     

    #generate adjacency matrix based on reference networks 
    #TO DO ... generate adjacency matrix based on matched gene names  
    def generateRefNetsAdj(self):
        netProperties = self.netProperties

        #select random reference network as base from the list of exact reference networks 
        reference_indices = self.exactNetworksIndices  
        if reference_indices is None:
            sys.exit("Error: list of exact networks indices missing!") 

        indx = np.random.choice(reference_indices)     

        refNet = netProperties.referenceNets[indx]     
        adjRef = refNet.getAdjacencyMatrix().copy() 
        nNodesRef = (adjRef.shape)[0]      

        if nNodesRef != self.nNodes: 
            sys.exit("Error: number of genes in reference network do not match time series data!")      

        return adjRef           

    #0 ... random (equal probability for ede/non-edge)
    #1 ... random (by folowing distribution of number of regulators)
    #2 ... based on reg. weights (select proportionate to edge probability and distribution of number of regulators)
    #3 ... based on reg. weights (select top k regulators based on distribution of number of regulators - in-degree)  
    #4 ... based on reg. weights (select top k regulations based on distribution of number of regulations - out-degree)           
    #5 ... based on top k ranked regulations, k is obtained from expected number of edges  
    #6 ... based on reg. weights and given threshold, if threshold is not given use dynamic threshold 
    #7 ... extracted from reference networks  
    def generateInitialAdjMatrix(self, mode):
        if mode == 0:
            return self.generateRandomAdj() 
        elif mode == 1:
            return self.generateRandomRegDistAdj() 
        elif mode == 2:
            return self.generateRegWeightsAdj() 
        elif mode == 3:
            return self.generateRegWeightsTopKAdjInDegree() 
        elif mode == 4:
            return self.generateRegWeightsTopKAdjOutDegree()     
        elif mode == 5: 
            return self.generateTopRegulationsAdj(stochasticEdgeNumber = False)     
        elif mode == 6:
            return self.generateRegWeightsThresholdAdj() 
        elif mode == 7:
            return self.generateRefNetsAdj()  
        else:
            print(f"Invalid parameter for generation of initial population. Switching to random subjects following distribution of number of regulators from reference networks (mode = 1).")   
            return self.generateInitialAdjMatrix(1)           

    #generate idividual, i.e. regulatory network represented as adjacency matrix
    def generate_subject(self): 
        mode = self.initialPop      

        if isinstance(mode, list): 
            mode = np.random.choice(mode, p = self.initialPopProb)    

        ind = creator.Individual() #Individual inherits all properties of Network   
        ind.setnNodes(self.nNodes) 
        #set adjacency matrix and calculate necesssary properties   
        ind.setAdjacencyMatrix(self.generateInitialAdjMatrix(mode)) 
        ind.setTriadicCensus()  
        ind.setMaxRegs(self.netProperties.maxRegs)    
        ind.setInDegs()     
        return ind      

    #mutate subject by addition or deletion of edges  
    def mutation(self, sub):     
        adjM = sub.getAdjacencyMatrix()           

        #add edge or remove edge with same probability 
        add_edge = True
        edge_diff = 1
        rnd = np.random.rand()  
        if rnd < 0.5:
            add_edge = False
            edge_diff = -1      

        indices = np.where(adjM == int(not add_edge))  
        ind_num = np.random.choice(len(indices[0]))            
        row = indices[0][ind_num]  
        column = indices[1][ind_num]   

        adjM[row, column] = int(add_edge)   

        #limit number of regulators   
        maxRegs = self.netProperties.maxRegs    
        if np.sum(adjM[:, column]) > maxRegs:  
            adjM[row, column] = 0        

        else:      
            #set NetworkX graph object     
            sub.setGraph()   
            #dynamically update topological properties   
            sub.updateEdgeProb(edge_diff)
            sub.updateOutDegs(row, edge_diff) 
            sub.updateInDegs(column, edge_diff)              
            sub.diff_a = row   
            sub.diff_b = column    
            sub.updateTriadicCensus()                  

        return sub,    
    
    #apply a crossover by swaping columns (regulators) in adjacency matrix #and rows    
    #The general rule for crossover operators is that they only mate individuals, this means that an independent copies must be made prior 
    #to mating the individuals if the original individuals have to be kept or are references to other individuals (see the selection operator).
    def crossover(self, sub1, sub2):   
        nNodes = self.nNodes             
        cxNum = np.random.randint(1, nNodes)      
        crossColumns = np.random.choice(nNodes, cxNum, replace=False)       

        adjSub1 = sub1.getAdjacencyMatrix()      
        adjSub2 = sub2.getAdjacencyMatrix()       

        tmpCols = adjSub1[:, crossColumns]  
        adjSub1[:, crossColumns] = adjSub2[:, crossColumns]        
        adjSub2[:, crossColumns] = tmpCols    

        sub1.setGraph()     
        sub1.setTriadicCensus()  
        sub2.setGraph()         
        sub2.setTriadicCensus()     

        sub1.setEdgeProb() 
        sub2.setEdgeProb()    

        sub1.setOutDegs()
        sub2.setOutDegs()      

        tmp_nums = sub1.in_nums[crossColumns] 
        sub1.in_nums[crossColumns] = sub2.in_nums[crossColumns]  
        sub2.in_nums[crossColumns] = tmp_nums 

        sub1.setInDegs()        
        sub2.setInDegs()            

        return sub1, sub2     

def getGoldNetwork(goldpath, geneIndices, geneNames):
    net = Network(refFile = goldpath, geneIndices = geneIndices, geneNames = geneNames)    
    return net       

#calculate metrices for given adjacency matrices	
def getMetrics(y_true, y_pred):     
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = confusion_matrix.ravel()                       
    TPR = TP/(TP + FN) 
    TNR = TN/(TN + FP)          

    precision = TP/(TP + FP)  
    f1 = 2*(precision*TPR)/(precision + TPR)     
    accuracy = (TP+TN)/(TP+TN+FP+FN)              
    bm = TPR + TNR - 1.     
    mcc = (TN*TP - FN*FP)/(np.sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN)))    

    return {"Accuracy": accuracy, "Precision": precision,"Recall": TPR,"F1": f1,"MCC": mcc,"BM": bm}  

def getMetricsList(pop, adj_true):  
    lst = []
    y_true = adj_true.ravel()           
    
    for sub in pop:
        #use ravel instead of flatten, since y_true and y_pred are not modified       
        y_pred = sub.getAdjacencyMatrix().ravel()      
        metrics = getMetrics(y_true, y_pred)      
        lst.append(metrics)     
    return lst         

def scatterPlotSubjects(paretoFront, labels = None, savePath = None, gen=0):   
    fits = [sub.fitness.values for sub in paretoFront]    

    """
    dump_path = os.path.join(".", "results", "dump" + str(gen) + ".pkl")     
    file = open(dump_path, "wb")     
    pickle.dump(fits, file)  
    file.close()      
    """ 
    
    fits = list(zip(*fits))  
    fit1 = fits[0]  
    fit2 = fits[1]          

    plt.figure()      

    plt.scatter(fit1, fit2)
    plt.xlabel(r'Alignment with ranked regulations ($z_1$)') 
    plt.ylabel(r'Topological similarity ($z_2$)')                

    if labels is not None:
        # add labels to all points
        for indx, (xi, yi) in enumerate(zip(fit1, fit2)):
            plt.text(xi, yi, labels[indx], va='bottom', ha='center')  

    if savePath is not None: 
        pickle.dump()
        plt.savefig(savePath + "_scatter.pdf")     
        plt.clf()  
    else: 
        plt.show()           

def getDistances(paretoFront):
    fits = [sub.fitness.values for sub in paretoFront]
    fitnesses = np.array(list(zip(*fits)))     
    min_objectives = np.min(fitnesses, axis=1) 
    max_objectives = np.max(fitnesses, axis=1)    
    scale = max_objectives - min_objectives 
    #transalte
    fitnesses = fitnesses - min_objectives[:, np.newaxis] 
    #scale 
    fitnesses = fitnesses/(scale[:, np.newaxis])      
    distances = np.sqrt(np.sum(fitnesses*fitnesses, axis=0)) 
    return np.vstack([fitnesses, distances])      

# Class of expected network properties extracted from reference networks including  
# referenceNets ... list of reference networks
# regulatory weights ... numpy matrix containing weights of regulators  
# geneIndices ... gene names obtained from time series or steady state data 
# maxRegs ... maximum number of regulators        
class NetworkProperties: 
    def __init__(self, referenceNets, regWeights, nNodes, geneIndices, geneNames, exactNetworksIndices = None, maxRegs=10):
        self.referenceNets = referenceNets
        self.nNodes = nNodes 
        self.geneIndices = geneIndices    
        self.geneNames = geneNames 
        self.maxRegs = maxRegs 
        self.avgInDegs = self.getAvgInDegs()     
        self.avgOutDegs = self.getAvgOutDegs()   
        self.expEdgeProb = self.getExpectedEdgeProbability()   
        self.avgTriC = self.getAvgTriadicCensus() 
        self.exactNetworksIndices = exactNetworksIndices        
        
        self.regWeights = regWeights
        self.regWeightsSortIndicesInDegree = np.argsort(regWeights, axis=0)
        self.regWeightsSortIndicesOutDegree = np.argsort(regWeights, axis=1)
        #tuple of three lists (regulatorIndices, regulatedIndices, regWeights) 
        self.rankedListWeightsTuple = self.getRankedList() 
        #create dictionary of regulations ranks for fast lookup  
        rankedTuple = list(zip(self.rankedListWeightsTuple[0], self.rankedListWeightsTuple[1]))
        rankedTuple = list(zip(rankedTuple, list(range(1,len(rankedTuple)+1))))  
        self.rankedDictionary = dict(rankedTuple)         

    def getRankedList(self):
        regWeights = self.regWeights 
        #negate array to sort descending   
        indices = np.argsort(-regWeights, axis = None) 
        indices = np.unravel_index(indices, regWeights.shape) 
        regulatorIndices = indices[0]
        regulatedIndices = indices[1]
        return (regulatorIndices, regulatedIndices, regWeights[indices])


    def getExpectedEdgeProbability(self):
        eProb = 0 
        for refNet in self.referenceNets:
            eProb = eProb + refNet.getEdgeProb()
        return eProb/len(self.referenceNets)     

    def getAvgTriadicCensus(self):
        triC = np.zeros(13) #number of triads    
        for refNet in self.referenceNets:
            triC_i = refNet.getTriadicCensus()    
            triC = triC + triC_i     
        sumTriC = np.sum(triC)  
        triC = triC/sumTriC   
        return triC    

    def getAvgInDegs(self):
        avgInDegs = np.zeros(self.maxRegs + 1)
        for refNet in self.referenceNets:
             avgInDegs = avgInDegs + refNet.getInDegs() 

        inSum = np.sum(avgInDegs)     
        return avgInDegs/inSum    

    def getAvgOutDegs(self):   
        avgOutDegs = np.zeros(self.maxRegs + 1)
        for refNet in self.referenceNets: 
             avgOutDegs = avgOutDegs + refNet.getOutDegs()       

        outSum = np.sum(avgOutDegs)   
        return avgOutDegs/outSum    

#calculate semi tensor product for Boolean expressions  
def getBooleanBySemiTensorProduct(mf, X): 
    value = mf 
    
    numCols = np.shape(mf)[1]   
    numRegs = np.shape(X)[1]
    #calculate semi tensor product for each variable in b 
    for i in range(numRegs): 
        numCols = numCols//2 
        b0 = X[0,i] 
        b1 = X[1,i] 
        value = value[:,:numCols]*b0 + value[:,numCols:]*b1         
    return value   

def getDecimalFromBinary(row):
    return int(row, 2)                 

def getBinaryFromDecimal(value, nBits):
    return [int(i) for i in bin(value)[2:].zfill(nBits)]   

def getUniqueSubjects(subjects):
    unique_subjects = []  
    unique_subjects_set = set()    

    for subject in subjects:
        adj = subject.getAdjacencyMatrix() 
        adj_byte_array = adj.tobytes()   

        if adj_byte_array not in unique_subjects_set:
            unique_subjects.append(subject)
            unique_subjects_set.add(adj_byte_array)

    return unique_subjects

#Main class for context specific inference
class ContextSpecificDecoder:
    #timeSeriesPath   ... file path to time series data
    #steadyStatesPaths ... file paths list to steady states data
    #referenceNetPaths ... file paths list to reference networks 
    #maxRegs ... maximum number of regulators  
    #obj2Weights ... weights for 2nd objective (topological properties) 
    #decimal ... decimal separator, default is '.'   

    def __init__(self, timeSeriesPath, steadyStatesPaths=None, referenceNetPaths = None, goldNetPath = None, binarisedPath = None, savePath = None, maxRegs = 10, debug = False, obj2Weights = None, initialPop = None, initialPopProb = None, exactNetworksIndices = None, decimal = "."):         
        self.decimal = decimal   
        
        if initialPop is None:
            initialPop = [3, 5]

        if initialPopProb is None:
            initialPopProb = [0.99, 0.01] 

        #set empty list to None 
        if isinstance(steadyStatesPaths, list):  
            if not steadyStatesPaths:  
                steadyStatesPaths = None    

        self.timeSeriesPath = timeSeriesPath  
        timeSeriesDf = self.readFiles([timeSeriesPath]) 

        self.timeSeriesDf = timeSeriesDf 
        #set experiment count
        self.timeSeriesDf["Experiment"] = 0  
        times = self.timeSeriesDf[["Time"]]  
        times_shifted = times.shift(periods=1)    
        
        self.timeSeriesDf["Experiment"] = (times["Time"] < times_shifted["Time"]).astype(int)    
        self.timeSeriesDf["Experiment"] = self.timeSeriesDf["Experiment"].cumsum() 
        self.experiments = self.timeSeriesDf["Experiment"].max() + 1 

        nNodes = len(self.timeSeriesDf.columns) - 2 #substract two for Experiment and Time
        print("Number of nodes is " + str(nNodes)) 

        geneNamesList = list(self.timeSeriesDf.columns)   
        if "Time" in geneNamesList:
            geneNamesList.remove("Time")     
        if "Experiment" in geneNamesList: 
            geneNamesList.remove("Experiment")   

        self.geneIndices = {value: indx for indx, value in enumerate(geneNamesList)} 
        self.geneNames =  {indx: value for indx, value in enumerate(geneNamesList)} 


        self.goldNetPath = goldNetPath          
        self.savePath = savePath    
        self.obj2Weights = obj2Weights     

        if binarisedPath is None:
            tmp_path = timeSeriesPath 
            binarisedPath = tmp_path.rsplit(".", 1)[0] + "_binarised.tsv"      

        if os.path.exists(binarisedPath): 
            #if binarised file exists read from file   
            binarised_df = pd.read_csv(binarisedPath, sep="\t") 
        else:
            #else binarise time series data and save       
            binarised_df = self.binarise(self.timeSeriesDf)    
            binarised_df.to_csv(binarisedPath, index=False, sep="\t")      

        #if binarised dataframe does not contain columns Time and Experiment append them
        if not "Time" in binarised_df:
            binarised_df["Time"] = self.timeSeriesDf["Time"]  
            
        if not "Experiment" in binarised_df: 
            binarised_df["Experiment"] = self.timeSeriesDf["Experiment"]  

        self.binarised_df = binarised_df             

        self.steadyStatesDf = None    
        self.steadyStatesPaths = steadyStatesPaths     
        if steadyStatesPaths is not None:    
            steadyStatesDf = self.readFiles(steadyStatesPaths)   
            self.steadyStatesDf = steadyStatesDf 
            if nNodes != len(self.steadyStatesDf.columns):
                sys.exit("Error: Gene number mismatch between time series and steady states!")

        method_args = {}   

        regWeights = self.getRegulatoryWeights(**method_args)      
        referenceNets = self.getReferenceNetworks(referenceNetPaths, maxRegs, self.geneIndices, self.geneNames, exactNetworksIndices = exactNetworksIndices)                   
        netProperties = NetworkProperties(referenceNets, regWeights, nNodes, self.geneIndices, self.geneNames, exactNetworksIndices = exactNetworksIndices, maxRegs = maxRegs)      
        self.netProperties = netProperties      
        self.qm = QuineMcCluskey(use_xor = False)     
        self.genSolver = GeneticSolver(netProperties, obj2Weights=obj2Weights, initialPop = initialPop, initialPopProb = initialPopProb, exactNetworksIndices = exactNetworksIndices)               

    def test(self, subjects, debug = False):    
        start = time.time()   
        gold_standard = getGoldNetwork(self.goldNetPath, self.geneIndices, self.geneNames)     
        gold_standard_adj = gold_standard.getAdjacencyMatrix() 

        end = time.time() 
        elapsed = end - start 
        if debug:
            print(f"Extracting gold standard network: {elapsed} seconds elapsed!")    

        start = time.time()  
        metrics = getMetricsList(subjects, gold_standard_adj) 
        end = time.time()  
        elapsed = end - start
        if debug: 
            print(f"Calculating metrics for subjects: {elapsed} seconds elapsed!")             

        start = time.time()     
        distances = getDistances(subjects)     
        end = time.time()  
        elapsed = end - start 
        if debug:
            print(f"Calculating distances: {elapsed} seconds elapsed!")   

        baseAdj = self.genSolver.generateTopRegulationsAdj(stochasticEdgeNumber = False)      
        baseMetrics = getMetrics(gold_standard_adj.ravel(), baseAdj.ravel())           

        baseNetwork = Network(nNodes = baseAdj.shape[0], adjM = baseAdj)   
        
        if debug:
            print("Regulates degrees") 
            print(gold_standard.getOutDegs())  
            print(baseNetwork.getOutDegs())     
            print(self.netProperties.avgOutDegs)   

            print("Regulatory degrees") 
            print(gold_standard.getInDegs())  
            print(baseNetwork.getInDegs())    
            print(self.netProperties.avgInDegs)    

            print("Triadic census") 
            print(gold_standard.getNormalisedTriadicCensus())    
            print(baseNetwork.getNormalisedTriadicCensus())       
            print(self.netProperties.avgTriC)      

            print("Edge probability")    
            print(gold_standard.getEdgeProb())    
            print(baseNetwork.getEdgeProb())       
            print(self.netProperties.expEdgeProb)  

            print(baseNetwork.getAdjacencyMatrix())         

        return distances, metrics, baseMetrics          
   

    #returns Boolean model that best matches training data         
    #bNetworks ... list of Boolean networks   
    #bin_df, shift_bin_df, experiments_df    
    def getBestBooleanModel(self, bNetworks, bin_df, shift_bin_df, experiments_df, distances):       

        print("Evaluating Boolean networks!")   

        start = time.time() 
        iterableList = [(bNetwork, bin_df, experiments_df, self.geneNames, self.experiments, index) for index, bNetwork in enumerate(bNetworks)]    
        pool = multiprocessing.Pool(4) #(multiprocessing.cpu_count())     
        accuracies = pool.starmap(getDynamicAccuracy, iterableList)   
        pool.close()       

        bestAcc, bestInd = accuracies[0]  
        bestDist = distances[bestInd]       

        for accuracy, index in accuracies:
            dist = distances[index]    
            if (accuracy > bestAcc) or (accuracy == bestAcc and dist < bestDist):
                bestAcc = accuracy 
                bestInd = index  
                bestDist = dist      
                
        bestNetwork = bNetworks[bestInd]  

        #traverse Boolean functions and replace constants with functions from bNetworks
        for index, bFun in enumerate(bestNetwork):
            regulators = bFun[1]
            bexpr = bFun[4]
            target = bFun[0] 
            if not regulators.size > 0 or bexpr == f"Gene{target + 1} = 0" or bexpr == f"Gene{target + 1} = 1":

                print("************************")
                print("Constant for Boolean function")
                print(bFun) 
            
                bestDist = distances[bestInd]
                bestFun = bNetworks[bestInd][index]  
                found = False

                for dist_index, bNetwork in enumerate(bNetworks):
                    newFun = bNetwork[index] 
                    new_regulators = newFun[1]
                    new_distance = distances[dist_index] 
                    new_bexpr = newFun[4] 
                    new_target = newFun[0]    

                    if new_regulators.size > 0 and new_bexpr != f"Gene{new_target + 1} = 0" and new_bexpr != f"Gene{new_target + 1} = 1":
                        if not found or (new_distance < bestDist):
                            bestFun = newFun
                            bestDist = new_distance
                            found = True 

                #assign new Boolean function
                bestNetwork[index] = bestFun

                print("New Boolean function")
                print(bestFun) 
                print("************************") 

        #print(bestNetwork)

        end = time.time() 
        elapsed = end - start  
        print(str(elapsed) + " seconds elapsed!")  

        print("Best accuracy after multiprocessing map function")
        print(bestAcc) 
        print("------------------------------------------------") 

        """
        #sequential implementation 
        start = time.time()
        bestAcc, _ = getDynamicAccuracy(bNetworks[0], bin_df, experiments_df, self.geneNames, self.experiments, 0)     
        bestNetwork = bNetworks[0]   
        #iterate through every network except first  
        for bNetwork in bNetworks[1:]:  
            myAcc, _ = getDynamicAccuracy(bNetwork, bin_df, experiments_df, self.geneNames, self.experiments, 0)       
            print(bestAcc)          
            if myAcc > bestAcc: 
                bestAcc =  myAcc   
                bestNetwork = bNetwork  
        end = time.time() 
        elapsed = end - start  
        print(str(elapsed) + " seconds elapsed!")        
        """     

        return bestNetwork, bestAcc                        
                         

    #infers Boolean networks   
    def inferBooleanNetworks(self, subjects, bin_df, shift_bin_df, experiments_df):  
        #list of Boolean networks        
    
        nNodes = subjects[0].nNodes    

        print("Inferring Boolean networks!")   
        start = time.time()
        iterableList = [(subject.adjM, nNodes, bin_df, shift_bin_df, experiments_df, self.geneNames, self.qm, self.experiments) for subject in subjects]        
        pool = multiprocessing.Pool(4) #(multiprocessing.cpu_count() )     
        bNetworks = pool.starmap(inferBooleanNetwork, iterableList)  
        pool.close()         
        end = time.time()      
        elapsed = end - start  
        print(str(elapsed) + " seconds elapsed!")       
        

        """
        #sequential implementation 
        bNetworks = []  
        i = 0 
        start = time.time() 
        for subject in subjects:
            i = i + 1 
            #print("Inferring Boolean network")   
            bNetwork = inferBooleanNetwork(subject.adjM, nNodes, bin_df, shift_bin_df, experiments_df, self.geneNames, self.qm, self.experiments)    
            bNetworks.append(bNetwork)    
        end = time.time()  
        elapsed = end - start  
        print(str(elapsed) + " seconds elapsed!")
        """
        
        return bNetworks 

    def getUniqueRegulators(self, subjects):
        nNodes = self.netProperties.nNodes  
        unique_regulators = {i: ([], set()) for i in range(nNodes)}   

        for subject in subjects:      
            adj = subject.getAdjacencyMatrix()  
            for i in range(nNodes):
                column = adj[:, i] 
                column_byte_array = column.tobytes()  

                regulators_list, regulators_byte_array_set = unique_regulators[i] 
                if column_byte_array not in regulators_byte_array_set:
                    regulators_list.append(column)  
                    regulators_byte_array_set.add(column_byte_array)       
        
        return unique_regulators

    #subjects ... list of unique networks 
    #bin_df   ... binarised time series data
    #shift_bin_df ... shifted binarised time series data 
    #experiments_df ... dataframe denoting experiments and time points 
    def getBestCombinedBooleanModel(self, subjects, bin_df, shift_bin_df, experiments_df):
        nNodes = self.netProperties.nNodes 
        b_functions = []      
        unique_regulators = self.getUniqueRegulators(subjects)     
        for target in range(nNodes):    
            regulators_list, _ = unique_regulators[target]
            min_error = len(bin_df.index)*nNodes    
            min_function = None  
            for regulator_array in regulators_list: 
                regulators, gT, bfun, bexpr, tT = inferBooleanFunction(regulator_array, target, bin_df, shift_bin_df, experiments_df, self.geneNames, self.qm, self.experiments)  

                if gT is not None:     
                    sumS = gT["T"] + gT["F"]   
                    eT = (gT[tT.value == 1]["F"]/sumS[tT.value == 1]).sum()            
                    eF = (gT[tT.value == 0]["T"]/sumS[tT.value == 0]).sum()   
                    error = eT + eF 
                else: 
                    #define dummy error smaller than the initial error value 
                    error = len(bin_df.index)*nNodes - 1         

                if error < min_error:     
                    min_error = error   
                    min_function = (target, regulators, gT, bfun, bexpr, tT)       

            b_functions.append(min_function)  
        return b_functions          

    def iterativeKmeans(self, data, d=3):     
        data = np.array(data)                
        data = np.reshape(data, (-1,1)) #reshape to array with one feature  
        while d > 0:       
            clusters = pow(2, d) 
            kmeans = KMeans(n_clusters=clusters, random_state=0).fit(data)     
            data = kmeans.cluster_centers_[kmeans.labels_]   
            d = d - 1        
        #binarize     	
        boolV = kmeans.cluster_centers_[0,0] > kmeans.cluster_centers_[1,0] 
        centers = np.array([int(boolV), int(not boolV)])      
        return pd.Series(centers[kmeans.labels_].tolist())           

    def binarise(self, timeSeriesDf):
        columns = timeSeriesDf.columns.tolist()  
        columns.remove("Time")     
        columns.remove("Experiment")             
        data = timeSeriesDf.loc[:, columns]   
        binarised_df = data.apply(self.iterativeKmeans, axis=0)   
        binarised_df = binarised_df.astype(int) 
        binarised_df = pd.concat([binarised_df, timeSeriesDf.loc[:, ["Time", "Experiment"]]], axis=1)              
        return binarised_df                                

    def getNetworkCandidates(self, debug=False):  
        return self.genSolver.run()        

    def run(self, debug=False):    
        fronts = self.getNetworkCandidates() 
        print("Number of networks in first Pareto front")   
        print(len(fronts[0]))   

        #get unique subjects of first Pareto front    
        subjects = getUniqueSubjects([sub for front in fronts for sub in front])  #getUniqueSubjects(fronts[0])          
        print("Number of unique networks in last population")     
        print(len(subjects))     

        #get distances 
        distances = getDistances(subjects)          

        binarised_df = self.binarised_df   
        #get Boolean networks from population    
        columns = binarised_df.columns.tolist()   
        columns.remove("Time")
        columns.remove("Experiment")   

        experiments_df = binarised_df[["Time", "Experiment"]]   
        bin_df = binarised_df[columns] 
        #since shifting appends nan values ints are converted to floats   
        shift_bin_df = bin_df.shift(periods = 1).fillna(0).astype(int)     
        shift_bin_df = shift_bin_df.astype(str)  

        #bNetworks ... list of Boolean networks
        ##bNetwork[i] ... [target, regulators, generalised truth table, boolean function, boolean expression, truth table]  
        bNetworks = self.inferBooleanNetworks(subjects, bin_df, shift_bin_df, experiments_df)           

        #select best network           
        best, bestAcc = self.getBestBooleanModel(bNetworks, bin_df, shift_bin_df, experiments_df, distances[2])                
        
        """
        #code for combined Boolean model
        bestCombined = self.getBestCombinedBooleanModel(subjects, bin_df, shift_bin_df, experiments_df)    
        myAcc, _ = getDynamicAccuracy(bestCombined, bin_df, experiments_df, self.geneNames, self.experiments, 0)                       
        print("Dynamic accuracy of best combined Boolean model: " + str(myAcc))   
        if myAcc > bestAcc:  
            best = bestCombined   
        """ 
            
        return best                  

    def readFiles(self, filePaths): 
        df_all = pd.DataFrame() 
        for filePath in filePaths:
            print(filePath) 
            if os.path.exists(filePath):   
                df = pd.read_csv(filePath, sep="\t", decimal=self.decimal)    
                df = df.apply(pd.to_numeric) 
                df = df.dropna() 
                df_all = pd.concat([df_all, df], ignore_index=True)   
            else: 
                print(f"Unable to read file {filePath}. File does not exists!")  
        return df_all

    def getReferenceNetworks(self, referenceNetPaths, maxRegs, geneIndices, geneNames, exactNetworksIndices=None):
        reference_nets = []  

        if exactNetworksIndices is None:
            exactNetworksIndices = [] #initialize to empty list if none 

        for ind, refPath in enumerate(referenceNetPaths):  
            if ind in exactNetworksIndices: 
                temp_geneIndices = geneIndices
                temp_geneNames = geneNames
            else:
                temp_geneIndices = None
                temp_geneNames = None  

            reference_net = Network(refFile = refPath, geneIndices = temp_geneIndices, geneNames = temp_geneNames)  
            reference_net.setMaxRegs(maxRegs)  
            if reference_net.adjM is not None:
                reference_nets.append(reference_net) 
        return reference_nets       

    #infer putative regulatory probability estimates 
    def getRegulatoryWeights(self, method="dynGENIE3", **method_args):  
        supportedMethods = ["dynGENIE3"]  
        
        if method not in supportedMethods:
            print(f"{method} is not supported! Using dynGENIE3.") 
            method = "dynGENIE3"  
        
        if method == "dynGENIE3":  
            #if nthreads argument is not defined use possible number of threads
            if not "nthreads" in method_args:
                method_args["nthreads"] = multiprocessing.cpu_count()     

            return self.run_dynGENIE3(**method_args)         

    #runs dynGENIE3
    def run_dynGENIE3(self, **method_args):
        #TS_data ... list of time series experiments as numpy arrays
        #time_points ... list of time points as one dimensional numpy arrays 
        TS_data = []    
        time_points = []         
        
        columnNames = list(self.timeSeriesDf.columns) 
        columnNames.remove("Experiment")  
        columnNames.remove("Time")       
        
        for experiment in range(self.experiments): 
            df = self.timeSeriesDf[self.timeSeriesDf["Experiment"] == experiment]      
            time_points_df = df["Time"] 
            time_points.append(time_points_df.to_numpy()) 
            df = df[columnNames]   
            TS_data.append(df.to_numpy())  

        #add steady states
        if self.steadyStatesDf is not None:
            method_args["SS_data"] = self.steadyStatesDf.to_numpy() 

        #probabilites are normalised column-wise   
        regulatory_probs, _, _, _, _ = dynGENIE3(TS_data, time_points, **method_args) 
        return regulatory_probs    

def main():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--timeSeriesPath")  
    parser.add_argument("--binarisedPath")        
    parser.add_argument("--referencePaths")   
    parser.add_argument("--outputFilePath") 
    parser.add_argument("--decimal")  
    parser.add_argument("--exactNetworksIndices") 
    parser.add_argument("--initialPop")
    parser.add_argument("--initialPopProb")  

    args =  parser.parse_args()   

    if args.timeSeriesPath is None or args.binarisedPath is None or args.referencePaths is None:   
        sys.exit("Error: Invalid arguments!")      

    decimal = args.decimal
    if decimal is None:
        decimal = "."   

    exactNetworksIndices = args.exactNetworksIndices 
    initialPop = args.initialPop
    initialPopProb = args.initialPopProb 

    referencePaths = args.referencePaths
    referencePaths = referencePaths.replace("\\'", "")
    referencePaths = ast.literal_eval(referencePaths) 

    decoder = ContextSpecificDecoder(args.timeSeriesPath, referenceNetPaths = referencePaths, binarisedPath=args.binarisedPath, obj2Weights=None, initialPop = initialPop, initialPopProb = initialPopProb, exactNetworksIndices=exactNetworksIndices, decimal=decimal)     
    best = decoder.run() 

    boolean_expressions = [] 

    for bfun in best:  
        boolean_expressions.append(bfun[4])          

    if args.outputFilePath is None:
        print("Output file path not provided! Printing to standard output.")  
        for expression in boolean_expressions:
            print(expression)  
    else: 
        with open(args.outputFilePath, "w+") as outFile:  
            outFile.writelines([expression + "\n" for expression in boolean_expressions])      

if __name__ == "__main__":    
	main()           