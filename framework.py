from dynGENIE3 import *
from deap import creator, base, tools, algorithms
#from inferelator import inferelator_workflow 
from sklearn.cluster import KMeans  
from qm import QuineMcCluskey  
from triadic_census import count_triads, count_local_triads   

import csv
import networkx as nx
import numpy as np 
import matplotlib.pyplot as plt 
import math    
#import multiprocessing
import os 
import pandas as pd 
import pickle 
import sklearn.metrics as metrics 
import sys  
import time         

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
            self.constructReferenceNetwork(refFile, geneIndices, geneNames)        

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

            """
            countTri, _ = self.countTriads() 
            if np.abs(np.sum(countTri - self.triadicCensus) > 0): 
                print("Error")   
                print(countTri - self.triadicCensus)     
            """   
                
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
    def constructReferenceNetwork(self, file, geneIndices, geneNames):  
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
    #nNodes ... network size
    #nGen ... number of generations
    #nSub ... number of subjects - population size
    #cxP  ... crossover probability
    #mutP ... mutation probability - defined as a ratio of expected changes in adjacency matrix      
    #indP ... independent probability of flipping an edge 
    #networkProperties ... network properties extracted from reference networks, expression data and user defined 
    def __init__(self, networkProperties, obj2Weights = None, nGen=10, nSub=500, cxP=1, mutP=1):                        
        
        self.netProperties = networkProperties     
        self.nNodes = self.netProperties.nNodes   
        self.nNodesSquared = self.nNodes*self.nNodes 

        self.obj2Weights = obj2Weights  
        
        self.nGen = nGen     
        self.nSub = nSub  
        self.cxP = cxP  
        self.mutP = mutP                     
        self.plotParetoPerGeneration = False           
        self.plotPopulationPerGeneration = False                                         

        self.initialPop = [3, 5]      
        self.initialPopProb = [0.99, 0.01]                                                                                                         
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
                scatterPlotSubjects(paretoFront)   
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

        #regWeights = netProperties.regWeights   
        #favorize less sparsely connected networks  
        expEdgeProb = 1*netProperties.expEdgeProb  
        avgTriC = netProperties.avgTriC   
        maxRegs = netProperties.maxRegs 
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
            obj2Weights = np.array([0.25, 0.25, 0.25, 0.25])                          

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


        if debug:
            print("outDeg cost is")
            print(outDegCost)

            print("inDeg cost is")
            print(inDegCost) 

            print("Triadic cost is")
            print(triCost) 

            print("eProb cost is")
            print(eProbCost)  

            print("Topological cost is")
            print(obj2)  

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

        #select random reference network as base 
        numNets = len(netProperties.referenceNets)  
        indx = np.random.choice(numNets, 1)[0]     
        refNet = netProperties.referenceNets[indx]     
        adjRef = refNet.getAdjacencyMatrix().copy() 
        nNodesRef = (adjRef.shape)[0]      

        if nNodesRef < self.nNodes:
            #zerp pad adjacency matrix 
            pad_size = self.nNodes - nNodesRef
            if pad_size % 2 == 1:
                pad_size = (int(pad_size/2), int(pad_size/2) + 1)
            else: 
                pad_size = int(pad_size/2)       

            adjRef = np.pad(adjRef)   

        elif nNodesRef > self.nNodes:
            indices = np.random.choice(nNodesRef, self.nNodes, replace=False)  
            adjRef = adjRef[indices,:][:,indices]

        perm = np.random.permutation(self.nNodes)     
        adjRef = adjRef[perm,:][:,perm]   

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

        ind = creator.Individual(Network) #Individual inherits all properties of Network   
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

        """   
        #flip bits in adjacency matrix with predefined probability
        mutationMatrix = (rndVals < self.mutM).astype(int)     
        newAdjM = np.absolute(sub.getAdjacencyMatrix() - mutationMatrix) 
        """       

        """
        #flip ones in ajdacency matrix with probability p1 and zeros with probability p2 to approximately preserve number of edges  
        rndVals = np.random.rand(self.nNodes, self.nNodes)
        adjM = sub.getAdjacencyMatrix()  

        indOnes = np.where(adjM == 1)   
        indZeros = np.where(adjM == 0)   

        adjM[indOnes] = (rndVals[indOnes] > self.p1).astype(int)         
        adjM[indZeros] = (rndVals[indZeros] < self.p2).astype(int)    
        """          

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

        #print("Cross columns are")
        #print(crossColumns) 
        #print("Before")
        #print(sub1.in_nums)
        #print(sub2.in_nums) 
        tmp_nums = sub1.in_nums[crossColumns] 
        sub1.in_nums[crossColumns] = sub2.in_nums[crossColumns]  
        sub2.in_nums[crossColumns] = tmp_nums 
        #print("After")
        #print(sub1.in_nums)
        #print(sub2.in_nums)  

        if np.sum(sub1.adjM - adjSub1) != 0 or np.sum(sub2.adjM - adjSub2) != 0: 
            print("Difference in numpy matrices is")
            print(np.sum(sub1.adjM - adjSub1))  
            print(np.sum(sub2.adjM - adjSub2))    

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

def scatterPlotSubjects(paretoFront, labels = None, savePath = None):   
    fits = [sub.fitness.values for sub in paretoFront]    
    fits = list(zip(*fits))  
    fit1 = fits[0]  
    fit2 = fits[1]          

    plt.figure()      

    plt.scatter(fit1, fit2)
    plt.xlabel('Objective 1: (reg. weights)') 
    plt.ylabel('Objective 2: (topology)')     

    if labels is not None:
        # add labels to all points
        for indx, (xi, yi) in enumerate(zip(fit1, fit2)):
            plt.text(xi, yi, labels[indx], va='bottom', ha='center')  

    if savePath is not None: 
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
    def __init__(self, referenceNets, regWeights, nNodes, geneIndices, geneNames, maxRegs=10):
        self.referenceNets = referenceNets
        self.nNodes = nNodes 
        self.geneIndices = geneIndices    
        self.geneNames = geneNames   
        self.maxRegs = maxRegs 
        self.avgInDegs = self.getAvgInDegs()     
        self.avgOutDegs = self.getAvgOutDegs()   
        self.expEdgeProb = self.getExpectedEdgeProbability()   
        self.avgTriC = self.getAvgTriadicCensus()       
        
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
    #timeSeriesPaths   ... file paths list to time series data
    #steadyStatesPaths ... file paths list to steady states data
    #referenceNetPaths ... file paths list to reference networks 
    #maxRegs ... maximum number of regulators   
    def __init__(self, timeSeriesPaths, steadyStatesPaths=None, referenceNetPaths = None, goldNetPath = None, binarisedPath = None, savePath = None, maxRegs = 10, debug = False, obj2Weights = None):        
        #set empty list to None 
        if isinstance(steadyStatesPaths, list):  
            if not steadyStatesPaths:  
                steadyStatesPaths = None   

        self.timeSeriesPaths = timeSeriesPaths 
        timeSeriesDf = self.readFiles(timeSeriesPaths)    
        self.timeSeriesDf = timeSeriesDf 
        nNodes = len(self.timeSeriesDf.columns) - 1
        geneNamesList = list(self.timeSeriesDf.columns)  
        if "Time" in geneNamesList:
            geneNamesList.remove("Time")           
        self.geneIndices = {value: indx for indx, value in enumerate(geneNamesList)} 
        self.geneNames =  {indx: value for indx, value in enumerate(geneNamesList)} 
        self.goldNetPath = goldNetPath          
        self.savePath = savePath    
        self.obj2Weights = obj2Weights     

        if binarisedPath is None:
            tmp_path = timeSeriesPaths[0] 
            binarisedPath = tmp_path.rsplit(".", 1)[0] + "_binarised.tsv"      

        self.binarisedPath = binarisedPath      

        self.steadyStatesDf = None   
        self.steadyStatesPaths = steadyStatesPaths    
        if steadyStatesPaths is not None:    
            steadyStatesDf = self.readFiles(steadyStatesPaths)   
            self.steadyStatesDf = steadyStatesDf
            if nNodes != len(self.steadyStatesDf.columns):
                sys.exit("Error: Gene number mismatch between time series and steady states!")
        
        method_args = {} 
        
        #tmp_path = timeSeriesPaths[0] 
        #weights_path = tmp_path.rsplit(".", 1)[0] + "_weights.pkl"   

        regWeights = self.getRegulatoryWeights(**method_args)   

        """
        if not os.path.exists(weights_path):  
            regWeights = self.getRegulatoryWeights(**method_args)     
            with open(weights_path, "wb") as file:      
                pickle.dump(regWeights, file)      
        else:            
            with open(weights_path, "rb") as file:      
                regWeights = pickle.load(file)         
        """

        referenceNets = self.getReferenceNetworks(referenceNetPaths, maxRegs)                 
        netProperties = NetworkProperties(referenceNets, regWeights, nNodes, self.geneIndices, self.geneNames, maxRegs = maxRegs)      
        self.netProperties = netProperties 

        if debug:
            plt.bar(np.arange(13), netProperties.avgTriC)       
            plt.show()       

        use_xor = False  
        self.qm = QuineMcCluskey(use_xor = use_xor) 
        self.genSolver = GeneticSolver(netProperties, obj2Weights=obj2Weights)                          

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

    #simulates Boolean network based on provided mode  
    # mode = 0 ... exec and eval (dynamic evaluation of Boolean expressions) 
    # mode = 1 ... truth table lookup 
    # mode = 2 ... semi-tensor product      
    #TO DO ... different modes of simulation - exec and eval, lookup from truth table, semi-tensor product   
    def simulateBooleanModel(self, bNetwork, initialState, simNum, mode=2):     
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
                    exec(self.geneNames[target] + " = " + str(simulations[time_stamp - 1, target]))        

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
                        ind = getDecimalFromBinary(''.join(map(str, list(simulations[time_stamp-1, regulators]))))            
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

    #returns dynamic accuraccy based on training time series data 
    #bNetwork ... list of nodes
    #bNetwork[i] ... [target, regulators, generalised truth table, boolean function, boolean expression, truth table]  
    def getDynamicAccuracy(self, bNetwork, bin_df, experiments_df): 
        total_errors = 0
        timeSteps = 0 

        numNodes = bin_df.shape[1] 
        #simulate Boolean model for each experiment   
        for experiment in range(self.experiments):  
            sel = experiments_df["Experiment"] == experiment 
            bin_df_exp = bin_df[sel]   
            simNum = len(bin_df_exp.index)     
            initialState = bin_df_exp.iloc[0,:]    

            bin_df_exp = bin_df_exp.to_numpy()                     
            simulation = self.simulateBooleanModel(bNetwork, list(initialState), simNum)                                 

            errors = np.absolute(simulation - bin_df_exp).sum()   
            total_errors = total_errors + errors  
            timeSteps = timeSteps + simNum - 1 #exclude first time point    

        #return dynamic accuracy
        return 1 -  total_errors/(timeSteps*numNodes)        

    #returns Boolean model that best matches training data         
    #bNetworks ... list of Boolean networks   
    #bin_df, shift_bin_df, experiments_df    
    def getBestBooleanModel(self, bNetworks, bin_df, shift_bin_df, experiments_df):     
        bestAcc = self.getDynamicAccuracy(bNetworks[0], bin_df, experiments_df)     
        bestNetwork = bNetworks[0]   
        #iterate through every network except first  
        for bNetwork in bNetworks[1:]:  
            myAcc = self.getDynamicAccuracy(bNetwork, bin_df, experiments_df)   
            print(bestAcc)          
            if myAcc > bestAcc:
                bestAcc =  myAcc   
                bestNetwork = bNetwork       
        return bestNetwork, bestAcc                    
    
    def getGeneralisedTruthTable(self, target, regulators, bin_df, shift_bin_df, experiments_df):  
        reg_shift_df = shift_bin_df.iloc[:,regulators]    
        target_df = bin_df.iloc[:,target]     
        numReg = len(regulators)       
        numRows = pow(2, numReg)         
        rowValues = list(range(numRows))          

        gT = pd.DataFrame(rowValues, columns = ["inputVector"])
        #create zero columns for T and F   
        gT["T"] = 0          
        gT["F"] = 0              

        for experiment in range(self.experiments):
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

    def getExpression(self, bfun, target, regulators):
        expr = self.geneNames[target] + " = "

        if bfun is None:
            return expr + "0" 

        regulator_names = [self.geneNames[regulator] for regulator in regulators]        
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

    #returns truth table based on Boolean expression  
    def getTruthTable(self, bexpr, regulators):  
        numReg = len(regulators)        
        numRows = pow(2, numReg)          
        rowValues = list(range(numRows))    
        regNames = [self.geneNames[regulator] for regulator in regulators]

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

    def inferBooleanFunction(self, regulatorsArray, target, bin_df, shift_bin_df, experiments_df):
            gT = None        
            bfun = None  
            regulators = np.argwhere(regulatorsArray == 1).flatten()            

            if regulators.size != 0:  
                gT =  self.getGeneralisedTruthTable(target, regulators, bin_df, shift_bin_df, experiments_df)
                minterms = list(gT[gT["T"] > gT["F"]]["inputVector"])   
                dont_cares = list(gT[gT["T"] == gT["F"]]["inputVector"])     
                if len(minterms) != 0 or len(dont_cares) != 0:   
                    bfun = self.qm.simplify(minterms, dont_cares, num_bits=len(regulators))   
            """
            else:  
                print(f"No regulators found for {self.geneNames[target]}")             
            """

            bexpr = self.getExpression(bfun, target, regulators)  
            tT = self.getTruthTable(bexpr, regulators)  

            return regulators, gT, bfun, bexpr, tT  


    def inferBooleanNetwork(self, adjM, nNodes, bin_df, shift_bin_df, experiments_df):    
        b_functions = [] 
        for target in range(nNodes):  

            regulators, gT, bfun, bexpr, tT = self.inferBooleanFunction(adjM[:, target], target, bin_df, shift_bin_df, experiments_df)

            b_fun = (target, regulators, gT, bfun, bexpr, tT)            
            b_functions.append(b_fun)         

        return b_functions                            

    #infers Boolean networks   
    def inferBooleanNetworks(self, subjects, bin_df, shift_bin_df, experiments_df):  
        #list of Boolean networks        
        bNetworks = []   
        i = 0
        for subject in subjects:
            i = i + 1
            print(i) 
            #print("Inferring Boolean network") 
            adjM = subject.adjM
            nNodes = subject.nNodes 
            bNetwork = self.inferBooleanNetwork(adjM, nNodes, bin_df, shift_bin_df, experiments_df)    
            bNetworks.append(bNetwork)         
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
                regulators, gT, bfun, bexpr, tT = self.inferBooleanFunction(regulator_array, target, bin_df, shift_bin_df, experiments_df)  

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

        #get unique subjects on last population    
        subjects = getUniqueSubjects([sub for front in fronts for sub in front])      
        print("Number of unique networks in final population of evolutionary algorithm")    
        print(len(subjects))    

        binarisedPath = self.binarisedPath  
        if os.path.exists(binarisedPath): 
            #if binarised file exists read from file   
            binarised_df = pd.read_csv(binarisedPath, sep="\t") 
        else:
            #else binarise time series data and save       
            binarised_df = self.binarise(self.timeSeriesDf)   
            binarised_df.to_csv(binarisedPath, index=False, sep="\t")       

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
        best, bestAcc = self.getBestBooleanModel(bNetworks, bin_df, shift_bin_df, experiments_df)         
        bestCombined = self.getBestCombinedBooleanModel(subjects, bin_df, shift_bin_df, experiments_df)     
        myAcc = self.getDynamicAccuracy(bestCombined, bin_df, experiments_df)                  
        print(myAcc) 
        if myAcc > bestAcc: 
            best = bestCombined  

        return best            

    def readFiles(self, filePaths): 
        df_all = pd.DataFrame() 
        for filePath in filePaths:
            if os.path.exists(filePath):   
                df = pd.read_csv(filePath, sep="\t", decimal=",")  
                df = df.apply(pd.to_numeric) 
                df = df.dropna() 
                df_all = pd.concat([df_all, df], ignore_index=True)   
            else: 
                print(f"Unable to read file {filePath}. File does not exists!")  
        return df_all

    def getReferenceNetworks(self, referenceNetPaths, maxRegs):
        reference_nets = [] 
        for refPath in referenceNetPaths:
            reference_net = Network(refFile = refPath)
            reference_net.setMaxRegs(maxRegs)  
            if reference_net.adjM is not None:
                reference_nets.append(reference_net) 
        return reference_nets       

    #infer putative regulatory probability estimates 
    def getRegulatoryWeights(self, method="dynGENIE3", **method_args):  
        supportedMethods = ["dynGENIE3", "inferelator"]    

        #set experiment count
        self.timeSeriesDf["Experiment"] = 0  
        times = self.timeSeriesDf[["Time"]]  
        times_shifted = times.shift(periods=1)  
        
        self.timeSeriesDf["Experiment"] = (times["Time"] < times_shifted["Time"]).astype(int)    
        self.timeSeriesDf["Experiment"] = self.timeSeriesDf["Experiment"].cumsum() 
        self.experiments = self.timeSeriesDf["Experiment"].max() + 1
        
        if method not in supportedMethods:
            print(f"{method} is not supported! Using dynGENIE3.") 
        
        if method == "dynGENIE3":
            #define inference method specific arguments   
            #method_args["ntrees"] = 500    
            #method_args["K"] = "all"      
            return self.run_dynGENIE3(**method_args)   
        elif method == "inferelator": 
            return self.run_inferelator(**method_args)   

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
    
    def run_inferelator(self, **method_args): 
        return None
        """
        worker = inferelator_workflow(regression="bbsr", workflow="tfa")         
        worker.set_run_parameters(num_bootstraps=5, random_seed=42) 

        #self.timeSeriesDf 
        #time series paths  
        expression_matrix_file = self.timeSeriesPaths[0].rsplit(".", 1)[0] + "_inferelator_expression_matrix_file.tsv"
        metadata_file = self.timeSeriesPaths[0].rsplit(".", 1)[0] + "_inferelator_metadata.tsv" 
        tf_names_file = self.timeSeriesPaths[0].rsplit(".", 1)[0] + "_tf_names.tsv"     

        #if expression matrix file and metadata file does not exists generate file 
        if not os.path.exists(expression_matrix_file): 
            df = self.timeSeriesDf  
            columns = list(df.columns)    
            columns.remove("Time")  
            columns.remove("Experiment")
            df = df[columns] 
            time_series_df = self.timeSeriesDf[["Time", "Experiment"]] 
            time_series_shift_df = time_series_df.shift(periods=1) 
            time_series_shift_down_df = time_series_df.shift(periods=-1)   


            tf_names = list(self.geneIndices.keys()) 
            tf_names_df = pd.DataFrame({"tf_names": tf_names}) 

            print(time_series_df)  

            if self.steadyStatesPaths is not None:   
                df = pd.concat([df, self.steadyStatesDf])              

            rowNumAll = len(df.index) 
            rowNumTimeSeries = len(time_series_df.index)   

            df.insert(0, "cond", ['"' + str(el) + '"' for el in range(rowNumAll)])      

            df_metadata =  pd.DataFrame({"isTs": [False]*rowNumAll, "is1stLast":["NA"]*rowNumAll, "prevCol":["NA"]*rowNumAll, "del.t":[np.nan]*rowNumAll, "condName":['"' + str(el) + '"' for el in range(rowNumAll)]})   
            df_metadata.loc[0:rowNumTimeSeries-1, "isTs"] = True      
            
            print(df_metadata)

            print(time_series_df["Time"] - time_series_shift_df["Time"])   
            df_metadata.loc[0:rowNumTimeSeries-1, "del.t"] = time_series_df["Time"] - time_series_shift_df["Time"] 
            
            boolList = list(time_series_df["Experiment"] > time_series_shift_df["Experiment"]) 
            boolList[0] = True  
            boolList.extend([False]*(rowNumAll - rowNumTimeSeries)) 
            df_metadata.loc[boolList, "is1stLast"] = "f"         
            df_metadata.loc[df_metadata["del.t"] > 0, "is1stLast"] = "m"  
            boolList = list(time_series_df["Experiment"] < time_series_shift_down_df["Experiment"]) 
            boolList[-1] = True   
            boolList.extend([False]*(rowNumAll - rowNumTimeSeries)) 
            df_metadata.loc[boolList, "is1stLast"] = "l"              

            df_metadata.loc[df_metadata["del.t"] < 0, "del.t"] = "NA"   

            df_metadata.loc[0:rowNumTimeSeries-1, "prevCol"] = ['"' + str(el) + '"' for el in range(-1, rowNumTimeSeries - 1)]            
            df_metadata.loc[df_metadata["is1stLast"] == "f", "prevCol"] = "NA"     
            df_metadata.fillna("NA", inplace=True)   

            df.to_csv(expression_matrix_file, sep="\t", index=False)       
            tf_names_df.to_csv(tf_names_file, sep="\t", index=False, header=None,  quotechar='"', quoting=csv.QUOTE_ALL)   
            df_metadata.to_csv(metadata_file, sep="\t", index= False, quotechar='"')            

        worker.set_network_data_flags(use_no_gold_standard=True, use_no_prior=True)          
        worker.set_file_paths(input_dir=".", 
                      output_dir="./output_inferelator",
                      expression_matrix_file=expression_matrix_file, 
                      tf_names_file = tf_names_file,       
                      meta_data_file=metadata_file)       
        
        network_result = worker.run()      
        regulatory_probs = network_result.combined_confidences  
        regulatory_probs = regulatory_probs.to_numpy()  
        max_value = regulatory_probs.max()
        regulatory_probs = regulatory_probs + (1 - max_value)/2.0    
        print(regulatory_probs) 

        return regulatory_probs            
        """    






