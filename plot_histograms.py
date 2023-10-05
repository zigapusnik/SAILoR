import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import SAILoR


from matplotlib.ticker import FormatStrFormatter
from SAILoR import NetworkProperties

def plotTriadicCensus(data):
    return None 

def plotHistogram(data, xlabel):
    fig, axs = plt.subplots(1, 1)

    ind = np.arange(len(data))     


    plt.bar(ind, data, width=.9, alpha = 1)
    plt.tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False)
    plt.show()  

netProperties_path = os.path.join(".", "results", "netProperties.pkl")  
file = open(netProperties_path, "rb")        
netProperties = pickle.load(file)
file.close()

avgInDegs = netProperties.avgInDegs     
avgOutDegs = netProperties.avgOutDegs   
expEdgeProb = netProperties.expEdgeProb 
avgTriC = netProperties.avgTriC   


plotHistogram(avgInDegs, r"$k$")
plotHistogram(avgOutDegs, r"$k$")
#plotHistogram(expEdgeProb)



