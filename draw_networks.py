import matplotlib.pyplot as plt
import networkx as nx
import numpy as np  
import os 
import re

def draw_network(path,  layout):

    adjM = np.zeros((size, size))  

    lines = list()
    with open(path) as f:
        lines = f.readlines()

    for line in lines: 
        print(line)  
        result = re.search("([0-9]+) <- ([0-9]+)", line)
        target = int(result.group(1))
        regulator = int(result.group(2)) 
        print(target)
        print(regulator)

        adjM[regulator-1, target-1] = 1

    G = nx.from_numpy_array(adjM, create_using=nx.DiGraph) 
    pos = nx.nx_agraph.graphviz_layout(G, prog=layout) #dot za SAILoR
    nx.draw(G, pos=pos, node_size=200, node_color="#1a1a1a", edge_color="#4d4d4d", with_labels = False, width=1.25, arrowsize=10)  


network = "Ecoli-7_5_structure.tsv" 
size = 32


fig, ax = plt.subplots(figsize=(13,5))

method = "SAILoR"
network_path = os.path.join(".", "results", "EcoliExtractedNetworks", str(size), method, network) 
ax1 = plt.subplot(131)
draw_network(network_path, "dot")

method = "MIBNI" 
network_path = os.path.join(".", "results", "EcoliExtractedNetworks", str(size), method, network) 
ax2 = plt.subplot(132)
draw_network(network_path, "circo")

method = "GABNI" 
network_path = os.path.join(".", "results", "EcoliExtractedNetworks", str(size), method, network) 
ax3 = plt.subplot(133)
draw_network(network_path, "circo")

ax1.text(0.5,-0.1, "a)", size=14, ha="center", transform=ax1.transAxes)
ax2.text(0.5,-0.1, "b)", size=14, ha="center", transform=ax2.transAxes)
ax3.text(0.5,-0.1, "c)", size=14, ha="center", transform=ax3.transAxes)

fig.tight_layout() 
plt.show() 

