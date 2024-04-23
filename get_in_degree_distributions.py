import matplotlib
import matplotlib.pyplot as plt 
import os
import pandas as pd 
import seaborn as sns   

def getInDegreeDistribution(network_structure_df, size):
    targets = list(network_structure_df[0])
    regulator_counts = [0]*(size) 
    for target in targets:
        regulator_counts[target-1] = regulator_counts[target-1] + 1  

    return regulator_counts     

def getInDegreePerSize(size):

    folder_path = os.path.join(".", "results", "EcoliExtractedNetworks", str(size), "SAILoR")  

    regulator_counts = []

    for filename in os.listdir(folder_path):
        if filename.endswith("_structure.tsv"):
            
            network_structure_df = pd.read_csv(os.path.join(folder_path, filename), sep=" <- ", header=None, index_col=False, engine='python') 
            regulator_counts_network = getInDegreeDistribution(network_structure_df, size)  
            
            regulator_counts.extend(regulator_counts_network)

    return regulator_counts  

network_sizes = [16, 32, 64]

colors = [0]*3
cmap = matplotlib.cm.get_cmap('tab10') 
all_colors = list(cmap.colors) 

colors[0] = cmap.colors[0]
colors[1] = cmap.colors[1]
colors[2] = cmap.colors[2]

df_degrees = pd.DataFrame()

fig, axes = plt.subplots(1, 3)  
plt.rcParams['patch.linewidth'] = 1
plt.rcParams['patch.edgecolor'] = 'white'

regulator_counts = getInDegreePerSize(16) 
ax1 = plt.subplot(131)
ax1.title.set_text('Network size 16') 
ax1.set_ylim([0, 0.5]) 
sns.histplot(regulator_counts, binrange=(0,10), discrete=True, stat = "probability", color = colors[0], alpha = 1,  ax=ax1)       
ax1.set(ylabel=None)

regulator_counts = getInDegreePerSize(32)
ax2 = plt.subplot(132)
ax2.title.set_text('Network size 32')
ax2.set_ylim([0, 0.5])
sns.histplot(regulator_counts, binrange=(0,10), discrete=True, stat = "probability", color = colors[1], alpha = 1, ax=ax2)     
ax2.set(ylabel=None)

regulator_counts = getInDegreePerSize(64)
ax3 = plt.subplot(133)
ax3.title.set_text('Network size 64')   
ax3.set_ylim([0, 0.5])
sns.histplot(regulator_counts, binrange=(0,10), discrete=True, stat = "probability", color = colors[2], alpha = 1, ax=ax3)      
ax3.set(ylabel=None) 

###################################################
#count Boolean functions with 7 or more regulators for logic functions from networks with 64 nodes#
regulator_counts_dist = [0]*20
regulator_counts = getInDegreePerSize(64) 
for count in regulator_counts:
    regulator_counts_dist[count] = regulator_counts_dist[count] + 1

total = sum(regulator_counts_dist)  

sevenOrHigher = sum(regulator_counts_dist[7:]) 
print(regulator_counts_dist)
print(sevenOrHigher)
print(total)  

###################################################

fig.supxlabel('Number of regulators') 
fig.supylabel('Probability')  
fig.tight_layout()    
plt.show()       


