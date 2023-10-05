import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import matplotlib

from matplotlib.ticker import FormatStrFormatter


histograms_path = os.path.join(".", "results", "histograms.pkl") 
file = open(histograms_path, "rb")      
avgTriC,triC = pickle.load(file)

print(avgTriC) 

file.close()

fig, axs = plt.subplots(1, 1)

triad_names = ["021D", "021U", "021C", "111D", "111U", "030T", "030C", "201", "120D", "120U", "120C", "210", "300"]  
ind = np.arange(len(triad_names))     

width = 0.45

plt.bar(ind, avgTriC, width, label = "Expected", color="#17BECF")
plt.bar(ind + width, triC, width, label = "Actual", color = "#E377C2")  
plt.legend(loc='upper right') 

plt.xticks(ind + width / 2, triad_names)
plt.xticks(rotation=45)

plt.yticks(np.arange(0, 1.2, step=0.2)) 

plt.xlabel("Triad") 
plt.ylabel("Frequency") 

axs.spines[['right', 'top']].set_visible(False) 


plt.show()    
