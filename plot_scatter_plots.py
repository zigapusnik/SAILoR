import matplotlib.pyplot as plt
import os
import pickle

from matplotlib.ticker import FormatStrFormatter

gens = [0,1,4,9]

scatter_path = os.path.join(".", "results") 


fig, axs = plt.subplots(2, 2)


for count, gen in enumerate(gens):
    path = os.path.join(scatter_path, "dump" + str(gen) + ".pkl")
    file = open(path, "rb")      
    fits = pickle.load(file)

    fits = list(zip(*fits)) 

    fit1 = fits[0]  
    fit2 = fits[1]           

    i = count // 2
    j = count % 2 

    axs[i,j].set_xlim([-0.1, 2.3])   
    axs[i,j].set_ylim([0.30, 0.45])  
    axs[i,j].set_title('Generation ' + str(gen + 1))  
    axs[i,j].spines['top'].set_visible(False)
    axs[i,j].spines['right'].set_visible(False)   

    axs[i,j].scatter(fit1, fit2, s = 75, alpha=0.85, linewidths=0)      

    axs[i,j].yaxis.set_major_formatter(FormatStrFormatter('%.2f')) 
    axs[i,j].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))  


fig.supxlabel(r"Alignment with ranked regulations ($z_1$)") 
fig.supylabel(r"Topological similarity ($z_2$)")       

plt.show()    
