from matplotlib import pyplot as plt
import numpy as np 
import matplotlib


params = {'legend.fontsize': 16,
          'axes.labelsize': 16,
          'axes.titlesize': 20,
          'mathtext.fontset' : 'stix'
         }
matplotlib.rcParams.update(params)


x1 = np.linspace(0, 0.1, 10000)
x2 = np.linspace(0.1, 1, 90000)

expected = 0.1

plt.plot(x1, np.abs(x1 - expected)/expected, color = "black", linewidth=1.7, alpha=0.8)
plt.plot(x2, np.abs(x2 - expected)/(1 - expected), color = "black", linewidth=1.7, alpha=0.8) 

plt.ylim(0,1.1)
plt.xlim(0,1.1)  

plt.xlabel(r"$\hat x$", fontsize=13)
plt.ylabel(r"$L_4(\hat x, x)$", fontsize=13)

ax = plt.gca()

xt = ax.get_xticks() 
xt[-1]=expected 

xtl=xt.tolist() 
xtl[-1] = r"$x$" 

xtl[3] = r"0.6" 

print(xtl) 

ax.set_xticks(xt)
ax.set_xticklabels(xtl)    
ax.spines[['right', 'top']].set_visible(False) 

plt.show()     



