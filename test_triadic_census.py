from triadic_census import count_triads

import networkx as nx 
import numpy as np  
import time 



G = nx.scale_free_graph(16) 
print(G)

start = time.time() 
triadic_census = nx.triadic_census(G)
end = time.time()   
elapsed = end - start 
print('{:.16f}'.format(elapsed))     

triNames = ["021D", "021U", "021C", "111D", "111U", "030T", "030C", "201", "120D", "120U", "120C", "210", "300"]  
triadic_census = list([triadic_census[key] for key in triNames])    
print(triadic_census)


start = time.time() 
triadic_census, _ = count_triads(G) 
end = time.time()   
elapsed = end - start 
print('{:.16f}'.format(elapsed))  
print(triadic_census)

print("--------------------------------------")
adjM = np.random.randint(2, size=(64, 64))  
G = nx.from_numpy_array(adjM, create_using=nx.DiGraph)  

triadic_census = nx.triadic_census(G)
triadic_census = list([triadic_census[key] for key in triNames])  
print(triadic_census) 
triadic_census, triad_pair_count = count_triads(G) 

x = triad_pair_count[0,1,:][3:] + triad_pair_count[1,0,:][3:]  
print(x) 

if G.has_edge(0,1): 
    G.remove_edge(0,1)  
else:
    G.add_edge(0,1)  
 
triadic_census, triad_pair_count = count_triads(G)  
print(triadic_census)
y = triad_pair_count[0,1,:][3:] + triad_pair_count[1,0,:][3:] 
print(y)     