from grakel import Graph
from grakel import GraphletSampling
import networkx as nx

#Extract motifs for all graphs given by list of adjacency matrices for given depth. 
def extract_motifs(adj_matrices, depth=3):
    graphs = list()
    for adj_matrix in adj_matrices:
        G = Graph(adj_matrix) 
        graphs.append(G)

    """
    if depth < 3:
        depth = 3
        print("Warning invalid graphlet size: {depth}, setting depth to 3!")     
    if depth > 5:
        depth = 5
        print("Warning invalid graphlet size: {depth}, setting depth to 5!")       
    """

    for k in range (3, depth + 1):
        #initialize kernel
        gk = GraphletSampling(k = k, sampling = {"a":-1})
        #check if fit is necessary
        gk.fit_transform(graphs)
        print("printing features")

        features = gk.parse_input(graphs)    
        print(features)

        print(gk._phi_X)

    for adj_matrix in adj_matrices:
        G = nx.from_numpy_matrix(adj_matrix, create_using=nx.DiGraph)
        census = nx.triadic_census(G)
        print([census[x] for x in census.keys()])  

    return None    
