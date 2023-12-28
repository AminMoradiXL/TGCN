import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import grid2op
from grid2op.PlotGrid import PlotMatplot
from grid2op.Runner import Runner
import networkx as nx
import matplotlib.pyplot as plt 

env_name = 'rte_case14_realistic'

env = grid2op.make(env_name, test=True) # default = "rte_case14_realistic" environment 

obs = env.reset() 

graph = obs.as_networkx()

linegraph=nx.line_graph(graph)
oldAdj = nx.adjacency_matrix(linegraph)
oldAdj = oldAdj.todense()

dict_={'0':18, '1':3, '2':16, '3':2, '4':15, '5':7, 
       '6':8, '7':1, '8':6, '9':9, '10':4, '11':19, 
       '12':13, '13':5, '14':14, '15':11, '16':12, '17':17, 
       '18':0, '19':10}

Adj=np.zeros((20,20))
for i in range(20):
    for j in range(20):
        if oldAdj[i,j]==1:
            m = dict_[str(i)]
            n = dict_[str(j)]
            Adj[m,n] = 1

df = pd.DataFrame(Adj)
df.to_csv('Adj.csv')
     
