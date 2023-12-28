import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import grid2op
from grid2op.PlotGrid import PlotMatplot
from grid2op.Runner import Runner
import networkx as nx
import matplotlib.pyplot as plt 

env_name = 'rte_case14_realistic'

env = grid2op.make(env_name, test=True) # default = "rte_case14_realistic" environment 

obs = env.reset() 
plot_helper = PlotMatplot(env.observation_space)
# _ = plot_helper.plot_obs(obs)
fig_info = plot_helper.plot_info(line_values=obs.name_line)
fig_info.show()
# _ = plot_helper.plot_obs(obs,line_info='rho')
# obs.name_line



graph = obs.as_networkx()

linegraph=nx.line_graph(graph)
Adj = nx.adjacency_matrix(linegraph)
Adj = Adj.todense()

f1 = plt.figure()
nx.draw_networkx(graph,
                        with_labels=True,
                        # I want to plot the "rho" value for edges
                        edge_color=[graph.edges[el]["rho"] for el in graph.edges], 
                        # i use the position computed with grid2op
                        # NB: this code only works if the number of bus per substation is 1 !
                        pos=[plot_helper._grid_layout[sub_nm] for sub_nm in obs.name_sub]
                      )
f1.savefig("graph_{}.png".format(env_name),dpi=600)


for edgename in graph.edges:
    rho=graph.edges[edgename]['rho']
    linegraph.nodes[edgename].update({'rho': rho})

f2 = plt.figure()
nx.draw_networkx(linegraph,
                        with_labels=True,
                        node_color=[linegraph.nodes[el]['rho'] for el in linegraph.nodes]
                      )
f2.savefig("line_graph_{}.png".format(env_name),dpi=600)

# for edge in graph.edges(data=True):
#     print(edge)

for node in linegraph.nodes(data=True):
    print(node)

