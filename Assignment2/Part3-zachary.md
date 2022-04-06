---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.7
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python

```

```python
import pandas as pd
import networkx as nx
from community import community_louvain
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import json
import random
import numpy as np
import netwulf as nw
import random
import copy
import scipy.stats as stats
import statsmodels.api as sm
```

## Part 3 Communities for the Zachary Karate Club Network
1. > Visualize the Zachary Karate Club network as a graph
 > - Set the color of each node based on the club split.


#### 1. Visualize Zachary Karate Club Network

```python
'''
Init the graph and change attribute name to 'group' for easier implementation i netwulf i.e. ..
netwulf recognizes the node attributes named 'group', 'size' and the link attributes named weight ..
and will automatically colorize different groups
'''
G = nx.karate_club_graph()
N = G.number_of_nodes()

clubs = nx.attr_matrix(G, node_attr='club')[1]  # Gets the set of node attributes i.e 'Mr. Hi' and 'Officer'
```

```python
#stylized_club_network, config = nw.visualize(G) 
#nw.save("../config/assignment2/p3.json", stylized_club_network, config) # Save the config as p3clubnet.json
```

<img src="../plots/assignment2/Part3_club_network.png" width="800" height="400">



2. > Write a function to compute the modularity of a graph partitioning (use equation 9.12 in the book)
 > - The function should take a networkX Graph and a partitioning as inputs and return the modularity.   
3. > Explain in your own words the concept of modularity.


Modularity explains a component of a real network which can provide clues to whether a possible community pattern is merely observed due to chance. The method is based on comparing a random null network to the actual graph and measuring if there's a difference in the real networks wirring diagram and that of the null network. If more links are observed in a partition of the graph than expected by chance i.e. the same resulting parameter from the randomized network, then there's a potential for a community.

The expected number of links in a completely randomized subgraph *C* is calculated using the following equation:

$M_{c}=\frac{L_{c}}{L}-\left(\frac{k_{c}}{2 L}\right)^{2}$

The generalized version below measure the same statistic for a full graph by summing over each observed subgraph or partition *C* i.e. the observed potential communities, where $C = \{1, 2, \ldots, n_{c}\}$ and *$n_{c}$* is the total number of partitions. Thus $L_{c}$ and $k_{c}$ is the number of links or edges within a community *c* and the total degree of the nodes in the community respectively.


\begin{align}
    M &=\sum_{c=1}^{n_{c}}\left[\frac{L_{c}}{L}-\left(\frac{k_{c}}{2 L}\right)^{2}\right] \label{eq9.12}\tag{9.12}\\
\end{align}


#### 2. Writes the function to compute modularity:

```python
def modularity(network):
    '''
    This function implements the modularity parameter.
    requires attribute which can be used to distinguish/partition nodes.
    This function also returns the communities which is used to evaluate the result againts..
    the built-in networkx modularity function
    '''
    partitions = list()
    no_of_groups = len(nx.attr_matrix(network, node_attr='club')[1])
    for _community in range(no_of_groups):
        partitions.append({n for n in range(G.number_of_nodes()) if G.nodes[n]['club'] == _community})
    #partitions = [{n for n in range(G.number_of_nodes()) if G.nodes[n]['club'] == 'Mr. Hi'}, 
                  #{n for n in range(G.number_of_nodes()) if G.nodes[n]['club'] == 'Officer'}]
    
    k = dict(G.degree)
    sumdeg = sum(k.values())
    L = sumdeg / 2
    norm = 1 / sumdeg ** 2
    L_c = np.zeros(no_of_groups)
    sumk = np.zeros(no_of_groups)
    for c in range(len(partitions)):
        parts = set(partitions[c])
        L_c[c] = sum([1 for u, v in G.edges(parts) if v in parts])
        sumk[c] = sum([k[u] for u in parts])
    return sum(L_c / L - (sumk ** 2) * norm), partitions
```

***
4. > Compute the modularity of the Karate club split partitioning using the function you just wrote. Note: the Karate club split partitioning is avilable as a node attribute, called "club".

```python
'''
This code changes the club strings into integers.
It is needed to generalize the modularity function above to accomodate the more common..
attribute type of integers, which is used later.
'''
for _node in G.nodes: # Change string attribute to int
    G.nodes[_node]['club'] = (0 if G.nodes[_node]['club'] == 'Mr. Hi' else 1)   
mod_of_G = modularity(G)[0]
print(f'-------------| Results |------------- \n',
      f'Modularity of Zachary club network: \n',
      f'{mod_of_G:.2f} \n',
      f'[Built-in networkx function] Modularity of Zachary club network: \n',
      f'{nx.algorithms.community.quality.modularity(G, modularity(G)[1], weight="club"):.2f} \n')

```

***
5. > **Randomization experiment**  
>We will now perform a small randomization experiment to assess if the modularity you just computed is statitically different from 0. To do so, we will implement a configuration model. In short, we will create a new network, such that each node has the same degree as in the original network, but different connections. Here is how the algorithm works.  

**Algorithm**: ***Double edge swap***  
a. Create an identical copy of your original network.  
b. Consider two edges in your new network (u,v) and (x,y), such that u!=v and v!=x.  
c. If none of edges (u,y) and (x,v) exists already, add them to the network and remove edges (u,v) and (x,y).  
Repeat steps b. and c. to achieve at least N swaps (I suggest N to be larger than the number of edges).  

```python
'''
We set a seed in this when creating the configuration to help with reprocudability.
The function double_edge_swap from networkx implements the Algorithm above.
'''
seed = 42
L = G.number_of_edges()
model = nx.algorithms.double_edge_swap(copy.deepcopy(G), nswap=L+10, max_tries=1000, seed=seed)
```

***
6. >Double check that your algorithm works well, by showing that the degree of nodes in the original network and the new 'randomized' version of the network are the same.


##### Comparing the two models' node degree 

```python
print(f'-------------| Results |------------- \n',
      f'Degree lists are identical (True/False):', list(model.degree) == list(G.degree) )
```

***
7. > Create 1000 randomized version of the Karate Club network using the double edge swap algorithm you wrote in step 5. For each of them, compute the modularity of the "club" split and store it in a list.

```python
'''
This code holds the 1000 randomized configuration models in a nested list.
The 'seed=random.seed()' parameter and argument is there to ensure each new model is created independently 
from different seeds.
The list variable 'modularity' holds the modularities computed from each model.
'''
modeldict = list(range(1000))
modularities = list(range(1000))

for _model in modeldict:
        
        modeldict[_model] = nx.algorithms.double_edge_swap(G, nswap=L+10, max_tries=1200, seed=np.random.randint(1000))
        modularities[_model] = modularity(modeldict[_model])[0]
```

***
8. >Compute the average and standard deviation of the modularity for the random network.


##### Using numpy

```python
'''
Numpy function numpy.mean and numpy.std is used to compute the parameters on our list of modularities
'''
print(f'-------------| Results |------------- \n',
      f'Average modularity of 1000 random networks: \n {np.mean(modularities):.2f} \n',
      '\n', 
      f'Standard deviation of 1000 random networks: \n {np.std(modularities):.2f}')
```

***
9. >Plot the distribution of the "random" modularity. Plot the actual modularity of the club split as a vertical line (use axvline).

```python
# Setup matplotlib formatting
def setup_mpl():
    mpl.rcParams['font.family'] = 'Helvetica Neue'
    mpl.rcParams['font.size'] = 18
    mpl.rcParams['lines.linewidth'] = 1
    mpl.rcParams['axes.linewidth'] = 1.2
setup_mpl()
myFormat = mpl.dates.DateFormatter('%b %Y')
```

```python
fig, ax = plt.subplots(figsize=(12,8))
h1 = ax.hist(modularities, label="Random networks")[2]
l1  = ax.axvline(x=mod_of_G, color='r', linewidth=4, label="Actual")
ax.legend((h1, l1), ('Random networks', "Zacharys's Club"), loc='upper right', shadow=True,)
ax.set_xlabel("Modularity")
ax.set_ylabel("Networks")
```

```python
'''
A qqplot showing difference in residuals is helpful when the variance is very small as seen in the histogram above
'''
pplot = sm.ProbPlot(np.array(modularities).cumsum(), stats.t, fit=True)
#fig = pplot.qqplot()
fig = pplot.qqplot(line="45")
h = plt.title("QQ-plot - of squared residuals from mean [random networks]")
plt.show()
```

10. >Comment on the figure. Is the club split a good partitioning? Why do you think I asked you to perform a randomization experiment? What is the reason why we preserved the nodes degree?


The difference between actual modularity and the observed from our 1000 instances of a configuration model seems to be very high, which indicates that the original club split is based on actual communities interacting.  
This experiment is very important to support our hypothesis that there is in fact several communities present and the wirring of the Zachary network is not just present because of chance. Without these experiments the observed communities could not be supported, especially given the small amount of nodes, i.e. a slight modification to the graph as a whole would drastically change the modularity. By preserving the nodes degree we back our claim that the community structuring is in fact encoded in the wiring diagram i.e. the total amount of edges and the node degrees does support the existence of an actual community.


***
11. >Use the Python Louvain-algorithm implementation to find communities in this graph. Report the value of modularity found by the algorithm. Is it higher or lower than what you found above for the club split? What does this comparison reveal?

```python
'''
This code uses the python_louvain package implementing the algorithm.
It returns a partitioning table/dict formatted as {node: community}, which is different
than the required format by the modularity functions used earlier. 
The returned table is then converted in to a nested list containing individual lists of all
member nodes of a specific community, which is suitable for the modularity functions.
For our own custom function we first assign node attributes with values corresponding to the
assigned community provided by the Python Louvain algorithm
'''

louvain_partition = community_louvain.best_partition(G) # Creates dictionary {node: community}

louvain_set_of_communities = set(louvain_partition.values()) # Creates an iterable list over the found communities
model1 = copy.deepcopy(G)
for _node in model1.nodes(): # Assigns commnunity attributes to our configuration model based on P. Louvain partitioning
    model1.nodes[_node]['club'] = louvain_partition[_node]

print(f'Modularity of Louvain partition: \n',
      f'{modularity(model1)[0]:.2f} \n')

louvain_communities = list()
for _community in louvain_set_of_communities:  #add node to corresponding community and append the communities to a list of communities
    louvain_communities.append({key for key in louvain_partition.keys() if louvain_partition.get(key) == _community})

print(f' Modularity of Zachary club network [Built-in networkx function]: \n',
      f'{nx.algorithms.community.modularity(model1, louvain_communities):.2f}')
```

***
12. >Compare the communities found by the Louvain algorithm with the club split partitioning by creating a matrix D with dimension (2 times A), where A is the number of communities found by Louvain. We set entry D(i,j) to be the number of nodes that community i has in common with group split j. The matrix D is what we call a confusion matrix. Use the confusion matrix to explain how well the communities you've detected correspond to the club split partitioning.


<div class="alert alert-block alert-danger">
There's unfortunately no more
</div>
