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
import pandas as pd
import networkx as nx
import datetime
import matplotlib.pyplot as plt
import json
import random
import numpy as np
import netwulf as nw
```

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
## Part 4: Properties of the real-world network of Redditors

For this part of the assignment, consider the directed network of redditors posting about GME on r/wallstreetbets in the period included between Jan 1st and Dec 31st, 2020 (the one you built in Week 3, Part 3).

<!-- #endregion -->

```python
#Loading undirected graph from Week4 part 2

with open('../data/json_network.json', 'r') as infile:
    DG = nx.readwrite.json_graph.node_link_graph(json.load(infile))
infile.close()
N = DG.number_of_nodes()
L = DG.number_of_edges()
```

<div class="alert alert-block alert-danger">
1. Compute the value of p such that the number of expected edges of the random network equals the number of edges in the redditor network (see equation 3.2 in your Network Science Book).
</div>


<div class="alert alert-block alert-info">

<b>Note:</b> The equation (3.2) from Ch. 3 seem to be in conflict with the equation in Ch. 2 eq (2.5) which is specifically stated to apply to directed graphs. Since most of chapter 3 is revolving around undirected networks and since we had discrepencies getting the prober amount of links in our random network using these equation we will be using (2.5) instead and verifying with emperical values. </div>


>2.What is the value of p? Compute the average value of the degree < k > (using the formula).


First we calculate p and \<k> for the **directed network DG** using eq. (2.5) to create our random null network using p.  
Using (2.5),  L and N (No. of edges and No. of nodes) in the directed network DG the average degree of the directed network i.e. $\left< k \right>$ which by Ch. 2.3 is equal to the average degree **in** and average degree **out**.   
(2.5): $\left< k^{in} \right> = \left< k^{out} \right> = \frac{L}{N}$  
Since the amount of possible distinct pairs is doubled from the undirected to the directed network i.e. in the directed network en edge from A to B doesnt equal an edge from B to A as these two edges are now distinguishable, we should simply use $p = frac{k}{N}$ to calculate the probability of pairing two random nodes.

```python
# Using (3.2) to calculate p
# Using (3.3) to calcuae <k>

p = L / ( N*(N-1) / 2)
print("Using eq. (3.2)")
print("Values p and <k> for redditors directed network:")
print(f"p   = {p:.5f}")
k = p*(N-1)
print(f"<k> = {k:.5f}")
print()

# Using (2.5) to calculate <k>

k = L/N
p = k / N

print("Using eq. (2.5) and p = k/n")
print("Values p and <k> for redditors directed network:")
print(f"p   = {p:.5f}")
print(f"<k> = {k:.5f}")
print()
```

Later we will calcuate \<k> empirically for both networks.


> 2. [continued] Create a Random network with the same number of nodes as the redditor networks, and p as computed above. Generate a random network by linking nodes in every possible pair with probability p.


Creating a null model using N number of nodes to simply generate a network with the nodes {1,2,3 ... N}.  
Self loops are allowed and (A, B), (B, A) are two distinguishable pairs because the graphs are directed.

```python
# Building null DiGraph network with nodes in {1,2,3, ..., N}

null = nx.DiGraph()
null.add_nodes_from(list(range(1,N+1)))

# Generate edges list randomly

np.random.seed(2460)    
edgs = [(u,v) for u in list(range(1,N+1)) for v in list(*np.where(np.random.uniform(size=N) < p))] # Add "if v != u" if no self_loops are to be allow
null.add_edges_from(edgs)
```

```python
# Calculating emperical values for <k>

avg_degree_of_DG = np.mean([d for n, d in DG.in_degree()])
avg_degree_of_null = np.mean([d for n, d in null.in_degree()])
print(f"Avg degree 'k' of reditors directed network: {avg_degree_of_DG:.3f}")
print(f"Avg degree 'k' of null network: {avg_degree_of_null:.3f}")
```

> 3. Visualize the Redditors Network and the Random Network. Comment on the differences between the two.

```python
# Convert DiGraph DG to undirected Graph G reciprocal = True
G = DG.to_undirected(True, False)
Gnull = null.to_undirected(True, False)


```

```python

# Creates lists of self loop edges and removes these edges

G_loops = list(nx.selfloop_edges(G))
G.remove_edges_from(G_loops)
null_loops = list(nx.selfloop_edges(null))
G.remove_edges_from(null_loops)

# Creates lists of isolated nodes and removes these

G_isolated = [(n) for n in list(G.nodes) if G.degree[n] == 0]
G.remove_nodes_from(G_isolated)
null_isolated = [(n) for n in list(null.nodes) if null.degree[n] == 0]
null.remove_nodes_from(null_isolated)


```

```python
stylized_network, config = nw.visualize(DG)

fig, ax = nw.draw_netwulf(stylized_network)
plt.show()
```

```python

#network_properties = nw.tools.bind_properties_to_network(network,
                                         network_properties,
                                         bind_node_positions=True,
                                         bind_node_color=True,
                                         bind_node_radius=True,
                                         bind_node_stroke_color=True,
                                         bind_node_stroke_width=True,
                                         bind_link_width=True,
                                         bind_link_color=True,
                                         bind_link_alpha=True)

#nw.tools.draw_netwulf(network_properties,
                           fig=None,
                           ax=None,
                           figsize=None,
                           draw_links=True,
                           draw_nodes=True,
                           link_zorder=-1,
                           node_zorder=1000)
```

```python
temp = nx.Graph()
snodes = [n for n in list(random.sample(DG.nodes, 100))]
temp.add_nodes_from(snodes)
sedges = [(u,v) for u in snodes for v in list(*np.where(np.random.uniform(size=len(snodes)) < p))]
temp.add_edges_from(sedges)

```

```python
# Visualize subgraph of redditors directed network
k = 10000
sampled_nodes = random.sample(DG.nodes, k)
sampled_graph = DG.subgraph(sampled_nodes)
```

```python
stylized_network, config = nw.visualize(sampled_graph)

# Reproduce the visualisation
fig, ax = nw.draw_netwulf(stylized_network)
plt.show()
```

```python
#Use previous style and config and start the API
nw.visualize(G, config=config)
```

```python
#Can be used to create a null network with same node names
#null = nx.classes.function.create_empty_copy(DG).to_undirected()
```

```python
#k = 100
#sampled_nodes = random.sample(DG.nodes, k)
#sampled_graph = DG.subgraph(sampled_nodes)
```

```python
fig, ax = plt.subplots(figsize=(16,16))
#edge_collection = nx.draw_networkx_edges(DG, ax=ax)
#edge_labels = nx.draw_networkx_labels(DG, ax=ax)
#nx.draw(sampled_graph, with_labels=True)
```

# Part 4: Preliminary analysis of the GME redditors network


We begin with a preliminary analysis of the network.

> 
> *Exercise: Basic Analysis of the Redditors Network*
> * Why do you think I want you guys to use a _directed_ graph? Could have we used an undirected graph instead?
> * What is the total number of nodes in the network? What is the total number of links? What is the density of the network (the total number of links over the maximum number of links)?
> * What are the average, median, mode, minimum and maximum value of the in-degree (number of incoming edges per redditor)? And of the out-degree (number of outgoing edges per redditor)? How do you intepret the results?
> * List the top 5 Redditors by in-degree and out-degree. What is their average score over time? At which point in time did they join the discussion on GME? When did they leave it?
> * Plot the distribution of in-degrees and out-degrees, using a logarithmic binning (see last week's exercise 4). 
> * Plot a scatter plot of the the in- versus out- degree for all redditors. Comment on the relation between the two.
> * Plot a scatter plot of the the in- degree versus average score for all redditors. Comment on the relation between the two.


```python
2+2
```

```python

```
