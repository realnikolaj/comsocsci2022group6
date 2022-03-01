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
# Placeholder cell for visualizing entire network
# Don't run this, it is slow and cluttered, run next cell which uses
# a sampled subgrap



#stylized_network, config = nw.visualize(DG)
#fig, ax = nw.draw_netwulf(stylized_network)
#plt.show()
```

```python
# Visualize a subgraph of redditors directed network

k = 5000
sampled_nodes = random.sample(DG.nodes, k)
sampled_graph = DG.subgraph(sampled_nodes)
stylized_redditors_network, config = nw.visualize(sampled_graph)
```

```python
# Visualize a subgraph of the random null directed network

sampled_nodes = random.sample(null.nodes, k)
sampled_graph = null.subgraph(sampled_nodes)
stylized_null_network, config = nw.visualize(sampled_graph)

```

```python
# Sets up config loading from netwulf API and plots

stylized_network, config = nw.visualize(sampled_graph)

# Reproduce the visualisation
#fig, ax = nw.draw_netwulf(stylized_network)
#plt.show()
```

# Part 4.2 Clustering:
## Compare the clustering coefficient in the Redditors Network and its random counterpart.

>Compute the clustering coefficient for all nodes in the random network, using the formula 2.15 in your book.



(2.15) $C_{ı} = \frac{2L_{i}}{k_{i}\left(k_{i}-1\right)}$

```python
# Computes the list and prints a few non-zero values

c_coef = list(nx.clustering(null).values())
sampls = random.sample([c_coef[i] for i in range(1, N+1) if c_coef[i] > 0], 5)
print('Randomely selected non-zero clustering coefficient values')
print(" c_r =", *(f"[{i:.3f}]" for i in sampls))
```

>Compute the average clustering across nodes of the random network. Is it consistent with the analytical prediction (network science book equation 3.21)?


To calculate degree of clustering of the whole network we find the the average clustering coefficient \<k> given by eq. (2.16) below.  
(2.16) $\left< C \right> = \frac{1}{N} \sum_{i=1}^{N} C_{i}$

```python
# Average clustering coefficient for the null network

null_avg_c = np.mean(c_coef)
print(f'<C> of null network = {null_avg_c:.3g}') 
```

```python
# Simply verifying using the Networkx API's build-in function

#api_null_c = nx.average_clustering(null)
#print(f'<C> of null network using build-in method = {api_null_c:.3g}') 
```

>Compute the average clustering coefficient for the Redditors network. How does it compare to its random counterpart? Is it something you would expect? Why?

```python
# Average clustering coefficient for the redditors directed network

DG_avg_c = np.mean(list(nx.clustering(DG).values()))
print(f'<C> of redditors network = {DG_avg_c:.3g}') 
```

We observe a large difference in the degree of clustering between the redditors network and the random null network.  
Several reasons for this difference could be thought of:
 1. Users posting on reddit or submitting to posts all have various backgrounds, specifically considering different political orientation, belief systems w.r.t. stock market analysis and investments which could result in groups of threads with common submitters/posters i.e. various users might have certain tendencies to comment on certain types of posts which in turn could be posted by the same poster.
 2. Reddit is a social network and posters and submitters are not anonymous therefore users might be following other users and posting more frequently to a specific subset of users.
 3. Certain topics especially meme-stocks such as GME, AMC, BB etc. might attract certain groups of redditors and when we condition the entire network of redditors to only return GME specific comments submitted in a single subreddit out of millions of subreddits it is expected that niche groupings of threads containing common submitters are more likely which results in a larger spread in the degree distribution $p_{k}$ of the redditor network but maintains the same average degree $\left<k\right>$ e.g. when we sample some spefic redditors post history some redditors might engage more on particular discussions and some might simply post to submissions that doesn't gain a lot of traction i.e. to smaller threads.
 4. Follow up on degree disitribution.  In this analysis we did not consider the degree distribution of the redditors network. Using a uniform distribution to sample edges in the random network might not capture the actual distribution of closely- and not so closely related redditors and basically averages out on the typical redditor and its attributes.
**Conlusion: Using estimated values to generate random edges between nodes in our null network misses most if not all of the mentioned dynamics above. Dynamics that portray different caracteristics for various redditors which in turn results in closer relations between some redditors and some redditors are more isolated. The random network simply links redditors with no underlying mechanism other than the probability parameter p.**

```python
#Use previous style and config and start the API
#nw.visualize(G, config=config)
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
#fig, ax = plt.subplots(figsize=(16,16))
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



