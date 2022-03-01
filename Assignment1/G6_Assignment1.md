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

# Assignment 1
## Group 6
> s183930 - Nikolaj S. Povlsen
> 
> 


<!-- #region -->
## Link to Git repository
https://github.com/realnikolaj/comsocsci2022group6



## Contribution statement

!!!!!!!!!!!!!!!Husk contribution statement her!!!!!!!!!!!!!!



<!-- #endregion -->

# Part 1: Datasets for Computational Social Science


### Big. How large is this data (approximately)? Could you collect the same amount of information via surveys?


We have 14715 submissions by 9375 unique authors and 644,397 comments on those submissions. It would probabaly be costly to obtain the same amount of information by surveys.


### Always-on. Can you keep collecting data over time?


The data is always-on, at least submission and comment are saved in https://files.pushshift.io, while scores will not be correctly calculated but was not here.


### Non-reactive. Is the dataset non-reactive?


The data is non-reactive, as people are probably not aware that data is used for experiments like these.


### Incomplete. Do you think the dataset captures entirely the unfolding of events leading to the GME stock rise in price?


The dataset does not capture all events leading to the GME stock rise. There are probably other online platforms like telegram and also the search criterion for reddit information may not capture all relevant data. Other than that market commentators, stock analysts and word of mouth are also sources of information exchange regarding the matter.


### Inaccessible. Is the data accessible? 


Reddit data is publicly accessible from PushShift API.


### Non-representative. Do you think that the conclusions we will draw by analyzing this dataset are specific to the GME events? Or could they instead help us understand social phenomena more in general? If yes, which phenomena could you think of? Of not, what are the aspects that make this dataset non-representative?


The conclusions will not be specific to the GME events, they could also tell us about the phenomena of manipulating stock prices in groups.


### Drifting. Is there any source of drift in this dataset (within the period observed)? 


Public perception of the phenomena, e.g. conveyed thorugh public news, will change the influx of users in this subreddit and investors.


### Algorithmically confounded. Is the dataset algorithmically confounded? If yes, why?


The score system will influence the type of posts, as thought provoking posts will probably get a score and comments. Some users will try to post/comment from a principle to improve ones score.


### Dirty. What aspect may make this dataset dirty?


As everyone can post/comment, there are unwanted data such as deleted comments and users, plus junk and spam.


### Sensitive. Is there any sensitive information in the data?


Usernames would not be considered sensitive, but could potentially be combined with other information, such as timestamps and crosreferencing other sites to identify the person.


# Part 2: activity on Reddit and GME prices


## Part 2.1 : Plotting prices and comments using line-graphs.

```python
import matplotlib.pylab as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import json
import netwulf as nw
from scipy import stats
import networkx as nx
```

```python
# Setting up data
GME_marketdata = pd.read_csv('../data/GME.csv', parse_dates=['Date']).set_index('Date')
rolled_marketdata = GME_marketdata['Volume'].rolling('7D', center=True).mean()

GME_comments = pd.read_csv('../data/wallstreetbet_gamestop_comments_cleaned.csv')
GME_comments['date_of_creation'] = pd.to_datetime(GME_comments['date_of_creation'], unit='s')
GME_comments = GME_comments.sort_values('date_of_creation').set_index('date_of_creation')
GME_comments = GME_comments.loc['2020-1-1':'2021-1-25']
GME_comments['daily'] = GME_comments.rolling('1D', center=True).count()['id']
GME_comments['weekly'] = GME_comments['daily'].rolling('7D', center=True).mean()
```

```python
# Setup matplotlib formatting
def setup_mpl():
    mpl.rcParams['font.family'] = 'Helvetica Neue'
    mpl.rcParams['lines.linewidth'] = 1
setup_mpl()
myFormat = mpl.dates.DateFormatter('%b %Y')
```

### 1. Plot the daily volume of the GME stock over time using the GME market data. On top of the daily data, plot the rolling average, using a 7 days window (you can use the function pd.rolling). Use a log-scale on the y-axis.

```python
fig, ax = plt.subplots(1, 1, figsize=(10,5), dpi=400)
ax.plot(GME_marketdata.index,GME_marketdata.Volume, ls = '--', alpha = 0.5, label="Daily volume")
ax.plot(rolled_marketdata.index,rolled_marketdata.values, color='black', label="1 week rolling average")
ax.set_ylabel('Volume (USD)')
ax.set_yscale('log')
ax.legend()
ax.xaxis.set_major_formatter(myFormat)
plt.title("Figure 1: Daily GME stock volume ", y=-0.15, fontsize=8)
plt.show()
```

### 2. Now make a second plot where you plot the total number of comments on Reddit per day. Follow the same steps you followed in step 1.

```python
fig, ax = plt.subplots(1, 1, figsize=(10,5), dpi=400)
ax.plot(GME_comments['daily'].index,GME_comments['daily'].values, ls = '--', alpha = 0.5, label="Daily comments")
ax.plot(GME_comments['weekly'].index,GME_comments['weekly'].values, color='black', label="1 week rolling average")
ax.set_ylabel('Comments')
ax.set_yscale('log')
ax.legend()
ax.xaxis.set_major_formatter(myFormat)
plt.title("Figure 2: Daily number of comments on r/wallstreetbet subreddit", y=-0.15, fontsize=8)
plt.show()
```

### 3. What is the advantage of using the log-scale on the y-axis? What is the advantage of using a rolling-window?


Using a log scale on the y-axis allows us to see long term trends, ranging from very few comments in the start and the multiple orders of comments in the end of the time period. If we had used cartesian coordinates the trend of the small numbers would not be visible.
The advantage of using a rolling window, here centered on the period, smoothes the data in order to capture the overall trend of the data.


### 4. Now take a minute to look at these two figures. Then write in a couple of lines: What are the three most important observations you can draw by looking at the figures?

```python
fig, ax = plt.subplots(figsize=(10,5), dpi=400)
ax.plot(rolled_marketdata.index,rolled_marketdata.values, color='black', label="1 week rolling average volume")
ax.set_ylabel('Volume (USD)')
ax.set_yscale('log')
ax.xaxis.set_major_formatter(myFormat)
ax2 = ax.twinx()
ax2.plot(GME_comments['weekly'].index,GME_comments['weekly'].values, color='red', label="1 week rolling average comments")
ax2.set_ylabel('No. of comments')
ax2.set_yscale('log')
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
fig.tight_layout()
plt.title("Figure 3: Rolling weekly GME stock volume and number of comments on r/wallstreetbet subreddit", y=-0.15, fontsize=10)
plt.show()
```

Figure 3 shows the rolling weekly GME stock volume and number of comments in a dual axis chart, showing a correlation between volume and number of comments, specifically we see that when the volume goes up, we see an increase in number of comments. 
After september 2020 we see a sharp rise in the volume of trades in GME stock, also the boom and bust cycles in trade volume from september 2020 indicate a degree of speculative trading.


## Part 2.2: Returns vs number of comments using scatter-plots. In this part of the assignment, we will look at the association between GME market indicators and the volume of comments on Reddit


### 1. & 2. Compute the daily log-returns and the daily log-change in number of new comments

```python
# Resample to daily and compute log-returns and log-change in comments
log_return_comments = pd.DataFrame(np.log(GME_marketdata['Close'] / GME_marketdata['Close'].shift(1)))
daily_comments = GME_comments.resample('1D').count()['id']
daily_comments = daily_comments[daily_comments!=0]
daily_log_change = np.log(daily_comments / daily_comments.shift(1))
log_return_comments['comments'] = daily_log_change
log_return_comments.dropna(inplace=True)
```

### 3. Compute the correlation coefficient between the series computed in step 1 and step 2. Is the correlation statistically significant?

```python
print(f" Pearson correlation coeffcient for the daily log change in comments and return:{stats.pearsonr(log_return_comments['Close'],log_return_comments['comments'])[0]:.2f}, p-value:{stats.pearsonr(log_return_comments['Close'],log_return_comments['comments'])[1]:.9f}")
```

The Pearson correlation coefficient shows a weak positive linear relationship between daily log change and return, but the p-value = 0.00 suggests that it is statistically significant.


### 4. Make a scatterplot of the daily log-return on investment for the GME stock against the daily log-change in number of comments.

```python
# Prepare data for scatterplot
log_return_comments['close_price'] = GME_marketdata['Close']
log_return_comments['color'] = 'blue'
log_return_comments['color'].loc['2021'] = 'green'

fig, ax = plt.subplots(dpi=400)
ax.scatter(log_return_comments['Close'],log_return_comments['comments'], s=log_return_comments['close_price'], c=log_return_comments['color'], alpha=0.7)
ax.set_ylabel('Daily log-change in number of comments')
ax.set_xlabel('Daily log-return on investment for the GME stock')
ax.set_xlim(-0.5,0.5)
ax.set_ylim(-5,5)

for size in [2, 10, 50]:
    plt.scatter([], [], c='k', alpha=0.3, s=size, label=str(size) +' USD')

legend1 = plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='Close price')
legend_elements = [mpl.lines.Line2D([0], [0], marker='o', color='w', label='2020', markerfacecolor='b', markersize=10),
                   mpl.lines.Line2D([0], [0], marker='o', color='w', label='2021', markerfacecolor='g', markersize=10)
                   ]
ax.legend(handles=legend_elements)
ax.add_artist(legend1)
plt.title("Figure 4", y=-0.22, fontsize=6)
plt.show()
```

### 5. Now take a minute to look at the figure you just prepared. Then write in a couple of lines: What are the three most salient observations you can draw by looking at it? 


When looking at Figure 4 we see an overall correlating trend between the daily log-change in comments and log-return on  investment, specifically focusing on greater positive changes in closing price semms to positively affect change in number of comments, and vice versa. 
This trend is even more pronounced for the 2021 data where we observe a linear relationship which suggests a power law relationship, log-change in comments change by an order of magnitude compared to the daily log-return, equal to the slope of the fitted line. 
We see a greater positive log-return in 2021, than 2020.


### 6. Based on the exploratory data visualization in Exercises 2 and 3, what can you conclude on the research question: *Is the activity on wallstreetbet related to the price of the GME stock?*


There is evidence of correlation between the activity on wallstreetbet and the price of the GME stock. Correlation does equal causation and a hidden confounder could be the reason for the correlation found, although it was on on the subreddit that the meme stock phenomena of 2020 began.
The activity on reddit is certainly to a degreee a result of market changes for the GME stock, but the reverse could also easily be the case.


# Part3


## Exercise 2.1 the 'Königsberg Problem

### Which of the icons in Image 2.19 can be drawn without raising yourpencil from the paper, and without drawing any line more than once? Why?

A walking path that goes through all bridges can have only one starting and one end point. Thus such a path cannot exist on a graph that has more than two nodes with an odd number of links.

Graph a) Consists of two nodes of degree 2, and two of degree three. Therefore, it does not have more than two nodes with an odd number of links, and it can be drawn be drawn without raising your pencil from the paper, and without drawing any line more than once.

Graph b) Consists of one node of degree 4, and four of degree three. Therefore, it does have more than two nodes with an odd number of links, and it can not be drawn be drawn without raising your pencil from the paper, and without drawing any line more than once.

Graph c) Consists of 6 node of degree 2, and 6 of degree 4. Therefore, it does have more than two nodes with an odd number of links, and it can be drawn be drawn without raising your pencil from the paper, and without drawing any line more than once.

Graph d) Consists of 1 node of degree 1, 2 nodes of degree 2, one of degree 3 and 2 of degree 6. Therefore, it does have more than two nodes with an odd number of links, and it can be drawn be drawn without raising your pencil from the paper, and without drawing any line more than once.

<!-- #region tags=[] -->
## Exercise 2.3 Graph representation
<!-- #endregion -->

Construct for the networks (a) and (b) in Imade 2.20:
- Undirected graph of 6 nodes and 7 links.
- Directed graph of 6 nodes and 8 directed links.
- The corresponding adjacency matrices.
- The corresponding link lists.

```python
# Undirected graph

undirected_graph = nx.Graph()
undirected_graph.add_nodes_from([1, 2, 3, 4, 5, 6])
undirected_graph.add_edges_from(
    [(1, 2), (1, 3), (1, 6), (6, 3), (3, 2), (2, 4), (1, 4)])

adjacency_matrix_undirected = nx.adjacency_matrix(undirected_graph).todense()

print("Adjacency matrix undirected:\n",adjacency_matrix_undirected)

link_list_undirected = list(undirected_graph.edges)
print("\nLinked_list undirected:\n", link_list_undirected)
```

```python
directed_graph = nx.DiGraph()

directed_graph.add_nodes_from([1, 2, 3, 4, 5, 6])

directed_graph.add_edges_from(
    [(1, 2), (6, 1), (2, 3), (3, 2), (4, 1), (3, 1), (2, 4), (6, 3)])
adjacency_matrix_directed = nx.adjacency_matrix(directed_graph).todense()
link_list_directed = list(directed_graph.edges)
print("Adjacency matrix directed: \n", adjacency_matrix_directed)

print("\nLinked_list directed: \n", link_list_directed)
```

### Determine the average clustering coefficient of the network shown in Image 2.20a

```python
clustering_coefficient_undirected = nx.average_clustering(undirected_graph)
print(clustering_coefficient_undirected)
```

### If you switch the labels of nodes 5 and 6 in Image 2.20a, how does that move change the adjacency matrix? And the link list?

If you swithc the labels of nodes 5 and 6, you swap the 5 column with the 6 column, and the 5 row with the 6 row. For the linked list, you swap out every instance of 6 with 5

### What kind of information can you not infer from the link list representation of the network that you can infer from the adjacency matrix?

you can not detect nodes that are disconnected from the rest of the graph in the link list, but you can detect disconnected nodes from the adjacency matrix


### In the (a) network, how many paths (with possible repetition of nodes and links) of length 3 exist starting from node 1 and ending at node 3? And in (b)?

For network a, there are 4 paths of length 3 from node 1 to node 3: 
[1,4,1,3], [1,4,2,3], [1,2,1,3], [1,6,1,3]

For network b, no path exsists



## Solve exercise 2.5 ('Bipartite Networks') from NS using NetworkX in your notebook.


### Construct its adjacency matrix. Why is it a block-diagonal matrix?

```python
undirected_graph = nx.Graph()
undirected_graph.add_nodes_from([x for x in range(1, 12)])
undirected_graph.add_edges_from(
    [(1, 7), (2, 9), (7, 3), (8, 3), (3, 9), (9, 4), (4, 10), (9, 5), (5, 11), (6, 11)])
adjacency_matrix_undirected = nx.adjacency_matrix(undirected_graph).todense()

print("Adjacency matrix: \n", adjacency_matrix_undirected)
```

Every matrix is block-diagonal with atleast one block. Every block represents one connected component. Since this network consists of one connected component, it has one block.

<!-- #region tags=[] -->
### Construct the adjacency matrix of its two projections, on the purple and on the green nodes, respectively.
<!-- #endregion -->

```python
purple_projection = nx.bipartite.projected_graph(
    undirected_graph, [1, 2, 3, 4, 5, 6])
adjacency_matrix_purple = nx.adjacency_matrix(purple_projection).todense()
print("Adjacecny matrix purple projection: \n", adjacency_matrix_purple)

green_projection = nx.bipartite.projected_graph(
    undirected_graph, [7, 8, 9, 10, 11])
adjacency_matrix_green= nx.adjacency_matrix(green_projection).todense()
print("\nAdjacecny matrix green projection: \n", adjacency_matrix_green)
```

### Calculate the average degree of the purple nodes and the average degree of the green nodes in the bipartite network.

```python
def calculateAverageDegree(graph, nodes):
    n = len(graph)
    sumOfDegrees = 0
    for node in nodes:
        sumOfDegrees += graph.degree[node]
    return sumOfDegrees / n


calculateAverageDegree(undirected_graph, [1,2,3,4,5,6])

```

```python
calculateAverageDegree(undirected_graph, [7,8,9,10,11])
```

### Calculate the average degree in each of the two network projections. Is it surprising that the values are different from those obtained in point (c)?

```python
nx.draw(purple_projection, with_labels=True, font_weight='bold')
plt.show()

calculateAverageDegree(purple_projection, [1,2,3,4,5,6])
```

```python
nx.draw(green_projection, with_labels=True, font_weight='bold')
plt.show()
calculateAverageDegree(green_projection,  [7,8,9,10,11])
```

It is not suprising that the values are different from those abained in point c. From the two graphs above, it is clear that the bottom graph has a lower amount of average degree than the top one.


### What is the maximum number of links Lmax the network can have?

Maximum numer of links is N1 * N2. Every node in N1 can have a link to every node in N2 wich results in N1*N2 link.


### How many links cannot occur compared to a non-bipartite network of size N = N1 + N2 ?



Maximum links non-bipartite network of size N = N1 + N2 = N(N-1)/2

(N(N-1)/2) - (N1*N2)

= (1/2)(N1^2 + N2^2 - N1 - N2)



### If N1‹‹N2 , what can you say about the network density, that is the total number of links over the maximum number of links, Lmax?



This will lead to a sparse graph, as a larger difference in N1 compared to N2 leads to fewer edges


### Find an expression connecting N1, N2 and the average degree for the two sets in the bipartite network, 〈k1〉 and 〈k2〉.


Sum of degrees in set 1 = s1, and sum of degrees in set 2  = s2. since it is a bipartite network, they must be the same.

s1=s2=L.
=>
k1=s1/N1 and k2=s2/N2
=>
N1k1=N2k2


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

k = 10000
sampled_nodes = random.sample(DG.nodes, k)
sampled_graph = DG.subgraph(sampled_nodes)

```

```python
# Visualize a subgraph of the random null directed network

k = 10000
sampled_nodes = random.sample(null.nodes, k)
sampled_graph = null.subgraph(sampled_nodes)
```

```python
# Sets up config loading from netwulf API and plots

stylized_network, config = nw.visualize(sampled_graph)

# Reproduce the visualisation
fig, ax = nw.draw_netwulf(stylized_network)
plt.show()
```

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


