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

```python
import networkx as nx
import matplotlib.pyplot as plt
```

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
import networkx as nx

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


### Find an expression connecting N1, N2 and the average degree for the two sets in the bipartite network, 〈k1〉 and 〈k2〉.


Sum of degrees in set 1 = s1, and sum of degrees in set 2  = s2. since it is a bipartite network, they must be the same.

s1=s2=L.
=>
k1=s1/N1 and k2=s2/N2
=>
N1k1=N2k2

```python

```
