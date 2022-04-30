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
```

```python
import bz2
import pickle
import _pickle as cPickle
data = bz2.BZ2File('../data/picl_data.pbz2', 'rb')
data = cPickle.load(data)
```

```python
data = data.head(30)
```

```python
#This is maybe redundant
def getIdFromUrl(url):
    try:
        id = url.split("lta/")
        if (len(id) == 1):
            id = url.split("ft/")
        if (len(id) == 1):
            id = url.split("ltb/")
        id = id[1]
    except:
        print("error getting id from. ", url)
        return -1 #May be bad
    return id
    
```

## Create graph

```python
G = nx.DiGraph()

idToIndexMapper = {}

uniqueIndex = 0

for index, row in data.iterrows():
    parent_id = getIdFromUrl(row.url)
    if parent_id not in idToIndexMapper:
        idToIndexMapper[parent_id] = {"indexInGraph": uniqueIndex, "url": row.url}
        uniqueIndex += 1
    edges = row["metadata"]
    for edge in edges:
        for key in edge:
            edge_id = getIdFromUrl(key)
            if edge_id not in idToIndexMapper:
                idToIndexMapper[edge_id] = {"indexInGraph": uniqueIndex, "url": edge}
                uniqueIndex += 1
            G.add_edge(idToIndexMapper[parent_id]["indexInGraph"], idToIndexMapper[edge_id]["indexInGraph"])

nx.draw(G)
```

```python
sorted(G.degree, key=lambda x: x[1], reverse=True)

for key in idToIndexMapper:
    if idToIndexMapper[key]["indexInGraph"] == 91:
        print("Highest degree", idToIndexMapper[key])

```

```python
nx.degree_centrality(G)
```
