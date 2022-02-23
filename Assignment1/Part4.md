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

# Overview

Networks (a.k.a. graphs) are widely used mathematical objects for representing and analysing social systems. 
This week is about getting familiar with networks, and we'll focus on four main aspects:

* Basic mathematical description of networks
* The `NetworkX` library.
* Building the network of GME redditors.
* Basic analysis of the network of GME redditors.

test


# Part 1: Basic mathematical description of networks

This week, let's start with some lecturing. You will watch some videos made by Sune for his course _Social Graphs and Interactions_, where he covers networks in details.  

> **_Video Lecture_**. Start by watching the ["History of Networks"](https://youtu.be/qjM9yMarl70). 


```python
from IPython.display import YouTubeVideo
YouTubeVideo("qjM9yMarl70",width=800, height=450)
```

> **_Video Lecture_**. Then check out a few comments on ["Network Notation"](https://youtu.be/MMziC5xktHs). 

```python
YouTubeVideo("MMziC5xktHs",width=800, height=450)
```

> _Reading_. We'll be reading the textbook _Network Science_ (NS) by Laszlo Barabasi. You can read the whole 
> thing for free [**here**](http://barabasi.com/networksciencebook/). 
> 
> * Read chapter 1\.
> * Read chapter 2\.
> 


> _Exercises_  
> _Chapter 1_ (Don't forget that you should be answering these in a Jupyter notebook.) 
> 
> * List three different real networks and state the nodes and links for each of them.
> * Tell us of the network you are personally most interested in. Address the following questions:
>   * What are its nodes and links? 
>   * How large is it? 
>   * Can be mapped out? 
>   * Why do you care about it? 
> * In your view what would be the area where network science could have the biggest impact in the next decade? Explain your answer - and base it on the text in the book. 
> 
> _Chapter 2_
> 
> * Section 2.5 states that real networks are sparse. Can you think of a real network where each node has _many_ connections? Is that network still sparse? If yes, can you explain why?
> 
> There are more questions on Chapter 2 below.
> 


# Part 2: Exercises using the `NetworkX` library

We will analyse networks in Python using the [NetworkX](https://networkx.org/) library. The cool thing about networkx is that it includes a lot of algorithms and metrics for analysing networks, so you don't have to code things from scratch. Get started by running the magic ``pip install networkx`` command. Then, get familiar with the library through the following exercises: 

>  *Exercises*:

> * Go to the NetworkX project's [tutorial page](https://networkx.org/documentation/stable/tutorial.html). The goal of this exercise is to create your own notebook that contains the entire tutorial. You're free to add your own (e.g. shorter) comments in place of the ones in the official tutorial - and change the code to make it your own where ever it makes sense.
> * Go to Section 2.12: [Homework](http://networksciencebook.com/chapter/2#homework2), then
>     * Write the solution for exercise 2.1 (the 'Königsberg Problem') from NS in your notebook.
>     * Solve exercise 2.3 ('Graph representation') from NS using NetworkX in your notebook. (You don't have to solve the last sub-question about cycles of length 4 ... but I'll be impressed if you do it).
>     * Solve exercise 2.5 ('Bipartite Networks') from NS using NetworkX in your notebook.


# Part 3: Building the GME redditors network


Ok, enough with theory :) It is time to go back to our cool dataset it took us so much pain to download! And guess what? We will build the network of GME Redditors. Then, we will use some Network Science to study some of its properties.

<!-- #region -->


> 
> *Exercise*: Build the network of Redditors discussing about GME on r\wallstreetbets. In this network, nodes correspond to authors of comments, and a direct link going from node _A_ to node _B_ exists if _A_ ever answered a submission or a comment by _B_. The weight on the link corresponds to the number of times _A_ answered _B_. You can build the network as follows:
>
> 1. Open the _comments dataset_ and the _submission datasets_ (the first contains all the comments and the second cointains all the submissions) and store them in two Pandas DataFrames.
> 2. Create three dictionaries, using the command ``dict(zip(keys,values))``, where keys and values are columns in your dataframes. The three dictionaries are the following:
>     * __comment_authors__: (_comment id_, _comment author_)
>     * __parent__:  (_comment id_ , _parent id_)
>     * __submission_authors__: (_submission id_, _submission author_)
>
> where above I indicated the (key, value) tuples contained in each dictionary.
>
> 3. Create a function that take as input a _comment id_ and outputs the author of its parent. The function does two things:
>     * First, it calls the dictionary __parent__, to find the _parent id_ of the comment identified by a given _comment id_. 
>     * Then, it finds the author of  _parent id_. 
>          * if the _parent id_ starts with "t1_", call the __comment_authors__ dictionary (for key=parent_id[3:])
>          * if the _parent id_ starts with "t3_", call the __submission_authors__ dictionars (for key=parent_id[3:])
>
> where by parent_id[3:], I mean that the first three charachters of the _parent id_ (either "t1_" or "t3_" should be ingnored).
>
> 4. Apply the function you created in step 3. to all the comment ids in your comments dataframe. Store the output in a new column, _"parent author"_, of the comments dataframe. 
> 5. For now, we will focus on the genesis of the GME community on Reddit, before all the hype started and many new redditors jumped on board. For this reason, __filter all the comments written before Dec 31st, 2020__. Also, remove deleted users by filtering all comments whose author or parent author is equal to "[deleted]". 
> 6. Create the weighted edge-list of your network as follows: consider all comments (after applying the filtering step above), groupby ("_author_", _"parent author"_) and count. 
> 7. Create a [``DiGraph``](https://networkx.org/documentation/stable//reference/classes/digraph.html) using networkx. Then, use the networkx function [``add_weighted_edges_from``](https://networkx.org/documentation/networkx-1.9/reference/generated/networkx.DiGraph.add_weighted_edges_from.html) to create a weighted, directed, graph starting from the edgelist you created in step 5.
> 8. Save the Network as a json file. 
<!-- #endregion -->

```python
import pandas as pd
import networkx as nx
import datetime
import matplotlib.pyplot as plt
import json
```

```python
comments = pd.read_csv("wsb_comments_formatted.csv")
submissions = pd.read_csv("WSB_w1.csv")


```

```python
comment_authors = dict(zip(comments.id, comments.author))
parent = dict(zip(comments.id, comments.parent_id))
submission_authors = dict(zip(submissions.id, submissions.author))
```

```python
def outputAuthor(commentIdKey, 
                 commentAuthorsDict = comment_authors,
                 parentDict = parent,
                 submissionAuthorsDict = submission_authors):
    
    parentIdValue = parentDict.get(commentIdKey)
    if parentIdValue[1] == "1":
        return commentAuthorsDict.get(parentIdValue[3:])
    else:
        return submissionAuthorsDict.get(parentIdValue[3:])

outputAuthor('fkqnewk')
```

```python
# Adding parent authors to comments data
comments["parent_author"] = comments.apply(lambda row : outputAuthor(row["id"]), axis=1)
```

```python
# Filtering date and dropping deleted submitters and parents
date = datetime.datetime(year=2020, month=12, day=31)
comments = comments.drop(labels = comments[comments["author"] ==  "[deleted]"].index)
comments = comments.drop(labels = comments[comments["parent_author"] ==  "[deleted]"].index)
comments = comments.drop(labels = comments[comments["created"] <  date.timestamp()].index)
```

```python
edgeList = comments.groupby(by = ["author", "parent_author"],
                            as_index=False).count()

edgeList = pd.DataFrame(edgeList, columns=["author", "parent_author", "created_utc"])
edgeList.rename(columns={"author" : "source", "parent_author" : "target", "created_utc" : "weight"}, inplace=True)
#edgeList = edgeList.sample(250)
#G = nx.from_pandas_edgelist(edgeList)
#                            source="author",
#                            target="parent_author")
#                            edge_attr="created_utc")
```

```python
DG = nx.DiGraph()
DG.add_weighted_edges_from(edgeList.values)
```

```python
dataJson = nx.readwrite.json_graph.node_link_data(DG)
with open('json_network.json', 'w') as outfile:
    json.dump(dataJson, outfile)
```

```python
fig, ax = plt.subplots(figsize=(16,16))
#edge_collection = nx.draw_networkx_edges(DG, ax=ax)
#edge_labels = nx.draw_networkx_labels(DG, ax=ax)
nx.draw(DG, with_labels=True)
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
