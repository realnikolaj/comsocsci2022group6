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

<!-- #region -->
# Assignment 2
## Group 6
> s183930 - Nikolaj S. Povlsen

> s184208 - Steffen Holm Cordes

> s217176 - Johan Fredrik Bjørnland

## Link to Git repository
https://github.com/realnikolaj/comsocsci2022group6



## Contribution statement

We worked collaboratory as a group, main responsibles are:
Part 1: Steffen S184208  
Part 2: Johan   S217176  
Part 3: Nikolaj S183930
<!-- #endregion -->

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
from collections import Counter
from nltk.tokenize import word_tokenize
import re, nltk
from nltk.stem import WordNetLemmatizer
import math
import pickle
import collections
from wordcloud import WordCloud
import labMTsimple

```
## Part 1: TF-IDF


1. Tokenize the __text__ of each submission. Create a column __tokens__ in your dataframe containing the tokens. Remember to follow the instructions in Week 6, Exercise 3.  

```python

wallstreet_subs = pd.read_csv('../data/wallstreet_subs.csv', parse_dates=['created_utc']).set_index('created_utc')
wallstreet_subs['text'] = wallstreet_subs[['title', 'selftext']].agg(' '.join, axis=1)

# Identify stock mentions before we clean the data
def find_tickers(text):
    tick = re.findall("\$[a-zA-Z0-9]*",text)
    tick =[t[1:].upper() for t in tick]
    tick = [t for t in tick if not t.isdigit()]
    return tick

result = [find_tickers(w) for w in wallstreet_subs['text'] if not find_tickers(w) == []] # Find the tickers for every text
result = [item for sublist in result for item in sublist] # flatten the list of lists
result = [i for i in result if i and len(i) < 5] # remove empty strings and strings longer than 4 letters
pattern = re.compile("([0-9]+)[KC]")
result = [i for i in result if not pattern.match(i)] # remove 1K, 10K, 100K, 50C, 2K so on
tickers = {item:result.count(item) for item in result} # update dictionary with ticker and count


top15 = Counter(tickers).most_common(15)


# Now we tokenize and clean the text

# Tokenize and filter out punctuation and numbers
wallstreet_subs['tokens'] = [word_tokenize(re.sub(r'[^\w\s]|\d','',w.lower())) for w in wallstreet_subs['text']]

# url filter
wallstreet_subs['tokens'] = [[c for c in w if not "http" in c] for w in wallstreet_subs['tokens'] ]

# Lemmatize is probably a better idea than stemming, as the word cloud is more understandable

lemmatizer = WordNetLemmatizer()
wallstreet_subs['tokens'] = [[lemmatizer.lemmatize(t) for t in token] for token in wallstreet_subs['tokens']]

# remove stopwords
stopwords = nltk.corpus.stopwords.words('english')
wallstreet_subs['tokens'] = [[t for t in sub if t not in stopwords] for sub in wallstreet_subs['tokens']]
```

2. Find submissions discussing at least one of the top 15 stocks you identified above (follow the instructions in Week 6, Exercise 3).

```python
# We only need the stock name to find the mention, not the count
top15 = [t[0] for t in top15]

def find_stock(text,stocklist):
    return [s for s in stocklist if '$'+s in text.upper()]

wallstreet_subs['stocks'] = [find_stock(t,top15) for t in wallstreet_subs['text']]

# Explode multiple stocks associated with one comment
wallstreet_subs = wallstreet_subs.explode('stocks')

# Replace NaN with "Other"
wallstreet_subs.fillna("Other", inplace=True)
```

3. Now, we want to find out which words are important for each *stock*, so we're going to create several ***large documents, one for each stock***. Each document includes all the tokens related to the same stock. We will also have a document including discussions that do not relate to the top 15 stocks.

```python
stocklist = top15.copy()
stocklist.insert(0,'Other')

stock_documents = pd.DataFrame()
for stock in stocklist:
    stock_documents[stock] = [[t for tokens in wallstreet_subs['tokens'][wallstreet_subs['stocks'] == stock] for t in tokens]]
```

4. Now, we're ready to calculate the TF for each word. Find the top 5 terms within __5 stocks of your choice__. 

```python


# TFterms = {}
# TFtop5terms = {}
# for stock in stocklist:
#     TFterms[stock] = dict(Counter(stock_documents[stock][0]))
#     for key in TFterms[stock]:
#         TFterms[stock][key] = TFterms[stock][key]/len(TFterms[stock])
#         # Use a set instead
#         #TFterms[stock][key].add(TFterms[stock][key] / len(TFterms[stock]))
#
#         TFtop5terms[stock] = {k: TFterms[stock][k] for k in list(TFterms[stock])[:5]}

TFterms = pickle.load(open('../data/TFterms.pickle', 'rb'))
TFtop5terms = pickle.load(open('../data/TFtop5terms.pickle', 'rb'))

# Print top5 terms for the chosen 5 stocks: Tesla, Virgin Galactic, Zoom, Palantir Technologies and GameStop 
top5_stocklist = ['TSLA', 'SPCE', 'ZM', 'PLTR', 'GME']
for stock in top5_stocklist:
    print(stock)
    print(sorted(TFterms[stock].items(), key=lambda x: x[1], reverse=True)[:5])
```

Describe similarities and differences between the stocks.





The top 5 TF terms are mostly about generic stock trading terms like "stock", "earnings" and their stock ticker. The information from the top 5 TF terms do not reveal any noticeable differences between the stocks.


Why aren't the TFs not necessarily a good description of the stocks?





Because TF produces the most frequent words in a document, these will most often also be generic terms that appear often in the english language pertaining to the subject.


Next, we calculate IDF for every word. 

```python
# IDFterms = {}
# number_documents = len(stock_documents.T)
# for stock in stocklist:
#     IDFterms[stock] = {}
#     for term in stock_documents[stock][0]:
#         term_in_num_doc = 0
#         for s in stocklist:
#             if term in stock_documents[s][0]:
#                 term_in_num_doc += 1
#         if not term_in_num_doc == 0:
#             IDFterms[stock][term] = math.log10(number_documents / float(term_in_num_doc))
#         else:
#             IDFterms[stock][term] = 0

IDFterms = pickle.load(open('../data/IDFterms.pickle', 'rb'))
```

What base logarithm did you use? Is that important?


Using log means that general words used in a lot of other documents will not count as much, so we can compare high frequency terms with low frequency.
Log10 is used because it will reduce the impact of the small number of documents that a term appears in by a factor of of a power of 10. It is not that important which base logarithm is being used, but it will scale the actual numbers e.g. Log10 will produce much smaller numbers than Log2.


5. We're ready to calculate TF-IDF. Do that for the __5 stock of your choice__. 

```python
# Calculate TF-IDF for each word
TFIDF = {}
for stock in stocklist:
    TFIDF[stock] = {}
    for term in TFterms[stock]:
        TFIDF[stock][term] = TFterms[stock][term] * IDFterms[stock][term]
```

List the 10 top TF words for each stock.

```python
# Print top10 TF terms for the chosen 5 stocks

for stock in top5_stocklist:
    print(stock)
    print(collections.Counter(TFterms[stock]).most_common(10))
```

List the 10 top TF-IDF words for each stock.

```python
# Print top10 TF-IDF terms for the chosen 5 stocks
for stock in top5_stocklist:
    print(stock)
    print(collections.Counter(TFIDF[stock]).most_common(10))
```

Are these 10 words more descriptive of the stock? If yes, what is it about IDF that makes the words more informative?


For the top 10 TF words for the Tesla stock, we see a lot of non-descriptive words such as: tsla, stock, call, market, share and and this holds true for the other stocks. For the TF-IDF we see more descriptive words such as vale and nickel, which refers to Tesla signing a nickel supply deal with the mining company Vale.

We also see a mention of Citron Research for the Palantir stock TF-IDF, who announced they would initiate a short position on Palantir in late 2020.

IDF makes the top terms more informative, because it takes into account how many other documents contain that term and therefore gives a measure of specificity. This is good because this is exactly the opposite what we saw with the generic words produced by TF and if these words are mentioned across many documents their TF-IDF score will be lowered, so more unique but still frequent terms gets a higher score.


6. Visualize the results in a Wordcloud and comment your results (follow the instrutions in Week 6, Exercise 4).

```python

for stock in top5_stocklist:
    wordcloud = WordCloud(width=1500, height=1000).generate_from_frequencies(TFIDF[stock])
    plt.figure(figsize=[15,10])
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.suptitle(stock)
    plt.show()
```

We see that the wordcloud is a powerful representation of the TF-IDF terms to identify important terms relatin to the stocks but also associated terms. E.g. Virgin Space is often compared to Maxar Technologies, Gamestop to Blockbuster and Zoom is often compared to Slack. This gives the reader an interesting and visual way to investigate terms relating to a topic.

# Part 2: Sentiment analysis

```python

```

1. > Pick a day of your choice in 2020. We call it ${d}$. It is more interesting if you pick a day where you expect something relevant to occur (e.g. Christmas, New Year, Corona starting, the market crashes...).

Choose d = March 13 2020, corona is anounced a national emergency

2. > Build two lists ${l}$ and ${l_{ref}}$ containing all tokens for submissions posted on r/wallstreebets on day ${d}$, and in the 7 days preceding day ${d}$, respectively.

```python
# March 13 00:01 2020 Represented as timestamp
start_l = "1584054011"
# March 13 23:059 2020 Represented as timestamp

end_l = "1584140351"

# March 6th 2020 00:01 2020 Represented as timestamp
start_l_ref = "1583535551"
# March 13 00:01 2020 Represented as timestamp
end_l_ref = "1584054011"

```

### Create the two lists ${l}$ and ${l_{ref}}$

```python
wsb_submissions = pd.read_csv('../data/wallstreet_subs_tokens_cleaned_lemma.csv', parse_dates=['created_utc']).set_index('created_utc')
wsb_submissions = wsb_submissions.sort_values('created_utc')


'''
Create a function that parses the many lists of tokens, removes unwanted characters and
creates one list with unique tokens
'''
def getAllTokens(dfCol):
    regex_pat = re.compile(r'\[|\]|\'', flags=re.IGNORECASE)
    l_tokens = dfCol.str.replace(regex_pat, '', regex=True)
    l_tokens = l_tokens.tolist()


    l_tokens = [x.split(",") for x in l_tokens]
    l_tokens = [item.strip() for sublist in l_tokens for item in sublist]
    return l_tokens

def getAllUniquetokens(tokens):
    return list(set(tokens))

#Filter out all rows that is not the correct time intervalls"""
l = wsb_submissions.loc[start_l:end_l]
l_ref = wsb_submissions.loc[start_l_ref:end_l_ref]
all_tokens = wsb_submissions.loc[start_l_ref:end_l]

# Convert from dataframe to list of tokens
l, l_ref, all_tokens = getAllTokens(l["tokens"]), getAllTokens(l_ref["tokens"]), getAllTokens(all_tokens["tokens"])


#Get the lists l, l_ref and all_tokens
unique_l, unique_l_ref, unique_all_tokens = getAllUniquetokens(l), getAllUniquetokens(l_ref), getAllUniquetokens(all_tokens)

```

3. > For each token ${i}$ , compute the relative frequency in the two lists ${l}$ and ${l_{ref}}$ . We call them ${p(i, l)}$ and ${p(i, l_{ref})}$, respectively. The relative frequency is computed as the number of times a token occurs over the total length of the document. Store the result in a dictionary.

## Create functions that calculate frequency and relative frequency

```python
def getFrequencyDist(tokens):
    frequencyDist = dict()
    for token in tokens:
        if token not in frequencyDist:
            frequencyDist[token] = 1
        else:
            frequencyDist[token] = frequencyDist[token] + 1
    return frequencyDist

freqDist = getFrequencyDist(l)
freqDist_rel = getFrequencyDist(l_ref)
```

```python
def getRelativeFrequencyDist(freqDistribution):
    distribution = dict(freqDistribution)
    totalWords = len(distribution)
    relativeFrequencyDist = distribution
    for key in relativeFrequencyDist:
        relativeFrequencyDist[key] = relativeFrequencyDist[key] / totalWords
    return relativeFrequencyDist


p_i_l = getRelativeFrequencyDist(freqDist)
p_i_l_ref = getRelativeFrequencyDist(freqDist_rel)


```

4. > For each token ${i}$ , compute the difference in relative frequency ${\delta p(i) = p(i,l)-p(i,l_{ref})}$ . Store the values in a dictionary. Print the top 10 tokens (those with largest relative frequency). Do you notice anything interesting?

### Compute difference in relative frequency

```python
def calculateDifRelFreq(l1, l_ref):
    difRelFreq = dict()
    for token in l1:
        l_ref_token = l_ref[token] if token in l_ref else 0
        difRelFreq[token] = l1[token] - l_ref_token
    return difRelFreq
```

### Top 10 tokens with largest difference in relative frequency

```python
differenceDict = calculateDifRelFreq(p_i_l, p_i_l_ref)

top10LargestDiff = sorted(differenceDict, key=differenceDict.get, reverse=True)[:10]
print("The 10 tokens with largest relative frequency are:")
for j in range(0, 10):
    print(f'{j + 1}. {top10LargestDiff[j]}')
```

5. > Now, for each token, compute the happiness ${h(i) = labMT(i) - 5}$ , using the labMT dictionary. Here, we subtract , so that positive tokens will have a positive value and negative tokens will have a negative value. Then, compute the product ${\delta\Phi = h(i) * \delta p(i)}$. Store the results in a dictionary.

## Compute ${h(i)}$

```python
def createHappinesDict(labMt_data, tokens):
    happinesDict = dict()
    for token in tokens:
        try:
            happinesDict[token] = labMt_data.loc[token].values[2] - 5
        except:
            pass
    return happinesDict
```

```python
labMt = pd.read_csv('../data/Hedonometer.csv').set_index("Word")

happinesDict = createHappinesDict(labMt, unique_l)
happined_dict_ref = createHappinesDict(labMt, unique_l_ref)

```

```python
def calculateDeltaPhi(happinesDict, differenceDict):
    sigmaDict = dict()
    for key in happinesDict:
        try:
            sigmaDict[key] = abs(happinesDict[key] * differenceDict[key])
        except:
            pass
    return sigmaDict

deltaPhiDict = calculateDeltaPhi(happinesDict, differenceDict)
```

6. > Print the top 10 tokens, ordered by the absolute value of ${abs(\delta\Phi)}$. Explain in your own words the meaning of ${\delta\Phi}$. If that is unclear, have a look at this page.

```python
top10LargestDiff = sorted(deltaPhiDict, key=deltaPhiDict.get, reverse=True)[:10]
print('The top 10 tokens ordered by the absolute value of delta phi')
for k in range(0,10):
    print(f'{k +1}. {top10LargestDiff[k]}')
```

## Explain in your own words the meaning of $\delta\Phi$.

$\delta\Phi$ represents the amount of impact a word has on the sentiment of a text. Taking the product of the happinesscore and the relative frequence of a word gives a good indication of how much that word contributes to a certain emotion in the text.

7. > Now install the shifterator Python package. We will use it for plotting Word Shifts.

```python
import shifterator as sh
```

8. > Use the function shifterator.WeightedAvgShift to plot the WordShift, showing which words contributed the most to make your day of choice d happier or more sad then days in the preceding 7 days. Comment on the figure.

```python
%%capture --no-stdout --no-display 
labMtDict = dict(zip(labMt["Word in English"], labMt["Happiness Score"]))

sentiment_shift = sh.WeightedAvgShift(type2freq_1=freqDist,
                                      type2freq_2=freqDist_rel,
                                      type2score_1=labMtDict,
                                      reference_value=5,
                                      stop_lens=[(4,6)]
                                     )
sentiment_shift.get_shift_graph(detailed=False,
                                system_names=['13. March.', '6. march - 13 march'])
```

9. > How do words that you printed in step 6 relate to those shown by the WordShift?

We can see a realy good corelation between the words from step 6 and those show by the wordshift. All the words from the top 10 is represented, except "like" and "can't".  

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



$M =\sum_{c=1}^{n_{c}}\left[\frac{L_{c}}{L}-\left(\frac{k_{c}}{2 L}\right)^{2}\right]$
(9.12)



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

#print(f'Modularity of Louvain partition: \n',
#      f'{modularity(model1)[0]:.2f} \n') #<-- Didn't seem to work

louvain_communities = list()
for _community in louvain_set_of_communities:  #add node to corresponding community and append the communities to a list of communities
    louvain_communities.append({key for key in louvain_partition.keys() if louvain_partition.get(key) == _community})

print(f' Modularity of Zachary club network [Built-in networkx function]: \n',
      f'{nx.algorithms.community.modularity(model1, louvain_communities):.2f}')
```

The Louvain split has a slightly lower modularity with a delta of 0.06 which means there's a less 'clean' partitioning.  
This split finds more communities which could indicate more intricacies or sub-commnunities within the network than just the Mr. Hi and Officer relationship. This is also hinted in the initial graph where it is visiually apparent that there's a few nodes acting as hubs or links between the two communities, which is probably what this algorithm finds.


***
12. >Compare the communities found by the Louvain algorithm with the club split partitioning by creating a matrix D with dimension (2 times A), where A is the number of communities found by Louvain. We set entry D(i,j) to be the number of nodes that community i has in common with group split j. The matrix D is what we call a confusion matrix. Use the confusion matrix to explain how well the communities you've detected correspond to the club split partitioning.


<div class="alert alert-block alert-danger">
END
</div>

