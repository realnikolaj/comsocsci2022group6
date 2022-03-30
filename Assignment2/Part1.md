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
    display_name: Python [conda env:comsocsci2022]
    language: python
    name: conda-env-comsocsci2022-py
---

## Part 1: TF-IDF


1. Tokenize the __text__ of each submission. Create a column __tokens__ in your dataframe containing the tokens. Remember to follow the instructions in Week 6, Exercise 3.  

```python
import pandas as pd
import re
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

from collections import Counter
top15 = Counter(tickers).most_common(15)


# Now we tokenize and clean the text
from nltk.tokenize import word_tokenize
import re, nltk
# Tokenize and filter out punctuation and numbers
wallstreet_subs['tokens'] = [word_tokenize(re.sub(r'[^\w\s]|\d','',w.lower())) for w in wallstreet_subs['text']]

# url filter
wallstreet_subs['tokens'] = [[c for c in w if not "http" in c] for w in wallstreet_subs['tokens'] ]

# Lemmatize is probably a better idea than stemming, as the word cloud is more understandable
from nltk.stem import WordNetLemmatizer
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
import math
import pickle

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





Why aren't the TFs not necessarily a good description of the stocks?





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


!!!!!!!!!!!!!Rewrite!!!!!!!!!!!!!!!!!!!!!

Using log means that general words used in a lot of other documents will not count as much, so we can compare high frequency terms with low frequency.
Log10 is used because it will reduce the impact of the small number of documents that a term appears in by a factor of of a power of 10.


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
import collections
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


6. Visualize the results in a Wordcloud and comment your results (follow the instrutions in Week 6, Exercise 4).

```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt
for stock in top5_stocklist:
    wordcloud = WordCloud(width=1500, height=1000).generate_from_frequencies(TFIDF[stock])
    plt.figure(figsize=[15,10])
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.suptitle(stock)
    plt.show()
```

!!!!!!!!!!!! Comment on results !!!!!!!!!!!!
