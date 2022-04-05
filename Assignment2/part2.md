---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: "1.3"
      jupytext_version: 1.13.7
  kernelspec:
    display_name: comsci-venv
    language: python
    name: comsci-venv
---

# Part 2: Sentiment analysis

```python
import pandas as pd
import nltk
import re
import labMTsimple
```

1. > Pick a day of your choice in 2020. We call it ${d}$. It is more interesting if you pick a day where you expect something relevant to occur (e.g. Christmas, New Year, Corona starting, the market crashes...).

Choose d = March 13 2020, corona is anounced a national emergency

2. > Build two lists ${l}$ and ${l_ref}$ containing all tokens for submissions posted on r/wallstreebets on day ${d}$, and in the 7 days preceding day ${d}$, respectively.

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

### Create the two lists ${l}$ and ${l_ref}$

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

3. > For each token ${i}$ , compute the relative frequency in the two lists ${l}$ and ${l_ref}$ . We call them ${p(i, l)}$ and ${p(i, l_ref)}$, respectively. The relative frequency is computed as the number of times a token occurs over the total length of the document. Store the result in a dictionary.

## Create functions that

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

4. > For each token ${i}$ , compute the difference in relative frequency ${\delta p(i) = p(i,l)-p(i,l_ref)}$ . Store the values in a dictionary. Print the top 10 tokens (those with largest relative frequency). Do you notice anything interesting?

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
happinesDict = createHappinesDict(labMt, i)
happined_dict_ref = createHappinesDict(labMt, i_ref)

```

```python
def calculateDeltaPhi(happinesDict, differenceDict):
    sigmaDict = dict()
    for key in i:
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

$\delta\Phi$ identifies which words contribute the most to this difference

7. > Now install the shifterator Python package. We will use it for plotting Word Shifts.

```python
import shifterator as sh
```

8. > Use the function shifterator.WeightedAvgShift to plot the WordShift, showing which words contributed the most to make your day of choice d happier or more sad then days in the preceding 7 days. Comment on the figure.

```python
sentiment_shift = sh.WeightedAvgShift(type2freq_1=freqDist,
                                      type2freq_2=freqDist_rel,
                                      type2score_1='labMT_English',
                                      reference_value=5,
                                      stop_lens=[(4,6)]
                                     )
sentiment_shift.get_shift_graph(detailed=False,
                                system_names=['13. March.', '6. march - 13 march'])
```

9. > How do words that you printed in step 6 relate to those shown by the WordShift?
