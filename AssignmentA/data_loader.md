---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
#from gevent import monkey as curious_george
#curious_george.patch_all(thread=False, select=False)

import bz2
import pickle
import _pickle as cPickle
import nest_asyncio
import asyncio
from requests_html import AsyncHTMLSession
import requests
import re
import pandas as pd
from itertools import islice
nest_asyncio.apply()
data/law_data.csv", index=False)
```


"""
Usage: Exports a compressed pickled dataframe object 'picl_law_data.pbz2'
To read back the object use:
    import bz2
    import pickle
    import _pickle as cPickle
    data = bz2.BZ2File('../data/picl_law_data.pbz2', 'rb')
    data = cPickle.load(data)
"""
'''
USAGE:
Load url's produced by get_edges.py
Setting coloumns to produce in the resulting law_data.csv file
'''
```python
nodes = pd.read_table(open('../data/edges.csv', 'r'), sep=",", header=None, dtype=str)
columns = ['id', 'title', 'documentTypeId', 'shortName', 'document_text', 'isHistorical', 'ressort', 'EliUrl', 'edges'] # FJERNET 'Nummer', Add attributes to this list

# Using list to append for optimization note: don't increase dataframes in a loop!
listdata = []
```


```python
async def fetch(session, url):
    """
    :param session: AsyncHTMLSession
    :param url: Self-explanatory, defined in main() 'tasks' list and called by await.gather(*tasks)
    :return: The variables to append to the output csv file

    Using an async fetcher to handle asynchronous request calls
    Note that this function utilizes two different .get() function (requsts_html.AsyncHTMLSession and requests.get()) \
    the later which talks directly to source provided API and the former which renders the browser destined Javascript \
    and collects edges (references to other documents)
    """
    url = url._1                                    # We really should cleanup the edges.csv formatting (low-priority)
    postfix = url.split('document/')[1]             # Used to call the AsyncHTMLSession.get() below
    r = await session.get(
        'https://www.retsinformation.dk/{}'.format(postfix))
    pattern = '/eli/.*'                             # Match this pattern to find links (urls) for actual documents and not other site-links like FAQ etc.
    await r.html.arender(
        retries=20,
        timeout=25,
        sleep=15,
        keep_page=True
    )                                                # Asynchronous await on the browser render NOTE the parameters
    edges = [edge for edge in r.html.links
             if (re.match(pattern, edge)
                 and len(edge) < 30)
             ]                                       # Condition found references
    resp = requests.get(url)                         # Requests session i.e. NOT Async - Maybe it should be awaited (await requests.get(url) - remains to be tested
    document = resp.json()                           # Source API response

    '''
    Add varaibles below to get more info
    OBS!!! Don't forget to add them to the return of fetch() (this function), otherwise the main() call won't append them to the data
    '''
    unique_identity = document[0]["id"]
    title = document[0]["title"]
    ressort = document[0]["ressort"]
    documentTypeId = document[0]["documentTypeId"]
    shortName = document[0]["shortName"]
    EliUrl = url
    isHistorical = document[0]["isHistorical"]
    document_text = document[0]["documentHtml"]

    return unique_identity, title, documentTypeId, shortName, document_text, isHistorical, ressort, EliUrl, edges

async def main(df, step):
    """
    :param df: List of URL's
    :param step: Dumb fix for errors caused by potentially hundreds of async render requests (request_html.AsyncHTMLSession)
    :return: List of row, data
    """

    with AsyncHTMLSession() as session:
        # Dirty batch split for smaller async sessions
        for begin in range(0, df.size-1, step):
            stop = begin+step
            tasks = [fetch(session, url) for url in islice(df.itertuples(), begin, stop)]
            for response in await asyncio.gather(*tasks):
                listdata.append(response)
            print(f'Done:{begin}')
    return
```


```python
step = 18 # Batch size, Higher is faster but more prone to timeouts and connections errors
asyncio.run(main(nodes.T, step))
result = pd.DataFrame(data=listdata, columns=columns)

with bz2.BZ2File('../data/picl_law_data' + '.pbz2', 'w') as f:
    cPickle.dump(result, f)
```


