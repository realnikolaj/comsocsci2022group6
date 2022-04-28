import argparse

# from gevent import monkey
# monkey.patch_all()
import bz2
import pickle
import _pickle as cPickle
import nest_asyncio
import asyncio
from requests_html import AsyncHTMLSession
import re
import pandas as pd
import json
from itertools import islice
nest_asyncio.apply()
parser = argparse.ArgumentParser()
args = parser.parse_args()

df = bz2.BZ2File('../data/picl_law_data.pbz2', 'rb')
#df = bz2.BZ2File('data/picl_law_data.pbz2', 'rb')
df = cPickle.load(df)
edgedata = []


columns = ['id', 'title', 'documentTypeId', 'shortName', 'full_text', 'isHistorical', 'ressort', "url", "caseHistoryReferenceGroup", "stateLabel", "edges", "metadata"] # FJERNET 'Nummer', Add attributes to this list

# Using list to append for optimization note: don't increase dataframes in a loop!
listdata = []
pattern = '/eli/.*'

async def fetch(session, id, url, edges=True):
    """
    :param session: AsyncHTMLSession
    :param url: Self-explanatory, defined in main() 'tasks' list and called by await.gather(*tasks)
    :return: The variables to append to the output csv file

    Using an async fetcher to handle asynchronous request calls
    Note that this function utilizes two different .get() function (requsts_html.AsyncHTMLSession and requests.get()) \
    the later which talks directly to source provided API and the former which renders the browser destined Javascript \
    and collects edges (references to other documents)
    """
    if edges:
        urlR = url.split('api')[0] + url.split('document/')[1]
        resp = await session.get(urlR, timeout=120)
        while (resp.status_code != 200):
            resp = await session.get(url, timeout=120)
        await resp.html.arender(retries=100, wait=20, timeout=120, sleep=15)
        edges = [{url.split('/eli')[0]+edge:'{placeholder}'} for edge in resp.html.links if (re.match(pattern, edge) and len(edge) < 30)]
        session.close()
        #[{url.split('document/')[1]+edge:shortName} for edge, shortName in zip(resp.html.links, resp.html.text.find("MISSING IMPLEMEN") if (re.match(pattern, edge) and len(edge) < 30)]
    else:
       edges = None

    resp = await session.get(url, timeout=120)
    while (resp.status_code != 200):
        resp = await session.get(url, timeout=120)
    document = json.loads(resp.text)                          # Source API response
    '''
    Add varaibles below to get more info
    OBS!!! Don't forget to add them to the return of fetch() (this function), otherwise the main() call won't append them to the data
    '''
    unique_identity = document[0]["id"]
    title = document[0]["title"]
    ressort = document[0]["ressort"]
    documentTypeId = document[0]["documentTypeId"]
    shortName = document[0]["shortName"]
    url = resp.html.url
    isHistorical = document[0]["isHistorical"]
    full_text = str(resp.html.full_text)  #document[0]["documentHtml"]
    try:
        caseHistoryReferenceGroup = document[0]["caseHistoryReferenceGroup"][0]['references']
        stateLabel = document[0]["caseHistoryReferenceGroup"][0]['stateLabel']
    except:
        caseHistoryReferenceGroup = None
        stateLabel = None
    metadata = document[0]["metadata"]
    await print('Yeahhhhhhhhh')
    return id, unique_identity, title, documentTypeId, shortName, full_text, isHistorical, ressort, url, caseHistoryReferenceGroup, stateLabel, edges, metadata

async def main(df, loop=None, stepsize=10, edges=True, workers=None):
    """
    :param df: List of URL's
    :param step: Dumb fix for errors caused by potentially hundreds of async render requests (request_html.AsyncHTMLSession)
    :return: List of row, data
    """
    # Dirty batch split for smaller async sessions
    for begin in range(0, df.size - 1, step):
        with AsyncHTMLSession(workers=workers) as session:
            # Initialize the event loop
            loop = session.loop
            thread_pool = session.thread_pool

            # Batch split intervals
            stop = begin + step
            # Use list comprehension to create a list of
            # tasks to complete. The executor will run the `fetch`
            # function for each url in the urlslist
            tasks = [await loop.run_in_executor(
                    thread_pool,
                    fetch,
                    *(session, name, url, True)
                )
                     #for id, name, url in df.itertuples()  # For multiple arguments to fetch function
                     #]
                     for id, name, url in islice(df.itertuples(), begin, stop)  # For multiple arguments to fetch function
                     ]
            for response in await asyncio.gather(*tasks):
                listdata.append(response)
    return


if __name__ == '__main__':
    step = 18 # Batch size, Higher is faster but more prone to timeouts and connections errors
    asyncio.run(main(df[['id', 'url']][:18], stepsize=step, edges=True, workers=16))
    result = pd.DataFrame(data=edgedata, columns=['id', 'edges'])
    print(result.size)
    print(result.head())

    with bz2.BZ2File('../data/picl_with_edge_data' + '.pbz2', 'w') as f:
        cPickle.dump(result, f)
