from gevent import monkey
monkey.patch_all()

import bz2
import pickle
import _pickle as cPickle
import nest_asyncio
import asyncio
from requests_html import AsyncHTMLSession
#import requests
#import re
import pandas as pd
import csv
#import time
#from timeit import default_timer
import json
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
nest_asyncio.apply()



'''
Load url's produced by get_edges.py
Setting coloumns to produce in the resulting law_data.csv file
'''
nodes = pd.read_table(open('../data/edges.csv', 'r'), sep=",", header=None, dtype=str)
columns = ['id', 'title', 'documentTypeId', 'shortName', 'full_text', 'isHistorical', 'ressort', "url", "caseHistoryReferenceGroup", "stateLabel", "metadata"] # FJERNET 'Nummer', Add attributes to this list

# Using list to append for optimization note: don't increase dataframes in a loop!
listdata = []


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
    resp = await session.get(url, timeout=20)
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
    return unique_identity, title, documentTypeId, shortName, full_text, isHistorical, ressort, url, caseHistoryReferenceGroup, stateLabel, metadata

async def main(df, loop=None, step=10):
    """
    :param df: List of URL's
    :param step: Dumb fix for errors caused by potentially hundreds of async render requests (request_html.AsyncHTMLSession)
    :return: List of row, data
    """
    await AsyncHTMLSession().close()
    with ThreadPoolExecutor(max_workers=16) as executor:
        with AsyncHTMLSession(workers=32) as session:
            # Dirty batch split for smaller async sessions
            for begin in range(0, df.size - 1, step):
                stop = begin + step
                # Initialize the event loop
                loop = asyncio.get_event_loop()

                # Use list comprehension to create a list of
                # tasks to complete. The executor will run the `fetch`
                # function for each url in the urlslist
                tasks = [await loop.run_in_executor(
                        executor,
                        fetch,
                        *(session, url)
                    )
                         for url in islice(df.itertuples(), begin, stop)  # For multiple arguments to fetch function
                         ]
                for response in await asyncio.gather(*tasks):
                    listdata.append(response)
                print(f'Done:{begin}')
    return


if __name__ == '__main__':
    step = 100 # Batch size, Higher is faster but more prone to timeouts and connections errors
    #loop = asyncio.get_event_loop()
    #future = asyncio.ensure_future(main(nodes.T, stop=50))
    #loop.run_until_complete(future)
    asyncio.run(main(nodes.T, step=step))
    result = pd.DataFrame(data=listdata, columns=columns)

    with bz2.BZ2File('../data/picl_law_data' + '.pbz2', 'w') as f:
        cPickle.dump(result, f)

#result = pd.DataFrame(data=listdata, columns=columns)