from gevent import monkey
monkey.patch_all()

import bz2
import pickle
import random
import _pickle as cPickle
import nest_asyncio
import asyncio
from requests_html import AsyncHTMLSession
import re
import pandas as pd
import json
from more_itertools import ichunked
from contextlib import closing

nest_asyncio.apply()

# Til at se resultat efter:
# df = bz2.BZ2File('..data/picl_with_edge_data.pbz2', 'rb')
df = bz2.BZ2File('../data/picl_law_data.pbz2', 'rb')
df = cPickle.load(df)

columns = ['idx', 'id', 'title', 'documentTypeId', 'shortName', 'full_text', 'isHistorical', 'ressort', "url",
           "caseHistoryReferenceGroup", "stateLabel", "edges",
           "metadata"]  # FJERNET 'Nummer', Add attributes to this list

# Using list to append for optimization note: don't increase dataframes in a loop!
listdata = []
pattern = '/eli/.*'


class Retsinfo:
    def __init__(self, data, batchsize=25, Timeout=10, workers=None, edges=True):
        self._timeout = Timeout
        self._batchsize = batchsize
        self.loop = asyncio.new_event_loop()
        self._workers = workers
        self._df = data
        self._edges = edges
        self._listdata = []
        self.done = 0

    async def main(self):
        iter = self._df.itertuples(index=False)
        batch = ichunked(iter, self._batchsize)
        for _batch in batch:
            for response in await self.run(_batch):
                await self.append(response)

    async def run(self, batch):  # , df, loop=None, stepsize=10, edges=True, workers=None):
        """
        :param df: List of URL's
        :param Batch: Dumb fix for errors caused by potentially hundreds of async render requests (request_html.AsyncHTMLSession)
        :return: List of row, data
        """

        # Initialize the class object event loop
        loop = asyncio.get_running_loop()
        with AsyncHTMLSession(workers=self._workers) as session:
            # Use list comprehension to create a list of
            # tasks to complete. The executor will run the `fetch`
            # function for each url in the urlslist

            tasks = [await loop.run_in_executor(
                session.thread_pool,
                self.fetch,
                *(session, _idx, url)
            )
                     for _idx, url in batch  # For multiple arguments to fetch function
                     ]
            return await asyncio.gather(*tasks)

    def running(self):
        return self.done < self._df.size

    async def get_edges(self, session, id, url):
        url = url.split('api')[0] + url.split('document/')[1]
        resp = await session.get(url, timeout=self._timeout)
        await resp.html.arender(retries=30, wait=random.randint(5, 10), timeout=self._timeout, sleep=8, keep_page=False)
        edges = [{url.split('/eli')[0] + edge: '{placeholder}'} for edge in resp.html.links if
                 (re.match(pattern, edge) and len(edge) < 30)]
        return edges

    async def get_meta(self, session, id, url):
        resp = await session.get(url, timeout=self._timeout)
        document = json.loads(resp.text)  # Source API response
        '''
        Add varaibles below to get more info
        OBS!!! Don't forget to add them to the return of get_meta() (this function), otherwise the append() call won't append them to the final data output
        '''

        unique_identity = document[0]["id"]
        title = document[0]["title"]
        ressort = document[0]["ressort"]
        documentTypeId = document[0]["documentTypeId"]
        shortName = document[0]["shortName"]
        url = resp.html.url
        isHistorical = document[0]["isHistorical"]
        full_text = str(resp.html.full_text)  # document[0]["documentHtml"]
        try:
            caseHistoryReferenceGroup = document[0]["caseHistoryReferenceGroup"][0]['references']
            stateLabel = document[0]["caseHistoryReferenceGroup"][0]['stateLabel']
        except:
            caseHistoryReferenceGroup = None
            stateLabel = None
        metadata = document[0]["metadata"]
        return [
            id,
            unique_identity,
            title,
            documentTypeId,
            shortName,
            full_text,
            isHistorical,
            ressort,
            url,
            caseHistoryReferenceGroup,
            stateLabel,
            metadata]


    async def fetch(self, session, id, url):  # , id, name, url):  #, session, id, url, edges=True):
        # edges, metadata = session.run(self.get_edges(id, url), self.get_meta(id, url))
        edges = await self.get_edges(session, id, url)
        metadata = await self.get_meta(session, id, url)

        # return [id, unique_identity, title, documentTypeId, shortName, full_text, isHistorical, ressort, url,
        #         caseHistoryReferenceGroup, stateLabel, edges, metadata]
        return [*metadata, edges]

    async def display_status(self):
        while self.running():
            await asyncio.sleep(2)
            print('\rdone:', self.done)

    async def append(self, node):
        self._listdata.append(node)
        self.done += 1
        # await asyncio.sleep(0.01)
        # Print the result
        print('\rdone:', self.done)

    def write(self):
        result = pd.DataFrame(data=self._listdata, columns=columns)
        with bz2.BZ2File('../data/picl_with_edge_data' + '.pbz2', 'w') as f:
            cPickle.dump(result, f)
        print('Success')
        print(result.size)
        print(result.head())


if __name__ == '__main__':
    batchsize = 16
    retsinfo = Retsinfo(data=df[['id', 'url']][:1024], batchsize=batchsize, Timeout=80, workers=16, edges=True)
    asyncio.run(retsinfo.main())
    retsinfo.write()

    # with closing(asyncio.get_event_loop()) as loop:
