import argparse

from gevent import monkey
monkey.patch_all()
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
# Til at se resultat efter:
#df = bz2.BZ2File('..data/picl_with_edge_data.pbz2', 'rb')
df = bz2.BZ2File('../data/picl_law_data.pbz2', 'rb')
df = cPickle.load(df)


columns = ['idx', 'id', 'title', 'documentTypeId', 'shortName', 'full_text', 'isHistorical', 'ressort', "url", "caseHistoryReferenceGroup", "stateLabel", "edges", "metadata"] # FJERNET 'Nummer', Add attributes to this list

# Using list to append for optimization note: don't increase dataframes in a loop!
listdata = []
pattern = '/eli/.*'

class Retsinfo:
    def __init__(self, session, data, interval, workers=None, stepsize=None, edges=True):
        self._session = session
        # self._loop = loop
        self._workers = workers
        self._df = data
        self._interval = interval
        self._stepsize = stepsize
        self._edges = edges
        self._listdata = []
        self.done = 0

    async def run(self):  #, df, loop=None, stepsize=10, edges=True, workers=None):
        """
        :param df: List of URL's
        :param step: Dumb fix for errors caused by potentially hundreds of async render requests (request_html.AsyncHTMLSession)
        :return: List of row, data
        """
        #with AsyncHTMLSession(loop=self._loop, workers=self._workers) as session:
        # Dirty batch split for smaller async sessions
        # for begin in range(self._interval[0], self._interval[1], self._stepsize):
        #     stop = begin + self._stepsize
        tasks = [await self._session.loop.run_in_executor(
            self._session.thread_pool,
            self.fetch,
            *(id, url)
        )
                 for _, id, url in islice(self._df.itertuples(), interval[0], interval[1])
                 ]
        await asyncio.gather(*tasks)

        # for response in await asyncio.gather(*tasks):
        #tasks = [self.fetch(id, name, url) for id, name, url in self._df.itertuples()]
        #await asyncio.wait(self.fetch(id, name, url))

    def running(self):
        return self.done < self._interval[1]

    async def fetch(self, id, url): #, id, name, url):  #, session, id, url, edges=True):
        """
        :param session: AsyncHTMLSession
        :param url: Self-explanatory, defined in main() 'tasks' list and called by await.gather(*tasks)
        :param id: Document number
        :return: The variables to append to the output csv file

        Using an async fetcher to handle asynchronous request calls
        Note that this function utilizes two different .get() function (requsts_html.AsyncHTMLSession and requests.get()) \
        the later which talks directly to source provided API and the former which renders the browser destined Javascript \
        and collects edges (references to other documents)
        """
        if self._edges:
            urlR = url.split('api')[0] + url.split('document/')[1]
            resp = await self._session.get(urlR, timeout=120)
            while (resp.status_code != 200):
                resp = await self._session.get(url, timeout=120)
            await resp.html.arender(retries=10, wait=10, timeout=10, sleep=5, keep_page=True)
            edges = [{url.split('/eli')[0] + edge: '{placeholder}'} for edge in resp.html.links if
                     (re.match(pattern, edge) and len(edge) < 30)]
            # [{url.split('document/')[1]+edge:shortName} for edge, shortName in zip(resp.html.links, resp.html.text.find("MISSING IMPLEMEN") if (re.match(pattern, edge) and len(edge) < 30)]
        else:
            self._edges = None

        resp = await self._session.get(url, timeout=120)
        while (resp.status_code != 200):
            resp = await self._session.get(url, timeout=120)
        document = json.loads(resp.text)  # Source API response
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
        full_text = str(resp.html.full_text)  # document[0]["documentHtml"]
        try:
            caseHistoryReferenceGroup = document[0]["caseHistoryReferenceGroup"][0]['references']
            stateLabel = document[0]["caseHistoryReferenceGroup"][0]['stateLabel']
        except:
            caseHistoryReferenceGroup = None
            stateLabel = None
        metadata = document[0]["metadata"]

        node = [id, unique_identity, title, documentTypeId, shortName, full_text, isHistorical, ressort, url, caseHistoryReferenceGroup, stateLabel, edges, metadata]
        self._listdata.append(node)
        self.done += 1
        #self.append([_id, unique_identity, title, documentTypeId, shortName, full_text, isHistorical, ressort, url, caseHistoryReferenceGroup, stateLabel, edges, metadata])


    async def display_status(self):
        while self.running():
            await asyncio.sleep(2)
            print('\rdone:', self.done)

    # def append(self, node):
    #     self._listdata.append(node)
    #     self.done += 1

    def write(self):
        result =  pd.DataFrame(data=self._listdata, columns=columns)
        with bz2.BZ2File('../data/picl_with_edge_data' + '.pbz2', 'w') as f:
            cPickle.dump(result, f)
        print('Success')
        print(result.size)
        print(result.head())

if __name__ == '__main__':
    interval = (0, 8)
    stepsize = 2
    # with closing(asyncio.get_event_loop()) as loop:
    with AsyncHTMLSession(workers=None) as session:
        retsinfo = Retsinfo(session, data=df[['id', 'url']], interval=interval, stepsize=stepsize, edges=True)
        session.loop.run_until_complete(asyncio.wait([
            retsinfo.run(),
            retsinfo.display_status()
            ]))
        retsinfo.write()