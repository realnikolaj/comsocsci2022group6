import pandas as pd
import requests
from requests_html import HTMLSession
import re

pattern = '/eli/.*'

def work(url):
    r = s.get(url)
    r.html.render(keep_page=True, timeout=10, sleep=10)
    #for edge in res:
    return [edge for edge in r.html.links if re.match(pattern, edge)]

metadata = pd.read_csv('../data/metadata.csv', sep=";", encoding="latin1", header=0)
metadata = metadata.head(4)
law_data = pd.DataFrame(columns=['year', 'number', 'text_content', 'edges'])
s = HTMLSession()

for index, row in metadata.iterrows():
    number  = row['Nummer']
    year = row['År']
    edges = work(row['EliUrl'])
    url = row['EliUrl'].split("dk/")
    law_id = url[1]
    print(edges)
    r = requests.get('https://www.retsinformation.dk/api/document/{}'.format(law_id))
    document= r.json()
    try:
        document_text = document[0]["documentHtml"]
        df2 = pd.DataFrame([[year, number, document_text, edges]], columns=['year', 'number', 'text_content', 'edges'])
        law_data = pd.concat([law_data, df2])
    except:
        pass

law_data.to_csv("../data/law_data.csv", index=False)