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
    display_name: comsci-venv
    language: python
    name: comsci-venv
---

Metadata.csv is a csv manually downloaded from the advanced search on https://www.retsinformation.dk. This can be automated if we want to

```python
import pandas as pd
import requests
from requests_html import HTMLSession
```

```python
def work(url):
    r = s.get(url)
    r.html.render(keep_page=True, timeout=10, sleep=10)
    #for edge in res:
    pattern = '/eli/.*'
    return [edge for edge in r.html.links if re.match(pattern, edge)]
```

```python
metadata = pd.read_csv('../data/metadata.csv', sep=";", encoding="latin1", header=0)

law_data = pd.DataFrame(columns=['id', 'year', 'number', 'text_content', 'edges', 'isHistorical'])
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
        isHistorical = document[0]["documentHtml"]
        df2 = pd.DataFrame([[law_id, year, number, document_text, edges, isHistorical]], columns=['id', 'year', 'number', 'text_content', 'edges'])
        law_data = pd.concat([law_data, df2])
    except:
        pass

```

```python
law_data.to_csv("../data/law_data.csv", index=False)
```

```python
test = pd.read_csv('../data/law_data.csv', sep=",", encoding='utf8')

print(test.iat[1,2])
```
