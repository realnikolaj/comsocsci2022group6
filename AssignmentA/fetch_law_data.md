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

metadata = pd.read_csv('../data/metadata.csv', sep=",", encoding='utf8')


law_data = pd.DataFrame(columns=['year', 'number', 'content'])

for index, row in metadata.iterrows():
    number  = row['Nummer']
    year = row['År']
    r = requests.get('https://www.retsinformation.dk/api/document/eli/lta/{}/{}'.format(year, number))
    document= r.json()
    try:
        document_text = document[0]["documentHtml"]
        df2 = pd.DataFrame([[year, number, document_text]], columns=['year', 'number', 'content'])
        law_data = pd.concat([law_data, df2])
    except:
        pass

```

```python
law_data.to_csv("../data/law_data.csv", index=False)
```

```python
test = pd.read_csv('../data/law_data.csv', sep=",", encoding='utf8')


```
