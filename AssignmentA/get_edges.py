from requests_html import HTMLSession
import re
import pandas as pd
import csv
df = pd.read_csv('../data/metadata.csv', sep=";", encoding="latin1", header=0, usecols=[0,27])

urls = []
for x in df.EliUrl.values:
    urls.append(f'{x}')

results = []
pattern = '/eli/.*'
def work(url):
    r = s.get(url)
    r.html.render(keep_page=True, timeout=10, sleep=10)
    res = [edge for edge in r.html.links if re.match(pattern, edge)]
    for edge in res:
        if len(edge) < 30:
            results.append(edge)
    return





def main(urls):
    for url in urls:
        work(url)
    return

s = HTMLSession()

main(urls)
#main(urls[339:])
with open('../data/edges.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(results)



# create an HTML Session object
#session = HTMLSession()

# Use the object above to connect to needed webpage
#resp = session.get('https://www.retsinformation.dk/eli/lta/2021/2616')

# Run JavaScript code on webpage
#resp.html.render(keep_page=True, timeout=10, sleep=10)
#resp.html.render(keep_page=True)







#dl = pd.DataFrame()
# for url in df.EliUrl[:10].values:
#     create an HTML Session object
#     session = HTMLSession()
#     Use the object above to connect to needed webpage
    # resp = session.get(url)
    # time.sleep(20.0)
    #
    # Run JavaScript code on webpage
    # resp.html.render(retries=20, timeout=20)
    # test = [*resp.html.links]
    # dl = pd.concat([dl, pd.DataFrame(test)])