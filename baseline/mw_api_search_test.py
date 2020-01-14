import mwclient
from more_itertools import take

site = mwclient.Site('en.wikipedia.org')
meta = []
for i,result in enumerate(take(10, site.search("Albert Einstein", limit=10))):
    meta.append({
        'word_count_doc': result['wordcount'],
        'size_doc': result["size"],
        'title': result['title'].replace('\'', '_-apos-_').replace("\"", "_-dq-_"),
        'relevance_score': i
    })
print(meta)