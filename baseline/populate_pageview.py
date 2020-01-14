import csv
import json

from tqdm import tqdm

from baseline.cache_builder import _undo_safe
from baseline.mediawiki_api import batch_get_pageview_data

fp = open('caches/cache_request.txt', 'r')
all_titles = set()
for line in csv.reader(fp):
    list_docs = eval(_undo_safe(line[1]))
    titles = [x["title"].replace('_-apos-_', "'").replace("_-dq-_", "\"") for x in list_docs]
    all_titles = all_titles.union(titles)

batch_titles = []
for t in tqdm(list(all_titles)[4004+20000+19000:]):
    batch_titles.append(t[:])
    if len(batch_titles) > 300:
        batch_get_pageview_data(batch_titles)
        batch_titles = []