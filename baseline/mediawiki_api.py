from collections import defaultdict
from time import sleep

import mwclient
from flask import json
from more_itertools import take
from mwviews.api import PageviewsClient
import unidecode

from baseline import cache_builder

p = PageviewsClient('jh2045@cam.ac.uk')
fp_pageviews = open('caches/_get_pageview_data.txt', "a+")
site = mwclient.Site('en.wikipedia.org')


def fix_title(t):
    replacements = {
        "-LRB-": "(",
        "-RRB-": ")",
        "-COLON-": ":",
        "½": "_1/2",
        "χ": "X"
    }
    for k, v in replacements.items():
        t = t.replace(k, v)
    return unidecode.unidecode(t)


def _get_pageview_data(title):
    title = fix_title(title)
    try:
        months = p.article_views('en.wikipedia', [title], start="20180101", end="20181231", granularity="monthly")
        total = 0
        for m in months.values():
            for k, v in m.items():
                assert k == title
                if v is not None:
                    total += v
        return total
    except Exception as e:
        print("_get_pageview_data", title)
        print(e)
        return -1


def batch_get_pageview_data(titles):
    global p
    titles = [fix_title(t) for t in titles]
    months = None
    while months is None:
        try:
            months = p.article_views('en.wikipedia', titles, start="20180101", end="20181231", granularity="monthly")
        except:
            sleep(20)
            p = PageviewsClient('jh2045@cam.ac.uk')
            months = None
    totals = defaultdict(int)
    for m in months.values():
        for k, v in m.items():
            if v is not None:
                totals[k] += v
    for title, total in totals.items():
        fp_pageviews.write('{},{}\n'.format(title, total))
    fp_pageviews.flush()


def _get_doc_meta_for_query(query):
    if len(query) > 300:
        query = query[:300]
        final_space = query.rfind(' ')
        if final_space > 10:  # the 10 here is arbitrarily chosen
            query = query[:final_space]
    # query = query.replace('#', '%23')
    meta = []
    for i, result in enumerate(take(10, site.search(query, limit=10))):
        meta.append({
            'word_count_doc': result['wordcount'],
            'size_doc': result["size"],
            'title': result['title'].replace('\'', '_-apos-_').replace("\"", "_-dq-_"),
            'relevance_score': i
        })
    return str(meta)


def get_doc_meta_for_query(query):
    query = query.strip(".").replace(" '", "'")
    str_val = cache_builder.cache_call(_get_doc_meta_for_query, query).replace('\'', '"')
    vals = json.loads(str_val)
    for x in vals:
        x["title"] = x["title"].replace('_-apos-_', "'").replace("_-dq-_", "\"")
    return vals


def get_pageview_data(title):
    str_val = cache_builder.cache_call(_get_pageview_data, title)
    try:
        return int(str_val)
    except:
        print("get_pageview_data",title, print(str_val))


cache_builder.setup_cache(_get_doc_meta_for_query, cache_filename='cache_request')
cache_builder.setup_cache(_get_pageview_data)

if __name__ == "__main__":
    # _get_pageview_data("Albert_Einstein")
    # print(get_doc_meta_for_query("Fox 2000 Pictures released the film Soul Food."))

    from tqdm import tqdm

    from baseline.get_dataset import get_fever_dataset
    from baseline.utils import extract_titles_from_evidence

    # data = get_fever_dataset()
    # for c, evidence in tqdm(data[1000:]):
    #     titles = extract_titles_from_evidence(evidence)
    #     for t in titles:
    #         views = get_pageview_data(t)
    #         if not views > -1:
    #             print("*********", t)

