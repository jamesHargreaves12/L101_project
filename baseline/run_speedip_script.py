

from tqdm import tqdm

from baseline.get_dataset import get_fever_dataset
from baseline.mediawiki_api import batch_get_pageview_data
from baseline.results import get_possible_docs
from baseline.utils import extract_titles_from_evidence

# data = get_fever_dataset()
# for c, evidence in tqdm(data[1000:]):
#     titles = extract_titles_from_evidence(evidence)
#     for t in titles:
#         views = get_pageview_data(t)
#         if not views > -1:
#             print("*********", t)

data = get_fever_dataset()
batch_titles = []
for c, evidence in tqdm(data[2000+5500+1400:len(data) // 8]):
    possible_docs = get_possible_docs(c)
    doc_titles = [x["title"] for x in possible_docs]
    batch_titles.extend(doc_titles)
    if len(batch_titles) > 500:
        batch_get_pageview_data(batch_titles)
        batch_titles = []
