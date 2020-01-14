import os
# This line must be above retriever import since it requires that the variable be set at run time
os.environ["CLASSPATH"] = "./DrQA/data/corenlp/*"

from drqa import retriever

from baseline.cache_builder import setup_cache, cache_call


ranker = None
ranker_sent = None

def title_to_database_norm(title):
    return title.replace("(", "-LRB-").replace(")","-RRB-").replace(":","-COLON-")


def setup_ranker():
    global ranker
    model = 'models/data-tfidf-ngram=2-hash=16777216-tokenizer=corenlp.npz'
    print("Setting up DrQA doc ranker (~20secs)")
    ranker = retriever.get_class('tfidf')(tfidf_path=model)
    print("DrQA doc ranker set up")


def setup_ranker_sent():
    print("Setting up DrQA sentence ranker (~20secs)")
    global ranker_sent
    model_sent = 'models/data_sent-tfidf-ngram=2-hash=16777216-tokenizer=corenlp_sentence.npz'
    ranker_sent = retriever.get_class('tfidf')(tfidf_path=model_sent)
    print("DrQA sentence ranker set up")


def process_sent(query, k=1):
    if not ranker_sent:
        setup_ranker_sent()
    doc_names, doc_scores = ranker_sent.closest_docs(query, k)
    return doc_names, doc_scores


def process(query, k=1):
    if not ranker:
        setup_ranker()
    doc_names, doc_scores = ranker.closest_docs(query, k)
    return doc_names, doc_scores


def get_titles(statment, k=5):
    titles, _ = process(statment, k)
    return set(titles)


def get_titles_sent(statment, k=5):
    titles, _ = process_sent(statment, k)
    return set(titles)


def _get_drqa_score_doc(query_doc_tuple):
    if not ranker:
        setup_ranker()
    query, doc_name = query_doc_tuple
    return ranker.get_score(query, title_to_database_norm(doc_name))


def _get_drqa_score_sent(query_doc_tuple):
    if not ranker_sent:
        setup_ranker_sent()
    query, doc_name = query_doc_tuple
    return ranker_sent.get_score(query, title_to_database_norm(doc_name))


def get_drqa_score_sent(query, doc_name):
    result = cache_call(_get_drqa_score_sent,(query, doc_name))
    return float(result)


def get_drqa_score_doc(query, doc_name):
    result = cache_call(_get_drqa_score_doc,(query, doc_name))
    return float(result)


setup_cache(_get_drqa_score_doc)
setup_cache(_get_drqa_score_sent)

if __name__ == "__main__":
    print("start")
    query = "Does Spain have a royal family?"
    names, scores = process("Waladu?", k=1)
    docs = ['British_royal_family', 'Royal_Family_-LRB-disambiguation-RRB-', 'Royal_families_of_the_United_Arab_Emirates',
     'Operation_Candid', 'List_of_Danish_royal_residences']
    print(names)
    print(scores)
    print([get_drqa_score_sent(query, x) for x in docs])
