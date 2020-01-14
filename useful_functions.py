import unicodedata
from collections import defaultdict

from DrQA.drqa.tokenizers.corenlp_tokenizer import CoreNLPTokenizer

tokenizer = CoreNLPTokenizer(**{"classpath":
                                    "./DrQA/data/corenlp/*"})


def tokenize(query):
    unicodedata.normalize('NFD', query)
    return tokenizer.tokenize(query)


def get_inverse_term_to_doc(docs):
    idict = defaultdict(set)
    for doc in docs:
        toks = tokenize(doc["sentences"]).data
        for term in toks:
            idict[hash(term[0])].add(doc["index"])
    return idict


docs = [
    {'index': 0, 'sentences': "The inverse document frequency is a measure of how much information the word provides"},
    {'index': 1,
     'sentences': "GloVe is an unsupervised learning algorithm for obtaining vector representations for words. "
                  "Training is performed on aggregated global word-word co-occurrence statistics from a corpus, "
                  "and the resulting representations showcase interesting linear substructures of the word vector "
                  "space."}]

tokens = tokenize("The inverse document frequency is a measure of how much information the word provides")
print(tokens.data[:3])

idict = get_inverse_term_to_doc(docs)
x = 1