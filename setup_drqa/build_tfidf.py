import math
import os

os.environ["CLASSPATH"] = "./DrQA/data/corenlp/*"
# Require CLASSPATH=./DrQA/data/corenlp/* in runtime

from DrQA.drqa import retriever
from DrQA.scripts.retriever.build_tfidf import get_doc_freqs, get_tfidf_matrix, get_count_matrix

db_path = "./database/data_sent.db"
ngram = 2
hash_size = int(math.pow(2, 24))
tokenizer = "corenlp"
out_dir = "./models/"
num_works = None

args = {
    "db_path": db_path,
    "ngram": ngram,
    "hash_size": hash_size,
    "tokenizer": tokenizer,
    "out_dir": out_dir,
    "num_workers": num_works
}
args = type('obj', (object,), args)

count_matrix, doc_dict = get_count_matrix(
    args, 'sqlite', {'db_path': args.db_path}
)

tfidf = get_tfidf_matrix(count_matrix)

freqs = get_doc_freqs(count_matrix)

basename = os.path.splitext(os.path.basename(args.db_path))[0]
basename += ('-tfidf-ngram=%d-hash=%d-tokenizer=%s_sentence' %
             (args.ngram, args.hash_size, args.tokenizer))
filename = os.path.join(args.out_dir, basename)

metadata = {
    'doc_freqs': freqs,
    'tokenizer': args.tokenizer,
    'hash_size': args.hash_size,
    'ngram': args.ngram,
    'doc_dict': doc_dict
}
retriever.utils.save_sparse_csr(filename, tfidf, metadata)
