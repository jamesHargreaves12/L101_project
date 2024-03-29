#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Interactive mode for the tfidf DrQA retriever module."""

import argparse
import code
import prettytable
import logging
from drqa import retriever

# logger = logging.getLogger()
# logger.setLevel(logging.INFO)
# fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
# console = logging.StreamHandler()
# console.setFormatter(fmt)
# logger.addHandler(console)

model = 'DrQA/data/wikipedia/docs-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz'

ranker = retriever.get_class('tfidf')(tfidf_path=model)




def process(query, k=1):
    doc_names, doc_scores = ranker.closest_docs(query, k)
    table = prettytable.PrettyTable(
        ['Rank', 'Doc Id', 'Doc Score']
    )
    for i in range(len(doc_names)):
        table.add_row([i + 1, doc_names[i], '%.5g' % doc_scores[i]])
    return doc_names,doc_scores, table


banner = """
Interactive TF-IDF DrQA Retriever
>> process(question, k=1)
>> usage()
"""


def usage():
    print(banner)


if __name__ == "__main__":
    print("start")
    names, scores, table = process("Does Spain have a royal family?", k=5)
    print(table)
    print(names)
    print(scores)
# code.interact(banner=banner, local=locals())
