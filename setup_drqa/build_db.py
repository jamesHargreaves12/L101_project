from DrQA.scripts.retriever.build_db import store_contents
import os

os.remove("./database/data_sent.db")
store_contents("./data/first_sentences/", "./database/data_sent.db", None)
