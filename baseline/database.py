import html
import sqlite3
from time import time

from baseline.cache_builder import setup_cache, cache_call

conn = sqlite3.connect("./database/data.db")
# accent_to_val = {"acute": "´", "circ": "ˆ", "uml": "¨", "grave": "`", "tilde": "˜", "cedilla": "¸"}
accent_to_val = {"acute": "_", "circ": "_", "uml": "_", "grave": "_", "tilde": "_", "cedilla": "_"}
table = {k: '{}{}'.format(v[0], accent_to_val[v[1:]] if v[1:] in accent_to_val else v[1:]) for k, v in
         html.entities.codepoint2name.items()}


def convert_to_form_in_db(title):
    return title.replace(" ", "_").replace("(", "-LRB-").replace(")", "-RRB-")


def get_docs(id, number=5):
    id = convert_to_form_in_db(id)
    c = conn.cursor()
    res = c.execute('SELECT * FROM documents WHERE id="{}" LIMIT {};'.format(id, number))
    return [x for x in res]


# needed since new pages may have been created since the 2017 dump
def _is_in_database(id):
    id = convert_to_form_in_db(id)
    c = conn.cursor()
    res = c.execute('Select * From documents where id = ? limit 1', (id,))
    if not bool([1 for _ in res]):
        replaced_id = id.translate(table)
        if not replaced_id == id:
            replaced_id = id.replace("_","'\_").translate(table)
            start = time()
            res.execute('Select * From documents where id Like ? ESCAPE \'\\\' limit 1', (replaced_id,))
            print(time() - start)
    return bool([1 for _ in res])


def is_in_database(id):
    return bool(cache_call(_is_in_database,id))


setup_cache(_is_in_database)




# c = conn.cursor()
# res = c.execute('SELECT * FROM documents WHERE id LIKE "?";', ('%\"Adult_Party_Cartoon\"%'))
# x = 1