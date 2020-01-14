import json
from time import time
from baseline import cache_builder
from baseline.tree_parse import load_tree

predictor = None


def set_up_predictor():
    global predictor
    from allennlp.models import load_archive
    from allennlp.predictors.predictor import Predictor
    model_archive = load_archive("parsers/elmo-constituency-parser-2018.03.14.tar.gz")
    predictor = Predictor.from_archive(model_archive)


def _get_noun_phrases(sentence):
    global predictor
    if not predictor:
        print("Predictor not setup, this will take about 20secs")
        set_up_predictor()
    parse_tree_str = predictor.predict(
        sentence=sentence
    )['trees']
    # print(parse_tree_str)
    parse_tree = load_tree(parse_tree_str)[0]
    nps = parse_tree.get_noun_phrases()
    return [str(np).replace("'", "_-apos-_") for np in nps]


def get_noun_phrases(sentence):
    result = cache_builder.cache_call(_get_noun_phrases, sentence)
    list_val =  eval(result) if type(result) == str else result
    return [x.replace("_-apos-_","'") for x in list_val]


cache_builder.setup_cache(_get_noun_phrases)

if __name__ == "__main__":
    print(get_noun_phrases("Fox 2000 Pictures released the film Soul Food."))
