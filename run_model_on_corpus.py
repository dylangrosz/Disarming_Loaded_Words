from w2v_test import *
from pprint import pprint
import stanford_parser2.src.stanford_parser.parser as parser

# parser.startJvm()
parser = parser.Parser()
# read in corpus and split into list, remove stop words
corpus = ["I", "have", "an", "idea", "that", "this", "model", "should", "work", "submissively"]
dependencies = parser.parseToStanfordDependencies("She had a nurturing presence")
tupleResult = [(rel, gov.text, dep.text) for rel, gov, dep in dependencies.dependencies]

def parse_article(article):
    pass
    # return parsed article

def collect_suspect_words(corpus, sentences, thresh = 0):
    fem_bias_words, masc_bias_words = {}, {}
    word_not_in_model = []
    for w in corpus:
        if w in model:
            male_bias = relative_single_gender_score(w, base_m_score, model, comp=1)
            female_bias = relative_single_gender_score(w, base_f_score, model, comp=1)
            isFemale = abs(female_bias) >= male_bias
            if isFemale and abs(female_bias) >= thresh:
                fem_bias_words[w] = abs(female_bias)
            elif not isFemale and abs(male_bias) >= thresh:
                masc_bias_words[w] = male_bias
        else:
            word_not_in_model.append(w)
    return masc_bias_words, fem_bias_words, word_not_in_model

if __name__ == '__main__':
    print("loading model...")
    model = load_model(limit=1000000)
    print("loaded.")
    base_f_score = get_single_base_score(female_list, model, isFemale=True)
    base_m_score = get_single_base_score(male_list, model, isFemale=False)


    pprint(fem_bias_words)
    pprint(masc_bias_words)
    pprint(word_not_in_model)