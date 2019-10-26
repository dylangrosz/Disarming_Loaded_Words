from w2v_test import *
from pprint import pprint
corpus = [] # read in corpus and split into list, remove stop words

fem_bias_words, masc_bias_words = {}, {}

model = load_model(limit=1000000)
base_f_score = get_single_base_score(female_list, isFemale=True)
base_m_score = get_single_base_score(male_list, isFemale=False)
word_not_in_model = []
thresh = 0.5

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

pprint(fem_bias_words)
pprint(masc_bias_words)