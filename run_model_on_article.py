from w2v_utils import *
import sys, os
from pprint import pprint
import nltk.data
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.tag import StanfordNERTagger
#import stanford_parser.src.stanford_parser.parser as nlpparser --dependency issue with java

def parse_article(article_fn):
    sentences, filtered_words = [], []
    with open(article_fn, 'r', encoding="utf8") as file:
        data = file.read().replace('\n', ' ')
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences.extend(tokenizer.tokenize(data))
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(data)
        filtered_words.extend([w for w in word_tokens if not w in stop_words])
    return filtered_words, sentences

def collect_suspect_words(corpus, sentences, base_f_score, base_m_score, thresh = 0.5):
    fem_bias_words, masc_bias_words = {}, {}
    suspect_to_sentence = {}
    word_not_in_model = []
    sentence_cnt = 0
    word_cnt = 0
    # handle quotes
    for w in corpus:
        if w in model:
            male_bias = relative_single_gender_score(w, base_m_score, model, comp=1)
            female_bias = relative_single_gender_score(w, base_f_score, model, comp=1)
            male_bias_adj = pos_score_adjust(w, sentences[sentence_cnt], abs(male_bias))
            fem_bias_adj = pos_score_adjust(w, sentences[sentence_cnt], abs(female_bias))
            isFemale = abs(fem_bias_adj) >= male_bias_adj
            if isFemale and abs(fem_bias_adj) >= thresh:
                fem_bias_words[w] = fem_bias_adj
                suspect_to_sentence[(w, 'F', sentences[sentence_cnt])] = abs(fem_bias_adj)
            elif not isFemale and abs(male_bias_adj) >= thresh:
                masc_bias_words[w] = male_bias_adj
                suspect_to_sentence[(w, 'M', sentences[sentence_cnt])] = male_bias_adj
        elif len(w) == 1 and w in '.?!':
            sentence_cnt += 1
        else:
            word_not_in_model.append(w)

    return suspect_to_sentence, word_not_in_model

def fetch_pos(word, sentence):
    sentence_tagged = nltk.pos_tag(word_tokenize(sentence))
    pos_token = None
    for w, t in sentence_tagged:
        if word == w:
            pos_token = t
            break
    return pos_token

def pos_score_adjust(word, sentence, bias_score):
    noun_weight = 0.6
    pos_deweights = {
        "NN": noun_weight,
        "NNP": noun_weight,
        "NNS": noun_weight,
        "PRP": 0,
        "PRP$": 0,
        "WDT": 0,
        "WP": 0,
        "WRB": 0,
        "TO": 0
    }
    print(fetch_pos(word, sentence))
    return bias_score * pos_deweights.get(fetch_pos(word, sentence), 1)


def pos_cleave(suspect_to_sentence):
    #parser = nlpparser.Parser()
    suspect_to_sentence_cleaved = {}
    for k, score in suspect_to_sentence.items():
        suspect, gender_id, sentence = k
#         dependencies = parser.parseToStanfordDependencies(sentence)
#         dependency_list = [(rel, gov.text, dep.text) for rel, gov, dep in dependencies.dependencies]
#         print(suspect + ": " + str(pos_tag([suspect])))
#         print(sentence)
#         new_dependency_list = []
#         for dep in dependency_list:
#             pos, modder, modded = dep
#             if suspect == modder:
#                 new_dependency_list.append(dep)
#                 if pos == 'nsubj':
#                     base_score = base_f_score if gender_id == 'F' else base_m_score
#                     print(modded)
# #                    if relative_single_gender_score(modded, base_score, model) > 0.7:
#                     # TODO implement gender check, validate_modded_gender(modded, gender_id)
#                     suspect_to_sentence_cleaved[(suspect, gender_id, modded, sentence)] = score
#                 # TODO take care of more complex cases
#         pprint(dependency_list)
    return suspect_to_sentence_cleaved

if __name__ == '__main__':
    article_fn = sys.argv[1]
    print("loading model...")
    model = load_model(limit=1000000)
    print("loaded.")
    base_f_score = get_single_base_score(female_list, model, isFemale=True)
    base_m_score = get_single_base_score(male_list, model, isFemale=False)

    # Set environmental variables programmatically.
    # Set the classpath to the path where the jar file is located
    os.environ['CLASSPATH'] = "C:/Users/dylan/PycharmProjects/disarming_loaded_words/stanford-ner-2015-04-20/stanford-ner-3.5.2.jar"

    # Set the Stanford models to the path where the models are stored
    os.environ[
        'STANFORD_MODELS'] = 'C:/Users/dylan/PycharmProjects/disarming_loaded_words/stanford-corenlp-caseless-2015-04-20-models.jar'

    stanford_classifier = "C:/Users/dylan/PycharmProjects/disarming_loaded_words/stanford-ner-2015-04-20/english.all.3class.caseless.distsim.crf.ser.gz" # 'C:/Users/dylan/PycharmProjects/disarming_loaded_words/stanford-corenlp-caseless-2015-04-20-models/edu/stanford/nlp/models/ner/english.all.3class.caseless.distsim.crf.ser.gz'
    ner_fp = "C:/Users/dylan/PycharmProjects/disarming_loaded_words/stanford-ner-2015-04-20/stanford-ner-3.5.2.jar"
    # # Build NER tagger object
    st = StanfordNERTagger(stanford_classifier)
    #
    # # A sample text for NER tagging
    text = 'srinivas ramanujan went to the united kingdom. There he studied at cambridge university.'
    #
    # # Tag the sentence and print output
    tagged = st.tag(str(text).split())
    print(tagged)

    filtered_words, sentences = parse_article(article_fn)
    print(filtered_words)
    suspect_to_sentence, word_not_in_model = collect_suspect_words(filtered_words, sentences,
                                                                base_f_score, base_m_score, thresh=0)
    pprint(suspect_to_sentence)
    cleaved_suspects = pos_cleave(suspect_to_sentence)
    pprint(cleaved_suspects)
    # print(word_not_in_model)