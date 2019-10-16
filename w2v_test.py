import gensim
import gensim.downloader as api
import pprint as pp
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from literal_tool import feminine_coded_words, masculine_coded_words
import seaborn as sns

model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, limit=500000)
#model = api.load('word2vec-google-news-300')

female_list = ["she", "her", "woman", "Mary", "herself", "daughter", "mother", "gal", "girl", "female"]
male_list = ["he", "his", "man", "John", "himself", "son", "father", "guy", "boy", "male"]

feminine_coded_words_unstemmed = [
    "agree",
    "nurse",
    "affectionate",
    "bossy",
    "child",
    "cheer",
    "collaborative",
    "committed",
    "communal",
    "compassionate",
    "connected",
    "considerate",
    "cooperative",
    "co-operative",
    "dependable",
    "emotional",
    "empathetic",
    "feel",
    "flatterable",
    "gentle",
    "honest",
    "interpersonal",
    "interdependent",
    "interpersonal",
    "inter-personal",
    "inter-dependent",
    "inter-personal",
    "kind",
    "kinship",
    "loyal",
    "modesty",
    "nag",
    "nurturing",
    "pleasant",
    "polite",
    "quiet",
    "responsive",
    "sassy",
    "sensitive",
    "submissive",
    "support",
    "sympathetic",
    "tender",
    "together",
    "trusting",
    "understand",
    "warm",
    "whiny",
    "enthusiastic",
    "inclusive",
    "yield",
    "share",
    "sharing"
]

masculine_coded_words_unstemmed = [
    "active",
    "doctor",
    "adventurous",
    "aggressive",
    "ambitious",
    "analytical",
    "assertive",
    "athletic",
    "autonomous",
    "battle",
    "boasting",
    "challenging",
    "champion",
    "competitive",
    "confident",
    "courageous",
    "decide",
    "decision",
    "decisive",
    "defend",
    "determined",
    "dominant",
    "driven",
    "fearless",
    "fight",
    "force",
    "greedy",
    "head-strong",
    "headstrong",
    "hierarchy",
    "hostile",
    "impulsive",
    "independent",
    "individualistic",
    "intellectual",
    "lead",
    "logical",
    "objective",
    "opinion",
    "outspoken",
    "persistent",
    "principle",
    "reckless",
    "self-confident",
    "self-reliant",
    "self-sufficient",
    "selfconfident",
    "selfreliant",
    "selfsufficient",
    "stubborn",
    "superior",
    "unreasonable"
]


def unit_vec(vec):
    return vec / (vec ** 2).sum() ** 0.5

def build_gender_subspace(model):
    gender_subspace = np.zeros((model.wv['news'].shape[0],0))
    for v_f, v_m in zip(female_list, male_list):
        diff = np.subtract(model.wv[v_m], model.wv[v_f])
        diff = np.expand_dims(diff, axis=1)
        gender_subspace = np.concatenate((gender_subspace, diff), axis=1)
    return gender_subspace

def build_gender_direction(model, comp = 1):
    g_s = build_gender_subspace(model)
    pca = PCA(n_components = comp)
    pcs = pca.fit_transform(g_s).squeeze()
    #print(pca.explained_variance_ratio_)
    return pcs, pca.explained_variance_ratio_


def absolute_gender_score(word_a, word_b, model, delta = 1, comp = 1):
    v_a, v_b = model.wv[word_a], model.wv[word_b]
    g_d, weights = build_gender_direction(model, comp = comp)
    diff = np.subtract(v_a, v_b)
    norm_d = np.linalg.norm(diff)
    #print("Norm: {}".format(norm_d))
    if True or norm_d <= delta:
        return gender_proj(diff, g_d, model, weights = weights)
    else:
        return 0 # are not semantically similar relative to delta

def gender_proj(word, g_d, model, weights = None):
    v_r = np.zeros((g_d.shape[0], 1))
    max = g_d.shape[1] if len(g_d.shape) > 1 else 1
    weights_n = weights_n = weights / np.linalg.norm(weights, 1) if weights is not None else None
    proj_factor_sum = 0
    for i in range(max):
        v = model.wv[word] if isinstance(word, str) else word
        g_d_i = g_d[:, i] if len(g_d.shape) > 1 else g_d
        proj_factor = (np.dot(v, g_d_i) / np.dot(g_d_i, g_d_i))
        if weights is not None:
            proj_factor_sum += weights_n[i] * abs(proj_factor)
        v_r += np.expand_dims(proj_factor * g_d_i, axis=1)
    return v_r, proj_factor_sum

def relative_gender_score(word_a, word_b, base_score, model, delta = 1, comp = 1):
    _, score = absolute_gender_score(word_a, word_b, model, delta = delta, comp = comp)
    return score / base_score

if __name__ == '__main__':
    g_d = build_gender_direction(model, comp = 1)
    p1, p2 = "father", "doctor"
    n1 = "mother"
    print("{} is to {} as {} is to: ".format(p1, p2, n1))
    #pp.pprint(model.most_similar(positive=[p1, n1], negative=[p2]))

    base_scores = []
    for v_f, v_m in zip(female_list, male_list):
        base_scores.append(absolute_gender_score(v_m, v_f, model, comp = 1)[1])
    print(male_list)
    print(female_list)
    print(base_scores)
    base_score = max(base_scores)

    word_m_bias, word_f_bias = "doctor", "nurse"
    score = absolute_gender_score(word_m_bias, word_f_bias, model, comp = 1)[1]
    print("Relative gender score of {} vs. {}: {} / {} = {}".format(word_m_bias, word_f_bias,
                                                                    score, base_score,
                                                                    relative_gender_score(word_m_bias, word_f_bias, base_score, model, comp = 1)))
    w_concs = []
    scores_rel = []
    for w_f in feminine_coded_words_unstemmed:
        for w_m in masculine_coded_words_unstemmed:
            if w_f in model and w_m in model:
                scores_rel.append(("{} vs. {}".format(w_f, w_m),
                                   relative_gender_score(w_m, w_f, base_score, model, comp = 4)))
    scores_rel.sort(reverse=True, key=lambda x: x[1])
    w_concs = [w for w, s in scores_rel]
    scores_rel_s = [s for w, s in scores_rel]
    pp.pprint(scores_rel[:5])
    TOP_N = 25
    plt.figure(figsize=(8, 4))
    bp = sns.barplot(w_concs[:TOP_N], scores_rel_s[:TOP_N])
    bp.set_xticklabels(bp.get_xticklabels(), rotation=45, ha='right')
    bp.set(ylabel='Proximity to Gender Bias', xlabel='F vs. M Word Pairs')
    plt.tight_layout()
    bp.get_figure().savefig('most_biased.png')
    plt.show()


