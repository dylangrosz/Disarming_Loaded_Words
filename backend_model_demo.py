import gensim
import gensim.downloader as api
import pprint as pp
import numpy as np
import sys
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from literal_tool import feminine_coded_words, masculine_coded_words
import seaborn as sns
 
def load_model(limit=1000000):
    return gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, limit=limit)

def unit_vec(vec):
    return vec / (vec ** 2).sum() ** 0.5

def build_gender_subspace(model):
    gender_subspace = np.zeros((model.wv['news'].shape[0], 0))
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

def build_single_gender_directions(model, comp = 1):
    g_m, g_f = build_male_subspace(model), build_female_subspace(model)
    pca_m, pca_f = PCA(n_components = comp), PCA(n_components = comp)
    pcs_m, pcs_f = pca_m.fit_transform(g_m).squeeze(), \
                   pca_f.fit_transform(g_f).squeeze()
    # print(pca_m.explained_variance_ratio_)
    # print(pca_f.explained_variance_ratio_)
    return pcs_m, pca_m.explained_variance_ratio_, \
           pcs_f, pca_f.explained_variance_ratio_

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

def build_male_subspace(model):
    male_subspace = np.zeros((model.wv['news'].shape[0],0))
    for w_m in male_list:
        v_m = np.expand_dims(model.wv[w_m], axis=1)
        male_subspace = np.concatenate((male_subspace, v_m), axis=1)
    return male_subspace

def build_female_subspace(model):
    female_subspace = np.zeros((model.wv['news'].shape[0],0))
    for w_f in female_list:
        v_f = np.expand_dims(model.wv[w_f], axis=1)
        female_subspace = np.concatenate((female_subspace, v_f), axis=1)
    return female_subspace

def absolute_single_gender_score(word, model, delta = 1, comp = 1):
    v = model.wv[word]
    g_m, weights_m, g_f, weights_f = build_single_gender_directions(model, comp = comp)
    norm = np.linalg.norm(v)
    #print("Norm: {}".format(norm_d))
    if True or norm <= delta:
        gp_m, gp_factor_m = gender_proj(v, g_m, model, weights = weights_m)
        gp_f, gp_factor_f = gender_proj(v, g_f, model, weights = weights_m)
        return (gp_m, gp_factor_m) if gp_factor_m > gp_factor_f else (gp_f, -1 * gp_factor_f)
    else:
        return 0 # are not semantically similar relative to delta

def relative_single_gender_score(word, base_score, model, delta = 1, comp = 1):
    _, score = absolute_single_gender_score(word, model, delta = delta, comp = comp)
    return score / base_score if score > 0 else -1 * (score / base_score)

def get_base_score(male_list, female_list):
    base_scores = []
    for v_f, v_m in zip(female_list, male_list):
        base_scores.append(absolute_gender_score(v_m, v_f, model, comp = 1)[1])
    return max(base_scores)

def get_single_base_score(gender_list, model, isFemale=True):
    scores = []
    for w in gender_list:
        scores.append(absolute_single_gender_score(w, model, comp=1)[1])
    # print(isFemale)
    # print(scores)
    return min(scores) if isFemale else max(scores)

def plot_single_bias(model, gender_list, coded_words_unstemmed,
                     isFemale=True, TOP_N = 25, comp = 1, fn_name="bias.png"):
    scores_rel = []
    base_score = get_single_base_score(gender_list, model, isFemale=isFemale)
    for w in coded_words_unstemmed:
        if w in model:
            scores_rel.append((w,
                               relative_single_gender_score(w, base_score, model, comp = comp)))
    scores_rel.sort(reverse=(not isFemale), key=lambda x: x[1])
    w_concs = [w for w, s in scores_rel]
    scores_rel_s = [s for w, s in scores_rel]
    plt.figure(figsize=(14, 7))
    bp = sns.barplot(w_concs[:TOP_N], scores_rel_s[:TOP_N])
    bp.set_xticklabels(bp.get_xticklabels(), rotation=45, ha='right')
    ylabel_name = 'Projection Proximity to \'Femininity\'' if isFemale else 'Projection Proximity to \'Masculinity\''
    bp.set(ylabel=ylabel_name, xlabel='Words')
    plt.tight_layout()
    bp.get_figure().savefig(fn_name)
    plt.show()

def plot_pair_bias(model, female_list, male_list,
                   feminine_coded_words_unstemmed, masculine_coded_words_unstemmed,
                   TOP_N=25, comp = 1, fn_name="pair_bias.png"):
    scores_rel = []
    base_score = get_base_score(male_list, female_list)
    for w_f in feminine_coded_words_unstemmed:
        for w_m in masculine_coded_words_unstemmed:
            if w_f in model and w_m in model:
                scores_rel.append(("{} vs. {}".format(w_f, w_m),
                                   relative_gender_score(w_m, w_f, base_score, model, comp = comp)))
    scores_rel.sort(reverse=True, key=lambda x: x[1])
    w_concs = [w for w, s in scores_rel]
    scores_rel_s = [s for w, s in scores_rel]
    #plt.figure(figsize=(14, 7))
    #bp = sns.barplot(w_concs[:TOP_N], scores_rel_s[:TOP_N]) ---dependency
    #bp.set_xticklabels(bp.get_xticklabels(), rotation=45, ha='right')
    #bp.set(ylabel='Projection Proximity to Gender Rift', xlabel='F vs. M Word Pairs')
    #plt.tight_layout()
    #bp.get_figure().savefig(fn_name)
    #plt.show()


# Used to construct gender directions / subspaces
female_list = ["she", "her", "woman", "Mary", "herself", "daughter", "mother", "gal", "girl", "female"]
male_list = ["he", "his", "man", "John", "himself", "son", "father", "guy", "boy", "male"]

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
    return bias_score * pos_deweights.get(fetch_pos(word, sentence), 1)


if __name__ == '__main__':
    feminine_coded_words_unstemmed = sys.argv[1:]
    print("Scoring the following words based on female gender-bias score: " + str(feminine_coded_words_unstemmed))
    print("building model...")
    model = load_model(limit=500000)
    print("building gender direction...")
    g_d = build_gender_direction(model, comp=1)

    base_f_score = get_single_base_score(female_list, model, isFemale=True)
    base_m_score = get_single_base_score(male_list, model, isFemale=False)

    print("Visualization being produced at: feminine_bias_demo.png")
    plot_single_bias(model, female_list, feminine_coded_words_unstemmed, fn_name='feminine_bias_demo.png', isFemale=True)

