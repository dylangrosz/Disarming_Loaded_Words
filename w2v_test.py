import gensim
import gensim.downloader as api
import pprint as pp
import numpy as np
from sklearn.decomposition import PCA

model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, limit=500000)
#model = api.load('word2vec-google-news-300')
p1, p2 = "father", "doctor"
n1 = "mother"

def unit_vec(vec):
    return vec / (vec ** 2).sum() ** 0.5

def build_gender_subspace(model):
    female_list = ["she", "her", "woman", "Mary", "herself", "daughter", "mother", "gal", "girl", "female"]
    male_list = ["he", "his", "man", "John", "himself", "son", "father", "guy", "boy", "male"]
    gender_subspace = np.zeros((model.wv['bed'].shape[0],0))
    for v_f, v_m in zip(female_list, male_list):
        diff = np.subtract(model.wv[v_f], model.wv[v_m])
        diff = np.expand_dims(diff, axis=1)
        gender_subspace = np.concatenate((gender_subspace, diff), axis=1)
    return gender_subspace

def build_gender_direction(g_s):
    pca = PCA(n_components = 10)
    pcs = pca.fit_transform(g_s).squeeze()
    print(pca.explained_variance_ratio_)
    return unit_vec(pcs)


def direct_bias(word, g_d, model):
    pass

def gender_proj(word, g_d, model):
    v = unit_vec(model.wv[word])
    v_he, v_she = unit_vec(model.wv['he']), unit_vec(model.wv['she'])
    diff = unit_vec(np.subtract(v_he, v_she))
    proj_factor = (np.dot(v, diff) / np.dot(diff, diff))
    print(proj_factor)
    return proj_factor * diff

g_s = build_gender_subspace(model)
g_d = build_gender_direction(g_s)
print("{} is to {} as {} is to: ".format(p1,p2,n1))
pp.pprint(model.most_similar(positive=[p1, n1], negative=[p2]))

print(gender_proj('lashing', g_d, model))