import pandas as pd
import numpy as np
import re
from nltk.stem import SnowballStemmer, PorterStemmer, LancasterStemmer

snow = SnowballStemmer('english')
lanc = LancasterStemmer()
port = PorterStemmer()



# from

# bernie heart attack -> yes there's been coverage but compared to Hillary's faining spell?
# pull week with Bernie
# nasty vs. forceful | bossy vs. decisive
# detect comments on looks, comments on personal life
# look at verbs | plot verbs on pos vs. neg connotation vs. strong vs. weak verbs
# feedback by readers (possibly at end of article)
# usage for live blogs/tweets

feminine_coded_words = [
    "agree",
    "affectionate",
    "child",
    "cheer",
    "collab",
    "commit",
    "communal",
    "compassion",
    "connect",
    "considerate",
    "cooperat",
    "co-operat",
    "depend",
    "emotiona",
    "empath",
    "feel",
    "flatterable",
    "gentle",
    "honest",
    "interpersonal",
    "interdependen",
    "interpersona",
    "inter-personal",
    "inter-dependen",
    "inter-persona",
    "kind",
    "kinship",
    "loyal",
    "modesty",
    "nag",
    "nurtur",
    "pleasant",
    "polite",
    "quiet",
    "respon",
    "sensitiv",
    "submissive",
    "support",
    "sympath",
    "tender",
    "together",
    "trust",
    "understand",
    "warm",
    "whin",
    "enthusias",
    "inclusive",
    "yield",
    "share",
    "sharin"
]

masculine_coded_words = [
    "active",
    "adventurous",
    "aggress",
    "ambitio",
    "analy",
    "assert",
    "athlet",
    "autonom",
    "battle",
    "boast",
    "challeng",
    "champion",
    "compet",
    "confident",
    "courag",
    "decid",
    "decision",
    "decisive",
    "defend",
    "determin",
    "domina",
    "dominant",
    "driven",
    "fearless",
    "fight",
    "force",
    "greedy",
    "head-strong",
    "headstrong",
    "hierarch",
    "hostil",
    "impulsive",
    "independen",
    "individual",
    "intellect",
    "lead",
    "logic",
    "objective",
    "opinion",
    "outspoken",
    "persist",
    "principle",
    "reckless",
    "self-confiden",
    "self-relian",
    "self-sufficien",
    "selfconfiden",
    "selfrelian",
    "selfsufficien",
    "stubborn",
    "superior",
    "unreasonab"
]

hyphenated_coded_words = [
    "co-operat",
    "inter-personal",
    "inter-dependen",
    "inter-persona",
    "self-confiden",
    "self-relian",
    "self-sufficien"
]

def count_bias(input):
    words = re.split('; |, |\*|\n|\t| ', input)
    f_score, m_score, n_score = 0, 0, 0
    f_list, m_list, n_list = [], [], []
    word_amt = len(words)
    for w_raw in words:
        w_stems = [snow.stem(w_raw), lanc.stem(w_raw), port.stem(w_raw)]
        print(w_stems)
        if any(w_i in feminine_coded_words for w_i in w_stems):
            f_list.append(w_raw)
            f_score += 1
        elif any(w_i in masculine_coded_words for w_i in w_stems):
            m_list.append(w_raw)
            m_score += 1
        elif any(w_i in hyphenated_coded_words for w_i in w_stems):
            n_list.append(w_raw)
            n_score += 1
    if f_score > m_score:
        print("This sentence is subtly coded as feminine")
        words_buf = "This determination is specifically caused by the following words: "
        for i in range(len(f_list) - 1):
            words_buf += str(f_list[i]) + ", "
        words_buf += str(f_list[len(f_list) - 1])
        print(words_buf)

    elif f_score < m_score:
        print("This sentence is subtly coded as masculine")
        words_buf = "This determination is specifically caused by the following words: "
        for i in range(len(m_list) - 1):
            words_buf += str(m_list[i]) + ", "
        words_buf += str(m_list[len(m_list) - 1])
        print(words_buf)
    elif f_score == m_score and f_score > 0:
        print("This sentence has gender bias but is equally subtly coded as both masculine and feminine.")

        m_words_buf = "The masculine-coded terms are specifically caused by the following words: "
        for i in range(len(m_list) - 1):
            m_words_buf += str(m_list[i]) + ", "
        m_words_buf += str(m_list[len(m_list) - 1])

        f_words_buf = "The feminine-coded terms are specifically caused by the following words: "
        for i in range(len(m_list) - 1):
            f_words_buf += str(m_list[i]) + ", "
        f_words_buf += str(m_list[len(m_list) - 1])

        print(m_words_buf)
        print(f_words_buf)
    elif f_score == 0 and m_score == 0:
        print("According to our repository of gender-coded words, this sentence is bias-free!")
while True:
    user_input = input("Write your sentence here (leave blank and press ENTER if you want exit):\n")
    count_bias(user_input)
    if len(user_input) == 0:
        break
    print()