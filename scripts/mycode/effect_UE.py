'''
To analyze some factors of UE.
'''
#%%
import pandas as pd
import numpy as np
from scipy import stats
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

root = "/home/liuzhu/WSD/multilabel-wsd/"
#%%
def hyp_test(x1, x2):

    ztest, pval = stats.ttest_ind(x1, x2, equal_var=True)

    return ztest, pval

def splitBy(data, by, level=None, EqualBin=False):
    if by == "pos":
        split = [data[data[by]==p]['UE'].to_numpy() for p in ["NOUN", "VERB", "ADJ", "ADV"]]
        return split
    else:
        if not EqualBin:
            gap = (data[by].max() + 1 - data[by].min()) // level
            split = [data[[data[by].min() + i * gap <= j <= data[by].min() + (i+1) * gap for j in data[by]]]['UE'].to_numpy() for i in range(level)]
        else:
            gap = len(data) // level
            by_set = Counter(data.sort_values(by=by)[by].to_list())
            by_dict = dict(sorted(dict(by_set).items()))
            sumv = 0
            sk = [list(by_dict.keys())[0] - 1]
            ek = []
            for k,v in by_dict.items():
                sumv = v + sumv
                if sumv > gap:
                    sk.append(k)
                    ek.append(k)
                    sumv = 0
            ek.append(list(by_dict.keys())[-1]+1)

            split = [data[[s < j <= e for j in data[by]]]['UE'].to_numpy() for s,e in zip(sk,ek)]
            split_pd = [data[[s < j <= e for j in data[by]]]['polysemy_degree'].to_numpy() for s,e in zip(sk,ek)]

    return split, split_pd, sk, ek

#%%
# Experiment-1: pos
df = pd.read_csv(root+"results2/analysis/unique_lemma_mean.csv")
# feats = ["nGT", "word_length", "polysemy_degree", "synset_size", "GT_freq", "hyper_depth", "nMorphs", "nMorphs_lemma", "nWP"]
feats = ["pos"]
# pos_split = splitBy(df, by="pos")
for f in feats:
    print("====%s======"%f)
    # pos_split, sk, ek = splitBy(df, by=f, level=5, EqualBin=True)
    pos_split = splitBy(df, by=f, level=5, EqualBin=True)
    print([np.mean(i) for i in pos_split])
    print([len(i) for i in pos_split])
    # print(sk)
    # print(ek)

    pvals = [[hyp_test(j, i)[0] for i in pos_split] for j in pos_split]
    print(pvals)


#%%
# wordcloud
root = "/home/liuzhu/WSD/multilabel-wsd/"
df = pd.read_csv(root+"results2/analysis/unique_lemma_mean.csv")
df_bylem = df.groupby(by="lemma").mean().reset_index()
freq_bylem = {l:u for l, u in zip(df_bylem['lemma'], df_bylem['UE'])}
pd_bylem = {l:u for l, u in zip(df_bylem['lemma'], df_bylem['polysemy_degree'])}
print("The number of lemmas is %d"%len(freq_bylem))
# wordcloud = WordCloud().generate_from_frequencies(freq_bylem)
# plt.imshow(wordcloud)
# %%
freq_bylem_sort = sorted(freq_bylem.items(), key=lambda x:x[1], reverse=True)
for i in range(100):
    df_temp = df[df['lemma'] == freq_bylem_sort[i][0]]
    df_dict = {l:u for l, u in zip(df_temp['GT_sense'], df_temp['UE'])}
    print("The number of group is %d"%len(df_dict))
    if len(df_dict) != 1:
        wordcloud = WordCloud().generate_from_frequencies(df_dict)
        plt.figure()   
        plt.imshow(wordcloud)
        plt.title(freq_bylem_sort[i][0])

# %%
# Experiment 2
# dfs = [df[df['pos'] == p] for p in ['NOUN', 'VERB', 'ADJ', 'ADV']]
dfs = [df]
dfs = [i[i['nGT'] == 1] for i in dfs]
dfs = [i.groupby(by="lemma").mean().reset_index() for i in dfs]

# for df_pos,pp in zip(dfs, ['NOUN', 'VERB', 'ADJ', 'ADV']):
#     print("====%s======"%pp)
for df_pos in dfs:
    pos_split, pos_split_pd, sk, ek = splitBy(df_pos, by='polysemy_degree', level=3, EqualBin=True)
    print([np.mean(i) for i in pos_split])
    print([np.mean(i) for i in pos_split_pd])
    print([len(i) for i in pos_split])
    print(sk)
    print(ek)

    pvals = [[hyp_test(j, i)[1] for i in pos_split] for j in pos_split]
    print(pvals)
    # print([[hyp_test(j, i)[1] for i in pos_split_pd] for j in pos_split_pd])

#%%
# Experiment 2 - Morphology
dfs = [df[df['pos'] == p] for p in ['NOUN', 'VERB', 'ADJ', 'ADV']]
dfs = [i[i['nGT'] == 1] for i in dfs]
dfs = [i.groupby(by="lemma").mean().reset_index() for i in dfs]

for df_pos,pp in zip(dfs, ['NOUN', 'VERB', 'ADJ', 'ADV']):
    print("====%s======"%pp)
    pos_split, pos_split_pd, sk, ek = splitBy(df_pos, by='nMorphs', level=3, EqualBin=True)
    print([np.mean(i) for i in pos_split])
    print([np.mean(i) for i in pos_split_pd])
    print([len(i) for i in pos_split])
    print(sk)
    print(ek)

    pvals = [[hyp_test(j, i)[1] for i in pos_split] for j in pos_split]
    print(pvals)
    # print([[hyp_test(j, i)[1] for i in pos_split_pd] for j in pos_split_pd])
# %%
# Experiment 3
df_hyps =  df[df['nGT'] == 1]
df_hyps = df_hyps[df_hyps['hyper_depth'] > 0]
df_hyps = [df_hyps[df_hyps['pos'] == p] for p in ['NOUN', 'VERB']] 

for df_pos,pp in zip(df_hyps, ['NOUN', 'VERB']):
    print("====%s======"%pp)
    pos_split, pos_split_pd, sk, ek = splitBy(df_pos, by='hyper_depth', level=3, EqualBin=True)
    print([np.mean(i) for i in pos_split])
    print([np.mean(i) for i in pos_split_pd])
    print([len(i) for i in pos_split])
    print(sk)
    print(ek)

    pvals = [[hyp_test(j, i)[1] for i in pos_split] for j in pos_split]
    print(pvals)
# %%
# Experiment 3-2
df_syn =  df[df['nGT'] == 1]
# df_hyps = df_hyps[df_hyps['hyper_depth'] > 0]
df_syn = [df_syn[df_syn['pos'] == p] for p in ['NOUN', 'VERB', 'ADJ', 'ADV']] 

for df_pos,pp in zip(df_syn, ['NOUN', 'VERB', 'ADJ', 'ADV']):
    print("====%s======"%pp)
    pos_split, pos_split_pd, sk, ek = splitBy(df_pos, by='synset_size', level=3, EqualBin=True)
    print([np.mean(i) for i in pos_split])
    print([np.mean(i) for i in pos_split_pd])
    print([len(i) for i in pos_split])
    print(sk)
    print(ek)

    pvals = [[hyp_test(j, i)[1] for i in pos_split] for j in pos_split]
    print(pvals)
# %%
